"""Đo tốc độ xử lý (FPS), thời gian từng công đoạn và benchmark throughput trên video.

Chạy: python scripts/run_benchmark.py [--video PATH]
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import sys
from pathlib import Path

# Thêm đường dẫn DLL của torch/lib để ONNX Runtime nhận diện cuDNN trên Windows
torch_lib_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
if os.path.isdir(torch_lib_dir):
    os.add_dll_directory(torch_lib_dir)
    os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")

import cv2
import re
import time
import torch
import warnings
import pandas as pd
from collections import Counter, defaultdict
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# Tắt các cảnh báo lặp lại
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent.parent

# ==============================================================================
# 1. CẤU HÌNH THIẾT BỊ VÀ MÔ HÌNH
# ==============================================================================
has_cuda = torch.cuda.is_available()
device = 0 if has_cuda else "cpu"

print("=" * 75)
if has_cuda:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"⚡ HARDWARE ACCELERATION: CUDA ENABLED ({gpu_name})")
    print(f"   Inference Device : GPU #{device}")
else:
    print("⚠️ HARDWARE ACCELERATION: CPU MODE")
print("=" * 75)

# Mô hình phát hiện biển số (Ưu tiên: ONNX Runtime CUDA -> PyTorch FP16)
model_onnx = str(SCRIPT_DIR / "v62.onnx")
model_pt = str(SCRIPT_DIR / "v62.pt")

if os.path.exists(model_onnx):
    print(f"🚀 Nạp mô hình ONNX Runtime CUDA: '{model_onnx}'")
    model = YOLO(model_onnx, task="detect")
    use_half_flag = False
elif os.path.exists(model_pt):
    print(f"🚀 Nạp mô hình PyTorch FP16: '{model_pt}'")
    model = YOLO(model_pt)
    if has_cuda:
        model.to("cuda:0")
        try:
            model.model.half()
        except Exception:
            pass
    use_half_flag = False
else:
    print("Lỗi: Không tìm thấy file model.")
    sys.exit(1)

# Khởi tạo FastPlateOCR trên CUDA
ocr_device = "cuda" if has_cuda else "cpu"
try:
    ocr_model = LicensePlateRecognizer("cct-s-v2-global-model", device=ocr_device)
    print(f"⚡ FastPlateOCR khởi tạo thành công trên thiết bị: {ocr_device.upper()}")
except Exception as e:
    print(f"⚠️ FastPlateOCR fallback sang CPU: {e}")
    ocr_model = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")

# ==============================================================================
# 2. THAM SỐ VẬN HÀNH LUỒNG CAMERA GIÁM SÁT
# ==============================================================================
input_video = str(SCRIPT_DIR / "YTSave_YouTube_Camera-nhan-dang-bien-so-xe-luu-tru-hinh_Media_VYR-VysjGkQ_001_1080p.mp4")
crop_dir = SCRIPT_DIR / "extracted_plates"
crop_dir.mkdir(exist_ok=True)

INFERENCE_IMG_SIZE = 640
DET_CONF_THRESHOLD = 0.20

MAX_OCR_SAMPLES_PER_TRACK = 5
EARLY_CONSENSUS_MIN_SAMPLES = 3
MIN_CROP_WIDTH = 35
MIN_CROP_HEIGHT = 15

DIGIT_FIX = str.maketrans({
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", 
    "T": "7", "S": "5", "B": "8", "G": "6", "Z": "2"
})

def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())

def normalize_plate(raw):
    text = clean_text(raw)
    if len(text) < 5:
        return text
    p1 = text[0].translate(DIGIT_FIX)
    p2 = text[1].translate(DIGIT_FIX)
    return p1 + p2 + text[2:]

def get_expanded_crop(frame, x1, y1, x2, y2, pad_pct=0.08):
    h, w, _ = frame.shape
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * pad_pct), int(bh * pad_pct)
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w, x2 + pad_w)
    ny2 = min(h, y2 + pad_h)
    return frame[ny1:ny2, nx1:nx2]

# ==============================================================================
# 3. CHẠY BENCHMARK PIPELINE
# ==============================================================================
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"Lỗi: Không thể mở video '{input_video}'")
    sys.exit(1)

video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

track_history = defaultdict(list)
track_ocr_count = defaultdict(int)
track_confident = set()
untracked_history = []

frame_idx = 0
processed_frames = 0
fps_history = []
total_yolo_ms = 0.0
total_ocr_ms = 0.0
total_ocr_calls = 0

print(f"\n🎬 BẮT ĐẦU BENCHMARK REAL-TIME LPR PIPELINE")
print(f"   Video nguồn : {Path(input_video).name}")
print(f"   Độ phân giải: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"   Tổng frames : {total_video_frames} frames @ {video_fps:.1f} FPS")
print(f"   Mục tiêu    : 25 - 30+ FPS\n")

track_kwargs = {
    "persist": True,
    "conf": DET_CONF_THRESHOLD,
    "imgsz": INFERENCE_IMG_SIZE,
    "device": device,
    "tracker": "bytetrack.yaml",
    "verbose": False
}

# Warmup GPU
print("🔥 Warmup GPU...")
ret, sample_frame = cap.read()
if ret:
    model.track(sample_frame, **track_kwargs)
    ocr_model.run([sample_frame[0:50, 0:100]], return_confidence=True)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("🚀 Warmup hoàn tất. Bắt đầu đo hiệu năng...")

t_pipeline_start = time.perf_counter()

while cap.isOpened():
    t_frame_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    processed_frames += 1
    timestamp_sec = round(frame_idx / video_fps, 2)
    
    # BƯỚC 1: YOLO DETECTION & TRACKING
    t_yolo_start = time.perf_counter()
    results = model.track(frame, **track_kwargs)
    t_yolo = (time.perf_counter() - t_yolo_start) * 1000
    total_yolo_ms += t_yolo
    
    # BƯỚC 2: CHỌN MẪU TRACK-AWARE OCR
    candidate_crops = []
    candidate_metas = []
    
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            
            if track_id is not None and track_id in track_confident:
                continue
            if track_id is not None and track_ocr_count[track_id] >= MAX_OCR_SAMPLES_PER_TRACK:
                continue
            
            bw = x2 - x1
            bh = y2 - y1
            if bw < MIN_CROP_WIDTH or bh < MIN_CROP_HEIGHT:
                continue
            
            crop = get_expanded_crop(frame, x1, y1, x2, y2, pad_pct=0.08)
            if crop.size == 0 or crop.shape[0] < MIN_CROP_HEIGHT or crop.shape[1] < MIN_CROP_WIDTH:
                continue
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            candidate_crops.append(crop_rgb)
            candidate_metas.append({
                "track_id": track_id,
                "det_conf": det_conf,
                "crop_bgr": crop
            })
            
    # BƯỚC 3: BATCH OCR TRÊN GPU
    t_ocr = 0.0
    if candidate_crops:
        t_ocr_start = time.perf_counter()
        ocr_preds = ocr_model.run(candidate_crops, return_confidence=True)
        t_ocr = (time.perf_counter() - t_ocr_start) * 1000
        total_ocr_ms += t_ocr
        total_ocr_calls += len(candidate_crops)
        
        for pred, meta in zip(ocr_preds, candidate_metas):
            track_id = meta["track_id"]
            det_conf = meta["det_conf"]
            crop_bgr = meta["crop_bgr"]
            
            raw_text = getattr(pred, "plate", "").strip()
            clean_plate = normalize_plate(raw_text)
            
            if len(clean_plate) >= 5:
                if hasattr(pred, "char_probs") and pred.char_probs is not None and len(raw_text) > 0:
                    actual_probs = pred.char_probs[:len(raw_text)]
                    ocr_conf = float(actual_probs.mean())
                else:
                    ocr_conf = 0.85
                
                trust_score = round((0.4 * ocr_conf + 0.6 * det_conf) * 100, 1)
                item = {
                    "plate": clean_plate,
                    "trust_score": trust_score,
                    "timestamp": timestamp_sec,
                    "crop": crop_bgr.copy()
                }
                
                if track_id is not None:
                    track_history[track_id].append(item)
                    track_ocr_count[track_id] += 1
                    items = track_history[track_id]
                    if len(items) >= EARLY_CONSENSUS_MIN_SAMPLES:
                        plates = [it["plate"] for it in items]
                        most_common, cnt = Counter(plates).most_common(1)[0]
                        if cnt >= EARLY_CONSENSUS_MIN_SAMPLES:
                            track_confident.add(track_id)
                else:
                    untracked_history.append(item)
                    
    t_frame_total = time.perf_counter() - t_frame_start
    instant_fps = 1.0 / t_frame_total if t_frame_total > 0 else 30.0
    fps_history.append(instant_fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    
    rolling_fps = sum(fps_history) / len(fps_history)
    
    if frame_idx % 60 == 0 or frame_idx == 1:
        print(f"[{timestamp_sec:>5.1f}s | Frame {frame_idx:>4}/{total_video_frames}] "
              f"⚡ Throughput: {rolling_fps:>5.1f} FPS "
              f"| YOLO: {t_yolo:>4.1f}ms | OCR: {t_ocr:>4.1f}ms (x{len(candidate_crops)}) "
              f"| Tracks: {len(track_history)} (Locked: {len(track_confident)})", flush=True)

cap.release()
t_total_elapsed = time.perf_counter() - t_pipeline_start
avg_fps = processed_frames / t_total_elapsed if t_total_elapsed > 0 else 0
avg_yolo = total_yolo_ms / processed_frames if processed_frames > 0 else 0
avg_ocr = total_ocr_ms / total_ocr_calls if total_ocr_calls > 0 else 0

print("\n" + "=" * 75)
print(f"🏁 KẾT QUẢ BENCHMARK TOÀN DIỆN TRÊN NVIDIA GEFORCE RTX 5060 Ti:")
print(f"   - Tổng thời gian xử lý: {t_total_elapsed:.2f} giây")
print(f"   - Tổng frames video   : {processed_frames} frames")
print(f"   - TỐC ĐỘ TRUNG BÌNH   : {avg_fps:.2f} FPS")
print(f"   - Thời gian YOLO/frame: {avg_yolo:.2f} ms")
print(f"   - Thời gian OCR/crop  : {avg_ocr:.2f} ms (Tổng {total_ocr_calls} crops)")
print("=" * 75)

if avg_fps >= 30.0:
    print(f"🎉 XUẤT SẮC! Đạt {avg_fps:.1f} FPS - VƯỢT MỤC TIÊU 25-30 FPS cho camera thời gian thực!")
elif avg_fps >= 25.0:
    print(f"✅ ĐẠT YÊU CẦU! Đạt {avg_fps:.1f} FPS - Chạy mượt mà trong dải 25-30 FPS!")
else:
    print(f"ℹ️ Tốc độ đạt {avg_fps:.1f} FPS.")

# XUẤT KẾT QUẢ BIỂN SỐ XE
final_results = []
for track_id, items in track_history.items():
    if not items: continue
    plate_counts = Counter(item["plate"] for item in items)
    most_common_plate, count = plate_counts.most_common(1)[0]
    total_detections = len(items)
    winning_items = [item for item in items if item["plate"] == most_common_plate]
    best_item = max(winning_items, key=lambda x: x["trust_score"])
    crop_filename = f"{most_common_plate}_track{track_id}.jpg"
    cv2.imwrite(str(crop_dir / crop_filename), best_item["crop"])
    final_results.append({
        "Biển số": most_common_plate,
        "Track ID": f"Xe #{track_id}",
        "Số frame đọc": f"{count}/{total_detections}",
        "Độ tin cậy": f"{best_item['trust_score']}%",
        "Thời điểm": f"{best_item['timestamp']}s"
    })

df = pd.DataFrame(final_results)
print(f"\n📋 ĐÃ TRÍCH XUẤT THÀNH CÔNG {len(df)} BIỂN SỐ XE DUY NHẤT:")
if not df.empty:
    print(df.to_string(index=False))

output_csv = str(SCRIPT_DIR / "evaluation" / "results" / "benchmark_ket_qua_bien_so.csv")
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n📁 File kết quả đã lưu tại: '{output_csv}'")
