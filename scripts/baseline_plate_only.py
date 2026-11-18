"""Baseline: Chạy nhận diện và đọc biển số xe trực tiếp không sử dụng vehicle tracker.

Chạy: python scripts/baseline_plate_only.py [--video PATH]
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import sys
from pathlib import Path

# Đảm bảo ONNX Runtime tìm thấy cuDNN trong torch/lib trên Windows
torch_lib_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
if os.path.isdir(torch_lib_dir):
    os.add_dll_directory(torch_lib_dir)
    os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")

import cv2
import re
import time
import torch
import pandas as pd
from collections import Counter, defaultdict
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

SCRIPT_DIR = Path(__file__).resolve().parent.parent

# ==============================================================================
# 1. CẤU HÌNH PHẦN CỨNG & TĂNG TỐC (CUDA / TENSORRT / FP16)
# ==============================================================================
# Kiểm tra khả năng tăng tốc phần cứng (NVIDIA CUDA / TensorRT)
has_cuda = torch.cuda.is_available()
device = 0 if has_cuda else "cpu"
use_half = has_cuda  # FP16 half-precision trên GPU tăng 2x-3x tốc độ và tiết kiệm VRAM

print("=" * 75)
if has_cuda:
    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)
    print(f"⚡ HARDWARE ACCELERATION: CUDA ENABLED ({gpu_name})")
    print(f"   Inference Device : {device}")
    print(f"   Precision Mode   : FP16 (Half-Precision)")
else:
    print("⚠️ HARDWARE ACCELERATION: CPU MODE (CUDA không khả dụng trong PyTorch)")
    print("   Gợi ý: Cài PyTorch CUDA (cu121/cu124) để tận dụng tối đa GPU NVIDIA.")
print("=" * 75)

# Khởi tạo mô hình phát hiện biển số (Ưu tiên: TensorRT Engine -> ONNX Runtime CUDA -> PyTorch .pt)
model_pt = str(SCRIPT_DIR / "v62.pt")
model_onnx = str(SCRIPT_DIR / "v62.onnx")
model_engine = str(SCRIPT_DIR / "v62.engine")

if os.path.exists(model_engine):
    print(f"🚀 Nạp mô hình TensorRT Engine: '{model_engine}' (Tốc độ tối đa!)")
    model = YOLO(model_engine, task="detect")
elif os.path.exists(model_onnx):
    print(f"🚀 Nạp mô hình ONNX Runtime CUDA: '{model_onnx}'")
    model = YOLO(model_onnx, task="detect")
elif os.path.exists(model_pt):
    print(f"🚀 Nạp mô hình PyTorch: '{model_pt}' (Tăng tốc FP16 trên GPU)")
    model = YOLO(model_pt)
    if has_cuda:
        model.to("cuda:0")
        try:
            model.model.half()
        except Exception:
            pass
else:
    print(f"Lỗi: Không tìm thấy weights '{model_pt}', '{model_onnx}' hoặc '{model_engine}'")
    exit(1)

# Khởi tạo FastPlateOCR (tự động chọn thiết bị)
ocr_device = "cuda" if has_cuda else "cpu"
try:
    ocr_model = LicensePlateRecognizer("cct-s-v2-global-model", device=ocr_device)
    print(f"⚡ FastPlateOCR khởi tạo thành công trên thiết bị: {ocr_device.upper()}")
except Exception as e:
    print(f"⚠️ FastPlateOCR chuyển sang CPU fallback (Lý do: {e})")
    ocr_model = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")

# ==============================================================================
# 2. THÔNG SỐ VẬN HÀNH LUỒNG CAMERA GIÁM SÁT (MỤC TIÊU 25-30 FPS)
# ==============================================================================
input_video = str(SCRIPT_DIR / "YTSave_YouTube_Camera-nhan-dang-bien-so-xe-luu-tru-hinh_Media_VYR-VysjGkQ_001_1080p.mp4")

crop_dir = SCRIPT_DIR / "extracted_plates"
crop_dir.mkdir(exist_ok=True)

# Độ phân giải chuẩn Real-time cho camera giám sát (640x640)
INFERENCE_IMG_SIZE = 640
DET_CONF_THRESHOLD = 0.20

# Chiến lược tối ưu hóa Track-Aware Selective OCR:
# Camera giám sát ghi lại mỗi phương tiện trong 30-100 frames. Việc gọi OCR ở TẤT CẢ các frames
# là nguyên nhân chính làm tụt FPS (<10 FPS). Tối ưu:
# 1. Chỉ lấy mẫu 3-5 frame chất lượng cao nhất cho mỗi xe.
# 2. Khi đạt đồng thuận (consensus) cao thì khóa OCR cho xe đó, chỉ giữ lại tracking cực nhẹ.
MAX_OCR_SAMPLES_PER_TRACK = 5       # Số lần OCR tối đa cho mỗi track_id
EARLY_CONSENSUS_MIN_SAMPLES = 3     # Đạt 3 lần đọc khớp nhau với điểm cao -> dừng OCR cho xe đó
MIN_CROP_WIDTH = 35                 # Lọc các bounding box quá bé/ở xa (nhiễu)
MIN_CROP_HEIGHT = 15

# ==============================================================================
# 3. CÁC HÀM BỔ TRỢ CHUẨN HÓA BIỂN SỐ VN
# ==============================================================================
VN_PROVINCE_CODES = {
    "11","12","14","15","16","17","18","19","20","21","22","23","24","25","26",
    "27","28","29","30","31","32","33","34","35","36","37","38","39","40","41",
    "43","47","48","49","50","51","52","53","54","55","56","57","58","59","60",
    "61","62","63","64","65","66","67","68","69","70","71","72","73","74","75",
    "76","77","78","79","80","81","82","83","84","85","86","88","89","90","92",
    "93","94","95","97","98","99"
}
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
    # Sửa lỗi OCR nhầm lẫn ký tự số ở 2 ký tự mã tỉnh đầu tiên
    p1 = text[0].translate(DIGIT_FIX)
    p2 = text[1].translate(DIGIT_FIX)
    text = p1 + p2 + text[2:]
    return text

def get_expanded_crop(frame, x1, y1, x2, y2, pad_pct=0.08):
    """Mở rộng bounding box một chút để OCR không bị cắt lẹm viền biển số"""
    h, w, _ = frame.shape
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(bw * pad_pct)
    pad_h = int(bh * pad_pct)
    
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w, x2 + pad_w)
    ny2 = min(h, y2 + pad_h)
    return frame[ny1:ny2, nx1:nx2]

# ==============================================================================
# 4. ĐỌC VIDEO & XỬ LÝ THEO THỜI GIAN THỰC (REAL-TIME SURVEILLANCE PIPELINE)
# ==============================================================================
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"Lỗi: Không thể mở file video '{input_video}'")
    exit(1)

video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

track_history = defaultdict(list)
track_ocr_count = defaultdict(int)
track_confident = set()
untracked_history = []

frame_idx = 0
processed_frames = 0

# Đo throughput thời gian thực (Moving Average FPS)
fps_history = []
t_pipeline_start = time.perf_counter()

print(f"\n🎬 Bắt đầu xử lý luồng camera giám sát...")
print(f"   Nguồn video : '{input_video}'")
print(f"   Độ phân giải: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"   Tổng frames : {total_video_frames} frames @ {video_fps:.1f} FPS")
print(f"   Mục tiêu    : Throughput vận hành thời gian thực 25–30 FPS\n")

# Cấu hình tham số track cho YOLO (Sử dụng ByteTrack tối ưu cho camera giám sát tĩnh)
track_kwargs = {
    "persist": True,
    "conf": DET_CONF_THRESHOLD,
    "imgsz": INFERENCE_IMG_SIZE,
    "device": device,
    "tracker": "bytetrack.yaml",
    "verbose": False
}

while cap.isOpened():
    t_frame_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    processed_frames += 1
    timestamp_sec = round(frame_idx / video_fps, 2)
    
    # --------------------------------------------------------------------------
    # BƯỚC 1: YOLO DETECTION & TRACKING
    # --------------------------------------------------------------------------
    t_yolo_start = time.perf_counter()
    results = model.track(frame, **track_kwargs)
    t_yolo = (time.perf_counter() - t_yolo_start) * 1000  # ms
    
    # --------------------------------------------------------------------------
    # BƯỚC 2: CHỌN MẪU OCR THÔNG MINH (TRACK-AWARE SELECTIVE FILTERING)
    # --------------------------------------------------------------------------
    candidate_crops = []
    candidate_metas = []
    
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            
            # Nếu xe này đã đạt độ tin cậy đồng thuận cao, bỏ qua OCR để bảo toàn 25-30 FPS
            if track_id is not None and track_id in track_confident:
                continue
            
            # Nếu đã lấy đủ số mẫu tối đa cho xe này, bỏ qua OCR
            if track_id is not None and track_ocr_count[track_id] >= MAX_OCR_SAMPLES_PER_TRACK:
                continue
            
            bw = x2 - x1
            bh = y2 - y1
            # Bỏ qua crop quá nhỏ (ở xa/nhiễu)
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
    
    # --------------------------------------------------------------------------
    # BƯỚC 3: BATCH OCR INFERENCE (CHẠY ĐỒNG THỜI GOM NHÓM TRÊN GPU)
    # --------------------------------------------------------------------------
    t_ocr_start = time.perf_counter()
    if candidate_crops:
        # Gom nhóm toàn bộ biển số trong frame và suy luận trong 1 lần gọi
        ocr_preds = ocr_model.run(candidate_crops, return_confidence=True)
        
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
                    
                    # Kiểm tra xem xe này đã đạt độ tin cậy đồng thuận sớm chưa (Early Consensus)
                    items = track_history[track_id]
                    if len(items) >= EARLY_CONSENSUS_MIN_SAMPLES:
                        plates = [it["plate"] for it in items]
                        most_common, cnt = Counter(plates).most_common(1)[0]
                        if cnt >= EARLY_CONSENSUS_MIN_SAMPLES:
                            track_confident.add(track_id)
                else:
                    untracked_history.append(item)
                    
    t_ocr = (time.perf_counter() - t_ocr_start) * 1000  # ms
    
    # --------------------------------------------------------------------------
    # ĐO ĐẠC THROUGHPUT THỜI GIAN THỰC
    # --------------------------------------------------------------------------
    t_frame_total = time.perf_counter() - t_frame_start
    instant_fps = 1.0 / t_frame_total if t_frame_total > 0 else 30.0
    fps_history.append(instant_fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    
    rolling_fps = sum(fps_history) / len(fps_history)
    
    # In thông số throughput định kỳ mỗi 15 frame
    if frame_idx % 15 == 0 or frame_idx == 1:
        ocr_batch_size = len(candidate_crops)
        active_tracks = len(track_history)
        print(f"[{timestamp_sec:>5.1f}s | Frame {frame_idx:>4}/{total_video_frames}] "
              f"⚡ Throughput: {rolling_fps:>5.1f} FPS "
              f"| YOLO: {t_yolo:>4.1f}ms | OCR: {t_ocr:>4.1f}ms (x{ocr_batch_size}) "
              f"| Tracks: {active_tracks} (Locked: {len(track_confident)})")

cap.release()
t_total_elapsed = time.perf_counter() - t_pipeline_start
avg_fps = processed_frames / t_total_elapsed if t_total_elapsed > 0 else 0

print("\n" + "=" * 75)
print(f"🏁 HOÀN TẤT XỬ LÝ VIDEO TRONG: {t_total_elapsed:.2f} GIÂY")
print(f"📊 TỔNG SỐ FRAME ĐÃ XỬ LÝ    : {processed_frames} frames")
print(f"🚀 TỐC ĐỘ THROUGHPUT TRUNG BÌNH: {avg_fps:.2f} FPS")
if avg_fps >= 25.0:
    print("✅ ĐẠT YÊU CẦU: Tốc độ vận hành thời gian thực 25–30 FPS ổn định!")
else:
    print(f"ℹ️ Tốc độ hiện tại: {avg_fps:.2f} FPS (Khuyến nghị dùng CUDA/TensorRT để đạt 30+ FPS)")
print("=" * 75)

# ==============================================================================
# 5. TỔNG HỢP KẾT QUẢ BẰNG ĐẾM TẦN SUẤT (MAJORITY VOTING)
# ==============================================================================
final_results = []

# Xử lý nhóm có Track ID
for track_id, items in track_history.items():
    if not items:
        continue
    plate_counts = Counter(item["plate"] for item in items)
    most_common_plate, count = plate_counts.most_common(1)[0]
    total_detections = len(items)
    consensus_rate = round((count / total_detections) * 100, 1)
    
    winning_items = [item for item in items if item["plate"] == most_common_plate]
    best_item = max(winning_items, key=lambda x: x["trust_score"])
    
    crop_filename = f"{most_common_plate}_track{track_id}.jpg"
    cv2.imwrite(str(crop_dir / crop_filename), best_item["crop"])
    
    final_results.append({
        "Biển số (Số đông nhất)": most_common_plate,
        "Track ID": f"Xe #{track_id}",
        "Số frame xuất hiện": f"{count}/{total_detections}",
        "Tỷ lệ đồng thuận": f"{consensus_rate}%",
        "Độ tin cậy cao nhất": f"{best_item['trust_score']}%",
        "Thời gian (giây)": f"{best_item['timestamp']}s",
        "File ảnh crop": str(crop_dir / crop_filename)
    })

# Xử lý các biển số untracked (nếu có)
if untracked_history:
    plate_groups = defaultdict(list)
    for item in untracked_history:
        plate_groups[item["plate"]].append(item)
        
    already_found = {r["Biển số (Số đông nhất)"] for r in final_results}
    
    for plate, items in plate_groups.items():
        if plate not in already_found and (len(items) >= 2 or items[0]["trust_score"] >= 75.0):
            best_item = max(items, key=lambda x: x["trust_score"])
            crop_filename = f"{plate}_untracked.jpg"
            cv2.imwrite(str(crop_dir / crop_filename), best_item["crop"])
            final_results.append({
                "Biển số (Số đông nhất)": plate,
                "Track ID": "Bổ sung (Untracked)",
                "Số frame xuất hiện": f"{len(items)} frames",
                "Tỷ lệ đồng thuận": "100%",
                "Độ tin cậy cao nhất": f"{best_item['trust_score']}%",
                "Thời gian (giây)": f"{best_item['timestamp']}s",
                "File ảnh crop": str(crop_dir / crop_filename)
            })

# ==============================================================================
# 6. XUẤT KẾT QUẢ RA MÀN HÌNH & FILE CSV
# ==============================================================================
df = pd.DataFrame(final_results)

print("\n" + "=" * 75)
print("📋 DANH SÁCH BIỂN SỐ XE TRÍCH XUẤT ĐƯỢC (MAJORITY VOTING):")
print("=" * 75)
if not df.empty:
    print(df.to_string(index=False))
else:
    print("Không tìm thấy biển số xe nào hợp lệ trong video.")

output_csv = str(SCRIPT_DIR / "evaluation" / "results" / "danh_sach_bien_so_trich_xuat_new_25-30.csv")
try:
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n📁 Đã lưu file kết quả: '{output_csv}'")
except PermissionError:
    output_csv = str(SCRIPT_DIR / "evaluation" / "results" / "danh_sach_bien_so_trich_xuat_new.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n📁 File cũ bị khóa, đã lưu ra file mới: '{output_csv}'")
