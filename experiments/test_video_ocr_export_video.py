import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import re
import time
import torch
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN
# ==============================================================================
parser = argparse.ArgumentParser(description="Hệ thống ALPR & xuất video kết quả nhận diện biển số")
parser.add_argument("--video", type=str, default="test/test_3.mp4", help="Đường dẫn video đầu vào")
parser.add_argument("--model", type=str, default=None, help="Đường dẫn file trọng số YOLO")
parser.add_argument("--output", type=str, default="outputs/videos/output_video_nhan_dien_bien_so1.mp4", help="Đường dẫn video xuất ra")
parser.add_argument("--csv", type=str, default="outputs/csv/danh_sach_bien_so_xuat_video.csv", help="Đường dẫn file CSV xuất ra")
parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy detection")
cli_args, _ = parser.parse_known_args()

# Thiết bị xử lý (Tự động nhận diện GPU CUDA nếu có, fallback về CPU)
has_cuda = torch.cuda.is_available()
device = "cuda:0" if has_cuda else "cpu"

print("=" * 78)
print("🚀 HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE (ALPR) & XUẤT VIDEO TRÍCH XUẤT")
print(f"   Thiết bị tính toán: {device.upper()} {'(NVIDIA GPU)' if has_cuda else '(CPU)'}")
print("=" * 78)

# Nạp mô hình YOLO nhận diện biển số
model_candidates = [
    cli_args.model if cli_args.model else "",
    r"model\Wiliv5_best.pt",
    r"model\v62.pt",
    r"DSP1\Wiliv5_best.pt",
    r"archives\DSP1\Wiliv5_best.pt"
]
model_path = next((p for p in model_candidates if p and os.path.exists(p)), r"model\Wiliv5_best.pt")
if not os.path.exists(model_path):
    print(f"❌ Lỗi: Không tìm thấy weights '{model_path}'")
    exit(1)

model = YOLO(model_path)
if has_cuda:
    model.to(device)

# Nạp FastPlateOCR
ocr_device = "cuda" if has_cuda else "cpu"
try:
    ocr_model = LicensePlateRecognizer("cct-s-v2-global-model", device=ocr_device)
    print(f"⚡ FastPlateOCR khởi tạo thành công trên: {ocr_device.upper()}")
except Exception as e:
    print(f"⚠️ Chuyển FastPlateOCR sang CPU fallback: {e}")
    ocr_model = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")

# ==============================================================================
# 2. CẤU HÌNH ĐẦU VÀO, ĐẦU RA & THAM SỐ VẬN HÀNH
# ==============================================================================
input_video = cli_args.video if os.path.exists(cli_args.video) else r"test\test_3.mp4"
output_video_path = cli_args.output
output_csv_path = cli_args.csv

Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
crop_dir = Path("outputs/extracted_plates")
crop_dir.mkdir(parents=True, exist_ok=True)

# THAM SỐ XỬ LÝ VIDEO
FRAME_STRIDE = 1
CONF_THRESHOLD = cli_args.conf
IMGSZ = 768

# Ngưỡng kích thước tối thiểu và hình học của biển số
MIN_PLATE_WIDTH = 35    # Chiều rộng tối thiểu của bounding box
MIN_PLATE_HEIGHT = 16   # Chiều cao tối thiểu của bounding box

# Tối ưu hóa OCR theo xe (Tránh gọi OCR lặp lại quá nhiều lần cho cùng 1 xe)
MAX_OCR_PER_TRACK = 5       # Số lần OCR tối đa cho mỗi xe
EARLY_CONSENSUS_COUNT = 3   # Khi 3 lần OCR ra cùng 1 biển số với điểm cao -> khóa OCR xe đó

# Giới hạn số frame muốn xử lý (đặt None để chạy hết toàn bộ video, hoặc số nguyên ví dụ 150 để test nhanh)
MAX_FRAMES_TO_PROCESS = None

# Hiển thị cửa sổ xem trực tiếp khi đang chạy (True để mở màn hình xem, False để xuất ngầm nhanh hơn)
SHOW_LIVE_PREVIEW = False

# ==============================================================================
# 3. CÁC HÀM BỔ TRỢ CHUẨN HÓA BIỂN SỐ VN & VẼ GIAO DIỆN
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

# Cấu trúc biển số xe cơ giới Việt Nam chuẩn:
# 1. Xe dân sự / thương mại / liên doanh: 
#    - 2 số đầu là mã tỉnh (11-99 trong danh mục VN_PROVINCE_CODES)
#    - Tiếp theo là 1 chữ cái (ô tô) hoặc 1 chữ + 1 số/chữ (xe máy/liên doanh như 51LD, 59MD)
#    - Phần sau là 4 hoặc 5 chữ số
#    Tổng chiều dài: 7 đến 10 ký tự.
VN_CIVIL_PATTERN = re.compile(r"^[0-9]{2}[A-Z](?:[0-9A-Z])?[0-9]{4,5}$")

# 2. Xe quân sự đặc thù (nếu có): 2 chữ cái quân đội + 4-6 chữ số
VN_MILITARY_PATTERN = re.compile(
    r"^(?:TM|TC|TH|TK|TT|TL|TN|TP|TQ|TS|TU|TB|QA|QB|QC|QD|QP|HA|HB|HC|HD|HE|HH|KA|KB|KC|KD|KP|VT|BB)[0-9]{4,6}$"
)

def clean_text(text):
    """Lọc ký tự lạ, giữ lại chữ in hoa và số"""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())

def normalize_plate(raw):
    """Chuẩn hóa biển số xe VN, sửa nhầm ký tự ở 2 vị trí mã tỉnh"""
    text = clean_text(raw)
    if len(text) < 5:
        return text
    p1 = text[0].translate(DIGIT_FIX)
    p2 = text[1].translate(DIGIT_FIX)
    text = p1 + p2 + text[2:]
    return text

def is_valid_vietnamese_plate(plate: str) -> bool:
    """
    Kiểm tra định dạng biển số xe Việt Nam.
    Loại bỏ hoàn toàn các loại logo, tem mác, nhãn hiệu như 'PMI', 'NP111', 'PP111', số điện thoại,...
    """
    if not plate or len(plate) < 7 or len(plate) > 10:
        return False
    
    # Trường hợp xe quân đội
    if VN_MILITARY_PATTERN.match(plate):
        return True
        
    # Mã tỉnh phải thuộc danh mục hợp lệ
    prov = plate[:2]
    if prov not in VN_PROVINCE_CODES:
        return False
        
    # Cấu trúc ký tự dân sự chuẩn
    return bool(VN_CIVIL_PATTERN.match(plate))

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

def draw_plate_box(frame, x1, y1, x2, y2, label, color=(0, 230, 115), box_thickness=2):
    """Vẽ bounding box phong cách hiện đại với góc nhấn và thẻ nhãn rõ ràng"""
    h, w, _ = frame.shape
    
    # 1. Khung chữ nhật bao quanh biển số
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
    
    # 2. Điểm nhấn 4 góc (Giao diện ALPR chuyên nghiệp)
    corner_len = max(6, min(int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)))
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, box_thickness + 2)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, box_thickness + 2)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, box_thickness + 2)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, box_thickness + 2)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, box_thickness + 2)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, box_thickness + 2)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, box_thickness + 2)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, box_thickness + 2)
    
    # 3. Thẻ nhãn hiển thị biển số và độ tin cậy
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    tag_y1 = y1 - text_h - 10
    tag_y2 = y1
    if tag_y1 < 45:  # Tránh che thanh HUD ở trên cùng
        tag_y1 = y2
        tag_y2 = y2 + text_h + 10
        text_y = tag_y2 - 5
    else:
        text_y = y1 - 5
        
    tag_x1 = max(0, x1)
    tag_x2 = min(w, tag_x1 + text_w + 14)
    
    # Nền thẻ nhãn
    cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
    # Màu chữ tương phản
    text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 380 else (255, 255, 255)
    cv2.putText(frame, label, (tag_x1 + 7, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def draw_hud(frame, frame_idx, total_frames, fps, total_vehicles, active_vehicles):
    """Vẽ thanh trạng thái HUD giám sát bán trong suốt ở mép trên video"""
    h, w, _ = frame.shape
    banner_h = 42
    
    # Lớp phủ nền tối bán trong suốt (Alpha Blending)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, banner_h), (w, banner_h), (0, 230, 115), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    progress_pct = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
    left_text = f"ALPR SYSTEM | Frame: {frame_idx}/{total_frames} ({progress_pct}%) | FPS: {fps:.1f}"
    cv2.putText(frame, left_text, (15, 27), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
    right_text = f"Total Detected: {total_vehicles} | In Frame: {active_vehicles}"
    (rw, rh), _ = cv2.getTextSize(right_text, font, 0.65, 2)
    cv2.putText(frame, right_text, (w - rw - 15, 27), font, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

# ==============================================================================
# 4. KHỞI TẠO BỘ ĐỌC VÀ GHI VIDEO (OPENCV VIDEO WRITER)
# ==============================================================================
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"❌ Lỗi: Không thể mở file video '{input_video}'")
    exit(1)

video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

target_total_frames = total_video_frames if MAX_FRAMES_TO_PROCESS is None else min(total_video_frames, MAX_FRAMES_TO_PROCESS)

# Thiết lập VideoWriter (codec MP4V chuẩn tương thích tốt trên Windows)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))

if not out_writer.isOpened():
    print(f"❌ Lỗi: Không thể khởi tạo VideoWriter lưu vào '{output_video_path}'")
    exit(1)


print(f"\n🎬 THÔNG TIN LUỒNG VIDEO:")
print(f"   Video nguồn   : '{input_video}'")
print(f"   Độ phân giải  : {video_width}x{video_height}")
print(f"   Tốc độ khung  : {video_fps:.1f} FPS")
print(f"   Tổng số frames: {target_total_frames} frames (~{target_total_frames / video_fps:.1f}s)")
print(f"   Video xuất ra : '{output_video_path}'")
print(f"   Chế độ Stride : {FRAME_STRIDE} (Xử lý từng frame: {'Bật' if FRAME_STRIDE == 1 else 'Tắt (tăng tốc)'})")
print("=" * 78)

# ==============================================================================
# 5. VÒNG LẶP XỬ LÝ VIDEO & ĐẾM TẦN SUẤT ĐỒNG THUẬN (MAJORITY VOTING)
# ==============================================================================
track_history = defaultdict(list)
track_ocr_count = defaultdict(int)
track_locked = set()
untracked_history = []

# Lưu trữ cache các box của frame trước (dùng khi FRAME_STRIDE > 1 để video mượt mà)
last_frame_annotations = []

frame_idx = 0
processed_frames = 0
fps_history = []
t_start = time.perf_counter()

print("⏳ Bắt đầu phân tích khung hình và ghi video...\n")

while cap.isOpened():
    t_frame_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    if MAX_FRAMES_TO_PROCESS is not None and frame_idx > MAX_FRAMES_TO_PROCESS:
        break
    
    timestamp_sec = round(frame_idx / video_fps, 2)
    is_key_frame = (frame_idx % FRAME_STRIDE == 0)
    
    current_annotations = []
    
    if is_key_frame:
        processed_frames += 1
        
        # YOLO Detection & Tracking
        track_results = model.track(
            frame, 
            persist=True, 
            conf=CONF_THRESHOLD, 
            imgsz=IMGSZ, 
            device=device, 
            verbose=False
        )
        
        for r in track_results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
                
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None
                
                # 1. Lọc hình học: Kích thước và Tỷ lệ khung hình (Aspect Ratio)
                bw = x2 - x1
                bh = y2 - y1
                if bw < MIN_PLATE_WIDTH or bh < MIN_PLATE_HEIGHT:
                    continue
                
                aspect_ratio = bw / max(1, bh)
                # Biển số chuẩn VN: Biển vuông (1.1 <= ratio <= 2.2) hoặc Biển dài (2.6 <= ratio <= 5.5)
                if not ((1.1 <= aspect_ratio <= 2.2) or (2.6 <= aspect_ratio <= 5.5)):
                    continue
                
                # Cắt ảnh biển số có lề bảo vệ
                crop = get_expanded_crop(frame, x1, y1, x2, y2, pad_pct=0.08)
                if crop.size == 0 or crop.shape[0] < MIN_PLATE_HEIGHT or crop.shape[1] < MIN_PLATE_WIDTH:
                    continue
                
                # Kiểm tra xem có cần gọi OCR cho xe này không (Selective OCR)
                need_ocr = True
                if track_id is not None:
                    if track_id in track_locked:
                        need_ocr = False
                    elif track_ocr_count[track_id] >= MAX_OCR_PER_TRACK:
                        need_ocr = False
                        
                current_plate = None
                current_trust = None

                if need_ocr:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    ocr_preds = ocr_model.run(crop_rgb, return_confidence=True)
                    
                    if ocr_preds:
                        pred = ocr_preds[0]
                        raw_text = getattr(pred, "plate", "").strip()
                        clean_plate = normalize_plate(raw_text)
                        
                        # CHỈ CHẤP NHẬN NẾU ĐÚNG ĐỊNH DẠNG BIỂN SỐ VIỆT NAM HỢP LỆ
                        if is_valid_vietnamese_plate(clean_plate):
                            # Tính điểm tin cậy trung bình của các ký tự
                            if hasattr(pred, "char_probs") and pred.char_probs is not None and len(raw_text) > 0:
                                actual_probs = pred.char_probs[:len(raw_text)]
                                ocr_conf = float(actual_probs.mean())
                            else:
                                ocr_conf = 0.85
                            
                            # Điểm tin cậy tổng hợp (Detection + OCR)
                            trust_score = round((0.4 * ocr_conf + 0.6 * det_conf) * 100, 1)
                            
                            current_plate = clean_plate
                            current_trust = trust_score
                            
                            item = {
                                "plate": clean_plate,
                                "trust_score": trust_score,
                                "timestamp": timestamp_sec,
                                "crop": crop.copy()
                            }
                            
                            if track_id is not None:
                                track_history[track_id].append(item)
                                track_ocr_count[track_id] += 1
                                
                                # Kiểm tra xem đã đạt Early Consensus chưa
                                items = track_history[track_id]
                                if len(items) >= EARLY_CONSENSUS_COUNT:
                                    cnt = Counter(it["plate"] for it in items).most_common(1)[0][1]
                                    if cnt >= EARLY_CONSENSUS_COUNT:
                                        track_locked.add(track_id)
                            else:
                                untracked_history.append(item)
                        elif track_id is not None:
                            # Tăng đếm số lần OCR để không cố đọc mãi các vật thể giả
                            track_ocr_count[track_id] += 1
                
                # Xác định nhãn hiển thị theo đa số (Majority Voting)
                if track_id is not None and track_id in track_history and len(track_history[track_id]) > 0:
                    items = track_history[track_id]
                    plate_counts = Counter(it["plate"] for it in items)
                    voted_plate, v_count = plate_counts.most_common(1)[0]
                    matching = [it for it in items if it["plate"] == voted_plate]
                    best_trust = max(it["trust_score"] for it in matching)
                    
                    label_str = f"Xe #{track_id} | {voted_plate} ({best_trust:.0f}%)"
                    box_color = (0, 230, 115)  # Xanh lá cây (Đã chốt biển số)
                elif track_id is not None and current_plate:
                    label_str = f"Xe #{track_id} | {current_plate} ({current_trust:.0f}%)"
                    box_color = (0, 230, 115)  # Xanh lá cây
                elif track_id is not None:
                    # Nếu xe đã theo vết nhiều frame mà không lần nào ra biển hợp lệ -> logo/tem dán -> BỎ QUA KHÔNG VẼ
                    if track_ocr_count[track_id] >= 3 and len(track_history[track_id]) == 0:
                        continue
                    label_str = f"Xe #{track_id} | Dang nhan dien..."
                    box_color = (0, 215, 255)  # Vàng (Đang phân tích)
                elif current_plate:
                    label_str = f"Bien so: {current_plate} ({current_trust:.0f}%)"
                    box_color = (0, 215, 255)  # Vàng / Cyan
                else:
                    # Không có track_id và cũng không đọc được biển số hợp lệ (logo, sticker, nhiễu) -> BỎ QUA HOÀN TOÀN
                    continue
                
                current_annotations.append({
                    "box": (x1, y1, x2, y2),
                    "label": label_str,
                    "color": box_color
                })
        
        last_frame_annotations = current_annotations
    else:
        # Nếu frame bị bỏ qua do STRIDE > 1, tái sử dụng annotation để video không bị giật
        current_annotations = last_frame_annotations

    # --------------------------------------------------------------------------
    # VẼ CÁC ANNOTATIONS LÊN KHUNG HÌNH
    # --------------------------------------------------------------------------
    for ann in current_annotations:
        bx1, by1, bx2, by2 = ann["box"]
        draw_plate_box(frame, bx1, by1, bx2, by2, ann["label"], color=ann["color"])
    
    # Tính toán FPS tức thời
    t_frame_end = time.perf_counter()
    instant_fps = 1.0 / (t_frame_end - t_frame_start) if (t_frame_end - t_frame_start) > 0 else 30.0
    fps_history.append(instant_fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    smooth_fps = sum(fps_history) / len(fps_history)
    
    # Vẽ HUD trên đỉnh video
    active_in_frame = len(current_annotations)
    unique_plates_count = len(set(it["plate"] for items in track_history.values() for it in items) | set(it["plate"] for it in untracked_history))
    total_tracked_so_far = max(len(track_history), unique_plates_count)
    draw_hud(frame, frame_idx, target_total_frames, smooth_fps, total_tracked_so_far, active_in_frame)
    
    # Ghi frame vào file video output
    out_writer.write(frame)
    
    # Hiển thị trực tiếp nếu bật tùy chọn
    if SHOW_LIVE_PREVIEW:
        cv2.imshow("ALPR Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️ Người dùng đã dừng chương trình bằng phím 'q'.")
            break

    # In log tiến độ định kỳ mỗi 30 frame
    if frame_idx % 30 == 0 or frame_idx == 1:
        elapsed = time.perf_counter() - t_start
        fps_overall = frame_idx / elapsed if elapsed > 0 else 0
        eta_sec = (target_total_frames - frame_idx) / fps_overall if fps_overall > 0 else 0
        print(f"[{timestamp_sec:>5.1f}s | Frame {frame_idx:>4}/{target_total_frames}] "
              f"⚡ {smooth_fps:>5.1f} FPS | Đã ghi nhận: {total_tracked_so_far} xe "
              f"| ETA: {int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}")

# Giải phóng bộ nhớ và file
cap.release()
out_writer.release()
if SHOW_LIVE_PREVIEW:
    cv2.destroyAllWindows()

t_total_time = time.perf_counter() - t_start
avg_fps = frame_idx / t_total_time if t_total_time > 0 else 0

print("\n" + "=" * 78)
print(f"🏁 ĐÃ XUẤT VIDEO THÀNH CÔNG: '{output_video_path}'")
print(f"   Tổng số frame đã ghi: {frame_idx} frames")
print(f"   Thời gian thực hiện : {t_total_time:.2f} giây")
print(f"   Tốc độ trung bình   : {avg_fps:.2f} FPS")
print("=" * 78)

# ==============================================================================
# 6. TỔNG HỢP KẾT QUẢ BẰNG ĐẾM TẦN SUẤT (MAJORITY VOTING) & LƯU CSV
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

# Xử lý nhóm không gán được Track ID (Untracked)
if untracked_history:
    plate_groups = defaultdict(list)
    for item in untracked_history:
        plate_groups[item["plate"]].append(item)
        
    already_found = {r["Biển số (Số đông nhất)"] for r in final_results}
    
    for plate, items in plate_groups.items():
        if is_valid_vietnamese_plate(plate) and plate not in already_found and (len(items) >= 2 or items[0]["trust_score"] >= 75.0):
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

df = pd.DataFrame(final_results)

print("\n📋 BẢNG THỐNG KÊ BIỂN SỐ XE TRÍCH XUẤT ĐƯỢC (MAJORITY VOTING):")
print("-" * 78)
if not df.empty:
    print(df.to_string(index=False))
else:
    print("Không tìm thấy biển số xe nào hợp lệ trong video.")
print("-" * 78)

try:
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n📁 Đã lưu file danh sách: '{output_csv_path}'")
except PermissionError:
    alt_csv = "danh_sach_bien_so_xuat_video_backup.csv"
    df.to_csv(alt_csv, index=False, encoding="utf-8-sig")
    print(f"\n📁 File cũ bị khóa, đã lưu vào: '{alt_csv}'")
