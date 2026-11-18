import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import re
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐƯỜNG DẪN
# ==============================================================================
parser = argparse.ArgumentParser(description="Nhận diện biển số xe từ video bằng Majority Voting")
parser.add_argument("--video", type=str, default="test/YTSave_YouTube_Camera-nhan-dang-bien-so-xe-luu-tru-hinh_Media_VYR-VysjGkQ_001_1080p.mp4", help="Đường dẫn video đầu vào")
parser.add_argument("--model", type=str, default=None, help="Đường dẫn file trọng số YOLO")
parser.add_argument("--csv", type=str, default="outputs/csv/danh_sach_bien_so_trich_xuat_new_1.csv", help="Đường dẫn file CSV kết quả")
cli_args, _ = parser.parse_known_args()

model_candidates = [
    cli_args.model if cli_args.model else "",
    r"model\Wiliv5_best.pt",
    r"model\v62.pt",
    r"DSP1\Wiliv5_best.pt",
    r"archives\DSP1\Wiliv5_best.pt"
]
model_path = next((p for p in model_candidates if p and os.path.exists(p)), r"model\Wiliv5_best.pt")

model = YOLO(model_path)
ocr_model = LicensePlateRecognizer("cct-s-v2-global-model")

# 2. Cấu hình đường dẫn
input_video = cli_args.video if os.path.exists(cli_args.video) else (
    r"test\test_3.mp4" if os.path.exists(r"test\test_3.mp4") else cli_args.video
)

crop_dir = Path("outputs/extracted_plates")
crop_dir.mkdir(parents=True, exist_ok=True)

# 3. Các hàm bổ trợ chuẩn hóa biển số VN
VN_PROVINCE_CODES = {"11","12","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","43","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","88","89","90","92","93","94","95","97","98","99"}
DIGIT_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "7", "S": "5", "B": "8", "G": "6", "Z": "2"})

# Cấu trúc biển số xe cơ giới Việt Nam chuẩn
VN_CIVIL_PATTERN = re.compile(r"^[0-9]{2}[A-Z](?:[0-9A-Z])?[0-9]{4,5}$")
VN_MILITARY_PATTERN = re.compile(
    r"^(?:TM|TC|TH|TK|TT|TL|TN|TP|TQ|TS|TU|TB|QA|QB|QC|QD|QP|HA|HB|HC|HD|HE|HH|KA|KB|KC|KD|KP|VT|BB)[0-9]{4,6}$"
)

def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())

def normalize_plate(raw):
    text = clean_text(raw)
    if len(text) < 5:
        return text
    p1 = text[0].translate(DIGIT_FIX)
    p2 = text[1].translate(DIGIT_FIX)
    text = p1 + p2 + text[2:]
    return text

def is_valid_vietnamese_plate(plate: str) -> bool:
    """Kiểm tra định dạng biển số xe Việt Nam, lọc bỏ logo/tem dán như PMI."""
    if not plate or len(plate) < 7 or len(plate) > 10:
        return False
    if VN_MILITARY_PATTERN.match(plate):
        return True
    prov = plate[:2]
    if prov not in VN_PROVINCE_CODES:
        return False
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

# 4. Đọc Video & Trích xuất dữ liệu (dùng model.track & Counter đếm tần suất)
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

track_history = defaultdict(list)
untracked_history = []

frame_idx = 0
print(f"🚀 Đang tiến hành trích xuất biển số từ video '{input_video}' (Tối ưu độ nhạy phát hiện)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    # Xử lý cách frame linh hoạt (xét mỗi 2 frame để không bỏ lỡ xe lướt nhanh)
    if frame_idx % 2 != 0:
        continue

    timestamp_sec = round(frame_idx / fps, 1)
    
    # YOLO phát hiện & theo vết (track) với ngưỡng conf nhạy hơn (0.18) và kích thước 768
    results = model.track(frame, persist=True, conf=0.18, imgsz=768, verbose=False)
    
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            
            bw = x2 - x1
            bh = y2 - y1
            if bw < 35 or bh < 16:
                continue
            aspect_ratio = bw / max(1, bh)
            if not ((1.1 <= aspect_ratio <= 2.2) or (2.6 <= aspect_ratio <= 5.5)):
                continue

            # Crop vùng biển số có kèm lề an toàn
            crop = get_expanded_crop(frame, x1, y1, x2, y2, pad_pct=0.08)
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 35:
                continue
                
            # 1. Chuyển ảnh crop sang RGB (bắt buộc đối với fast_plate_ocr)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # 2. Dự đoán OCR kèm xác suất
            ocr_preds = ocr_model.run(crop_rgb, return_confidence=True)
            
            if ocr_preds:
                pred = ocr_preds[0]
                raw_text = getattr(pred, "plate", "").strip()
                
                # Chuẩn hóa biển số (lọc ký tự lạ, sửa nhầm số/chữ ở mã tỉnh)
                clean_plate = normalize_plate(raw_text)
                
                # Chỉ nhận biển số Việt Nam hợp lệ
                if is_valid_vietnamese_plate(clean_plate):
                    # 3. Tính độ tin cậy trung bình của các ký tự thực tế
                    if hasattr(pred, "char_probs") and pred.char_probs is not None and len(raw_text) > 0:
                        actual_probs = pred.char_probs[:len(raw_text)]
                        ocr_conf = float(actual_probs.mean())
                    else:
                        ocr_conf = 0.85
                    
                    # 4. Tính điểm tin cậy tổng hợp (Kết hợp Detection + OCR)
                    trust_score = round((0.4 * ocr_conf + 0.6 * det_conf) * 100, 1)
                    
                    item = {
                        "plate": clean_plate,
                        "trust_score": trust_score,
                        "timestamp": timestamp_sec,
                        "crop": crop.copy()
                    }
                    
                    if track_id is not None:
                        track_history[track_id].append(item)
                    else:
                        untracked_history.append(item)

cap.release()

# 5. Tổng hợp kết quả bằng Đếm Tần Suất (Majority Voting)
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

# Xử lý thêm các biển số nhận diện tốt nhưng không gán được track ID (không bỏ sót)
if untracked_history:
    plate_groups = defaultdict(list)
    for item in untracked_history:
        plate_groups[item["plate"]].append(item)
        
    already_found = {r["Biển số (Số đông nhất)"] for r in final_results}
    
    for plate, items in plate_groups.items():
        # Chỉ lấy các biển số hợp lệ xuất hiện từ 2 frame trở lên hoặc có độ tin cậy cao mà chưa có trong danh sách
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

# 6. Xuất kết quả ra màn hình & file CSV
df = pd.DataFrame(final_results)

print("\n" + "="*70)
print("📋 DANH SÁCH BIỂN SỐ XE TRÍCH XUẤT ĐƯỢC (ĐẾM TẦN SUẤT XUẤT HIỆN):")
print("="*70)
if not df.empty:
    print(df.to_string(index=False))
else:
    print("Không tìm thấy biển số xe nào hợp lệ trong video.")

output_csv = cli_args.csv
Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
try:
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n📁 Đã lưu file kết quả: '{output_csv}'")
except PermissionError:
    output_csv = "outputs/csv/danh_sach_bien_so_trich_xuat_new.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n📁 File cũ bị khóa, đã lưu ra file mới: '{output_csv}'")
