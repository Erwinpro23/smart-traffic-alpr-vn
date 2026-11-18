import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os
import re
import cv2
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# ==============================================================================
# 1. CẤU HÌNH & CHUẨN HÓA BIỂN SỐ VIỆT NAM (TỪ test_video_ocr_majority_voting.py)
# ==============================================================================
DIGIT_FIX = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "T": "7", "S": "5",
    "B": "8", "G": "6",
    "Z": "2"
})

def clean_text(text):
    """Lọc bỏ các ký tự đặc biệt, chỉ giữ chữ cái in hoa và chữ số."""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())

def normalize_plate(raw):
    """
    Chuẩn hóa biển số Việt Nam:
    - Sửa lỗi OCR nhầm lẫn chữ/số ở 2 ký tự mã tỉnh đầu tiên (VD: S->5, B->8, O/D->0, Z->2).
    """
    text = clean_text(raw)
    if len(text) < 5:
        return text
    
    p1 = text[0].translate(DIGIT_FIX)
    p2 = text[1].translate(DIGIT_FIX)
    text = p1 + p2 + text[2:]
    return text

VN_PROVINCE_CODES = {
    "11","12","14","15","16","17","18","19","20","21","22","23","24","25","26",
    "27","28","29","30","31","32","33","34","35","36","37","38","39","40","41",
    "43","47","48","49","50","51","52","53","54","55","56","57","58","59","60",
    "61","62","63","64","65","66","67","68","69","70","71","72","73","74","75",
    "76","77","78","79","80","81","82","83","84","85","86","88","89","90","92",
    "93","94","95","97","98","99"
}

VN_CIVIL_PATTERN = re.compile(r"^[0-9]{2}[A-Z](?:[0-9A-Z])?[0-9]{4,5}$")
VN_MILITARY_PATTERN = re.compile(
    r"^(?:TM|TC|TH|TK|TT|TL|TN|TP|TQ|TS|TU|TB|QA|QB|QC|QD|QP|HA|HB|HC|HD|HE|HH|KA|KB|KC|KD|KP|VT|BB)[0-9]{4,6}$"
)

def is_valid_vietnamese_plate(plate: str) -> bool:
    """Kiểm tra định dạng biển số xe Việt Nam, loại bỏ logo/tem dán như PMI."""
    if not plate or len(plate) < 7 or len(plate) > 10:
        return False
    if VN_MILITARY_PATTERN.match(plate):
        return True
    prov = plate[:2]
    if prov not in VN_PROVINCE_CODES:
        return False
    return bool(VN_CIVIL_PATTERN.match(plate))

def get_expanded_crop(frame, x1, y1, x2, y2, pad_pct=0.08):
    """Mở rộng bounding box một chút để không bị lẹm viền biển số khi OCR."""
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

def find_best_model():
    """Tự động tìm đường dẫn model tốt nhất (Wiliv5_best / v62 .pt hoặc .onnx)."""
    candidate_paths = [
        "model/Wiliv5_best.pt",
        "model/v62.pt",
        "model/v62.onnx",
        "Wiliv5_best.pt",
        "v62.pt",
        "v62.onnx"
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

# ==============================================================================
# 2. HÀM TEST OCR CHO 1 ẢNH ĐẦU VÀO
# ==============================================================================
def process_single_image(image_path, model_path=None, conf_thresh=0.25, output_dir="outputs/extracted_plates"):
    # --- 1. Đọc và kiểm tra file ảnh ---
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: Không tìm thấy file ảnh tại '{image_path}'")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Lỗi: Không thể mở ảnh '{image_path}'. Vui lòng kiểm tra lại định dạng file.")
        return None

    # --- 2. Nạp mô hình YOLO (v62.pt / v62.onnx) ---
    if not model_path:
        model_path = find_best_model()
        if not model_path:
            print("❌ Lỗi: Không tìm thấy weights 'model/v62.pt' hoặc 'model/v62.onnx' trong thư mục project!")
            return None

    has_cuda = torch.cuda.is_available()
    device = "cuda:0" if has_cuda else "cpu"
    
    print("=" * 70)
    print(f"🚀 Model YOLO  : '{model_path}'")
    print(f"📷 Ảnh đầu vào : '{image_path}'")
    print(f"⚡ Thiết bị     : {device.upper()}")
    print("=" * 70)

    if model_path.endswith(".onnx"):
        model = YOLO(model_path, task="detect")
    else:
        model = YOLO(model_path)
        if has_cuda:
            model.to(device)

    # --- 3. Nạp mô hình FastPlateOCR ---
    ocr_device = "cuda" if has_cuda else "cpu"
    try:
        ocr_model = LicensePlateRecognizer("cct-s-v2-global-model", device=ocr_device)
    except Exception:
        ocr_model = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")

    # Tạo thư mục chứa ảnh biển số output
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # --- 4. Chạy nhận diện biển số ---
    results = model.predict(img, conf=conf_thresh, imgsz=768, verbose=False)
    
    detected_plates = []
    
    # Trường hợp 1: YOLO phát hiện ra biển số từ ảnh full
    found_boxes = False
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            found_boxes = True
            for idx, box in enumerate(r.boxes, start=1):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                det_conf = float(box.conf[0])

                crop = get_expanded_crop(img, x1, y1, x2, y2, pad_pct=0.08)
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ocr_preds = ocr_model.run(crop_rgb, return_confidence=True)

                raw_text = ""
                clean_plate = ""
                ocr_conf = 0.85

                if ocr_preds:
                    pred = ocr_preds[0]
                    raw_text = getattr(pred, "plate", "").strip()
                    clean_plate = normalize_plate(raw_text)

                if not is_valid_vietnamese_plate(clean_plate):
                    continue

                if hasattr(pred, "char_probs") and pred.char_probs is not None and len(raw_text) > 0:
                    actual_probs = pred.char_probs[:len(raw_text)]
                    ocr_conf = float(actual_probs.mean())

                plate_name = clean_plate if clean_plate else (raw_text if raw_text else "UNKNOWN_PLATE")
                trust_score = round((0.4 * ocr_conf + 0.6 * det_conf) * 100, 1)

                # Lưu ảnh biển số crop với tên file là chuỗi OCR nhận diện được
                save_filename = f"{plate_name}.jpg"
                save_filepath = output_path / save_filename
                
                # Nếu bị trùng tên (do nhiều biển giống nhau), đánh số thứ tự
                dup_counter = 1
                while save_filepath.exists():
                    save_filepath = output_path / f"{plate_name}_{dup_counter}.jpg"
                    dup_counter += 1

                cv2.imwrite(str(save_filepath), crop)

                detected_plates.append({
                    "STT": idx,
                    "Biển số OCR": plate_name,
                    "Độ tin cậy (%)": f"{trust_score}%",
                    "File ảnh biển số đã lưu": str(save_filepath)
                })

    # Trường hợp 2: Nếu ảnh đầu vào đã là ảnh crop sẵn của biển số (YOLO không detect ra box nữa)
    if not found_boxes:
        print("ℹ️ YOLO không tìm thấy bounding box (có thể ảnh đầu vào đã là ảnh crop biển số). Đang đưa thẳng vào FastPlateOCR...")
        crop_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ocr_preds = ocr_model.run(crop_rgb, return_confidence=True)
        if ocr_preds:
            pred = ocr_preds[0]
            raw_text = getattr(pred, "plate", "").strip()
            clean_plate = normalize_plate(raw_text)
            plate_name = clean_plate if clean_plate else (raw_text if raw_text else "UNKNOWN_PLATE")

            save_filepath = output_path / f"{plate_name}.jpg"
            cv2.imwrite(str(save_filepath), img)

            detected_plates.append({
                "STT": 1,
                "Biển số OCR": plate_name,
                "Độ tin cậy (%)": "100%",
                "File ảnh biển số đã lưu": str(save_filepath)
            })

    # --- 5. In kết quả trực quan ra màn hình ---
    print("\n" + "=" * 70)
    print("✨ KẾT QUẢ OCR BIỂN SỐ XE ✨")
    print("=" * 70)

    if detected_plates:
        df = pd.DataFrame(detected_plates)
        print(df.to_string(index=False))
        print("\n" + "-" * 70)
        for p in detected_plates:
            print(f"🎉 BIỂN SỐ NHẬN DIỆN:  [ {p['Biển số OCR']} ]")
            print(f"🖼️ FILE ẢNH ĐÃ LƯU  :  {p['File ảnh biển số đã lưu']}")
        print("-" * 70)
    else:
        print("❌ Không nhận diện được biển số từ ảnh đầu vào.")

    return detected_plates

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH (CLI)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OCR biển số xe từ 1 ảnh đầu vào -> Xuất ảnh crop tên theo biển số OCR.")
    parser.add_argument("-i", "--image", type=str, help="Đường dẫn tới ảnh cần test (VD: anh_xe.jpg).")
    parser.add_argument("-m", "--model", type=str, default=find_best_model(), help="Đường dẫn file weights YOLO (Mặc định: tự tìm trong model/).")
    parser.add_argument("-c", "--conf", type=float, default=0.25, help="Ngưỡng tin cậy YOLO (Mặc định 0.25).")
    parser.add_argument("-o", "--output-dir", type=str, default="outputs/extracted_plates", help="Thư mục lưu ảnh biển số đầu ra (Mặc định: outputs/extracted_plates).")

    args = parser.parse_args()

    image_input = args.image
    if not image_input:
        print("\n👉 Bạn chưa truyền tham số ảnh.")
        image_input = input("Vui lòng nhập hoặc kéo thả đường dẫn ảnh vào đây: ").strip().strip('"').strip("'")

    if image_input:
        process_single_image(
            image_path=image_input,
            model_path=args.model,
            conf_thresh=args.conf,
            output_dir=args.output_dir
        )
