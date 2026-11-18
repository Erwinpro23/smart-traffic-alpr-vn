# Model Weights Directory

Thư mục này dùng để lưu trữ các trọng số (weights / checkpoint) của các mô hình trong pipeline ALPR. Các file trọng số nhị phân (`.pt`, `.onnx`, `.engine`) không được đưa lên Git repository vì dung lượng lớn.

## Danh sách trọng số cần thiết

| Tên file | Mô tả | Nguồn / Huấn luyện |
|---|---|---|
| `weights/vehicle_det.pt` | YOLO11s phát hiện 4 loại phương tiện (car, motorcycle, bus, truck) | Huấn luyện từ `scripts/train_vehicle_det.py` hoặc tải từ Releases |
| `weights/v62.pt` (hoặc `v62.pt`) | YOLOv9s phát hiện biển số xe Việt Nam | Huấn luyện từ `notebooks/train_plate_detector.ipynb` hoặc tải từ Releases |
| `weights/ocr_vn.onnx` (tùy chọn) | Mô hình fast-plate-ocr fine-tune riêng cho biển Việt Nam | Huấn luyện từ `ocr_train/train.sh` |
| `weights/ocr_vn_plate_config.yaml` (tùy chọn) | Cấu hình cho mô hình OCR fine-tuned | Sinh ra trong quá trình export ONNX của OCR |

*Ghi chú:* Khi chạy lần đầu, mô hình OCR mặc định `cct-s-v2-global` sẽ được thư viện `fast-plate-ocr` tự động tải về bộ nhớ đệm.

## Tải trọng số chuẩn bị sẵn

Bạn có thể tải các weights đã được huấn luyện sẵn tại:
- **GitHub Releases:** `https://github.com/your-username/ALPR-Vietnamese/releases/tag/v0.1.0`
- **Hugging Face Hub:** `https://huggingface.co/your-username/alpr-vietnamese-weights`

Sau khi tải về, đặt các file `.pt` trực tiếp vào thư mục `weights/`.
