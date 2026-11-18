# ALPR-Vietnamese: End-to-End Vietnamese Vehicle & License Plate Recognition

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-purple.svg)](LICENSE)
[![CUDA Supported](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Hệ thống nhận diện phương tiện và đọc biển số xe Việt Nam end-to-end trên luồng video giao thông thời gian thực (Real-time Video ALPR), tối ưu hóa tăng tốc suy luận trên GPU NVIDIA (CUDA Graphs & ONNX Runtime GPU).

![Demo Results](docs/demo.jpg)

---

## 🚀 Tính năng nổi bật

1. **Phát hiện & Phân loại phương tiện (Vehicle Detection)**: Mô hình YOLO11s nhận diện 4 lớp phương tiện giao thông phổ biến tại Việt Nam (xe máy, ô tô, xe buýt, xe tải), kết hợp thuật toán **ByteTrack** bám vết liên tục qua từng khung hình.
2. **Phát hiện biển số (License Plate Detection)**: YOLOv9s (`v62`) dò tìm biển số xe (biển vuông 2 dòng, biển dài 1 dòng) trên toàn bộ khung hình ở độ phân giải 640/768 px.
3. **Xử lý ảnh biển số thích ứng (Plate Preprocessing)**: Đánh giá độ nét (Laplacian variance), nắn góc phối cảnh (perspective rectification), cân bằng sáng thích ứng (CLAHE) và phân tách biển 2 dòng khi cần.
4. **Đọc biển số & Tinh chỉnh cú pháp (OCR & Syntax Correction)**: Sử dụng `fast-plate-ocr` (`cct-s-v2-global` / fine-tuned model), tự động chuẩn hóa dựa trên quy chuẩn 63 mã tỉnh/thành Việt Nam, cơ chế sửa nhầm ký tự theo vị trí (O/0, I/1, B/8, S/5).
5. **Bỏ phiếu đa khung hình (Temporal Majority Voting)**: Tổng hợp kết quả đọc biển số theo vị trí từng ký tự qua nhiều frame của từng phương tiện, kết hợp trọng số độ nét và độ tin cậy để loại bỏ biển rác và chống rung lắc.
6. **Hợp nhất đa phương thức (Vehicle-Plate Fusion)**: Tự động điều chỉnh phân loại xe dựa trên định dạng biển số thực tế và gộp các track bị đứt gãy khi có cùng biển số.

---

## 📊 Sơ đồ kiến trúc Pipeline

```
                       LUỒNG VIDEO GIAO THÔNG
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Phát hiện phương tiện             Phát hiện biển số
       (YOLO11s + ByteTrack)           (YOLOv9s v62 / 640px)
                 │                               │
                 ▼                               ▼
          Theo vết xe (Track)          Gán biển số vào xe
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     Tiền xử lý ảnh biển số
              (Đo độ nét, nắn góc, tách 2 dòng)
                                 │
                                 ▼
                        Nhận dạng ký tự (OCR)
                         (fast-plate-ocr)
                                 │
                                 ▼
                    Kiểm tra cú pháp biển số VN
                    (63 mã tỉnh, luật ký tự)
                                 │
                                 ▼
                    Bỏ phiếu theo vị trí ký tự
                 (Temporal Position-Level Voting)
                                 │
                                 ▼
                   Hợp nhất kết quả xe & biển số
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
        Dashboard Web (Flask)            Xuất file CSV báo cáo
          (http://localhost:5000)          (ID, Loại xe, Biển số)
```

---

## 📈 Kết quả thực nghiệm

Đánh giá trên 4 video cầu đi bộ thực tế do nhóm thu thập tại Việt Nam (chấm trên 130 biển số đã gán nhãn Ground Truth chi tiết tại [`evaluation/plate_reading.md`](evaluation/plate_reading.md)):

| Độ phân giải nguồn | Tỷ lệ đọc đúng biển số | Tốc độ Pipeline (FPS) | Độ chính xác loại xe |
|---|---|---|---|
| **Full HD (1080p)** | **88 / 130** (67.7%) | **29 – 58 FPS** | **> 92%** |
| **Ultra HD (4K)** | **112 / 130** (86.2%) | **~26 FPS** | **> 95%** |

So sánh với các giải pháp Baseline:
- **Tốc độ:** Nhanh hơn 3.2× so với mô hình phát hiện 2 giai đoạn truyền thống nhờ tích hợp CUDA Graph và ONNX Runtime GPU.
- **Biển rác:** Lọc bỏ hơn 85% phát hiện giả (biển báo, lan can, tem xe) nhờ cơ chế kiểm tra cú pháp và ngưỡng tin cậy tối thiểu.

---

## 📂 Cấu trúc Repository

```
ALPR-Vietnamese/
├── alpr/                    # Gói mã nguồn chính (core pipeline & modules)
│   ├── pipeline.py          # Lớp ALPRPipeline trung tâm
│   ├── vehicle_module.py    # Detector phương tiện & ByteTrack
│   ├── plate_preprocess.py  # Đo nét, nắn phối cảnh, tách biển
│   ├── plate_text.py        # Cú pháp biển số Việt Nam (63 tỉnh)
│   ├── fusion.py            # Quy tắc hợp nhất loại xe và biển số
│   ├── alpr_common.py       # Tải model, CUDA DLLs, CUDA Graphs
│   └── draw.py              # Hiển thị HUD và vẽ bounding box
├── scripts/                 # Các công cụ dòng lệnh (CLI entrypoints)
│   ├── stream_app.py        # Dashboard web giám sát thời gian thực
│   ├── run_video.py         # Chạy nhận diện trên video và xuất CSV
│   ├── run_benchmark.py     # Đo tốc độ và benchmark throughput
│   ├── train_vehicle_det.py # Huấn luyện mô hình nhận diện xe YOLO11
│   ├── eval_vehicle.py      # Đánh giá mAP detector phương tiện
│   └── baseline_plate_only.py # Baseline đọc biển số không tracking
├── experiments/             # Các kịch bản nghiên cứu & thử nghiệm mở rộng
│   ├── ocr_eval.py          # Đánh giá CER, Accuracy, McNemar test của OCR
│   ├── robustness.py        # Kiểm tra độ bền trước nhiễu, mờ, thiếu sáng
│   ├── eigencam.py          # Trực quan hóa bản đồ chú ý (EigenCAM)
│   └── tracker_compare.py   # So sánh hiệu năng các thuật toán tracking
├── data_tools/              # Công cụ thu thập và xử lý dataset
│   ├── balance.py           # Thuật toán repeat-factor sampling cân bằng lớp
│   ├── prepare_det_data.py  # Chuẩn bị dữ liệu từ Roboflow/Kaggle
│   └── check_data.py        # Kiểm tra tính toàn vẹn nhãn YOLO
├── configs/                 # File cấu hình YAML
│   ├── sources.yaml         # Khai báo nguồn dữ liệu huấn luyện
│   └── augmentation.yaml    # Cấu hình augmentation cho OCR
├── ocr_train/               # Huấn luyện & fine-tune mô hình OCR
│   ├── train.sh             # Script chạy huấn luyện fast-plate-ocr
│   ├── prepare_data.py      # Xử lý nhãn ảnh biển số
│   └── export_onnx.py       # Xuất mô hình sang định dạng ONNX
├── evaluation/              # Bộ công cụ đánh giá & dữ liệu Ground Truth
│   ├── demo_gt_reviewed.csv # Ground truth chuẩn sau rà soát
│   ├── score_demo.py        # Chấm điểm cấu hình trên video
│   └── scoring.py           # Tính toán mAP, IoU, độ chính xác
├── notebooks/               # Jupyter Notebooks tương tác
│   ├── train_plate_detector.ipynb      # Huấn luyện detector biển số
│   ├── eda_dataset.ipynb               # Phân tích khám phá dữ liệu (EDA)
│   └── pipeline_vehicle_ocr_counting.ipynb # Demo pipeline từng bước
├── docs/                    # Tài liệu kỹ thuật, báo cáo & hình ảnh minh họa
├── weights/                 # Thư mục chứa trọng số mô hình (.gitkeep)
├── tests/                   # Bộ kiểm thử tự động (Unit tests)
├── requirements.txt         # Khai báo các thư viện phụ thuộc
├── pyproject.toml           # Cấu hình project & công cụ kiểm tra mã nguồn
├── pytest.ini               # Cấu hình kiểm thử pytest
└── LICENSE                  # Giấy phép mã nguồn mở (GNU AGPLv3)
```

---

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- **Hệ điều hành:** Linux hoặc Windows 10/11
- **Python:** 3.10 trở lên (khuyên dùng Python 3.11 hoặc 3.12)
- **Phần cứng:** Khuyến nghị GPU NVIDIA có VRAM tối thiểu 4GB hỗ trợ CUDA 12.x (vẫn có thể chạy trên CPU nhưng tốc độ sẽ thấp hơn).

### Hướng dẫn cài đặt từng bước

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/ALPR-Vietnamese.git
   cd ALPR-Vietnamese
   ```

2. **Tạo môi trường ảo:**
   ```bash
   python -m venv .venv
   # Trên Windows:
   .venv\Scripts\activate
   # Trên Linux/macOS:
   source .venv/bin/activate
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

> ⚠️ **Lưu ý quan trọng về GPU:** Để OCR chạy với hiệu năng cao nhất, chỉ cài đặt `onnxruntime-gpu`, không cài `onnxruntime` (bản CPU). Kiểm tra sự tồn tại của `CUDAExecutionProvider`:
> ```bash
> python -c "import onnxruntime as o; print(o.get_available_providers())"
> ```

---

## 📦 Tải Trọng số Mô hình (Weights)

Các file trọng số mô hình lớn được lưu trữ tại GitHub Releases. Tải về và đặt vào thư mục `weights/`:

| Tên file | Mô tả | Vị trí lưu |
|---|---|---|
| `vehicle_det.pt` | YOLO11s phát hiện phương tiện 4 lớp | `weights/vehicle_det.pt` |
| `v62.pt` | YOLOv9s phát hiện biển số Việt Nam | `weights/v62.pt` |
| `ocr_vn.onnx` *(tùy chọn)* | Fast-plate-ocr fine-tune cho biển số VN | `weights/ocr_vn.onnx` |

👉 **Link tải trọn bộ weights:** [Tải từ GitHub Releases v0.1.0](https://github.com/your-username/ALPR-Vietnamese/releases/tag/v0.1.0) hoặc xem hướng dẫn chi tiết tại [`weights/README.md`](weights/README.md).

---

## 🚦 Hướng dẫn sử dụng

### 1. Giao diện Web Dashboard thời gian thực

Khởi chạy web server giám sát:
```bash
python scripts/stream_app.py --port 5000
```
Truy cập trình duyệt tại: `http://localhost:5000` để theo dõi video trực tiếp, danh sách biển số đã trích xuất, phân loại phương tiện và tốc độ khung hình (FPS).

### 2. Xử lý video offline và xuất báo cáo CSV

Chạy nhận diện trên một file video bất kỳ:
```bash
python scripts/run_video.py --video path/to/traffic_video.mp4 --out ket_qua.csv
```
Kết quả được xuất ra file CSV với đầy đủ: ID xe, loại xe, biển số, độ tin cậy, mốc thời gian xuất hiện và đường dẫn ảnh crop biển số.

### 3. Đo tốc độ và kiểm chuẩn (Benchmark)

```bash
python scripts/run_benchmark.py --video path/to/traffic_video.mp4
```

### 4. Chạy bộ kiểm thử tự động (Unit Tests)

```bash
python -m pytest tests/ -q
```

---

## 🎓 Huấn luyện & Tùy biến mô hình

### 1. Huấn luyện Detector Phương tiện (YOLO11s)
- Chuẩn bị dữ liệu bằng `python -m data_tools.prepare_det_data`
- Cân bằng lớp mẫu: `python -m data_tools.balance`
- Bắt đầu huấn luyện:
  ```bash
  python scripts/train_vehicle_det.py --balanced --epochs 100 --batch 16
  ```

### 2. Huấn luyện Detector Biển số
Mở notebook tương tác [`notebooks/train_plate_detector.ipynb`](notebooks/train_plate_detector.ipynb) để huấn luyện trên Google Colab / Kaggle GPU.

### 3. Fine-tune mô hình OCR cho biển số Việt Nam
```bash
bash ocr_train/train.sh 60 runs/ocr_vn
```

---

## 📄 Bản quyền (License)

Dự án được phân phối dưới giấy phép [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).
