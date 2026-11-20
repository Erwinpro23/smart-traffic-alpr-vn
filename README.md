# ALPR-Vietnamese: Nhận diện phương tiện & biển số xe Việt Nam thời gian thực

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-purple.svg)](LICENSE)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Hệ thống end-to-end phát hiện phương tiện, bám vết và đọc biển số xe Việt Nam trên video giao thông thời gian thực (Real-time Video ALPR). Suy luận được tăng tốc trên GPU NVIDIA bằng CUDA Graphs và ONNX Runtime GPU.

---

## 🚀 Tính năng

1. **Phát hiện & phân loại phương tiện:** YOLO11s nhận diện 4 lớp (xe máy, ô tô, xe buýt, xe tải), kết hợp **ByteTrack** để bám vết từng xe qua các khung hình.
2. **Phát hiện biển số:** YOLOv9s (`Wiliv5.pt`) tìm biển số trên toàn khung hình ở 640/768 px, gồm cả biển vuông 2 dòng và biển dài 1 dòng.
3. **Tiền xử lý ảnh biển số:** đo độ nét (Laplacian variance), nắn phối cảnh, cân bằng sáng CLAHE, tách biển 2 dòng khi cần.
4. **OCR & chuẩn hoá cú pháp:** `fast-plate-ocr` (`cct-s-v2-global` hoặc bản fine-tune), kiểm tra theo 63 mã tỉnh/thành và sửa ký tự hay nhầm theo vị trí (O/0, I/1, B/8, S/5).
5. **Bỏ phiếu đa khung hình:** gộp kết quả đọc theo từng vị trí ký tự qua nhiều frame của mỗi xe, có trọng số theo độ nét và độ tin cậy, giúp loại biển rác.
6. **Hợp nhất xe & biển số:** điều chỉnh loại xe theo định dạng biển thực tế và gộp các track bị đứt khi trùng biển số.
7. **Dashboard web** xem video trực tiếp, danh sách biển số, loại xe và FPS. **Xuất CSV** khi xử lý video offline.

---

## 📊 Kiến trúc pipeline

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
                                 ▼
                    Nhận dạng ký tự (fast-plate-ocr)
                                 ▼
                    Kiểm tra cú pháp biển số VN
                       (63 mã tỉnh, luật ký tự)
                                 ▼
                    Bỏ phiếu theo vị trí ký tự
                                 ▼
                   Hợp nhất kết quả xe & biển số
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
          Dashboard web                  Xuất file CSV
      (http://localhost:5000)       (ID, loại xe, biển số)
```

---

## 📈 Kết quả thực nghiệm

Đánh giá trên 4 video quay từ cầu đi bộ do nhóm tự thu thập tại Việt Nam, chấm trên 130 biển số đã gán nhãn (chi tiết: [`evaluation/plate_reading.md`](evaluation/plate_reading.md)).

| Độ phân giải nguồn | Đọc đúng biển số | Tốc độ pipeline | Độ chính xác loại xe |
|---|---|---|---|
| **Full HD (1080p)** | 88 / 130 (67,7%) | 29 – 58 FPS | > 92% |
| **4K** | 112 / 130 (86,2%) | ~26 FPS | > 95% |

So với baseline:
- **Tốc độ:** nhanh hơn khoảng 3,2× so với pipeline 2 giai đoạn truyền thống nhờ CUDA Graph và ONNX Runtime GPU.
- **Biển rác:** lọc bỏ hơn 85% phát hiện giả (biển báo, lan can, tem xe) nhờ kiểm tra cú pháp và ngưỡng tin cậy.

> Kết quả đo trên tập video của nhóm. Với camera, góc quay và điều kiện ánh sáng khác, độ chính xác có thể thay đổi.

---

## 📂 Cấu trúc repository

```
smart-traffic-alpr-vn/
├── alpr/                    # Gói mã nguồn chính
│   ├── pipeline.py          # Lớp ALPRPipeline trung tâm
│   ├── vehicle_module.py    # Detector phương tiện & ByteTrack
│   ├── plate_preprocess.py  # Đo nét, nắn phối cảnh, tách biển
│   ├── plate_text.py        # Cú pháp biển số Việt Nam (63 tỉnh)
│   ├── fusion.py            # Quy tắc hợp nhất loại xe và biển số
│   ├── alpr_common.py       # Tải model, CUDA DLLs, CUDA Graphs
│   └── draw.py              # Vẽ HUD và bounding box
├── scripts/                 # Lệnh chạy chính
│   ├── stream_app.py        # Dashboard web thời gian thực
│   ├── run_video.py         # Chạy trên video, xuất CSV
│   ├── run_benchmark.py     # Đo tốc độ
│   ├── train_vehicle_det.py # Huấn luyện detector phương tiện
│   ├── eval_vehicle.py      # Đánh giá mAP detector phương tiện
│   └── baseline_plate_only.py # Baseline đọc biển không tracking
├── experiments/             # Thử nghiệm: OCR, độ bền, EigenCAM, so sánh tracker
├── data_tools/              # Chuẩn bị, kiểm tra, cân bằng dataset
├── configs/                 # sources.yaml (nguồn dữ liệu), augmentation.yaml (OCR)
├── ocr_train/               # Fine-tune & xuất ONNX mô hình OCR
├── evaluation/              # Ground truth & công cụ chấm điểm
├── notebooks/               # Huấn luyện detector biển số, EDA, demo pipeline
├── weights/                 # Đặt file model vào đây (không lưu trên Git)
├── tests/                   # Unit tests
├── requirements.txt
├── pyproject.toml
└── LICENSE                  # GNU AGPL-3.0
```

---

## 🛠️ Cài đặt

### Yêu cầu
- **Hệ điều hành:** Linux hoặc Windows 10/11
- **Python:** 3.10 – 3.12 (khuyên dùng 3.11 hoặc 3.12)
- **Phần cứng:** khuyến nghị GPU NVIDIA từ 4 GB VRAM, CUDA 12.x. Chạy được trên CPU nhưng chậm hơn nhiều.

### Các bước

**1. Clone repository**
```bash
git clone https://github.com/Erwinpro23/smart-traffic-alpr-vn.git
cd smart-traffic-alpr-vn
```

**2. Tạo môi trường ảo**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

**3. Cài PyTorch đúng phiên bản CUDA** (xem lệnh phù hợp tại [pytorch.org](https://pytorch.org/get-started/locally/)), ví dụ CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**4. Cài các thư viện còn lại**
```bash
pip install -r requirements.txt
```

> ⚠️ **ONNX Runtime:** chỉ cài **một** trong hai bản, không cài cả hai.
> - Có GPU NVIDIA: giữ `onnxruntime-gpu` như trong `requirements.txt`, rồi kiểm tra có `CUDAExecutionProvider`:
>   ```bash
>   python -c "import onnxruntime as o; print(o.get_available_providers())"
>   ```
> - Chỉ có CPU: `pip uninstall -y onnxruntime-gpu && pip install onnxruntime`

---

## 📦 Tải trọng số mô hình

File model không lưu trong Git. Tải tại **[GitHub Releases](https://github.com/Erwinpro23/smart-traffic-alpr-vn/releases)** và đặt vào thư mục `weights/`:

| File | Mô tả | Bắt buộc |
|---|---|---|
| `vehicle_det.pt` | YOLO11s phát hiện 4 loại phương tiện | ✅ |
| `Wiliv5.pt` | YOLOv9s phát hiện biển số Việt Nam | ✅ |
| `ocr_vn.onnx` + `ocr_vn_plate_config.yaml` | OCR fine-tune cho biển VN | Tuỳ chọn |

Nếu không có `ocr_vn.onnx`, chương trình dùng mô hình OCR mặc định `cct-s-v2-global`, được `fast-plate-ocr` tự tải ở lần chạy đầu.

Cấu trúc sau khi tải:
```
weights/
├── vehicle_det.pt
├── Wiliv5.pt
├── ocr_vn.onnx                 # tuỳ chọn
└── ocr_vn_plate_config.yaml    # tuỳ chọn
```

---

## 🚦 Hướng dẫn sử dụng

### 1. Xử lý video và xuất CSV

```bash
python scripts/run_video.py --video duong/dan/video.mp4 --out ket_qua.csv --full-frame
```

| Tham số | Ý nghĩa |
|---|---|
| `--video` | Đường dẫn video đầu vào |
| `--out` | File CSV kết quả (mặc định `ket_qua_xe_va_bien_so.csv`) |
| `--full-frame` | Xử lý toàn khung hình, không cắt vùng ROI của video demo |
| `--max-frames N` | Chỉ chạy N khung hình đầu (để thử nhanh) |

CSV gồm: ID xe, loại xe, biển số, độ tin cậy, thời điểm xuất hiện và đường dẫn ảnh biển số. Ảnh biển số cắt ra được lưu trong `extracted_plates/`.

### 2. Dashboard web thời gian thực

```bash
python scripts/stream_app.py --port 5000
```

Mở trình duyệt tại `http://localhost:5000`. Dashboard được viết bằng `http.server` có sẵn của Python, không cần cài thêm web framework.

| Tham số | Ý nghĩa |
|---|---|
| `--port` | Cổng web (mặc định 5000) |
| `--video` | Video phát đầu tiên (có thể đổi trên giao diện web) |
| `--show-window` | Mở thêm cửa sổ OpenCV trên máy |

Dashboard phát các video demo đặt tại những vị trí sau (video nào không có sẽ được bỏ qua):
```
<thư mục gốc repo>/
├── YTSave_YouTube_Camera-nhan-dang-bien-so-xe-luu-tru-hinh_Media_VYR-VysjGkQ_001_1080p.mp4   # video "demo" (cổng ra vào)
├── demovid_1080/   # IMG_8355.mp4, IMG_8357.mp4, IMG_8355_crop.mp4, IMG_8356_crop.mp4, IMG_8358_crop.mp4
└── demovid/        # bản 4K: IMG_8355.MOV, IMG_8357.MOV, IMG_8355_crop.mp4, IMG_8356_crop.mp4, IMG_8358_crop.mp4
```

### 3. Đo tốc độ (benchmark)

```bash
python scripts/run_benchmark.py
```

Script đo trên video demo cổng ra vào ở thư mục gốc (tên file như trên).

### 4. Chạy unit test

```bash
python -m pytest tests/ -q
```

Phần lớn test không cần GPU hay file model.

---

## 🎓 Huấn luyện & tuỳ biến mô hình

### 1. Detector phương tiện (YOLO11s)

Bước chuẩn bị dữ liệu tải dataset từ Roboflow và Kaggle (khai báo trong `configs/sources.yaml`), nên cần:
- File `.env` ở thư mục gốc chứa API key Roboflow:
  ```
  ROBOFLOW_API_KEY=your_roboflow_api_key
  ```
- Tài khoản Kaggle đã cấu hình API (file `kaggle.json`, xem [hướng dẫn của Kaggle](https://www.kaggle.com/docs/api)).
- Cài thêm: `pip install roboflow kaggle imagehash`

```bash
python -m data_tools.prepare_det_data          # tải & gộp dữ liệu vào datasets/vehicle_det/
python -m data_tools.balance                   # cân bằng lớp (repeat-factor sampling)
python scripts/train_vehicle_det.py --balanced --epochs 100 --batch 16
```

### 2. Detector biển số

Mở [`notebooks/train_plate_detector.ipynb`](notebooks/train_plate_detector.ipynb) trên Kaggle hoặc Google Colab (bật GPU và Internet). Sau khi huấn luyện, đổi tên model thu được thành `Wiliv5.pt` (hoặc tên bạn chọn) và đặt vào `weights/`.

### 3. Fine-tune OCR cho biển số Việt Nam

Cần chuẩn bị:
- Mô hình gốc `cct-s-v2-global` trong `ocr_train/base/`: `cct_s_v2_global.keras`, `cct_s_v2_global_model_config.yaml`, `cct_s_v2_global_plate_config.yaml`.
- Dữ liệu ảnh biển số kèm nhãn: chạy `python -m ocr_train.prepare_data` để tải 2 bộ dữ liệu Kaggle và tạo `datasets/ocr_vn/train.csv`, `val.csv`.

```bash
# 1. Huấn luyện (backend Keras = torch), 60 epoch, lưu vào runs/ocr_vn
bash ocr_train/train.sh 60 runs/ocr_vn

# 2. Bỏ optimizer khỏi checkpoint trước khi xuất
KERAS_BACKEND=torch python ocr_train/strip_optimizer.py runs/ocr_vn/<run>/best.keras   # -> best_clean.keras

# 3. Xuất ONNX (cần môi trường riêng có TensorFlow) -> weights/ocr_vn.onnx + weights/ocr_vn_plate_config.yaml
KERAS_BACKEND=tensorflow python ocr_train/export_onnx.py runs/ocr_vn/<run>/best_clean.keras ocr_train/base/cct_s_v2_global_plate_config.yaml
```

---

## ⚠️ Hạn chế

- Chưa có chức năng đo tốc độ xe hay phát hiện vi phạm giao thông.
- Biển số ở xa hoặc quá nhỏ trên video 1080p dễ đọc sai. Video 4K cho kết quả tốt hơn nhưng chậm hơn.
- Tốc độ công bố ở trên đạt được với GPU NVIDIA. Trên CPU, pipeline chậm hơn đáng kể.

---

## 📄 Giấy phép

Dự án phát hành theo giấy phép [GNU AGPL-3.0](LICENSE). Nếu bạn sửa đổi và cung cấp hệ thống dưới dạng dịch vụ qua mạng, bạn phải công khai mã nguồn phần đã sửa.

Nếu dùng dự án trong nghiên cứu, vui lòng trích dẫn theo [`CITATION.cff`](CITATION.cff).
