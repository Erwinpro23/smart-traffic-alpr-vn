# Evaluation & Benchmarking Suite

Thư mục này chứa toàn bộ công cụ đánh giá, chấm điểm (evaluation harness) và dữ liệu Ground Truth cho pipeline nhận diện phương tiện và biển số xe Việt Nam.

## Cấu trúc thư mục

```
evaluation/
├── demo_gt_reviewed.csv       # [Canonical GT] Ground truth chuẩn đã qua rà soát
├── demo_gt.csv                # Ground truth cơ sở cho video cổng demo
├── build_gt.py                # Script tổng hợp nhãn ground truth
├── make_gt_draft.py           # Sinh bản nháp GT ban đầu từ tracking
├── review_kit.py              # Công cụ rà soát crop ảnh và gán nhãn
├── merge_review.py            # Gộp nhãn review và xuất GT chính thức
├── score_demo.py              # Chấm điểm 1 cấu hình pipeline trên video demo
├── scoring.py                 # Hàm tính toán IoU, mAP, độ chính xác loại xe, biển số
├── videos.py                  # Khai báo danh sách video benchmark
├── plate_reading.md           # Báo cáo chi tiết kết quả đọc biển trên 4 video cầu đi bộ
├── heldout/                   # Đánh giá trên tập held-out (giao lộ, ngõ hẹp)
└── archive/                   # Lưu trữ các file nháp & nhãn trung gian
    ├── gt_draft.csv
    ├── labels_part1.csv
    ├── labels_part2.csv
    └── labels_resolved.csv
```

---

## Ground Truth & Quy trình tạo dữ liệu kiểm chuẩn

File Ground Truth chính thức dùng cho benchmark là **`demo_gt_reviewed.csv`** (và `demo_gt.csv`).

### Quy trình tạo Ground Truth (`build_gt` → `review_kit` → `merge_review`):

```
+------------------+       +------------------+       +-------------------+       +-----------------------+
|  make_gt_draft   | ----> |    review_kit    | ----> |   merge_review    | ----> | demo_gt_reviewed.csv  |
| (Sinh bản nháp)  |       | (Rà soát nhãn)   |       | (Giải quyết xung  |       | (Ground Truth chuẩn)  |
|                  |       |                  |       |  đột & gộp nhãn)  |       |                       |
+------------------+       +------------------+       +-------------------+       +-----------------------+
```

1. **Khởi tạo bản nháp (`make_gt_draft.py`):**
   - Chạy pipeline cơ sở trên video, lấy các track có ít nhất 10 khung hình.
   - Trích xuất bounding box tham chiếu, crop ảnh xe và crop ảnh biển số rõ nhất.
2. **Rà soát độc lập (`review_kit.py`):**
   - Kiểm tra trực quan từng crop ảnh để xác định loại phương tiện thực tế và ký tự biển số.
   - Đánh dấu các track bị đứt gãy thuộc cùng một phương tiện vật lý.
3. **Gộp và chuẩn hóa (`merge_review.py` & `build_gt.py`):**
   - So sánh các bản rà soát độc lập, giải quyết các điểm không thống nhất (conflicts).
   - Xuất ra file chuẩn hóa cuối cùng `demo_gt_reviewed.csv` với độ chính xác cao nhất.

Các file nháp trung gian trong quá trình thực hiện được lưu trữ gọn gàng tại thư mục `evaluation/archive/`.

---

## Hướng dẫn chạy benchmark & chấm điểm

Chấm điểm một cấu hình pipeline trên video demo:

```bash
python -m evaluation.score_demo --name test_run --plate-model v62 --fusion expand_assoc plate_buffer merge_by_plate single_line_rule
```

Kết quả sẽ được tính toán qua `scoring.py` và lưu trữ dưới dạng JSON/CSV.

### Các chỉ số đánh giá:

| Chỉ số | Ý nghĩa |
|---|---|
| `type_acc` | Tỷ lệ phương tiện GT được nhận diện đúng loại xe (car, motorcycle, truck, bus). |
| `plate_acc` | Tỷ lệ phương tiện GT có biển đọc được mà pipeline đọc chính xác 100% ký tự. |
| `duplicates` | Số bản ghi trùng lặp cho cùng một phương tiện vật lý. |
| `false_records` | Số bản ghi phát hiện biển rác hoặc phương tiện không có trong GT. |
| `fps` | Tốc độ xử lý khung hình/giây của toàn bộ pipeline end-to-end. |
