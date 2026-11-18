# Đọc biển số trên 4 video cầu đi bộ (nhóm quay, 4K gốc)

## Nhãn

`plate_reading/gt_draft.py` chạy pipeline trên bản 4K gốc (`demovid/IMG_8355-8358.MOV`) và lưu crop biển của mọi bản ghi (253 crop).
Các crop được đọc bằng mắt (AI) trong `plate_reading/labels.py`:

- `SURE`: 130 biển đọc chắc từng ký tự (vid1 39, vid2 37, vid3 23, vid4 31).
- `GUESS`: 11 biển hơi mờ, đoán được. Không cấu hình nào đọc đúng biển nào trong nhóm này.
- Crop không đọc được hoặc không phải biển (lan can, biển báo "QUAN SÁT", chữ trên áo) không có nhãn.

**Thiên lệch.** Danh sách biển lấy từ những gì cấu hình `base` tìm được trên 4K. Vì vậy `base@4k` được lợi, và biển mà pipeline không bao giờ phát hiện không được tính.

## Chấm

Mỗi video được chấm trên tập chuỗi biển của các bản ghi:

| Cột | Ý nghĩa |
|---|---|
| đúng | Số biển `SURE` có trong kết quả |
| gần đúng | Chuỗi cách một biển có nhãn đúng 1 ký tự |
| khác | Chuỗi còn lại: rác, hoặc biển không kiểm được |

`plate_reading/ablate.py [cấu hình ...]` ghi kết quả vào `results/plate_reading.json`. Mọi cấu hình đều bật đủ luật ghép (fusion).

## Kết quả (tổng 4 video)

| Cấu hình | Nguồn | Đúng /130 | Gần đúng | Khác |
|---|---|---|---|---|
| base (code cũ) | 1080p | 80 | 22 | 137 |
| + cú pháp | 1080p | 80 | 24 | 77 |
| + phiếu theo cỡ ảnh | 1080p | 80 | 12 | 148 |
| + làm nét (CLAHE + unsharp) | 1080p | 81 | 24 | 140 |
| cú pháp + phiếu theo cỡ | 1080p | 79 | 16 | 87 |
| cả 3 | 1080p | 81 | 13 | 99 |
| base | 4K | 108 | 15 | 130 |
| **cú pháp** | **4K** | **110** | **13** | **76** |
| cả 3 | 4K | 105 | 13 | 91 |

FPS chỉ tính pipeline: khoảng 23–31 ở 4K. Tính cả giải mã HEVC 4K trên CPU thì còn 15–18.

## Kết luận (đợt 1: code cũ cộng 3 bước rời)

- **Độ phân giải nguồn là yếu tố chính.** Cùng code, 1080p đọc đúng 80 biển, 4K đọc đúng 108–110. Phần lớn biển sót ở 1080p là OCR nhầm 1–2 ký tự, ví dụ X→A, 5→3 ở mã tỉnh (50→30), H→M, 8→6. Các lỗi đó vẫn ra chuỗi đúng cú pháp, nên bộ lọc cú pháp không sửa được.
- **Cú pháp** giữ nguyên số biển đúng, còn số chuỗi rác giảm khoảng một nửa (vid2: từ 30–36 xuống 1–3).
- **Phiếu theo cỡ ảnh** thay đổi ±1–4 biển, nằm trong mức nhiễu. Bước này không có trong báo cáo, nên tắt.

## Pipeline theo báo cáo mục 5–6 (mặc định của demo)

Demo và báo cáo phải chạy cùng một pipeline, nên `PlateConfig()` mặc định bật đủ các bước xử lý biển mà báo cáo mô tả:

| Bước | Báo cáo | Code |
|---|---|---|
| Kiểm độ nét: phương sai Laplacian trên vùng biển; crop mất nét hẳn (< 50 ở cỡ 128×64) không gửi OCR | 6, Blur filtering | `plate_preprocess.sharpness` |
| Nắn phối cảnh: Otsu, contour sáng lớn nhất thành tứ giác, perspective transform; không có tứ giác thì giữ crop | 5, 6 | `plate_preprocess.rectify` |
| OCR CCT `cct-s-v2-global` | 14 | `alpr_common.load_ocr` |
| Đọc lại khi OCR tin < 0,8: ảnh CLAHE + unsharp; biển 2 dòng đọc thêm từng nửa rồi ghép | 2, 6 | `ALPRPipeline._retry` |
| Chọn ứng viên theo điểm cú pháp: 63 mã tỉnh, sửa O/0, I/1, B/8, S/5 theo vị trí, mỗi ký tự phải sửa trừ 0,1 điểm | 2, 5 | `plate_text.correct`, `_parse_joined` |
| Bỏ phiếu từng vị trí ký tự trên các lần đọc của 1 xe, trọng số = xác suất ký tự × độ nét | 5 | `pipeline.position_vote` |
| Trust = (0,4·OCR + 0,6·det) × (1 − 0,1·số ký tự phải sửa) | 5, 14 | `_read_plates` |
| Dừng đọc xe khi 3 lần đọc trùng (tối đa 5) | 5, 11 | `PlateVoter` |

Các phần báo cáo đề xuất nhưng mục 11/15 ghi là chưa cài đặt (dò biển trên vùng từng xe ở 768, phân loại biển theo màu nền, fine-tune OCR) không bật trong demo.

| Cấu hình | Nguồn | Đúng /130 | Gần đúng | Khác | FPS pipeline |
|---|---|---|---|---|---|
| cú pháp (demo trước đây) | 1080p | 80 | 24 | 77 | 13–18 |
| **theo báo cáo** | **1080p** | **88** | 17 | 92 | 16–28 |
| − nắn phối cảnh | 1080p | 88 | 17 | 92 | 16–30 |
| − đọc lại (làm nét, 2 dòng) | 1080p | 86 | 20 | 76 | 30–42 |
| − phiếu theo vị trí ký tự | 1080p | 80 | 24 | 93 | 17–30 |
| cú pháp (demo trước đây) | 4K | 110 | 13 | 76 | 23–31 |
| **theo báo cáo** | **4K** | **112** | 14 | 85 | 14–25 |

FPS chỉ tính pipeline và dao động theo máy đang bận, nên chỉ dùng để so thô.

- **Phiếu theo vị trí ký tự đóng góp nhiều nhất:** +8 biển ở 1080p. Nhiều xe có các lần đọc sai mỗi lần một ký tự khác nhau; ghép theo từng vị trí ra được chuỗi đúng, dù có khi không lần đọc nào đúng nguyên chuỗi.
- **Đọc lại** thêm 2 biển, nhưng thêm khoảng 16 chuỗi rác và làm chậm pipeline (khoảng 40 → 22 FPS ở 1080p), vì crop khó phải chạy OCR 2–3 lần.
- **Nắn phối cảnh** đổi khoảng 5–7% số lần đọc, nhưng biển chốt cuối cùng không đổi. Biển trong 4 video gần như nhìn thẳng. Lần cài đầu làm mất 8 biển, do hai nguyên nhân: vùng sáng của đường hoặc thân xe dính vào biển làm tứ giác méo, và ảnh nắn sát chữ không có lề. Đã sửa: tứ giác chạm mép crop thì bỏ, và ảnh nắn chừa lề 8%.
- **Độ nét** không tách được lần đọc rác khỏi lần đọc đúng: crop lan can đọc ra `11T1111` vẫn có phương sai khoảng 450, ngang ảnh nét. Vì vậy ngưỡng đặt thấp, chỉ loại ảnh mất nét hẳn như hình mờ trong báo cáo; độ nét chủ yếu dùng làm trọng số phiếu.

## OCR fine-tune cho biển VN (nhánh feat/ocr-finetune, không dùng trong demo)

Bắt đầu từ `cct-s-v2-global`, train trên 8.751 crop biển VN (Kaggle) cộng ảnh tổng hợp (`ocr_train/`). Training bị dừng ở epoch 22/60, lúc val plate_acc đạt 83,7% và đã chững. Mô hình lưu ở `weights/ocr_vn.onnx` (không đưa vào git) và chỉ chạy khi đặt `OCR_MODEL=vn`.

| | global | VN fine-tune |
|---|---|---|
| test phm (300): đúng cả biển | 88,0% | 89,3% |
| test topkek (645): đúng cả biển | 65,9% | 78,0% |
| 4 video 1080p + cú pháp: đúng /130 | 80 | 103 |
| 4 video 4K + cú pháp: đúng /130 | 110 | 114 |
| 4 video, cột "khác" (1080p / 4K) | 77 / 76 | 142 / 141 |

Mô hình VN đọc ra nhiều chuỗi rác hơn trên crop hỏng, kiểu `50H00000` hay `71TT11111` (khoảng 45 chuỗi có phần số là một chữ số lặp lại). Báo cáo ghi OCR là mô hình global chưa fine-tune (mục 14–15), nên demo vẫn dùng mô hình global.

## Lọc biển rác theo độ tin OCR (mặc định)

Bộ lọc cú pháp đã bỏ mọi chuỗi sai form, nên các chuỗi rác còn lại đều đúng form. Đó là chỗ OCR đọc bừa trên crop lan can, biển báo hoặc biển quá nhỏ, ví dụ `11T1111` hay `30D4111`. `junk_features.py` ghi đặc trưng của từng bản ghi (`results/junk_features_{1080,4k}.json`). Đặc trưng tách tốt nhất là độ tin OCR cao nhất trong các lần đọc của xe:

- Biển đúng có độ tin cao nhất ít nhất 0,908 ở 1080p và 0,863 ở 4K.
- Biển rác có trung vị khoảng 0,46.
- Trust score, mức đồng thuận và kiểu số lặp (`1111`) đều tách kém hơn.

`PlateConfig.min_best_conf = 0.85`: xe phải có ít nhất 1 lần đọc với độ tin ≥ 0,85 thì mới được chốt biển. Xe không đạt vẫn được đếm và có loại xe, chỉ để trống biển. Ngưỡng 0,90 sẽ mất 1 biển đúng ở 4K.

| Cấu hình | Nguồn | Đúng /130 | Gần đúng | Khác |
|---|---|---|---|---|
| theo báo cáo | 1080p | 88 | 17 | 92 |
| **+ lọc độ tin** | **1080p** | **88** | 16 | **17** |
| theo báo cáo | 4K | 112 | 14 | 85 |
| **+ lọc độ tin** | **4K** | **112** | 12 | **22** |

Ngưỡng được chọn trên chính 4 video này. Biên an toàn ở 1080p rộng (0,85 so với 0,908), nhưng ở 4K hẹp (0,85 so với 0,863).

## Tăng tốc và xe máy chạy chậm không ra biển

Đo trên 4 video cầu bản 1080p, cùng nhãn 130 biển.

| Cấu hình | Đúng /130 | Gần đúng | Khác | Ghi chú |
|---|---|---|---|---|
| mặc định trước đây (dò biển mọi khung) | 88 | 16 | 17 | |
| **bỏ khung không có xe cần đọc (`skip_idle`, nay là mặc định)** | **88** | 16 | 17 | cùng từng chuỗi biển |
| dò biển 1/2 khung | 71 | 16 | 13 | đã gỡ |
| dò biển 1/3 khung | 56 | 16 | 15 | đã gỡ |
| dò biển 1/5 khung | 48 | 14 | 14 | đã gỡ |
| xe chưa chốt đọc tiếp, thay lần đọc có biển nhỏ nhất | 85 | 14 | 16 | đã gỡ |
| xe chưa chốt đọc tiếp, thay lần đọc OCR kém nhất | 88 | 15 | 18 | đã gỡ |

- **Bỏ khung là mất biển.** Nhiều biển chỉ đọc được trong vài khung, lúc xe ở gần và biển chưa bị che. Vì vậy chỉ bỏ được những khung chắc chắn không cần dò: không có xe, hoặc mọi xe đã chốt biển hay đã đủ lượt đọc.
- **Đọc tiếp quá 5 lượt không giúp.** Xe không chốt được biển thường là do biển quá nhỏ, chứ không phải do thiếu lượt đọc.
- **Dashboard:** phần vẽ box, nén JPEG và ghi ảnh crop chạy ở luồng riêng, chỉ giữ khung mới nhất. Bảng đếm loại xe tính lại mỗi 15 khung. FPS hiển thị là số khung chia cho thời gian thật.

**Vì sao xe máy ở lâu trong khung vẫn không ra biển.** Xét 192 xe máy ở lại trong khung từ 1,5 giây trở lên (`pipeline.py` mặc định, bản 1080p):

| Video | Đọc được | Detector không thấy biển | Biển < 35×15 px | Crop mờ / OCR sai cú pháp | Bị lọc rác (OCR < 0,85) |
|---|---|---|---|---|---|
| 1 | 25 | 8 | 1 | 3 | 17 |
| 2 | 34 | 65 | 0 | 5 | 0 |
| 3 (bỏ làn xe máy) | 3 | 1 | 1 | 0 | 0 |
| 4 (bỏ làn xe máy) | 5 | 10 | 3 | 3 | 8 |

- **Video 2: biển quá nhỏ để detector thấy.** Detector biển chạy trên cả khung 1080p thu về 640 px. Biển xe máy ở đây rộng 24–31 px, sau khi thu chỉ còn khoảng 8–10 px. Lấy 5 xe như vậy, đưa crop từng xe vào cùng detector thì cả 5 đều thấy biển, 4 xe có độ tin 0,88. Dò biển trên vùng từng xe (báo cáo mục 11, chưa cài đặt) hoặc dùng bản 4K sẽ giải quyết được. Tuy vậy, biển rộng 25–30 px ở 1080p vẫn khó đọc đúng.
- **Video 1 và 4: biển rộng 35–54 px.** OCR ra chuỗi gần đúng nhưng độ tin cao nhất chỉ khoảng 0,8, dưới ngưỡng 0,85 của bộ lọc rác.
- **Detector không thấy biển ở video 1 và 4:** phần lớn là xe ở xa, cả xe chỉ rộng 26–83 px.
