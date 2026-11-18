# Contributing to ALPR-Vietnamese

Cảm ơn bạn đã quan tâm đến việc đóng góp cho dự án ALPR-Vietnamese! Dưới đây là các hướng dẫn để bạn có thể bắt đầu đóng góp mã nguồn hoặc báo lỗi một cách hiệu quả.

---

## 🛠️ Quy trình đóng góp (Workflow)

1. **Fork repository** về tài khoản GitHub cá nhân của bạn.
2. **Clone** fork về máy:
   ```bash
   git clone https://github.com/<your-username>/ALPR-Vietnamese.git
   cd ALPR-Vietnamese
   ```
3. **Tạo nhánh mới (branch)** mô tả tính năng hoặc lỗi bạn định xử lý:
   ```bash
   git checkout -b feature/ten-tinh-nang-moi
   # hoặc
   git checkout -b fix/ten-loi-can-sua
   ```
4. **Cài đặt môi trường phát triển:**
   ```bash
   pip install -r requirements.txt
   pip install pytest ruff
   ```
5. **Thực hiện thay đổi và kiểm thử:**
   - Viết unit tests tương ứng trong thư mục `tests/` nếu thêm tính năng mới.
   - Chạy kiểm thử:
     ```bash
     python -m pytest tests/
     ```
   - Kiểm tra định dạng mã nguồn:
     ```bash
     ruff check alpr/ scripts/
     ```
6. **Commit và Push:**
   - Viết commit message rõ ràng (theo chuẩn Conventional Commits, ví dụ: `feat: add adaptive contrast for night plates`, `fix: handle edge case in 2-line plate splitting`).
   - Push nhánh lên GitHub:
     ```bash
     git push origin feature/ten-tinh-nang-moi
     ```
7. **Tạo Pull Request (PR):**
   - Mở PR tới nhánh `main` của repository gốc kèm mô tả chi tiết các thay đổi.

---

## 📋 Tiêu chuẩn mã nguồn (Coding Standards)

- **Ngôn ngữ & Phiên bản:** Tương thích Python 3.10+.
- **Docstrings & Comments:** Mỗi hàm, lớp hoặc script cần có docstring giải thích rõ đầu vào (Args), đầu ra (Returns) và cách thức hoạt động.
- **Không commit file nặng:** Tuyệt đối không commit file trọng số (`.pt`, `.onnx`), video (`.mp4`), ảnh dung lượng lớn hoặc nhãn tạm thời vào git. Sử dụng `.gitignore`.

---

## 💬 Báo cáo lỗi (Issue Reporting)

Nếu gặp lỗi hoặc có đề xuất cải tiến:
- Mở **Issue** trên GitHub kèm thông tin:
  - Hệ điều hành (Windows / Linux / macOS)
  - Phiên bản Python & PyTorch
  - GPU & phiên bản CUDA (nếu có)
  - Lệnh đã chạy và Log lỗi đầy đủ (Traceback)
