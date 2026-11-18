import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR
DEMO_VIDEO = SCRIPT_DIR / "YTSave_YouTube_Camera-nhan-dang-bien-so-xe-luu-tru-hinh_Media_VYR-VysjGkQ_001_1080p.mp4"
# Video demo là bản quay màn hình phần mềm: cột bên phải (x >= ~0.83 W) là khung giao diện chứa ảnh biển cũ
DEMO_ROI = (0.0, 0.0, 0.83, 1.0)

# Chuẩn hóa biển số: 2 ký tự đầu là mã tỉnh nên luôn là chữ số
DIGIT_FIX = str.maketrans({
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
    "T": "7", "S": "5", "B": "8", "G": "6", "Z": "2"
})


def setup_cuda_dlls():
    """Nạp DLL cuDNN của torch cho ONNX Runtime trên Windows."""
    torch_lib_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib_dir):
        os.add_dll_directory(torch_lib_dir)
        os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")


class CudaGraphForward:
    """Chạy forward của model PyTorch bằng CUDA Graph, mỗi kích thước input một graph.

    Với batch 1, YOLO bị giới hạn bởi CPU phát hàng trăm lệnh nhỏ xuống GPU (~24 ms/frame trên máy này);
    phát lại graph chỉ còn ~2 ms và cho kết quả giống hệt.
    """

    def __init__(self, module):
        self.forward = module.forward
        self.graphs = {}  # (shape, dtype) -> (input tĩnh, graph, output tĩnh)

    def __call__(self, x, *args, **kwargs):
        import torch
        from torch.utils._pytree import tree_map

        if args or any(kwargs.values()) or not x.is_cuda:  # augment/embed hoặc CPU -> chạy thường
            return self.forward(x, *args, **kwargs)
        key = (tuple(x.shape), x.dtype)
        if key not in self.graphs:
            self.graphs[key] = self._capture(x)
        static_x, graph, static_y = self.graphs[key]
        static_x.copy_(x)
        graph.replay()
        # Clone vì lần replay sau sẽ ghi đè output tĩnh
        return tree_map(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, static_y)

    def _capture(self, x):
        import torch

        static_x = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warmup trên stream phụ trước khi capture
            for _ in range(3):
                self.forward(static_x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = self.forward(static_x)
        return static_x, graph, static_y


def enable_cuda_graphs(yolo):
    """Gắn CUDA Graph vào model PyTorch của predictor YOLO. Gọi sau lần predict/track đầu tiên
    (predictor được tạo lúc đó); gọi lại nhiều lần không sao. Model ONNX/TensorRT được bỏ qua."""
    backend = getattr(getattr(yolo.predictor, "model", None), "backend", None)
    module = getattr(backend, "model", None)
    if module is None or not hasattr(module, "forward") or isinstance(module.forward, CudaGraphForward):
        return
    if next(module.parameters()).is_cuda:
        module.forward = CudaGraphForward(module)


def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def normalize_plate(raw):
    text = clean_text(raw)
    if len(text) < 5:
        return text
    return text[0].translate(DIGIT_FIX) + text[1].translate(DIGIT_FIX) + text[2:]


# Biển xe máy VN: 2 số tỉnh + series chữ+số + 5 số (59-G3 073.26). Ô tô không có dạng 9 ký tự này.
MOTO_PLATE = re.compile(r"\d{2}[A-Z]\d\d{5}")


def plate_vehicle_class(plate):
    """'motorcycle' nếu biển chắc chắn là biển xe máy, None nếu không kết luận được."""
    return "motorcycle" if plate and MOTO_PLATE.fullmatch(plate) else None


def get_expanded_crop(frame, x1, y1, x2, y2, pad_pct=0.08):
    h, w = frame.shape[:2]
    pad_w, pad_h = int((x2 - x1) * pad_pct), int((y2 - y1) * pad_pct)
    return frame[max(0, y1 - pad_h):min(h, y2 + pad_h), max(0, x1 - pad_w):min(w, x2 + pad_w)]


def plate_model_candidates(script_dir, stem, has_cuda=False):
    """Thứ tự thử: engine -> onnx -> pt (CPU) hoặc engine -> pt -> onnx (GPU: .pt chạy CUDA Graph nhanh hơn
    ONNX Runtime ~3 lần); mỗi loại tìm ở thư mục gốc rồi weights/."""
    script_dir = Path(script_dir)
    exts = (".engine", ".pt", ".onnx") if has_cuda else (".engine", ".onnx", ".pt")
    return [d / f"{stem}{ext}" for ext in exts for d in (script_dir, script_dir / "weights")]


def load_plate_model(script_dir, has_cuda, stem="v62"):
    """Ưu tiên TensorRT engine, rồi PyTorch FP16 + CUDA Graph (GPU) hoặc ONNX Runtime (CPU)."""
    from ultralytics import YOLO

    for path in plate_model_candidates(script_dir, stem, has_cuda):
        if not path.exists():
            continue
        print(f"[Plate] Nạp mô hình: '{path}'")
        if path.suffix != ".pt":
            return YOLO(str(path), task="detect")
        model = YOLO(str(path))
        if has_cuda:
            model.to("cuda:0")  # FP32: bản FP16 đọc sai thêm 1 biển trên video demo; CUDA Graph đã đủ nhanh
        return model
    raise FileNotFoundError(f"Không tìm thấy weights biển số '{stem}' (.engine/.onnx/.pt) trong {script_dir} hoặc weights/")


OCR_VN_MODEL = SCRIPT_DIR / "weights" / "ocr_vn.onnx"
OCR_VN_CONFIG = SCRIPT_DIR / "weights" / "ocr_vn_plate_config.yaml"


def ocr_model_kwargs(choice=None):
    """Tham số LicensePlateRecognizer. Mặc định mô hình global cct-s-v2 như báo cáo (mục 14).
    choice (hoặc biến môi trường OCR_MODEL) = "vn" dùng mô hình fine-tune biển VN (ocr_train/) nếu có file."""
    choice = choice or os.environ.get("OCR_MODEL", "global")
    if choice == "vn" and OCR_VN_MODEL.exists() and OCR_VN_CONFIG.exists():
        return {"onnx_model_path": OCR_VN_MODEL, "plate_config_path": OCR_VN_CONFIG}
    return {"hub_ocr_model": "cct-s-v2-global-model"}


def load_ocr(has_cuda, choice=None):
    from fast_plate_ocr import LicensePlateRecognizer

    kwargs = ocr_model_kwargs(choice)
    print(f"[OCR] Mô hình: {kwargs.get('onnx_model_path') or kwargs['hub_ocr_model']}")
    device = "cuda" if has_cuda else "cpu"
    try:
        return LicensePlateRecognizer(device=device, **kwargs)
    except Exception as e:
        print(f"[OCR] CPU fallback: {e}")
        return LicensePlateRecognizer(device="cpu", **kwargs)
