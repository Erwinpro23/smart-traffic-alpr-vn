import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from alpr.alpr_common import clean_text, get_expanded_crop, normalize_plate, plate_model_candidates, plate_vehicle_class


def test_clean_text_strips_separators_and_uppercases():
    assert clean_text("51g-671.46") == "51G67146"
    assert clean_text(None) == ""


def test_normalize_plate_fixes_province_digits_only():
    # first two chars are province digits: O->0, I->1; the series letter stays
    assert normalize_plate("5IG67146") == "51G67146"
    assert normalize_plate("OS-B12345") == "05B12345"


def test_normalize_plate_short_text_unchanged():
    assert normalize_plate("AB1") == "AB1"


def test_get_expanded_crop_pads_and_clips():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = get_expanded_crop(frame, 10, 10, 110, 60, pad_pct=0.10)
    # box 100x50, pad 10x5 -> (0..120, 5..65)
    assert crop.shape == (60, 120, 3)
    edge = get_expanded_crop(frame, 0, 0, 200, 100, pad_pct=0.10)
    assert edge.shape == (100, 200, 3)


def test_plate_vehicle_class_detects_motorcycle_series():
    # biển xe máy: 2 số tỉnh + series chữ+số + 5 số (59-G3 073.26)
    assert plate_vehicle_class("59G307326") == "motorcycle"
    assert plate_vehicle_class("50N215155") == "motorcycle"
    # biển ô tô (51A-072.04) và biển xe máy cũ 4 số (59-V1 3814) trùng mẫu 8 ký tự -> không kết luận
    assert plate_vehicle_class("51A07204") is None
    assert plate_vehicle_class("59V13814") is None
    assert plate_vehicle_class("51LD12345") is None
    assert plate_vehicle_class("") is None


def test_plate_model_candidates_engine_then_onnx_then_pt(tmp_path):
    c = plate_model_candidates(tmp_path, "wiliv5_768")
    assert c[:2] == [tmp_path / "wiliv5_768.engine", tmp_path / "weights" / "wiliv5_768.engine"]
    assert c[2:4] == [tmp_path / "wiliv5_768.onnx", tmp_path / "weights" / "wiliv5_768.onnx"]
    assert c[4:] == [tmp_path / "wiliv5_768.pt", tmp_path / "weights" / "wiliv5_768.pt"]


def test_plate_model_candidates_on_gpu_prefer_pt_over_onnx(tmp_path):
    c = plate_model_candidates(tmp_path, "v62", has_cuda=True)
    assert [p.suffix for p in c] == [".engine", ".engine", ".pt", ".pt", ".onnx", ".onnx"]


def test_ocr_model_kwargs_falls_back_to_global(monkeypatch, tmp_path):
    import alpr.alpr_common as alpr_common

    monkeypatch.setattr(alpr_common, "OCR_VN_MODEL", tmp_path / "missing.onnx")
    assert alpr_common.ocr_model_kwargs("vn") == {"hub_ocr_model": "cct-s-v2-global-model"}


def test_ocr_model_kwargs_global_by_default_vn_on_request(monkeypatch, tmp_path):
    import alpr.alpr_common as alpr_common

    m, c = tmp_path / "ocr_vn.onnx", tmp_path / "cfg.yaml"
    m.write_bytes(b"x"), c.write_text("x")
    monkeypatch.setattr(alpr_common, "OCR_VN_MODEL", m)
    monkeypatch.setattr(alpr_common, "OCR_VN_CONFIG", c)
    monkeypatch.delenv("OCR_MODEL", raising=False)
    assert alpr_common.ocr_model_kwargs() == {"hub_ocr_model": "cct-s-v2-global-model"}
    assert alpr_common.ocr_model_kwargs("vn")["onnx_model_path"] == m
    monkeypatch.setenv("OCR_MODEL", "vn")
    assert alpr_common.ocr_model_kwargs()["onnx_model_path"] == m
