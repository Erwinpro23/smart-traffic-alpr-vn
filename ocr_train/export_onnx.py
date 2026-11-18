"""Xuất mô hình .keras sang ONNX và chép vào weights/ (pipeline tự dùng, xem alpr_common.ocr_model_kwargs).

Checkpoint train bằng torch phải bỏ optimizer trước (ocr_train/strip_optimizer.py), rồi chạy bằng môi trường conda riêng có TensorFlow (backend torch không xuất được lớp blur-pool của CCT):
    KERAS_BACKEND=tensorflow /c/Users/admin/miniconda3/envs/fpo-export/python.exe ocr_train/export_onnx.py MODEL.keras PLATE_CONFIG.yaml
Trên Windows, NamedTemporaryFile mặc định giữ khoá file nên tf2onnx không ghi được; dùng delete=False.
"""
import functools
import shutil
import sys
import tempfile
from pathlib import Path

from click.testing import CliRunner

import fast_plate_ocr.cli.export as export_mod

ROOT = Path(__file__).resolve().parents[1]
export_mod.NamedTemporaryFile = functools.partial(tempfile.NamedTemporaryFile, delete=False)


def main(model, plate_config, out_name="ocr_vn"):
    save_dir = Path(tempfile.mkdtemp())
    res = CliRunner().invoke(export_mod.export, ["-m", model, "--plate-config-file", plate_config,
                                                 "--save-dir", str(save_dir), "--dynamic-batch"])
    if res.exit_code != 0:
        raise SystemExit(f"Xuất lỗi:\n{res.output}\n{res.exception!r}")
    onnx = next(save_dir.glob("*.onnx"))
    weights = ROOT / "weights"
    shutil.copy(onnx, weights / f"{out_name}.onnx")
    shutil.copy(plate_config, weights / f"{out_name}_plate_config.yaml")
    print("ghi", weights / f"{out_name}.onnx")


if __name__ == "__main__":
    main(*sys.argv[1:])
