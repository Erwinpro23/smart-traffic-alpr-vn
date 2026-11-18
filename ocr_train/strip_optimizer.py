"""Lưu lại checkpoint train bằng backend torch mà không kèm optimizer, để môi trường TF (export_onnx.py) đọc được.

KERAS_BACKEND=torch python ocr_train/strip_optimizer.py runs/ocr_vn/<run>/best.keras  ->  best_clean.keras
"""
import sys
from pathlib import Path

import keras
from fast_plate_ocr.train.utilities.utils import _register_custom_keras

if __name__ == "__main__":
    src = Path(sys.argv[1])
    _register_custom_keras()
    model = keras.models.load_model(src, compile=False)
    model.save(src.with_name(src.stem + "_clean.keras"))
    print("ghi", src.with_name(src.stem + "_clean.keras"))
