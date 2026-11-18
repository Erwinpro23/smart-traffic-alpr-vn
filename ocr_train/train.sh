#!/usr/bin/env bash
# Fine-tune cct-s-v2-global trên biển VN. Chạy từ thư mục gốc repo:  bash ocr_train/train.sh [EPOCHS] [OUT_DIR]
# Backend Keras = torch (TensorFlow trên máy lỗi). Máy 16 GB RAM -> 2 worker dạng thread.
set -euo pipefail
EPOCHS=${1:-60}
OUT=${2:-runs/ocr_vn}
export KERAS_BACKEND=torch PYTHONIOENCODING=utf-8
fast-plate-ocr train \
  --model-config-file ocr_train/base/cct_s_v2_global_model_config.yaml \
  --plate-config-file ocr_train/base/cct_s_v2_global_plate_config.yaml \
  --weights-path ocr_train/base/cct_s_v2_global.keras \
  --annotations datasets/ocr_vn/train.csv \
  --val-annotations datasets/ocr_vn/val.csv \
  --augmentation-path configs/augmentation.yaml \
  --lr 3e-4 --warmup-fraction 0.03 \
  --batch-size 64 --workers 2 --no-use-multiprocessing \
  --epochs "$EPOCHS" --early-stopping-patience 12 --early-stopping-metric val_plate_acc \
  --output-dir "$OUT" --seed 0
