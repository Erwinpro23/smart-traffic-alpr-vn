"""Chấm OCR (đọc đúng cả biển, đúng từng ký tự) trên các tập test của datasets/ocr_vn, so mô hình global và VN.

python -m ocr_train.evaluate  ->  evaluation/results/ocr_vn_test.json
"""
import csv
import json
from pathlib import Path

import cv2

from alpr_common import ocr_model_kwargs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "ocr_vn"
TESTS = ("test_phm", "test_topkek")


def char_acc(pred, gold):
    return sum(a == b for a, b in zip(pred, gold)) / max(len(pred), len(gold), 1)


def evaluate(recognizer, rows):
    imgs = [cv2.cvtColor(cv2.imread(str(DATA / r["image_path"])), cv2.COLOR_BGR2RGB) for r in rows]
    preds = []
    for i in range(0, len(imgs), 64):
        preds += [getattr(p, "plate", p).replace("_", "") for p in recognizer.run(imgs[i:i + 64])]
    gold = [r["plate_text"] for r in rows]
    errors = [(g, p) for g, p in zip(gold, preds) if g != p]
    return {"n": len(rows), "plate_acc": round(1 - len(errors) / len(rows), 4),
            "char_acc": round(sum(char_acc(p, g) for g, p in zip(gold, preds)) / len(rows), 4),
            "errors_sample": errors[:15]}


def main():
    from fast_plate_ocr import LicensePlateRecognizer

    out = {}
    for model in ("global", "vn"):
        kwargs = ocr_model_kwargs(model)
        if model == "vn" and "hub_ocr_model" in kwargs:
            print("Chưa có weights/ocr_vn.onnx, bỏ qua mô hình VN")
            continue
        rec = LicensePlateRecognizer(device="cpu", **kwargs)
        for t in TESTS:
            rows = list(csv.DictReader(open(DATA / f"{t}.csv", encoding="utf-8")))
            out[f"{model}/{t}"] = evaluate(rec, rows)
            r = out[f"{model}/{t}"]
            print(f"{model:6s} {t:12s} n={r['n']} plate_acc={r['plate_acc']:.3f} char_acc={r['char_acc']:.3f}")
    res = ROOT / "evaluation" / "results" / "ocr_vn_test.json"
    res.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
