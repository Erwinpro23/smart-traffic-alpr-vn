"""Bước 3: chấm detector trên frame held-out. Nhãn: evaluation/heldout/<video>/gt.json (labels.py apply).
Chỉ tính box cao >= MIN_H và tâm ngoài vùng IGNORE, áp dụng như nhau cho nhãn và dự đoán.
AP@0.5 kiểu COCO (nội suy 101 điểm) theo từng lớp; P/R tại conf 0.25. Ghi evaluation/results/heldout_<video>.json
Dùng: python -m evaluation.heldout.score hem junction"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from evaluation.heldout.labels import CLASSES, HERE, counted, in_ignore, iou

ROOT = HERE.parents[1]
DETECTORS = {"plain": "weights/vehicle_det.pt", "balanced": "weights/vehicle_det_bal.pt", "coco": "yolo11s.pt"}
CONF_PR = 0.25


def match(gt, preds):
    """preds: [(frame, score, box)] -> danh sách (score, is_tp) và số nhãn. Mỗi nhãn được khớp tối đa một lần."""
    used = {f: [False] * len(b) for f, b in gt.items()}
    out = []
    for f, s, box in sorted(preds, key=lambda p: -p[1]):
        best, bi = 0.5, -1
        for i, g in enumerate(gt.get(f, [])):
            v = iou(box, g)
            if v >= best and not used[f][i]:
                best, bi = v, i
        if bi >= 0:
            used[f][bi] = True
        out.append((s, bi >= 0))
    return out, sum(len(b) for b in gt.values())


def ap50(gt, preds):
    m, n = match(gt, preds)
    if n == 0 or not m:
        return 0.0
    tp = np.cumsum([t for _, t in m])
    fp = np.cumsum([not t for _, t in m])
    rec, prec = tp / n, tp / np.maximum(tp + fp, 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]))


def match_counts(gt, preds):
    m, n = match(gt, preds)
    tp = sum(t for _, t in m)
    return tp, len(m) - tp, n - tp


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 4), round(min(1, c + h), 4))


def load_gt(video):
    g = json.loads((HERE / video / "gt.json").read_text(encoding="utf-8"))["frames"]
    return {c: {f: [b["box"] for b in bs if b["cls"] == c and counted(b["box"]) and not in_ignore(video, b["box"])]
                for f, bs in g.items()} for c in CLASSES}


def predict(video, weights):
    from ultralytics import YOLO
    det = YOLO(str(ROOT / weights))
    name_to_cls = {i: n for i, n in det.names.items() if n in CLASSES}
    preds = {c: [] for c in CLASSES}
    for fp in sorted((HERE / video / "frames").glob("*.jpg")):
        r = det.predict(str(fp), imgsz=640, conf=0.001, iou=0.7, device=0, verbose=False,
                        classes=list(name_to_cls))[0]
        for b, s, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            box = [round(v) for v in b]
            if counted(box) and not in_ignore(video, box):
                preds[name_to_cls[int(k)]].append((fp.name, s, box))
    return preds


def main(videos):
    for video in videos:
        gt = load_gt(video)
        res = {"video": video, "n_frames": len(next(iter(gt.values()))),
               "n_gt": {c: sum(len(b) for b in gt[c].values()) for c in CLASSES}, "detectors": {}}
        for name, w in DETECTORS.items():
            preds = predict(video, w)
            per = {}
            for c in CLASSES:
                n = res["n_gt"][c]
                hi = [p for p in preds[c] if p[1] >= CONF_PR]
                tp, fp, fn = match_counts(gt[c], hi)
                per[c] = {"n_gt": n, "ap50": round(ap50(gt[c], preds[c]), 4) if n else None,
                          "tp": tp, "fp": fp, "fn": fn,
                          "recall": round(tp / n, 4) if n else None, "recall_ci": wilson(tp, n),
                          "precision": round(tp / (tp + fp), 4) if tp + fp else None}
            present = [c for c in CLASSES if res["n_gt"][c]]
            tp = sum(per[c]["tp"] for c in CLASSES)
            fp = sum(per[c]["fp"] for c in CLASSES)
            n = sum(res["n_gt"].values())
            res["detectors"][name] = {
                "weights": w, "map50": round(float(np.mean([per[c]["ap50"] for c in present])), 4),
                "recall": round(tp / n, 4), "recall_ci": wilson(tp, n),
                "precision": round(tp / (tp + fp), 4) if tp + fp else None, "per_class": per}
            print(video, name, res["detectors"][name]["map50"], res["detectors"][name]["recall"],
                  res["detectors"][name]["precision"], flush=True)
        (ROOT / "evaluation" / "results" / f"heldout_{video}.json").write_text(json.dumps(res, indent=1),
                                                                               encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
