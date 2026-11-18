import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Accuracy vs. speed trade-off: input size (and optionally TensorRT FP16) of a YOLO detector.

Usage:
  python imgsz_speed.py --weights model/Wiliv5_best.pt --data v58_merged_dataset_real/v58_merged_dataset_real/data.yaml --out outputs/reports/imgsz_plate
  python imgsz_speed.py --tensorrt
"""
import argparse, csv, os
from pathlib import Path
from ultralytics import YOLO


def find_default_weights():
    cands = [
        "model/Wiliv5_best.pt",
        "model/v62.pt",
        "model/yolo11l.pt",
        "Wiliv5_best.pt"
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def find_default_data():
    cands = [
        "v58_merged_dataset_real/v58_merged_dataset_real/data.yaml",
        "v58_merged_dataset_real/data.yaml",
        "dataset/data.yaml"
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def run(model, data, split, imgsz, half, out, tag):
    m = model.val(data=data, split=split, imgsz=imgsz, batch=1, half=half, plots=False, verbose=False,
                  project=str(out / "val"), name=f"{tag}_{imgsz}", exist_ok=True)
    return dict(backend=tag, imgsz=imgsz, P=m.box.mp, R=m.box.mr, mAP50=m.box.map50, mAP50_95=m.box.map,
                pre_ms=m.speed["preprocess"], inf_ms=m.speed["inference"], post_ms=m.speed["postprocess"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=find_default_weights(), help="Path to YOLO weights (.pt)")
    ap.add_argument("--data", default=find_default_data(), help="Path to data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/reports/imgsz_speed", help="Output directory for reports")
    ap.add_argument("--sizes", type=int, nargs="+", default=[416, 512, 640, 768, 960])
    ap.add_argument("--tensorrt", action="store_true", help="also export a TensorRT FP16 engine per size")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in a.sizes:
        rows.append(run(YOLO(a.weights), a.data, a.split, s, True, out, "pytorch_fp16"))
        if a.tensorrt:  # needs `pip install tensorrt`; an engine is fixed to one input size
            eng = YOLO(a.weights).export(format="engine", imgsz=s, half=True, batch=1)
            rows.append(run(YOLO(eng), a.data, a.split, s, True, out, "tensorrt_fp16"))
        print(rows[-1])

    with open(out / "imgsz_speed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag in dict.fromkeys(r["backend"] for r in rows):
        rr = [r for r in rows if r["backend"] == tag]
        ax.plot([r["inf_ms"] for r in rr], [r["mAP50_95"] * 100 for r in rr], marker="o", label=tag)
        for r in rr:
            ax.annotate(str(r["imgsz"]), (r["inf_ms"], r["mAP50_95"] * 100), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Inference time per image, batch 1 (ms)"); ax.set_ylabel("mAP@0.5:0.95 (%)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "imgsz_speed.pdf"); fig.savefig(out / "imgsz_speed.png", dpi=200)


if __name__ == "__main__":
    main()
