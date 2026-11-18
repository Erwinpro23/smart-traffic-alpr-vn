import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Robustness of a YOLO detector to image degradations (night, blur, noise, JPEG, low resolution, rain, fog).

Builds degraded copies of the test split (labels copied unchanged), runs model.val on each and writes
a CSV, a line plot (mAP@0.5 vs severity) and an example grid for the report.

Usage (from the project root):
  python robustness.py --weights model/Wiliv5_best.pt --data v58_merged_dataset_real/v58_merged_dataset_real/data.yaml --out outputs/reports/robust_plate --max 1000
"""
import argparse, csv, os, shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import img2label_paths

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def night(img, s, rng):  # gamma darkening + blue cast + sensor noise
    g = (1.6, 2.3, 3.0)[s - 1]
    out = 255.0 * (img / 255.0) ** g
    out[..., 0] *= 1.10; out[..., 2] *= 0.85            # BGR: a bit more blue, less red
    out += rng.normal(0, 3 + 3 * s, img.shape)
    return out

def motion_blur(img, s, rng):
    k = (5, 9, 15)[s - 1]
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, ker)

def gaussian_noise(img, s, rng):
    return img + rng.normal(0, (10, 20, 35)[s - 1], img.shape)

def jpeg(img, s, rng):
    q = (30, 15, 7)[s - 1]
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def low_res(img, s, rng):
    f = (0.5, 0.33, 0.25)[s - 1]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def rain(img, s, rng):  # random slanted streaks, slightly blurred, plus a darker sky
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.uint8)
    for _ in range(int((300, 800, 1500)[s - 1] * h * w / (1280 * 720))):
        x, y, L = rng.integers(0, w), rng.integers(0, h), rng.integers(10, 25)
        cv2.line(layer, (int(x), int(y)), (int(x + L * 0.25), int(y + L)), 200, 1)
    layer = cv2.blur(layer, (3, 3)).astype(np.float32)[..., None]
    return img * (1 - 0.08 * s) + layer * 0.6

def fog(img, s, rng):
    a = (0.3, 0.5, 0.7)[s - 1]
    return img * (1 - a) + 200 * a

CORRUPTIONS = dict(night=night, motion_blur=motion_blur, gaussian_noise=gaussian_noise,
                   jpeg=jpeg, low_res=low_res, rain=rain, fog=fog)


def test_images(data_yaml, split):
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    root = Path(d.get("path") or Path(data_yaml).parent)
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    entries = d[split] if isinstance(d[split], list) else [d[split]]
    files = []
    for e in entries:
        p = Path(e) if Path(e).is_absolute() else root / e
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
        else:  # txt list of image paths
            files += [Path(l.strip()) if Path(l.strip()).is_absolute() else root / l.strip()
                      for l in open(p, encoding="utf-8") if l.strip()]
    return d["names"], files


def build_split(files, out_dir, fn, s, seed):
    rng = np.random.default_rng(seed)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    for i, (f, lab) in enumerate(zip(files, img2label_paths([str(f) for f in files]))):
        img = cv2.imread(str(f))
        if img is None:
            continue
        if fn is not None:
            img = np.clip(fn(img.astype(np.float32), s, rng), 0, 255).astype(np.uint8)
        stem = f"{i:06d}_{f.stem}"  # index prefix: test images from different folders may share a name
        # PNG keeps the degradation exact (a JPEG re-save would add its own artefacts to every corruption)
        cv2.imwrite(str(out_dir / "images" / (stem + ".png")), img)
        if Path(lab).exists():
            shutil.copy(lab, out_dir / "labels" / (stem + ".txt"))
        else:
            open(out_dir / "labels" / (stem + ".txt"), "w").close()  # image without objects


def example_grid(img_path, out_png):
    img = cv2.imread(str(img_path)).astype(np.float32)
    rng = np.random.default_rng(0)
    rows = []
    for name, fn in CORRUPTIONS.items():
        tiles = [img] + [fn(img.copy(), s, rng) for s in (1, 2, 3)]
        tiles = [cv2.resize(np.clip(t, 0, 255).astype(np.uint8), (320, int(320 * img.shape[0] / img.shape[1])))
                 for t in tiles]
        row = np.hstack(tiles)
        cv2.putText(row, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        rows.append(row)
    cv2.imwrite(str(out_png), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=find_default_weights(), help="Path to YOLO detector weights")
    ap.add_argument("--data", default=find_default_data(), help="Path to dataset data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/reports/robust_plate", help="Output directory for reports")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max", type=int, default=0, help="use only the first N test images (0 = all)")
    ap.add_argument("--only", nargs="*", default=list(CORRUPTIONS), help="subset of corruptions")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names, files = test_images(a.data, a.split)
    if a.max:
        files = files[: a.max]
    print(f"{len(files)} test images")
    example_grid(files[0], out / "examples.jpg")

    model = YOLO(a.weights)
    rows = []
    jobs = [("clean", None, 0)] + [(c, CORRUPTIONS[c], s) for c in a.only for s in (1, 2, 3)]
    for name, fn, s in jobs:
        d = out / "data" / f"{name}_{s}"
        if not (d / "images").exists():
            build_split(files, d, fn, s, seed=s)
        y = out / "data" / f"{name}_{s}.yaml"
        yaml.safe_dump({"path": str(d.resolve()), "train": "images", "val": "images", "names": names}, open(y, "w"))
        m = model.val(data=str(y), split="val", imgsz=a.imgsz, batch=16, plots=False, verbose=False,
                      project=str(out / "val"), name=f"{name}_{s}", exist_ok=True)
        row = dict(corruption=name, severity=s, P=m.box.mp, R=m.box.mr, mAP50=m.box.map50, mAP50_95=m.box.map)
        for ci, ap50 in zip(m.box.ap_class_index, m.box.ap50):
            row[f"AP50_{names[int(ci)]}"] = ap50
        rows.append(row)
        print({k: (round(v * 100, 2) if isinstance(v, float) else v) for k, v in row.items()})

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out / "robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    clean = rows[0]["mAP50"] * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in a.only:
        ys = [clean] + [r["mAP50"] * 100 for r in rows if r["corruption"] == c]
        ax.plot([0, 1, 2, 3], ys, marker="o", label=c)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xlabel("Severity (0 = clean)"); ax.set_ylabel("mAP@0.5 (%)")
    ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(out / "robustness.pdf"); fig.savefig(out / "robustness.png", dpi=200)
    print("wrote", out / "robustness.csv")


if __name__ == "__main__":
    main()
