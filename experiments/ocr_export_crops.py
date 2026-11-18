import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Step 1 of the OCR evaluation: cut plate crops from the plate test split and label their text by hand.

  python ocr_export_crops.py export --data v58_merged_dataset_real/v58_merged_dataset_real/data.yaml --out outputs/ocr_test --n 600
  python ocr_export_crops.py label --out outputs/ocr_test        # type each plate; Enter = keep, ? = illegible, q = stop

The crops use the plate box of the ground truth, so OCR is measured without detector errors.
Text is stored without separators, top line then bottom line, e.g. 59G307326 or 51A12345.
Label by hand from the image: do not pre-fill with an OCR guess, it biases the test towards that OCR.
"""
import argparse, csv, os, random, sys
from pathlib import Path

import cv2
from ultralytics.data.utils import img2label_paths

sys.path.insert(0, str(Path(__file__).parent))
from robustness import test_images  # same split reader

FIELDS = ["file", "src", "w", "h", "aspect", "line", "text"]


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


def export(a):
    _, files = test_images(a.data, a.split)
    out = Path(a.out); (out / "crops").mkdir(parents=True, exist_ok=True)
    cands = []
    for f, lab in zip(files, img2label_paths([str(f) for f in files])):
        if not Path(lab).exists():
            continue
        for line in open(lab):
            p = line.split()
            if len(p) >= 5:
                cands.append((f, *map(float, p[1:5])))
    random.Random(0).shuffle(cands)

    rows, per = [], {"one": 0, "two": 0}
    for f, cx, cy, bw, bh in cands:
        img = cv2.imread(str(f))
        if img is None:
            continue
        H, W = img.shape[:2]
        w, h = bw * W, bh * H
        if w < 35 or h < 15:          # same minimum size as the pipeline (Table 14.2)
            continue
        kind = "one" if w / h >= 2.5 else "two"   # same ratio as the single-line rule
        if per[kind] >= a.n // 2:     # half one-line, half two-line plates
            continue
        px, py = 0.08 * w, 0.08 * h   # pipeline crop padding 8%
        x1, y1 = int(max(0, cx * W - w / 2 - px)), int(max(0, cy * H - h / 2 - py))
        x2, y2 = int(min(W, cx * W + w / 2 + px)), int(min(H, cy * H + h / 2 + py))
        name = f"{len(rows):05d}.png"
        cv2.imwrite(str(out / "crops" / name), img[y1:y2, x1:x2])
        rows.append(dict(file=name, src=str(f), w=round(w), h=round(h), aspect=round(w / h, 2), line=kind, text=""))
        per[kind] += 1
        if sum(per.values()) >= a.n:
            break
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, FIELDS); wr.writeheader(); wr.writerows(rows)
    print(f"{len(rows)} crops ({per}) -> {out}")


def label(a):
    out = Path(a.out)
    rows = list(csv.DictReader(open(out / "labels.csv", encoding="utf-8")))
    todo = [r for r in rows if not r["text"]]
    print(f"{len(todo)} crops left. Type the plate (no dashes/dots), Enter = skip, ? = illegible, q = quit.")
    try:
        for r in todo:
            img = cv2.imread(str(out / "crops" / r["file"]))
            s = 300 / max(1, img.shape[1])
            cv2.imshow("plate", cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)); cv2.waitKey(1)
            t = input(f"{r['file']} [{r['line']}]: ").strip().upper()
            if t == "Q":
                break
            if t:
                r["text"] = "ILLEGIBLE" if t == "?" else "".join(ch for ch in t if ch.isalnum())
    finally:
        with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, FIELDS); wr.writeheader(); wr.writerows(rows)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "label"])
    ap.add_argument("--data", default=find_default_data(), help="Path to data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="outputs/ocr_test", help="Directory to save crops and labels.csv")
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else label(a)
