import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Baseline: off-the-shelf COCO YOLO11 (no Vietnamese training) on the vehicle test split.
Labels are remapped to COCO ids (motorcycle 3, car 2, bus 5, truck 7) and validation is restricted to those classes."""
import json
import os
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"D:\test_biensoxe")
SRC = ROOT / "datasets" / "vehicle_det"
DST = ROOT / "datasets" / "vehicle_det_coco"
OURS_TO_COCO = {0: 3, 1: 2, 2: 5, 3: 7}


def build():
    (DST / "images" / "test").mkdir(parents=True, exist_ok=True)
    (DST / "labels" / "test").mkdir(parents=True, exist_ok=True)
    for img in (SRC / "images" / "test").iterdir():
        dst = DST / "images" / "test" / img.name
        if not dst.exists():
            os.link(img, dst)
        lab = SRC / "labels" / "test" / (img.stem + ".txt")
        lines = []
        if lab.exists():
            for ln in lab.read_text().splitlines():
                p = ln.split()
                if p:
                    lines.append(" ".join([str(OURS_TO_COCO[int(p[0])])] + p[1:]))
        (DST / "labels" / "test" / (img.stem + ".txt")).write_text("\n".join(lines))


def main():
    build()
    out = {}
    for w in sys.argv[1:]:
        det = YOLO(str(ROOT / w))
        yaml = DST / "data.yaml"
        yaml.write_text(f"path: {DST}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n" +
                        "".join(f"  {i}: {n}\n" for i, n in det.names.items()))
        m = det.val(data=str(yaml), split="test", imgsz=640, device=0, verbose=False, classes=[2, 3, 5, 7],
                    project=str(ROOT / "runs"), name=f"test_coco_{Path(w).stem}", exist_ok=True)
        per_class = {}
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per_class[det.names[int(c)]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap}
        out[w] = {"mAP50": m.box.map50, "mAP50_95": m.box.map, "precision": m.box.mp, "recall": m.box.mr,
                  "speed_ms": m.speed, "per_class": per_class}
        print(w, json.dumps(out[w], indent=1))
    (ROOT / "reports" / "vehicle_eval" / "metrics_coco_baseline.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
