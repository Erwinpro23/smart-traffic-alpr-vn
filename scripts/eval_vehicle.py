import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Đánh giá detector xe (4 lớp) trên test split. Chạy: python eval_vehicle.py [--weights PATH]"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "vehicle_eval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "weights" / "vehicle_det.pt"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    det = YOLO(args.weights)
    m = det.val(data=str(ROOT / "datasets" / "vehicle_det" / "data.yaml"), split="test", imgsz=640,
                device=0, verbose=False, project=str(ROOT / "runs"), name=f"test_{Path(args.weights).stem}", exist_ok=True)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[det.names[int(c)]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap}
    metrics = {"weights": args.weights, "mAP50": m.box.map50, "mAP50_95": m.box.map,
               "precision": m.box.mp, "recall": m.box.mr, "speed_ms": m.speed, "per_class": per_class}

    (OUT / f"metrics_{Path(args.weights).stem}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Detector mAP50 >= 0.90:", "PASS" if metrics["mAP50"] >= 0.90 else "FAIL")


if __name__ == "__main__":
    main()
