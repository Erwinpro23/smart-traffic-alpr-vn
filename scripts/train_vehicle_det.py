import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Train YOLO11s phát hiện xe 4 lớp.

Train từ đầu:  python train_vehicle_det.py
Fine-tune:     python train_vehicle_det.py --finetune
  (từ runs/vehicle_det/weights/best.pt, augmentation phóng to mạnh để mô phỏng xe sát camera,
   lưu weights/vehicle_det_ft.pt — không ghi đè model đang dùng)
Đặt tên run và file lưu: python train_vehicle_det.py --name vehicle_det_v2s --out weights/vehicle_det_v2s.pt
Cân bằng lớp:  python -m data.balance && python train_vehicle_det.py --balanced
  (repeat-factor sampling + class weight căn bậc hai nghịch đảo tần suất + mixup 0.1,
   lưu weights/vehicle_det_bal.pt)
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)  # máy 16 GB RAM: không chạy song song 2 lần train
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    data_yaml = ROOT / "datasets" / "vehicle_det" / "data.yaml"

    if args.finetune:
        model_path = args.model or str(ROOT / "runs" / "vehicle_det" / "weights" / "best.pt")
        name, out = "vehicle_det_ft", ROOT / "weights" / "vehicle_det_ft.pt"
        # scale 0.9: phóng/thu ±90% -> có mẫu xe máy chiếm gần cả khung như camera cổng
        extra = {"epochs": args.epochs or 50, "scale": 0.9, "optimizer": "SGD", "lr0": 0.002, "warmup_epochs": 1}
    elif args.balanced:
        model_path = args.model or "yolo11s.pt"
        name, out = "vehicle_det_bal", ROOT / "weights" / "vehicle_det_bal.pt"
        # cls_pw 0.5: trọng số lớp = (1/số box)^0.5, chuẩn hoá trung bình 1
        extra = {"epochs": args.epochs or 100, "cls_pw": 0.5, "mixup": 0.1}
        data_yaml = ROOT / "datasets" / "vehicle_det" / "data_rfs.yaml"
    else:
        model_path = args.model or "yolo11s.pt"
        name, out = "vehicle_det", ROOT / "weights" / "vehicle_det.pt"
        extra = {"epochs": args.epochs or 100}

    if args.name:
        name = args.name
    if args.out:
        out = ROOT / args.out

    model = YOLO(model_path)
    model.train(data=str(data_yaml), imgsz=640, patience=20,
                batch=args.batch, device=0, seed=0, deterministic=True, workers=args.workers,
                project=str(ROOT / "runs"), name=name, exist_ok=True, **extra)
    best = ROOT / "runs" / name / "weights" / "best.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(best, out)
    print(f"Đã lưu {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
