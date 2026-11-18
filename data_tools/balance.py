"""Cân bằng lớp cho tập train detector bằng repeat-factor sampling (Gupta et al., LVIS 2019).

Mỗi lớp c có f_c = tỷ lệ ảnh train chứa c; r_c = max(1, sqrt(t / f_c)).
Ảnh lặp r_i = max r_c trên các lớp có trong ảnh (phần lẻ làm tròn ngẫu nhiên, seed cố định),
nên ảnh có bus/truck xuất hiện nhiều lần hơn mỗi epoch; val/test giữ nguyên.

Chạy: python -m data.balance            (ghi datasets/vehicle_det/train_rfs.txt và data_rfs.yaml)
"""
import argparse
import math
import random
from collections import Counter
from pathlib import Path

import yaml

from data_tools.prepare_det_data import COARSE_CLASSES, IMG_EXTS

ROOT = Path(__file__).resolve().parent.parent
DET = ROOT / "datasets" / "vehicle_det"


def image_classes(label_path):
    return {int(line.split()[0]) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def repeat_factors(classes_per_image, t):
    """-> (r_c theo lớp, r_i theo ảnh)."""
    n = len(classes_per_image)
    freq = Counter(c for cls in classes_per_image for c in cls)
    r_class = {c: max(1.0, math.sqrt(t / (k / n))) for c, k in freq.items()}
    r_image = [max((r_class[c] for c in cls), default=1.0) for cls in classes_per_image]
    return r_class, r_image


def repeat_counts(r_image, seed=0):
    """Số lần lặp nguyên: phần nguyên + 1 với xác suất bằng phần lẻ."""
    rng = random.Random(seed)
    return [int(r) + (rng.random() < r - int(r)) for r in r_image]


def main():
    parser = argparse.ArgumentParser()
    # 0.5: bus (có ở 22.8% ảnh) lặp x1.48, truck (42.3%) x1.09; ảnh nào cũng có xe máy nên không lặp mạnh hơn được
    parser.add_argument("--t", type=float, default=0.5, help="ngưỡng tần suất: lớp có f_c < t được lặp")
    args = parser.parse_args()

    images = sorted(p for p in (DET / "images" / "train").iterdir() if p.suffix.lower() in IMG_EXTS)
    classes = [image_classes(DET / "labels" / "train" / f"{p.stem}.txt") for p in images]
    r_class, r_image = repeat_factors(classes, args.t)
    counts = repeat_counts(r_image)

    lines = [str(p) for p, k in zip(images, counts) for _ in range(k)]
    (DET / "train_rfs.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = {"path": str(DET), "train": "train_rfs.txt", "val": "images/val", "test": "images/test",
            "names": dict(enumerate(COARSE_CLASSES))}
    (DET / "data_rfs.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    print("r_c:", {COARSE_CLASSES[c]: round(r, 2) for c, r in sorted(r_class.items())})
    print(f"Ảnh train: {len(images)} -> {len(lines)} mẫu/epoch (x{len(lines) / len(images):.2f})")
    for name, before, after in box_shares(classes, counts, DET / "labels" / "train", images):
        print(f"  {name}: {before:.1%} -> {after:.1%} số box")


def box_shares(classes, counts, label_dir, images):
    before, after = Counter(), Counter()
    for p, k in zip(images, counts):
        for line in (label_dir / f"{p.stem}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = int(line.split()[0])
                before[c] += 1
                after[c] += k
    tb, ta = sum(before.values()), sum(after.values())
    return [(COARSE_CLASSES[c], before[c] / tb, after[c] / ta) for c in sorted(before)]


if __name__ == "__main__":
    main()
