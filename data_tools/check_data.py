"""Kiểm tra dataset: python -m data.check_data"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data.prepare_det_data import COARSE_CLASSES, IMG_EXTS, compute_phash, phash_groups

ROOT = Path(__file__).resolve().parent.parent
DET = ROOT / "datasets" / "vehicle_det"


def check_det():
    errors, paths, split_of = 0, [], []
    for split in ("train", "val", "test"):
        counts = Counter()
        for label in sorted((DET / "labels" / split).glob("*.txt")):
            for line in label.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                vals = line.split()
                ok = len(vals) == 5 and vals[0].isdigit() and int(vals[0]) < len(COARSE_CLASSES) \
                    and all(0.0 <= float(v) <= 1.0 for v in vals[1:])
                if not ok:
                    errors += 1
                    print("Label lỗi:", label, line)
                    continue
                counts[COARSE_CLASSES[int(vals[0])]] += 1
        imgs = [p for p in (DET / "images" / split).glob("*") if p.suffix.lower() in IMG_EXTS]
        paths += imgs
        split_of += [split] * len(imgs)
        print(f"[det] {split}: {len(imgs)} ảnh, boxes {dict(counts)}")
    groups = phash_groups([compute_phash(p) for p in paths])
    splits_by_group = defaultdict(set)
    for s, g in zip(split_of, groups):
        splits_by_group[g].add(s)
    leaks = sum(1 for v in splits_by_group.values() if len(v) > 1)
    print(f"[det] nhóm near-duplicate nằm ở >1 split: {leaks}")
    return errors + leaks


if __name__ == "__main__":
    problems = check_det()
    print("OK" if problems == 0 else f"{problems} vấn đề")
    sys.exit(0 if problems == 0 else 1)
