"""Tải và gộp dữ liệu detector xe (4 lớp thô) -> datasets/vehicle_det.

Chạy: python -m data.prepare_det_data
"""
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
COARSE_CLASSES = ["motorcycle", "car", "bus", "truck"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAW_DIR = ROOT / "datasets" / "raw" / "det"
OUT_DIR = ROOT / "datasets" / "vehicle_det"


def load_env():
    """Nạp các dòng KEY=VALUE trong .env vào os.environ (không ghi đè biến đã có)."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def get_roboflow_key():
    load_env()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Thiếu ROBOFLOW_API_KEY (biến môi trường hoặc file .env)")
    return key


def download_source(src, raw_dir=RAW_DIR):
    dest = raw_dir / src["name"]
    if dest.exists() and any(dest.rglob("*.txt")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if src["type"] == "roboflow":
        from roboflow import Roboflow
        rf = Roboflow(api_key=get_roboflow_key())
        rf.workspace(src["workspace"]).project(src["project"]).version(src["version"]).download(
            "yolov11", location=str(dest), overwrite=True)
    elif src["type"] == "kaggle":
        load_env()  # KAGGLE_API_TOKEN phải có trước khi import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(src["slug"], path=str(dest), unzip=True)
    else:
        raise ValueError(f"Loại nguồn không hỗ trợ: {src['type']}")
    return dest


def read_names(root):
    for yml in sorted(Path(root).rglob("*.yaml")):
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return names
    raise FileNotFoundError(f"Không tìm thấy data.yaml có 'names' trong {root}")


def find_yolo_pairs(root):
    """Cặp (ảnh, nhãn): nhãn ở thư mục labels/ song song images/, hoặc .txt cùng thư mục với ảnh."""
    pairs = []
    for img in sorted(Path(root).rglob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        if "images" in img.parts:
            parts = list(img.parts)
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            label = Path(*parts).with_suffix(".txt")
        else:
            label = img.with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))
    return pairs


def remap_label_lines(lines, names, class_map):
    out = []
    for line in lines:
        vals = line.split()
        if len(vals) < 5 or len(vals) % 2 == 0:
            continue
        try:
            cls = int(float(vals[0]))
            coords = [float(v) for v in vals[1:]]
        except ValueError:
            continue
        if cls >= len(names) or names[cls] not in class_map:
            continue
        target = COARSE_CLASSES.index(class_map[names[cls]])
        if len(coords) == 4:
            cx, cy, w, h = coords
        else:  # polygon x1 y1 x2 y2 ... -> bbox
            xs, ys = coords[0::2], coords[1::2]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w <= 0 or h <= 0:
            continue
        out.append(f"{target} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def compute_phash(path):
    import imagehash
    from PIL import Image
    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("RGB"))), 16)


def phash_groups(hashes, max_dist=5):
    """Union-find: ảnh có Hamming <= max_dist cùng nhóm (bắc cầu)."""
    arr = np.array(hashes, dtype=np.uint64)
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(arr)):
        dists = np.bitwise_count(arr[i + 1:] ^ arr[i])
        for j in np.nonzero(dists <= max_dist)[0]:
            ri, rj = find(i), find(i + 1 + int(j))
            if ri != rj:
                parent[rj] = ri
    return [find(i) for i in range(len(hashes))]


def group_split(groups, ratios=(0.8, 0.1, 0.1), seed=0):
    """Chia theo nhóm nhưng cân theo số ẢNH: nhóm lớn xếp trước, vào split đang thiếu nhiều nhất."""
    sizes = Counter(groups)
    unique = sorted(sizes)
    random.Random(seed).shuffle(unique)
    unique.sort(key=lambda g: -sizes[g])  # sort ổn định: cùng kích thước giữ thứ tự ngẫu nhiên
    names = ("train", "val", "test")
    total = len(groups)
    filled = dict.fromkeys(names, 0)
    assign = {}
    for g in unique:
        split = max(names, key=lambda s: ratios[names.index(s)] * total - filled[s])
        assign[g] = split
        filled[split] += sizes[g]
    return [assign[g] for g in groups]


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "sources.yaml").read_text(encoding="utf-8"))
    samples = []  # (img_path, remapped_lines, source_name)
    for src in cfg["det_sources"]:
        root = download_source(src) / src.get("subdir", "")
        names = [str(n) for n in src["names"]] if "names" in src else read_names(root)
        pairs = find_yolo_pairs(root)
        kept = 0
        for img, label in pairs:
            lines = remap_label_lines(label.read_text(encoding="utf-8").splitlines(), names, src["class_map"])
            if lines:
                samples.append((img, lines, src["name"]))
                kept += 1
        print(f"[{src['name']}] names={names} pairs={len(pairs)} kept={kept}")

    print(f"Tính pHash cho {len(samples)} ảnh...")
    groups = phash_groups([compute_phash(s[0]) for s in samples])
    splits = group_split(groups)
    print(f"Nhóm near-duplicate: {len(set(groups))} nhóm / {len(samples)} ảnh; "
          f"lớn nhất: {[n for _, n in Counter(groups).most_common(5)]}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for (img, lines, src_name), split in zip(samples, splits):
        stem = f"{src_name}__{img.stem}"
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, OUT_DIR / "images" / split / f"{stem}{img.suffix.lower()}")
        (OUT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = {"path": str(OUT_DIR), "train": "images/train", "val": "images/val", "test": "images/test",
                 "names": dict(enumerate(COARSE_CLASSES))}
    (OUT_DIR / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    for s in ("train", "val", "test"):
        print(f"{s}: {splits.count(s)} ảnh")


if __name__ == "__main__":
    main()
