"""Chuẩn bị dữ liệu fine-tune OCR biển VN (docs/superpowers/specs/2026-11-14-ocr-finetune-design.md).

Tải 2 bộ Kaggle vào datasets/ocr_vn/raw (nếu chưa có), chuẩn hoá nhãn, loại ảnh trùng, chia tập theo chuỗi biển,
ghi datasets/ocr_vn/{train,val,test_topkek,test_phm}.csv theo định dạng của fast-plate-ocr
(image_path tương đối so với file CSV, plate_text, plate_region).
"""
import csv
import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "ocr_vn"
RAW = OUT / "raw"
SOURCES = {"topkek": "topkek69/vietnamese-license-plate-ocr", "phm": "phmthanhlm/license-plates-vietnamese"}
ALPHABET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
N_SYNTH = 2500  # số ảnh tổng hợp đưa vào train (ảnh thật là chính)


def normalize_label(text):
    """Nhãn -> chuỗi 0-9A-Z (Đ -> D), None nếu không dùng được."""
    t = str(text or "").upper().replace("Đ", "D")
    t = "".join(c for c in t if c in ALPHABET)
    return t if 4 <= len(t) <= 10 else None


def split_by_plate(rows, fracs, seed=0):
    """Chia rows (dict có 'plate_text') thành len(fracs) phần theo chuỗi biển: một biển chỉ nằm ở một phần."""
    plates = sorted({r["plate_text"] for r in rows})
    random.Random(seed).shuffle(plates)
    bounds, acc = [], 0.0
    for f in fracs[:-1]:
        acc += f
        bounds.append(round(acc * len(plates)))
    part_of = {}
    for i, p in enumerate(plates):
        part_of[p] = sum(i >= b for b in bounds)
    parts = [[] for _ in fracs]
    for r in rows:
        parts[part_of[r["plate_text"]]].append(r)
    return parts


def download():
    sys.path.insert(0, str(ROOT))
    from data.prepare_det_data import load_env

    load_env()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    for ref in SOURCES.values():
        dst = RAW / ref.replace("/", "__")
        if not dst.exists():
            print("Tải", ref)
            api.dataset_download_files(ref, path=str(dst), unzip=True, quiet=True)


def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_sources():
    """-> dict nguồn -> list (đường dẫn ảnh, nhãn gốc)."""
    t = RAW / SOURCES["topkek"].replace("/", "__")
    p = RAW / SOURCES["phm"].replace("/", "__") / "dataset"
    read = lambda f: list(csv.DictReader(open(f, encoding="utf-8")))  # noqa: E731
    phm = lambda f: [(p / a, b) for a, b in (ln.rstrip("\n").split("\t") for ln in open(f, encoding="utf-8") if "\t" in ln)]  # noqa: E731
    return {
        "topkek_real": [(t / "cropped" / r["Name"], r["Label"]) for r in read(t / "labels" / "crop_labels.csv")],
        "topkek_synth": [(t / "generated" / r["Name"], r["Label"]) for r in read(t / "labels" / "gen_labels.csv")],
        "phm_train": phm(p / "rec_gt_train.txt"),
        "phm_test": phm(p / "rec_gt_test.txt"),
    }


def clean(items, seen):
    """Chuẩn hoá nhãn, bỏ ảnh thiếu hoặc trùng nội dung (seen: hash đã gặp, dùng chung giữa các nguồn)."""
    rows, dropped = [], Counter()
    for path, label in items:
        text = normalize_label(label)
        if text is None:
            dropped["label"] += 1
            continue
        if not path.exists():
            dropped["missing"] += 1
            continue
        h = file_hash(path)
        if h in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(h)
        rows.append({"path": path, "plate_text": text})
    return rows, dropped


def write_csv(name, rows):
    f = OUT / f"{name}.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path", "plate_text", "plate_region"])
        for r in rows:
            w.writerow([Path(r["path"]).relative_to(OUT).as_posix(), r["plate_text"], "Vietnam"])
    print(f"{name}: {len(rows)} ảnh")


def main():
    download()
    src = load_sources()
    seen = set()
    # Tập test trước để ảnh trùng được giữ ở test và bỏ khỏi train
    phm_test, d1 = clean(src["phm_test"], seen)
    real, d2 = clean(src["topkek_real"], seen)
    phm_train, d3 = clean(src["phm_train"], seen)
    synth, d4 = clean(src["topkek_synth"], seen)
    print("bỏ:", {"phm_test": dict(d1), "topkek_real": dict(d2), "phm_train": dict(d3), "topkek_synth": dict(d4)})

    real_tr, real_va, real_te = split_by_plate(real, [0.8, 0.1, 0.1])
    test_plates = {r["plate_text"] for r in real_te + phm_test}
    phm_tr, phm_va = split_by_plate([r for r in phm_train if r["plate_text"] not in test_plates], [0.9, 0.1], seed=1)
    synth = [r for r in synth if r["plate_text"] not in test_plates]
    random.Random(2).shuffle(synth)

    train = real_tr + phm_tr + synth[:N_SYNTH]
    random.Random(3).shuffle(train)
    write_csv("train", train)
    write_csv("val", real_va + phm_va)
    write_csv("test_topkek", real_te)
    write_csv("test_phm", phm_test)


if __name__ == "__main__":
    main()
