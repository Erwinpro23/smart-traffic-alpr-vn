"""Gộp nhãn người duyệt (evaluation/review/gt_review_<video>.csv) vào GT -> <gt>_reviewed.csv.
Ô human_* để trống = giữ nhãn AI; human_plate = 'none' = xe không có biển đọc được; human_type = 'not_vehicle' = bỏ.
Dùng: python -m evaluation.merge_review demo"""
import csv
import sys

from evaluation.scoring import load_gt
from evaluation.videos import EVAL_DIR, gt_path

TYPES = {"motorcycle", "car", "bus", "truck", "not_vehicle"}


def merge(gt_rows, review_rows):
    fix = {r["gt_id"]: r for r in review_rows}
    out, t_changed, p_changed = [], set(), set()
    for r in gt_rows:
        r = dict(r)
        h = fix.get(r["gt_id"], {})
        ht, hp = (h.get("human_type") or "").strip(), (h.get("human_plate") or "").strip().upper()
        if ht:
            if ht not in TYPES:
                raise ValueError(f"gt_id {r['gt_id']}: loại '{ht}' không hợp lệ")
            if ht != r["type"]:
                t_changed.add(r["gt_id"])
            r["type"] = ht
        if hp:
            new = "" if hp == "NONE" else hp
            if new != r["plate"]:
                p_changed.add(r["gt_id"])
            r["plate"] = new
        out.append(r)
    return out, len(t_changed), len(p_changed)


def main(video):
    with open(EVAL_DIR / "review" / f"gt_review_{video}.csv", encoding="utf-8-sig") as f:
        review = list(csv.DictReader(f))
    rows, nt, np_ = merge(load_gt(gt_path(video)), review)
    with open(gt_path(video, "reviewed"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_id", "frame", "x1", "y1", "x2", "y2", "type", "plate"])
        for r in rows:
            w.writerow([r["gt_id"], r["frame"], *r["box"], r["type"], r["plate"]])
    print(f"{video}: {nt} xe đổi loại, {np_} xe đổi biển, {len(review)} xe đã duyệt")


if __name__ == "__main__":
    main(sys.argv[1])
