"""Chấm điểm pipeline trên video demo bằng ground truth evaluation/demo_gt.csv (chỉ để đánh giá)."""
import csv
from collections import defaultdict


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(path):
    with open(path, encoding="utf-8") as f:
        return [{"gt_id": r["gt_id"], "frame": int(r["frame"]),
                 "box": tuple(int(r[k]) for k in ("x1", "y1", "x2", "y2")),
                 "type": r["type"], "plate": r["plate"]} for r in csv.DictReader(f)]


def match_gt(gt_rows, track_log, thr=0.5):
    """Mỗi dòng GT -> track id có IoU lớn nhất (>= thr) với box GT tại frame của dòng đó; không có -> None."""
    by_frame = {e["frame"]: e["tracks"] for e in track_log}
    out = []
    for r in gt_rows:
        best, best_iou = None, thr
        for t in by_frame.get(r["frame"], []):
            if t[0] is None:
                continue
            v = iou(r["box"], t[1:5])
            if v >= best_iou:
                best, best_iou = t[0], v
        out.append(best)
    return out


def score(gt_rows, matches, records):
    rec_of = {vid: i for i, r in enumerate(records) for vid in r["vehicle_ids"]}
    gt_info, recs_of_gt, used = {}, defaultdict(list), set()
    for r, tid in zip(gt_rows, matches):
        gt_info.setdefault(r["gt_id"], r)
        if tid is not None and tid in rec_of:
            recs_of_gt[r["gt_id"]].append(rec_of[tid])
            used.add(rec_of[tid])
    gt_plates = {r["plate"] for r in gt_rows if r["plate"]}

    n_veh = n_type_ok = n_plate = n_plate_ok = dup = false = 0
    for gid, r in gt_info.items():
        recs = recs_of_gt.get(gid, [])
        # record chính: phủ nhiều dòng GT nhất, ưu tiên record có biển
        distinct = sorted(set(recs), key=lambda i: (-recs.count(i), records[i]["plate"] == "", i))
        if r["type"] == "not_vehicle":
            false += sum(1 for i in distinct if records[i]["plate"])
            continue
        dup += max(0, len(distinct) - 1)
        n_veh += 1
        main = records[distinct[0]] if distinct else None
        n_type_ok += bool(main and main["type"] == r["type"])
        if r["plate"]:
            n_plate += 1
            n_plate_ok += bool(main and main["plate"] == r["plate"])
    for i, rec in enumerate(records):
        if i in used or not rec["plate"]:
            continue
        if not rec["vehicle_ids"] and rec["plate"] in gt_plates:
            dup += 1
        else:
            false += 1
    return {"type_acc": round(n_type_ok / n_veh, 4) if n_veh else 0.0,
            "plate_acc": round(n_plate_ok / n_plate, 4) if n_plate else 0.0,
            "duplicates": dup, "false_records": false, "n_vehicles": n_veh, "n_plates": n_plate}
