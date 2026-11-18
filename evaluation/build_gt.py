"""Gộp nhãn của các labeller thành file GT của 1 video. Chạy: python -m evaluation.build_gt [--video street]
Track được 2 labeller gán khác nhau phải có dòng trong labels_resolved.csv, nếu không sẽ báo lỗi."""
import argparse
import csv
import sys

from alpr_common import normalize_plate
from evaluation.videos import VIDEOS, gt_path

TYPES = {"motorcycle", "car", "bus", "truck", "not_vehicle"}


def read(folder, name):
    path = folder / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(r["track_id"]): r for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="demo", choices=sorted(VIDEOS))
    video = parser.parse_args().video
    folder = VIDEOS[video]["dir"]
    draft = read(folder, "gt_draft.csv")
    parts = [read(folder, "labels_part1.csv"), read(folder, "labels_part2.csv")]
    resolved = read(folder, "labels_resolved.csv")
    labels, conflicts = {}, []
    for tid in draft:
        votes = [p[tid] for p in parts if tid in p]
        if tid in resolved:
            labels[tid] = resolved[tid]
        elif not votes:
            conflicts.append((tid, "chưa có nhãn"))
        elif len(votes) == 2 and any(votes[0][k].strip() != votes[1][k].strip() for k in ("same_as", "type", "plate")):
            conflicts.append((tid, f"{dict(votes[0])} != {dict(votes[1])}"))
        else:
            labels[tid] = votes[0]
    bad = [(tid, r["type"]) for tid, r in labels.items() if r["type"].strip() not in TYPES]
    if conflicts or bad:
        for c in conflicts + bad:
            print("Cần xử lý:", c)
        sys.exit(1)
    with open(gt_path(video), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_id", "frame", "x1", "y1", "x2", "y2", "type", "plate"])
        for tid in sorted(labels):
            r, d = labels[tid], draft[tid]
            gid = r["same_as"].strip() or str(tid)
            w.writerow([gid, d["frame"], d["x1"], d["y1"], d["x2"], d["y2"], r["type"].strip(),
                        normalize_plate(r["plate"])])
    print(f"Đã ghi {len(labels)} dòng, {len({(labels[t]['same_as'].strip() or str(t)) for t in labels})} xe")


if __name__ == "__main__":
    main()
