"""Bộ duyệt nhãn cho người: mỗi xe GT một ô (crop tại frame GT giữa, kèm ID, loại và biển do AI gán) + CSV để điền.
Dùng: python -m evaluation.review_kit demo street"""
import csv
import sys

import cv2
import numpy as np

from evaluation.scoring import load_gt
from evaluation.videos import EVAL_DIR, VIDEOS, gt_path

OUT = EVAL_DIR / "review"
CELL, COLS, ROWS = 300, 4, 5


def frames_needed(rows):
    by_id = {}
    for r in rows:
        by_id.setdefault(r["gt_id"], []).append(r)
    return {gid: sorted(rs, key=lambda r: r["frame"])[len(rs) // 2] for gid, rs in by_id.items()}


def grab(video_path, wanted):
    cap, n, got = cv2.VideoCapture(str(video_path)), 0, {}
    while len(got) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        n += 1  # frame GT đánh số từ 1, giống pipeline.frame_idx
        if n in wanted:
            got[n] = frame
    cap.release()
    return got


def cell(frame, r):
    x1, y1, x2, y2 = r["box"]
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    h, w = frame.shape[:2]
    crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    s = CELL / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    img = np.full((CELL + 60, CELL, 3), 255, np.uint8)
    img[:crop.shape[0], :crop.shape[1]] = crop
    for i, t in enumerate((f"ID {r['gt_id']}  frame {r['frame']}", f"{r['type']}  {r['plate'] or '-'}")):
        cv2.putText(img, t, (6, CELL + 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def main(videos):
    OUT.mkdir(exist_ok=True)
    for video in videos:
        mids = {g: r for g, r in frames_needed(load_gt(gt_path(video))).items() if r["type"] != "not_vehicle"}
        frames = grab(VIDEOS[video]["path"], {r["frame"] for r in mids.values()})
        ids = sorted(mids, key=int)
        cells = [cell(frames[mids[g]["frame"]], mids[g]) for g in ids]
        per = COLS * ROWS
        for k in range(0, len(cells), per):
            chunk = cells[k:k + per]
            chunk += [np.full_like(cells[0], 255)] * (per - len(chunk))
            grid = np.vstack([np.hstack(chunk[i:i + COLS]) for i in range(0, per, COLS)])
            cv2.imwrite(str(OUT / f"{video}_sheet_{k // per + 1}.jpg"), grid)
        with open(OUT / f"gt_review_{video}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["gt_id", "frame", "ai_type", "ai_plate", "human_type", "human_plate", "note"])
            for g in ids:
                w.writerow([g, mids[g]["frame"], mids[g]["type"], mids[g]["plate"], "", "", ""])
        print(video, len(ids), "xe")


if __name__ == "__main__":
    main(sys.argv[1:])
