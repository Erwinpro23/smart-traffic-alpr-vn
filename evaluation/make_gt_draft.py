"""Sinh nháp ground truth cho 1 video: mỗi track >= 10 frame -> 1 dòng + crop xe + crop biển.
Chạy: python -m evaluation.make_gt_draft [--video street]"""
import argparse
import csv
from collections import defaultdict

import cv2

from alpr_common import SCRIPT_DIR, get_expanded_crop
from evaluation.score_demo import run_pipeline
from evaluation.videos import VIDEOS
from pipeline import ALPRPipeline

MIN_FRAMES = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="demo", choices=sorted(VIDEOS))
    video = VIDEOS[parser.parse_args().video]
    out_dir = video["dir"]
    crop_dir = out_dir / "gt_crops"
    pipe = ALPRPipeline(SCRIPT_DIR, roi=video["roi"], log_tracks=True)
    run_pipeline(pipe, video=video["path"])
    seen = defaultdict(list)  # track_id -> [(frame, box)]
    for e in pipe.track_log:
        for t in e["tracks"]:
            if t[0] is not None:
                seen[t[0]].append((e["frame"], tuple(t[1:5])))
    rows = {}
    for tid, obs in seen.items():
        if len(obs) >= MIN_FRAMES:
            frame, box = obs[len(obs) // 2]
            rows[tid] = (frame, box)

    crop_dir.mkdir(parents=True, exist_ok=True)
    wanted = defaultdict(list)
    for tid, (frame, box) in rows.items():
        wanted[frame].append((tid, box))
    cap = cv2.VideoCapture(str(video["path"]))
    n = 0
    while wanted:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        for tid, (x1, y1, x2, y2) in wanted.pop(n, []):
            cv2.imwrite(str(crop_dir / f"t{tid}_veh.jpg"), get_expanded_crop(img, x1, y1, x2, y2, pad_pct=0.15))
    cap.release()

    with open(out_dir / "gt_draft.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "frame", "x1", "y1", "x2", "y2", "pred_type", "pred_plate", "veh_crop", "plate_crop"])
        for tid in sorted(rows):
            frame, (x1, y1, x2, y2) = rows[tid]
            best = pipe.plate_voter.best(tid)
            plate_crop = ""
            if best is not None:
                plate_crop = f"gt_crops/t{tid}_plate.jpg"
                c = best["crop"]
                cv2.imwrite(str(out_dir / plate_crop), cv2.resize(c, (c.shape[1] * 3, c.shape[0] * 3)))
            w.writerow([tid, frame, x1, y1, x2, y2, pipe.vehicle_type(tid)[0], best["plate"] if best else "",
                        f"gt_crops/t{tid}_veh.jpg", plate_crop])
    print(f"Đã ghi {len(rows)} dòng nháp vào {out_dir / 'gt_draft.csv'}")


if __name__ == "__main__":
    main()
