import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Compare multi-object trackers (ByteTrack, BoT-SORT, ByteTrack variants) on the team videos.

Without ground truth it reports proxy measures: number of IDs, track length, share of short tracks
(fragments) and FPS. With a MOTChallenge gt.txt (e.g. a 300-frame clip labelled in CVAT and exported
as "MOT 1.1") it also reports MOTA, IDF1, ID switches, MT/ML via motmetrics.

Usage:
  python tracker_compare.py --weights model/Wiliv5_best.pt --video test/test_3.mp4 --out outputs/reports/trackers
  python tracker_compare.py --weights model/Wiliv5_best.pt --video test/test_3.mp4 --gt clip_gt/gt.txt --out outputs/reports/trackers
"""
import argparse, csv, os, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils import ROOT

TRACKERS = {  # name -> (base yaml, overrides)
    "bytetrack": ("bytetrack.yaml", {}),
    "bytetrack_buf60": ("bytetrack.yaml", {"track_buffer": 60}),
    "bytetrack_match0.9": ("bytetrack.yaml", {"match_thresh": 0.9}),
    "botsort": ("botsort.yaml", {"with_reid": False}),
}


def find_default_weights():
    cands = [
        "model/Wiliv5_best.pt",
        "model/v62.pt",
        "model/yolo11l.pt",
        "Wiliv5_best.pt"
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def find_default_video():
    cands = [
        "test/test_3.mp4",
        "test/test_1.mp4",
        "test/Download.mp4"
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    # Hoặc tìm file mp4 bất kỳ trong test/
    test_dir = Path("test")
    if test_dir.exists():
        mp4s = list(test_dir.glob("*.mp4"))
        if mp4s:
            return str(mp4s[0])
    return cands[0]


def tracker_yaml(name, out):
    base, over = TRACKERS[name]
    cfg = yaml.safe_load(open(ROOT / "cfg" / "trackers" / base))
    cfg.update(over)
    p = out / f"{name}.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    return str(p)


def track_video(weights, video, cfg, conf, imgsz):
    model = YOLO(weights)
    rows, t0, n = [], time.perf_counter(), 0
    for f, r in enumerate(model.track(source=video, tracker=cfg, conf=conf, imgsz=imgsz, half=True,
                                      stream=True, persist=True, verbose=False), start=1):
        n += 1
        if r.boxes.id is None:
            continue
        for (x1, y1, x2, y2), tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
            rows.append((f, tid, x1, y1, x2 - x1, y2 - y1, c))
    return rows, n / (time.perf_counter() - t0)


def proxy_metrics(rows, short=10):
    frames = defaultdict(set)
    for f, tid, *_ in rows:
        frames[tid].add(f)
    lengths = np.array([len(v) for v in frames.values()]) if frames else np.array([0])
    gaps = sum(max(v) - min(v) + 1 - len(v) > 0 for v in frames.values())  # tracks with holes
    return dict(ids=len(frames), mean_len=float(lengths.mean()), median_len=float(np.median(lengths)),
                short_share=float((lengths < short).mean()), tracks_with_gaps=int(gaps))


def iou_dist(a, b, iou):
    """1 - IoU between xywh boxes, NaN (= cannot match) below the IoU threshold.
    Own version because motmetrics 1.4 iou_matrix uses np.asfarray, removed in NumPy 2."""
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, 0][:, None], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, 1][:, None], b[:, 1]), 0, None)
    inter = iw * ih
    d = 1 - inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter)
    d[d > 1 - iou] = np.nan
    return d


def mot_metrics(rows, gt_path, iou=0.5):
    import motmetrics as mm
    gt = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    gt = gt[gt[:, 6] != 0] if gt.shape[1] > 6 else gt  # MOT: column 7 = "consider" flag
    acc = mm.MOTAccumulator(auto_id=False)
    hyp = defaultdict(list)
    for f, tid, x, y, w, h, _ in rows:
        hyp[f].append((tid, [x, y, w, h]))
    for f in sorted(set(gt[:, 0].astype(int)) | set(hyp)):
        g = gt[gt[:, 0] == f]
        hy = hyp.get(f, [])
        d = iou_dist(g[:, 2:6], np.array([b for _, b in hy], float).reshape(-1, 4), iou)
        acc.update(g[:, 1].astype(int).tolist(), [t for t, _ in hy], d, frameid=f)
    s = mm.metrics.create().compute(acc, metrics=["mota", "motp", "idf1", "num_switches", "mostly_tracked",
                                                  "mostly_lost", "num_fragmentations"], name="x")
    return {k: float(v) for k, v in s.loc["x"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=find_default_weights(), help="Path to YOLO detector weights")
    ap.add_argument("--video", default=find_default_video(), help="Path to input test video")
    ap.add_argument("--gt", help="MOTChallenge gt.txt (frame,id,x,y,w,h,conf,...), frames from 1")
    ap.add_argument("--out", default="outputs/reports/trackers", help="Output directory for reports")
    ap.add_argument("--conf", type=float, default=0.10, help="same as the pipeline's tracker input confidence")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--trackers", nargs="+", default=list(TRACKERS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    res = []
    for name in a.trackers:
        rows, fps = track_video(a.weights, a.video, tracker_yaml(name, out), a.conf, a.imgsz)
        r = dict(tracker=name, fps=fps, **proxy_metrics(rows))
        if a.gt:
            r.update(mot_metrics(rows, a.gt))
        res.append(r)
        print(r)
        with open(out / f"tracks_{name}.txt", "w") as f:  # MOT format, reusable with TrackEval
            for fr, tid, x, y, w, h, c in rows:
                f.write(f"{fr},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,{c},-1\n")
    with open(out / "trackers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, res[0].keys()); w.writeheader(); w.writerows(res)


if __name__ == "__main__":
    main()
