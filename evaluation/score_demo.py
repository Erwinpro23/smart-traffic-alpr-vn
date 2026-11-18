"""Chạy pipeline với 1 cấu hình trên 1 video có GT rồi chấm điểm. Ví dụ:
python -m evaluation.score_demo --name wiliv5_768 --plate-model wiliv5_768 --plate-imgsz 768 --fusion expand_assoc
python -m evaluation.score_demo --video street --name street_base --fusion expand_assoc plate_buffer
"""
import argparse
import json
import time

import cv2

from alpr_common import DEMO_VIDEO, SCRIPT_DIR
from evaluation.scoring import load_gt, match_gt, score
from evaluation.videos import EVAL_DIR, VIDEOS, gt_path
from fusion import RULES, FusionConfig
from pipeline import ALPRPipeline

GT_PATH = gt_path("demo")
RESULTS_DIR = EVAL_DIR / "results"


def run_pipeline(pipe, max_frames=0, video=DEMO_VIDEO):
    cap = cv2.VideoCapture(str(video))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n, t0 = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        n += 1
        pipe.process(frame, round(n / fps_video, 2))
    cap.release()
    return n, n / (time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--video", default="demo", choices=sorted(VIDEOS))
    parser.add_argument("--plate-model", default="v62")
    parser.add_argument("--plate-imgsz", type=int, default=640)
    parser.add_argument("--vehicle-weights", default=None)
    parser.add_argument("--fusion", nargs="*", default=list(RULES), help="Luật bật (mặc định: tất cả; --fusion không tên = tắt hết)")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    video = VIDEOS[args.video]
    pipe = ALPRPipeline(SCRIPT_DIR, roi=video["roi"], plate_model=args.plate_model, plate_imgsz=args.plate_imgsz,
                        vehicle_weights=args.vehicle_weights, fusion=FusionConfig.from_names(args.fusion),
                        log_tracks=True)
    frames, fps = run_pipeline(pipe, args.max_frames, video["path"])
    records = pipe.finalize()
    gt_rows = load_gt(gt_path(args.video))
    result = score(gt_rows, match_gt(gt_rows, pipe.track_log), records)
    result.update({"name": args.name, "video": args.video, "plate_model": args.plate_model, "plate_imgsz": args.plate_imgsz,
                   "vehicle_weights": args.vehicle_weights or "weights/vehicle_det.pt", "fusion": args.fusion,
                   "frames": frames, "fps": round(fps, 2), "n_records": len(records)})
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
