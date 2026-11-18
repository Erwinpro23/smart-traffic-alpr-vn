import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Chạy pipeline xe + biển số trên video. Chạy: python run_video.py [--video PATH] [--max-frames N]"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from alpr.alpr_common import DEMO_ROI, DEMO_VIDEO, SCRIPT_DIR
from alpr.pipeline import ALPRPipeline

DEFAULT_VIDEO = DEMO_VIDEO
CROP_DIR = SCRIPT_DIR / "extracted_plates"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--out", default=str(SCRIPT_DIR / "ket_qua_xe_va_bien_so.csv"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--full-frame", action="store_true", help="Không cắt ROI (mặc định cắt khung giao diện của video demo)")
    args = parser.parse_args()

    use_roi = not args.full_frame and Path(args.video).resolve() == DEMO_VIDEO.resolve()
    pipe = ALPRPipeline(SCRIPT_DIR, roi=DEMO_ROI if use_roi else None)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {args.video}")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sums = {"vehicle_ms": 0.0, "type_ms": 0.0, "plate_ms": 0.0, "ocr_ms": 0.0}
    n = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        n += 1
        res = pipe.process(frame, round(n / fps_video, 2))
        for k in sums:
            sums[k] += res.timings[k]
    cap.release()
    elapsed = time.perf_counter() - t_start

    CROP_DIR.mkdir(exist_ok=True)
    rows = []
    for r in pipe.finalize():
        crop_path = ""
        if r["crop"] is not None and r["plate"]:
            crop_path = str(CROP_DIR / f"{r['plate']}_veh{r['vehicle_id']}.jpg")
            cv2.imwrite(crop_path, r["crop"])
        rows.append({
            "ID xe": r["vehicle_id"] if r["vehicle_id"] is not None else "Untracked",
            "Các ID gộp": ",".join(map(str, r["vehicle_ids"])),
            "Loại xe": r["type"],
            "Tỷ lệ phiếu loại": (f"{r['type_conf'] * 100:.1f}%" if r["type_conf"] is not None
                                 else "" if r["vehicle_id"] is None else "theo biển số"),
            "Biển số": r["plate"],
            "Tỷ lệ đồng thuận": "" if r["consensus"] is None else f"{r['consensus']}%",
            "Trust": "" if r["trust_score"] is None else f"{r['trust_score']}%",
            "Thời điểm (s)": r["first_seen"],
            "File crop": crop_path,
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nFrames: {n} | Thời gian: {elapsed:.1f}s | FPS trung bình: {n / elapsed:.2f}")
    print("ms/frame:", {k: round(v / max(1, n), 1) for k, v in sums.items()})
    print("Phân bố loại xe:", df["Loại xe"].value_counts().to_dict())
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
