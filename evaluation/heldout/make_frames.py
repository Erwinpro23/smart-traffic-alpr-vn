"""Chuẩn bị dữ liệu held-out (camera không có trong train): cắt video và lấy N frame cách đều, không dùng detector.
Dùng: python -m evaluation.heldout.make_frames
- hem: video quảng cáo quay màn hình NVR Hikvision; chỉ giữ 3 đoạn xem trực tiếp, cắt lấy khung camera
  (bỏ thanh menu, cột ảnh biển số), ghép thành demo_videos/hem_cut.mp4.
- junction: demo_videos/IMG_8328.MP4 (nhóm tự quay), dùng nguyên khung.
Ghi evaluation/heldout/<video>/frames/*.jpg và frames.json (frame gốc của từng ảnh)."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
N = 40
# (frame đầu, frame cuối, vùng cắt x1, y1, x2, y2) trong video gốc 1920x1080
HEM_SEGMENTS = [(1035, 1800, 352, 0, 1920, 1040),   # xem trực tiếp toàn màn hình
                (1830, 2160, 352, 80, 1920, 1040),  # xem trực tiếp có thanh menu NVR
                (6450, 6900, 352, 80, 1920, 1040)]
OUT_W, OUT_H = 1568, 1040


def letterbox(img):
    h, w = img.shape[:2]
    s = min(OUT_W / w, OUT_H / h)
    r = cv2.resize(img, (round(w * s), round(h * s)))
    canvas = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    y, x = (OUT_H - r.shape[0]) // 2, (OUT_W - r.shape[1]) // 2
    canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
    return canvas


def cut_hem():
    src = cv2.VideoCapture(str(ROOT / "demo_videos" / "hem_hikvision_hr-ZCWK1WTk.mp4"))
    fps = src.get(cv2.CAP_PROP_FPS)
    dst = cv2.VideoWriter(str(ROOT / "demo_videos" / "hem_cut.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (OUT_W, OUT_H))
    origin, i = [], 0
    while True:
        ok = src.grab()
        if not ok:
            break
        for a, b, x1, y1, x2, y2 in HEM_SEGMENTS:
            if a <= i <= b:
                _, f = src.retrieve()
                dst.write(letterbox(f[y1:y2, x1:x2]))
                origin.append(i)
        i += 1
    src.release()
    dst.release()
    return ROOT / "demo_videos" / "hem_cut.mp4", origin


def sample(video_path, name, origin=None):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sorted({int((k + 0.5) * n / N) for k in range(N)})
    out = HERE / name / "frames"
    out.mkdir(parents=True, exist_ok=True)
    meta, i = {}, 0
    while i <= want[-1]:
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            _, f = cap.retrieve()
            fn = f"{name}_{i:05d}.jpg"
            cv2.imwrite(str(out / fn), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            meta[fn] = {"frame": i, "source_frame": origin[i] if origin else i}
        i += 1
    cap.release()
    (HERE / name / "frames.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(name, n, "frames ->", len(meta))


if __name__ == "__main__":
    path, origin = cut_hem()
    sample(path, "hem", origin)
    sample(ROOT / "demo_videos" / "IMG_8328.MP4", "junction")
