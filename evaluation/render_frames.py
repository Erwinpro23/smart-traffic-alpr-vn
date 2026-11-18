"""Render frame minh hoạ cho báo cáo theo luật cố định (không chọn tay). Dùng: python -m evaluation.render_frames
Luật chọn:
- Video có nhãn (demo, street), detector plain và balanced: với mỗi kết quả (correct, wrong_type, missed) lấy K xe đầu
  theo gt_id, render tại frame GT giữa của xe đó. Video demo + plain: thêm 2 xe đọc biển đúng và 2 xe đọc sai đầu tiên.
- Video không nhãn: IMG_8328 tại 25/50/75% độ dài (plain); các video từ trên cao tại 50% độ dài, plain và COCO ghép
  cạnh nhau.
Ghi ảnh + manifest.json vào docs/report/main/figures/demo/."""
import json

import cv2
import numpy as np

from alpr_common import SCRIPT_DIR
from draw import annotate
from evaluation.scoring import load_gt
from evaluation.videos import VIDEOS, gt_path
from fusion import RULES, FusionConfig
from pipeline import ALPRPipeline

OUT = SCRIPT_DIR / "docs" / "report" / "main" / "figures" / "demo"
PV = SCRIPT_DIR / "docs" / "paper" / "analysis" / "per_vehicle_out"
WEIGHTS = {"plain": "weights/vehicle_det.pt", "balanced": "weights/vehicle_det_bal.pt", "coco": "yolo11s.pt"}
OVERHEAD = ["tren_cao_saigon_Kv2Z_NHz9KE", "tren_cao_ha_noi_UzKw4O9qOZo", "pho_tu_tren_cao_l9Euku8i3b0"]
K = 3
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 85]


def outcome(v):
    if not v["matched"]:
        return "missed"
    return "correct" if v["pred_type"] == v["gt_type"] else "wrong_type"


def middle_frames(rows):
    frames = {}
    for r in rows:
        frames.setdefault(r["gt_id"], []).append(r["frame"])
    return {g: sorted(f)[len(f) // 2] for g, f in frames.items()}


def pick_labelled(video, det, k=K):
    veh = json.load(open(PV / f"{video}_{det}.json", encoding="utf-8"))["by_iou"]["0.5"]["vehicles"]
    mid = middle_frames(load_gt(gt_path(video)))
    picks = []
    for oc in ("correct", "wrong_type", "missed"):
        ids = sorted((g for g, v in veh.items() if outcome(v) == oc), key=int)[:k]
        picks += [{"video": video, "det": det, "frame": mid[g], "gt_id": g, "outcome": oc,
                   "gt_type": veh[g]["gt_type"], "pred_type": veh[g]["pred_type"],
                   "gt_plate": veh[g]["gt_plate"], "pred_plate": veh[g]["pred_plate"],
                   "rule": f"first {k} {oc} by gt_id"} for g in ids]
    if video == "demo" and det == "plain":
        plated = sorted((g for g, v in veh.items() if v["gt_plate"]), key=int)
        for want, tag in ((True, "plate_ok"), (False, "plate_wrong")):
            ids = [g for g in plated if (veh[g]["pred_plate"] == veh[g]["gt_plate"]) == want][:2]
            picks += [{"video": video, "det": det, "frame": mid[g], "gt_id": g, "outcome": tag,
                       "gt_type": veh[g]["gt_type"], "pred_type": veh[g]["pred_type"],
                       "gt_plate": veh[g]["gt_plate"], "pred_plate": veh[g]["pred_plate"],
                       "rule": f"first 2 {tag} by gt_id"} for g in ids]
    return picks


def render(path, roi, weights, frames):
    """Chạy pipeline từ đầu video tới frame lớn nhất cần (để tracker và bỏ phiếu có lịch sử thật), vẽ các frame cần."""
    pipe = ALPRPipeline(SCRIPT_DIR, roi=roi, vehicle_weights=weights, fusion=FusionConfig.from_names(list(RULES)))
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n, got, last = 0, {}, max(frames)
    while n < last:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        res = pipe.process(frame, round(n / fps, 2))
        if n in frames:
            got[n] = annotate(frame, res, roi, hud=f"frame {n}")
    cap.release()
    del pipe
    return got


def n_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for video in ("demo", "street"):
        for det in ("plain", "balanced"):
            picks = pick_labelled(video, det)
            imgs = render(VIDEOS[video]["path"], VIDEOS[video]["roi"], WEIGHTS[det], {p["frame"] for p in picks})
            for p in picks:
                f = f"{video}_{det}_{p['outcome']}_gt{p['gt_id']}.jpg"
                cv2.imwrite(str(OUT / f), imgs[p["frame"]], JPEG)
                manifest.append({"file": f, **p})
    junction = SCRIPT_DIR / "demo_videos" / "IMG_8328.MP4"
    fr = {int(n_frames(junction) * q) for q in (0.25, 0.5, 0.75)}
    for n, img in sorted(render(junction, None, WEIGHTS["plain"], fr).items()):
        f = f"junction_plain_f{n}.jpg"
        cv2.imwrite(str(OUT / f), img, JPEG)
        manifest.append({"file": f, "video": "IMG_8328", "det": "plain", "frame": n, "rule": "25/50/75% of length"})
    for stem in OVERHEAD:
        path = SCRIPT_DIR / "demo_videos" / f"{stem}.mp4"
        n = int(n_frames(path) * 0.5)
        a = render(path, None, WEIGHTS["plain"], {n})[n]
        b = render(path, None, WEIGHTS["coco"], {n})[n]
        f = f"overhead_{stem}_plain_vs_coco.jpg"
        cv2.imwrite(str(OUT / f), np.hstack([a, b]), JPEG)
        manifest.append({"file": f, "video": stem, "det": "plain|coco", "frame": n, "rule": "50% of length"})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(manifest), "images")


if __name__ == "__main__":
    main()
