"""Bước 2 của nhãn held-out: vẽ ảnh duyệt và áp dụng sửa đổi.
python -m evaluation.heldout.labels render <video>  -> review2/*.jpg (box cao >= MIN_H đánh số; box nhỏ hơn vẽ mờ)
python -m evaluation.heldout.labels apply <video>   -> labels/*.txt (YOLO, lớp 0..3) + gt.json
Sửa đổi nằm trong evaluation/heldout/<video>/corrections.json:
  {"<frame>.jpg": {"drop": [i, ...], "cls": {"i": "car"}, "add": [["motorcycle", x1, y1, x2, y2], ...]}}
Frame không có trong corrections.json = giữ nguyên đề xuất (đã duyệt, không cần sửa)."""
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
CLASSES = ["motorcycle", "car", "bus", "truck"]
MIN_H = 32  # chỉ chấm xe có box cao >= 32 px (áp dụng cho cả nhãn và dự đoán)
# Vùng bỏ qua (x1, y1, x2, y2): box có tâm nằm trong vùng không được chấm, cả nhãn lẫn dự đoán.
# hem: cụm xe máy đậu sát nhau sau dòng ngày giờ ở đầu hẻm, không vẽ được box riêng từng xe cho chắc chắn.
IGNORE = {"hem": [(200, 0, 500, 165)], "junction": []}
DEDUP_IOU = 0.6


def in_ignore(video, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in IGNORE.get(video, []))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def dedup(boxes):
    """Sau khi đổi lớp có thể có 2 box cùng lớp trên một xe: giữ box đứng trước (điểm đề xuất cao hơn)."""
    keep = []
    for b in boxes:
        if all(not (k["cls"] == b["cls"] and iou(k["box"], b["box"]) > DEDUP_IOU) for k in keep):
            keep.append(b)
    return keep


COLORS = {"motorcycle": (255, 200, 0), "car": (0, 255, 128), "bus": (255, 0, 200), "truck": (60, 60, 255)}


def counted(box):
    return box[3] - box[1] >= MIN_H


def render(video):
    vdir = HERE / video
    props = json.loads((vdir / "proposals.json").read_text(encoding="utf-8"))
    (vdir / "review2").mkdir(exist_ok=True)
    for fn, dets in props.items():
        im = cv2.imread(str(vdir / "frames" / fn))
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["box"]
            if not counted(d["box"]):
                cv2.rectangle(im, (x1, y1), (x2, y2), (120, 120, 120), 1)
                continue
            c = COLORS[d["cls"]]
            cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
            t = f"{i} {d['cls'][:4]}"
            cv2.putText(im, t, (x1 + 2, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(im, t, (x1 + 2, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        cv2.putText(im, fn, (10, im.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(vdir / "review2" / fn), im)


def apply(video):
    vdir = HERE / video
    props = json.loads((vdir / "proposals.json").read_text(encoding="utf-8"))
    cpath = vdir / "corrections.json"
    corr = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    (vdir / "labels").mkdir(exist_ok=True)
    gt, log = {}, {"dropped": 0, "reclassed": 0, "added": 0, "kept": 0}
    for fn, dets in props.items():
        c = corr.get(fn, {})
        drop = set(c.get("drop", []))
        recls = {int(k): v for k, v in c.get("cls", {}).items()}
        boxes = []
        for i, d in enumerate(dets):
            if i in drop:
                log["dropped"] += 1
                continue
            cls = recls.get(i, d["cls"])
            if i in recls and recls[i] != d["cls"]:
                log["reclassed"] += 1
            else:
                log["kept"] += 1
            boxes.append({"cls": cls, "box": d["box"]})
        for a in c.get("add", []):
            assert a[0] in CLASSES, a
            boxes.append({"cls": a[0], "box": [int(v) for v in a[1:]]})
            log["added"] += 1
        n0 = len(boxes)
        boxes = dedup(boxes)
        log["deduplicated"] = log.get("deduplicated", 0) + n0 - len(boxes)
        gt[fn] = boxes
        h, w = cv2.imread(str(vdir / "frames" / fn)).shape[:2]
        lines = []
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            lines.append(f"{CLASSES.index(b['cls'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (vdir / "labels" / fn.replace(".jpg", ".txt")).write_text("\n".join(lines), encoding="utf-8")
    (vdir / "gt.json").write_text(json.dumps({"min_h": MIN_H, "ignore": IGNORE[video], "log": log, "frames": gt}, indent=1), encoding="utf-8")
    n = sum(counted(b["box"]) and not in_ignore(video, b["box"]) for bs in gt.values() for b in bs)
    print(video, log, "boxes counted (h >= %d):" % MIN_H, n)


if __name__ == "__main__":
    {"render": render, "apply": apply}[sys.argv[1]](sys.argv[2])
