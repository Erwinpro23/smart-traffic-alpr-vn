"""Bước 1 của nhãn held-out: đề xuất box bằng Grounding DINO (độc lập với các detector YOLO được chấm),
rồi vẽ box đã đánh số lên ảnh để người/AI duyệt. Dùng: python -m evaluation.heldout.propose <video_name>
Đọc  evaluation/heldout/<video>/frames/*.jpg
Ghi  evaluation/heldout/<video>/proposals.json, evaluation/heldout/<video>/review/*.jpg"""
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

HERE = Path(__file__).resolve().parent
MODEL = "IDEA-Research/grounding-dino-tiny"
PROMPT = "motorcycle. car. bus. truck."
CLASSES = ["motorcycle", "car", "bus", "truck"]
BOX_T, TEXT_T = 0.25, 0.25
COLORS = {"motorcycle": (255, 200, 0), "car": (0, 255, 128), "bus": (255, 0, 200), "truck": (60, 60, 255)}


def to_class(label):
    for c in CLASSES:
        if c in label:
            return c
    return None


def nms_across(dets, iou_t=0.7):
    """Grounding DINO có thể trả cùng một vật cho 2 nhãn: giữ box điểm cao nhất."""
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(iou(d["box"], k["box"]) < iou_t for k in keep):
            keep.append(d)
    return keep


def main(video):
    vdir = HERE / video
    frames = sorted((vdir / "frames").glob("*.jpg"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(dev)
    out = {}
    (vdir / "review").mkdir(exist_ok=True)
    for fp in frames:
        img = Image.open(fp).convert("RGB")
        inputs = proc(images=img, text=PROMPT, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inputs)
        r = proc.post_process_grounded_object_detection(o, inputs.input_ids, threshold=BOX_T, text_threshold=TEXT_T,
                                                        target_sizes=[img.size[::-1]])[0]
        labels = r.get("text_labels", r.get("labels"))
        dets = []
        for b, s, l in zip(r["boxes"].tolist(), r["scores"].tolist(), labels):
            c = to_class(str(l))
            w, h = img.size
            if c and (b[2] - b[0]) < 0.9 * w:
                dets.append({"cls": c, "score": round(s, 3), "box": [round(x) for x in b]})
        dets = nms_across(dets)
        out[fp.name] = dets
        im = cv2.imread(str(fp))
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(im, (x1, y1), (x2, y2), COLORS[d["cls"]], 2)
            cv2.putText(im, f"{i}:{d['cls'][:4]}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, f"{i}:{d['cls'][:4]}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLORS[d["cls"]], 1, cv2.LINE_AA)
        cv2.imwrite(str(vdir / "review" / fp.name), im)
        print(fp.name, len(dets), flush=True)
    (vdir / "proposals.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
