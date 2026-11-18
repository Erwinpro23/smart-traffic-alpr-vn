import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""EigenCAM heat maps for a YOLO detector: where in the image the chosen layer responds most.

EigenCAM (Muhammad & Yeasin, 2020) projects the feature map of one layer on its first principal
component; it needs no gradients and no class score, so it works for any detector.

Usage:
  python eigencam.py --weights model/Wiliv5_best.pt --images outputs/extracted_plates/*.jpg --out outputs/reports/cam_plate
  python eigencam.py --images outputs/extracted_plates/*.jpg
"""
import argparse, os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def find_default_weights():
    cands = [
        "model/Wiliv5_best.pt",
        "model/v62.pt",
        "Wiliv5_best.pt"
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def eigencam(act):
    """act: (C, H, W) feature map -> (H, W) map in [0, 1]."""
    c, h, w = act.shape
    a = act.reshape(c, -1).T                      # (HW, C)
    a = a - a.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    cam = (a @ vt[0]).reshape(h, w)
    if -cam.min() > cam.max():                    # the sign of a principal component is arbitrary
        cam = -cam
    cam = np.maximum(cam, 0)
    return cam / (cam.max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=find_default_weights(), help="Path to YOLO weights")
    ap.add_argument("--images", nargs="+", default=[], help="Image paths to visualize")
    ap.add_argument("--out", default="outputs/reports/cam_plate", help="Output directory")
    ap.add_argument("--layer", type=int, default=-2, help="index into model.model.model; -2 = last neck layer")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if not a.images:
        # Tự động tìm vài ảnh mẫu trong outputs/extracted_plates
        plate_dir = Path("outputs/extracted_plates")
        if plate_dir.exists():
            a.images = [str(p) for p in list(plate_dir.glob("*.jpg"))[:5]]
        if not a.images:
            print("⚠️ Vui lòng cung cấp ít nhất 1 ảnh: python eigencam.py --images <anh1.jpg> <anh2.jpg>")
            return

    yolo = YOLO(a.weights)
    net = yolo.model.float().eval()
    dev = next(net.parameters()).device
    feats = {}
    layer = net.model[a.layer]
    layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", (o[0] if isinstance(o, (list, tuple)) else o)))
    print("layer", a.layer, type(layer).__name__)

    for p in a.images:
        img = cv2.imread(p)
        if img is None:
            continue
        x = cv2.resize(img, (a.imgsz, a.imgsz))[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)[None].float().div(255).to(dev)
        with torch.no_grad():
            net(x)
        cam = eigencam(feats["a"][0].float().cpu().numpy())
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        boxes = yolo.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0].plot(img=cv2.addWeighted(img, 0.5, heat, 0.5, 0))
        cv2.imwrite(str(out / (Path(p).stem + "_cam.jpg")), np.hstack([img, boxes]))
    print("wrote", out)


if __name__ == "__main__":
    a = np.zeros((4, 5, 5), np.float32); a[:, 2, 2] = [1, 2, 3, 4]  # self-check: one hot spot -> max there
    assert np.unravel_index(eigencam(a).argmax(), (5, 5)) == (2, 2)
    main()
