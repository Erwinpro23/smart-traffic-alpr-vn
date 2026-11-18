import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from alpr.plate_preprocess import enhance, find_quad, order_corners, rectify, sharpness, split_two_line


def plate_image(warp=True):
    """Biển trắng chữ đen 200x60 trên nền tối, nghiêng phối cảnh."""
    plate = np.full((60, 200, 3), 235, np.uint8)
    cv2.putText(plate, "51G30069", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    canvas = np.full((110, 260, 3), 40, np.uint8)
    src = np.float32([[0, 0], [199, 0], [199, 59], [0, 59]])
    dst = np.float32([[30, 25], [225, 15], [232, 80], [25, 92]]) if warp else (src + 25).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    mask = cv2.warpPerspective(np.full((60, 200), 255, np.uint8), m, (260, 110))
    warped = cv2.warpPerspective(plate, m, (260, 110))
    canvas[mask > 0] = warped[mask > 0]
    return canvas, dst


def test_order_corners():
    pts = [[10, 50], [0, 0], [50, 0], [60, 50]]
    assert order_corners(pts).tolist() == [[0, 0], [50, 0], [60, 50], [10, 50]]


def test_find_quad_recovers_plate_corners():
    img, dst = plate_image()
    q = find_quad(img)
    assert q is not None
    assert np.abs(q - dst).max() < 5


def test_rectify_straightens_plate():
    img, _ = plate_image()
    out, ok = rectify(img)
    assert ok
    h, w = out.shape[:2]
    assert 2.8 < w / h < 3.9  # biển gốc 200x60, cộng lề 8% mỗi phía
    assert out[8:-8, 12:-12].mean() > 160  # bên trong lề chỉ còn nền biển trắng và chữ
    assert out[:3].mean() < 120            # lề trên là nền tối quanh biển


def test_rectify_falls_back_without_plate():
    flat = np.full((50, 120, 3), 128, np.uint8)
    out, ok = rectify(flat)
    assert not ok and out is flat


def test_sharpness_drops_with_blur():
    img, _ = plate_image(warp=False)
    sharp = sharpness(img)
    assert sharpness(cv2.GaussianBlur(img, (0, 0), 3)) < sharp / 4


def test_enhance_keeps_shape():
    img, _ = plate_image()
    assert enhance(img).shape == img.shape


def test_split_two_line_halves_overlap():
    img = np.zeros((100, 140, 3), np.uint8)
    top, bottom = split_two_line(img)
    assert top.shape[0] == 56 and bottom.shape[0] == 56


def test_find_quad_rejects_region_touching_crop_border():
    img, _ = plate_image(warp=False)
    assert find_quad(img) is not None
    bright = img.copy()
    bright[:, :40] = 235  # nền sáng dính vào biển và chạm mép trái
    assert find_quad(bright) is None
