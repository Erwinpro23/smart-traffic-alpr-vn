"""Tiền xử lý ảnh biển trước OCR, theo báo cáo mục 6 (Blur filtering, Plate-level preprocessing):

- Độ nét: phương sai Laplacian trên vùng biển. Crop quá mờ thì không gửi OCR, và độ nét làm trọng số phiếu.
- Nắn phối cảnh: nhị phân Otsu, contour sáng lớn nhất xấp xỉ thành tứ giác, 4 góc -> perspective transform.
  Không tìm được tứ giác hợp lý thì dùng crop thẳng trục.
- Làm nét: CLAHE trên kênh sáng + unsharp mask.
- Tách biển 2 dòng thành nửa trên và nửa dưới để đọc lại.
"""
import cv2
import numpy as np

SHARP_SIZE = (128, 64)  # (w, h) cỡ đầu vào OCR; đo độ nét ở cùng cỡ để biển to/nhỏ so được với nhau
QUAD_MIN_AREA = 0.35    # tứ giác phải chiếm 35-97% crop: nhỏ hơn là chữ/vết bẩn, lớn hơn là thân xe sáng
QUAD_MAX_AREA = 0.97
QUAD_ASPECT = (0.9, 6.5)  # biển xe máy ~1.4:1, ô tô 1 dòng ~4.7:1
QUAD_BORDER = 2         # góc tứ giác cách mép crop <= 2 px: vùng sáng tràn ra nền (đường, thân xe), bỏ
RECT_MARGIN = 0.08      # chừa lề quanh biển đã nắn như crop thường (get_expanded_crop), OCR quen đọc có lề
SPLIT_OVERLAP = 0.06


def sharpness(img):
    """Phương sai Laplacian của ảnh xám đã đưa về SHARP_SIZE."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, SHARP_SIZE, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def order_corners(pts):
    """4 điểm -> thứ tự trên-trái, trên-phải, dưới-phải, dưới-trái."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def find_quad(crop):
    """4 góc biển trong crop (thứ tự order_corners) hoặc None."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = crop.shape[0] * crop.shape[1]
    if not QUAD_MIN_AREA * area <= cv2.contourArea(c) <= QUAD_MAX_AREA * area:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = order_corners(approx)
            ch, cw = crop.shape[:2]
            if (q[:, 0].min() <= QUAD_BORDER or q[:, 1].min() <= QUAD_BORDER
                    or q[:, 0].max() >= cw - 1 - QUAD_BORDER or q[:, 1].max() >= ch - 1 - QUAD_BORDER):
                return None
            w, h = quad_size(q)
            if h > 0 and QUAD_ASPECT[0] <= w / h <= QUAD_ASPECT[1]:
                return q
            return None
    return None


def quad_size(q):
    tl, tr, br, bl = q
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return float(w), float(h)


def rectify(crop):
    """(ảnh đã nắn, True) hoặc (crop gốc, False) nếu không tìm thấy tứ giác."""
    q = find_quad(crop)
    if q is None:
        return crop, False
    w, h = quad_size(q)
    mx, my = w * RECT_MARGIN, h * RECT_MARGIN
    W, H = max(int(round(w + 2 * mx)), 1), max(int(round(h + 2 * my)), 1)
    dst = np.array([[mx, my], [mx + w - 1, my], [mx + w - 1, my + h - 1], [mx, my + h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(crop, m, (W, H), borderMode=cv2.BORDER_REPLICATE), True


def enhance(img):
    """CLAHE trên kênh sáng + unsharp mask."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.5)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


def split_two_line(img):
    """Nửa trên, nửa dưới của biển 2 dòng (chồng nhau SPLIT_OVERLAP để không cắt chân chữ)."""
    h = img.shape[0]
    ov = max(1, int(h * SPLIT_OVERLAP))
    return img[:h // 2 + ov], img[h // 2 - ov:]
