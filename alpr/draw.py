"""Vẽ kết quả pipeline lên frame (dùng chung cho stream_app và evaluation/render_frames)."""
import cv2

from .pipeline import roi_pixels

TYPE_COLORS = {"motorcycle": (255, 200, 0), "car": (0, 255, 128), "truck": (60, 60, 255),
               "bus": (255, 0, 200), "unknown": (180, 180, 180)}


def annotate(frame, res, roi, hud=None):
    """Cắt vùng ROI, vẽ box xe màu theo loại với nhãn "#id loại | biển"; hud != None -> thanh thông tin cao 45px (hud="" vẽ thanh trống để ghi chữ sau)."""
    rx0, ry0, rx1, ry1 = roi_pixels(frame.shape, roi)
    img = frame[ry0:ry1, rx0:rx1].copy()
    top = 45 if hud is not None else 0
    for t in res.tracks:
        x1, y1, x2, y2 = t.box[0] - rx0, t.box[1] - ry0, t.box[2] - rx0, t.box[3] - ry0
        vtype, _ = res.types.get(t.id, (t.coarse, None)) if t.id is not None else (t.coarse, None)
        best = res.plates.get(t.id) if t.id is not None else None
        label = f"#{t.id} {vtype}" if t.id is not None else vtype
        if best:
            label += f" | {best['plate']}"
        color = TYPE_COLORS.get(vtype, TYPE_COLORS["unknown"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = y1 if y1 - 24 >= top else max(y1, top) + 24  # tránh bị thanh HUD che
        cv2.rectangle(img, (x1, ly - 24), (x1 + lw + 10, ly), color, -1)
        cv2.putText(img, label, (x1 + 5, ly - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    if hud is not None:
        cv2.rectangle(img, (0, 0), (img.shape[1], 45), (15, 15, 20), -1)
        if hud:
            draw_hud_text(img, hud)
    return img


def draw_hud_text(img, text):
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)
