"""Các luật ghép xe <-> biển số, bật/tắt riêng qua FusionConfig (chọn bằng evaluation/score_demo.py)."""
import statistics
from dataclasses import dataclass

SINGLE_LINE_RATIO = 2.5  # biển 1 dòng (ô tô) dài ~4.5:1, biển 2 dòng ~1.4:1
TALL_ASPECT = 0.75  # box xe máy có người lái w/h ~0.4-0.6; ô tô/bus thấy trọn w/h >= ~1.0
# Các luật ghép; mặc định bật những luật được giữ qua ablation (evaluation/README.md)
RULES = ("expand_assoc", "plate_buffer", "merge_by_plate", "single_line_rule", "aspect_rule")


@dataclass
class FusionConfig:
    expand_assoc: bool = True       # biển ngoài mọi box xe -> thử box nới expand_pct
    plate_buffer: bool = True       # biển chưa có chủ -> chờ tối đa buffer_frames frame
    merge_by_plate: bool = True     # gộp các track cùng biển (ID bị đổi giữa chừng)
    single_line_rule: bool = True   # biển 1 dòng -> không phải xe máy
    aspect_rule: bool = True        # car/bus mà box cao hẹp -> xe máy
    expand_pct: float = 0.20
    buffer_frames: int = 10
    merge_gap_s: float = 3.0

    @classmethod
    def from_names(cls, names):
        unknown = set(names) - set(RULES)
        if unknown:
            raise ValueError(f"Luật ghép không tồn tại: {sorted(unknown)} (có: {RULES})")
        return cls(**{r: r in names for r in RULES})


def expand_box(box, pct):
    """Nới box thêm pct tổng chiều rộng/cao (mỗi bên pct/2)."""
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * pct / 2, (y2 - y1) * pct / 2
    return x1 - dw, y1 - dh, x2 + dw, y2 + dh


def owner_of(plate_box, tracks, expand=0.0):
    """ID xe có box (nới expand) chứa tâm biển; nhiều xe chứa thì chọn box nhỏ nhất; không có -> None."""
    x1, y1, x2, y2 = plate_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    best_id, best_area = None, None
    for t in tracks:
        if t.id is None:
            continue
        vx1, vy1, vx2, vy2 = expand_box(t.box, expand) if expand else t.box
        if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
            area = (vx2 - vx1) * (vy2 - vy1)
            if best_area is None or area < best_area:
                best_id, best_area = t.id, area
    return best_id


class PlateBuffer:
    """Biển đọc được nhưng chưa thuộc xe nào: giữ tối đa max_age frame chờ track xe xuất hiện."""

    def __init__(self, max_age=10, expand=0.20):
        self.max_age = max_age
        self.expand = expand
        self.items = []  # (frame_idx, plate_box, item)

    def add(self, frame_idx, plate_box, item):
        self.items.append((frame_idx, plate_box, item))

    def step(self, frame_idx, tracks):
        assigned, expired, keep = [], [], []
        for f, box, item in self.items:
            vid = owner_of(box, tracks, self.expand)
            if vid is not None:
                assigned.append((vid, item))
            elif frame_idx - f >= self.max_age:
                expired.append(item)
            else:
                keep.append((f, box, item))
        self.items = keep
        return assigned, expired

    def flush(self):
        items = [item for _, _, item in self.items]
        self.items = []
        return items


def plates_similar(a, b):
    """Cùng biển: giống hệt hoặc cùng độ dài và khác đúng 1 vị trí (1 ký tự OCR sai)."""
    if not a or not b or len(a) != len(b):
        return False
    return sum(x != y for x, y in zip(a, b)) <= 1


def group_ids_by_plate(entries, max_gap_s=3.0):
    """entries = [(vid, plate, first_seen)] -> các nhóm vid cùng biển, xuất hiện cách nhau <= max_gap_s."""
    parent = {vid: vid for vid, _, _ in entries}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for i, (va, pa, ta) in enumerate(entries):
        for vb, pb, tb in entries[i + 1:]:
            if ta is None or tb is None or abs(ta - tb) > max_gap_s or not plates_similar(pa, pb):
                continue
            ra, rb = find(va), find(vb)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for vid, _, _ in entries:
        groups.setdefault(find(vid), []).append(vid)
    return sorted(sorted(g) for g in groups.values())


def is_single_line(wh):
    w, h = wh
    return h > 0 and w / h >= SINGLE_LINE_RATIO


def single_line_type(vote, votes, reads, min_reads=2):
    """Biển 1 dòng đọc >= min_reads lần thì xe không phải xe máy: lấy lớp khác có phiếu cao nhất (mặc định car)."""
    if vote[0] != "motorcycle" or sum(is_single_line(it["wh"]) for it in reads if "wh" in it) < min_reads:
        return vote
    others = [c for c, _ in votes.most_common() if c != "motorcycle"]
    return (others[0] if others else "car"), None


def box_touches_border(box, shape, margin=2):
    """Box chạm mép khung hình (xe bị cắt, tỉ lệ w/h không đúng)."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


def aspect_type(vote, aspects, max_aspect=TALL_ASPECT, min_boxes=5):
    """Track được vote car/bus nhưng trung vị w/h các box (không chạm mép) < max_aspect thì là xe máy."""
    if vote[0] not in ("car", "bus") or len(aspects) < min_boxes:
        return vote
    return ("motorcycle", None) if statistics.median(aspects) < max_aspect else vote
