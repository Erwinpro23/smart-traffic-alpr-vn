import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .alpr_common import enable_cuda_graphs

COARSE_CLASSES = ["motorcycle", "car", "bus", "truck"]


@dataclass
class Track:
    id: int | None
    box: tuple[int, int, int, int]
    coarse: str
    conf: float


class TypeVoter:
    """Loại xe của mỗi track = lớp detector có tổng confidence lớn nhất qua các frame."""

    def __init__(self):
        self.votes = defaultdict(Counter)

    def process(self, tracks):
        for t in tracks:
            if t.id is not None:
                self.votes[t.id][t.coarse] += t.conf

    def get(self, track_id):
        """Trả về (loại xe, tỷ lệ phiếu của lớp thắng) hoặc ("unknown", None) nếu chưa thấy track."""
        votes = self.votes.get(track_id)
        if not votes:
            return "unknown", None
        label, weight = votes.most_common(1)[0]
        return label, weight / sum(votes.values())

    def seen_ids(self):
        return list(self.votes)


class VehicleTracker:
    """YOLO11 phát hiện xe (4 lớp) + ByteTrack."""

    # conf thấp để ByteTrack dùng box điểm thấp nối dài track (track mới vẫn cần >= new_track_thresh 0.25)
    def __init__(self, weights, device, imgsz=640, conf=0.10):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Không tìm thấy weights detector xe: '{weights}' (chạy train_vehicle_det.py)")
        self.model = YOLO(str(weights))
        self.kwargs = {"persist": True, "conf": conf, "imgsz": imgsz, "device": device,
                       "tracker": "bytetrack.yaml", "verbose": False}
        if device != "cpu":
            self.kwargs["quantize"] = 16  # FP16 (ultralytics >= 8.4 thay tham số half cũ, half in cảnh báo mỗi khung)
        # Model nhiều lớp hơn (vd. COCO 80 lớp): chỉ giữ 4 lớp xe
        vehicle_ids = [i for i, n in self.model.names.items() if n in COARSE_CLASSES]
        if len(vehicle_ids) < len(self.model.names):
            self.kwargs["classes"] = vehicle_ids

    def reset(self):
        """Xóa mọi track và đánh số ID lại từ 1 (sang video mới / phát lại), giữ nguyên model đã nạp."""
        for tracker in getattr(self.model.predictor, "trackers", None) or []:
            tracker.reset()

    def update(self, frame):
        r = self.model.track(frame, **self.kwargs)[0]
        enable_cuda_graphs(self.model)
        tracks = []
        if r.boxes is None:
            return tracks
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            if not all(map(math.isfinite, xyxy)):  # ByteTrack hiếm khi ra track NaN (box cao 0 -> chia 0): bỏ qua
                continue
            track_id = int(box.id[0]) if box.id is not None else None
            x1, y1, x2, y2 = map(int, xyxy)
            tracks.append(Track(track_id, (x1, y1, x2, y2), r.names[int(box.cls[0])], float(box.conf[0])))
        return tracks
