import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .alpr_common import (enable_cuda_graphs, get_expanded_crop, load_ocr, load_plate_model, normalize_plate,
                         plate_vehicle_class, setup_cuda_dlls)
from .plate_preprocess import enhance as enhance_crop, rectify, sharpness, split_two_line
from .plate_text import clean as clean_plate, correct as correct_plate, is_valid as valid_plate, n_changes
from .fusion import (FusionConfig, PlateBuffer, aspect_type, box_touches_border, group_ids_by_plate, is_single_line,
                    owner_of, single_line_type)
from .vehicle_module import Track, TypeVoter, VehicleTracker

MIN_PLATE_W, MIN_PLATE_H = 35, 15
PLATE_CONF = 0.20
RETRY_CONF = 0.80  # OCR tin dưới mức này (hoặc chuỗi sai cú pháp) -> đọc lại (làm nét, tách 2 dòng)
SHARP_MIN = 50.0   # crop có phương sai Laplacian (ở 128x64) dưới mức này không gửi OCR (mất nét hẳn, báo cáo hình f_blur)
SHARP_REF = 100.0  # trọng số độ nét của phiếu = sharp / (sharp + SHARP_REF)
FIX_PENALTY = 0.10  # mỗi ký tự phải sửa theo cú pháp trừ 10% điểm ứng viên và trust
# Biển chỉ được chốt khi xe có ít nhất 1 lần đọc OCR tin >= mức này. Biển rác (lan can, biển báo, biển quá nhỏ)
# vẫn đúng cú pháp nhưng OCR không bao giờ chắc: trên 4 video cầu đi bộ giữ đủ biển đúng (88/88 ở 1080p, 112/112
# ở 4K) và bỏ 75/92 và 63/85 chuỗi rác (evaluation/plate_reading.md).
MIN_BEST_CONF = 0.85


@dataclass
class PlateConfig:
    """Các bước đọc biển theo báo cáo mục 5-6, bật hết mặc định. Kết quả đo từng bước trên 4 video cầu đi bộ:
    evaluation/plate_reading.md."""
    syntax: bool = True         # sửa/loại chuỗi theo cú pháp biển VN (63 mã tỉnh, sửa O/0, I/1, B/8, S/5 theo vị trí)
    rectify: bool = True        # nắn phối cảnh từ 4 góc biển (Otsu + tứ giác), không thấy tứ giác thì giữ crop
    sharp_min: float = SHARP_MIN  # kiểm độ nét: crop mờ hơn mức này không đọc (0 = không lọc, chỉ làm trọng số)
    enhance: bool = True        # OCR không chắc -> đọc lại trên ảnh CLAHE + unsharp và trên 2 nửa biển 2 dòng
    position_vote: bool = True  # bỏ phiếu từng vị trí ký tự, trọng số = xác suất ký tự x độ nét
    min_best_conf: float = MIN_BEST_CONF  # lọc biển rác: cần >= 1 lần đọc OCR tin từ mức này (0 = tắt)
    size_vote: bool = False     # (thử nghiệm) phiếu theo bề rộng ảnh biển, không chốt sớm
    max_reads: int = 7
    # Tốc độ: bỏ dò biển ở khung không có xe nào cần đọc (không có xe, hoặc mọi xe đã chốt / đủ lượt đọc).
    # Trên 4 video cầu kết quả giống hệt khi tắt (88/130 biển đúng, cùng chuỗi rác).
    skip_idle: bool = True


def roi_pixels(shape, roi):
    """roi = (x1, y1, x2, y2) theo tỷ lệ khung hình -> pixel. None = cả khung hình."""
    h, w = shape[:2]
    if roi is None:
        return 0, 0, w, h
    return round(roi[0] * w), round(roi[1] * h), round(roi[2] * w), round(roi[3] * h)


def associate_plates(plate_boxes, tracks, expand=0.0):
    """Biển thuộc về xe có box chứa tâm biển; nhiều xe chứa thì chọn box nhỏ nhất."""
    return [owner_of(b, tracks, expand) for b in plate_boxes]


def resolve_type(vote, best_plate, min_reads=2):
    """Loại xe = vote detector, trừ khi biển đọc được >= min_reads lần là biển xe máy.
    Detector hay nhầm xe máy chở hàng/cận cảnh thành car/truck; định dạng biển thì không nhầm."""
    if best_plate and best_plate["count"] >= min_reads and plate_vehicle_class(best_plate["plate"]):
        label = plate_vehicle_class(best_plate["plate"])
        return vote if vote[0] == label else (label, None)
    return vote


def sharp_weight(item):
    sharp = item.get("sharp")
    return 1.0 if sharp is None else sharp / (sharp + SHARP_REF)


def position_vote(items):
    """Bỏ phiếu từng vị trí ký tự trên các lần đọc của 1 xe (báo cáo mục 5: per-position voting weighted by
    confidence and sharpness). Độ dài chuỗi lấy theo đa số có trọng số. Trả về (chuỗi, mức đồng thuận 0-1)."""
    lengths = Counter()
    for it in items:
        lengths[len(it["plate"])] += sharp_weight(it)
    n = lengths.most_common(1)[0][0]
    scores = [Counter() for _ in range(n)]
    for it in items:
        if len(it["plate"]) != n:
            continue
        probs = it.get("char_probs") or [it.get("ocr_conf", 1.0)] * n
        for i, ch in enumerate(it["plate"]):
            scores[i][ch] += probs[i] * sharp_weight(it)
    plate = "".join(sc.most_common(1)[0][0] for sc in scores)
    agree = sum(sc.most_common(1)[0][1] / sum(sc.values()) for sc in scores) / n
    return plate, agree


def best_of(items, weighted=False, by_position=False, min_conf=0.0):
    """Biển chốt của 1 xe. by_position: bỏ phiếu từng vị trí ký tự (nếu ra chuỗi đúng cú pháp); không thì biển đa
    số (weighted: mỗi phiếu nặng bằng bề rộng ảnh biển). Trong biển đó lấy lần đọc có trust cao nhất.
    None nếu không lần đọc nào có độ tin OCR >= min_conf."""
    if not items or max(it.get("ocr_conf", 1.0) for it in items) < min_conf:
        return None
    if by_position and len(items) > 1:
        plate, agree = position_vote(items)
        if valid_plate(plate):
            same = [it for it in items if it["plate"] == plate]
            # Chuỗi ghép từ nhiều lần đọc có thể không trùng lần đọc nào: lấy ảnh của lần đọc gần nhất
            top = max(same, key=lambda it: it["trust_score"]) if same else min(
                items, key=lambda it: (n_changes(it["plate"], plate), -it["trust_score"]))
            return {**top, "plate": plate, "count": len(same), "total": len(items),
                    "consensus": round(agree * 100, 1)}
    weight = (lambda it: it.get("wh", (1, 1))[0]) if weighted else (lambda it: 1)
    votes = Counter()
    for it in items:
        votes[it["plate"]] += weight(it)
    plate, w = votes.most_common(1)[0]
    count = sum(it["plate"] == plate for it in items)
    top = max((it for it in items if it["plate"] == plate), key=lambda it: it["trust_score"])
    return {**top, "count": count, "total": len(items), "consensus": round(w / sum(votes.values()) * 100, 1)}


class PlateVoter:
    """Majority voting biển số theo ID xe: tối đa max_reads lần đọc, khóa khi lock_count lần khớp."""

    def __init__(self, max_reads=5, lock_count=3, by_size=False, by_position=False, min_conf=0.0):
        self.max_reads = max_reads
        self.lock_count = lock_count
        self.by_size = by_size  # True: không chốt; đầy thì thay lần đọc có ảnh biển nhỏ nhất bằng ảnh to hơn
        self.by_position = by_position
        self.min_conf = min_conf
        self.reads = defaultdict(list)
        self.locked = set()

    def needs_read(self, vid, width=None):
        reads = self.reads.get(vid, [])
        if self.by_size:
            return len(reads) < self.max_reads or width is None or width > min(it["wh"][0] for it in reads)
        return vid not in self.locked and len(reads) < self.max_reads

    def add(self, vid, item):
        self.reads[vid].append(item)
        if self.by_size:
            if len(self.reads[vid]) > self.max_reads:
                self.reads[vid].remove(min(self.reads[vid], key=lambda it: it["wh"][0]))
            return
        _, cnt = Counter(it["plate"] for it in self.reads[vid]).most_common(1)[0]
        if cnt >= self.lock_count:
            self.locked.add(vid)

    def best(self, vid):
        return best_of(self.reads.get(vid), self.by_size, self.by_position, self.min_conf)

    def ids(self):
        return list(self.reads)


@dataclass
class FrameResult:
    tracks: list
    types: dict
    plates: dict
    timings: dict = field(default_factory=dict)


class ALPRPipeline:
    def __init__(self, script_dir, roi=None, plate_model="v62", plate_imgsz=640, vehicle_weights=None,
                 fusion=None, log_tracks=False, plate_cfg=None):
        setup_cuda_dlls()
        import torch

        script_dir = Path(script_dir)
        has_cuda = torch.cuda.is_available()
        self.device = 0 if has_cuda else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
        self.roi = roi
        self.tracker = VehicleTracker(vehicle_weights or script_dir / "weights" / "vehicle_det.pt", self.device)
        self.plate_model = load_plate_model(script_dir, has_cuda, plate_model)
        self.plate_imgsz = plate_imgsz
        self.ocr = load_ocr(has_cuda)
        self.plate_cfg = plate_cfg or PlateConfig()
        self.fusion = fusion or FusionConfig()
        self.log_tracks = log_tracks
        self._reset_state()

    def _reset_state(self):
        cfg = self.plate_cfg
        self.type_voter = TypeVoter()
        self.plate_voter = (PlateVoter(max_reads=cfg.max_reads, by_size=True, by_position=cfg.position_vote,
                                       min_conf=cfg.min_best_conf) if cfg.size_vote
                            else PlateVoter(by_position=cfg.position_vote, min_conf=cfg.min_best_conf))
        self.buffer = PlateBuffer(self.fusion.buffer_frames, self.fusion.expand_pct)
        self.untracked = []
        self.first_seen = {}
        self.aspects = defaultdict(list)  # track id -> w/h các box không chạm mép
        self.frame_idx = 0
        self.track_log = [] if self.log_tracks else None

    def reset(self, roi=None):
        """Bắt đầu video mới (hoặc phát lại) mà không nạp lại model: xóa track, phiếu bầu, biển đã đọc."""
        self.roi = roi
        self.tracker.reset()
        self._reset_state()

    def _plate_due(self, tracks):
        """Khung này có cần chạy detector biển không."""
        if not self.plate_cfg.skip_idle:
            return True
        return any(t.id is None or self.plate_voter.needs_read(t.id) for t in tracks)

    def _read_plates(self, frame, timestamp, tracks):
        if not self._plate_due(tracks):
            self._step_buffer(tracks)
            return 0.0
        r = self.plate_model.predict(frame, conf=PLATE_CONF, imgsz=self.plate_imgsz, device=self.device, rect=False,
                                     verbose=False)[0]
        enable_cuda_graphs(self.plate_model)
        boxes, confs = [], []
        if r.boxes is not None:
            for b in r.boxes:
                boxes.append(tuple(map(int, b.xyxy[0])))
                confs.append(float(b.conf[0]))
        owners = associate_plates(boxes, tracks)
        if self.fusion.expand_assoc:
            owners = [o if o is not None else owner_of(b, tracks, self.fusion.expand_pct)
                      for o, b in zip(owners, boxes)]

        cfg = self.plate_cfg
        inputs, metas = [], []
        for box, det_conf, owner in zip(boxes, confs, owners):
            x1, y1, x2, y2 = box
            if owner is not None and not self.plate_voter.needs_read(owner, x2 - x1):
                continue
            if x2 - x1 < MIN_PLATE_W or y2 - y1 < MIN_PLATE_H:
                continue
            crop = get_expanded_crop(frame, x1, y1, x2, y2)
            if crop.size == 0 or crop.shape[0] < MIN_PLATE_H or crop.shape[1] < MIN_PLATE_W:
                continue
            sharp = sharpness(crop)
            if sharp < cfg.sharp_min:
                continue
            inputs.append(rectify(crop)[0] if cfg.rectify else crop)
            metas.append((owner, det_conf, crop, box, sharp))

        t_ocr = time.perf_counter()
        if inputs:
            reads = [self._parse(pred) for pred in self._ocr(inputs)]
            if cfg.enhance:
                self._retry(inputs, metas, reads)
            for read, (owner, det_conf, crop, box, sharp) in zip(reads, metas):
                if read is None:
                    continue
                trust = (0.4 * read["conf"] + 0.6 * det_conf) * max(0.0, 1 - FIX_PENALTY * read["fixes"])
                item = {"plate": read["plate"], "trust_score": round(trust * 100, 1), "ocr_conf": read["conf"],
                        "char_probs": read["probs"], "sharp": round(sharp, 1), "timestamp": timestamp,
                        "crop": crop.copy(), "wh": (box[2] - box[0], box[3] - box[1])}
                if owner is not None:
                    self.plate_voter.add(owner, item)
                elif self.fusion.plate_buffer:
                    self.buffer.add(self.frame_idx, box, item)
                else:
                    self.untracked.append(item)
        ocr_ms = (time.perf_counter() - t_ocr) * 1000
        self._step_buffer(tracks)
        return ocr_ms

    def _step_buffer(self, tracks):
        """Biển chưa có xe chủ nằm trong buffer vài khung, chờ xe xuất hiện quanh nó (fusion plate_buffer)."""
        if not self.fusion.plate_buffer:
            return
        assigned, expired = self.buffer.step(self.frame_idx, tracks)
        for vid, item in assigned:
            self.plate_voter.add(vid, item)
        self.untracked.extend(expired)

    def _ocr(self, images):
        return self.ocr.run([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images], return_confidence=True)

    def _retry(self, inputs, metas, reads):
        """Đọc lại biển OCR không chắc (báo cáo mục 2: low-confidence OCR retries): ảnh CLAHE + unsharp, và với biển
        2 dòng thêm 2 nửa trên/dưới đọc riêng rồi ghép. Giữ ứng viên có điểm cú pháp cao nhất."""
        batch, jobs = [], []
        for i, r in enumerate(reads):
            if r is not None and r["conf"] >= RETRY_CONF:
                continue
            enh = enhance_crop(inputs[i])
            batch.append(enh)
            jobs.append((i, 1))
            x1, y1, x2, y2 = metas[i][3]
            if not is_single_line((x2 - x1, y2 - y1)):
                batch.extend(split_two_line(enh))
                jobs.append((i, 2))
        if not batch:
            return
        preds = iter(self._ocr(batch))
        for i, n in jobs:
            parts = [next(preds) for _ in range(n)]
            cand = self._parse(parts[0]) if n == 1 else self._parse_joined(parts)
            if cand is not None and (reads[i] is None or cand["score"] > reads[i]["score"]):
                reads[i] = cand

    def _parse(self, pred):
        """Kết quả OCR -> {plate, conf, probs, fixes, score} hoặc None nếu không dùng được."""
        return self._parse_joined([pred])

    def _parse_joined(self, preds):
        """Ghép chuỗi và xác suất ký tự của 1 hoặc nhiều kết quả OCR (biển 2 dòng đọc từng nửa)."""
        raw, probs = "", []
        for pred in preds:
            part = getattr(pred, "plate", "").strip()
            p = getattr(pred, "char_probs", None)
            raw += part
            probs += [float(x) for x in p[:len(part)]] if p is not None else [0.85] * len(part)
        conf = sum(probs) / len(probs) if probs else 0.85
        if self.plate_cfg.syntax:
            plate = correct_plate(raw)
            fixes = n_changes(plate, clean_plate(raw)) if plate else 0
        else:
            plate, fixes = normalize_plate(raw), 0
        if plate is None or len(plate) < 5:
            return None
        if len(probs) != len(plate):
            probs = [conf] * len(plate)
        # Điểm cú pháp của ứng viên: độ tin OCR trừ phần ký tự phải sửa
        return {"plate": plate, "conf": conf, "probs": probs, "fixes": fixes, "score": conf - FIX_PENALTY * fixes}

    def process(self, frame, timestamp):
        self.frame_idx += 1
        # Chỉ xử lý vùng ROI (vd. bỏ khung giao diện phần mềm trong video demo)
        x0, y0, x1, y1 = roi_pixels(frame.shape, self.roi)
        frame = frame[y0:y1, x0:x1]

        t0 = time.perf_counter()
        tracks = self.tracker.update(frame)
        t1 = time.perf_counter()
        self.type_voter.process(tracks)
        t2 = time.perf_counter()
        ocr_ms = self._read_plates(frame, timestamp, tracks)
        t3 = time.perf_counter()

        ids = [t.id for t in tracks if t.id is not None]
        for vid in ids:
            self.first_seen.setdefault(vid, timestamp)
        for t in tracks:
            bx1, by1, bx2, by2 = t.box
            if t.id is not None and by2 > by1 and not box_touches_border(t.box, frame.shape):
                self.aspects[t.id].append((bx2 - bx1) / (by2 - by1))
        full = [Track(t.id, (t.box[0] + x0, t.box[1] + y0, t.box[2] + x0, t.box[3] + y0), t.coarse, t.conf)
                for t in tracks]
        if self.track_log is not None:
            self.track_log.append({"frame": self.frame_idx,
                                   "tracks": [[t.id, *t.box, t.coarse, round(t.conf, 3)] for t in full]})
        return FrameResult(
            tracks=full,
            types={vid: self.vehicle_type(vid) for vid in ids},
            plates={vid: self.plate_voter.best(vid) for vid in ids},
            timings={"vehicle_ms": (t1 - t0) * 1000, "type_ms": (t2 - t1) * 1000,
                     "plate_ms": (t3 - t2) * 1000 - ocr_ms, "ocr_ms": ocr_ms},
        )

    def _reads_of(self, vids):
        return [it for v in vids for it in self.plate_voter.reads.get(v, [])]

    def _type_of(self, vids):
        votes = sum((self.type_voter.votes.get(v, Counter()) for v in vids), Counter())
        if votes:
            label, weight = votes.most_common(1)[0]
            vote = (label, weight / sum(votes.values()))
        else:
            vote = ("unknown", None)
        reads = self._reads_of(vids)
        if self.fusion.aspect_rule:
            vote = aspect_type(vote, [a for v in vids for a in self.aspects.get(v, [])])
        if self.fusion.single_line_rule:
            vote = single_line_type(vote, votes, reads)
        return resolve_type(vote, self._best(reads))

    def _best(self, reads):
        c = self.plate_cfg
        return best_of(reads, c.size_vote, c.position_vote, c.min_best_conf)

    def vehicle_type(self, vid):
        return self._type_of([vid])

    def finalize(self):
        if self.fusion.plate_buffer:
            self.untracked.extend(self.buffer.flush())
        ids = sorted(set(self.type_voter.seen_ids()) | set(self.plate_voter.ids()))
        if self.fusion.merge_by_plate:
            entries = [(v, (self.plate_voter.best(v) or {}).get("plate", ""), self.first_seen.get(v)) for v in ids]
            groups = group_ids_by_plate(entries, self.fusion.merge_gap_s)
        else:
            groups = [[v] for v in ids]
        records = []
        for vids in groups:
            vtype, type_conf = self._type_of(vids)
            best = self._best(self._reads_of(vids))
            seen = [self.first_seen[v] for v in vids if v in self.first_seen]
            records.append({
                "vehicle_id": vids[0], "vehicle_ids": vids, "type": vtype, "type_conf": type_conf,
                "plate": best["plate"] if best else "",
                "consensus": best["consensus"] if best else None,
                "trust_score": best["trust_score"] if best else None,
                "first_seen": min(seen) if seen else None,
                "crop": best["crop"] if best else None,
            })
        # Biển không nằm trong xe nào: giữ như code cũ (>=2 lần đọc hoặc trust >= 75)
        groups_u = defaultdict(list)
        for it in self.untracked:
            groups_u[it["plate"]].append(it)
        known = {r["plate"] for r in records}
        for plate, items in groups_u.items():
            top = max(items, key=lambda it: it["trust_score"])
            if max(it.get("ocr_conf", 1.0) for it in items) < self.plate_cfg.min_best_conf:
                continue
            if plate not in known and (len(items) >= 2 or top["trust_score"] >= 75.0):
                records.append({"vehicle_id": None, "vehicle_ids": [], "type": "unknown", "type_conf": None,
                                "plate": plate, "consensus": None, "trust_score": top["trust_score"],
                                "first_seen": top["timestamp"], "crop": top["crop"]})
        return records
