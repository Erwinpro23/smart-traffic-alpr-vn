import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collections import Counter

import numpy as np

from alpr.fusion import FusionConfig, PlateBuffer
from alpr.pipeline import (ALPRPipeline, PlateConfig, PlateVoter, associate_plates, best_of, position_vote, resolve_type,
                      roi_pixels)
from alpr.vehicle_module import Track, TypeVoter


def item(plate, trust=80.0, ts=1.0):
    return {"plate": plate, "trust_score": trust, "timestamp": ts, "crop": np.zeros((2, 2, 3), np.uint8)}


def test_plate_assigned_to_containing_vehicle():
    tracks = [Track(1, (0, 0, 100, 100), "car", 0.9), Track(2, (200, 0, 300, 100), "car", 0.9)]
    assert associate_plates([(210, 50, 250, 70), (40, 40, 60, 60)], tracks) == [2, 1]


def test_nested_boxes_pick_smallest_vehicle():
    truck = Track(1, (0, 0, 500, 500), "truck", 0.9)
    moto = Track(2, (100, 100, 200, 300), "motorcycle", 0.9)
    assert associate_plates([(140, 250, 160, 270)], [truck, moto]) == [2]


def test_plate_outside_or_untracked_vehicle_is_none():
    tracks = [Track(None, (0, 0, 100, 100), "car", 0.9)]
    assert associate_plates([(40, 40, 60, 60), (500, 500, 520, 510)], tracks) == [None, None]


def test_plate_voter_locks_after_three_identical_reads():
    v = PlateVoter(max_reads=5, lock_count=3)
    for p in ("51G67146", "51G67148", "51G67146", "51G67146"):
        assert v.needs_read(9)
        v.add(9, item(p))
    assert not v.needs_read(9)
    best = v.best(9)
    assert best["plate"] == "51G67146"
    assert (best["count"], best["total"]) == (3, 4)
    assert best["consensus"] == 75.0


def test_plate_voter_stops_at_max_reads_and_picks_best_trust():
    v = PlateVoter(max_reads=2, lock_count=3)
    v.add(4, item("AAA111", trust=60.0, ts=1.0))
    v.add(4, item("AAA111", trust=90.0, ts=2.0))
    assert not v.needs_read(4)
    assert v.best(4)["trust_score"] == 90.0
    assert v.best(4)["timestamp"] == 2.0


def test_roi_pixels():
    assert roi_pixels((864, 1920, 3), None) == (0, 0, 1920, 864)
    assert roi_pixels((864, 1920, 3), (0.0, 0.0, 0.83, 1.0)) == (0, 0, 1594, 864)


def test_plate_voter_unknown_id():
    v = PlateVoter()
    assert v.best(123) is None
    assert v.ids() == []


def test_resolve_type_motorcycle_plate_overrides_detector_vote():
    best = {"plate": "59G307326", "count": 2}
    assert resolve_type(("car", 0.9), best) == ("motorcycle", None)
    # khớp sẵn với vote -> giữ tỷ lệ phiếu
    assert resolve_type(("motorcycle", 0.8), best) == ("motorcycle", 0.8)


def test_resolve_type_keeps_vote_without_reliable_plate():
    assert resolve_type(("car", 0.9), None) == ("car", 0.9)
    # chỉ 1 lần đọc -> có thể là OCR lỗi, không ghi đè
    assert resolve_type(("car", 0.9), {"plate": "59G307326", "count": 1}) == ("car", 0.9)
    assert resolve_type(("car", 0.9), {"plate": "51A07204", "count": 3}) == ("car", 0.9)


def bare_pipeline(fusion):
    """ALPRPipeline không nạp model: chỉ phần voting/finalize."""
    p = ALPRPipeline.__new__(ALPRPipeline)
    p.type_voter, p.plate_voter = TypeVoter(), PlateVoter()
    p.fusion, p.buffer = fusion, PlateBuffer()
    p.plate_cfg = PlateConfig(size_vote=False)
    p.untracked, p.first_seen, p.aspects = [], {}, {}
    return p


def test_best_of_majority_then_trust():
    items = [item("A", 70), item("B", 99), item("A", 80)]
    best = best_of(items)
    assert best["plate"] == "A" and best["trust_score"] == 80 and best["count"] == 2 and best["consensus"] == 66.7
    assert best_of([]) is None


def test_finalize_merges_tracks_with_same_plate_when_enabled():
    for merge, expected in ((False, [[1], [2]]), (True, [[1, 2]])):
        p = bare_pipeline(FusionConfig(merge_by_plate=merge))
        p.type_voter.votes[1] = Counter({"car": 2.0})
        p.type_voter.votes[2] = Counter({"motorcycle": 3.0})
        p.first_seen = {1: 10.0, 2: 11.0}
        p.plate_voter.add(1, item("59S290774"))
        p.plate_voter.add(2, item("59S290774"))
        recs = p.finalize()
        assert [r["vehicle_ids"] for r in recs] == expected
    assert recs[0]["type"] == "motorcycle" and recs[0]["first_seen"] == 10.0 and recs[0]["vehicle_id"] == 1


def test_single_line_rule_only_when_enabled():
    for rule, expected in ((False, "motorcycle"), (True, "car")):
        p = bare_pipeline(FusionConfig(single_line_rule=rule))
        p.type_voter.votes[1] = Counter({"motorcycle": 3.0, "car": 1.0})
        for _ in range(2):
            p.plate_voter.add(1, {**item("51A07204"), "wh": (120, 30)})
        assert p.vehicle_type(1)[0] == expected


def test_aspect_rule_relabels_tall_car_track_only_when_enabled():
    for on, expected in ((False, "car"), (True, "motorcycle")):
        p = bare_pipeline(FusionConfig(aspect_rule=on))
        p.type_voter.process([Track(7, (0, 0, 50, 100), "car", 0.9)])
        p.aspects = {7: [0.5] * 6}
        assert p.vehicle_type(7)[0] == expected


def _read(plate, w, trust=50.0):
    return {"plate": plate, "trust_score": trust, "wh": (w, w // 2)}


def test_size_voter_keeps_largest_reads_and_never_locks():
    v = PlateVoter(max_reads=3, by_size=True)
    for w in (40, 45, 50):
        v.add(1, _read("59X300000", w))  # đọc sai lúc xe còn xa (ảnh nhỏ)
    assert v.needs_read(1, 60) and not v.needs_read(1, 30)
    for w in (90, 95):
        v.add(1, _read("59X321224", w))  # lại gần: ảnh to, đọc đúng
    assert sorted(it["wh"][0] for it in v.reads[1]) == [50, 90, 95]
    assert v.best(1)["plate"] == "59X321224"


def test_best_of_weighted_by_width():
    items = [_read("A", 40), _read("A", 40), _read("B", 120)]
    assert best_of(items)["plate"] == "A"
    assert best_of(items, weighted=True)["plate"] == "B"


def read(plate, probs, sharp=100.0, trust=80.0):
    return {"plate": plate, "char_probs": probs, "sharp": sharp, "trust_score": trust}


def test_position_vote_combines_characters_across_reads():
    # 2 lần đọc mỗi lần sai 1 ký tự khác nhau, ký tự sai có xác suất thấp -> ghép ra chuỗi đúng
    a = read("59X321224", [.99] * 8 + [.30])        # ký tự cuối sai
    b = read("59X321220", [.99] * 8 + [.20])
    c = read("50X321224", [.99, .40] + [.99] * 7)  # mã tỉnh sai
    plate, agree = position_vote([a, b, c])
    assert plate == "59X321224" and 0.5 < agree < 1
    best = best_of([a, b, c], by_position=True)
    assert best["plate"] == "59X321224" and best["count"] == 1


def test_position_vote_weights_sharp_reads_more():
    blurry = [read("51G30069", [.9] * 8, sharp=5), read("51G30069", [.9] * 8, sharp=5)]
    sharp = read("51G38069", [.9] * 8, sharp=900)
    assert position_vote(blurry + [sharp])[0] == "51G38069"


def test_best_of_position_vote_falls_back_to_majority_when_invalid():
    items = [read("AB", [.9, .9]), read("AB", [.9, .9]), read("XY", [.99, .99])]
    assert best_of(items, by_position=True)["plate"] == "AB"


class FakePred:
    def __init__(self, plate, p=0.95):
        self.plate, self.char_probs = plate, np.full(10, p)


def test_retry_picks_candidate_with_best_syntax_score():
    p = bare_pipeline(FusionConfig())
    calls = []

    def fake_ocr(images):
        calls.append(len(images))
        # ảnh làm nét đọc ra chuỗi lỗi, 2 nửa biển 2 dòng đọc ra "59X3" + "21224"
        return [FakePred("5"), FakePred("59X3"), FakePred("21224")]

    p._ocr = fake_ocr
    img = np.zeros((60, 80, 3), np.uint8)
    metas = [(1, 0.9, img, (0, 0, 80, 60), 50.0)]  # box 80x60 -> biển 2 dòng
    reads = [p._parse(FakePred("59X32I22", 0.5))]
    p._retry([img], metas, reads)
    assert calls == [3]
    assert reads[0]["plate"] == "59X321224" and reads[0]["fixes"] == 0


def test_best_of_rejects_vehicle_without_confident_read():
    unsure = [{"plate": "11T1111", "ocr_conf": 0.55, "trust_score": 40.0},
              {"plate": "11T1111", "ocr_conf": 0.62, "trust_score": 45.0}]
    assert best_of(unsure, min_conf=0.85) is None
    sure = unsure + [{"plate": "11T1111", "ocr_conf": 0.9, "trust_score": 70.0}]
    assert best_of(sure, min_conf=0.85)["plate"] == "11T1111"
    v = PlateVoter(min_conf=0.85)
    for it in unsure:
        v.add(1, it)
    assert v.best(1) is None


def test_plate_due_skips_idle_frames():
    p = bare_pipeline(FusionConfig())
    p.plate_cfg = PlateConfig(skip_idle=True)
    car = Track(1, (0, 0, 10, 10), "car", 0.9)
    assert p._plate_due([car])  # xe 1 chưa có lần đọc nào
    for _ in range(3):
        p.plate_voter.add(1, {"plate": "59X321224", "trust_score": 80.0})
    assert not p._plate_due([car])  # xe 1 đã chốt
    assert not p._plate_due([])  # khung không có xe
    assert p._plate_due([car, Track(None, (20, 0, 30, 10), "car", 0.9)])  # xe chưa có ID: vẫn dò


def test_plate_due_without_skip_runs_every_frame():
    p = bare_pipeline(FusionConfig())
    p.plate_cfg = PlateConfig(skip_idle=False)
    assert p._plate_due([])
