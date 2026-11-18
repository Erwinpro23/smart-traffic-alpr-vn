import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collections import Counter

try:
    import pytest
except ImportError:
    class _Mark:
        @staticmethod
        def parametrize(argnames, argvalues):
            def decorator(func):
                return func
            return decorator
    class _Pytest:
        mark = _Mark()
    pytest = _Pytest()


from alpr.fusion import (FusionConfig, PlateBuffer, aspect_type, box_touches_border, expand_box, group_ids_by_plate,
                    is_single_line, owner_of, plates_similar, single_line_type)
from alpr.vehicle_module import Track


def test_from_names_enables_only_named_rules():
    cfg = FusionConfig.from_names(["merge_by_plate"])
    assert cfg.merge_by_plate and not cfg.expand_assoc and not cfg.plate_buffer and not cfg.single_line_rule


def test_defaults_enable_all_kept_rules_and_empty_names_disable_all():
    cfg = FusionConfig()
    assert cfg.expand_assoc and cfg.plate_buffer and cfg.merge_by_plate and cfg.single_line_rule and cfg.aspect_rule
    off = FusionConfig.from_names([])
    assert not (off.expand_assoc or off.plate_buffer or off.merge_by_plate or off.single_line_rule or off.aspect_rule)
    with pytest.raises(ValueError):
        FusionConfig.from_names(["nope"])


def test_expand_box_grows_20_percent_total():
    assert expand_box((100, 100, 200, 300), 0.20) == (90.0, 80.0, 210.0, 320.0)


def test_owner_of_needs_expansion_for_plate_just_outside():
    tracks = [Track(1, (100, 100, 200, 300), "motorcycle", 0.9)]
    plate = (195, 290, 215, 310)  # tâm (205, 300): ngoài box, trong box nới 20%
    assert owner_of(plate, tracks) is None
    assert owner_of(plate, tracks, expand=0.20) == 1


def test_owner_of_prefers_smallest_and_skips_untracked():
    tracks = [Track(None, (0, 0, 50, 50), "car", 0.9), Track(1, (0, 0, 500, 500), "truck", 0.9),
              Track(2, (0, 0, 100, 100), "motorcycle", 0.9)]
    assert owner_of((20, 20, 30, 30), tracks) == 2


def test_plate_buffer_assigns_when_track_appears_and_expires_otherwise():
    buf = PlateBuffer(max_age=10, expand=0.20)
    buf.add(1, (100, 100, 120, 110), {"plate": "A"})
    buf.add(1, (900, 900, 920, 910), {"plate": "B"})
    assigned, expired = buf.step(5, [Track(7, (80, 80, 200, 200), "motorcycle", 0.5)])
    assert assigned == [(7, {"plate": "A"})] and expired == []
    assigned, expired = buf.step(11, [])
    assert assigned == [] and expired == [{"plate": "B"}]
    assert buf.flush() == []


def test_plates_similar_allows_one_substitution_only():
    assert plates_similar("59S290774", "59S290774")
    assert plates_similar("59S290774", "59S290776")
    assert not plates_similar("59S290774", "59S290766")
    assert not plates_similar("59S29077", "59S290774")
    assert not plates_similar("", "")


def test_group_ids_by_plate_merges_close_in_time_only():
    entries = [(192, "59S290774", 10.0), (214, "59S290774", 11.5), (228, "59S290776", 12.0),
               (300, "59S290774", 40.0), (5, "", 10.0), (6, "", 10.2)]
    assert group_ids_by_plate(entries, max_gap_s=3.0) == [[5], [6], [192, 214, 228], [300]]


def test_is_single_line_threshold():
    assert is_single_line((100, 30))
    assert not is_single_line((60, 45))
    assert not is_single_line((60, 0))


def test_single_line_type_overrides_motorcycle_with_best_other_class():
    votes = Counter({"motorcycle": 5.0, "truck": 2.0, "car": 1.0})
    reads = [{"wh": (120, 30)}, {"wh": (118, 31)}]
    assert single_line_type(("motorcycle", 0.625), votes, reads) == ("truck", None)
    assert single_line_type(("motorcycle", 1.0), Counter({"motorcycle": 3.0}), reads) == ("car", None)


def test_single_line_type_keeps_vote_when_not_enough_evidence():
    votes = Counter({"motorcycle": 5.0, "car": 1.0})
    assert single_line_type(("motorcycle", 0.8), votes, [{"wh": (120, 30)}]) == ("motorcycle", 0.8)
    assert single_line_type(("motorcycle", 0.8), votes, [{"wh": (60, 45)}] * 3) == ("motorcycle", 0.8)
    assert single_line_type(("car", 0.8), votes, [{"wh": (120, 30)}] * 3) == ("car", 0.8)


def test_aspect_type_turns_tall_car_or_bus_into_motorcycle():
    tall = [0.5, 0.55, 0.6, 0.5, 0.45]
    assert aspect_type(("car", 0.8), tall) == ("motorcycle", None)
    assert aspect_type(("bus", 0.7), tall) == ("motorcycle", None)
    assert aspect_type(("truck", 0.9), tall) == ("truck", 0.9)
    assert aspect_type(("car", 0.8), [1.1, 1.2, 1.0, 1.3, 1.1]) == ("car", 0.8)


def test_aspect_type_needs_enough_boxes():
    assert aspect_type(("car", 0.8), [0.5, 0.5]) == ("car", 0.8)
    assert aspect_type(("car", 0.8), []) == ("car", 0.8)


def test_box_touches_border():
    assert box_touches_border((0, 10, 50, 60), (100, 200))
    assert box_touches_border((10, 10, 199, 60), (100, 200))
    assert not box_touches_border((10, 10, 50, 60), (100, 200))
