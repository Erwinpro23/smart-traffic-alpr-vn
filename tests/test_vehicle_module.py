import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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


from alpr.vehicle_module import COARSE_CLASSES, Track, TypeVoter


def test_coarse_classes_order():
    assert COARSE_CLASSES == ["motorcycle", "car", "bus", "truck"]


def test_unknown_before_any_frame():
    assert TypeVoter().get(1) == ("unknown", None)


def test_confidence_weighted_vote_beats_frame_count():
    voter = TypeVoter()
    box = (0, 0, 10, 10)
    # 2 frame "car" điểm thấp vs 1 frame "motorcycle" điểm cao
    voter.process([Track(5, box, "car", 0.2)])
    voter.process([Track(5, box, "car", 0.2)])
    voter.process([Track(5, box, "motorcycle", 0.9)])
    label, share = voter.get(5)
    assert label == "motorcycle"
    assert share == pytest.approx(0.9 / 1.3)


def test_untracked_detection_is_ignored():
    voter = TypeVoter()
    voter.process([Track(None, (0, 0, 10, 10), "car", 0.9)])
    assert voter.seen_ids() == []


def test_tracks_are_independent():
    voter = TypeVoter()
    voter.process([Track(1, (0, 0, 10, 10), "bus", 0.8), Track(2, (20, 0, 30, 10), "truck", 0.7)])
    assert voter.get(1) == ("bus", 1.0)
    assert voter.get(2) == ("truck", 1.0)
    assert sorted(voter.seen_ids()) == [1, 2]
