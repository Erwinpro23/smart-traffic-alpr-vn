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


from evaluation.merge_review import merge

GT = [{"gt_id": "2", "frame": 36, "box": (1, 2, 3, 4), "type": "motorcycle", "plate": "50H197901"},
      {"gt_id": "2", "frame": 40, "box": (1, 2, 3, 4), "type": "motorcycle", "plate": "50H197901"},
      {"gt_id": "3", "frame": 68, "box": (5, 6, 7, 8), "type": "truck", "plate": ""}]


def test_blank_human_fields_keep_ai_label():
    rows, nt, np_ = merge(GT, [{"gt_id": "2", "human_type": "", "human_plate": ""}])
    assert rows == GT and (nt, np_) == (0, 0)


def test_human_type_overrides_all_rows_of_vehicle():
    rows, nt, np_ = merge(GT, [{"gt_id": "2", "human_type": "car", "human_plate": ""}])
    assert [r["type"] for r in rows] == ["car", "car", "truck"] and (nt, np_) == (1, 0)


def test_plate_none_clears_plate():
    rows, _, np_ = merge(GT, [{"gt_id": "2", "human_type": "", "human_plate": "none"}])
    assert rows[0]["plate"] == "" and np_ == 1


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        merge(GT, [{"gt_id": "3", "human_type": "lorry", "human_plate": ""}])
