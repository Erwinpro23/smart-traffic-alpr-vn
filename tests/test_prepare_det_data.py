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


from data_tools.prepare_det_data import find_yolo_pairs, group_split, phash_groups, remap_label_lines

NAMES = ["bus", "car", "motor", "person"]
MAP = {"bus": "bus", "car": "car", "motor": "motorcycle"}


def test_remap_boxes_and_drop_unmapped():
    lines = ["2 0.5 0.5 0.1 0.2", "3 0.1 0.1 0.05 0.05", "0 0.4 0.4 0.3 0.3"]
    assert remap_label_lines(lines, NAMES, MAP) == ["0 0.500000 0.500000 0.100000 0.200000",
                                                     "2 0.400000 0.400000 0.300000 0.300000"]


def test_polygon_converted_to_box():
    out = remap_label_lines(["1 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6"], NAMES, MAP)
    assert out == ["1 0.200000 0.400000 0.200000 0.400000"]


def test_malformed_lines_skipped():
    assert remap_label_lines(["", "1 0.5", "9 0.5 0.5 0.1 0.1"], NAMES, MAP) == []


def test_find_yolo_pairs_both_layouts(tmp_path):
    for p in ("a/images/train/x.jpg", "a/labels/train/x.txt", "b/y.jpg", "b/y.txt", "b/z.jpg"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text("")
    pairs = [(i.relative_to(tmp_path).as_posix(), l.relative_to(tmp_path).as_posix())
             for i, l in find_yolo_pairs(tmp_path)]
    assert pairs == [("a/images/train/x.jpg", "a/labels/train/x.txt"), ("b/y.jpg", "b/y.txt")]


def test_phash_groups_links_near_duplicates_transitively():
    a = 0b0
    b = 0b111          # dist 3 from a
    c = 0b111111       # dist 3 from b, 6 from a
    d = (1 << 63) | (0xFFFF << 20)  # far from all
    g = phash_groups([a, b, c, d], max_dist=5)
    assert g[0] == g[1] == g[2]
    assert g[3] != g[0]


def test_group_split_keeps_groups_together_and_ratios():
    groups = [i // 2 for i in range(200)]  # 100 groups of 2
    splits = group_split(groups, seed=0)
    for i in range(0, 200, 2):
        assert splits[i] == splits[i + 1]
    assert splits.count("train") == pytest.approx(160, abs=10)
    assert set(splits) == {"train", "val", "test"}
    assert group_split(groups, seed=0) == splits  # deterministic


def test_group_split_balances_images_with_uneven_groups():
    groups = [0] * 300 + list(range(1, 701))  # one big group (300 imgs) + 700 singletons
    splits = group_split(groups, seed=0)
    assert len({splits[i] for i in range(300)}) == 1
    assert splits.count("train") == pytest.approx(800, abs=5)
    assert splits.count("val") == pytest.approx(100, abs=5)
    assert splits.count("test") == pytest.approx(100, abs=5)
