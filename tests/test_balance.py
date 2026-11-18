import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_tools.balance import repeat_counts, repeat_factors


def test_rare_class_repeated_common_class_not():
    # lớp 0 có ở mọi ảnh, lớp 2 chỉ ở 1/10 ảnh
    classes = [{0}] * 9 + [{0, 2}]
    r_class, r_image = repeat_factors(classes, t=0.4)
    assert r_class[0] == 1.0
    assert r_class[2] == 2.0  # sqrt(0.4 / 0.1)
    assert r_image[:9] == [1.0] * 9 and r_image[9] == 2.0


def test_image_without_labels_kept_once():
    _, r_image = repeat_factors([set(), {1}], t=0.3)
    assert r_image[0] == 1.0


def test_repeat_counts_round_fraction_by_probability():
    counts = repeat_counts([1.5] * 2000, seed=0)
    assert set(counts) == {1, 2}
    assert abs(sum(counts) / 2000 - 1.5) < 0.05
    assert repeat_counts([3.0, 1.0]) == [3, 1]
