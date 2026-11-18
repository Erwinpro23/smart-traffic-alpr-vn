import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ocr_train.prepare_data import normalize_label, split_by_plate


def test_normalize_label():
    assert normalize_label("60MĐ1 016.57") == "60MD101657"
    assert normalize_label("59-X3 212.24") == "59X321224"
    assert normalize_label("abc") is None          # quá ngắn
    assert normalize_label("12345678901") is None  # quá 10 ô của mô hình


def test_split_by_plate_keeps_each_plate_in_one_part():
    rows = [{"plate_text": f"P{i % 30}"} for i in range(300)]
    parts = split_by_plate(rows, [0.8, 0.1, 0.1])
    sets = [{r["plate_text"] for r in p} for p in parts]
    assert sum(len(p) for p in parts) == 300
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(sets[0]) == 24 and len(sets[1]) == 3 and len(sets[2]) == 3
