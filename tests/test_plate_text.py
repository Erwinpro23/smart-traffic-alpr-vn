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


from alpr.plate_text import correct, is_valid


@pytest.mark.parametrize("plate", ["59X321224", "59AA12345", "51G30069", "52Z0914", "70L91791", "50H54223", "59R218328"])
def test_valid_plates_kept(plate):
    assert is_valid(plate)
    assert correct(plate) == plate


@pytest.mark.parametrize("raw,want", [
    ("68I173125", "68T173125"),   # I không có trong sê-ri -> T
    ("5OAE37O51", "50AE37051"),   # O ở chỗ số -> 0
    ("59-S3 510.38", "59S351038"),  # bỏ dấu
    ("598321224", "59B321224"),   # chữ sê-ri đọc thành số
])
def test_corrections(raw, want):
    assert correct(raw) == want


@pytest.mark.parametrize("raw", ["0UANSAT", "77TTTT", "531111", "11TTTII", "CHUYQUANST", "45L11277", ""])
def test_garbage_rejected(raw):
    assert correct(raw) is None


def test_unknown_province_rejected():
    assert not is_valid("42A12345")
