"""Kiểm tra và sửa chuỗi biển số theo cú pháp biển Việt Nam (mục hậu xử lý trong báo cáo).

Dạng chuỗi sau khi bỏ dấu: 2 số mã tỉnh + sê-ri (1 chữ, có thể thêm 1 chữ hoặc 1 số) + 4-5 số.
  59X321224 (xe máy 9 ký tự), 59AA12345, 51G30069 (ô tô), 52Z0914 (ô tô cũ 4 số), 70L91791 (xe máy cũ).
Sửa theo vị trí: chỗ phải là số mà đọc ra chữ (O->0, B->8...) và ngược lại ở chữ sê-ri đầu (8->B, 0->D...).
"""
import re

VN_PROVINCE_CODES = frozenset("""
11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41
43 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75
76 77 78 79 80 81 82 83 84 85 86 88 89 90 92 93 94 95 97 98 99""".split())

TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "1",
                          "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"})
TO_LETTER = str.maketrans({"0": "D", "1": "T", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"})
# Biển VN không dùng I, O, Q, W làm chữ sê-ri; OCR hay đọc nhầm T thành I
SERIES_FIX = str.maketrans({"I": "T", "O": "D", "Q": "D"})

MAX_FIXES = 2
PLATE_RE = re.compile(r"(\d\d)([A-Z])([A-Z0-9]?)(\d{4,5})")


def clean(text):
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def is_valid(plate):
    m = PLATE_RE.fullmatch(plate or "")
    return bool(m) and m.group(1) in VN_PROVINCE_CODES and 7 <= len(plate) <= 9


def correct(raw):
    """Sửa chuỗi OCR theo vị trí; trả về biển hợp lệ hoặc None nếu không đưa được về dạng biển VN."""
    t = clean(raw)
    if not 7 <= len(t) <= 9:
        return None
    head = t[:2].translate(TO_DIGIT) + t[2].translate(TO_LETTER).translate(SERIES_FIX)
    rest = t[3:]
    # Ký tự 4 có thể là chữ (59AA..) hoặc số (59X3..). Phần cuối luôn là 4-5 số.
    candidates = []
    for n_tail in (5, 4):
        mid = rest[:-n_tail] if len(rest) > n_tail else ""
        if len(rest) - len(mid) != n_tail or len(mid) > 1:
            continue
        tail = rest[-n_tail:].translate(TO_DIGIT)
        mid = mid.translate(SERIES_FIX) if mid.isalpha() else mid
        cand = head + mid + tail
        if is_valid(cand):
            candidates.append(cand)
    # Ưu tiên cách sửa ít ký tự nhất; sửa quá MAX_FIXES ký tự thì coi là rác (vd. lan can đọc ra 11TTTII)
    best = min(candidates, key=lambda c: n_changes(c, t), default=None)
    return best if best is not None and n_changes(best, t) <= MAX_FIXES else None


def n_changes(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
