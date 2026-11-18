import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.scoring import iou, load_gt, match_gt, score


def gt(gid, frame, box, typ, plate=""):
    return {"gt_id": gid, "frame": frame, "box": box, "type": typ, "plate": plate}


def rec(ids, typ, plate=""):
    return {"vehicle_ids": ids, "type": typ, "plate": plate}


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_match_gt_picks_best_iou_above_threshold():
    log = [{"frame": 5, "tracks": [[1, 0, 0, 10, 10, "car", 0.9], [2, 1, 0, 11, 10, "car", 0.9],
                                   [None, 0, 0, 10, 10, "car", 0.9]]}]
    rows = [gt("a", 5, (1, 0, 11, 10), "car"), gt("b", 5, (50, 50, 60, 60), "car"), gt("c", 6, (0, 0, 10, 10), "car")]
    assert match_gt(rows, log) == [2, None, None]


def test_score_counts_type_plate_duplicates_and_false_records():
    rows = [gt("A", 1, (0, 0, 1, 1), "motorcycle", "59S290774"),
            gt("A", 9, (0, 0, 1, 1), "motorcycle", "59S290774"),   # cùng xe, track khác (đổi ID)
            gt("B", 2, (0, 0, 1, 1), "car", "51A07204"),
            gt("C", 3, (0, 0, 1, 1), "truck"),                      # không đọc được biển
            gt("D", 4, (0, 0, 1, 1), "car"),                        # bị bỏ sót
            gt("X", 5, (0, 0, 1, 1), "not_vehicle")]
    matches = [10, 11, 20, 30, None, 40]
    records = [rec([10], "motorcycle", "59S290774"), rec([11], "car", "59S290774"),
               rec([20], "car", "51A07209"), rec([30], "truck"), rec([40], "car", "99X99999"),
               rec([50], "car", "12A34567"), rec([], "unknown", "59S290774"), rec([], "unknown", "77B11111")]
    s = score(rows, matches, records)
    assert s["n_vehicles"] == 4 and s["n_plates"] == 2
    assert s["type_acc"] == 0.75          # A, B, C đúng; D bỏ sót
    assert s["plate_acc"] == 0.5          # A đúng, B sai 1 ký tự
    assert s["duplicates"] == 2           # A có 2 record + 1 untracked trùng biển GT
    assert s["false_records"] == 3        # record của X, record 50 không khớp GT, untracked 77B11111


def test_load_gt_parses_types(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("gt_id,frame,x1,y1,x2,y2,type,plate\nA,12,1,2,3,4,car,51A07204\n", encoding="utf-8")
    assert load_gt(p) == [gt("A", 12, (1, 2, 3, 4), "car", "51A07204")]


def test_gt_path_suffix():
    from evaluation.videos import gt_path
    assert gt_path("demo").name == "demo_gt.csv"
    assert gt_path("demo", "reviewed").name == "demo_gt_reviewed.csv"
    assert gt_path("street", "reviewed").name == "gt_reviewed.csv"
