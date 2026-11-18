import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.heldout.score import ap50, match_counts


def test_ap50_perfect_and_empty():
    gt = {"a.jpg": [[0, 0, 10, 40]]}
    assert ap50(gt, [("a.jpg", 0.9, [0, 0, 10, 40])]) == 1.0
    assert ap50(gt, []) == 0.0


def test_ap50_false_positive_ranked_first_halves_precision():
    gt = {"a.jpg": [[0, 0, 10, 40]]}
    preds = [("a.jpg", 0.9, [50, 50, 60, 90]), ("a.jpg", 0.8, [0, 0, 10, 40])]
    assert abs(ap50(gt, preds) - 0.5) < 0.02


def test_match_counts_each_gt_once():
    gt = {"a.jpg": [[0, 0, 10, 40]]}
    preds = [("a.jpg", 0.9, [0, 0, 10, 40]), ("a.jpg", 0.8, [0, 0, 10, 41])]
    assert match_counts(gt, preds) == (1, 1, 0)  # tp, fp, fn
