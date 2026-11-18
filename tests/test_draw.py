import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from alpr.draw import annotate
from alpr.pipeline import FrameResult
from alpr.vehicle_module import Track


def test_annotate_crops_roi_and_draws_box():
    frame = np.zeros((100, 200, 3), np.uint8)
    res = FrameResult(tracks=[Track(1, (20, 60, 80, 95), "car", 0.9)], types={1: ("car", None)},
                      plates={1: {"plate": "51A12345"}})
    out = annotate(frame, res, roi=(0.0, 0.0, 0.5, 1.0))
    assert out.shape == (100, 100, 3)
    assert out[60, 20].any() and not frame.any()


def test_annotate_hud_draws_top_bar():
    frame = np.zeros((100, 200, 3), np.uint8)
    out = annotate(frame, FrameResult(tracks=[], types={}, plates={}), roi=None, hud="FPS 50")
    assert (out[5, 5] == (15, 15, 20)).all()
