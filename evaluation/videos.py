import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Các video dùng để chấm điểm: đường dẫn, ROI và thư mục chứa nháp, nhãn và GT của từng video."""
from pathlib import Path

from alpr.alpr_common import DEMO_ROI, DEMO_VIDEO, SCRIPT_DIR

EVAL_DIR = Path(__file__).resolve().parent

VIDEOS = {
    # video cổng (camera gần), GT gốc nằm thẳng trong evaluation/
    "demo": {"path": DEMO_VIDEO, "roi": DEMO_ROI, "dir": EVAL_DIR, "gt": "demo_gt.csv"},
    # video đường phố quay từ phía sau (nhóm gửi), file video không đưa vào git
    "street": {"path": SCRIPT_DIR / "demo_videos" / "8785301236677161723.mp4", "roi": None,
               "dir": EVAL_DIR / "street", "gt": "gt.csv"},
}


def gt_path(video, suffix=""):
    """suffix='reviewed' -> <gt>_reviewed.csv (GT đã được người duyệt, xem evaluation/merge_review.py)."""
    v = VIDEOS[video]
    if not suffix:
        return v["dir"] / v["gt"]
    stem, ext = v["gt"].rsplit(".", 1)
    return v["dir"] / f"{stem}_{suffix}.{ext}"
