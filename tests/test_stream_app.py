import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from alpr.pipeline import FrameResult
from scripts.stream_app import DISPLAY_W, FrameReader, to_display
from alpr.vehicle_module import Track


def test_to_display_scales_frame_and_boxes():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    res = FrameResult(tracks=[Track(1, (300, 600, 900, 1500), "car", 0.9)], types={}, plates={})
    small, res2 = to_display(frame, res)
    assert small.shape[1] == DISPLAY_W and small.shape[0] == 720
    assert res2.tracks[0].box == (100, 200, 300, 500)
    assert res.tracks[0].box == (300, 600, 900, 1500)  # bản gốc không đổi


def test_to_display_keeps_small_frames():
    frame = np.zeros((720, 1280, 3), np.uint8)
    res = FrameResult(tracks=[], types={}, plates={})
    assert to_display(frame, res)[0] is frame


def test_frame_reader_reads_all_frames_then_stops(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(12):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    r = FrameReader(path, maxsize=2)
    n = 0
    while r.read()[0]:
        n += 1
    r.release()
    assert n == 12


def test_frame_reader_release_while_queue_full(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for _ in range(20):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    r = FrameReader(path, maxsize=2)
    r.read()
    r.release()  # không được treo
    assert not r.thread.is_alive()
