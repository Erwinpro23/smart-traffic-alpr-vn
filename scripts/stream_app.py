"""Web Dashboard thời gian thực (HTTP Server) hiển thị luồng video, danh sách biển số và FPS.

Chạy: python scripts/stream_app.py [--port 5000]
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import queue
import threading
import time
import traceback
import warnings
from collections import Counter
from dataclasses import replace
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from alpr.alpr_common import DEMO_ROI, DEMO_VIDEO, SCRIPT_DIR
from alpr.draw import annotate, draw_hud_text
from alpr.pipeline import ALPRPipeline
from alpr.vehicle_module import Track

warnings.filterwarnings("ignore")

CROP_DIR = SCRIPT_DIR / "extracted_plates"
CROP_DIR.mkdir(exist_ok=True)
DEMO_1080 = SCRIPT_DIR / "demovid_1080"
DEMO_4K = SCRIPT_DIR / "demovid"
DISPLAY_W = 1280  # bề rộng ảnh gửi lên dashboard
# Dashboard chỉ phát video cổng và 4 video cầu đi bộ nhóm quay: bản 1080p (thu từ 4K gốc) và bản 4K gốc.
# 4K đọc đúng 112/130 biển so với 88/130 ở 1080p (evaluation/plate_reading.md) nhưng chậm hơn.
# Video 3 và 4 dùng bản đã cắt bỏ làn xe máy bên trong dải phân cách (giữ 66% / 72% bề rộng bên trái).
# Video 2 có thêm bản chỉ giữ làn xe máy: x 24-72%, bỏ 30% phía trên (xe ở xa, biển quá nhỏ để đọc).
BRIDGE_VIDEOS = [
    ("vid1", "Cầu đi bộ 1 - 1080p", DEMO_1080 / "IMG_8357.mp4"),
    ("vid2", "Cầu đi bộ 2 - 1080p", DEMO_1080 / "IMG_8355.mp4"),
    ("vid2_lane", "Cầu đi bộ 2 - 1080p (chỉ làn xe máy)", DEMO_1080 / "IMG_8355_crop.mp4"),
    ("vid3", "Cầu đi bộ 3 - 1080p (bỏ làn xe máy)", DEMO_1080 / "IMG_8356_crop.mp4"),
    ("vid4", "Cầu đi bộ 4 - 1080p (bỏ làn xe máy)", DEMO_1080 / "IMG_8358_crop.mp4"),
    ("vid1_4k", "Cầu đi bộ 1 - 4K", DEMO_4K / "IMG_8357.MOV"),
    ("vid2_4k", "Cầu đi bộ 2 - 4K", DEMO_4K / "IMG_8355.MOV"),
    ("vid2_lane_4k", "Cầu đi bộ 2 - 4K (chỉ làn xe máy)", DEMO_4K / "IMG_8355_crop.mp4"),
    ("vid3_4k", "Cầu đi bộ 3 - 4K (bỏ làn xe máy)", DEMO_4K / "IMG_8356_crop.mp4"),
    ("vid4_4k", "Cầu đi bộ 4 - 4K (bỏ làn xe máy)", DEMO_4K / "IMG_8358_crop.mp4"),
]


def list_videos():
    """key -> (nhãn, đường dẫn, ROI). ROI chỉ dùng để cắt khung giao diện của video demo cổng."""
    videos = {"demo": ("Cổng ra vào (video demo)", DEMO_VIDEO, DEMO_ROI)}
    for key, label, path in BRIDGE_VIDEOS:
        if path.exists():
            videos[key] = (label, path, None)
    return videos


VIDEOS = list_videos()



class FrameReader:
    """Giải mã video ở luồng riêng, song song với nhận diện (giải mã HEVC 4K mất ~30 ms/khung trên CPU)."""

    def __init__(self, path, maxsize=4):
        self.cap = cv2.VideoCapture(str(path))
        self.q = queue.Queue(maxsize)
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            self.q.put(frame if ok else None)
            if not ok:
                break

    def read(self):
        frame = self.q.get()
        return frame is not None, frame

    def release(self):
        self.stopped = True
        while self.thread.is_alive():  # luồng có thể đang chờ put vào hàng đợi đầy
            try:
                self.q.get(timeout=0.1)
            except queue.Empty:
                pass
        self.cap.release()


def to_display(frame, res):
    """Thu khung về DISPLAY_W trước khi vẽ để box và chữ giữ cỡ dễ đọc với video 4K; box được đổi tỷ lệ theo."""
    s = DISPLAY_W / frame.shape[1]
    if s >= 1:
        return frame, res
    small = cv2.resize(frame, (DISPLAY_W, round(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
    tracks = [Track(t.id, tuple(int(v * s) for v in t.box), t.coarse, t.conf) for t in res.tracks]
    return small, replace(res, tracks=tracks)


# Shared State giữa luồng Inference và Web Server
class StreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_jpeg = None
        self.fps = 0.0
        self.yolo_ms = 0.0
        self.ocr_ms = 0.0
        self.frame_idx = 0
        self.total_frames = 0
        self.active_tracks = 0
        self.gpu_name = "CPU"
        self.recognized_plates = []  # 25 xe + biển số mới nhất (danh sách trên dashboard)
        self.plate_total = 0  # số biển khác nhau đã ghi nhận từ đầu video
        self.type_counts = {}
        self.video_key = "demo"
        self.switch_to = None  # key video người dùng vừa chọn, worker sẽ chuyển ở frame kế tiếp
        self.is_running = True
        self.mode = "play"  # play / pause / stop (tắt xử lý, GPU nghỉ)
        self.restart = False  # phát lại video hiện tại từ đầu

state = StreamState()

# ------------------------------------------------------------------------------
# LUỒNG XỬ LÝ: PHÁT HIỆN + PHÂN LOẠI XE, NHẬN DIỆN BIỂN SỐ TRÊN GPU
# ------------------------------------------------------------------------------
class Renderer:
    """Vẽ box/HUD, thu nhỏ và nén JPEG ở luồng riêng (~5 ms/khung) để luồng nhận diện không phải chờ.
    Chỉ giữ khung mới nhất: nếu vẽ chậm hơn nhận diện thì bỏ khung cũ, dashboard vẫn hiện khung gần nhất."""

    def __init__(self, show_window=False):
        self.show_window = show_window
        self.q = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, job):
        try:
            self.q.get_nowait()  # bỏ khung chưa kịp vẽ
        except queue.Empty:
            pass
        self.q.put(job)

    def _run(self):
        while state.is_running:
            try:
                frame, res, roi, hud_text, saves = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            for path, img in saves:
                cv2.imwrite(str(path), img)
            if state.mode == "stop":
                continue
            annotated = annotate(*to_display(frame, res), roi, hud="")
            draw_hud_text(annotated, hud_text)
            h, w = annotated.shape[:2]
            display = annotated if w == DISPLAY_W else cv2.resize(annotated, (DISPLAY_W, int(DISPLAY_W * h / w)))
            ok, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with state.lock:
                    state.current_jpeg = jpeg.tobytes()
            if self.show_window:
                cv2.imshow("Vehicle + License Plate Recognition", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    state.is_running = False
        if self.show_window:
            cv2.destroyAllWindows()


TYPE_COUNT_EVERY = 15  # đếm loại xe (bỏ phiếu lại trên mọi xe đã thấy) mỗi 15 khung thay vì mỗi khung


def idle_frame(text):
    """Ảnh JPEG báo trạng thái (đang tắt / đang nạp) khi không có khung video."""
    img = np.full((720, DISPLAY_W, 3), 24, np.uint8)
    cv2.putText(img, text, (60, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def wait_while_paused():
    """Đứng yên khi tạm dừng (giữ khung cuối trên dashboard). Trả về False nếu phải thoát vòng video."""
    while state.mode == "pause" and state.is_running and not state.switch_to and not state.restart:
        time.sleep(0.05)
    return state.mode == "play" and state.is_running and not state.switch_to and not state.restart


def inference_worker(show_window=False):
    renderer = Renderer(show_window)
    state.current_jpeg = idle_frame("Dang nap model...")
    pipe = ALPRPipeline(SCRIPT_DIR)  # nạp model 1 lần; mỗi video / lần phát lại chỉ reset trạng thái
    state.gpu_name = pipe.gpu_name
    print("[Worker] Bắt đầu stream video...")
    while state.is_running:
        if state.mode == "stop" and not state.switch_to:
            time.sleep(0.1)
            continue
        with state.lock:
            if state.switch_to:
                state.video_key, state.switch_to = state.switch_to, None
                state.mode = "play"  # chọn video khi đang tắt thì phát luôn
            state.restart = False
            key = state.video_key
            state.recognized_plates = []
            state.plate_total = 0
            state.type_counts = {}
            state.active_tracks = 0
            state.frame_idx = 0
            state.fps = 0.0
        _, video_path, roi = VIDEOS[key]
        pipe.reset(roi)
        published = {}  # vehicle_id -> biển số đã đẩy lên dashboard
        plate_set = set()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[Worker] Lỗi mở video: {video_path}")
            time.sleep(2)
            continue
        print(f"[Worker] Đang phát: {video_path.name}")
        state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        reader = FrameReader(video_path)
        frame_idx = 0
        stamps = []  # thời điểm xong các khung gần nhất -> FPS thực = số khung / thời gian

        while wait_while_paused():
            if stamps and time.perf_counter() - stamps[-1] > 0.5:
                stamps = []  # vừa tiếp tục sau tạm dừng: không tính khoảng dừng vào FPS
            ret, frame = reader.read()
            if not ret:
                break
            frame_idx += 1
            state.frame_idx = frame_idx

            try:
                res = pipe.process(frame, round(frame_idx / fps_video, 2))
            except Exception:  # 1 khung lỗi không được làm chết luồng xử lý: ghi log, xóa track, chạy tiếp
                traceback.print_exc()
                pipe.tracker.reset()
                continue

            # Đẩy xe có biển số mới lên dashboard (ảnh crop được ghi ở luồng vẽ)
            saves = []
            for vid, best in res.plates.items():
                if not best or published.get(vid) == best["plate"]:
                    continue
                published[vid] = best["plate"]
                vtype, _ = res.types.get(vid, ("unknown", None))
                crop_filename = f"{best['plate']}_veh{vid}.jpg"
                saves.append((CROP_DIR / crop_filename, best["crop"]))
                with state.lock:
                    state.recognized_plates.insert(0, {
                        "plate": best["plate"], "trust_score": best["trust_score"],
                        "track_id": f"Xe #{vid}", "vehicle_type": vtype,
                        "crop_file": crop_filename, "time": time.strftime("%H:%M:%S")})
                    del state.recognized_plates[25:]
                    plate_set.add(best["plate"])
                    state.plate_total = len(plate_set)

            stamps.append(time.perf_counter())
            del stamps[:-31]
            rolling_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
            det_ms = res.timings["vehicle_ms"] + res.timings["type_ms"] + res.timings["plate_ms"]

            with state.lock:
                state.fps = rolling_fps
                state.yolo_ms = det_ms
                state.ocr_ms = res.timings["ocr_ms"]
                state.active_tracks = sum(t.id is not None for t in res.tracks)  # xe đang có trong khung
                if frame_idx % TYPE_COUNT_EVERY == 1:
                    seen = pipe.type_voter.seen_ids()  # mọi xe đã thấy từ đầu video
                    state.type_counts = dict(Counter(pipe.vehicle_type(v)[0] for v in seen))

            hud_text = (f"{state.gpu_name} | FPS: {rolling_fps:.1f} | YOLO: {det_ms:.1f}ms | "
                        f"OCR: {res.timings['ocr_ms']:.1f}ms | Xe: {state.active_tracks}")
            renderer.submit((frame, res, roi, hud_text, saves))

        reader.release()
        if state.mode == "stop":
            state.fps = 0.0
            state.active_tracks = 0
            state.current_jpeg = idle_frame("Da tat xu ly - bam Bat de chay lai")
            print("[Worker] Đã tắt xử lý")
        elif not state.switch_to and not state.restart:
            print("[Worker] Hết video, tự động lặp lại luồng...")


# ------------------------------------------------------------------------------
# HTTP SERVER & WEBPAGE DASHBOARD
# ------------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALPR Live Stream | NVIDIA RTX 5060 Ti</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #07090e;
            --card-bg: rgba(18, 24, 38, 0.8);
            --border: rgba(255, 255, 255, 0.08);
            --neon-green: #10b981;
            --neon-cyan: #06b6d4;
            --neon-purple: #8b5cf6;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
            overflow-x: hidden;
            background-image: radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.05) 0%, transparent 40%);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }
        .title-group h1 {
            font-size: 24px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            letter-spacing: -0.5px;
        }
        .badge-live {
            background: rgba(16, 185, 129, 0.15);
            color: var(--neon-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 10px;
            font-size: 12px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .badge-live::before {
            content: "";
            width: 7px;
            height: 7px;
            background: var(--neon-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--neon-green);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .kpi-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px 20px;
            backdrop-filter: blur(12px);
            transition: transform 0.2s, border-color 0.2s;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
        }
        .kpi-label {
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 28px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #fff;
        }
        .kpi-value.green { color: var(--neon-green); text-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
        .kpi-value.cyan { color: var(--neon-cyan); }
        .kpi-value.purple { color: var(--neon-purple); }

        .main-layout {
            display: grid;
            grid-template-columns: 2.2fr 1fr;
            gap: 24px;
        }
        .stream-container {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            position: relative;
        }
        .stream-header {
            padding: 14px 20px;
            background: rgba(10, 14, 23, 0.9);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
            color: var(--text-muted);
        }
        .video-box {
            position: relative;
            width: 100%;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .video-box img {
            width: 100%;
            height: auto;
            display: block;
        }

        .side-panel {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 240px);
            min-height: 540px;
            overflow: hidden;
        }
        .panel-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            font-size: 16px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .plate-list {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .plate-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 12px;
            display: flex;
            align-items: center;
            gap: 14px;
            animation: fadeIn 0.3s ease-out;
            transition: background 0.2s;
        }
        .plate-card:hover {
            background: rgba(255, 255, 255, 0.06);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .plate-thumb {
            width: 76px;
            height: 42px;
            border-radius: 6px;
            object-fit: cover;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: #000;
        }
        .plate-info {
            flex: 1;
        }
        .plate-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 17px;
            font-weight: 700;
            color: #fff;
            letter-spacing: 0.5px;
        }
        .plate-meta {
            display: flex;
            gap: 8px;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .score-pill {
            color: var(--neon-green);
            font-weight: 600;
        }
        .ctl-btn {
            background: var(--card-bg);
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px 12px;
            font-family: inherit;
            font-size: 14px;
            cursor: pointer;
        }
        .ctl-btn:hover { border-color: var(--text-muted); }
        .ctl-btn:disabled { opacity: 0.4; cursor: default; }
        .video-select {
            background: var(--card-bg);
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px 10px;
            font-family: inherit;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="title-group">
            <h1>🚗 Giám Sát Phương Tiện & Biển Số (Module A + B)</h1>
            <span class="badge-live">Live Stream</span>
        </div>
        <div style="font-size: 14px; color: var(--text-muted); display: flex; align-items: center; gap: 10px;">
            <label for="video-select">Video:</label>
            <select id="video-select" class="video-select"></select>
            <button id="btn-pause" class="ctl-btn" onclick="togglePause()">⏸ Tạm dừng</button>
            <button id="btn-restart" class="ctl-btn" onclick="control('restart')">⟲ Phát lại</button>
            <button id="btn-power" class="ctl-btn" onclick="togglePower()">⏻ Tắt</button>
            <span>| YOLO11 + ByteTrack + FastPlateOCR</span>
        </div>
    </header>

    <section class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Throughput Realtime</div>
            <div class="kpi-value green" id="fps-val">-- FPS</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Độ Trễ YOLO + OCR</div>
            <div class="kpi-value cyan" id="latency-val">-- ms</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Phương Tiện Đang Bám</div>
            <div class="kpi-value purple" id="tracks-val">0</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Biển Số Ghi Nhận</div>
            <div class="kpi-value" id="plates-count-val">0</div>
        </div>
    </section>

    <main class="main-layout">
        <div class="stream-container">
            <div class="stream-header">
                <span id="video-name">📹 Luồng Video Trực Tiếp</span>
                <span id="frame-progress">Đang tải luồng...</span>
            </div>
            <div class="video-box">
                <img src="/video_feed" alt="Live Stream Video">
            </div>
        </div>

        <aside class="side-panel">
            <div class="panel-header">
                <span>📋 Danh Sách Biển Số Mới</span>
                <span style="font-size: 12px; color: var(--neon-green);" id="live-dot">● Cập nhật liên tục</span>
            </div>
            <div id="type-counts" style="padding: 8px 20px; font-size: 13px; color: var(--text-muted);"></div>
            <div class="plate-list" id="plate-feed">
                <!-- Danh sách card biển số xe sẽ tự động render tại đây -->
            </div>
        </aside>
    </main>

    <script>
        async function updateStats() {
            try {
                const res = await fetch('/api/stats');
                if (!res.ok) return;
                const data = await res.json();
                
                document.getElementById('fps-val').innerText = data.fps.toFixed(1) + ' FPS';
                document.getElementById('latency-val').innerText = (data.yolo_ms + data.ocr_ms).toFixed(1) + ' ms';
                document.getElementById('tracks-val').innerText = data.active_tracks;
                document.getElementById('plates-count-val').innerText = data.plate_total;
                document.getElementById('type-counts').innerText =
                    'Tổng xe đã qua — ' + Object.entries(data.type_counts || {}).map(([k, v]) => `${k}: ${v}`).join('  ·  ');
                document.getElementById('frame-progress').innerText = `Frame: ${data.frame_idx} / ${data.total_frames}`;
                if (data.mode && data.mode !== mode) { mode = data.mode; renderControls(); }

                // Cập nhật feed biển số
                const sel = document.getElementById('video-select');
                if (document.activeElement !== sel) sel.value = data.video;
                document.getElementById('video-name').innerText = '📹 ' + (data.video_label || '');

                const feed = document.getElementById('plate-feed');
                {
                    feed.innerHTML = (data.plates || []).map(p => `
                        <div class="plate-card">
                            <img class="plate-thumb" src="/crops/${p.crop_file}" onerror="this.style.display='none'">
                            <div class="plate-info">
                                <div class="plate-num">${p.plate}</div>
                                <div class="plate-meta">
                                    <span>${p.track_id}</span>
                                    <span>•</span>
                                    <span>${p.vehicle_type}</span>
                                    <span>•</span>
                                    <span class="score-pill">${p.trust_score}%</span>
                                    <span>•</span>
                                    <span>${p.time}</span>
                                </div>
                            </div>
                        </div>
                    `).join('');
                }
            } catch (err) {
                console.error("Lỗi cập nhật stats:", err);
            }
        }
        let mode = 'play';
        async function control(action) {
            const res = await fetch('/api/control?action=' + action);
            const data = await res.json();
            if (data.mode) { mode = data.mode; renderControls(); }
        }
        function togglePause() { control(mode === 'pause' ? 'resume' : 'pause'); }
        function togglePower() { control(mode === 'stop' ? 'start' : 'stop'); }
        function renderControls() {
            const pause = document.getElementById('btn-pause');
            pause.innerText = mode === 'pause' ? '▶ Tiếp tục' : '⏸ Tạm dừng';
            pause.disabled = mode === 'stop';
            document.getElementById('btn-power').innerText = mode === 'stop' ? '⏻ Bật' : '⏻ Tắt';
            document.querySelector('.badge-live').innerText =
                mode === 'play' ? 'Live Stream' : (mode === 'pause' ? 'Tạm dừng' : 'Đã tắt');
        }
        async function loadVideos() {
            const res = await fetch('/api/videos');
            const data = await res.json();
            const sel = document.getElementById('video-select');
            sel.innerHTML = data.videos.map(v => `<option value="${v.key}">${v.label}</option>`).join('');
            sel.value = data.current;
            sel.onchange = async () => {
                await fetch('/api/select?video=' + encodeURIComponent(sel.value));
                sel.blur();
            };
        }
        loadVideos();
        setInterval(updateStats, 500);
        updateStats();
    </script>
</body>
</html>
"""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Tắt log console HTTP để không spam terminal
        return

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while state.is_running:
                    jpeg_bytes = None
                    with state.lock:
                        jpeg_bytes = state.current_jpeg
                    if jpeg_bytes:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.025)  # ~40 FPS stream rate
            except (ConnectionResetError, BrokenPipeError):
                pass

        elif self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state.lock:
                data = {
                    "fps": round(state.fps, 1),
                    "yolo_ms": round(state.yolo_ms, 1),
                    "ocr_ms": round(state.ocr_ms, 1),
                    "active_tracks": state.active_tracks,
                    "plate_total": state.plate_total,
                    "mode": state.mode,
                    "frame_idx": state.frame_idx,
                    "total_frames": state.total_frames,
                    "gpu_name": state.gpu_name,
                    "type_counts": dict(state.type_counts),
                    "plates": list(state.recognized_plates),
                    "video": state.switch_to or state.video_key,
                    "video_label": VIDEOS[state.switch_to or state.video_key][0],
                }
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif self.path == "/api/videos":
            self.send_json({"current": state.video_key,
                            "videos": [{"key": k, "label": v[0]} for k, v in VIDEOS.items()]})

        elif self.path.startswith("/api/select"):
            key = parse_qs(urlparse(self.path).query).get("video", [""])[0]
            if key not in VIDEOS:
                self.send_json({"error": f"không có video '{key}'"}, status=404)
                return
            with state.lock:
                if key != state.video_key or state.switch_to:
                    state.switch_to = key
            self.send_json({"current": key})

        elif self.path.startswith("/api/control"):
            action = parse_qs(urlparse(self.path).query).get("action", [""])[0]
            with state.lock:
                if action == "pause" and state.mode == "play":
                    state.mode = "pause"
                elif action == "resume" and state.mode == "pause":
                    state.mode = "play"
                elif action == "stop":
                    state.mode = "stop"
                elif action == "start" and state.mode == "stop":
                    state.mode = "play"
                elif action == "restart":
                    state.mode, state.restart = "play", True
                else:
                    self.send_json({"error": f"lệnh '{action}' không hợp lệ khi đang {state.mode}", "mode": state.mode},
                                   status=400)
                    return
            self.send_json({"mode": state.mode})

        elif self.path.startswith("/crops/"):
            fname = self.path[len("/crops/"):]
            fpath = CROP_DIR / fname
            if fpath.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ALPR Real-Time Video Streaming App")
    parser.add_argument("--port", type=int, default=5000, help="Port Web Server (default: 5000)")
    parser.add_argument("--show-window", action="store_true", help="Mở thêm cửa sổ hiển thị desktop (cv2.imshow)")
    parser.add_argument("--video", default="demo", choices=list(VIDEOS), help="Video phát đầu tiên (đổi được trên web)")
    args = parser.parse_args()
    state.video_key = args.video

    # Khởi chạy luồng suy luận GPU
    worker_thread = threading.Thread(target=inference_worker, kwargs={"show_window": args.show_window}, daemon=True)
    worker_thread.start()

    # Khởi chạy HTTP Server
    server = ThreadedHTTPServer(("0.0.0.0", args.port), StreamHandler)
    print("=" * 75)
    print(f"🚀 ALPR LIVE STREAM SERVER ĐÃ SẴN SÀNG!")
    print(f"👉 Mời bạn mở trình duyệt xem tại : http://localhost:{args.port}")
    print(f"⚡ Pipeline: YOLO11 xe + v62 biển số + FastPlateOCR")
    print(f"🎯 Nhấn Ctrl + C trong terminal để dừng server.")
    print("=" * 75)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Đang dừng...")
        state.is_running = False
        server.server_close()

if __name__ == "__main__":
    main()
