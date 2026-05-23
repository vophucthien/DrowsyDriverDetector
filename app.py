import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from drowsy_driver.validation import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_TRANSITION_BUFFER_FRAMES,
    LATEST_VALIDATION_RESULTS,
    VALIDATION_CLIP,
    VALIDATION_SCHEDULE,
)
from drowsy_driver.detector import (
    CONSEC_FRAMES,
    EAR_THRESHOLD,
    FACE_GATE_GRACE_FRAMES,
    FACEMESH_ROI_MARGIN_RATIO,
    MP_LEFT_EYE,
    MP_RIGHT_EYE,
    configure_runtime_cache,
    draw_overlay,
    expand_box,
    eye_aspect_ratio,
    init_alarm,
    load_face_mesh_class,
    SilentAlarm,
    short_stage_name,
    suppress_native_stderr,
)

cv2 = None
np = None

BASE_DIR = Path(__file__).resolve().parent
WINDOW_NAME = "Drowsy Driver Detector"
DEFAULT_RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
]
REFERENCE_LAYOUT_RESOLUTION = (640, 480)
DEFAULT_WINDOW_SIZE = (960, 540)
MIN_RENDER_SIZE = (480, 270)
GUI_SCALES = [0.85, 1.0, 1.2, 1.45]
PERFORMANCE_SAMPLE_LIMIT = 120
PERFORMANCE_COLORS = [
    (77, 163, 255),
    (87, 196, 137),
    (255, 188, 66),
    (236, 112, 99),
    (157, 125, 255),
    (76, 201, 240),
    (255, 139, 92),
]


def load_gui_dependencies():
    global cv2, np
    if cv2 is None:
        import cv2 as cv2_module
        import numpy as np_module

        cv2 = cv2_module
        np = np_module


def log_status(message):
    print(f"[GUI] {message}", flush=True)


def resolution_layout_scale(frame):
    h, w = frame.shape[:2]
    ref_w, ref_h = REFERENCE_LAYOUT_RESOLUTION
    return max(0.65, min(w / ref_w, h / ref_h))


def effective_layout_scale(frame, gui_scale):
    return gui_scale * resolution_layout_scale(frame)


def current_window_size(default=DEFAULT_WINDOW_SIZE):
    try:
        _x, _y, width, height = cv2.getWindowImageRect(WINDOW_NAME)
    except Exception:
        return default

    min_w, min_h = MIN_RENDER_SIZE
    if width < min_w or height < min_h:
        return default
    return int(width), int(height)


def render_video_canvas(frame, canvas_size):
    canvas_w, canvas_h = canvas_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    if frame is None:
        return canvas

    frame_h, frame_w = frame.shape[:2]
    if frame_w <= 0 or frame_h <= 0:
        return canvas

    scale = max(canvas_w / frame_w, canvas_h / frame_h)
    target_w = max(canvas_w, int(round(frame_w * scale)))
    target_h = max(canvas_h, int(round(frame_h * scale)))
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    crop_x = max(0, (target_w - canvas_w) // 2)
    crop_y = max(0, (target_h - canvas_h) // 2)
    canvas[:, :] = resized[crop_y:crop_y + canvas_h, crop_x:crop_x + canvas_w]
    return canvas


def percent_text(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def load_validation_summary_lines():
    if not LATEST_VALIDATION_RESULTS.exists():
        return ["No validation results yet."]
    try:
        with open(LATEST_VALIDATION_RESULTS, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Could not read validation summary: {exc}"]

    results = payload.get("results", [])
    if not results:
        return ["No validation results yet."]

    result = results[-1]
    lines = [
        f"Latest validation: {payload.get('updated_at', 'unknown time')}",
        f"Overall score: {percent_text(result.get('f1'))}",
        (
            f"Correct alarms: {percent_text(result.get('precision'))} | "
            f"Sleepy frames caught: {percent_text(result.get('recall'))}"
        ),
        (
            f"Frames: {result.get('tp', 0)} sleepy caught, "
            f"{result.get('fp', 0)} false alarms, "
            f"{result.get('fn', 0)} sleepy missed, "
            f"{result.get('tn', 0)} awake correct"
        ),
        (
            f"Closed-eye periods detected: "
            f"{result.get('detected_closed_segments', 0)} of "
            f"{result.get('total_closed_segments', 0)}"
        ),
    ]
    report = result.get("report")
    if report:
        lines.append(f"Report: {report}")
    return lines[:6]


def draw_text_lines(frame, lines, x, y, scale=1.0, line_height=None, max_width=None):
    if line_height is None:
        line_height = int(24 * scale)
    for index, line in enumerate(lines):
        color = (255, 255, 255) if index == 0 else (210, 225, 235)
        font_scale = 0.46 * scale
        if max_width is not None:
            while font_scale > 0.30 * scale:
                (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                if text_width <= max_width:
                    break
                font_scale -= 0.03 * scale
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_fitted_text(frame, text, x, y, max_width, scale, color, thickness=1, font=None):
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = scale
    min_scale = max(0.26, scale * 0.62)
    while font_scale > min_scale:
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        if text_width <= max_width:
            break
        font_scale -= 0.03
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def panel_bounds(frame, scale=1.0):
    h, w = frame.shape[:2]
    margin = max(14, int(20 * scale))
    panel_w = min(int(760 * scale), w - 2 * margin)
    panel_h = min(int(580 * scale), h - 2 * margin)
    x1 = (w - panel_w) // 2
    y1 = (h - panel_h) // 2
    return x1, y1, x1 + panel_w, y1 + panel_h


def make_button_grid(panel_rect, rows, top, scale=1.0):
    x1, _y1, x2, _y2 = panel_rect
    pad = max(16, int(24 * scale))
    gap = max(8, int(12 * scale))
    bh = max(30, int(38 * scale))
    inner_w = x2 - x1 - 2 * pad
    bw = max(120, (inner_w - gap) // 2)
    buttons = []

    for row_index, row in enumerate(rows):
        y = int(top + row_index * (bh + gap))
        for col_index, item in enumerate(row):
            if item is None:
                continue
            key, label, enabled = item
            x = int(x1 + pad + col_index * (bw + gap))
            buttons.append(Button(key, label, (x, y, x + bw, y + bh), enabled=enabled))
    return buttons


def performance_summary(samples):
    if not samples:
        return None

    stage_names = []
    for sample in samples:
        for stage in sample:
            if stage not in stage_names:
                stage_names.append(stage)

    stages = []
    for stage in stage_names:
        values = []
        for sample in samples:
            try:
                values.append(float(sample.get(stage, 0.0)))
            except (TypeError, ValueError):
                pass
        if values:
            stages.append({
                "name": stage,
                "short_name": short_stage_name(stage),
                "ms": sum(values) / len(values),
            })

    total_ms = sum(stage["ms"] for stage in stages)
    if total_ms <= 0:
        return None

    for stage in stages:
        stage["percent"] = stage["ms"] / total_ms * 100.0
    return {
        "stages": stages,
        "total_ms": total_ms,
        "fps": 1000.0 / total_ms if total_ms > 0 else 0.0,
    }


def draw_performance_chart(frame, samples, rect, scale=1.0):
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    if width < 260 or height < 92:
        return

    cv2.line(frame, (x1, y1), (x2, y1), (86, 96, 108), 1)
    summary = performance_summary(samples)
    if summary is None:
        draw_fitted_text(
            frame,
            "Performance Test: collecting frame timings",
            x1,
            y1 + max(24, int(24 * scale)),
            width,
            0.44 * scale,
            (230, 230, 230),
        )
        return

    title = (
        f"Performance Test  Total {summary['total_ms']:.1f} ms  "
        f"~{summary['fps']:.1f} FPS"
    )
    draw_fitted_text(
        frame,
        title,
        x1,
        y1 + max(22, int(22 * scale)),
        width,
        0.42 * scale,
        (255, 255, 255),
    )

    radius = min(max(34, int(height * 0.30)), max(38, int(width * 0.14)), int(56 * scale))
    center = (x1 + radius + max(10, int(14 * scale)), y1 + height // 2 + int(12 * scale))
    start_angle = 0.0
    for index, stage in enumerate(summary["stages"]):
        sweep = stage["percent"] / 100.0 * 360.0
        end_angle = start_angle + sweep
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            0,
            start_angle,
            end_angle,
            PERFORMANCE_COLORS[index % len(PERFORMANCE_COLORS)],
            -1,
        )
        start_angle = end_angle
    cv2.ellipse(frame, center, (radius, radius), 0, 0, 360, (235, 235, 235), 1)

    legend_x = center[0] + radius + max(18, int(24 * scale))
    legend_y = y1 + max(42, int(42 * scale))
    row_h = max(13, int(16 * scale))
    font_scale = max(0.27, 0.32 * scale)
    sorted_stages = sorted(summary["stages"], key=lambda item: item["ms"], reverse=True)
    max_rows = max(1, (y2 - legend_y - 2) // row_h)

    for index, stage in enumerate(sorted_stages[:max_rows]):
        y = legend_y + index * row_h
        color = PERFORMANCE_COLORS[summary["stages"].index(stage) % len(PERFORMANCE_COLORS)]
        cv2.rectangle(frame, (legend_x, y - 10), (legend_x + 10, y), color, -1)
        label = f"{stage['short_name']}: {stage['percent']:.1f}%  {stage['ms']:.2f} ms"
        draw_fitted_text(
            frame,
            label,
            legend_x + 16,
            y,
            max(80, x2 - legend_x - 16),
            font_scale,
            (225, 235, 242),
        )


@dataclass
class Button:
    key: str
    label: str
    rect: tuple
    enabled: bool = True

    def contains(self, x, y):
        x1, y1, x2, y2 = self.rect
        return self.enabled and x1 <= x <= x2 and y1 <= y <= y2


class GuiState:
    def __init__(self):
        self.buttons = []
        self.mode = "live"
        self.message = "Starting live detector..."
        self.last_frame = None
        self.clicked = None
        self.validation_lines = load_validation_summary_lines()
        self.performance_samples = []

    def set_buttons(self, buttons):
        self.buttons = buttons

    def consume_click(self):
        clicked = self.clicked
        self.clicked = None
        return clicked

    def add_performance_sample(self, timings):
        if not timings:
            return
        self.performance_samples.append(dict(timings))
        overflow = len(self.performance_samples) - PERFORMANCE_SAMPLE_LIMIT
        if overflow > 0:
            del self.performance_samples[:overflow]


def on_mouse(event, x, y, _flags, state):
    if event != cv2.EVENT_LBUTTONUP:
        return
    for button in state.buttons:
        if button.contains(x, y):
            state.clicked = button.key
            return


def draw_button(frame, button, scale=1.0):
    x1, y1, x2, y2 = button.rect
    bg = (45, 52, 60) if button.enabled else (70, 70, 70)
    border = (230, 230, 230) if button.enabled else (120, 120, 120)
    text = (255, 255, 255) if button.enabled else (160, 160, 160)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)
    font_scale = 0.46 * scale
    thickness = 1 if scale < 1.3 else 2
    (tw, th), _ = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = x1 + max(6, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(frame, button.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text, thickness, cv2.LINE_AA)


def show_startup_frame(message):
    width, height = DEFAULT_WINDOW_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Drowsy Driver Detector",
        (32, 160),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (32, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "If this stays here, check the terminal for startup details.",
        (32, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)


def draw_panel(frame, title, lines, buttons, scale=1.0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    x1, y1, x2, y2 = panel_bounds(frame, scale)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 28, 34), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)

    pad = max(16, int(24 * scale))
    draw_fitted_text(
        frame,
        title,
        x1 + pad,
        y1 + max(42, int(46 * scale)),
        x2 - x1 - 2 * pad,
        0.82 * scale,
        (255, 255, 255),
        thickness=2,
        font=cv2.FONT_HERSHEY_DUPLEX,
    )
    y = y1 + max(76, int(88 * scale))
    for line in lines:
        draw_fitted_text(
            frame,
            line,
            x1 + pad,
            y,
            x2 - x1 - 2 * pad,
            0.46 * scale,
            (220, 220, 220),
        )
        y += max(20, int(24 * scale))

    for button in buttons:
        draw_button(frame, button, scale=scale)

    return x1, y1, x2, y2


def top_buttons(frame, scale, enabled=True):
    h, w = frame.shape[:2]
    bw = int(130 * scale)
    bh = int(36 * scale)
    gap = int(8 * scale)
    y = int(12 * scale)
    settings = Button("settings", "Settings", (w - bw - gap, y, w - gap, y + bh), enabled=enabled)
    validation = Button(
        "validation",
        "Validation",
        (w - 2 * bw - 2 * gap, y, w - bw - 2 * gap, y + bh),
        enabled=enabled,
    )
    return [validation, settings]


def open_camera(source, resolution):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/source: {source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(source, int) and resolution:
        width, height = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def load_facemesh(quiet_native_logs=True):
    if quiet_native_logs:
        with suppress_native_stderr():
            FaceMesh = load_face_mesh_class()
            return FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
    FaceMesh = load_face_mesh_class()
    return FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def available_camera_indexes(scan_limit=3):
    sources = []
    for index in range(scan_limit):
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened()
        cap.release()
        if ok:
            sources.append(index)
    return sources or [0]


def command_label(command):
    return " ".join(str(part) for part in command)


class CommandRunner:
    def __init__(self):
        self.process = None
        self.command = None
        self.on_complete = None

    @property
    def busy(self):
        return self.process is not None

    def start(self, command, state, on_complete=None):
        if self.busy:
            state.message = "A validation action is already running."
            return False

        state.message = "Running validation command. Watch terminal output..."
        print("\n[VALIDATION] " + command_label(command))
        try:
            self.process = subprocess.Popen(
                command,
                cwd=BASE_DIR,
                stdin=subprocess.DEVNULL,
            )
        except Exception as exc:
            self.process = None
            self.command = None
            self.on_complete = None
            state.message = f"Could not start validation command: {exc}"
            return False

        self.command = command
        self.on_complete = on_complete
        return True

    def poll(self, state):
        if self.process is None:
            return

        returncode = self.process.poll()
        if returncode is None:
            return

        on_complete = self.on_complete
        self.process = None
        self.command = None
        self.on_complete = None

        if returncode == 0:
            message = "Validation action completed."
        else:
            message = f"Validation action failed with exit code {returncode}."
        if on_complete is not None:
            callback_message = on_complete(returncode)
            if callback_message:
                message = callback_message
        state.message = message

    def terminate(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


def process_frame(frame, face_cascade, face_mesh, detector_state):
    raw_frame = frame.copy()
    fh, fw = frame.shape[:2]
    timings = {}

    t = time.perf_counter()
    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    timings["1. Grayscale"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    timings["2. Gaussian Blur"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    faces = face_cascade.detectMultiScale(
        blurred, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    face_box = None
    if len(faces):
        face_box = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = face_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        detector_state["haar_seen"] = True
        detector_state["haar_miss_frames"] = 0
        detector_state["last_face_box"] = face_box
    elif detector_state["haar_seen"]:
        detector_state["haar_miss_frames"] += 1

    should_run_facemesh = (
        face_box is not None
        or (
            detector_state["haar_seen"]
            and detector_state["haar_miss_frames"] <= FACE_GATE_GRACE_FRAMES
        )
    )
    active_box = face_box if face_box is not None else detector_state["last_face_box"]
    roi = expand_box(active_box, fw, fh, FACEMESH_ROI_MARGIN_RATIO) if should_run_facemesh and active_box is not None else None
    timings["3. Face Detection"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 255), 1)
    if face_box is not None:
        x, y, w, h = face_box
        ey1 = y + int(h * 0.30)
        ey2 = y + int(h * 0.55)
        cv2.rectangle(frame, (x, ey1), (x + w, ey2), (0, 165, 255), 1)
    timings["4. Eye ROI Draw"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    ear = 0.0
    if roi is not None and face_mesh is not None:
        x1, y1, x2, y2 = roi
        roi_bgr = raw_frame[y1:y2, x1:x2]
        roi_h, roi_w = roi_bgr.shape[:2]
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            points = np.array([
                [int(lm.x * roi_w) + x1, int(lm.y * roi_h) + y1]
                for lm in result.multi_face_landmarks[0].landmark
            ])
            left_eye = points[MP_LEFT_EYE]
            right_eye = points[MP_RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            for p in np.vstack([left_eye, right_eye]):
                cv2.circle(frame, tuple(p), 2, (0, 255, 100), -1)
    timings["5. Landmark + EAR"] = (time.perf_counter() - t) * 1000

    return ear, timings


def reset_detector_state():
    return {
        "closed_frames": 0,
        "alarm_on": False,
        "haar_seen": False,
        "haar_miss_frames": 0,
        "last_face_box": None,
        "threshold": EAR_THRESHOLD,
        "calibrating": True,
        "calibration_start": time.perf_counter(),
        "calibration_ears": [],
        "drowsy": False,
    }


def run_gui():
    load_gui_dependencies()
    configure_runtime_cache()
    cv2.setUseOptimized(True)
    state = GuiState()
    log_status("Opening detector window.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *DEFAULT_WINDOW_SIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)
    show_startup_frame("Starting live detector...")

    camera_sources = [0]
    camera_sources_scanned = False
    camera_index = 0
    source = camera_sources[camera_index]
    resolution_index = 0
    gui_scale_index = 1
    gui_scale = GUI_SCALES[gui_scale_index]

    cap = None
    face_mesh = None
    face_mesh_loader = {"mesh": None, "error": None, "done": False}
    alarm = SilentAlarm()
    alarm_ready = False
    audio_enabled = False
    try:
        log_status(f"Opening default camera/source {source}.")
        show_startup_frame(f"Opening default camera/source {source}...")
        cap = open_camera(source, DEFAULT_RESOLUTIONS[resolution_index])
        log_status("Loading Haar cascade.")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if face_cascade.empty():
            raise RuntimeError("Could not load Haar cascade.")
    except Exception:
        if cap is not None:
            cap.release()
        if face_mesh is not None:
            face_mesh.close()
        if audio_enabled:
            import pygame
            pygame.mixer.quit()
        raise
    detector = reset_detector_state()
    state.message = "Camera ready. Loading FaceMesh..."
    log_status("Camera ready. Loading FaceMesh in background.")

    def load_facemesh_background():
        try:
            face_mesh_loader["mesh"] = load_facemesh()
            log_status("MediaPipe FaceMesh ready.")
        except Exception as exc:
            face_mesh_loader["error"] = exc
            log_status(f"MediaPipe FaceMesh failed: {exc}")
        finally:
            face_mesh_loader["done"] = True

    threading.Thread(target=load_facemesh_background, daemon=True).start()

    last_raw_display = None
    validation_pause = False
    paused_frame = None
    command_runner = CommandRunner()

    def restart_camera_after_command(returncode):
        nonlocal cap, detector, validation_pause
        try:
            cap = open_camera(source, DEFAULT_RESOLUTIONS[resolution_index])
            detector = reset_detector_state()
            state.mode = "live"
            validation_pause = False
            if returncode == 0:
                return "Recording completed. Live detector restarted."
            return f"Recording failed with exit code {returncode}; live detector restarted."
        except Exception as exc:
            cap = None
            state.mode = "validation"
            validation_pause = True
            return f"Recording ended, but camera restart failed: {exc}"

    def refresh_validation_after_command(returncode):
        state.validation_lines = load_validation_summary_lines()
        if returncode == 0:
            return "Validation completed. Metrics updated below."
        return f"Validation failed with exit code {returncode}. Check terminal output."

    try:
        while True:
            command_runner.poll(state)
            command_running = command_runner.busy
            timings = {}
            overlay_data = None
            command_overlay = False
            if face_mesh is None and face_mesh_loader["mesh"] is not None:
                face_mesh = face_mesh_loader["mesh"]
                detector = reset_detector_state()
                state.message = "FaceMesh ready. Calibrating; keep eyes open."
            elif face_mesh is None and face_mesh_loader["error"] is not None:
                state.message = f"FaceMesh failed: {face_mesh_loader['error']}"

            if command_running:
                if paused_frame is not None:
                    display_base = paused_frame.copy()
                elif last_raw_display is not None:
                    display_base = last_raw_display.copy()
                else:
                    display_base = np.zeros((480, 640, 3), dtype=np.uint8)
                command_overlay = True
            elif validation_pause and paused_frame is not None:
                display_base = paused_frame.copy()
                timings = {}
                ear = 0.0
            else:
                if cap is None:
                    state.message = "Camera is unavailable. Open Validation or restart the app."
                    display_base = np.zeros((480, 640, 3), dtype=np.uint8)
                    ok = False
                else:
                    ok, frame = cap.read()
                if not ok:
                    state.message = "Could not read frame. Check camera input."
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ear, timings = process_frame(frame, face_cascade, face_mesh, detector)

                decision_start = time.perf_counter()
                if face_mesh is None:
                    if face_mesh_loader["error"] is None:
                        state.message = "Camera ready. Loading FaceMesh..."
                    detector["closed_frames"] = 0
                    detector["drowsy"] = False
                elif detector["calibrating"]:
                    elapsed = time.perf_counter() - detector["calibration_start"]
                    if ear > 0:
                        detector["calibration_ears"].append(ear)
                    if elapsed >= 5.0:
                        if detector["calibration_ears"]:
                            detector["threshold"] = float(np.median(detector["calibration_ears"]) * 0.75)
                            state.message = f"Calibration complete. Threshold={detector['threshold']:.3f}"
                        else:
                            detector["threshold"] = EAR_THRESHOLD
                            state.message = "Calibration failed; using default threshold."
                        detector["calibrating"] = False
                else:
                    if ear > 0 and ear < detector["threshold"]:
                        detector["closed_frames"] += 1
                        if detector["closed_frames"] >= CONSEC_FRAMES:
                            detector["drowsy"] = True
                            if not detector["alarm_on"]:
                                if not alarm_ready:
                                    alarm, audio_enabled = init_alarm(enabled=True)
                                    alarm_ready = True
                                alarm.play(-1)
                                detector["alarm_on"] = True
                    else:
                        detector["closed_frames"] = 0
                        detector["drowsy"] = False
                        if detector["alarm_on"]:
                            alarm.stop()
                            detector["alarm_on"] = False
                timings["6. Decision & Alert"] = (time.perf_counter() - decision_start) * 1000

                overlay_data = (
                    ear,
                    detector["drowsy"],
                    detector["closed_frames"],
                    timings,
                    detector["threshold"],
                    CONSEC_FRAMES,
                    detector["calibrating"],
                    time.perf_counter() - detector["calibration_start"] if detector["calibrating"] else 0.0,
                    5.0,
                )
                display_base = frame
                last_raw_display = frame.copy()
                state.add_performance_sample(timings)

            display = render_video_canvas(display_base, current_window_size())
            layout_scale = effective_layout_scale(display, gui_scale)

            if overlay_data is not None:
                draw_overlay(display, *overlay_data, ui_scale=layout_scale)
            if command_overlay:
                cv2.putText(
                    display,
                    "Validation command running...",
                    (int(16 * layout_scale), int(42 * layout_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72 * layout_scale,
                    (255, 255, 255),
                    max(1, int(round(2 * layout_scale))),
                    cv2.LINE_AA,
                )

            buttons = top_buttons(display, layout_scale, enabled=not command_running)
            for button in buttons:
                draw_button(display, button, scale=layout_scale)

            if state.message:
                cv2.putText(display, state.message, (12, display.shape[0] - int(18 * layout_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48 * layout_scale, (255, 255, 255), 1, cv2.LINE_AA)

            if state.mode == "settings":
                h, w = display.shape[:2]
                panel_rect = panel_bounds(display, layout_scale)
                _px1, py1, _px2, py2 = panel_rect
                button_top = py1 + max(96, int(104 * layout_scale))
                panel_buttons = make_button_grid(
                    panel_rect,
                    [
                        [
                            ("recalibrate", "Recalibrate EAR", not command_running),
                            ("reset_threshold", "Reset EAR Default", not command_running),
                        ],
                        [
                            ("scale", f"GUI Scale {gui_scale:.2f}x", not command_running),
                            ("camera", f"Camera {source}", not command_running),
                        ],
                        [
                            (
                                "resolution",
                                f"Resolution {DEFAULT_RESOLUTIONS[resolution_index][0]}x{DEFAULT_RESOLUTIONS[resolution_index][1]}",
                                not command_running,
                            ),
                            ("close_modal", "Back to Live", not command_running),
                        ],
                    ],
                    button_top,
                    scale=layout_scale,
                )
                draw_panel(
                    display,
                    "Settings",
                    [
                        "Live detector keeps running unless Validation is opened.",
                    ],
                    panel_buttons,
                    scale=layout_scale,
                )
                last_button_bottom = max((button.rect[3] for button in panel_buttons), default=button_top)
                chart_pad = max(16, int(24 * layout_scale))
                chart_top = last_button_bottom + max(18, int(24 * layout_scale))
                draw_performance_chart(
                    display,
                    state.performance_samples,
                    (
                        panel_rect[0] + chart_pad,
                        chart_top,
                        panel_rect[2] - chart_pad,
                        py2 - chart_pad,
                    ),
                    scale=layout_scale,
                )
                buttons.extend(panel_buttons)

            if state.mode == "validation":
                h, w = display.shape[:2]
                panel_rect = panel_bounds(display, layout_scale)
                px1, py1, px2, py2 = panel_rect
                button_top = py1 + max(128, int(138 * layout_scale))
                panel_buttons = make_button_grid(
                    panel_rect,
                    [
                        [
                            ("record_validation", "Record Validation Clip", not command_running),
                            ("run_validation", "Run Validation Test", not command_running and VALIDATION_CLIP.exists()),
                        ],
                        [
                            ("latest_report", "Show Latest Report", not command_running),
                            ("close_modal", "Resume Live", not command_running),
                        ],
                    ],
                    button_top,
                    scale=layout_scale,
                )
                draw_panel(
                    display,
                    "Validation",
                    [
                        "Live detection is paused here so validation actions are stable.",
                        "One validation clip covers normal open/closed and boundary durations.",
                    ],
                    panel_buttons,
                    scale=layout_scale,
                )
                last_button_bottom = max((button.rect[3] for button in panel_buttons), default=button_top)
                text_pad = max(16, int(24 * layout_scale))
                draw_text_lines(
                    display,
                    state.validation_lines,
                    px1 + text_pad,
                    last_button_bottom + max(42, int(52 * layout_scale)),
                    scale=layout_scale,
                    max_width=px2 - px1 - 2 * text_pad,
                )
                buttons.extend(panel_buttons)

            state.set_buttons(buttons)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            click = state.consume_click()

            if key == ord("q"):
                break
            if key == ord("s"):
                click = "settings"
            if key in (ord("v"), ord("d")):
                click = "validation"

            if command_runner.busy and click is not None:
                state.message = "Validation action is still running. Watch terminal output..."
                click = None

            if click == "settings":
                state.mode = "settings"
            elif click == "validation":
                state.mode = "validation"
                validation_pause = True
                paused_frame = last_raw_display.copy() if last_raw_display is not None else display_base.copy()
                if detector["alarm_on"]:
                    alarm.stop()
                    detector["alarm_on"] = False
            elif click == "close_modal":
                state.mode = "live"
                validation_pause = False
            elif click == "recalibrate":
                detector["calibrating"] = True
                detector["calibration_start"] = time.perf_counter()
                detector["calibration_ears"].clear()
                detector["closed_frames"] = 0
                detector["drowsy"] = False
                state.message = "Recalibrating. Keep eyes open."
                state.mode = "live"
            elif click == "reset_threshold":
                detector["threshold"] = EAR_THRESHOLD
                detector["calibrating"] = False
                detector["closed_frames"] = 0
                state.message = f"Threshold reset to default {EAR_THRESHOLD:.3f}."
                state.mode = "live"
            elif click == "scale":
                gui_scale_index = (gui_scale_index + 1) % len(GUI_SCALES)
                gui_scale = GUI_SCALES[gui_scale_index]
                state.message = f"GUI scale set to {gui_scale:.2f}x."
            elif click == "camera":
                if not camera_sources_scanned:
                    state.message = "Scanning cameras..."
                    camera_sources = available_camera_indexes()
                    camera_sources_scanned = True
                    camera_index = camera_sources.index(source) if source in camera_sources else 0
                if len(camera_sources) <= 1:
                    state.message = f"Only camera {source} found."
                    state.mode = "live"
                    continue
                camera_index = (camera_index + 1) % len(camera_sources)
                source = camera_sources[camera_index]
                if cap is not None:
                    cap.release()
                cap = open_camera(source, DEFAULT_RESOLUTIONS[resolution_index])
                detector = reset_detector_state()
                state.message = f"Camera changed to {source}. Recalibrating."
                state.mode = "live"
            elif click == "resolution":
                resolution_index = (resolution_index + 1) % len(DEFAULT_RESOLUTIONS)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_RESOLUTIONS[resolution_index][0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_RESOLUTIONS[resolution_index][1])
                detector = reset_detector_state()
                state.message = f"Resolution set to {DEFAULT_RESOLUTIONS[resolution_index][0]}x{DEFAULT_RESOLUTIONS[resolution_index][1]}."
                state.mode = "live"
            elif click == "record_validation":
                if cap is not None:
                    cap.release()
                    cap = None
                paused_frame = last_raw_display.copy() if last_raw_display is not None else display_base.copy()
                validation_pause = True
                if not command_runner.start(
                    [
                        sys.executable,
                        "-m",
                        "drowsy_driver.validation",
                        "--record",
                        "--output",
                        str(VALIDATION_CLIP),
                        "--schedule",
                        VALIDATION_SCHEDULE,
                    ],
                    state,
                    on_complete=restart_camera_after_command,
                ):
                    state.message = restart_camera_after_command(1)
            elif click == "run_validation":
                paused_frame = last_raw_display.copy() if last_raw_display is not None else display_base.copy()
                validation_pause = True
                command_runner.start(
                    [sys.executable, str(BASE_DIR / "app.py"), "--validate-all"],
                    state,
                    on_complete=refresh_validation_after_command,
                )
            elif click == "latest_report":
                paused_frame = last_raw_display.copy() if last_raw_display is not None else display_base.copy()
                validation_pause = True
                command_runner.start([sys.executable, str(BASE_DIR / "app.py"), "--latest-report"], state)

    finally:
        command_runner.terminate()
        if detector["alarm_on"]:
            alarm.stop()
        if cap is not None:
            cap.release()
        if face_mesh is not None:
            face_mesh.close()
        if audio_enabled:
            import pygame
            pygame.mixer.quit()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Drowsy Driver Detector app")
    parser.add_argument("--menu", action="store_true", help="Open the terminal validation menu")
    parser.add_argument("--validate", metavar="CLIP", help="Run validation directly")
    parser.add_argument("--validate-all", action="store_true", help="Run the standard validation test clip")
    parser.add_argument("--tag", default=None, help="Tag to use with --validate")
    parser.add_argument(
        "--transition-buffer-frames",
        type=int,
        default=None,
        help=f"Frames to exclude around schedule transitions during validation (default: {DEFAULT_TRANSITION_BUFFER_FRAMES})",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help=f"Bootstrap samples for validation confidence intervals (default: {DEFAULT_BOOTSTRAP_ITERATIONS})",
    )
    parser.add_argument("--assume-fps", default=None, help="Override video FPS for validation")
    parser.add_argument("--latest-report", action="store_true", help="Print the latest generated report")
    args = parser.parse_args()

    if args.menu:
        from drowsy_driver.validation import menu

        menu()
        return
    if args.latest_report:
        from drowsy_driver.validation import latest_report

        latest_report()
        return
    if args.validate_all:
        from drowsy_driver.validation import validate_standard_clip

        validate_standard_clip(
            buffer_frames=args.transition_buffer_frames,
            bootstrap_iterations=args.bootstrap_iterations,
            assume_fps=args.assume_fps,
        )
        return
    if args.validate:
        from drowsy_driver.validation import validate_clip

        validate_clip(
            input_path=args.validate,
            tag=args.tag,
            buffer_frames=args.transition_buffer_frames,
            bootstrap_iterations=args.bootstrap_iterations,
            assume_fps=args.assume_fps,
            interactive=False,
        )
        return

    run_gui()


if __name__ == "__main__":
    main()
