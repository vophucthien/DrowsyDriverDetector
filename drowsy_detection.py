import os
import time
import argparse
import sys
import json
import csv
import importlib.util
import types
from contextlib import contextmanager

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

# ── Constants ─────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.22   # below this → eye considered closed/drowsy
CONSEC_FRAMES = 15     # consecutive drowsy frames before alarm (~0.5s @ 30fps)
APP_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CAMERA_ALIASES_FILE = os.path.join(os.path.dirname(__file__), ".camera_aliases.json")
PERFORMANCE_LOG_FILE = os.path.join(os.path.dirname(__file__), "pipeline_performance_log.csv")

# Mediapipe FaceMesh 6-point EAR indices (Soukupová & Čech, 2016)
MP_LEFT_EYE  = [33,  160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ── Helpers ───────────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye_pts) -> float:
    from scipy.spatial import distance as dist

    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)


def parse_video_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def configure_runtime_cache():
    os.makedirs(APP_CACHE_DIR, exist_ok=True)
    matplotlib_dir = os.path.join(APP_CACHE_DIR, "matplotlib")
    xdg_cache_dir = os.path.join(APP_CACHE_DIR, "xdg")
    os.makedirs(matplotlib_dir, exist_ok=True)
    os.makedirs(xdg_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_dir)


def load_face_mesh_class():
    """Load MediaPipe FaceMesh without importing the unused mediapipe.tasks stack."""
    spec = importlib.util.find_spec("mediapipe")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("MediaPipe is not installed in this Python environment.")

    root = next(iter(spec.submodule_search_locations))
    if "mediapipe" not in sys.modules:
        mediapipe_pkg = types.ModuleType("mediapipe")
        mediapipe_pkg.__path__ = [root]
        mediapipe_pkg.__package__ = "mediapipe"
        sys.modules["mediapipe"] = mediapipe_pkg

    try:
        from mediapipe.python.solutions.face_mesh import FaceMesh
    except Exception as exc:
        raise RuntimeError(
            "Could not load MediaPipe FaceMesh. Recreate the virtual environment "
            "with Python 3.12 and install requirements.txt."
        ) from exc

    return FaceMesh


@contextmanager
def suppress_native_stderr():
    saved_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)


def load_camera_aliases():
    try:
        with open(CAMERA_ALIASES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if str(value).strip()}


def save_camera_aliases(aliases):
    try:
        with open(CAMERA_ALIASES_FILE, "w", encoding="utf-8") as f:
            json.dump(aliases, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as exc:
        print(f"[WARN] Could not save camera aliases: {exc}")


def describe_video_source(source):
    if isinstance(source, int):
        alias = load_camera_aliases().get(str(source), "").strip()
        return alias if alias else f"Camera index {source}"
    return str(source)


def performance_stage_column(stage):
    return f"{stage} (ms)"


def read_performance_log():
    try:
        with open(PERFORMANCE_LOG_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames or [], list(reader)
    except FileNotFoundError:
        return [], []
    except (OSError, csv.Error) as exc:
        print(f"[WARN] Could not read performance log: {exc}")
        return [], []


def write_performance_log(fieldnames, rows):
    with open(PERFORMANCE_LOG_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_performance_log(mean_timings, frame_count, total_mean_latency, mean_fps, source_label):
    existing_fields, rows = read_performance_log()
    stage_fields = [performance_stage_column(stage) for stage in mean_timings]
    standard_fields = ["timestamp", "source", "frames", *stage_fields, "total_ms", "fps"]

    fieldnames = []
    for field in [*existing_fields, *standard_fields]:
        if field and field not in fieldnames:
            fieldnames.append(field)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_label,
        "frames": str(frame_count),
        "total_ms": f"{total_mean_latency:.6f}",
        "fps": f"{mean_fps:.6f}",
    }
    for stage, ms in mean_timings.items():
        row[performance_stage_column(stage)] = f"{ms:.6f}"

    rows.append(row)

    try:
        write_performance_log(fieldnames, rows)
    except OSError as exc:
        print(f"[WARN] Could not write performance log: {exc}")

    return rows


def numeric_average(rows, field):
    values = []
    for row in rows:
        try:
            values.append(float(row.get(field, "")))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def short_stage_name(stage):
    name = stage.split(".", 1)[-1].strip() if "." in stage else stage
    aliases = {
        "Grayscale": "Gray",
        "Gaussian Blur": "Blur",
        "Face Detection": "Face",
        "Eye ROI Crop": "ROI",
        "Landmark + EAR": "EAR",
        "Decision & Alert": "Alert",
    }
    return aliases.get(name, name[:8])


def truncate_cell(value, width):
    value = str(value)
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[:width - 1] + "."


def format_log_number(row, field, width, decimals=2):
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return "-".rjust(width)
    return f"{value:.{decimals}f}".rjust(width)


def format_average_value(rows, field, width, decimals=2):
    value = numeric_average(rows, field)
    if value is None:
        return "-".rjust(width)
    return f"{value:.{decimals}f}".rjust(width)


def print_performance_log_table(rows, stage_names):
    if not rows:
        return

    stage_columns = [
        (short_stage_name(stage), performance_stage_column(stage), 8)
        for stage in stage_names
    ]
    columns = [
        ("Run", 4),
        ("Timestamp", 19),
        ("Source", 18),
        ("Frames", 6),
        *[(label, width) for label, _field, width in stage_columns],
        ("Total", 8),
        ("FPS", 7),
    ]
    line = "-+-".join("-" * width for _label, width in columns)

    print("\nPERFORMANCE LOG (RECORDED RUNS)")
    print("Log File:", PERFORMANCE_LOG_FILE)
    print("Delete this CSV to reset the history.\n")

    print(" | ".join(label.ljust(width) for label, width in columns))
    print(line)
    for run_number, row in enumerate(rows, start=1):
        cells = [
            str(run_number).rjust(4),
            truncate_cell(row.get("timestamp", "-"), 19).ljust(19),
            truncate_cell(row.get("source", "-"), 18).ljust(18),
            format_log_number(row, "frames", 6, decimals=0),
        ]
        for _label, field, width in stage_columns:
            cells.append(format_log_number(row, field, width))
        cells.extend([
            format_log_number(row, "total_ms", 8),
            format_log_number(row, "fps", 7, decimals=1),
        ])
        print(" | ".join(cells))

    print(line)
    average_cells = [
        "Avg".rjust(4),
        "-".ljust(19),
        f"{len(rows)} run{'s' if len(rows) != 1 else ''}".ljust(18),
        format_average_value(rows, "frames", 6, decimals=0),
    ]
    for _label, field, width in stage_columns:
        average_cells.append(format_average_value(rows, field, width))
    average_cells.extend([
        format_average_value(rows, "total_ms", 8),
        format_average_value(rows, "fps", 7, decimals=1),
    ])
    print(" | ".join(average_cells))

    stage_averages = {}
    for stage in stage_names:
        avg_ms = numeric_average(rows, performance_stage_column(stage))
        if avg_ms is not None:
            stage_averages[stage] = avg_ms

    total_latency = numeric_average(rows, "total_ms")
    if total_latency is None:
        total_latency = sum(stage_averages.values())
    average_fps = numeric_average(rows, "fps")
    if average_fps is None and total_latency > 0:
        average_fps = 1000.0 / total_latency

    print("\n" + "="*55)
    print(f"       AVERAGE PERFORMANCE ACROSS RUNS ({len(rows)} run{'s' if len(rows) != 1 else ''})")
    print("="*55)
    for stage, ms in stage_averages.items():
        percentage = (ms / total_latency) * 100 if total_latency > 0 else 0
        print(f"{stage:<25}: {ms:>6.2f} ms ({percentage:>5.1f}%)")
    print("-"*55)
    print(f"Avg Total Pipeline Latency: {total_latency:.2f} ms")
    if average_fps is not None:
        print(f"Avg System Throughput     : {average_fps:.1f} FPS")
    print("="*55)


def unique_labels(labels):
    unique = []
    for label in labels:
        label = str(label).strip()
        if label and label not in unique:
            unique.append(label)
    return unique


def get_platform_camera_labels():
    if sys.platform == "darwin":
        return get_macos_camera_labels()
    if sys.platform.startswith("linux"):
        return get_linux_camera_labels()
    if os.name == "nt":
        return get_windows_camera_labels()
    return []


def get_macos_camera_labels():
    import subprocess

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    return unique_labels(camera.get("_name", "") for camera in data.get("SPCameraDataType", []))


def get_linux_camera_labels():
    labels = []
    video_root = "/sys/class/video4linux"
    try:
        devices = sorted(os.listdir(video_root))
    except OSError:
        return []

    for device in devices:
        name_path = os.path.join(video_root, device, "name")
        try:
            with open(name_path, "r", encoding="utf-8") as f:
                name = f.read().strip()
        except OSError:
            continue
        if name:
            labels.append(name)
    return unique_labels(labels)


def get_windows_camera_labels():
    import subprocess

    commands = [
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { ($_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image') -and $_.Name } | "
            "Select-Object -ExpandProperty Name",
        ],
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName",
        ],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=4,
            )
        except Exception:
            continue
        if result.returncode == 0:
            labels = unique_labels(result.stdout.splitlines())
            if labels:
                return labels
    return []


def format_camera_label(index, resolution=None, aliases=None):
    aliases = aliases or {}
    alias = aliases.get(str(index), "").strip()
    label = alias if alias else "Camera"
    if resolution:
        return f"{label} [index {index}] ({resolution[0]}x{resolution[1]})"
    return f"{label} [index {index}]"


def find_camera_sources(cv2, scan_limit=5):
    sources = []
    aliases = load_camera_aliases()
    for index in range(scan_limit):
        cap = None
        try:
            with suppress_native_stderr():
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    continue

                ok, frame = cap.read()
            label = format_camera_label(index, aliases=aliases)
            if ok and frame is not None:
                h, w = frame.shape[:2]
                label = format_camera_label(index, resolution=(w, h), aliases=aliases)
            sources.append((str(index), label))
        finally:
            if cap is not None:
                cap.release()

    if not sources:
        sources.append(("0", format_camera_label(0, aliases=aliases)))
    return sources


def edit_camera_aliases(cameras):
    aliases = load_camera_aliases()
    print("\nCamera aliases are saved locally in .camera_aliases.json.")
    print("Press Enter to keep the current label, or type '-' to clear it.")

    for source, label in cameras:
        current = aliases.get(source, "")
        value = input(f"{label}\nName for index {source} [{current or 'none'}]: ").strip()
        if value == "":
            continue
        if value == "-":
            aliases.pop(source, None)
        else:
            aliases[source] = value

    save_camera_aliases(aliases)
    print("[INFO] Camera aliases updated.")


def prompt_camera_alias_after_run(source, detected_labels, quit_requested):
    if not quit_requested or not sys.stdin.isatty() or not isinstance(source, int):
        return

    aliases = load_camera_aliases()
    source_key = str(source)
    current = aliases.get(source_key, "")
    labels = unique_labels([current, *detected_labels])

    print(f"\nYou just used camera index {source}. What camera was that?")
    if labels:
        print("Detected camera labels from startup:")
        for i, label in enumerate(labels, start=1):
            print(f"  {i}. {label}")
    else:
        print("No camera labels were detected by the operating system at startup.")
    print("  C. Custom label")
    print("  S. Skip")

    while True:
        choice = input("Choose label, C for custom, or S to skip: ").strip()
        if not choice or choice.lower() == "s":
            return
        if choice.lower() == "c":
            label = input(f"Custom label for camera index {source}: ").strip()
            if not label:
                print("Custom label cannot be empty.")
                continue
            aliases[source_key] = label
            save_camera_aliases(aliases)
            print(f"[INFO] Saved camera index {source} as: {label}")
            return
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(labels):
                aliases[source_key] = labels[index - 1]
                save_camera_aliases(aliases)
                print(f"[INFO] Saved camera index {source} as: {labels[index - 1]}")
                return
        print("Choose one of the listed labels, C for custom, or S to skip.")


def choose_video_source(cv2, scan_limit=5):
    cameras = find_camera_sources(cv2, scan_limit=scan_limit)

    while True:
        print("\nAvailable camera sources:")
        for i, (_source, label) in enumerate(cameras, start=1):
            print(f"  {i}. {label}")
        print("  R. Rename/save camera labels")
        print("  C. Custom path, URL, or camera index")

        choice = input(f"Select camera [1-{len(cameras)}] or C (default 1): ").strip()
        if not choice:
            return cameras[0][0]
        if choice.lower() == "r":
            edit_camera_aliases(cameras)
            cameras = find_camera_sources(cv2, scan_limit=scan_limit)
            continue
        if choice.lower() == "c":
            custom = input("Enter custom path, URL, or camera index: ").strip()
            if custom:
                return custom
            print("Custom source cannot be empty.")
            continue
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(cameras):
                return cameras[index - 1][0]
        print("Choose a listed camera number or C for a custom source.")


def choose_startup_source(cv2, scan_limit=5):
    if not sys.stdin.isatty():
        print("[WARN] Interactive startup menu unavailable; using camera 0")
        return "0"

    print("\nStartup options:")
    print("  1. Use default built-in camera (0)")
    print("  2. Scan for available cameras")

    while True:
        choice = input("Choose startup option [1/2] (default 1): ").strip()
        if choice in ("", "1"):
            return "0"
        if choice == "2":
            return choose_video_source(cv2, scan_limit=scan_limit)
        print("Choose 1 for the default camera or 2 to scan cameras.")


def make_beep(pygame, freq=880, duration=0.5, rate=44100):
    import numpy as np

    t = np.linspace(0, duration, int(rate * duration), False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))


class SilentAlarm:
    """Fallback alarm used when the audio device or pygame mixer is unavailable."""

    def play(self, *args, **kwargs):
        return None

    def stop(self):
        return None


def init_alarm(enabled=True):
    if not enabled:
        return SilentAlarm(), False

    try:
        import pygame

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        return make_beep(pygame), True
    except Exception as exc:
        print(f"[WARN] Audio alarm disabled: {exc}")
        return SilentAlarm(), False


# ── Overlay ───────────────────────────────────────────────────────────────────

def draw_overlay(frame, ear, drowsy, frame_counter, timings, threshold, frame_limit,
                 calibrating=False):
    import cv2

    h, w = frame.shape[:2]

    if drowsy:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.putText(frame, "! DROWSY  WAKE UP !",
                    (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3)

    status_color = (0, 0, 255) if drowsy else (0, 220, 0)
    status = "Calibrating" if calibrating else ("DROWSY" if drowsy else "Awake")
    status_color = (0, 200, 255) if calibrating else status_color
    cv2.putText(frame, f"Status: {status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
    cv2.putText(frame,
                f"EAR: {ear:.3f}   threshold: {threshold:.3f}   closed frames: {frame_counter}/{frame_limit}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    y, total = 88, 0.0
    for label, ms in timings.items():
        cv2.putText(frame, f"{label}: {ms:.1f} ms",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)
        y += 18
        total += ms

    fps = 1000.0 / total if total > 0 else 0.0
    cv2.putText(frame, f"Pipeline: {total:.1f} ms  (~{fps:.1f} FPS)",
                (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 50), 1)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(cap, face_cascade, alarm, ear_threshold=EAR_THRESHOLD,
        consecutive_frames=CONSEC_FRAMES, display=True, calibrate=False,
        calibration_seconds=5.0, calibration_ratio=0.75,
        source_label="Unknown source", quiet_native_logs=True):
    import cv2
    import numpy as np

    print("[INFO] Loading MediaPipe FaceMesh...")
    if quiet_native_logs:
        with suppress_native_stderr():
            FaceMesh = load_face_mesh_class()
    else:
        FaceMesh = load_face_mesh_class()

    try:
        if quiet_native_logs:
            with suppress_native_stderr():
                face_mesh = FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
        else:
            face_mesh = FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
    except RuntimeError as exc:
        raise RuntimeError(
            "MediaPipe FaceMesh could not start. On macOS this can happen when "
            "Python cannot create an OpenGL context, such as from a restricted "
            "IDE/sandbox. Try running the command from a normal Terminal window."
        ) from exc
    print("[INFO] MediaPipe FaceMesh ready")

    frame_counter, alarm_on = 0, False
    calibration_start = time.perf_counter()
    calibration_ears = []
    current_threshold = ear_threshold
    is_calibrating = calibrate

    # Initialize data store list for performance monitoring metrics
    all_timings_data = []
    quit_requested = False
    quiet_frames_remaining = 2 if quiet_native_logs else 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fh, fw = frame.shape[:2]
            timings = {}

            # ① Grayscale
            t = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timings["1. Grayscale"] = (time.perf_counter() - t) * 1000

            # ② Gaussian blur
            t = time.perf_counter()
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            timings["2. Gaussian Blur"] = (time.perf_counter() - t) * 1000

            # ③ Haar face detection
            t = time.perf_counter()
            faces = face_cascade.detectMultiScale(
                blurred, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            if len(faces):
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            timings["3. Face Detection"] = (time.perf_counter() - t) * 1000

            # ④ Eye ROI crop (geometric)
            t = time.perf_counter()
            if len(faces):
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                ey1 = y + int(h * 0.30)
                ey2 = y + int(h * 0.55)
                cv2.rectangle(frame, (x, ey1), (x + w, ey2), (0, 165, 255), 1)
            timings["4. Eye ROI Crop"] = (time.perf_counter() - t) * 1000

            # ⑤ Mediapipe FaceMesh → EAR thực sự
            t = time.perf_counter()
            ear = 0.0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if quiet_frames_remaining > 0:
                with suppress_native_stderr():
                    result = face_mesh.process(rgb)
                quiet_frames_remaining -= 1
            else:
                result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                pts = np.array([[int(lm.x * fw), int(lm.y * fh)] for lm in lms])
                left_eye = pts[MP_LEFT_EYE]
                right_eye = pts[MP_RIGHT_EYE]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                for p in np.vstack([left_eye, right_eye]):
                    cv2.circle(frame, tuple(p), 2, (0, 255, 100), -1)
            timings["5. Landmark + EAR"] = (time.perf_counter() - t) * 1000

            # ⑥ Decision & alert
            t = time.perf_counter()
            drowsy = False
            if is_calibrating:
                if ear > 0:
                    calibration_ears.append(ear)
                elapsed = time.perf_counter() - calibration_start
                if elapsed >= calibration_seconds:
                    if calibration_ears:
                        current_threshold = float(np.median(calibration_ears) * calibration_ratio)
                        print(
                            f"[INFO] Calibration complete | open-eye EAR median: "
                            f"{np.median(calibration_ears):.3f} | threshold: {current_threshold:.3f}"
                        )
                    else:
                        print(
                            "[WARN] Calibration did not collect face landmarks; "
                            f"using threshold {current_threshold:.3f}"
                        )
                    is_calibrating = False
            elif ear > 0 and ear < current_threshold:
                frame_counter += 1
                if frame_counter >= consecutive_frames:
                    drowsy = True
                    if not alarm_on:
                        alarm.play(-1)
                        alarm_on = True
            else:
                frame_counter = 0
                if alarm_on:
                    alarm.stop()
                    alarm_on = False
            timings["6. Decision & Alert"] = (time.perf_counter() - t) * 1000

            # Append a copy of the recorded frame dictionary metrics to the data list
            all_timings_data.append(timings.copy())

            draw_overlay(
                frame, ear, drowsy, frame_counter, timings,
                current_threshold, consecutive_frames, is_calibrating
            )
            if display:
                cv2.imshow("Drowsy Driver Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_requested = True
                    break

    finally:
        face_mesh.close()
        if alarm_on:
            alarm.stop()

        # ── AUTOMATIC PERFORMANCE REPORTING & PLOTTING SYSTEM ──────────────────
        if all_timings_data:
            # 1. Process timing statistics from all completed frames.
            stage_names = list(all_timings_data[0])
            mean_timings = {
                stage: sum(row.get(stage, 0.0) for row in all_timings_data) / len(all_timings_data)
                for stage in stage_names
            }
            total_mean_latency = sum(mean_timings.values())
            mean_fps = 1000.0 / total_mean_latency if total_mean_latency > 0 else 0

            # Print an elegant markdown text summary in the console terminal
            print("\n" + "="*55)
            print("         PIPELINE PERFORMANCE BENCHMARK REPORT          ")
            print("="*55)
            for stage, ms in mean_timings.items():
                percentage = (ms / total_mean_latency) * 100
                print(f"{stage:<25}: {ms:>6.2f} ms ({percentage:>5.1f}%)")
            print("-"*55)
            print(f"Total Pipeline Latency   : {total_mean_latency:.2f} ms")
            print(f"Average System Throughput: {mean_fps:.1f} FPS")
            print("="*55)

            performance_rows = append_performance_log(
                mean_timings,
                len(all_timings_data),
                total_mean_latency,
                mean_fps,
                source_label,
            )
            print_performance_log_table(performance_rows, stage_names)

            # 2. Build horizontal bar graph visualization for pipeline bottleneck evaluation
            try:
                stages = list(mean_timings)
                latencies = list(mean_timings.values())
                max_latency = max(latencies) if latencies else 1.0
                row_h = 58
                width = 1200
                height = 115 + row_h * len(stages)
                label_w = 285
                bar_w = 760
                chart = np.full((height, width, 3), 255, dtype=np.uint8)
                colors = [
                    (218, 212, 206), (189, 181, 173), (125, 117, 108),
                    (87, 80, 73), (182, 119, 0), (216, 180, 0)
                ]

                cv2.putText(
                    chart,
                    f"Pipeline Bottleneck Analysis | Avg Total: {total_mean_latency:.1f} ms | ~{mean_fps:.1f} FPS",
                    (32, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (30, 30, 30),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    chart,
                    "Latency (milliseconds)",
                    (label_w, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (90, 90, 90),
                    1,
                    cv2.LINE_AA,
                )

                for i, (stage, latency) in enumerate(zip(stages, latencies)):
                    y = 112 + i * row_h
                    color = colors[i % len(colors)]
                    scaled_w = int((latency / max_latency) * bar_w) if max_latency > 0 else 0
                    cv2.putText(
                        chart, stage, (32, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.56, (45, 45, 45), 1, cv2.LINE_AA
                    )
                    cv2.rectangle(chart, (label_w, y), (label_w + scaled_w, y + 30), color, -1)
                    cv2.putText(
                        chart, f"{latency:.2f} ms", (label_w + scaled_w + 12, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA
                    )

                report_path = os.path.join(os.path.dirname(__file__), "pipeline_performance_benchmark.png")
                cv2.imwrite(report_path, chart)
                print(f"[INFO] Performance chart successfully generated and exported as '{report_path}'!")
            except Exception as e:
                print(f"[WARN] Error compiling system performance chart output: {e}")

    return quit_requested


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    configure_runtime_cache()

    ap = argparse.ArgumentParser(description="Drowsy Driver Detection — CV Group 10")
    ap.add_argument("--input", default=None,
                    help="Video source: '0' = webcam, path to .mp4, or IP-camera URL. "
                         "If omitted, an interactive startup menu opens.")
    ap.add_argument("--camera-scan-limit", type=int, default=5,
                    help="Number of camera indexes to scan when requested (default: 5)")
    ap.add_argument("--ear-threshold", type=float, default=EAR_THRESHOLD,
                    help=f"EAR value below which eyes count as closed (default: {EAR_THRESHOLD})")
    ap.add_argument("--frames", type=int, default=CONSEC_FRAMES,
                    help=f"Consecutive closed-eye frames before alert (default: {CONSEC_FRAMES})")
    ap.add_argument("--no-audio", action="store_true",
                    help="Disable the pygame alarm")
    ap.add_argument("--no-display", action="store_true",
                    help="Run without opening an OpenCV preview window")
    ap.add_argument("--calibrate", action="store_true",
                    help="Measure open-eye EAR first and derive a personal threshold")
    ap.add_argument("--calibration-seconds", type=float, default=5.0,
                    help="Seconds to collect open-eye EAR when --calibrate is used (default: 5)")
    ap.add_argument("--calibration-ratio", type=float, default=0.75,
                    help="Threshold = median open-eye EAR times this ratio (default: 0.75)")
    ap.add_argument("--native-logs", action="store_true",
                    help="Show low-level MediaPipe/TFLite startup diagnostics")
    args = ap.parse_args()
    if args.frames < 1:
        sys.exit("[ERROR] --frames must be at least 1")
    if args.ear_threshold <= 0:
        sys.exit("[ERROR] --ear-threshold must be greater than 0")
    if args.calibration_seconds <= 0:
        sys.exit("[ERROR] --calibration-seconds must be greater than 0")
    if args.calibration_ratio <= 0:
        sys.exit("[ERROR] --calibration-ratio must be greater than 0")
    if args.camera_scan_limit < 1:
        sys.exit("[ERROR] --camera-scan-limit must be at least 1")

    import cv2

    startup_camera_labels = get_platform_camera_labels()
    input_value = args.input
    if input_value is None:
        if args.no_display:
            input_value = "0"
            print("[INFO] No --input provided with --no-display; using camera 0")
        else:
            input_value = choose_startup_source(cv2, scan_limit=args.camera_scan_limit)

    source = parse_video_source(input_value)

    print("[INFO] Initializing alarm...")
    alarm, audio_enabled = init_alarm(enabled=not args.no_audio)

    print("[INFO] Loading Haar cascade...")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        sys.exit("[ERROR] Could not load Haar cascade.")

    print(f"[INFO] Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: '{source}'")

    print("[INFO] Drowsy Driver Detection started | press 'q' to quit")
    if args.calibrate:
        print(f"[INFO] Calibration: keep your eyes open for {args.calibration_seconds:.1f}s")

    quit_requested = False
    try:
        quit_requested = run(
            cap,
            face_cascade,
            alarm,
            ear_threshold=args.ear_threshold,
            consecutive_frames=args.frames,
            display=not args.no_display,
            calibrate=args.calibrate,
            calibration_seconds=args.calibration_seconds,
            calibration_ratio=args.calibration_ratio,
            source_label=describe_video_source(source),
            quiet_native_logs=not args.native_logs,
        )
    except RuntimeError as exc:
        sys.exit(f"[ERROR] {exc}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if audio_enabled:
            import pygame

            pygame.mixer.quit()
        print("[INFO] Stopped.")
        prompt_camera_alias_after_run(source, startup_camera_labels, quit_requested)


if __name__ == "__main__":
    main()
