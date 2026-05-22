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
FACE_GATE_GRACE_FRAMES = 5  # keep FaceMesh running briefly after Haar loses face
FACEMESH_ROI_MARGIN_RATIO = 0.20  # expand Haar face box before FaceMesh crop
APP_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CAMERA_ALIASES_FILE = os.path.join(os.path.dirname(__file__), ".camera_aliases.json")
PERFORMANCE_LOG_FILE = os.path.join(os.path.dirname(__file__), "pipeline_performance_log.csv")
FRAME_LOG_FILE = os.path.join(os.path.dirname(__file__), "frame_state_log.csv")
ALARM_EVENT_LOG_FILE = os.path.join(os.path.dirname(__file__), "alarm_events_log.csv")
CONTROLLED_EVAL_FILE = os.path.join(os.path.dirname(__file__), "controlled_evaluation_summary.json")
CONTROLLED_OPEN_SECONDS = 5.0
CONTROLLED_CLOSED_SECONDS = 3.0
CONTROLLED_EXPECTED_SECONDS = 20.0

# MediaPipe FaceMesh 6-point EAR indices (Soukupova and Cech, 2016)
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


def expand_box(box, frame_width, frame_height, margin_ratio):
    x, y, w, h = [int(v) for v in box]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame_width, x + w + margin_x)
    y2 = min(frame_height, y + h + margin_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bool_field(value):
    return "1" if value else "0"


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value or value < 0:
        return default
    return value


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


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


def write_csv_rows(path, fieldnames, rows):
    if not path or not rows:
        return
    try:
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"[WARN] Could not write CSV log '{path}': {exc}")


def append_csv_rows(path, fieldnames, rows):
    if not path or not rows:
        return
    try:
        ensure_parent_dir(path)
        needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"[WARN] Could not append CSV log '{path}': {exc}")


def write_json(path, data):
    if not path:
        return
    try:
        ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as exc:
        print(f"[WARN] Could not write JSON file '{path}': {exc}")


def ordered_fieldnames(rows):
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


def controlled_ground_truth(time_seconds, open_seconds, closed_seconds):
    cycle_seconds = open_seconds + closed_seconds
    if cycle_seconds <= 0:
        return "open"
    cycle_position = time_seconds % cycle_seconds
    return "closed" if cycle_position >= open_seconds else "open"


def calculate_controlled_evaluation(frame_rows, open_seconds, closed_seconds):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    closed_segment_starts = {}
    detected_segment_times = {}
    cycle_seconds = open_seconds + closed_seconds

    for row in frame_rows:
        try:
            time_seconds = float(row.get("time_seconds", ""))
        except (TypeError, ValueError):
            continue

        truth = controlled_ground_truth(time_seconds, open_seconds, closed_seconds)
        predicted_closed = row.get("drowsy") == "1"
        actual_closed = truth == "closed"

        if actual_closed and predicted_closed:
            counts["tp"] += 1
        elif actual_closed:
            counts["fn"] += 1
        elif predicted_closed:
            counts["fp"] += 1
        else:
            counts["tn"] += 1

        if actual_closed and cycle_seconds > 0:
            segment_index = int(time_seconds // cycle_seconds)
            segment_start = segment_index * cycle_seconds + open_seconds
            closed_segment_starts.setdefault(str(segment_index), segment_start)
            if predicted_closed:
                detected_segment_times.setdefault(str(segment_index), time_seconds)

    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = (2.0 * precision * recall / f1_denominator) if f1_denominator else 0.0

    latencies = []
    for segment_index, segment_start in closed_segment_starts.items():
        detected_at = detected_segment_times.get(segment_index)
        if detected_at is not None:
            latencies.append(detected_at - segment_start)

    return {
        "schedule": {
            "open_seconds": open_seconds,
            "closed_seconds": closed_seconds,
            "cycle_seconds": cycle_seconds,
        },
        "frames_evaluated": sum(counts.values()),
        "counts": counts,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "detection_latency_seconds": {
            "detected_closed_segments": len(latencies),
            "total_closed_segments": len(closed_segment_starts),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "values": latencies,
        },
    }


def print_controlled_evaluation(summary, output_path=None):
    counts = summary["counts"]
    metrics = summary["metrics"]
    latency = summary["detection_latency_seconds"]

    print("\n" + "=" * 55)
    print("         CONTROLLED CLIP EVALUATION REPORT          ")
    print("=" * 55)
    print(
        f"Schedule: open {summary['schedule']['open_seconds']:.2f}s, "
        f"closed {summary['schedule']['closed_seconds']:.2f}s"
    )
    print(f"Frames evaluated: {summary['frames_evaluated']}")
    print(
        f"TP: {counts['tp']}  FP: {counts['fp']}  "
        f"TN: {counts['tn']}  FN: {counts['fn']}"
    )
    print(
        f"Precision: {metrics['precision']:.3f}  "
        f"Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}"
    )
    print(
        "Closed segments detected: "
        f"{latency['detected_closed_segments']}/{latency['total_closed_segments']}"
    )
    if latency["mean"] is not None:
        print(
            f"Detection latency: mean {latency['mean']:.3f}s, "
            f"min {latency['min']:.3f}s, max {latency['max']:.3f}s"
        )
    if output_path:
        print(f"Summary JSON: {output_path}")
    print("=" * 55)


def warn_if_controlled_input_is_short(cap, expected_seconds):
    import cv2

    fps = safe_float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0 or expected_seconds <= 0:
        return

    duration = frame_count / fps
    print(
        f"[INFO] Controlled input metadata: {frame_count:.0f} frames "
        f"at {fps:.2f} FPS ({duration:.2f}s)"
    )
    if duration < expected_seconds * 0.95:
        print(
            f"[WARN] Controlled input is shorter than the expected "
            f"{expected_seconds:.1f}s recording. Re-run record_test.py and wait "
            "for it to finish; the full default clip should be 600 frames at 30 FPS."
        )


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
        "Eye ROI Draw": "ROI",
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
                 calibrating=False, calibration_elapsed=0.0, calibration_seconds=0.0):
    import cv2

    h, w = frame.shape[:2]

    if drowsy:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        alert = "! DROWSY  WAKE UP !"
        (alert_w, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_DUPLEX, 1.28, 3)
        cv2.putText(frame, alert,
                    (w // 2 - alert_w // 2, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.28, (0, 0, 255), 3)

    status_color = (0, 0, 255) if drowsy else (0, 220, 0)
    status = "Calibrating" if calibrating else ("DROWSY" if drowsy else "Awake")
    status_color = (0, 200, 255) if calibrating else status_color
    cv2.putText(frame, f"Status: {status}",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.98, status_color, 2)
    cv2.putText(frame,
                f"EAR: {ear:.3f}   threshold: {threshold:.3f}   closed frames: {frame_counter}/{frame_limit}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)

    y, total = 102, 0.0
    for label, ms in timings.items():
        cv2.putText(frame, f"{label}: {ms:.1f} ms",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1)
        y += 22
        total += ms

    fps = 1000.0 / total if total > 0 else 0.0
    cv2.putText(frame, f"Pipeline: {total:.1f} ms  (~{fps:.1f} FPS)",
                (10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 50), 1)

    if calibrating:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        title = "Calibrating"
        remaining = max(0.0, calibration_seconds - calibration_elapsed)
        details = f"Keep eyes open ({remaining:.1f}s)"
        cancel = "C/Esc: cancel and use default EAR threshold"

        title_scale = 1.58
        title_thickness = 3
        (title_w, _), _ = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_DUPLEX, title_scale, title_thickness
        )
        detail_scale = 0.72
        cancel_scale = 0.66
        (detail_w, _), _ = cv2.getTextSize(details, cv2.FONT_HERSHEY_SIMPLEX, detail_scale, 2)
        (cancel_w, _), _ = cv2.getTextSize(cancel, cv2.FONT_HERSHEY_SIMPLEX, cancel_scale, 1)

        cx = w // 2
        cy = h // 2
        cv2.putText(frame, title, (cx - title_w // 2, cy - 34),
                    cv2.FONT_HERSHEY_DUPLEX, title_scale, (0, 220, 255), title_thickness)
        cv2.putText(frame, details, (cx - detail_w // 2, cy + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, detail_scale, (255, 255, 255), 2)
        cv2.putText(frame, cancel, (cx - cancel_w // 2, cy + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, cancel_scale, (210, 210, 210), 1)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(cap, face_cascade, alarm, ear_threshold=EAR_THRESHOLD,
        consecutive_frames=CONSEC_FRAMES, display=True, calibrate=True,
        calibration_seconds=5.0, calibration_ratio=0.75,
        face_gate_grace_frames=FACE_GATE_GRACE_FRAMES,
        facemesh_roi_margin=FACEMESH_ROI_MARGIN_RATIO,
        source_label="Unknown source", quiet_native_logs=True,
        source_fps=0.0, frame_log_path=None, alarm_event_log_path=None,
        eval_controlled=False,
        eval_open_seconds=CONTROLLED_OPEN_SECONDS,
        eval_closed_seconds=CONTROLLED_CLOSED_SECONDS,
        eval_output_path=None):
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
    haar_face_seen = False
    haar_miss_frames = 0
    last_face_box = None
    source_fps = safe_float(source_fps)
    if eval_controlled and not frame_log_path:
        frame_log_path = FRAME_LOG_FILE
    frame_rows = []
    alarm_event_rows = []
    frame_index = 0
    run_started_at = time.perf_counter()
    alarm_event_fieldnames = [
        "timestamp",
        "source",
        "event",
        "frame",
        "time_seconds",
        "ear",
        "threshold",
        "closed_frames",
    ]

    def add_alarm_event(event, event_frame, event_time, event_ear):
        alarm_event_rows.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_label,
            "event": event,
            "frame": str(event_frame),
            "time_seconds": f"{event_time:.6f}",
            "ear": f"{event_ear:.6f}",
            "threshold": f"{current_threshold:.6f}",
            "closed_frames": str(frame_counter),
        })

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame.copy()
            fh, fw = frame.shape[:2]
            timings = {}
            wall_seconds = time.perf_counter() - run_started_at
            time_seconds = frame_index / source_fps if source_fps > 0 else wall_seconds

            # ① Grayscale
            t = time.perf_counter()
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
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
            face_box = None
            if len(faces):
                face_box = max(faces, key=lambda r: r[2] * r[3])
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                haar_face_seen = True
                haar_miss_frames = 0
                last_face_box = face_box
            elif haar_face_seen:
                haar_miss_frames += 1
            should_run_facemesh = (
                face_box is not None
                or (haar_face_seen and haar_miss_frames <= face_gate_grace_frames)
            )
            active_face_box = face_box if face_box is not None else last_face_box
            facemesh_roi = None
            reused_face_box = face_box is None and should_run_facemesh and active_face_box is not None
            if should_run_facemesh and active_face_box is not None:
                facemesh_roi = expand_box(active_face_box, fw, fh, facemesh_roi_margin)
            timings["3. Face Detection"] = (time.perf_counter() - t) * 1000

            # ④ Eye ROI draw (geometric visualization)
            t = time.perf_counter()
            if facemesh_roi is not None:
                x1, y1, x2, y2 = facemesh_roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 255), 1)
            if face_box is not None:
                x, y, w, h = face_box
                ey1 = y + int(h * 0.30)
                ey2 = y + int(h * 0.55)
                cv2.rectangle(frame, (x, ey1), (x + w, ey2), (0, 165, 255), 1)
            timings["4. Eye ROI Draw"] = (time.perf_counter() - t) * 1000

            # ⑤ MediaPipe FaceMesh and EAR
            t = time.perf_counter()
            ear = 0.0
            landmarks_found = False
            if facemesh_roi is not None:
                x1, y1, x2, y2 = facemesh_roi
                roi_bgr = raw_frame[y1:y2, x1:x2]
                roi_h, roi_w = roi_bgr.shape[:2]
                rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                if quiet_frames_remaining > 0:
                    with suppress_native_stderr():
                        result = face_mesh.process(rgb)
                    quiet_frames_remaining -= 1
                else:
                    result = face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    landmarks_found = True
                    lms = result.multi_face_landmarks[0].landmark
                    pts = np.array([
                        [int(lm.x * roi_w) + x1, int(lm.y * roi_h) + y1]
                        for lm in lms
                    ])
                    left_eye = pts[MP_LEFT_EYE]
                    right_eye = pts[MP_RIGHT_EYE]
                    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                    for p in np.vstack([left_eye, right_eye]):
                        cv2.circle(frame, tuple(p), 2, (0, 255, 100), -1)
            timings["5. Landmark + EAR"] = (time.perf_counter() - t) * 1000

            # ⑥ Decision & alert
            t = time.perf_counter()
            drowsy = False
            calibration_elapsed = (
                time.perf_counter() - calibration_start if is_calibrating else 0.0
            )
            if is_calibrating:
                if ear > 0:
                    calibration_ears.append(ear)
                if calibration_elapsed >= calibration_seconds:
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
                        add_alarm_event("alarm_start", frame_index, time_seconds, ear)
            else:
                frame_counter = 0
                if alarm_on:
                    alarm.stop()
                    alarm_on = False
                    add_alarm_event("alarm_stop", frame_index, time_seconds, ear)
            timings["6. Decision & Alert"] = (time.perf_counter() - t) * 1000

            # Append a copy of the recorded frame dictionary metrics to the data list
            all_timings_data.append(timings.copy())
            if frame_log_path or eval_controlled:
                frame_row = {
                    "source": source_label,
                    "frame": str(frame_index),
                    "time_seconds": f"{time_seconds:.6f}",
                    "wall_seconds": f"{wall_seconds:.6f}",
                    "ear": f"{ear:.6f}",
                    "threshold": f"{current_threshold:.6f}",
                    "closed_frames": str(frame_counter),
                    "drowsy": bool_field(drowsy),
                    "alarm_on": bool_field(alarm_on),
                    "calibrating": bool_field(is_calibrating),
                    "face_detected": bool_field(face_box is not None),
                    "face_gate_active": bool_field(facemesh_roi is not None),
                    "haar_box_reused": bool_field(reused_face_box),
                    "facemesh_ran": bool_field(facemesh_roi is not None),
                    "landmarks_found": bool_field(landmarks_found),
                    "haar_miss_frames": str(haar_miss_frames),
                }
                if facemesh_roi is not None:
                    x1, y1, x2, y2 = facemesh_roi
                    frame_row.update({
                        "roi_x1": str(x1),
                        "roi_y1": str(y1),
                        "roi_x2": str(x2),
                        "roi_y2": str(y2),
                    })
                else:
                    frame_row.update({
                        "roi_x1": "",
                        "roi_y1": "",
                        "roi_x2": "",
                        "roi_y2": "",
                    })
                for stage, ms in timings.items():
                    frame_row[performance_stage_column(stage)] = f"{ms:.6f}"
                frame_rows.append(frame_row)
            frame_index += 1

            draw_overlay(
                frame, ear, drowsy, frame_counter, timings,
                current_threshold, consecutive_frames, is_calibrating,
                calibration_elapsed, calibration_seconds
            )
            if display:
                cv2.imshow("Drowsy Driver Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    quit_requested = True
                    break
                if is_calibrating and key in (ord("c"), 27):
                    is_calibrating = False
                    calibration_ears.clear()
                    current_threshold = ear_threshold
                    frame_counter = 0
                    print(
                        "[INFO] Calibration cancelled; "
                        f"using default threshold {current_threshold:.3f}"
                    )

    finally:
        face_mesh.close()
        if alarm_on:
            alarm.stop()
            shutdown_time = frame_index / source_fps if source_fps > 0 else time.perf_counter() - run_started_at
            add_alarm_event("alarm_stop_shutdown", frame_index, shutdown_time, 0.0)
            alarm_on = False

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

        if frame_rows and frame_log_path:
            write_csv_rows(frame_log_path, ordered_fieldnames(frame_rows), frame_rows)
            print(f"[INFO] Frame state log exported as '{frame_log_path}'")

        if alarm_event_rows and alarm_event_log_path:
            append_csv_rows(alarm_event_log_path, alarm_event_fieldnames, alarm_event_rows)
            print(f"[INFO] Alarm event log appended as '{alarm_event_log_path}'")

        if eval_controlled and frame_rows:
            summary = calculate_controlled_evaluation(
                frame_rows,
                eval_open_seconds,
                eval_closed_seconds,
            )
            if eval_output_path:
                write_json(eval_output_path, summary)
            print_controlled_evaluation(summary, eval_output_path)

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
    ap.add_argument("--face-gate-grace", type=int, default=FACE_GATE_GRACE_FRAMES,
                    help="Frames to keep running FaceMesh after Haar loses the face "
                         f"(default: {FACE_GATE_GRACE_FRAMES})")
    ap.add_argument("--facemesh-roi-margin", type=float, default=FACEMESH_ROI_MARGIN_RATIO,
                    help="Haar face-box margin ratio for the FaceMesh processing crop "
                         f"(default: {FACEMESH_ROI_MARGIN_RATIO})")
    ap.add_argument("--no-audio", action="store_true",
                    help="Disable the pygame alarm")
    ap.add_argument("--no-display", action="store_true",
                    help="Run without opening an OpenCV preview window")
    calibration_group = ap.add_mutually_exclusive_group()
    calibration_group.add_argument("--calibrate", dest="calibrate", action="store_true", default=True,
                                   help="Measure open-eye EAR first and derive a personal threshold (default)")
    calibration_group.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                                   help="Skip startup calibration and use the default EAR threshold")
    ap.add_argument("--calibration-seconds", type=float, default=5.0,
                    help="Seconds to collect open-eye EAR during calibration (default: 5)")
    ap.add_argument("--calibration-ratio", type=float, default=0.75,
                    help="Threshold = median open-eye EAR times this ratio (default: 0.75)")
    ap.add_argument("--native-logs", action="store_true",
                    help="Show low-level MediaPipe/TFLite startup diagnostics")
    ap.add_argument("--frame-log", default=None,
                    help="Write per-frame detector state to this CSV path")
    ap.add_argument("--alarm-log", default=ALARM_EVENT_LOG_FILE,
                    help=f"Append alarm start/stop events to this CSV path (default: {ALARM_EVENT_LOG_FILE})")
    ap.add_argument("--no-alarm-log", action="store_true",
                    help="Disable alarm event CSV logging")
    ap.add_argument("--eval-controlled", action="store_true",
                    help="Score the run against the record_test.py open/closed schedule")
    ap.add_argument("--eval-open-seconds", type=float, default=CONTROLLED_OPEN_SECONDS,
                    help=f"Open-eye seconds per controlled cycle (default: {CONTROLLED_OPEN_SECONDS})")
    ap.add_argument("--eval-closed-seconds", type=float, default=CONTROLLED_CLOSED_SECONDS,
                    help=f"Closed-eye seconds per controlled cycle (default: {CONTROLLED_CLOSED_SECONDS})")
    ap.add_argument("--eval-output", default=CONTROLLED_EVAL_FILE,
                    help=f"Write controlled evaluation summary JSON here (default: {CONTROLLED_EVAL_FILE})")
    ap.add_argument("--eval-expected-seconds", type=float, default=CONTROLLED_EXPECTED_SECONDS,
                    help=f"Expected controlled recording duration (default: {CONTROLLED_EXPECTED_SECONDS})")
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
    if args.face_gate_grace < 0:
        sys.exit("[ERROR] --face-gate-grace must be at least 0")
    if args.facemesh_roi_margin < 0:
        sys.exit("[ERROR] --facemesh-roi-margin must be at least 0")
    if args.eval_open_seconds <= 0:
        sys.exit("[ERROR] --eval-open-seconds must be greater than 0")
    if args.eval_closed_seconds <= 0:
        sys.exit("[ERROR] --eval-closed-seconds must be greater than 0")
    if args.eval_expected_seconds <= 0:
        sys.exit("[ERROR] --eval-expected-seconds must be greater than 0")

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
    source_fps = safe_float(cap.get(cv2.CAP_PROP_FPS))
    frame_log_path = args.frame_log
    if args.eval_controlled and not frame_log_path:
        frame_log_path = FRAME_LOG_FILE
    alarm_event_log_path = None if args.no_alarm_log else args.alarm_log
    if args.eval_controlled:
        warn_if_controlled_input_is_short(cap, args.eval_expected_seconds)

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
            face_gate_grace_frames=args.face_gate_grace,
            facemesh_roi_margin=args.facemesh_roi_margin,
            source_label=describe_video_source(source),
            quiet_native_logs=not args.native_logs,
            source_fps=source_fps,
            frame_log_path=frame_log_path,
            alarm_event_log_path=alarm_event_log_path,
            eval_controlled=args.eval_controlled,
            eval_open_seconds=args.eval_open_seconds,
            eval_closed_seconds=args.eval_closed_seconds,
            eval_output_path=args.eval_output,
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
