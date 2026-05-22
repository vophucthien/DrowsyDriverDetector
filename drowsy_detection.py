import os
import time
import argparse
import sys
import importlib.util
import types
from contextlib import contextmanager

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# ── Constants ─────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.22   # below this → eye considered closed/drowsy
CONSEC_FRAMES = 15     # consecutive drowsy frames before alarm (~0.5s @ 30fps)
APP_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

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
    matplotlib_dir = os.path.join(APP_CACHE_DIR, "matplotlib")
    os.makedirs(matplotlib_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")


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


def find_camera_sources(cv2, scan_limit=5):
    sources = []
    for index in range(scan_limit):
        cap = None
        try:
            with suppress_native_stderr():
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    continue

                ok, frame = cap.read()
            label = f"Camera {index}"
            if ok and frame is not None:
                h, w = frame.shape[:2]
                label = f"Camera {index} ({w}x{h})"
            sources.append((str(index), label))
        finally:
            if cap is not None:
                cap.release()

    if not sources:
        sources.append(("0", "Camera 0"))
    return sources


def choose_video_source(cv2, scan_limit=5):
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        print(f"[WARN] Camera picker unavailable: {exc}; using camera 0")
        return "0"

    cameras = find_camera_sources(cv2, scan_limit=scan_limit)
    source_by_label = {label: source for source, label in cameras}
    custom_label = "Other video source..."
    selected = {"source": None}

    root = tk.Tk()
    root.title("Select Camera")
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)

    ttk.Label(root, text="Camera source").grid(
        row=0, column=0, columnspan=2, padx=16, pady=(16, 6), sticky="w"
    )

    source_box = ttk.Combobox(
        root,
        values=[label for _, label in cameras] + [custom_label],
        state="readonly",
        width=36,
    )
    source_box.current(0)
    source_box.grid(row=1, column=0, columnspan=2, padx=16, sticky="ew")

    ttk.Label(root, text="Custom path, URL, or camera index").grid(
        row=2, column=0, columnspan=2, padx=16, pady=(12, 6), sticky="w"
    )

    custom_entry = ttk.Entry(root, width=38)
    custom_entry.insert(0, "0")
    custom_entry.state(["disabled"])
    custom_entry.grid(row=3, column=0, columnspan=2, padx=16, sticky="ew")

    def on_source_change(_event=None):
        if source_box.get() == custom_label:
            custom_entry.state(["!disabled"])
            custom_entry.focus_set()
            custom_entry.selection_range(0, tk.END)
        else:
            custom_entry.state(["disabled"])

    def start():
        if source_box.get() == custom_label:
            value = custom_entry.get().strip()
        else:
            value = source_by_label[source_box.get()]

        if not value:
            messagebox.showerror("Missing Source", "Choose a camera or enter a source.")
            return

        selected["source"] = value
        root.destroy()

    def cancel():
        selected["source"] = None
        root.destroy()

    source_box.bind("<<ComboboxSelected>>", on_source_change)
    root.bind("<Return>", lambda _event: start())
    root.bind("<Escape>", lambda _event: cancel())
    root.protocol("WM_DELETE_WINDOW", cancel)

    ttk.Button(root, text="Cancel", command=cancel).grid(
        row=4, column=0, padx=(16, 6), pady=16, sticky="ew"
    )
    ttk.Button(root, text="Start", command=start).grid(
        row=4, column=1, padx=(6, 16), pady=16, sticky="ew"
    )

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.mainloop()

    if selected["source"] is None:
        sys.exit("[INFO] Camera selection cancelled.")
    return selected["source"]


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
        calibration_seconds=5.0, calibration_ratio=0.75):
    import cv2
    import numpy as np

    print("[INFO] Loading MediaPipe FaceMesh...")
    FaceMesh = load_face_mesh_class()

    try:
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    configure_runtime_cache()

    ap = argparse.ArgumentParser(description="Drowsy Driver Detection — CV Group 10")
    ap.add_argument("--input", default=None,
                    help="Video source: '0' = webcam, path to .mp4, or IP-camera URL. "
                         "If omitted, a camera picker opens.")
    ap.add_argument("--camera-scan-limit", type=int, default=5,
                    help="Number of camera indexes to scan for the picker (default: 5)")
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

    input_value = args.input
    if input_value is None:
        if args.no_display:
            input_value = "0"
            print("[INFO] No --input provided with --no-display; using camera 0")
        else:
            input_value = choose_video_source(cv2, scan_limit=args.camera_scan_limit)

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

    try:
        run(
            cap,
            face_cascade,
            alarm,
            ear_threshold=args.ear_threshold,
            consecutive_frames=args.frames,
            display=not args.no_display,
            calibrate=args.calibrate,
            calibration_seconds=args.calibration_seconds,
            calibration_ratio=args.calibration_ratio,
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


if __name__ == "__main__":
    main()
