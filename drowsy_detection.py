import os
import time
import argparse
import sys

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
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "This script requires the MediaPipe legacy solutions API. "
            "Recreate the virtual environment with Python 3.12 and install "
            "requirements.txt so mediapipe==0.10.21 is used."
        )

    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
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
            import pandas as pd
            import matplotlib.pyplot as plt

            # 1. Process timing statistics with Pandas DataFrame
            df = pd.DataFrame(all_timings_data)
            mean_timings = df.mean()
            total_mean_latency = mean_timings.sum()
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
                plt.figure(figsize=(10, 5))
                stages = list(mean_timings.index)
                latencies = list(mean_timings.values)
                
                # Dark slate aesthetic palette
                colors = ['#ced4da', '#adb5bd', '#6c757d', '#495057', '#0077b6', '#00b4d8']
                bars = plt.barh(stages, latencies, color=colors[:len(stages)])
                
                # Append millisecond latency labels directly outside structural data bars
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                             f'{width:.2f} ms', 
                             va='center', ha='left', fontsize=10, fontweight='bold')

                plt.title(f"Pipeline Bottleneck Analysis (Avg Total: {total_mean_latency:.1f} ms | ~{mean_fps:.1f} FPS)", 
                          fontsize=12, fontweight='bold', pad=15)
                plt.xlabel("Latency (milliseconds)", fontsize=10)
                plt.gca().invert_yaxis()  # Keeps Stage 1 execution layout on top row
                plt.tight_layout()
                
                # Export the diagnostic visualization asset directly into project directory
                report_path = "pipeline_performance_benchmark.png"
                plt.savefig(report_path, dpi=300)
                print(f"[INFO] Performance chart successfully generated and exported as '{report_path}'!")
            except Exception as e:
                print(f"[WARN] Error compiling system performance chart output: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    configure_runtime_cache()

    ap = argparse.ArgumentParser(description="Drowsy Driver Detection — CV Group 10")
    ap.add_argument("--input", default="0",
                    help="Video source: '0' = webcam, path to .mp4, or IP-camera URL")
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

    source = parse_video_source(args.input)

    import cv2

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