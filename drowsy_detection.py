import cv2
import numpy as np
import pygame
import time
import argparse
import sys
from scipy.spatial import distance as dist

# ── Constants ─────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.22   # below this → eye considered closed/drowsy
CONSEC_FRAMES = 15     # consecutive drowsy frames before alarm (~0.5s @ 30fps)

# Mediapipe FaceMesh 6-point EAR indices (Soukupová & Čech, 2016)
MP_LEFT_EYE  = [33,  160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ── Helpers ───────────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)


def make_beep(freq=880, duration=0.5, rate=44100) -> pygame.mixer.Sound:
    t = np.linspace(0, duration, int(rate * duration), False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))


# ── Overlay ───────────────────────────────────────────────────────────────────

def draw_overlay(frame, ear, drowsy, frame_counter, timings):
    h, w = frame.shape[:2]

    if drowsy:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.putText(frame, "! DROWSY  WAKE UP !",
                    (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3)

    status_color = (0, 0, 255) if drowsy else (0, 220, 0)
    cv2.putText(frame, f"Status: {'DROWSY' if drowsy else 'Awake'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
    cv2.putText(frame,
                f"EAR: {ear:.3f}   closed frames: {frame_counter}/{CONSEC_FRAMES}",
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

def run(cap, face_cascade, alarm):
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_counter, alarm_on = 0, False

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
            left_eye  = pts[MP_LEFT_EYE]
            right_eye = pts[MP_RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            for p in np.vstack([left_eye, right_eye]):
                cv2.circle(frame, tuple(p), 2, (0, 255, 100), -1)
        timings["5. Landmark + EAR"] = (time.perf_counter() - t) * 1000

        # ⑥ Decision & alert
        t = time.perf_counter()
        drowsy = False
        if ear > 0 and ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
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

        draw_overlay(frame, ear, drowsy, frame_counter, timings)
        cv2.imshow("Drowsy Driver Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    if alarm_on:
        alarm.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Drowsy Driver Detection — CV Group 10")
    ap.add_argument("--input", default="0",
                    help="Video source: '0' = webcam, path to .mp4, or IP-camera URL")
    args = ap.parse_args()

    try:
        source = int(args.input)
    except ValueError:
        source = args.input

    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    alarm = make_beep()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        sys.exit("[ERROR] Could not load Haar cascade.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: '{source}'")

    print("[INFO] Drowsy Driver Detection started | press 'q' to quit")

    try:
        run(cap, face_cascade, alarm)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
