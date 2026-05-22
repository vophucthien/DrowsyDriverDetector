import cv2
import time
import sys
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE_DIR, "test_video.mp4")
DURATION = 20  # seconds
OUTPUT_FPS = 30
OPEN_SECONDS = 5
CLOSED_SECONDS = 3


def current_phase(elapsed):
    cycle = OPEN_SECONDS + CLOSED_SECONDS
    position = elapsed % cycle
    if position < OPEN_SECONDS:
        return "open", "KEEP EYES OPEN", OPEN_SECONDS - position, (0, 220, 0)
    return "closed", "CLOSE EYES", cycle - position, (0, 0, 255)


def fit_text_scale(text, max_width, base_scale, thickness):
    scale = base_scale
    while scale > 0.35:
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        if text_width <= max_width:
            return scale
        scale -= 0.05
    return scale


def put_centered_text(frame, text, y, scale, color, thickness, font=cv2.FONT_HERSHEY_DUPLEX):
    h, w = frame.shape[:2]
    scale = fit_text_scale(text, int(w * 0.92), scale, thickness)
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (w - text_width) // 2)
    cv2.putText(frame, text, (x, y + text_height // 2), font, scale, color, thickness, cv2.LINE_AA)


def draw_ready_overlay(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, min(h, 170)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    put_centered_text(frame, "READY TO RECORD", 45, 1.15, (0, 240, 0), 3)
    put_centered_text(frame, "Open 5s -> Close 3s -> Repeat", 92, 0.58, (255, 255, 255), 1,
                      font=cv2.FONT_HERSHEY_SIMPLEX)
    put_centered_text(frame, "Press S or Space to start | Q to cancel", 126, 0.58, (255, 255, 255), 1,
                      font=cv2.FONT_HERSHEY_SIMPLEX)


def draw_recording_overlay(frame, remaining, phase_text, phase_remaining, phase_color):
    h, w = frame.shape[:2]
    panel_bottom = min(h, 220)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    phase_overlay = frame.copy()
    cv2.rectangle(phase_overlay, (0, 62), (w, min(h, 152)), phase_color, -1)
    cv2.addWeighted(phase_overlay, 0.42, frame, 0.58, 0, frame)

    cv2.circle(frame, (24, 28), 8, (0, 0, 255), -1)
    cv2.putText(frame, f"REC  {remaining:.1f}s remaining",
                (42, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    put_centered_text(frame, phase_text, 103, 1.75, (255, 255, 255), 4)
    put_centered_text(frame, f"{phase_remaining:.1f}s left in this phase", 170, 0.72,
                      (255, 255, 255), 2, font=cv2.FONT_HERSHEY_SIMPLEX)
    put_centered_text(frame, "Open 5s -> Close 3s -> Repeat | Q to stop", 202, 0.52,
                      (225, 225, 225), 1, font=cv2.FONT_HERSHEY_SIMPLEX)


def make_tone(pygame, np, freq, duration=0.18, rate=44100, volume=0.45):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    wave = volume * np.sin(2 * np.pi * freq * t)
    audio = (wave * 32767).astype(np.int16)
    stereo = np.column_stack((audio, audio))
    return pygame.sndarray.make_sound(stereo)


def init_audio_cues():
    try:
        import numpy as np
        import pygame

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        return {
            "open": make_tone(pygame, np, 880, duration=0.16),
            "closed": make_tone(pygame, np, 440, duration=0.30),
        }, pygame
    except Exception as exc:
        print(f"[WARN] Audio cues disabled: {exc}")
        return {}, None


def play_cue(cues, phase_key):
    cue = cues.get(phase_key)
    if cue is not None:
        cue.play()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Cannot open webcam")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_output_frames = DURATION * OUTPUT_FPS

print("[INFO] Camera ready.")
print("[INFO] Press 's' or Space when you are ready to start recording.")
print("[INFO] Press 'q' to cancel.")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit("[ERROR] Cannot read from webcam")

    draw_ready_overlay(frame)
    cv2.imshow("Recording setup", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("s"), ord(" ")):
        break
    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit("[INFO] Recording cancelled")

print(f"[INFO] Recording {DURATION}s -> {OUTPUT}")
print("[INFO] Open your eyes for 5 seconds -> Close your eyes for 3 seconds -> Repeat")
print("[INFO] Press 'q' to stop early")

writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), OUTPUT_FPS, (w, h))
if not writer.isOpened():
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(f"[ERROR] Cannot create output video: {OUTPUT}")

audio_cues, pygame_audio = init_audio_cues()
last_phase_key = None
frames_written = 0
stopped_early = False
last_raw_frame = None
start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()
    last_raw_frame = raw_frame
    elapsed = time.time() - start
    remaining = max(0, DURATION - elapsed)
    phase_key, phase_text, phase_remaining, phase_color = current_phase(elapsed)
    if phase_key != last_phase_key:
        play_cue(audio_cues, phase_key)
        last_phase_key = phase_key

    draw_recording_overlay(frame, remaining, phase_text, phase_remaining, phase_color)

    target_frames = min(int(elapsed * OUTPUT_FPS), total_output_frames)
    while frames_written < target_frames:
        writer.write(raw_frame)
        frames_written += 1

    cv2.imshow("Recording - press q to stop", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        stopped_early = True
        break
    if elapsed >= DURATION:
        break

while not stopped_early and last_raw_frame is not None and frames_written < total_output_frames:
    writer.write(last_raw_frame)
    frames_written += 1

cap.release()
writer.release()
if pygame_audio is not None:
    pygame_audio.mixer.quit()
cv2.destroyAllWindows()
saved_duration = frames_written / OUTPUT_FPS
print(f"[INFO] Saved clean video: {OUTPUT}")
print(f"[INFO] Frames written: {frames_written} at {OUTPUT_FPS} FPS ({saved_duration:.1f}s)")
