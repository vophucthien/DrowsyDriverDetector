"""
Driver Drowsiness Detector — Image Input Pipeline
Stages:
  1. Grayscale        → cv2.cvtColor
  2. Filtering        → Gaussian / Bilateral blur
  3. Face Detection   → Haar Cascade (frontal face)
  4. Eye ROI Extract  → Haar eye cascade + geometric crop
  5. Feature Transform→ EAR (Eye Aspect Ratio) via MediaPipe landmarks
"""

import cv2
import numpy as np
import sys
from scipy.spatial import distance as dist


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"

EAR_THRESH = 0.25   # below this → eye considered closed


# ─────────────────────────────────────────────────────────────
# STAGE 1 — Grayscale  (3 channels → 1 channel)
# ─────────────────────────────────────────────────────────────
def to_grayscale(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"[Stage 1] Grayscale shape: {gray.shape}")
    return gray


# ─────────────────────────────────────────────────────────────
# STAGE 2 — Filtering  (denoise)
# ─────────────────────────────────────────────────────────────
def filter_frame(gray: np.ndarray, method: str = "bilateral") -> np.ndarray:
    if method == "gaussian":
        filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    else:  # bilateral preserves edges better for face/eye detection
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    print(f"[Stage 2] Filtering method: {method}")
    return filtered


# ─────────────────────────────────────────────────────────────
# STAGE 3 — Face Detection  (Haar Cascade frontal face)
# ─────────────────────────────────────────────────────────────
def detect_faces(gray: np.ndarray, frame: np.ndarray):
    face_cascade    = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    # Try frontal first
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    # If frontal fails, relax parameters (catches tilted/drowsy faces)
    if len(faces) == 0:
        print("[Stage 3] Frontal strict failed, trying relaxed...")
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 60),
        )

    # If still nothing, try profile cascade (tilted head)
    if len(faces) == 0:
        print("[Stage 3] Frontal relaxed failed, trying profile...")
        faces = face_cascade.detectMultiScale(
            cv2.flip(gray, 1),  # try mirrored too
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 60),
        )
        # Mirror coordinates back
        if len(faces) > 0:
            h_img, w_img = gray.shape
            faces = np.array([(w_img - x - w, y, w, h) for (x, y, w, h) in faces])

    print(f"[Stage 3] Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 100), 2)
        cv2.putText(frame, "Face", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)

    return faces, frame


# ─────────────────────────────────────────────────────────────
# STAGE 4 — Eye ROI Extraction  (Haar eye cascade + geometric crop)
# ─────────────────────────────────────────────────────────────
def apply_nms(eyes, overlap_thresh=0.3):
    """Suppress overlapping eye boxes, keep the largest (highest area)."""
    if len(eyes) == 0:
        return []
    boxes  = np.array([[x, y, x+w, y+h] for (x,y,w,h) in eyes], dtype=float)
    areas  = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    order  = areas.argsort()[::-1]
    keep   = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < overlap_thresh]
    return [eyes[i] for i in keep]


def extract_eye_rois(gray: np.ndarray, frame: np.ndarray, faces):
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    all_eye_rois = []

    for (fx, fy, fw, fh) in faces:
        # Geometric crop: upper 55% of face only
        eye_region_gray = gray[fy : fy + int(fh * 0.55), fx : fx + fw]

        eyes_raw = eye_cascade.detectMultiScale(
            eye_region_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )

        print(f"[Stage 4] Raw detections: {len(eyes_raw)}")

        # NMS → remove overlapping duplicates
        eyes = apply_nms(list(eyes_raw) if len(eyes_raw) > 0 else [])

        # Ratio filter: keep only boxes whose size is plausible for a real eye
        # Eye width  should be 10-40% of face width
        # Eye height should be  5-25% of face height
        MIN_EYE_W, MAX_EYE_W = 0.10 * fw, 0.40 * fw
        MIN_EYE_H, MAX_EYE_H = 0.05 * fh, 0.25 * fh
        eyes = [(x,y,w,h) for (x,y,w,h) in eyes
                if MIN_EYE_W <= w <= MAX_EYE_W and MIN_EYE_H <= h <= MAX_EYE_H]
        print(f"[Stage 4] After ratio filter: {len(eyes)}")

        # Split face into left/right halves, pick best candidate from each
        mid_x = fw // 2
        left_candidates  = [(x,y,w,h) for (x,y,w,h) in eyes if (x + w//2) < mid_x]
        right_candidates = [(x,y,w,h) for (x,y,w,h) in eyes if (x + w//2) >= mid_x]

        best_left  = max(left_candidates,  key=lambda e: e[2]*e[3]) if left_candidates  else None
        best_right = max(right_candidates, key=lambda e: e[2]*e[3]) if right_candidates else None

        eyes = [e for e in [best_left, best_right] if e is not None]
        eyes = sorted(eyes, key=lambda e: e[0])  # left-to-right

        print(f"[Stage 4] Eyes after NMS + filter: {len(eyes)}")

        for (ex, ey, ew, eh) in eyes:
            abs_x, abs_y = fx + ex, fy + ey
            cv2.rectangle(frame,
                          (abs_x, abs_y),
                          (abs_x + ew, abs_y + eh),
                          (255, 180, 0), 2)
            roi_gray = eye_region_gray[ey : ey + eh, ex : ex + ew]
            all_eye_rois.append((roi_gray, (abs_x, abs_y, ew, eh)))

    return all_eye_rois, frame


# ─────────────────────────────────────────────────────────────
# STAGE 5 — Pupil detection via HoughCircles
# ─────────────────────────────────────────────────────────────
def detect_pupil(roi_gray: np.ndarray, roi_w: int, roi_h: int):
    """
    Detect pupil in eye ROI using HoughCircles.

    Steps:
    1. Resize ROI to fixed size for consistent detection
    2. Equalise histogram → boosts contrast of dark iris/pupil
    3. Blur to reduce noise (HoughCircles needs smooth input)
    4. HoughCircles to find circular pupil
    5. Filter by radius: pupil should be 10–40% of ROI width

    Returns: (circle, roi_display) where circle is (cx, cy, r) or None
    """
    if roi_gray is None or roi_gray.size == 0:
        return None, roi_gray

    # Step 1: resize to fixed width for consistent min/maxRadius
    scale  = 80 / roi_w
    rw     = 80
    rh     = max(1, int(roi_h * scale))
    roi    = cv2.resize(roi_gray, (rw, rh))

    # Step 2: histogram equalisation — improves contrast in varied lighting
    roi = cv2.equalizeHist(roi)

    # Step 3: Gaussian blur — required before HoughCircles
    roi_blur = cv2.GaussianBlur(roi, (7, 7), 1.5)

    # Step 4: HoughCircles
    # minRadius / maxRadius tuned to 10–40% of ROI width
    min_r = max(3,  int(rw * 0.10))
    max_r = max(min_r + 1, int(rw * 0.40))

    circles = cv2.HoughCircles(
        roi_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=rw // 2,      # only one pupil per eye
        param1=50,            # Canny edge threshold
        param2=15,            # accumulator threshold (lower = more detections)
        minRadius=min_r,
        maxRadius=max_r
    )

    # Build a colour display ROI for visualisation
    roi_display = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # Pick the circle closest to the ROI centre
        cx_roi, cy_roi = rw // 2, rh // 2
        best = min(circles, key=lambda c: abs(c[0]-cx_roi) + abs(c[1]-cy_roi))
        cx, cy, r = best
        cv2.circle(roi_display, (cx, cy), r,  (0, 255, 0), 1)
        cv2.circle(roi_display, (cx, cy), 2,  (0, 255, 0), -1)
        # Scale circle coords back to original ROI size
        orig_cx = int(cx / scale)
        orig_cy = int(cy / scale)
        orig_r  = int(r  / scale)
        return (orig_cx, orig_cy, orig_r), roi_display

    return None, roi_display


def feature_transform(eye_rois: list, frame: np.ndarray) -> list:
    """Detect pupil in each eye ROI and annotate the frame."""
    results_list = []
    eye_labels = ["Left", "Right"]

    for idx, (roi_gray, (ax, ay, aw, ah)) in enumerate(eye_rois):
        label = eye_labels[idx] if idx < 2 else f"Eye{idx+1}"

        circle, roi_display = detect_pupil(roi_gray, aw, ah)

        state = "OPEN" if circle is not None else "CLOSED"
        color = (0, 220, 0) if state == "OPEN" else (0, 0, 255)

        print(f"[Stage 5] {label} eye | Pupil: {'detected' if circle else 'not found'} → {state}")

        # Draw pupil circle on main frame if found
        if circle is not None:
            pcx, pcy, pr = circle
            cv2.circle(frame, (ax + pcx, ay + pcy), pr, (0, 255, 0), 1)
            cv2.circle(frame, (ax + pcx, ay + pcy), 2,  (0, 255, 0), -1)

        # Coloured border around eye box
        cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), color, 2)

        # Label
        cv2.putText(frame, f"{label}: {state}",
                    (ax, ay - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        results_list.append({"eye": label, "state": state, "pupil": circle})

    return results_list


# ─────────────────────────────────────────────────────────────
# DROWSINESS VERDICT
# ─────────────────────────────────────────────────────────────
def drowsiness_verdict(eye_results: list) -> str:
    if not eye_results:
        return "NO EYES DETECTED"

    closed = sum(1 for r in eye_results if r["state"] == "CLOSED")
    verdict = "DROWSY" if closed >= 1 else "AWAKE"
    print(f"\n[Result] Closed eyes: {closed}/{len(eye_results)} → {verdict}")
    return verdict


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    print(f"\n{'='*55}")
    print(f"  Driver Drowsiness Detector — Image: {image_path}")
    print(f"{'='*55}\n")

    # Stage 1
    gray = to_grayscale(frame)

    # Stage 2
    filtered = filter_frame(gray, method="bilateral")

    # Stage 3
    faces, frame = detect_faces(filtered, frame)

    if len(faces) == 0:
        print("[!] No face found. Exiting.")
        cv2.imshow("Result", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Stage 4
    eye_rois, frame = extract_eye_rois(filtered, frame, faces)

    # Stage 5
    eye_results = feature_transform(eye_rois, frame)

    # Verdict
    verdict = drowsiness_verdict(eye_results)
    color = (0, 0, 255) if "DROWSY" in verdict else (0, 200, 50)
    cv2.putText(frame, verdict, (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)

    # Save + show
    out_path = "drowsiness_result.jpg"
    cv2.imwrite(out_path, frame)
    print(f"\n[✓] Output saved → {out_path}")

    cv2.imshow("Driver Drowsiness Detector", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "driver.jpg"
    run_pipeline(img)
