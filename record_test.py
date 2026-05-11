import cv2
import time
import sys

OUTPUT = "test_video.mp4"
DURATION = 20  # seconds

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Cannot open webcam")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print(f"[INFO] Recording {DURATION}s → {OUTPUT}")
print("[INFO] Hướng dẫn: nhìn bình thường 5s → nhắm mắt 3s → mở mắt → lặp lại")
print("[INFO] Nhấn 'q' để dừng sớm")

start = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    elapsed = time.time() - start
    remaining = max(0, DURATION - elapsed)

    cv2.putText(frame, f"REC  {remaining:.1f}s remaining",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, "Mo mat 5s → Nham mat 3s → lap lai",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    writer.write(frame)
    cv2.imshow("Recording — press q to stop", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or elapsed >= DURATION:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"[INFO] Saved: {OUTPUT}")
