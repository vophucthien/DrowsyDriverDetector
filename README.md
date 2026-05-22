# Drowsy Driver Detector

Real-time drowsiness detection using OpenCV, MediaPipe FaceMesh landmarks, and
eye aspect ratio (EAR). The app reads from a webcam, video file, or IP camera,
draws the detection pipeline overlay, and can play an alarm when the driver's
eyes stay closed for too many consecutive frames.

## Setup

Use Python 3.12 for this project. The MediaPipe FaceMesh code uses the legacy
`mediapipe.solutions` API, so Python 3.13 environments can install a package
layout that does not work with this script.

```bash
cd /Users/ben/Documents/GitHub/DrowsyDriverDetector
rm -rf .venv
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the project environment is active:

```bash
which python
python --version
python -m pip show mediapipe
```

The Python path should point inside `.venv`, and MediaPipe should be
`0.10.20`.

## Run

In VS Code, open `drowsy_detection.py` and click **Run Python File**.

The terminal shows two startup options: use the default built-in camera (`0`),
or scan for available cameras and choose from the detected list. This is the
main workflow.

Terminal equivalent:

```bash
source .venv/bin/activate
python drowsy_detection.py
```

The scanner checks camera indexes `0` through `4` by default and shows each
OpenCV index with its detected resolution. After you quit with `q`, the app can
save a local label for the camera index you actually used, using camera names
detected at startup or a custom label. Command-line inputs are still supported
for testing and recorded videos.

Webcam without picker:

```bash
python drowsy_detection.py --input 0
```

Video file:

```bash
python drowsy_detection.py --input test_video.mp4
```

Record a short webcam test video:

```bash
python record_test.py
python drowsy_detection.py --input test_video.mp4
```

IP camera:

```bash
python drowsy_detection.py --input http://camera-url/video
```

Press `q` in the OpenCV window to quit.

When the run exits, the app prints the current run's per-stage performance
report, appends the run to `pipeline_performance_log.csv`, prints a table of
all recorded runs plus the average row, and exports
`pipeline_performance_benchmark.png` with the latest bottleneck chart. The log
table is local; delete that CSV when you want to reset the recorded history.

## Useful Options

Tune the EAR threshold and closed-frame count:

```bash
python drowsy_detection.py --input 0 --ear-threshold 0.22 --frames 15
```

Disable audio:

```bash
python drowsy_detection.py --input 0 --no-audio
```

Run without an OpenCV preview window:

```bash
python drowsy_detection.py --input test_video.mp4 --no-display --no-audio
```

Scan more camera indexes when using startup option 2:

```bash
python drowsy_detection.py --camera-scan-limit 8
```

Show low-level MediaPipe/TFLite diagnostics while debugging:

```bash
python drowsy_detection.py --input 0 --native-logs
```

Calibrate a personal threshold from your open eyes:

```bash
python drowsy_detection.py --input 0 --calibrate
```

During calibration, keep your eyes open and face the camera. The app measures
your median open-eye EAR, then sets:

```text
threshold = median_open_eye_ear * calibration_ratio
```

The default calibration ratio is `0.75`. You can change it:

```bash
python drowsy_detection.py --input 0 --calibrate --calibration-seconds 6 --calibration-ratio 0.72
```

## Common Errors

`ModuleNotFoundError: No module named 'pygame'`

You are probably running global Python instead of the project virtual
environment. In VS Code, run **Python: Select Interpreter** and choose:

```text
/Users/ben/Documents/GitHub/DrowsyDriverDetector/.venv/bin/python
```

Then click **Run Python File** again. Terminal equivalent:

```bash
source .venv/bin/activate
python drowsy_detection.py
```

`module 'mediapipe' has no attribute 'solutions'`

Recreate `.venv` with Python 3.12 and reinstall `requirements.txt`. The project
pins MediaPipe to a tested 0.10.x version because the code uses the legacy
FaceMesh API.

`CoreAudio error`

The app now falls back to a silent alarm if pygame cannot open your audio
device. You can also pass `--no-audio`.

Slow launch or repeated `Matplotlib is building the font cache`

MediaPipe imports Matplotlib internally. The app writes Matplotlib cache files
to `.cache/` inside this project so the cache can be reused instead of rebuilt
every launch. If startup is still very slow, make sure you are running the
latest script from this repo and that `.cache/` is writable.

Low-level MediaPipe/TFLite startup logs are hidden by default so normal runs
show only the app's own status and benchmark output. Use `--native-logs` if you
need those diagnostics while debugging.

Also avoid having both `opencv-python` and `opencv-contrib-python` installed in
the same virtual environment. MediaPipe depends on `opencv-contrib-python`, and
installing both can make `import cv2` slow or unstable. The simplest fix is to
recreate `.venv` from `requirements.txt`.

No GUI appears

The OpenCV window appears only after the webcam/video source opens and
MediaPipe FaceMesh starts successfully. If you see an OpenGL or
`NSOpenGLPixelFormat` error, run the command from a normal macOS Terminal
window instead of a restricted IDE runner. Also make sure you did not pass
`--no-display`.

`[ERROR] Cannot open: '0'`

OpenCV could not access your webcam. Check camera permissions, close other apps
using the camera, or pass a video file with `--input`.

## Pipeline

For each frame:

1. Convert BGR frame to grayscale.
2. Apply Gaussian blur to reduce noise.
3. Detect the face with Haar cascades.
4. Draw an approximate eye-region crop for visualization.
5. Use MediaPipe FaceMesh to get eye landmarks.
6. Compute EAR from six landmarks per eye.
7. Count consecutive frames where EAR is below the threshold.
8. Mark drowsy and trigger the alarm when the counter reaches `--frames`.
