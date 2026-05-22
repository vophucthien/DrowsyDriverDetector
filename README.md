# Drowsy Driver Detector

Real-time drowsiness detection using OpenCV, MediaPipe FaceMesh landmarks, and
eye aspect ratio (EAR). The app reads from a webcam, video file, or IP camera,
draws the detection pipeline overlay, and can play an alarm when the driver's
eyes stay closed for too many consecutive frames.

## Setup

Use Python 3.12 for this project. The MediaPipe FaceMesh code uses the legacy
FaceMesh solution implementation through `mediapipe.python.solutions`, so
Python 3.13 environments can install a package layout that does not work with
this script.

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

In the scan menu, choose a numbered camera, `R` to rename/save local camera
labels, or `C` to enter a custom path, URL, or camera index.

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

`record_test.py` opens a preview first and waits until you press `s` or Space.
It then records camera `0` to a clean `test_video.mp4` for 20 seconds at a
fixed 30 FPS timeline. The preview shows a large open-eye / closed-eye phase
overlay with audio cues, but that overlay is not burned into the saved MP4. It
overwrites the previous test video and can be stopped early with `q`.

IP camera:

```bash
python drowsy_detection.py --input http://camera-url/video
```

Calibration starts automatically when the OpenCV window opens. Keep your eyes
open and face the camera while the `Calibrating` overlay is visible. Press `c`
or `Esc` during calibration to cancel it and use the default EAR threshold.
Press `q` in the OpenCV window to quit.

In `--no-display` mode, there is no OpenCV window, overlay, or keyboard cancel
handler. Use `--no-calibrate` for headless runs unless the first calibration
seconds of the input show open eyes.

When the run exits, the app prints the current run's per-stage performance
report, appends the run to `pipeline_performance_log.csv`, prints a table of
all recorded runs plus the average row, and exports
`pipeline_performance_benchmark.png` with the latest bottleneck chart. The log
table is local; delete that CSV when you want to reset the recorded history.
Alarm start/stop transitions are appended to `alarm_events_log.csv` when they
occur.

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
python drowsy_detection.py --input test_video.mp4 --no-display --no-audio --no-calibrate
```

Export a per-frame detector-state CSV:

```bash
python drowsy_detection.py --input test_video.mp4 --frame-log frame_state_log.csv
```

Score the 5-second-open / 3-second-closed recording made by `record_test.py`:

```bash
python drowsy_detection.py --input test_video.mp4 --no-display --no-audio --no-calibrate --eval-controlled
```

This writes `frame_state_log.csv` and `controlled_evaluation_summary.json`,
then prints TP/FP/TN/FN, precision, recall, F1, and alarm latency.

Scan more camera indexes when using startup option 2:

```bash
python drowsy_detection.py --camera-scan-limit 8
```

Show low-level MediaPipe/TFLite diagnostics while debugging:

```bash
python drowsy_detection.py --input 0 --native-logs
```

Adjust the Haar face-detection gate. FaceMesh processes a padded crop around
the Haar face box, and keeps running for this many missed Haar frames before
being skipped:

```bash
python drowsy_detection.py --input 0 --face-gate-grace 5
```

Adjust the crop margin around the Haar face box before FaceMesh processing:

```bash
python drowsy_detection.py --input 0 --facemesh-roi-margin 0.20
```

Calibration is enabled by default. To skip it and use the default threshold:

```bash
python drowsy_detection.py --input 0 --no-calibrate
```

During calibration, the app measures your median open-eye EAR, then sets:

```text
threshold = median_open_eye_ear * calibration_ratio
```

The default calibration ratio is `0.75`. You can change it:

```bash
python drowsy_detection.py --input 0 --calibration-seconds 6 --calibration-ratio 0.72
```

Calibration uses valid EAR samples collected during the first
`--calibration-seconds` seconds of the selected source. If no face landmarks are
detected during that window, the app keeps the configured `--ear-threshold`.

## Generated Local Files

The app creates a few local files while running:

- `.cache/` stores Matplotlib and runtime cache files.
- `.camera_aliases.json` stores camera labels you save from the scan menu.
- `pipeline_performance_log.csv` stores per-run benchmark summaries.
- `pipeline_performance_benchmark.png` stores the latest benchmark chart.
- `alarm_events_log.csv` stores alarm start/stop transitions.
- `frame_state_log.csv` stores per-frame detector state when requested or when
  `--eval-controlled` is used.
- `controlled_evaluation_summary.json` stores the latest controlled-clip score.
- `test_video.mp4` is created by `record_test.py`.

These files are ignored by git.

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

MediaPipe FaceMesh import errors

Recreate `.venv` with Python 3.12 and reinstall `requirements.txt`. The project
pins MediaPipe to `0.10.20` because the code uses the legacy FaceMesh solution
implementation.

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

If you intentionally pass `--no-display`, keyboard controls such as `q`, `c`,
and `Esc` are unavailable from the OpenCV window. Video files stop when the file
ends; webcam streams keep running until interrupted from the terminal.

`[ERROR] Cannot open: '0'`

OpenCV could not access your webcam. Check camera permissions, close other apps
using the camera, or pass a video file with `--input`.

## Pipeline

For each frame:

1. Convert BGR frame to grayscale.
2. Apply Gaussian blur to reduce noise.
3. Detect the face with Haar cascades as a fast face-presence gate.
4. Draw the detected face, padded FaceMesh crop, and approximate eye-region band on the full frame.
5. Crop the padded face ROI from the raw frame and convert that crop to RGB.
6. Use MediaPipe FaceMesh on the crop and remap landmarks to full-frame coordinates.
7. Compute EAR from six landmarks per eye.
8. During calibration, collect open-eye EAR samples and set
   `threshold = median_open_eye_ear * calibration_ratio`.
9. Count consecutive frames where EAR is below the threshold.
10. Mark drowsy and trigger the alarm when the counter reaches `--frames`.
