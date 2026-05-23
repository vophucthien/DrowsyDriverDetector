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

Preferred workflow: open `app.py` in VS Code and click **Run Python File**, or
run the same entry point from a terminal:

```bash
python app.py
```

This opens the live detector window immediately and starts calibration. The
top-right buttons control the app:

```text
Settings     Recalibrate, reset EAR threshold, cycle GUI scale, camera, resolution
Validation   Pause live detection, record validation clip, run validation test, show latest report
```

The Validation panel pauses live detection before running test actions. A
validation run exports per-frame logs, computes metrics, runs failure analysis,
and writes a readable Markdown report into a timestamped folder under
`outputs/runs/`.

The old terminal menu is still available:

```bash
python app.py --menu
```

Direct validation without opening the menu:

```bash
python app.py --validate-all
```

## Structure

```text
app.py                         Main entry point
drowsy_driver/detector.py      Live detector and controlled-clip evaluator
drowsy_driver/validation.py    Recording, validation reports, and failure analysis
```

The implementation lives in the `drowsy_driver/` package. The lower-level
modules are still useful for debugging:

```bash
python -m drowsy_driver.detector --input 0
python -m drowsy_driver.validation --record
```

Direct webcam run without the GUI wrapper:

```bash
python -m drowsy_driver.detector --input 0
```

Video file:

```bash
python -m drowsy_driver.detector --input outputs/test_video.mp4
```

Record a webcam validation video:

```bash
python -m drowsy_driver.validation --record
python -m drowsy_driver.detector --input outputs/test_video.mp4
```

The recorder opens a preview first and waits until you press `s` or Space. It
then records camera `0` to a clean `outputs/test_video.mp4` using one validation
schedule that includes normal open-eye periods plus closed-eye durations near
the detector boundary: 0.5s, 1.0s, 2.0s, and 3.0s. The preview shows large
open-eye / closed-eye prompts with audio cues, but that overlay is not burned
into the saved MP4. It overwrites the previous test video and can be stopped
early with `q`.

The recorder writes `outputs/test_video.schedule.json` next to the MP4.
Controlled evaluation uses that sidecar automatically, so the one recording is
enough for the validation report.

IP camera:

```bash
python -m drowsy_driver.detector --input http://camera-url/video
```

Calibration starts automatically when the OpenCV window opens. Keep your eyes
open and face the camera while the `Calibrating` overlay is visible. Press `c`
or `Esc` during calibration to cancel it and use the default EAR threshold.
Press `q` in the OpenCV window to quit.

In `--no-display` mode, there is no OpenCV window, overlay, or keyboard cancel
handler. Use `--no-calibrate` for headless runs unless the first calibration
seconds of the input show open eyes.

When the run exits, the app prints the current run's per-stage performance
report, appends the run to `outputs/pipeline_performance_log.csv`, prints a table of
all recorded runs plus the average row, and exports
`outputs/pipeline_performance_benchmark.png` with the latest bottleneck chart.
The log table is local; delete that CSV when you want to reset the recorded
history. Alarm start/stop transitions are appended to
`outputs/alarm_events_log.csv` when they occur.

## Useful Options

Tune the EAR threshold and closed-frame count:

```bash
python -m drowsy_driver.detector --input 0 --ear-threshold 0.22 --frames 15
```

Disable audio:

```bash
python -m drowsy_driver.detector --input 0 --no-audio
```

Run without an OpenCV preview window:

```bash
python -m drowsy_driver.detector --input outputs/test_video.mp4 --no-display --no-audio --no-calibrate
```

Export a per-frame detector-state CSV:

```bash
python -m drowsy_driver.detector --input outputs/test_video.mp4 --frame-log outputs/frame_state_log.csv
```

Score the validation recording directly:

```bash
python -m drowsy_driver.detector --input outputs/test_video.mp4 --no-display --no-audio --no-calibrate --eval-controlled
```

This writes `outputs/frame_state_log.csv` and
`outputs/controlled_evaluation_summary.json`,
then prints plain-language accuracy percentages, frame counts, transition
buffer exclusions, tracking-failure exclusions, and alarm latency. The default
evaluation excludes 12 frames around each schedule transition to avoid
penalizing human reaction time. Use `--eval-transition-buffer-frames 0` to
reproduce the unbuffered historical metric.

Run the same validation through the unified app:

```bash
python app.py --validate-all
```

Break down false positives and false negatives after an eval run:

```bash
python -m drowsy_driver.validation --analyze-failures \
  --frame-log outputs/frame_state_log.csv \
  --summary outputs/controlled_evaluation_summary.json \
  --output outputs/fp_breakdown.json
```

Scan more camera indexes when using startup option 2:

```bash
python -m drowsy_driver.detector --camera-scan-limit 8
```

Show low-level MediaPipe/TFLite diagnostics while debugging:

```bash
python -m drowsy_driver.detector --input 0 --native-logs
```

Adjust the Haar face-detection gate. FaceMesh processes a padded crop around
the Haar face box, and keeps running for this many missed Haar frames before
being skipped:

```bash
python -m drowsy_driver.detector --input 0 --face-gate-grace 5
```

Adjust the crop margin around the Haar face box before FaceMesh processing:

```bash
python -m drowsy_driver.detector --input 0 --facemesh-roi-margin 0.20
```

Calibration is enabled by default. To skip it and use the default threshold:

```bash
python -m drowsy_driver.detector --input 0 --no-calibrate
```

During calibration, the app measures your median open-eye EAR, then sets:

```text
threshold = median_open_eye_ear * calibration_ratio
```

The default calibration ratio is `0.75`. You can change it:

```bash
python -m drowsy_driver.detector --input 0 --calibration-seconds 6 --calibration-ratio 0.72
```

Calibration uses valid EAR samples collected during the first
`--calibration-seconds` seconds of the selected source. If no face landmarks are
detected during that window, the app keeps the configured `--ear-threshold`.

## Generated Local Files

The app creates a few local files while running:

- `outputs/` stores local generated files and is ignored by git.
- `outputs/runs/` stores timestamped validation runs, each with config, logs, JSON
  summaries, failure breakdowns, and `report.md`.
- `outputs/cache/` stores Matplotlib and runtime cache files.
- `outputs/camera_aliases.json` stores camera labels you save from the scan menu.
- `outputs/pipeline_performance_log.csv` stores per-run benchmark summaries.
- `outputs/pipeline_performance_benchmark.png` stores the latest benchmark chart.
- `outputs/alarm_events_log.csv` stores alarm start/stop transitions.
- `outputs/frame_state_log.csv` stores per-frame detector state when requested or when
  `--eval-controlled` is used.
- `outputs/controlled_evaluation_summary.json` stores the latest controlled-clip score.
- `outputs/controlled_evaluation_summary_<tag>.json` is used for tagged evaluation
  runs.
- `outputs/fp_breakdown.json` stores the optional false-positive/false-negative bucket
  analysis.
- `*.schedule.json` files store custom recording schedules created by
  `python -m drowsy_driver.validation --record`.
- `outputs/test_video.mp4` is created by `python -m drowsy_driver.validation --record`.

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
python -m drowsy_driver.detector
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
to `outputs/cache/` so the cache can be reused instead of rebuilt every launch.
If startup is still very slow, make sure you are running the latest script from
this repo and that `outputs/cache/` is writable.

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
