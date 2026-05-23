import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from drowsy_driver.detector import (
    CONSEC_FRAMES,
    build_eval_segments,
    label_controlled_frame,
    row_float,
    row_int,
    row_truthy,
    tagged_output_path,
    tracking_failed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
RUNS_DIR = OUTPUT_DIR / "runs"
VALIDATION_CLIP = OUTPUT_DIR / "test_video.mp4"
LATEST_VALIDATION_RESULTS = RUNS_DIR / "latest_validation_results.json"

VALIDATION_SCHEDULE = (
    "open:5,closed:0.5,open:5,closed:1.0,"
    "open:5,closed:2.0,open:5,closed:3.0,open:5"
)

DEFAULT_TRANSITION_BUFFER_FRAMES = 12
DEFAULT_BOOTSTRAP_ITERATIONS = 1000

BASE_DIR = PROJECT_ROOT
APP_CACHE_DIR = CACHE_DIR
APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(APP_CACHE_DIR / "matplotlib").mkdir(exist_ok=True)
(APP_CACHE_DIR / "xdg").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(APP_CACHE_DIR / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(APP_CACHE_DIR / "xdg"))

DEFAULT_SCHEDULE = VALIDATION_SCHEDULE


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def ask(prompt, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return str(default)
    return value


def ask_int(prompt, default):
    while True:
        value = ask(prompt, default)
        try:
            return int(value)
        except ValueError:
            print("Enter an integer.")


def ask_yes_no(prompt, default=True):
    default_text = "Y/n" if default else "y/N"
    while True:
        value = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y or n.")


def safe_tag(value):
    value = value.strip() or "validation"
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "validation"


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def run_command(command):
    print("\n$ " + " ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def format_percent(value):
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def format_ci(ci):
    if not ci or ci[0] is None or ci[1] is None:
        return "n/a"
    return f"{float(ci[0]) * 100:.1f}%-{float(ci[1]) * 100:.1f}%"


BUCKET_LABELS = {
    "near_boundary_residual": "Near a scheduled open/closed switch",
    "tracking_unstable": "Face or landmark tracking was unstable",
    "squint_or_partial_closure": "Eyes looked partly closed during an open-eye period",
    "deep_closure_during_open": "Eyes looked fully closed during an open-eye period",
    "fsm_delay": "Detector needed its normal consecutive-frame delay",
    "ear_above_threshold": "Eyes measured above the closed-eye threshold",
    "other": "Other or unclear cause",
}


def bucket_label(bucket):
    return BUCKET_LABELS.get(bucket, bucket.replace("_", " "))


def schedule_text(summary):
    schedule = summary.get("schedule", {})
    if schedule.get("type") == "explicit":
        phases = schedule.get("schedule", [])
        return ", ".join(f"{phase}:{duration:g}s" for phase, duration in phases)
    return (
        f"open:{schedule.get('open_seconds', 5.0):g}s, "
        f"closed:{schedule.get('closed_seconds', 3.0):g}s repeating"
    )


def bucket_lines(section):
    lines = []
    total = max(1, int(section.get("total", 0)))
    for bucket, count in section.get("by_bucket", {}).items():
        lines.append(f"| {bucket_label(bucket)} | {count} | {count / total * 100:.1f}% |")
    return lines


def interpretation(summary, breakdown):
    counts = summary.get("counts", {})
    latency = summary.get("detection_latency_seconds", {})
    fp = counts.get("fp", 0)
    fn = counts.get("fn", 0)
    lines = []

    if fp == 0:
        lines.append("- No false positives remained after transition-buffer and tracking-failure filtering.")
    else:
        fp_buckets = breakdown.get("false_positives", {}).get("by_bucket", {})
        top = max(fp_buckets, key=fp_buckets.get) if fp_buckets else "unknown"
        lines.append(f"- False positives remain; the largest likely cause is {bucket_label(top)}.")

    if fn == 0:
        lines.append("- No false negatives remained in eligible frames.")
    else:
        fn_buckets = breakdown.get("false_negatives", {}).get("by_bucket", {})
        top = max(fn_buckets, key=fn_buckets.get) if fn_buckets else "unknown"
        lines.append(f"- False negatives remain; the largest likely cause is {bucket_label(top)}.")

    if latency.get("mean") is not None:
        lines.append(
            f"- Mean detection latency was {latency['mean']:.3f}s across "
            f"{latency.get('detected_closed_segments', 0)}/"
            f"{latency.get('total_closed_segments', 0)} detected closed segments."
        )

    tracking_rate = summary.get("tracking_failure_rate", 0.0)
    if tracking_rate > 0.05:
        lines.append("- Tracking failure rate is high; inspect lighting, face pose, and ROI stability.")
    else:
        lines.append("- Tracking failure rate is low for this controlled clip.")

    return lines


def generate_report(run_dir, config, summary_path, breakdown_path):
    summary = load_json(summary_path)
    breakdown = load_json(breakdown_path)
    counts = summary.get("counts", {})
    metrics = summary.get("metrics", {})
    latency = summary.get("detection_latency_seconds", {})

    lines = [
        "# Controlled Validation Report",
        "",
        "## Overview",
        "",
        f"- Run directory: `{rel(run_dir)}`",
        f"- Input: `{config['input']}`",
        f"- Tag: `{config['tag']}`",
        f"- Run ID: `{summary.get('run_id', 'n/a')}`",
        f"- Schedule: {schedule_text(summary)}",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Correct alarms (precision) | {format_percent(metrics.get('precision'))} |",
        f"| Precision 95% CI | {format_ci(summary.get('precision_ci_95'))} |",
        f"| Sleepy frames caught (recall) | {format_percent(metrics.get('recall'))} |",
        f"| Recall 95% CI | {format_ci(summary.get('recall_ci_95'))} |",
        f"| Overall score (F1) | {format_percent(metrics.get('f1'))} |",
        f"| F1 95% CI | {format_ci(summary.get('f1_ci_95'))} |",
        "",
        "## Confusion Matrix",
        "",
        "| Count | Value |",
        "|---|---:|",
        f"| Sleepy frames caught | {counts.get('tp', 0)} |",
        f"| False alarms | {counts.get('fp', 0)} |",
        f"| Awake frames correctly ignored | {counts.get('tn', 0)} |",
        f"| Sleepy frames missed | {counts.get('fn', 0)} |",
        "",
        "## Eval Filters",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Total frames | {summary.get('total_frames', 'n/a')} |",
        f"| Eligible frames | {summary.get('eligible_frames', summary.get('frames_evaluated', 'n/a'))} |",
        f"| Transition-buffer frames excluded | {summary.get('buffer_frames_excluded', 0)} |",
        f"| Tracking-failed frames excluded | {summary.get('tracking_failed_frames', 0)} |",
        f"| Tracking failure rate | {summary.get('tracking_failure_rate', 0.0):.3%} |",
        f"| Transition buffer | {summary.get('eval_transition_buffer_frames', 0)} frames |",
        f"| Bootstrap iterations | {summary.get('bootstrap_iterations', 0)} |",
        "",
        "## Segment Latency",
        "",
        "| Segment start | Duration | Detected | Latency |",
        "|---:|---:|:---:|---:|",
    ]

    for segment in summary.get("closed_segments", latency.get("closed_segments", [])):
        detected = "yes" if segment.get("detected") else "no"
        lat = segment.get("latency")
        lat_text = f"{lat:.3f}s" if lat is not None else "n/a"
        lines.append(
            f"| {segment.get('start', 0.0):.3f}s | "
            f"{segment.get('duration', 0.0):.3f}s | {detected} | {lat_text} |"
        )

    lines.extend([
        "",
        "## Failure Breakdown",
        "",
        "### False Positives",
        "",
        "| Likely cause | Count | Share |",
        "|---|---:|---:|",
        *bucket_lines(breakdown.get("false_positives", {})),
        "",
        "### False Negatives",
        "",
        "| Likely cause | Count | Share |",
        "|---|---:|---:|",
        *bucket_lines(breakdown.get("false_negatives", {})),
        "",
        "## Interpretation",
        "",
        *interpretation(summary, breakdown),
        "",
        "## Files",
        "",
        f"- Summary JSON: `{rel(summary_path)}`",
        f"- Failure breakdown JSON: `{rel(breakdown_path)}`",
        f"- Frame state CSV: `{rel(config['frame_log'])}`",
        f"- Alarm events CSV: `{rel(config['alarm_log'])}`",
    ])

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def print_report_summary(report_path, summary_path):
    summary = load_json(summary_path)
    metrics = summary.get("metrics", {})
    counts = summary.get("counts", {})
    latency = summary.get("detection_latency_seconds", {})
    print("\nValidation complete.")
    print(f"Report: {rel(report_path)}")
    print(
        f"Overall score={format_percent(metrics.get('f1'))} "
        f"[95% CI {format_ci(summary.get('f1_ci_95'))}], "
        f"Correct alarms={format_percent(metrics.get('precision'))}, "
        f"Sleepy frames caught={format_percent(metrics.get('recall'))}"
    )
    print(
        f"Sleepy caught={counts.get('tp', 0)} False alarms={counts.get('fp', 0)} "
        f"Awake correct={counts.get('tn', 0)} Sleepy missed={counts.get('fn', 0)}"
    )
    print(
        "Closed segments detected: "
        f"{latency.get('detected_closed_segments', 0)}/"
        f"{latency.get('total_closed_segments', 0)}"
    )


def latest_report():
    reports = sorted(RUNS_DIR.glob("*/report.md"), key=lambda p: p.stat().st_mtime)
    if not reports:
        print("No validation reports found yet.")
        return
    report = reports[-1]
    print(f"\nLatest report: {rel(report)}\n")
    print(report.read_text(encoding="utf-8"))


def validation_result(config, report_path, summary_path):
    summary = load_json(summary_path)
    metrics = summary.get("metrics", {})
    counts = summary.get("counts", {})
    latency = summary.get("detection_latency_seconds", {})
    return {
        "tag": config["tag"],
        "input": config["input"],
        "run_dir": rel(config["run_dir"]),
        "report": rel(report_path),
        "summary": rel(summary_path),
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "tp": counts.get("tp", 0),
        "fp": counts.get("fp", 0),
        "tn": counts.get("tn", 0),
        "fn": counts.get("fn", 0),
        "detected_closed_segments": latency.get("detected_closed_segments", 0),
        "total_closed_segments": latency.get("total_closed_segments", 0),
    }


def write_latest_validation_results(results):
    write_json(
        LATEST_VALIDATION_RESULTS,
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        },
    )
    print(f"GUI summary: {rel(LATEST_VALIDATION_RESULTS)}")


def parse_schedule(value):
    schedule = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Schedule part '{part}' must use phase:seconds")
        phase, duration = part.split(":", 1)
        phase = phase.strip().lower()
        if phase not in {"open", "closed"}:
            raise ValueError(f"Unsupported phase '{phase}'. Use open or closed.")
        duration = float(duration.strip())
        if duration <= 0:
            raise ValueError("Schedule durations must be greater than 0.")
        schedule.append((phase, duration))
    if not schedule:
        raise ValueError("Schedule cannot be empty.")
    return schedule


def resolve_output_path(value):
    if os.path.isabs(value):
        return value
    return str(BASE_DIR / value)


def schedule_sidecar_path(video_path):
    root, _ext = os.path.splitext(video_path)
    return f"{root}.schedule.json"


def schedule_label(schedule):
    return " -> ".join(f"{phase.capitalize()} {duration:g}s" for phase, duration in schedule)


def current_phase(elapsed, schedule):
    cursor = 0.0
    for phase, duration in schedule:
        end = cursor + duration
        if elapsed < end:
            remaining = end - elapsed
            if phase == "open":
                return "open", "KEEP EYES OPEN", remaining, (0, 220, 0)
            return "closed", "CLOSE EYES", remaining, (0, 0, 255)
        cursor = end

    phase, _duration = schedule[-1]
    if phase == "open":
        return "open", "KEEP EYES OPEN", 0.0, (0, 220, 0)
    return "closed", "CLOSE EYES", 0.0, (0, 0, 255)


def fit_text_scale(cv2, text, max_width, base_scale, thickness):
    scale = base_scale
    while scale > 0.35:
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        if text_width <= max_width:
            return scale
        scale -= 0.05
    return scale


def put_centered_text(cv2, frame, text, y, scale, color, thickness, font=None):
    if font is None:
        font = cv2.FONT_HERSHEY_DUPLEX
    h, w = frame.shape[:2]
    scale = fit_text_scale(cv2, text, int(w * 0.92), scale, thickness)
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (w - text_width) // 2)
    cv2.putText(frame, text, (x, y + text_height // 2), font, scale, color, thickness, cv2.LINE_AA)


def draw_ready_overlay(cv2, frame, schedule_text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, min(h, 185)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    put_centered_text(cv2, frame, "READY TO RECORD", 45, 1.15, (0, 240, 0), 3)
    put_centered_text(cv2, frame, schedule_text, 92, 0.58, (255, 255, 255), 1,
                      font=cv2.FONT_HERSHEY_SIMPLEX)
    put_centered_text(cv2, frame, "Preview prompts are not saved to the MP4", 126, 0.54,
                      (225, 225, 225), 1, font=cv2.FONT_HERSHEY_SIMPLEX)
    put_centered_text(cv2, frame, "Press S or Space to start | Q to cancel", 158, 0.58,
                      (255, 255, 255), 1, font=cv2.FONT_HERSHEY_SIMPLEX)


def draw_recording_overlay(cv2, frame, remaining, phase_text, phase_remaining, phase_color,
                           schedule_text):
    h, w = frame.shape[:2]
    panel_bottom = min(h, 230)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    phase_overlay = frame.copy()
    cv2.rectangle(phase_overlay, (0, 62), (w, min(h, 152)), phase_color, -1)
    cv2.addWeighted(phase_overlay, 0.42, frame, 0.58, 0, frame)

    cv2.circle(frame, (24, 28), 8, (0, 0, 255), -1)
    cv2.putText(frame, f"REC  {remaining:.1f}s remaining",
                (42, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    put_centered_text(cv2, frame, phase_text, 103, 1.75, (255, 255, 255), 4)
    put_centered_text(cv2, frame, f"{phase_remaining:.1f}s left in this phase", 170, 0.72,
                      (255, 255, 255), 2, font=cv2.FONT_HERSHEY_SIMPLEX)
    put_centered_text(cv2, frame, f"{schedule_text} | Q to stop", 207, 0.50,
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


def write_schedule_sidecar(path, schedule, total_seconds, stopped_early):
    sidecar = schedule_sidecar_path(path)
    data = {
        "schedule": [[phase, duration] for phase, duration in schedule],
        "total_seconds": total_seconds,
        "fps": 30,
        "stopped_early": stopped_early,
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"[INFO] Schedule sidecar: {sidecar}")


def record_validation_clip(output, schedule_text):
    import cv2

    try:
        schedule = parse_schedule(schedule_text)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    output = resolve_output_path(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    total_duration = sum(duration for _phase, duration in schedule)
    output_fps = 30
    total_output_frames = int(round(total_duration * output_fps))
    schedule_name = schedule_label(schedule)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("[INFO] Camera ready.")
    print("[INFO] Press 's' or Space when you are ready to start recording.")
    print("[INFO] Press 'q' to cancel.")
    print(f"[INFO] Schedule: {schedule_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Cannot read from webcam")

        draw_ready_overlay(cv2, frame, schedule_name)
        cv2.imshow("Recording setup", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("s"), ord(" ")):
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Recording cancelled")
            return

    print(f"[INFO] Recording {total_duration:.1f}s -> {output}")
    print("[INFO] Follow the preview prompts. They are not saved to the MP4.")
    print("[INFO] Press 'q' to stop early")

    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (w, h))
    if not writer.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError(f"Cannot create output video: {output}")

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
        remaining = max(0, total_duration - elapsed)
        phase_key, phase_text, phase_remaining, phase_color = current_phase(elapsed, schedule)
        if phase_key != last_phase_key:
            play_cue(audio_cues, phase_key)
            last_phase_key = phase_key

        draw_recording_overlay(cv2, frame, remaining, phase_text, phase_remaining, phase_color, schedule_name)

        target_frames = min(int(elapsed * output_fps), total_output_frames)
        while frames_written < target_frames:
            writer.write(raw_frame)
            frames_written += 1

        cv2.imshow("Recording - press q to stop", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stopped_early = True
            break
        if elapsed >= total_duration:
            break

    while not stopped_early and last_raw_frame is not None and frames_written < total_output_frames:
        writer.write(last_raw_frame)
        frames_written += 1

    cap.release()
    writer.release()
    if pygame_audio is not None:
        pygame_audio.mixer.quit()
    cv2.destroyAllWindows()
    saved_duration = frames_written / output_fps
    write_schedule_sidecar(output, schedule, total_duration, stopped_early)
    print(f"[INFO] Saved clean video: {output}")
    print(f"[INFO] Frames written: {frames_written} at {output_fps} FPS ({saved_duration:.1f}s)")


def run_live_detector():
    source = ask("Input source blank for startup menu, camera index/path/URL otherwise", "")
    command = [sys.executable, "-m", "drowsy_driver.detector"]
    if source:
        command.extend(["--input", source])
    if ask_yes_no("Disable audio", False):
        command.append("--no-audio")
    if ask_yes_no("Skip calibration", False):
        command.append("--no-calibrate")
    run_command(command)


def record_clip():
    output = ask("Output clip path", str(VALIDATION_CLIP))
    schedule_default = DEFAULT_SCHEDULE
    schedule = ask("Schedule", schedule_default)
    command = [
        sys.executable,
        "-m",
        "drowsy_driver.validation",
        "--record",
        "--output",
        output,
        "--schedule",
        schedule,
    ]
    run_command(command)


def validate_clip(
    input_path=None,
    tag=None,
    buffer_frames=None,
    bootstrap_iterations=None,
    assume_fps=None,
    interactive=None,
    update_latest=True,
):
    if interactive is None:
        interactive = input_path is None

    default_clip = VALIDATION_CLIP
    input_path = input_path or ask("Controlled clip path", str(default_clip))
    tag = safe_tag(tag or (ask("Run tag", "validation") if interactive else "validation"))
    if buffer_frames is None:
        buffer_frames = (
            ask_int("Transition buffer frames", DEFAULT_TRANSITION_BUFFER_FRAMES)
            if interactive
            else DEFAULT_TRANSITION_BUFFER_FRAMES
        )
    if bootstrap_iterations is None:
        bootstrap_iterations = (
            ask_int("Bootstrap iterations", DEFAULT_BOOTSTRAP_ITERATIONS)
            if interactive
            else DEFAULT_BOOTSTRAP_ITERATIONS
        )
    if assume_fps is None:
        assume_fps = ask("Assume FPS override blank to use video metadata", "") if interactive else ""

    buffer_frames = int(buffer_frames)
    bootstrap_iterations = int(bootstrap_iterations)
    if buffer_frames < 0:
        raise ValueError("Transition buffer frames must be at least 0")
    if bootstrap_iterations < 0:
        raise ValueError("Bootstrap iterations must be at least 0")
    if assume_fps in (None, ""):
        assume_fps = ""
    else:
        assume_fps = str(float(assume_fps))
        if float(assume_fps) <= 0:
            raise ValueError("Assume FPS must be greater than 0")

    run_dir = RUNS_DIR / f"{timestamp()}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_log_base = run_dir / "frame_state.csv"
    summary_base = run_dir / "summary.json"
    alarm_log = run_dir / "alarm_events.csv"
    frame_log = Path(tagged_output_path(str(frame_log_base), tag))
    summary_path = Path(tagged_output_path(str(summary_base), tag))
    breakdown_path = run_dir / "failure_breakdown.json"

    command = [
        sys.executable,
        "-m",
        "drowsy_driver.detector",
        "--input",
        input_path,
        "--no-display",
        "--no-audio",
        "--no-calibrate",
        "--eval-controlled",
        "--frame-log",
        str(frame_log_base),
        "--alarm-log",
        str(alarm_log),
        "--eval-output",
        str(summary_base),
        "--eval-tag",
        tag,
        "--eval-transition-buffer-frames",
        str(buffer_frames),
        "--eval-bootstrap-iterations",
        str(bootstrap_iterations),
    ]
    if assume_fps:
        command.extend(["--eval-assume-fps", assume_fps])

    config = {
        "input": input_path,
        "tag": tag,
        "run_dir": str(run_dir),
        "frame_log": str(frame_log),
        "summary": str(summary_path),
        "alarm_log": str(alarm_log),
        "failure_breakdown": str(breakdown_path),
        "transition_buffer_frames": buffer_frames,
        "bootstrap_iterations": bootstrap_iterations,
        "assume_fps": assume_fps or None,
        "detector_command": command,
    }
    write_json(run_dir / "config.json", config)

    run_command(command)
    analyze_failure_files(str(frame_log), str(summary_path), str(breakdown_path))

    report_path = generate_report(run_dir, config, summary_path, breakdown_path)
    print_report_summary(report_path, summary_path)
    result = validation_result(config, report_path, summary_path)
    if update_latest:
        write_latest_validation_results([result])
    return result


def validate_standard_clip(buffer_frames=None, bootstrap_iterations=None, assume_fps=None):
    if not VALIDATION_CLIP.exists():
        raise RuntimeError(f"Validation clip not found: {rel(VALIDATION_CLIP)}")

    result = validate_clip(
        input_path=str(VALIDATION_CLIP),
        tag="validation",
        buffer_frames=buffer_frames,
        bootstrap_iterations=bootstrap_iterations,
        assume_fps=assume_fps,
        interactive=False,
        update_latest=False,
    )
    write_latest_validation_results([result])
    print(
        "\nValidation summary\n"
        "| Overall score | Correct alarms | Sleepy frames caught | Sleepy caught | False alarms | Awake correct | Sleepy missed | Closed-eye periods |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {format_percent(result.get('f1'))} | "
        f"{format_percent(result.get('precision'))} | "
        f"{format_percent(result.get('recall'))} | "
        f"{result.get('tp', 0)} | {result.get('fp', 0)} | "
        f"{result.get('tn', 0)} | {result.get('fn', 0)} | "
        f"{result.get('detected_closed_segments', 0)}/"
        f"{result.get('total_closed_segments', 0)} |"
        )
    return [result]


def validate_all_clips(buffer_frames=None, bootstrap_iterations=None, assume_fps=None):
    return validate_standard_clip(
        buffer_frames=buffer_frames,
        bootstrap_iterations=bootstrap_iterations,
        assume_fps=assume_fps,
    )


FP_BUCKETS = [
    "near_boundary_residual",
    "tracking_unstable",
    "squint_or_partial_closure",
    "deep_closure_during_open",
    "other",
]

FN_BUCKETS = [
    "fsm_delay",
    "tracking_unstable",
    "ear_above_threshold",
    "other",
]


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def infer_fps(rows, fallback=30.0):
    times = [row_float(row, "time_seconds", default=-1.0) for row in rows]
    times = [value for value in times if value >= 0]
    deltas = [
        later - earlier
        for earlier, later in zip(times, times[1:])
        if later > earlier
    ]
    if not deltas:
        return fallback
    deltas.sort()
    median_delta = deltas[len(deltas) // 2]
    return 1.0 / median_delta if median_delta > 0 else fallback


def schedule_from_summary(summary):
    schedule = summary.get("schedule", {})
    if schedule.get("type") == "explicit":
        return (
            [(str(phase), float(duration)) for phase, duration in schedule.get("schedule", [])],
            None,
            None,
        )
    return (
        None,
        float(schedule.get("open_seconds", 5.0)),
        float(schedule.get("closed_seconds", 3.0)),
    )


def empty_breakdown(bucket_names):
    return {
        "total": 0,
        "by_bucket": {bucket: 0 for bucket in bucket_names},
        "examples": {bucket: [] for bucket in bucket_names},
    }


def add_example(section, bucket, row, ear, threshold):
    section["by_bucket"][bucket] += 1
    section["total"] += 1
    examples = section["examples"][bucket]
    if len(examples) < 10:
        examples.append({
            "frame": row_int(row, "frame"),
            "time_seconds": row_float(row, "time_seconds"),
            "ear": ear,
            "threshold": threshold,
            "bucket": bucket,
        })


def fp_bucket(row, label):
    ear = row_float(row, "ear")
    threshold = row_float(row, "threshold")
    if label["near_boundary"] or row_truthy(row, "near_boundary"):
        return "near_boundary_residual"
    if (
        row_truthy(row, "facemesh_ran")
        and row_int(row, "landmarks_found") >= 12
        and row_truthy(row, "haar_box_reused")
    ):
        return "tracking_unstable"
    if threshold > ear > threshold - 0.04:
        return "squint_or_partial_closure"
    if ear < threshold - 0.04:
        return "deep_closure_during_open"
    return "other"


def fn_bucket(row, label, fps):
    time_seconds = row_float(row, "time_seconds")
    segment_start = label["segment_start"]
    if segment_start is not None and 0 <= (time_seconds - segment_start) * fps < CONSEC_FRAMES:
        return "fsm_delay"
    if (
        row_truthy(row, "facemesh_ran")
        and row_int(row, "landmarks_found") >= 12
        and row_truthy(row, "haar_box_reused")
    ):
        return "tracking_unstable"
    if row_float(row, "ear") > row_float(row, "threshold"):
        return "ear_above_threshold"
    return "other"


def flatten_examples(section):
    examples = []
    for bucket in section["by_bucket"]:
        examples.extend(section["examples"][bucket])
    section["examples"] = examples


def analyze_failures(rows, summary):
    explicit_schedule, open_seconds, closed_seconds = schedule_from_summary(summary)
    fps = float(summary.get("schedule", {}).get("fps") or infer_fps(rows))
    max_time = max((row_float(row, "time_seconds", default=0.0) for row in rows), default=0.0)
    segments = build_eval_segments(
        open_seconds or 5.0,
        closed_seconds or 3.0,
        max_time,
        explicit_schedule,
    )
    transition_buffer = int(summary.get("eval_transition_buffer_frames", 0))

    false_positives = empty_breakdown(FP_BUCKETS)
    false_negatives = empty_breakdown(FN_BUCKETS)

    for row in rows:
        time_seconds = row_float(row, "time_seconds", default=-1.0)
        if time_seconds < 0:
            continue
        label = label_controlled_frame(time_seconds, segments, transition_buffer, fps)
        if label["near_boundary"] or tracking_failed(row):
            continue

        actual_closed = label["label"] == "closed"
        predicted_closed = row_truthy(row, "drowsy")
        ear = row_float(row, "ear")
        threshold = row_float(row, "threshold")

        if not actual_closed and predicted_closed:
            add_example(false_positives, fp_bucket(row, label), row, ear, threshold)
        elif actual_closed and not predicted_closed:
            add_example(false_negatives, fn_bucket(row, label, fps), row, ear, threshold)

    flatten_examples(false_positives)
    flatten_examples(false_negatives)
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def print_breakdown_section(title, section):
    print(f"{title} ({section['total']} total):")
    total = max(1, int(section["total"]))
    for bucket, count in section["by_bucket"].items():
        print(f"  {bucket_label(bucket):<52} {count:>4} ({count / total * 100:>5.1f}%)")


def analyze_failure_files(frame_log, summary_path, output):
    rows = read_csv(frame_log)
    summary = load_json(summary_path)
    result = analyze_failures(rows, summary)

    expected_fp = int(summary["counts"]["fp"])
    expected_fn = int(summary["counts"]["fn"])
    if result["false_positives"]["total"] != expected_fp:
        raise RuntimeError(
            f"FP bucket total {result['false_positives']['total']} "
            f"does not match summary FP {expected_fp}"
        )
    if result["false_negatives"]["total"] != expected_fn:
        raise RuntimeError(
            f"FN bucket total {result['false_negatives']['total']} "
            f"does not match summary FN {expected_fn}"
        )

    write_json(Path(output), result)
    print_breakdown_section("FP breakdown", result["false_positives"])
    print_breakdown_section("FN breakdown", result["false_negatives"])
    print(f"Output JSON: {output}")


def analyze_existing():
    frame_log = ask("Frame log CSV", str(BASE_DIR / "outputs" / "frame_state_log.csv"))
    summary = ask("Summary JSON", str(BASE_DIR / "outputs" / "controlled_evaluation_summary.json"))
    output = ask("Failure breakdown output", str(BASE_DIR / "outputs" / "fp_breakdown.json"))
    analyze_failure_files(frame_log, summary, output)


def menu():
    actions = {
        "1": ("Live detector", run_live_detector),
        "2": ("Record validation clip", record_clip),
        "3": ("Run validation test", validate_standard_clip),
        "4": ("Analyze existing frame log", analyze_existing),
        "5": ("Show latest report", latest_report),
        "0": ("Exit", None),
    }

    while True:
        print("\nDrowsy Driver Detector")
        print("======================")
        for key, (label, _handler) in actions.items():
            print(f"  {key}. {label}")
        choice = input("Choose an option: ").strip() or "4"
        if choice == "0":
            return
        action = actions.get(choice)
        if action is None:
            print("Choose one of the listed options.")
            continue
        try:
            action[1]()
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as exc:
            print(f"\n[ERROR] {exc}")


def main():
    parser = argparse.ArgumentParser(description="Validation recording, scoring, and reporting tools.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--record", action="store_true", help="Record the standard validation clip")
    actions.add_argument("--validate", metavar="CLIP", help="Validate a specific controlled clip")
    actions.add_argument("--validate-all", action="store_true", help="Run the standard validation clip")
    actions.add_argument("--analyze-failures", action="store_true", help="Break down false positives and false negatives")
    actions.add_argument("--latest-report", action="store_true", help="Print the latest generated validation report")
    actions.add_argument("--menu", action="store_true", help="Open the terminal validation menu")
    parser.add_argument("--tag", default=None, help="Tag to use with --validate")
    parser.add_argument("--output", default=None, help="Output path for --record or --analyze-failures")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE, help="Recording schedule for --record")
    parser.add_argument("--frame-log", default=None, help="Frame-state CSV for --analyze-failures")
    parser.add_argument("--summary", default=None, help="Evaluation summary JSON for --analyze-failures")
    parser.add_argument(
        "--transition-buffer-frames",
        type=int,
        default=None,
        help=f"Frames to exclude around schedule transitions during --validate (default: {DEFAULT_TRANSITION_BUFFER_FRAMES})",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=None,
        help=f"Bootstrap samples for validation confidence intervals (default: {DEFAULT_BOOTSTRAP_ITERATIONS})",
    )
    parser.add_argument(
        "--assume-fps",
        default=None,
        help="Override video FPS for validation when metadata is unreliable",
    )
    args = parser.parse_args()

    if args.record:
        record_validation_clip(args.output or str(VALIDATION_CLIP), args.schedule)
        return
    if args.validate:
        validate_clip(
            input_path=args.validate,
            tag=args.tag,
            buffer_frames=args.transition_buffer_frames,
            bootstrap_iterations=args.bootstrap_iterations,
            assume_fps=args.assume_fps,
            interactive=False,
        )
        return
    if args.validate_all:
        validate_all_clips(
            buffer_frames=args.transition_buffer_frames,
            bootstrap_iterations=args.bootstrap_iterations,
            assume_fps=args.assume_fps,
        )
        return
    if args.analyze_failures:
        if not args.frame_log or not args.summary:
            parser.error("--analyze-failures requires --frame-log and --summary")
        analyze_failure_files(
            args.frame_log,
            args.summary,
            args.output or str(BASE_DIR / "outputs" / "fp_breakdown.json"),
        )
        return
    if args.latest_report:
        latest_report()
        return
    if args.menu:
        menu()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
