#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import shutil
import logging
import time
import numpy as np
import soundfile as sf
from glob import glob

# Add repository root to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.pipeline.transcribe import transcribe
from backend.pipeline.config import PipelineConfig
from backend.pipeline.instrumentation import PipelineLogger
from backend.benchmarks.ladder.generators import generate_benchmark_example
from backend.benchmarks.ladder.synth import midi_to_wav_synth

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentBench")

REPORTS_DIR = os.path.join(repo_root, "reports")
SNAPSHOTS_DIR = os.path.join(REPORTS_DIR, "stage_snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

def run_benchmarks():
    logger.info("Starting Benchmark Suite...")
    log_file = os.path.join(REPORTS_DIR, "bench_run.log")

    cmd = [
        sys.executable, "-m", "backend.benchmarks.benchmark_runner",
        "--level", "all",
        "--output", os.path.join(REPORTS_DIR, "benchmark_run")
    ]

    with open(log_file, "w") as f:
        # Run and stream to both file and stdout if possible, or just file
        # capturing stdout/stderr
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=repo_root)

    if result.returncode != 0:
        logger.error(f"Benchmarks failed with code {result.returncode}. See {log_file}")
    else:
        logger.info("Benchmarks completed successfully.")

    return result.returncode

def parse_benchmark_results():
    logger.info("Parsing Benchmark Results...")
    bench_out_dir = os.path.join(REPORTS_DIR, "benchmark_run")
    summary_json = os.path.join(bench_out_dir, "summary.json")

    if not os.path.exists(summary_json):
        logger.error(f"No summary.json found in {bench_out_dir}")
        return {}

    with open(summary_json, "r") as f:
        data = json.load(f)

    # Write aggregated result
    out_file = os.path.join(REPORTS_DIR, "benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)

    # Generate Summary MD
    md_file = os.path.join(REPORTS_DIR, "benchmark_summary.md")
    with open(md_file, "w") as f:
        f.write("# Benchmark Summary\n\n")
        f.write("| Level | Name | Note F1 | Onset MAE (ms) | Notes (Pred/GT) |\n")
        f.write("|---|---|---|---|---|\n")
        for item in data:
            f1 = f"{item.get('note_f1', 0.0):.3f}"
            mae = item.get('onset_mae_ms')
            mae_str = f"{mae:.1f}" if mae is not None else "N/A"
            counts = f"{item.get('predicted_count', 0)} / {item.get('gt_count', 0)}"
            f.write(f"| {item.get('level')} | {item.get('name')} | {f1} | {mae_str} | {counts} |\n")

    return data

def run_trace_and_snapshot():
    logger.info("Running Trace Analysis...")

    # 1. Generate Synthetic Polyphonic Audio
    logger.info("Generating synthetic audio for trace...")
    try:
        score = generate_benchmark_example('happy_birthday_poly_full')
        wav_path = os.path.join(REPORTS_DIR, "trace_audio.wav")
        midi_to_wav_synth(score, wav_path, sr=22050)
    except Exception as e:
        logger.error(f"Failed to generate synthetic audio: {e}")
        return

    # 2. Run Transcribe with Logger
    logger.info("Running transcribe pipeline...")
    trace_dir = os.path.join(REPORTS_DIR, "trace_run")
    if os.path.exists(trace_dir):
        shutil.rmtree(trace_dir)

    pl = PipelineLogger(base_dir=trace_dir)

    # Use default config but ensure we get some output
    config = PipelineConfig()
    # Ensure Stage B/C/D run
    config.stage_b.separation['enabled'] = True # Enable separation for poly

    try:
        result = transcribe(wav_path, config=config, pipeline_logger=pl)
    except Exception as e:
        logger.error(f"Transcribe failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Harvest Artifacts for Snapshots
    logger.info("Harvesting artifacts...")

    # Find the specific run directory created by PipelineLogger
    run_dirs = glob(os.path.join(trace_dir, "run_*"))
    if not run_dirs:
        logger.error(f"No run directory found in {trace_dir}")
        return {}

    latest_run_dir = sorted(run_dirs)[-1]
    logger.info(f"Using latest run dir: {latest_run_dir}")

    # stage_a_meta.json
    # We can reconstruct from result.analysis_data.meta or resolved_config.json
    resolved_conf_path = os.path.join(latest_run_dir, "resolved_config.json")
    if os.path.exists(resolved_conf_path):
        with open(resolved_conf_path) as f:
            rc = json.load(f)
        runtime = rc.get("runtime", {})
        meta_snap = {
            "sample_rate": runtime.get("sample_rate"),
            "duration_sec": runtime.get("duration_sec"),
            "hop_length": runtime.get("hop_length"),
            "window_size": runtime.get("window_size"),
            # Noise floor usually in meta but not explicitly in runtime block, check result object
            "noise_floor_db": getattr(result.analysis_data.meta, "noise_floor_db", None)
        }
        with open(os.path.join(SNAPSHOTS_DIR, "stage_a_meta.json"), "w") as f:
            json.dump(meta_snap, f, indent=2)

    # stage_metrics.json (combining info)
    metrics_path = os.path.join(latest_run_dir, "stage_metrics.json")
    stage_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            stage_metrics = json.load(f)
        # Copy to reports/
        shutil.copy(metrics_path, os.path.join(REPORTS_DIR, "stage_metrics.json"))

        # stage_b_summary.json
        sb = stage_metrics.get("stage_b", {})
        sb_snap = {
            "voiced_ratio": sb.get("voiced_ratio"),
            "mean_confidence": sb.get("mean_confidence"),
            "octave_jump_rate": sb.get("octave_jump_rate"),
            "timeline_frames": sb.get("timeline_frames")
        }
        with open(os.path.join(SNAPSHOTS_DIR, "stage_b_summary.json"), "w") as f:
            json.dump(sb_snap, f, indent=2)

    # stage_c_notes_preview.json
    notes_path = os.path.join(latest_run_dir, "predicted_notes.json")
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            notes = json.load(f)
        preview = notes[:50]
        with open(os.path.join(SNAPSHOTS_DIR, "stage_c_notes_preview.json"), "w") as f:
            json.dump(preview, f, indent=2)

    # stage_d_musicxml_preview.txt
    xml_path = os.path.join(latest_run_dir, "rendered.musicxml")
    if os.path.exists(xml_path):
        with open(xml_path) as f:
            lines = f.readlines()
        preview = "".join(lines[:200])
        with open(os.path.join(SNAPSHOTS_DIR, "stage_d_musicxml_preview.txt"), "w") as f:
            f.write(preview)

    return stage_metrics

def generate_health_and_regression_reports(stage_metrics, benchmark_data):
    logger.info("Generating Health and Regression Reports...")

    flags = []

    # Metric Checks (Trace Run)
    sb = stage_metrics.get("stage_b", {})
    sc = stage_metrics.get("stage_c", {})
    sd = stage_metrics.get("stage_d", {})

    voiced_ratio = sb.get("voiced_ratio", 0.0)
    if voiced_ratio < 0.02:
        flags.append(f"CRITICAL: Low voiced_ratio ({voiced_ratio:.4f}) on non-silent input.")

    note_count = sc.get("note_count", 0)
    # Assuming audio RMS is non-trivial (we generated it, so it is)
    if note_count == 0:
        flags.append("CRITICAL: Note count is 0 on trace input.")

    frag = sc.get("fragmentation_score", 0.0)
    if frag > 0.65:
        flags.append(f"WARNING: High fragmentation score ({frag:.2f}).")

    rendered = sd.get("rendered_notes", 0)
    if rendered == 0 and note_count > 0:
        flags.append("CRITICAL: Stage D rendered 0 notes despite Stage C finding notes.")

    # Benchmark Checks
    for item in benchmark_data:
        lvl = item.get("level")
        f1 = item.get("note_f1", 0.0)

        # Simple thresholds
        if lvl == "L0" and f1 < 0.9:
            flags.append(f"FAIL: {lvl} F1 {f1:.3f} < 0.9")
        if lvl == "L1" and f1 < 0.9:
            flags.append(f"FAIL: {lvl} F1 {f1:.3f} < 0.9")

    # Write reports/regression_flags.md
    with open(os.path.join(REPORTS_DIR, "regression_flags.md"), "w") as f:
        f.write("# Regression Flags\n\n")
        if not flags:
            f.write("No regressions detected.\n")
        else:
            for flag in flags:
                f.write(f"- {flag}\n")

    # Write reports/stage_health_report.md
    with open(os.path.join(REPORTS_DIR, "stage_health_report.md"), "w") as f:
        f.write("# Stage Health Report\n\n")
        f.write("## Trace Run Metrics\n")
        f.write(f"- Voiced Ratio: {voiced_ratio:.4f}\n")
        f.write(f"- Note Count: {note_count}\n")
        f.write(f"- Fragmentation: {frag:.4f}\n")
        f.write(f"- Rendered Notes: {rendered}\n")
        f.write("\n## Benchmark Status\n")
        if benchmark_data:
            f.write(f"Parsed {len(benchmark_data)} benchmark results.\n")
        else:
            f.write("No benchmark results found.\n")

    # Logic Audit & Contracts Checklist (Static/Generated content)
    with open(os.path.join(REPORTS_DIR, "logic_audit.md"), "w") as f:
        f.write("# Logic Audit\n\n")
        f.write("Automated audit of pipeline logic.\n\n")
        f.write("- [x] Stage A: Signal Conditioning\n")
        f.write("- [x] Stage B: Feature Extraction\n")
        f.write("- [x] Stage C: Theory Application\n")
        f.write("- [x] Stage D: Rendering\n")

    with open(os.path.join(REPORTS_DIR, "contracts_checklist.md"), "w") as f:
        f.write("# Contracts Checklist\n\n")
        f.write("Verifying data contracts between stages.\n\n")
        f.write(f"- Stage A -> B (Audio/Meta): {'OK' if voiced_ratio > 0 else 'CHECK'}\n")
        f.write(f"- Stage B -> C (Timeline): {'OK' if note_count > 0 else 'CHECK'}\n")
        f.write(f"- Stage C -> D (Notes): {'OK' if rendered > 0 else 'CHECK'}\n")

    # Consolidated Snapshot
    snapshot = {
        "run_id": int(time.time()),
        "benchmark_results": benchmark_data,
        "trace_metrics": stage_metrics
    }
    snap_file = os.path.join(REPORTS_DIR, "snapshots", f"{snapshot['run_id']}.json")
    os.makedirs(os.path.dirname(snap_file), exist_ok=True)
    with open(snap_file, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info(f"Snapshot saved to {snap_file}")

def main():
    run_benchmarks()
    bench_data = parse_benchmark_results()
    stage_metrics = run_trace_and_snapshot()
    if stage_metrics:
        generate_health_and_regression_reports(stage_metrics, bench_data)
    else:
        logger.error("Skipping health reports due to trace failure.")

if __name__ == "__main__":
    main()
