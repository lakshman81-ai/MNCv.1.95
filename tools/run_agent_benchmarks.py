#!/usr/bin/env python3
"""
Run the full agent benchmark suite (L0-L6) and generate reports.
This script is the primary entry point for verifying Task 1 (Polyphonics Improvements).
"""

import os
import sys
import shutil
import time
import json
import logging
from pathlib import Path

# Add repo root to python path so we can import backend
sys.path.insert(0, os.getcwd())

from backend.benchmarks.benchmark_runner import main as benchmark_main

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_agent_benchmarks")

REPORTS_DIR = Path("reports")
SNAPSHOTS_DIR = REPORTS_DIR / "snapshots"

def setup_dirs():
    """Ensure report directories exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def run_benchmarks():
    """Run the benchmark runner logic."""
    # We use a timestamped output dir for the run to avoid collisions
    run_id = int(time.time())
    output_dir = f"results/benchmark_{run_id}"

    logger.info(f"Starting benchmark run. Output: {output_dir}")

    # Simulate command line arguments for benchmark_runner
    # We want to run all levels
    sys.argv = ["benchmark_runner.py", "--level", "all", "--output", output_dir]

    try:
        benchmark_main()
    except SystemExit as e:
        if e.code != 0:
            logger.error("Benchmark runner failed!")
            sys.exit(e.code)
    except Exception as e:
        logger.exception(f"Benchmark runner crashed: {e}")
        sys.exit(1)

    return Path(output_dir)

def collect_artifacts(run_dir: Path):
    """Copy critical artifacts to reports/ folder."""
    logger.info("Collecting artifacts...")

    # 1. benchmark_results.json (summary snapshot)
    summary_src = run_dir / "summary.json"
    if summary_src.exists():
        shutil.copy(summary_src, REPORTS_DIR / "benchmark_results.json")
        # Also save a timestamped snapshot
        shutil.copy(summary_src, SNAPSHOTS_DIR / f"benchmark_results_{int(time.time())}.json")

    # 2. stage_metrics.json (summary CSV converted or raw)
    # The runner produces summary.csv, let's copy that too
    summary_csv = run_dir / "summary.csv"
    if summary_csv.exists():
        shutil.copy(summary_csv, REPORTS_DIR / "summary.csv")

    # 3. Aggregated stage metrics from all levels
    stage_metrics = {}
    for f in run_dir.glob("*_metrics.json"):
        try:
            data = json.loads(f.read_text())
            key = f"{data.get('level', 'unknown')}_{data.get('name', 'unknown')}"
            stage_metrics[key] = data
        except Exception:
            pass

    with open(REPORTS_DIR / "stage_metrics.json", "w") as f:
        json.dump(stage_metrics, f, indent=2)

    # 4. Generate Health Report
    generate_health_report(stage_metrics)

    # 5. Generate Regression Flags
    generate_regression_flags(stage_metrics)

def generate_health_report(metrics: dict):
    """Generate Markdown health report."""
    lines = ["# Stage Health Report", "", f"Run Date: {time.ctime()}", ""]

    lines.append("## Level Summary")
    lines.append("| Level | Name | Note F1 | Onset MAE (ms) | Notes |")
    lines.append("|---|---|---|---|---|")

    for key, m in sorted(metrics.items()):
        f1 = m.get("note_f1", 0.0)
        mae = m.get("onset_mae_ms", "")
        if isinstance(mae, (float, int)):
            mae = f"{mae:.1f}"
        else:
            mae = "-"
        count = m.get("predicted_count", 0)
        lines.append(f"| {m.get('level')} | {m.get('name')} | {f1:.3f} | {mae} | {count} |")

    lines.append("")
    lines.append("## Polyphonic Diagnostics")
    # Check L2, L3, L5, L6 specifically
    poly_levels = [k for k in metrics.keys() if any(x in k for x in ["L2", "L3", "L5", "L6"])]
    if not poly_levels:
        lines.append("_No polyphonic levels run._")
    else:
        for key in sorted(poly_levels):
            m = metrics[key]
            lines.append(f"### {key}")
            lines.append(f"- **Voiced Ratio**: {m.get('voiced_ratio', 0.0):.3f}")
            lines.append(f"- **Pitch Jump Rate**: {m.get('pitch_jump_rate_cents_sec', 0.0):.1f} cents/sec")
            lines.append(f"- **Vocal Band Ratio**: {m.get('vocal_band_ratio', 0.0):.3f}")
            lines.append("")

    with open(REPORTS_DIR / "stage_health_report.md", "w") as f:
        f.write("\n".join(lines))

def generate_regression_flags(metrics: dict):
    """Generate Regression Flags report."""
    flags = []

    # Define thresholds
    MIN_F1 = 0.1  # Very low baseline to catch catastrophic failures
    MAX_MAE = 200.0

    for key, m in metrics.items():
        level = m.get("level", "")

        # 1. F1 Drop
        f1 = m.get("note_f1")
        if f1 is not None and f1 < MIN_F1 and level not in ["L4"]: # L4 doesn't have F1
            flags.append(f"🔴 {key}: F1 score {f1:.3f} is below safety threshold {MIN_F1}")

        # 2. Onset Accuracy
        mae = m.get("onset_mae_ms")
        if mae is not None and mae > MAX_MAE:
            flags.append(f"🟠 {key}: Onset MAE {mae:.1f}ms is high (> {MAX_MAE}ms)")

        # 3. Empty Output
        if m.get("predicted_count", 0) == 0:
            flags.append(f"🔴 {key}: Zero notes predicted!")

        # 4. Fragmentation
        frag = m.get("fragmentation_score", 0.0)
        if frag > 0.5:
             flags.append(f"🟡 {key}: High fragmentation score {frag:.2f}")

    content = ["# Regression Flags", "", f"Generated: {time.ctime()}", ""]
    if not flags:
        content.append("✅ No obvious regressions detected.")
    else:
        content.extend(flags)

    with open(REPORTS_DIR / "regression_flags.md", "w") as f:
        f.write("\n".join(content))

if __name__ == "__main__":
    setup_dirs()
    run_dir = run_benchmarks()
    collect_artifacts(run_dir)
    logger.info(f"Done. Reports generated in {REPORTS_DIR.absolute()}")
