#!/usr/bin/env python3
"""
tools/run_agent_benchmarks.py

Runs the benchmark ladder and writes WI-required artifacts under ./reports/.

Usage:
  python tools/run_agent_benchmarks.py --level all
  python tools/run_agent_benchmarks.py --level L0,L1,L2 --baseline reports/snapshots/<older_run_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.benchmarks.benchmark_runner import BenchmarkSuite


DEFAULT_LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5.1", "L5.2", "L6"]


def _now_run_id() -> str:
    # Stable run id usable as folder name
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_levels(level_arg: str) -> List[str]:
    if not level_arg or level_arg.lower() == "all":
        return list(DEFAULT_LEVELS)
    parts = []
    for token in level_arg.split(","):
        t = token.strip()
        if not t:
            continue
        parts.append(t.upper() if t.upper().startswith("L") else t)
    # Keep deterministic order, respecting DEFAULT_LEVELS
    ordered = []
    for lvl in DEFAULT_LEVELS:
        if lvl in parts:
            ordered.append(lvl)
    for lvl in parts:
        if lvl not in ordered:
            ordered.append(lvl)
    return ordered


def _collect_case_files(run_dir: str) -> List[Tuple[str, str]]:
    """Return list of (case_id, metrics_path) for files like L0_name_metrics.json."""
    out = []
    for fn in os.listdir(run_dir):
        if fn.endswith("_metrics.json"):
            case_id = fn[:-len("_metrics.json")]
            out.append((case_id, os.path.join(run_dir, fn)))
    out.sort(key=lambda x: x[0])
    return out


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_stage_timings(run_dir: str) -> Dict[str, Any]:
    timings: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}

    for fn in os.listdir(run_dir):
        if not fn.endswith("_run_info.json"):
            continue
        info = _load_json(os.path.join(run_dir, fn))
        st = info.get("stage_timings", {}) or {}
        if not isinstance(st, dict):
            continue
        for k, v in st.items():
            try:
                fv = float(v)
            except Exception:
                continue
            timings.setdefault(k, []).append(fv)

    summary: Dict[str, Any] = {"by_stage": {}, "n_runs": 0}
    n_runs = len([fn for fn in os.listdir(run_dir) if fn.endswith("_run_info.json")])
    summary["n_runs"] = n_runs

    for stage, vals in timings.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mean = sum(vals_sorted) / len(vals_sorted)
        p50 = vals_sorted[len(vals_sorted) // 2]
        p90 = vals_sorted[int(0.9 * (len(vals_sorted) - 1))]
        summary["by_stage"][stage] = {
            "mean_s": mean,
            "p50_s": p50,
            "p90_s": p90,
            "n": len(vals_sorted),
        }
    return summary


def _render_health_report(results: List[Dict[str, Any]]) -> str:
    # A compact, human-readable report
    lines = []
    lines.append("# Stage Health Report")
    lines.append("")
    lines.append(f"Cases: {len(results)}")
    lines.append("")
    # Worst by note_f1 (if present)
    with_f1 = [r for r in results if isinstance(r.get("note_f1"), (int, float))]
    if with_f1:
        worst = sorted(with_f1, key=lambda r: r.get("note_f1", 0.0))[:10]
        lines.append("## Worst note_f1 (lowest 10)")
        for r in worst:
            lines.append(f"- {r.get('level')} / {r.get('name')}: note_f1={r.get('note_f1')}")
        lines.append("")
    # Fragmentation
    with_frag = [r for r in results if isinstance(r.get("fragmentation_score"), (int, float))]
    if with_frag:
        worst = sorted(with_frag, key=lambda r: r.get("fragmentation_score", 0.0), reverse=True)[:10]
        lines.append("## Worst fragmentation_score (highest 10)")
        for r in worst:
            lines.append(f"- {r.get('level')} / {r.get('name')}: frag={r.get('fragmentation_score')}")
        lines.append("")
    # Voiced ratio
    with_vr = [r for r in results if isinstance(r.get("voiced_ratio"), (int, float))]
    if with_vr:
        worst = sorted(with_vr, key=lambda r: r.get("voiced_ratio", 1.0))[:10]
        lines.append("## Worst voiced_ratio (lowest 10)")
        for r in worst:
            lines.append(f"- {r.get('level')} / {r.get('name')}: voiced_ratio={r.get('voiced_ratio')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _compare_dirs(a_dir: str, b_dir: str) -> Tuple[str, bool]:
    """Return (markdown, has_regressions)."""
    # Compare per-case metrics.json files
    def load_metrics_map(d: str) -> Dict[str, Dict[str, Any]]:
        m = {}
        for case_id, mp in _collect_case_files(d):
            try:
                m[case_id] = _load_json(mp)
            except Exception:
                continue
        return m

    a = load_metrics_map(a_dir)
    b = load_metrics_map(b_dir)

    common = sorted(set(a.keys()) & set(b.keys()))
    md = []
    md.append("# Regression Flags")
    md.append("")
    md.append(f"Baseline: `{a_dir}`")
    md.append(f"Current: `{b_dir}`")
    md.append("")
    if not common:
        md.append("No comparable cases found.")
        return "\n".join(md) + "\n", False

    regressions = []
    for cid in common:
        am = a[cid]
        bm = b[cid]
        # Note count
        a_nc = float(am.get("note_count", am.get("predicted_count", 0.0)) or 0.0)
        b_nc = float(bm.get("note_count", bm.get("predicted_count", 0.0)) or 0.0)
        # Voiced ratio
        a_vr = float(am.get("voiced_ratio", 0.0) or 0.0)
        b_vr = float(bm.get("voiced_ratio", 0.0) or 0.0)
        # Fragmentation (higher is worse)
        a_fr = float(am.get("fragmentation_score", 0.0) or 0.0)
        b_fr = float(bm.get("fragmentation_score", 0.0) or 0.0)
        # F1 (if exists)
        a_f1 = am.get("note_f1", None)
        b_f1 = bm.get("note_f1", None)

        flags = []
        # Heuristics
        if a_nc > 0 and (b_nc < 0.8 * a_nc) and ((a_nc - b_nc) >= 10):
            flags.append(f"note_count drop {a_nc:.0f}→{b_nc:.0f}")
        if (a_vr - b_vr) > 0.05:
            flags.append(f"voiced_ratio drop {a_vr:.3f}→{b_vr:.3f}")
        if (b_fr - a_fr) > 0.05:
            flags.append(f"fragmentation increase {a_fr:.3f}→{b_fr:.3f}")
        if isinstance(a_f1, (int, float)) and isinstance(b_f1, (int, float)) and (a_f1 - b_f1) > 0.02:
            flags.append(f"note_f1 drop {a_f1:.3f}→{b_f1:.3f}")

        if flags:
            regressions.append((cid, flags))

    if not regressions:
        md.append("✅ No regressions detected by heuristic gates.")
        return "\n".join(md) + "\n", False

    md.append("## Regressions")
    for cid, flags in regressions:
        md.append(f"- **{cid}**: " + "; ".join(flags))
    md.append("")
    md.append("> Note: These are heuristic gates (not definitive). Inspect per-case *_run_info.json and timeline CSVs for root cause.")
    return "\n".join(md) + "\n", True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="all", help="Comma list (L0,L1,...) or 'all'")
    ap.add_argument("--reports", default="reports", help="Reports output folder")
    ap.add_argument("--tag", default="", help="Optional tag appended to run id")
    ap.add_argument("--baseline", default="", help="Baseline snapshot directory to compare against (folder containing *_metrics.json)")
    ap.add_argument("--skip-real", action="store_true", help="Skip L4 real songs (recommended for CI)")
    args = ap.parse_args()

    reports_root = os.path.abspath(args.reports)
    os.makedirs(reports_root, exist_ok=True)
    os.makedirs(os.path.join(reports_root, "snapshots"), exist_ok=True)

    run_id = _now_run_id() + (f"_{args.tag}" if args.tag else "")
    run_dir = os.path.join(reports_root, "snapshots", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(reports_root, "bench_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("run_agent_benchmarks")
    logger.info("Run dir: %s", run_dir)

    levels = _resolve_levels(args.level)
    if args.skip_real and "L4" in levels:
        levels = [l for l in levels if l != "L4"]

    runner = BenchmarkSuite(output_dir=run_dir)

    # Run levels (deterministic order)
    for lvl in levels:
        try:
            logger.info("Running %s ...", lvl)
            if lvl == "L0":
                runner.run_L0_mono_sanity()
            elif lvl == "L1":
                runner.run_L1_mono_musical()
            elif lvl == "L2":
                runner.run_L2_poly_dominant()
            elif lvl == "L3":
                runner.run_L3_full_poly_musicxml()
            elif lvl == "L4":
                runner.run_L4_real_songs()
            elif lvl == "L5.1":
                runner.run_L5_1_kal_ho_na_ho()
            elif lvl == "L5.2":
                runner.run_L5_2_tumhare_hi_rahenge()
            elif lvl == "L6":
                runner.run_L6_synthetic_pop_song()
            else:
                logger.warning("Unknown level %s (skipping)", lvl)
        except Exception as exc:
            logger.exception("Level %s failed: %s", lvl, exc)

    runner.generate_summary()

    # Required top-level reports (always written)
    # 1) benchmark_results.json
    bench_results_path = os.path.join(reports_root, "benchmark_results.json")
    try:
        with open(bench_results_path, "w", encoding="utf-8") as f:
            json.dump(getattr(runner, "results", []), f, indent=2)
    except Exception as exc:
        logger.exception("Failed writing benchmark_results.json: %s", exc)

    # 2) stage_metrics.json
    stage_metrics_path = os.path.join(reports_root, "stage_metrics.json")
    try:
        stage_metrics = _aggregate_stage_timings(run_dir)
        with open(stage_metrics_path, "w", encoding="utf-8") as f:
            json.dump(stage_metrics, f, indent=2)
    except Exception as exc:
        logger.exception("Failed writing stage_metrics.json: %s", exc)

    # 3) stage_health_report.md
    health_path = os.path.join(reports_root, "stage_health_report.md")
    try:
        report_md = _render_health_report(getattr(runner, "results", []))
        with open(health_path, "w", encoding="utf-8") as f:
            f.write(report_md)
    except Exception as exc:
        logger.exception("Failed writing stage_health_report.md: %s", exc)

    # 4) regression_flags.md (even if baseline missing)
    regression_path = os.path.join(reports_root, "regression_flags.md")
    try:
        if args.baseline:
            md, has_reg = _compare_dirs(os.path.abspath(args.baseline), run_dir)
        else:
            md, has_reg = ("# Regression Flags\n\nNo baseline provided; skipping diff.\n", False)
        with open(regression_path, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception as exc:
        logger.exception("Failed writing regression_flags.md: %s", exc)

    logger.info("Done.")
    logger.info("Artifacts: %s", reports_root)
    return 2 if args.baseline and os.path.exists(args.baseline) and has_reg else 0


if __name__ == "__main__":
    raise SystemExit(main())
