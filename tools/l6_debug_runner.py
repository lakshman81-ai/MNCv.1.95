#!/usr/bin/env python3
"""
tools/l6_debug_runner.py

Specialized runner for Benchmark L6 (Synthetic Pop Song) with:
- Baseline run (writes snapshot run_info + metrics)
- Mini sweep over Stage C post-merge + chord snap + quality gate threshold (writes reports/l6_sweep_results.json)
- Report generation:
    - reports/l6_gate_matrix.json
    - reports/l6_accuracy_report.md
- Prints the generated reports to stdout.

Run (repo root):
  python tools/l6_debug_runner.py --reports reports --tag l6_debug

Notes:
- Uses the same L6 generator as backend.benchmarks.benchmark_runner (create_pop_song_base + midi_to_wav_synth).
- Runs the pipeline via backend.pipeline.transcribe.transcribe so AnalysisData.diagnostics includes:
    decision_trace, quality_gate, stage_c_post (if you applied the stage patches).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf  # noqa: F401
except Exception:
    sf = None  # type: ignore


# -------------------------------
# Path bootstrap
# -------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# -------------------------------
# Imports from repo
# -------------------------------
from backend.pipeline.config import PipelineConfig
from backend.pipeline.instrumentation import PipelineLogger
from backend.pipeline.transcribe import transcribe

from backend.benchmarks.metrics import (
    note_f1,
    onset_offset_mae,
    dtw_note_f1,
    dtw_onset_error_ms,
    compute_symptom_metrics,
)

# We intentionally import L6 generator from benchmark_runner to match the benchmark.
from backend.benchmarks.benchmark_runner import BenchmarkSuite, create_pop_song_base
from backend.benchmarks.ladder.synth import midi_to_wav_synth


# -------------------------------
# Helpers
# -------------------------------
def _now_run_id(tag: str = "") -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{tag}" if tag else base


def _safe_mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _is_dataclass_instance(x: Any) -> bool:
    try:
        return dataclasses.is_dataclass(x)
    except Exception:
        return False


def _asdict_safe(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if _is_dataclass_instance(x):
        try:
            return dataclasses.asdict(x)
        except Exception:
            return {}
    # best-effort
    return {}


def _cfg_set_path(cfg: Any, path: str, value: Any) -> None:
    """
    Set cfg field given dotted path, supporting dataclasses/objects and dicts.
    Creates dict nodes if needed. Does not create new dataclass nodes.
    """
    parts = path.split(".")
    cur = cfg
    for i, p in enumerate(parts):
        last = (i == len(parts) - 1)
        if isinstance(cur, dict):
            if last:
                cur[p] = value
                return
            if p not in cur or cur[p] is None:
                cur[p] = {}
            cur = cur[p]
            continue

        # object attribute
        if not hasattr(cur, p):
            # if we can't set deeper, stop
            return
        if last:
            try:
                setattr(cur, p, value)
            except Exception:
                pass
            return
        nxt = getattr(cur, p)
        if nxt is None:
            # if it's a dict-like config field, create dict
            try:
                setattr(cur, p, {})
                nxt = getattr(cur, p)
            except Exception:
                return
        cur = nxt


def _voiced_ratio_from_analysis(analysis) -> float:
    """
    Prefer timeline-based voiced ratio if timeline frames carry active_pitches;
    otherwise approximate by note coverage.
    """
    try:
        timeline = getattr(analysis, "timeline", None)
        if timeline:
            total = max(1, len(timeline))
            voiced = 0
            for fr in timeline:
                ap = getattr(fr, "active_pitches", None)
                if ap is not None and len(ap) > 0:
                    voiced += 1
            return float(voiced / total)
    except Exception:
        pass

    # coverage fallback
    try:
        meta = getattr(analysis, "meta", None)
        dur = float(getattr(meta, "duration_sec", 0.0) or 0.0)
        notes = getattr(analysis, "notes_before_quantization", None) or getattr(analysis, "notes", None) or []
        if dur <= 0.0:
            return 0.0
        tot = 0.0
        for n in notes:
            s = float(getattr(n, "start_sec", 0.0))
            e = float(getattr(n, "end_sec", 0.0))
            tot += max(0.0, e - s)
        return float(min(1.0, tot / max(1e-6, dur)))
    except Exception:
        return 0.0


def _extract_notes_tuples(analysis) -> List[Tuple[int, float, float]]:
    notes = getattr(analysis, "notes_before_quantization", None) or getattr(analysis, "notes", None) or []
    out = []
    for n in notes:
        out.append((int(getattr(n, "midi_note", 0)), float(getattr(n, "start_sec", 0.0)), float(getattr(n, "end_sec", 0.0))))
    return out


def _compute_metrics(level: str, name: str, pred_list: List[Tuple[int, float, float]], gt: List[Tuple[int, float, float]], voiced_ratio: float) -> Dict[str, Any]:
    f1 = note_f1(pred_list, gt, onset_tol=0.05)
    onset_mae, offset_mae = onset_offset_mae(pred_list, gt)
    dtw_f1 = dtw_note_f1(pred_list, gt, onset_tol=0.05)
    dtw_onset_ms = dtw_onset_error_ms(pred_list, gt)

    symptoms = compute_symptom_metrics(pred_list)
    metrics = {
        "level": level,
        "name": name,
        "note_f1": f1,
        "onset_mae_ms": onset_mae * 1000 if onset_mae is not None else None,
        "dtw_note_f1": dtw_f1,
        "dtw_onset_error_ms": dtw_onset_ms,
        "predicted_count": int(len(pred_list)),
        "gt_count": int(len(gt)),
        "vocal_band_ratio": None,            # not computed in this debug runner
        "pitch_jump_rate_cents_sec": None,   # not computed in this debug runner
        "voiced_ratio": float(voiced_ratio),
        "voicing_ratio": float(voiced_ratio),
        "note_density": None,
        "note_count": int(len(pred_list)),
        **(symptoms or {}),
    }
    return metrics


def _render_accuracy_report(metrics: Dict[str, Any], run_info: Dict[str, Any]) -> str:
    f1 = metrics.get("note_f1")
    onset = metrics.get("onset_mae_ms")
    vr = metrics.get("voiced_ratio")
    frag = metrics.get("fragmentation_score")
    med = metrics.get("median_note_len_ms")
    ocr = metrics.get("octave_jump_rate")
    nps = metrics.get("note_count_per_10s")

    lines: List[str] = []
    lines.append("# L6 Accuracy Report\n")
    lines.append("## Headline metrics\n")
    lines.append(f"- note_f1: {f1}")
    lines.append(f"- onset_mae_ms: {onset}")
    lines.append(f"- voiced_ratio: {vr}")
    lines.append(f"- note_count: {metrics.get('note_count')}")
    lines.append(f"- fragmentation_score: {frag}")
    lines.append(f"- median_note_len_ms: {med}")
    lines.append(f"- octave_jump_rate: {ocr}")
    lines.append(f"- note_count_per_10s: {nps}")
    lines.append("")

    # Failure mode heuristic
    fail_modes = []
    try:
        if isinstance(vr, (int, float)) and vr < 0.35:
            fail_modes.append("Under-transcription / voicing loss (low voiced_ratio)")
        if isinstance(frag, (int, float)) and frag > 0.45:
            fail_modes.append("Over-fragmentation (high fragmentation_score)")
        if isinstance(nps, (int, float)) and nps > 120:
            fail_modes.append("Over-transcription (very high note density)")
        if isinstance(ocr, (int, float)) and ocr > 0.25:
            fail_modes.append("Octave instability (high octave_jump_rate)")
        if isinstance(onset, (int, float)) and onset > 80:
            fail_modes.append("Timing error (high onset MAE)")
    except Exception:
        pass

    lines.append("## Likely failure modes\n")
    if fail_modes:
        for fm in fail_modes[:3]:
            lines.append(f"- {fm}")
    else:
        lines.append("- No dominant failure mode detected by heuristics.")
    lines.append("")

    # Gate sensitivity recommendations
    dt = (run_info.get("decision_trace") or {})
    rf = (dt.get("routing_features") or {})
    sep = (dt.get("separation") or {})
    qg = (run_info.get("quality_gate") or {})

    lines.append("## Gate sensitivity recommendations (no code edits)\n")
    lines.append("| Gate / knob | Current | Observed | Direction | Suggestion |")
    lines.append("|---|---:|---:|---|---|")

    # Stage C knobs
    stage_c_post = run_info.get("stage_c_post") or {}
    lines.append(f"| stage_c.post_merge.max_gap_ms | {stage_c_post.get('merge_gap_ms')} | frag={frag} med_ms={med} | {'increase' if (isinstance(frag,(int,float)) and frag>0.45) else 'keep'} | try 60→80ms if fragmented |")
    lines.append(f"| stage_c.chord_onset_snap_ms | {stage_c_post.get('snap_tol_ms')} | octave_jump={ocr} onset_mae={onset} | {'increase' if (isinstance(onset,(int,float)) and onset>80) else 'keep'} | try 25→35ms if chord onsets smear |")

    # Separation knobs
    lines.append(f"| stage_b.separation.gates.min_mixture_score | {sep.get('gates',{}).get('min_mixture_score')} | mixture_score={rf.get('mixture_score')} | lower if skipping wrongly | try 0.45→0.35 for borderline mixes |")

    # Quality gate knobs
    lines.append(f"| quality_gate.threshold | {qg.get('threshold')} | selected={qg.get('selected_candidate_id')} score={qg.get('candidates',[{}])[0].get('score') if qg.get('candidates') else None} | lower if rejecting good candidate | try 0.45→0.35 if many rejects |")

    lines.append("")
    return "\n".join(lines) + "\n"


def _build_gate_matrix(run_info: Dict[str, Any]) -> Dict[str, Any]:
    dt = run_info.get("decision_trace") or {}
    out = {
        "stage_b_routing": {
            "requested": (dt.get("requested") or {}),
            "resolved": (dt.get("resolved") or {}),
            "routing_features": (dt.get("routing_features") or {}),
            "rule_hits": (dt.get("rule_hits") or []),
        },
        "separation": (dt.get("separation") or {}),
        "quality_gate": (run_info.get("quality_gate") or {}),
        "stage_c_post": (run_info.get("stage_c_post") or {}),
    }
    return out


def _print_file(path: str, title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    try:
        with open(path, "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print(f"(could not read {path}: {e})")


def _verify_artifacts(expected_paths: List[str]) -> List[Tuple[str, bool]]:
    results = []
    for p in expected_paths:
        results.append((p, os.path.exists(p)))
    return results


# -------------------------------
# Main
# -------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports", help="Reports folder")
    ap.add_argument("--tag", default="l6_debug", help="Run tag")
    ap.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    ap.add_argument("--seed", type=int, default=123, help="Determinism seed")
    ap.add_argument("--max-combos", type=int, default=9, help="Max sweep combinations to run (cap)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    reports_root = os.path.abspath(args.reports)
    snapshots_root = _safe_mkdir(os.path.join(reports_root, "snapshots"))
    run_id = _now_run_id(args.tag)
    run_dir = _safe_mkdir(os.path.join(snapshots_root, run_id))

    # Logging to reports/bench_run.log
    log_path = os.path.join(reports_root, "bench_run.log")
    _safe_mkdir(reports_root)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("l6_debug_runner")
    logger.info("Run ID: %s", run_id)
    logger.info("Run dir: %s", run_dir)

    # ---- Generate L6 audio + GT ----
    sr = 22050
    score = create_pop_song_base(duration_sec=60.0, tempo_bpm=110, seed=0)
    suite = BenchmarkSuite(output_dir=run_dir)
    gt = suite._score_to_gt(score, parts=["Lead"])

    wav_path = os.path.join(run_dir, "L6_synthetic_pop_song.wav")
    logger.info("Rendering audio to %s", wav_path)
    midi_to_wav_synth(score, wav_path, sr=sr)

    # ---- Baseline config (align with benchmark_runner L6 intent) ----
    base_cfg = PipelineConfig()
    try:
        base_cfg.stage_b.separation["enabled"] = False
    except Exception:
        pass

    try:
        base_cfg.stage_b.melody_filtering.update({"fmin_hz": 180.0, "fmax_hz": 1600.0, "voiced_prob_threshold": 0.40})
    except Exception:
        pass

    try:
        for det in ["rmvpe", "crepe", "swiftf0", "yin"]:
            if det in getattr(base_cfg.stage_b, "detectors", {}):
                base_cfg.stage_b.detectors[det]["enabled"] = True
    except Exception:
        pass

    # Ensure quality_gate structure exists (transcribe() reads it if implemented)
    if not hasattr(base_cfg, "quality_gate"):
        try:
            setattr(base_cfg, "quality_gate", {"enabled": True, "threshold": 0.45, "max_candidates": 3})
        except Exception:
            pass
    else:
        try:
            qg = getattr(base_cfg, "quality_gate", {}) or {}
            qg.setdefault("enabled", True)
            qg.setdefault("threshold", 0.45)
            qg.setdefault("max_candidates", 3)
            setattr(base_cfg, "quality_gate", qg)
        except Exception:
            pass

    # ---- Baseline run via transcribe() ----
    logger.info("Running baseline transcribe() ...")
    plog = PipelineLogger()
    t0 = time.time()
    tr = transcribe(wav_path, config=base_cfg, pipeline_logger=plog, device=args.device)
    elapsed = time.time() - t0

    analysis = tr.analysis_data
    pred_list = _extract_notes_tuples(analysis)
    voiced_ratio = _voiced_ratio_from_analysis(analysis)

    metrics = _compute_metrics("L6", "synthetic_pop_song_lead", pred_list, gt, voiced_ratio)
    metrics["timing_total_sec"] = float(elapsed)

    # run_info with full diagnostics
    diag = getattr(analysis, "diagnostics", {}) or {}
    run_info = {
        "level": "L6",
        "name": "synthetic_pop_song_lead",
        "wav_path": wav_path,
        "duration_sec": float(getattr(getattr(analysis, "meta", None), "duration_sec", 0.0) or 0.0),
        "note_count": int(len(pred_list)),
        "voiced_ratio": float(voiced_ratio),
        "decision_trace": diag.get("decision_trace", {}),
        "quality_gate": diag.get("quality_gate", {}),
        "stage_c_post": diag.get("stage_c_post", {}),
        "timing": diag.get("timing", {"total_sec": float(elapsed)}),
        "config": _asdict_safe(base_cfg),
    }

    # Save baseline snapshot files (requested names)
    base_metrics_path = os.path.join(run_dir, "L6_synthetic_pop_song_metrics.json")
    base_run_info_path = os.path.join(run_dir, "L6_synthetic_pop_song_run_info.json")
    with open(base_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(base_run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, default=str)

    # Also write benchmark-like files for consistency
    bench_results_path = os.path.join(reports_root, "benchmark_results.json")
    with open(bench_results_path, "w", encoding="utf-8") as f:
        json.dump([metrics], f, indent=2, default=str)

    stage_metrics_path = os.path.join(reports_root, "stage_metrics.json")
    stage_metrics = {"n_runs": 1, "by_stage": {"total": {"mean_s": float(elapsed), "p50_s": float(elapsed), "p90_s": float(elapsed), "n": 1}}}
    with open(stage_metrics_path, "w", encoding="utf-8") as f:
        json.dump(stage_metrics, f, indent=2, default=str)

    stage_health_path = os.path.join(reports_root, "stage_health_report.md")
    with open(stage_health_path, "w", encoding="utf-8") as f:
        f.write("# Stage Health Report\n\n")
        f.write(f"Run ID: `{run_id}`\n\n")
        f.write(f"- L6 note_f1: {metrics.get('note_f1')}\n")
        f.write(f"- L6 voiced_ratio: {metrics.get('voiced_ratio')}\n")
        f.write(f"- L6 fragmentation_score: {metrics.get('fragmentation_score')}\n")

    regression_flags_path = os.path.join(reports_root, "regression_flags.md")
    with open(regression_flags_path, "w", encoding="utf-8") as f:
        f.write("# Regression Flags\n\nNo baseline provided in l6_debug_runner.\n")

    # ---- Reports requested: gate matrix + accuracy report ----
    gate_matrix = _build_gate_matrix(run_info)
    gate_matrix_path = os.path.join(reports_root, "l6_gate_matrix.json")
    with open(gate_matrix_path, "w", encoding="utf-8") as f:
        json.dump(gate_matrix, f, indent=2, default=str)

    accuracy_report = _render_accuracy_report(metrics, run_info)
    accuracy_report_path = os.path.join(reports_root, "l6_accuracy_report.md")
    with open(accuracy_report_path, "w", encoding="utf-8") as f:
        f.write(accuracy_report)

    # ---- Sweep (mini) ----
    logger.info("Running mini sweep (capped to %d combos) ...", int(args.max_combos))
    max_gap_ms_values = [40.0, 60.0, 80.0]
    snap_ms_values = [15.0, 25.0, 35.0]
    q_thr_values = [0.35, 0.45, 0.55]

    combos = list(itertools.product(max_gap_ms_values, snap_ms_values, q_thr_values))
    combos = combos[: max(0, int(args.max_combos))]

    sweep_rows: List[Dict[str, Any]] = []
    for (gap_ms, snap_ms, thr) in combos:
        cfg = copy.deepcopy(base_cfg)

        # Stage C knobs (support both new and legacy config layouts)
        _cfg_set_path(cfg, "stage_c.post_merge.max_gap_ms", float(gap_ms))
        _cfg_set_path(cfg, "stage_c.gap_filling.max_gap_ms", float(gap_ms))
        _cfg_set_path(cfg, "stage_c.chord_onset_snap_ms", float(snap_ms))

        # Quality gate knob
        if hasattr(cfg, "quality_gate"):
            qg = getattr(cfg, "quality_gate", {}) or {}
            if isinstance(qg, dict):
                qg["enabled"] = True
                qg["threshold"] = float(thr)
                qg.setdefault("max_candidates", 3)
                setattr(cfg, "quality_gate", qg)
        else:
            try:
                setattr(cfg, "quality_gate", {"enabled": True, "threshold": float(thr), "max_candidates": 3})
            except Exception:
                pass

        t1 = time.time()
        tr_s = transcribe(wav_path, config=cfg, pipeline_logger=PipelineLogger(), device=args.device)
        dt = time.time() - t1

        analysis_s = tr_s.analysis_data
        pred_s = _extract_notes_tuples(analysis_s)
        vr_s = _voiced_ratio_from_analysis(analysis_s)
        met_s = _compute_metrics("L6", "synthetic_pop_song_lead", pred_s, gt, vr_s)
        met_s["timing_total_sec"] = float(dt)

        diag_s = getattr(analysis_s, "diagnostics", {}) or {}
        qg_s = diag_s.get("quality_gate", {}) or {}
        selected = qg_s.get("selected_candidate_id", None)

        sweep_rows.append({
            "overrides": {
                "stage_c.post_merge.max_gap_ms": float(gap_ms),
                "stage_c.chord_onset_snap_ms": float(snap_ms),
                "quality_gate.threshold": float(thr),
            },
            "selected_candidate_id": selected,
            "metrics": {
                "note_f1": met_s.get("note_f1"),
                "onset_mae_ms": met_s.get("onset_mae_ms"),
                "voiced_ratio": met_s.get("voiced_ratio"),
                "fragmentation_score": met_s.get("fragmentation_score"),
                "median_note_len_ms": met_s.get("median_note_len_ms"),
                "note_count": met_s.get("note_count"),
                "octave_jump_rate": met_s.get("octave_jump_rate"),
            },
        })

    sweep_path = os.path.join(reports_root, "l6_sweep_results.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "baseline": {"metrics": metrics, "gate_matrix_path": gate_matrix_path},
            "sweep": sweep_rows,
        }, f, indent=2, default=str)

    # ---- Print reports to stdout ----
    _print_file(gate_matrix_path, "reports/l6_gate_matrix.json")
    _print_file(accuracy_report_path, "reports/l6_accuracy_report.md")

    # ---- Verify artifacts ----
    expected = [
        log_path,
        bench_results_path,
        stage_metrics_path,
        stage_health_path,
        regression_flags_path,
        gate_matrix_path,
        accuracy_report_path,
        base_run_info_path,
        base_metrics_path,
    ]
    checks = _verify_artifacts(expected)
    print("\n" + "=" * 80)
    print("ARTIFACT CHECK")
    print("=" * 80)
    ok_all = True
    for p, ok in checks:
        print(f"{'OK ' if ok else 'MISSING '} {p}")
        ok_all = ok_all and ok

    logger.info("Done. Snapshot: %s", run_dir)
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
