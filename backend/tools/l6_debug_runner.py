#!/usr/bin/env python3
"""
tools/l6_debug_runner.py

FIXED + HARDENED VERSION

L6 ("Synthetic Pop Song") debug runner with:
- deterministic synth generation
- baseline run + metrics + artifacts
- parameter sweep for Stage C + quality gate
- **FORCED polyphonic routing for your current transcribe() implementation**
  (sets config.stage_b.transcription_mode = "classic_song")
- config-path setter that can create missing dict nodes + leaf attrs
- safe handling of decision_trace when it is not a dict
- clean error if synth backend (soundfile/libsndfile) is missing
- prints sweep results + effective applied params (so you can prove sweeps are real)

Run (repo root):
  python tools/l6_debug_runner.py --reports reports --tag l6_debug --device cpu
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
from typing import Any, Dict, List, Tuple

import numpy as np

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
    return {}


def _cfg_set_path(cfg: Any, path: str, value: Any) -> bool:
    """
    Set cfg field given dotted path, supporting dataclasses/objects and dicts.
    Creates dict nodes if needed.

    FIXES:
    - allows setting missing LEAF attributes (setattr on last segment)
    - creates missing INTERMEDIATE nodes as dict buckets (e.g., post_merge)
    """
    parts = path.split(".")
    cur = cfg
    for i, p in enumerate(parts):
        last = (i == len(parts) - 1)

        # dict node
        if isinstance(cur, dict):
            if last:
                cur[p] = value
                return True
            if p not in cur or cur[p] is None:
                cur[p] = {}
            cur = cur[p]
            continue

        # object node
        if last:
            try:
                setattr(cur, p, value)
                return True
            except Exception:
                return False

        # intermediate attribute might not exist -> create dict bucket
        if not hasattr(cur, p):
            try:
                setattr(cur, p, {})
            except Exception:
                return False

        nxt = getattr(cur, p)
        if nxt is None:
            try:
                setattr(cur, p, {})
                nxt = getattr(cur, p)
            except Exception:
                return False

        cur = nxt

    return False


def _safe_decision_trace(diag: Any) -> Dict[str, Any]:
    dt = diag.get("decision_trace", {}) if isinstance(diag, dict) else {}
    if isinstance(dt, dict):
        return dt
    return {"raw": dt}


def _extract_notes_tuples(analysis) -> List[Tuple[int, float, float]]:
    notes = getattr(analysis, "notes_before_quantization", None) or getattr(analysis, "notes", None) or []
    out: List[Tuple[int, float, float]] = []
    for n in notes:
        out.append(
            (
                int(getattr(n, "midi_note", 0)),
                float(getattr(n, "start_sec", 0.0)),
                float(getattr(n, "end_sec", 0.0)),
            )
        )
    return out


def _voiced_ratio_from_analysis(analysis) -> float:
    """Prefer timeline-based voiced ratio if available; else approximate by note coverage."""
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


def _compute_metrics(
    level: str,
    name: str,
    pred_list: List[Tuple[int, float, float]],
    gt: List[Tuple[int, float, float]],
    voiced_ratio: float,
) -> Dict[str, Any]:
    f1 = note_f1(pred_list, gt, onset_tol=0.05)
    onset_mae, offset_mae = onset_offset_mae(pred_list, gt)
    dtw_f1 = dtw_note_f1(pred_list, gt, onset_tol=0.05)
    dtw_onset_ms = dtw_onset_error_ms(pred_list, gt)

    symptoms = compute_symptom_metrics(pred_list) or {}
    return {
        "level": level,
        "name": name,
        "note_f1": f1,
        "onset_mae_ms": onset_mae * 1000 if onset_mae is not None else None,
        "offset_mae_ms": offset_mae * 1000 if offset_mae is not None else None,
        "dtw_note_f1": dtw_f1,
        "dtw_onset_error_ms": dtw_onset_ms,
        "predicted_count": int(len(pred_list)),
        "gt_count": int(len(gt)),
        "voiced_ratio": float(voiced_ratio),
        "note_count": int(len(pred_list)),
        **symptoms,
    }


def _render_accuracy_report(metrics: Dict[str, Any], run_info: Dict[str, Any]) -> str:
    f1 = metrics.get("note_f1")
    onset = metrics.get("onset_mae_ms")
    vr = metrics.get("voiced_ratio")
    frag = metrics.get("fragmentation_score")
    ocr = metrics.get("octave_jump_rate")
    nps = metrics.get("note_count_per_10s")

    stage_c_post = run_info.get("stage_c_post") or {}
    snap_ms = stage_c_post.get("snap_tol_ms")
    merge_gap_ms = stage_c_post.get("merge_gap_ms")

    lines: List[str] = []
    lines.append("# L6 Accuracy Report\n")
    lines.append("## Headline metrics\n")
    lines.append(f"- note_f1: {f1}")
    lines.append(f"- onset_mae_ms: {onset}")
    lines.append(f"- voiced_ratio: {vr}")
    lines.append(f"- note_count: {metrics.get('note_count')}")
    lines.append(f"- fragmentation_score: {frag}")
    lines.append(f"- octave_jump_rate: {ocr}")
    lines.append(f"- note_count_per_10s: {nps}")
    lines.append("")
    lines.append("## Effective Stage C knobs (from diagnostics)\n")
    lines.append(f"- stage_c.chord_onset_snap_ms (effective): {snap_ms}")
    lines.append(f"- stage_c.post_merge.max_gap_ms / gap_filling.max_gap_ms (effective): {merge_gap_ms}")
    lines.append("")

    diag_dt = run_info.get("decision_trace") or {}
    if not isinstance(diag_dt, dict):
        diag_dt = {"raw": diag_dt}
    resolved = diag_dt.get("resolved", {})

    lines.append("## Resolved routing (from decision_trace)\n")
    lines.append("```json")
    lines.append(json.dumps(resolved, indent=2, default=str))
    lines.append("```")
    lines.append("")

    # Failure heuristics
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
        for fm in fail_modes[:4]:
            lines.append(f"- {fm}")
    else:
        lines.append("- No dominant failure mode detected by heuristics.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_gate_matrix(run_info: Dict[str, Any]) -> Dict[str, Any]:
    dt = run_info.get("decision_trace") or {}
    if not isinstance(dt, dict):
        dt = {"raw": dt}
    return {
        "decision_trace": dt,
        "quality_gate": run_info.get("quality_gate") or {},
        "stage_c_post": run_info.get("stage_c_post") or {},
    }


def _print_file(path: str, title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    try:
        with open(path, "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print(f"(could not read {path}: {e})")


def _force_l6_mode_on_cfg(cfg: Any) -> None:
    """
    HARD FORCE for your current backend.pipeline.transcribe() routing:

    transcribe() calls:
      requested_mode = getattr(config.stage_b, "transcription_mode", "auto")

    So for L6 we *must* set:
      config.stage_b.transcription_mode = "classic_song"
    """
    try:
        if hasattr(cfg, "stage_b"):
            setattr(cfg.stage_b, "transcription_mode", "classic_song")
        else:
            cfg["stage_b"]["transcription_mode"] = "classic_song"
    except Exception:
        # Don't crash debug runner; the decision_trace will show what happened.
        pass


def _run_transcribe(wav_path: str, cfg: PipelineConfig, device: str, pipeline_logger: PipelineLogger):
    """
    Call transcribe() safely for this repo version.

    ✅ Tiny but recommended cleanup:
    Force intent explicitly via transcribe(..., requested_mode="classic_song")
    so the debug runner cannot silently regress if config routing changes.
    """
    return transcribe(
        wav_path,
        config=cfg,
        pipeline_logger=pipeline_logger,
        device=device,
        requested_mode="classic_song",
    )


# -------------------------------
# Main
# -------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports", help="Reports folder")
    ap.add_argument("--tag", default="l6_debug", help="Run tag")
    ap.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    ap.add_argument("--seed", type=int, default=123, help="Determinism seed")
    ap.add_argument("--max-combos", type=int, default=9, help="Max sweep combinations to run (cap)")
    ap.add_argument("--duration-sec", type=float, default=60.0, help="Synth duration")
    ap.add_argument("--tempo-bpm", type=float, default=110.0, help="Synth tempo")
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

    # Import synth lazily so missing soundfile doesn't crash at import time
    try:
        from backend.benchmarks.ladder.synth import midi_to_wav_synth
    except Exception as e:
        print("ERROR: Could not import backend.benchmarks.ladder.synth.midi_to_wav_synth.")
        print("Most common cause: missing 'soundfile' (pip install soundfile) and system 'libsndfile'.")
        print(f"Import error: {e}")
        return 2

    # ---- Generate L6 audio + GT ----
    sr = 22050
    score = create_pop_song_base(duration_sec=float(args.duration_sec), tempo_bpm=float(args.tempo_bpm), seed=0)
    suite = BenchmarkSuite(output_dir=run_dir)

    gt = suite._score_to_gt(score, parts=["Lead"])
    if not gt:
        raise RuntimeError("GT is empty. Lead part selection likely failed (score_to_gt parts=['Lead']).")

    wav_path = os.path.join(run_dir, "L6_synthetic_pop_song.wav")
    logger.info("Rendering audio to %s", wav_path)
    midi_to_wav_synth(score, wav_path, sr=sr)

    # ---- Baseline config (match L6 intent) ----
    base_cfg = PipelineConfig()

    # ✅ FIX 1: FORCE L6 MODE (safety net)
    _force_l6_mode_on_cfg(base_cfg)

    # Avoid Demucs on synthetic benches (as in benchmark_runner L6)
    try:
        base_cfg.stage_b.separation["enabled"] = False
    except Exception:
        pass

    # Encourage lead tracking in a poly mix: limit band to typical lead range
    try:
        base_cfg.stage_b.melody_filtering.update(
            {"fmin_hz": 180.0, "fmax_hz": 1600.0, "voiced_prob_threshold": 0.40}
        )
    except Exception:
        pass

    # Ensure Stage C is set to extract top voice (Lead-only GT)
    try:
        base_cfg.stage_c.polyphony_filter = {"mode": "skyline_top_voice"}
    except Exception:
        pass

    # chord snap default (sweep will override)
    _cfg_set_path(base_cfg, "stage_c.chord_onset_snap_ms", 25.0)

    # Ensure quality_gate exists as optional dict (transcribe() reads getattr)
    if not hasattr(base_cfg, "quality_gate"):
        try:
            setattr(base_cfg, "quality_gate", {"enabled": True, "threshold": 0.45, "max_candidates": 3})
        except Exception:
            pass

    # ---- Baseline run ----
    logger.info("Running baseline transcribe() ...")
    plog = PipelineLogger()
    t0 = time.time()
    tr = _run_transcribe(wav_path, cfg=base_cfg, device=args.device, pipeline_logger=plog)
    elapsed = time.time() - t0

    analysis = tr.analysis_data
    pred_list = _extract_notes_tuples(analysis)
    voiced_ratio = _voiced_ratio_from_analysis(analysis)

    metrics = _compute_metrics("L6", "synthetic_pop_song_lead", pred_list, gt, voiced_ratio)
    metrics["timing_total_sec"] = float(elapsed)

    diag = getattr(analysis, "diagnostics", {}) or {}
    dt = _safe_decision_trace(diag)

    run_info = {
        "level": "L6",
        "name": "synthetic_pop_song_lead",
        "wav_path": wav_path,
        "duration_sec": float(getattr(getattr(analysis, "meta", None), "duration_sec", 0.0) or 0.0),
        "note_count": int(len(pred_list)),
        "voiced_ratio": float(voiced_ratio),
        "decision_trace": dt,
        "quality_gate": diag.get("quality_gate", {}) if isinstance(diag, dict) else {},
        "stage_c_post": diag.get("stage_c_post", {}) if isinstance(diag, dict) else {},
        "timing": diag.get("timing", {"total_sec": float(elapsed)}) if isinstance(diag, dict) else {"total_sec": float(elapsed)},
        "config": _asdict_safe(base_cfg),
    }

    # Save baseline snapshot files
    base_metrics_path = os.path.join(run_dir, "L6_synthetic_pop_song_metrics.json")
    base_run_info_path = os.path.join(run_dir, "L6_synthetic_pop_song_run_info.json")
    with open(base_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(base_run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, default=str)

    # Repo-level report artifacts
    bench_results_path = os.path.join(reports_root, "benchmark_results.json")
    with open(bench_results_path, "w", encoding="utf-8") as f:
        json.dump([metrics], f, indent=2, default=str)

    gate_matrix = _build_gate_matrix(run_info)
    gate_matrix_path = os.path.join(reports_root, "l6_gate_matrix.json")
    with open(gate_matrix_path, "w", encoding="utf-8") as f:
        json.dump(gate_matrix, f, indent=2, default=str)

    accuracy_report = _render_accuracy_report(metrics, run_info)
    accuracy_report_path = os.path.join(reports_root, "l6_accuracy_report.md")
    with open(accuracy_report_path, "w", encoding="utf-8") as f:
        f.write(accuracy_report)

    # ---- Sweep ----
    logger.info("Running mini sweep (capped to %d combos) ...", int(args.max_combos))
    max_gap_ms_values = [40.0, 60.0, 80.0]
    snap_ms_values = [15.0, 25.0, 35.0]
    q_thr_values = [0.35, 0.45, 0.55]

    combos = list(itertools.product(max_gap_ms_values, snap_ms_values, q_thr_values))
    combos = combos[: max(0, int(args.max_combos))]

    sweep_rows: List[Dict[str, Any]] = []
    for (gap_ms, snap_ms, thr) in combos:
        cfg = copy.deepcopy(base_cfg)

        # ✅ FIX 2: force L6 mode for every sweep candidate (paranoia)
        _force_l6_mode_on_cfg(cfg)

        # Stage C knobs (both new + legacy paths)
        _cfg_set_path(cfg, "stage_c.post_merge.max_gap_ms", float(gap_ms))
        _cfg_set_path(cfg, "stage_c.gap_filling.max_gap_ms", float(gap_ms))  # legacy fallback
        _cfg_set_path(cfg, "stage_c.chord_onset_snap_ms", float(snap_ms))

        # Quality gate knob
        qg = getattr(cfg, "quality_gate", None)
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
        tr_s = _run_transcribe(wav_path, cfg=cfg, device=args.device, pipeline_logger=PipelineLogger())
        dt_s = time.time() - t1

        analysis_s = tr_s.analysis_data
        pred_s = _extract_notes_tuples(analysis_s)
        vr_s = _voiced_ratio_from_analysis(analysis_s)
        met_s = _compute_metrics("L6", "synthetic_pop_song_lead", pred_s, gt, vr_s)
        met_s["timing_total_sec"] = float(dt_s)

        diag_s = getattr(analysis_s, "diagnostics", {}) or {}
        stage_c_post_s = diag_s.get("stage_c_post", {}) if isinstance(diag_s, dict) else {}
        qg_s = diag_s.get("quality_gate", {}) if isinstance(diag_s, dict) else {}
        selected = qg_s.get("selected_candidate_id", None)

        # Effective values proof (so you can confirm sweep is real)
        eff_snap = stage_c_post_s.get("snap_tol_ms", None)
        eff_gap = stage_c_post_s.get("merge_gap_ms", None)

        sweep_rows.append(
            {
                "overrides": {
                    "stage_c.post_merge.max_gap_ms": float(gap_ms),
                    "stage_c.chord_onset_snap_ms": float(snap_ms),
                    "quality_gate.threshold": float(thr),
                },
                "effective": {
                    "stage_c_post.snap_tol_ms": eff_snap,
                    "stage_c_post.merge_gap_ms": eff_gap,
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
                    "timing_total_sec": met_s.get("timing_total_sec"),
                },
            }
        )

    sweep_path = os.path.join(reports_root, "l6_sweep_results.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "baseline": {"metrics": metrics, "run_info_path": base_run_info_path},
                "sweep": sweep_rows,
            },
            f,
            indent=2,
            default=str,
        )

    # ---- Print key reports to stdout ----
    _print_file(gate_matrix_path, "reports/l6_gate_matrix.json")
    _print_file(accuracy_report_path, "reports/l6_accuracy_report.md")
    _print_file(sweep_path, "reports/l6_sweep_results.json")

    logger.info("Done. Snapshot: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
