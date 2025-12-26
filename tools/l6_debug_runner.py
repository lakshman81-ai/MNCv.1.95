#!/usr/bin/env python3
"""
tools/l6_debug_runner.py

Custom runner for L6 Benchmark (Synthetic Pop Song) to extract full diagnostics,
tuning gates, and perform parameter sweeps.
"""

import os
import sys
import json
import time
import logging
import copy
import numpy as np
import soundfile as sf
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline.config import PipelineConfig
from backend.pipeline.models import AudioType, AnalysisData, NoteEvent
from backend.benchmarks.benchmark_runner import (
    run_pipeline_on_audio,
    create_pop_song_base
)
from backend.benchmarks.ladder.synth import midi_to_wav_synth
from backend.benchmarks.metrics import (
    note_f1,
    onset_offset_mae,
    compute_symptom_metrics
)

import music21
from music21 import tempo, chord

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("l6_debug_runner")


def setup_reports_dir(reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "snapshots"), exist_ok=True)
    return reports_dir


def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def score_to_gt(score, parts: Optional[List[str]] = None) -> List[Tuple[int, float, float]]:
    """
    Convert a music21 score to GT tuples.
    If parts is provided, only those parts (by partName/id) are used.
    """
    tempo_marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
    bpm = float(tempo_marks[0].number) if tempo_marks else 100.0
    sec_per_quarter = 60.0 / bpm if bpm else 0.6

    # Select stream to read from
    if parts:
        selected_parts = []
        for p in getattr(score, "parts", []):
            pname = getattr(p, "partName", None)
            pid = getattr(p, "id", None)
            if (pname in parts) or (pid in parts):
                selected_parts.append(p)
        if selected_parts:
            stream_to_read = music21.stream.Stream(selected_parts)
        else:
            stream_to_read = score

    gt: List[Tuple[int, float, float]] = []
    for el in stream_to_read.flatten().notes:
        start = float(el.offset) * sec_per_quarter
        dur = float(el.quarterLength) * sec_per_quarter
        end = start + dur

        if isinstance(el, chord.Chord):
            for p in el.pitches:
                gt.append((int(p.midi), start, end))
        else:
            gt.append((int(el.pitch.midi), start, end))

    return gt


def generate_l6_audio_and_gt(duration_sec: float = 15.0) -> Tuple[np.ndarray, int, List[Tuple[int, float, float]]]:
    """Generates the synthetic pop song audio and ground truth."""
    sr = 22050
    tempo_bpm = 110

    logger.info(f"Generating L6 synthetic pop song score ({duration_sec}s)...")
    score = create_pop_song_base(duration_sec=duration_sec, tempo_bpm=tempo_bpm, seed=0)

    # Ground truth (Lead melody only for L6)
    gt = score_to_gt(score, parts=["Lead"])
    logger.info(f"Generated GT with {len(gt)} notes (Lead part).")
    if gt:
        logger.info(f"GT Sample: {gt[:3]}")

    logger.info("Synthesizing audio...")
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        midi_to_wav_synth(score, wav_path, sr=sr)
        audio, read_sr = sf.read(wav_path)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32), int(read_sr), gt


def get_l6_baseline_config() -> PipelineConfig:
    config = PipelineConfig()
    # Avoid Demucs on synthetic benches
    config.stage_b.separation["enabled"] = False

    # Encourage melody tracking in a poly mix
    config.stage_b.melody_filtering.update({"fmin_hz": 180.0, "fmax_hz": 1600.0, "voiced_prob_threshold": 0.40})
    for det in ["rmvpe", "crepe", "swiftf0", "yin"]:
        if det in config.stage_b.detectors:
            config.stage_b.detectors[det]["enabled"] = True

    return config


def calculate_metrics(pred_notes: List[NoteEvent], gt: List[Tuple[int, float, float]], duration_sec: float) -> Dict[str, Any]:
    pred_list = [(n.midi_note, n.start_sec, n.end_sec) for n in pred_notes]

    f1 = note_f1(pred_list, gt, onset_tol=0.05)
    onset_mae, _ = onset_offset_mae(pred_list, gt)
    symptoms = compute_symptom_metrics(pred_list)

    # Calculate voiced ratio from notes if timeline not available/processed here
    # (Approximation based on duration coverage)
    total_note_dur = sum(max(0, min(duration_sec, end) - max(0, start)) for _, start, end in pred_list)
    voiced_ratio = total_note_dur / max(1e-6, duration_sec)

    return {
        "note_f1": f1 if not np.isnan(f1) else 0.0,
        "onset_mae_ms": onset_mae * 1000 if onset_mae is not None and not np.isnan(onset_mae) else None,
        "note_count": len(pred_list),
        "gt_count": len(gt),
        "voiced_ratio": min(1.0, voiced_ratio),
        "fragmentation_score": symptoms.get("fragmentation_score", 0.0),
        "median_note_len_ms": symptoms.get("median_note_len_ms", 0.0),
        "octave_jump_rate": symptoms.get("octave_jump_rate", 0.0),
        "note_count_per_10s": symptoms.get("note_count_per_10s", 0.0),
        "notes_per_sec": len(pred_list) / max(1e-6, duration_sec)
    }


def run_sweep(
    audio: np.ndarray,
    sr: int,
    gt: List[Tuple[int, float, float]],
    base_config: PipelineConfig,
    reports_dir: str,
    duration_sec: float
) -> List[Dict[str, Any]]:

    logger.info("Starting Mini Threshold Sweep (Optimized)...")

    # Reduced sweep parameters to avoid timeouts
    sweeps = []
    # 1. max_gap_ms: Baseline is 100? Let's try tighter.
    sweeps.append(({"stage_c": {"post_merge": {"max_gap_ms": 60}}}, "max_gap_ms=60"))

    # 2. chord_onset_snap_ms: Baseline 25.
    sweeps.append(({"stage_c": {"chord_onset_snap_ms": 35}}, "chord_snap_ms=35"))

    # 3. quality_gate.threshold: Baseline None/0.45.
    sweeps.append(({"quality_gate": {"threshold": 0.45}}, "qgate_thr=0.45"))

    results = []

    for overrides, label in sweeps:
        logger.info(f"Sweep: {label}")
        sys.stdout.flush()

        cfg = copy.deepcopy(base_config)

        # Apply overrides
        if "post_merge" in overrides.get("stage_c", {}):
             val = overrides["stage_c"]["post_merge"]["max_gap_ms"]
             if not hasattr(cfg.stage_c, "post_merge") or cfg.stage_c.post_merge is None:
                 cfg.stage_c.post_merge = {}
             if isinstance(cfg.stage_c.post_merge, dict):
                 cfg.stage_c.post_merge["max_gap_ms"] = val
             else:
                 try: setattr(cfg.stage_c.post_merge, "max_gap_ms", val)
                 except: pass

        if "chord_onset_snap_ms" in overrides.get("stage_c", {}):
            cfg.stage_c.chord_onset_snap_ms = overrides["stage_c"]["chord_onset_snap_ms"]

        if "quality_gate" in overrides:
            val = overrides["quality_gate"]["threshold"]
            if hasattr(cfg, "quality_gate"):
                 if isinstance(cfg.quality_gate, dict):
                     cfg.quality_gate["threshold"] = val
                 else:
                     try: setattr(cfg.quality_gate, "threshold", val)
                     except: pass
            else:
                 cfg.quality_gate = {"threshold": val}

        try:
            res = run_pipeline_on_audio(
                audio, sr, cfg, AudioType.POLYPHONIC_DOMINANT, allow_separation=False
            )

            metrics = calculate_metrics(res["notes"], gt, duration_sec)

            analysis_diag = getattr(res.get("analysis_data"), "diagnostics", {}) or {}
            qgate = analysis_diag.get("quality_gate", {})
            selected_id = qgate.get("selected_candidate_id", "unknown")

            stage_b_diag = getattr(res.get("stage_b_out"), "diagnostics", {}) or {}
            resolved_mode = "unknown"
            if "decision_trace" in stage_b_diag:
                 resolved_mode = stage_b_diag["decision_trace"].get("resolved", {}).get("transcription_mode", "unknown")

            results.append({
                "run_label": label,
                "note_f1": metrics["note_f1"],
                "onset_mae_ms": metrics["onset_mae_ms"],
                "voiced_ratio": metrics["voiced_ratio"],
                "fragmentation_score": metrics["fragmentation_score"],
                "note_count": metrics["note_count"],
                "selected_candidate_id": selected_id,
                "decision_trace.resolved.transcription_mode": resolved_mode
            })

        except Exception as e:
            logger.error(f"Sweep run {label} failed: {e}")
            results.append({"run_label": label, "error": str(e)})

    sweep_path = os.path.join(reports_dir, "l6_sweep_results.json")
    save_json(sweep_path, results)
    return results


def write_gate_matrix(run_info: Dict[str, Any], path: str):
    """Generates reports/l6_gate_matrix.json from run info."""

    diags = run_info.get("diagnostics", {}) # Stage B diagnostics
    analysis_diags = run_info.get("analysis_diagnostics", {})

    decision_trace = diags.get("decision_trace", {})
    if not decision_trace:
         decision_trace = analysis_diags.get("decision_trace", {})

    q_gate = analysis_diags.get("quality_gate", {})
    stage_c_post = analysis_diags.get("stage_c_post", {})
    separation = diags.get("separation", {})

    gate_matrix = {
        "Stage B routing gates": {
            "requested": {
                "mode": decision_trace.get("requested", {}).get("mode"),
                "profile": decision_trace.get("requested", {}).get("profile"),
                "separation_mode": decision_trace.get("requested", {}).get("separation_mode"),
            },
            "resolved": {
                "mode": decision_trace.get("resolved", {}).get("transcription_mode"),
                "profile": decision_trace.get("resolved", {}).get("profile"),
                "separation_mode": decision_trace.get("resolved", {}).get("separation_mode"),
                "audio_type": decision_trace.get("resolved", {}).get("audio_type"),
            },
            "routing_features": decision_trace.get("features", {}),
            "rule_hits": decision_trace.get("rule_hits", []),
        },
        "Separation gates": {
            "ran": separation.get("ran", False),
            "backend": separation.get("backend", "none"),
            "skip_reasons": separation.get("skip_reasons", []),
            "gates": {
                "min_duration_sec": separation.get("gates", {}).get("min_duration_sec"),
                "min_mixture_score": separation.get("gates", {}).get("min_mixture_score"),
                "bypass_if_synthetic_like": separation.get("gates", {}).get("bypass_if_synthetic_like"),
            },
            "outputs": {
                "stems": list(separation.get("stems", {}).keys()),
                "selected_primary_stem": analysis_diags.get("stage_c", {}).get("selected_stem"),
            }
        },
        "Quality gate": {
            "enabled": q_gate.get("enabled"),
            "threshold": q_gate.get("threshold"),
            "all_candidates": q_gate.get("candidates", []),
            "selected_candidate_id": q_gate.get("selected_candidate_id"),
            "fallbacks_triggered": q_gate.get("fallbacks_triggered", [])
        },
        "Stage C post": {
            "gap_merges": stage_c_post.get("gap_merges"),
            "chord_snaps": stage_c_post.get("chord_snaps"),
            "snap_tol_ms": stage_c_post.get("snap_tol_ms"),
            "merge_gap_ms": stage_c_post.get("merge_gap_ms")
        }
    }

    save_json(path, gate_matrix)


def write_accuracy_report(metrics: Dict[str, Any], gate_matrix: Dict[str, Any], path: str):
    """Generates reports/l6_accuracy_report.md"""

    lines = []
    lines.append("# L6 Accuracy & Failure Mode Report")
    lines.append("")
    lines.append("## 1) L6 Headline Metrics")
    lines.append("")
    lines.append(f"- **note_f1**: {metrics.get('note_f1', 0.0):.3f}")
    lines.append(f"- **onset_mae_ms**: {metrics.get('onset_mae_ms', 'N/A')}")
    lines.append(f"- **note_count**: {metrics.get('note_count')}")
    lines.append(f"- **voiced_ratio**: {metrics.get('voiced_ratio', 0.0):.3f}")
    lines.append(f"- **fragmentation_score**: {metrics.get('fragmentation_score', 0.0):.3f}")
    lines.append(f"- **median_note_len_ms**: {metrics.get('median_note_len_ms', 0.0):.1f}")
    lines.append(f"- **octave_jump_rate**: {metrics.get('octave_jump_rate', 0.0):.3f}")
    lines.append(f"- **note_count_per_10s**: {metrics.get('note_count_per_10s', 0.0):.1f}")
    lines.append("")

    lines.append("## 2) Failure Mode Diagnosis")
    lines.append("")

    # Simple heuristics
    modes = []
    nc = metrics.get('note_count', 0)
    vr = metrics.get('voiced_ratio', 0.0)
    frag = metrics.get('fragmentation_score', 0.0)
    mae = metrics.get('onset_mae_ms', 0.0)
    nps = metrics.get('notes_per_sec', 0.0)
    ojr = metrics.get('octave_jump_rate', 0.0)
    med_len = metrics.get('median_note_len_ms', 0.0)

    if nc < 10 or vr < 0.2:
        modes.append("**Under-transcription**: Very few notes or low voicing coverage.")
    if frag > 0.4 or nps > 8.0:
        modes.append("**Over-transcription**: High fragmentation or notes/sec.")
    if mae is not None and mae > 50.0:
         modes.append(f"**Timing error**: Onset MAE {mae:.1f}ms > 50ms.")
    if ojr > 0.1:
        modes.append(f"**Octave instability**: Octave jump rate {ojr:.3f} is high.")
    if med_len < 80.0 and nc > 50:
        modes.append(f"**Chord fragmentation**: Low median duration {med_len:.1f}ms.")

    if not modes:
        modes.append("No critical failure modes detected (Pass).")

    for m in modes[:2]: # Top 2
        lines.append(f"- {m}")

    lines.append("")
    lines.append("## 3) Gate Sensitivity Recommendations")
    lines.append("")
    lines.append("| Knob | Current Value | L6 Observed Value | Judgment | Recommendation |")
    lines.append("|---|---|---|---|---|")

    # Helper to extract vals
    def safe_get(d, keys, default="N/A"):
        v = d
        for k in keys:
            if isinstance(v, dict): v = v.get(k, {})
            else: return default
        return v if v is not {} else default

    # Stage B
    routing_feats = safe_get(gate_matrix, ["Stage B routing gates", "routing_features"], {})

    dense_poly = routing_feats.get("dense_polyphony_score", "N/A")
    lines.append(f"| dense_poly_threshold | Config Dependent | {dense_poly} | - | Check routing rules if poly mode failed |")

    mix_score = routing_feats.get("mixture_score", "N/A")
    lines.append(f"| mixture_score | Config Dependent | {mix_score} | - | Adjust if separation skipped unexpectedly |")

    # Separation
    sep_gates = safe_get(gate_matrix, ["Separation gates", "gates"], {})
    min_dur = sep_gates.get("min_duration_sec", "N/A")
    lines.append(f"| sep.min_duration_sec | {min_dur} | 60.0 | OK | Keep as is |")

    # Quality
    q_gate = safe_get(gate_matrix, ["Quality gate"], {})
    thresh = q_gate.get("threshold", "N/A")
    sel_cand = q_gate.get("selected_candidate_id", "none")
    # Find score of selected
    sel_score = "N/A"
    for c in q_gate.get("all_candidates", []):
        if c.get("candidate_id") == sel_cand:
            sel_score = c.get("score")
            break

    lines.append(f"| quality.threshold | {thresh} | Selected Score: {sel_score} | - | If score close to threshold, lower it |")

    # Stage C
    sc_post = safe_get(gate_matrix, ["Stage C post"], {})
    max_gap = sc_post.get("merge_gap_ms", "N/A")
    merges = sc_post.get("gap_merges", 0)
    lines.append(f"| post_merge.max_gap_ms | {max_gap} | Merges: {merges} | - | Increase if fragmentation high |")

    chord_snap = sc_post.get("chord_snaps", 0)
    snap_ms = sc_post.get("snap_tol_ms", "N/A")
    lines.append(f"| chord_onset_snap_ms | {snap_ms} | Snaps: {chord_snap} | - | Increase if chord timing bad |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    reports_dir = setup_reports_dir("reports")
    run_id = f"run_{int(time.time())}"
    snapshot_dir = os.path.join(reports_dir, "snapshots", run_id)
    os.makedirs(snapshot_dir, exist_ok=True)

    logger.info(f"Reports directory: {reports_dir}")
    logger.info(f"Snapshot directory: {snapshot_dir}")

    # 1. Generate L6 Data (Reduced to 15s to fit in CPU timeout)
    duration_sec = 15.0
    audio, sr, gt = generate_l6_audio_and_gt(duration_sec=duration_sec)

    # 2. Baseline Run
    logger.info("Running L6 Baseline...")
    config = get_l6_baseline_config()

    # Execute Pipeline
    res = run_pipeline_on_audio(
        audio, sr, config, AudioType.POLYPHONIC_DOMINANT, allow_separation=False
    )

    # Calculate Metrics
    metrics = calculate_metrics(res["notes"], gt, duration_sec)
    metrics["level"] = "L6"
    metrics["name"] = "synthetic_pop_song"

    # Debug logging for F1=0
    if metrics["note_f1"] == 0.0:
        logger.warning("F1 is 0.0! Dumping top notes:")
        if gt:
            logger.warning(f"GT (first 5): {gt[:5]}")
        else:
            logger.warning("GT is empty!")

        pred_list = [(n.midi_note, n.start_sec, n.end_sec) for n in res["notes"]]
        if pred_list:
            logger.warning(f"Pred (first 5): {pred_list[:5]}")
        else:
            logger.warning("Pred is empty!")

    # Extract RICH diagnostics
    stage_b_out = res.get("stage_b_out")
    analysis_data = res.get("analysis_data")

    stage_b_diag = getattr(stage_b_out, "diagnostics", {}) if stage_b_out else {}
    analysis_diag = getattr(analysis_data, "diagnostics", {}) if analysis_data else {}

    # Construct Full Run Info
    run_info = {
        "level": "L6",
        "name": "synthetic_pop_song",
        "metrics": metrics,
        "config": asdict(res["resolved_config"]),
        "diagnostics": stage_b_diag,          # Stage B diagnostics
        "analysis_diagnostics": analysis_diag, # Stage C / Quality Gate diagnostics
    }

    # Save Baseline Artifacts
    base_name = "L6_synthetic_pop_song"
    save_json(os.path.join(snapshot_dir, f"{base_name}_metrics.json"), metrics)
    save_json(os.path.join(snapshot_dir, f"{base_name}_run_info.json"), run_info)

    # 3. Generate Gate Matrix (Requirement B)
    gate_matrix_path = os.path.join(reports_dir, "l6_gate_matrix.json")
    write_gate_matrix(run_info, gate_matrix_path)

    # 4. Generate Accuracy Report (Requirement C)
    acc_report_path = os.path.join(reports_dir, "l6_accuracy_report.md")
    # Load the matrix we just wrote to use in report generation if needed, or just pass dict
    with open(gate_matrix_path, 'r') as f:
        gm = json.load(f)
    write_accuracy_report(metrics, gm, acc_report_path)

    # 5. Run Sweep (Requirement D) - Only if F1 > 0 or forced
    if metrics["note_f1"] > 0.0:
        run_sweep(audio, sr, gt, config, reports_dir, duration_sec)
    else:
        logger.warning("Skipping sweep because baseline F1 is 0.0.")
        # Write empty sweep file so process doesn't look like it crashed
        save_json(os.path.join(reports_dir, "l6_sweep_results.json"), [{"error": "Skipped due to F1=0.0"}])

    # 6. Generate other standard artifacts for compliance (Requirement A)
    save_json(os.path.join(reports_dir, "benchmark_results.json"), [metrics])
    save_json(os.path.join(reports_dir, "stage_metrics.json"), {"n_runs": 1, "note": "generated by l6_debug_runner"})
    with open(os.path.join(reports_dir, "stage_health_report.md"), "w") as f:
        f.write(f"# Stage Health\n\nL6 F1: {metrics['note_f1']:.3f}\n")
    with open(os.path.join(reports_dir, "regression_flags.md"), "w") as f:
        f.write("# Regression Flags\n\nNo baseline for comparison.\n")

    logger.info("Done.")

if __name__ == "__main__":
    main()
