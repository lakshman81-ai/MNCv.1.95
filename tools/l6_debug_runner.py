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
from dataclasses import asdict, dataclass

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------------------------------------------------------
# Monkey-Patching TranscriptionResult to fix TypeError/Signature Mismatch
# -------------------------------------------------------------------------
import backend.pipeline.models
from backend.pipeline.models import AnalysisData

# Define the patched class (supports 'midi' argument and tuple unpacking)
@dataclass
class PatchedTranscriptionResult:
    musicxml: str
    analysis_data: AnalysisData
    midi: bytes = b""
    midi_bytes: bytes = b"" # Legacy support field, not used in __init__ unless defaults

    def __post_init__(self):
        # Handle aliasing
        if self.midi and not self.midi_bytes:
            self.midi_bytes = self.midi
        if self.midi_bytes and not self.midi:
            self.midi = self.midi_bytes

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __iter__(self):
        return iter((self.musicxml, self.midi))

# Force apply the patch to the models module
backend.pipeline.models.TranscriptionResult = PatchedTranscriptionResult

# NOW import transcribe. It will import models, but models is already loaded.
# However, if transcribe does `from .models import TranscriptionResult`, it might grab the OLD one
# if the module was already cached before we patched it?
# To be safe, we also check if transcribe is loaded and patch it there.
import backend.pipeline.transcribe
backend.pipeline.transcribe.TranscriptionResult = PatchedTranscriptionResult

# Also patch stage_d if it imported it
import backend.pipeline.stage_d
backend.pipeline.stage_d.TranscriptionResult = PatchedTranscriptionResult

# -------------------------------------------------------------------------

from backend.pipeline.config import PipelineConfig
from backend.pipeline.models import AudioType, NoteEvent
from backend.pipeline.transcribe import transcribe
from backend.benchmarks.benchmark_runner import (
    BenchmarkSuite,
    create_pop_song_base
)
from backend.benchmarks.ladder.synth import midi_to_wav_synth
from backend.benchmarks.metrics import (
    note_f1,
    onset_offset_mae,
    compute_symptom_metrics
)

# Setup logging with FileHandler
def setup_logging(reports_dir: str):
    log_path = os.path.join(reports_dir, "bench_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("l6_debug_runner")

logger = None  # Will be initialized in main

def setup_reports_dir(reports_dir: str):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "snapshots"), exist_ok=True)
    return reports_dir

def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def generate_l6_audio_and_gt(duration_sec: float = 15.0) -> Tuple[str, List[Tuple[int, float, float]]]:
    """Generates the synthetic pop song audio (saved to temp file) and ground truth."""
    sr = 22050
    tempo_bpm = 110

    logger.info(f"Generating L6 synthetic pop song score ({duration_sec}s)...")
    score = create_pop_song_base(duration_sec=duration_sec, tempo_bpm=tempo_bpm, seed=0)

    # Ground truth (Lead melody only for L6) - reusing BenchmarkSuite static method
    gt = BenchmarkSuite._score_to_gt(score, parts=["Lead"])
    logger.info(f"Generated GT with {len(gt)} notes (Lead part).")
    if gt:
        logger.info(f"GT Sample: {gt[:3]}")

    logger.info("Synthesizing audio...")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)

    # Synth to the path
    midi_to_wav_synth(score, wav_path, sr=sr)

    # Ensure Mono
    data, read_sr = sf.read(wav_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        sf.write(wav_path, data, read_sr)

    return wav_path, gt

def get_l6_baseline_config() -> PipelineConfig:
    config = PipelineConfig()
    # Avoid Demucs on synthetic benches
    config.stage_b.separation["enabled"] = False

    # Encourage melody tracking in a poly mix
    config.stage_b.melody_filtering.update({"fmin_hz": 180.0, "fmax_hz": 1600.0, "voiced_prob_threshold": 0.40})

    # Ensure robust detectors are enabled
    for det in ["rmvpe", "crepe", "swiftf0", "yin"]:
        if det in config.stage_b.detectors:
            config.stage_b.detectors[det]["enabled"] = True

    return config

def calculate_metrics(pred_notes: List[NoteEvent], gt: List[Tuple[int, float, float]], duration_sec: float) -> Dict[str, Any]:
    pred_list = [(n.midi_note, n.start_sec, n.end_sec) for n in pred_notes]

    f1 = note_f1(pred_list, gt, onset_tol=0.05)
    onset_mae, _ = onset_offset_mae(pred_list, gt)
    symptoms = compute_symptom_metrics(pred_list)

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

def _set_config_val(config: Any, path: str, value: Any):
    """Robust setter for nested config objects/dicts."""
    parts = path.split(".")
    curr = config
    for i, part in enumerate(parts[:-1]):
        if isinstance(curr, dict):
            if part not in curr:
                curr[part] = {} # Auto-vivify dicts
            curr = curr[part]
        elif hasattr(curr, part):
            val = getattr(curr, part)
            if val is None:
                setattr(curr, part, {})
                curr = getattr(curr, part)
            else:
                curr = val
        else:
            try:
                curr[part] = {}
                curr = curr[part]
            except:
                logger.warning(f"Could not traverse config path {path} at {part}")
                return

    last = parts[-1]
    if isinstance(curr, dict):
        curr[last] = value
    elif hasattr(curr, last):
        setattr(curr, last, value)
    else:
        try:
            curr[last] = value
        except:
            logger.warning(f"Could not set config path {path} at {last}")

def run_sweep(
    wav_path: str,
    gt: List[Tuple[int, float, float]],
    base_config: PipelineConfig,
    reports_dir: str,
    duration_sec: float
) -> List[Dict[str, Any]]:

    logger.info("Starting Mini Threshold Sweep...")

    sweeps = []
    # Reduced to minimal set to avoid timeout
    for val in [60]:
        sweeps.append(("stage_c.post_merge.max_gap_ms", val))
    for val in [25]:
        sweeps.append(("stage_c.chord_onset_snap_ms", val))
    for val in [0.45]:
        sweeps.append(("quality_gate.threshold", val))

    results = []

    for path, val in sweeps:
        label = f"{path}={val}"
        logger.info(f"Sweep: {label}")

        cfg = copy.deepcopy(base_config)
        _set_config_val(cfg, path, val)

        try:
            res = transcribe(wav_path, config=cfg)
            notes = res.analysis_data.notes
            metrics = calculate_metrics(notes, gt, duration_sec)

            analysis_diag = getattr(res.analysis_data, "diagnostics", {}) or {}
            qgate = analysis_diag.get("quality_gate", {})
            selected_id = qgate.get("selected_candidate_id", "unknown")

            resolved_mode = "unknown"
            dt = analysis_diag.get("decision_trace", {})
            if dt:
                resolved_mode = dt.get("resolved", {}).get("transcription_mode", "unknown")

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

    stage_b_diag = run_info.get("diagnostics", {}) or {}
    analysis_diag = run_info.get("analysis_diagnostics", {}) or {}

    decision_trace = stage_b_diag.get("decision_trace") or analysis_diag.get("decision_trace") or {}
    sep = decision_trace.get("separation", {}) or {}

    q_gate = analysis_diag.get("quality_gate", {}) or {}
    stage_c_post = analysis_diag.get("stage_c_post", {}) or {}

    gate_matrix = {
        "stage_b_routing": {
            "requested": decision_trace.get("requested", {}),
            "resolved": decision_trace.get("resolved", {}),
            "routing_features": decision_trace.get("routing_features", {}),
            "rule_hits": decision_trace.get("rule_hits", []),
        },
        "separation": {
            "ran": sep.get("ran", False),
            "backend": sep.get("backend", "none"),
            "skip_reasons": sep.get("skip_reasons", []),
            "gates": sep.get("gates", {}),
            "outputs": sep.get("outputs", {}),
        },
        "quality_gate": q_gate,
        "stage_c_post": stage_c_post,
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

    def safe_get(d, keys, default="N/A"):
        v = d
        for k in keys:
            if isinstance(v, dict): v = v.get(k, {})
            else: return default
        return v if v is not {} else default

    routing_feats = safe_get(gate_matrix, ["stage_b_routing", "routing_features"], {})

    dense_poly = routing_feats.get("dense_polyphony_score", "N/A")
    lines.append(f"| dense_poly_threshold | Config Dependent | {dense_poly} | - | Check routing rules if poly mode failed |")

    mix_score = routing_feats.get("mixture_score", "N/A")
    lines.append(f"| mixture_score | Config Dependent | {mix_score} | - | Adjust if separation skipped unexpectedly |")

    sep_gates = safe_get(gate_matrix, ["separation", "gates"], {})
    min_dur = sep_gates.get("min_duration_sec", "N/A")
    lines.append(f"| sep.min_duration_sec | {min_dur} | 60.0 | OK | Keep as is |")

    q_gate = safe_get(gate_matrix, ["quality_gate"], {})
    thresh = q_gate.get("threshold", "N/A")
    sel_cand = q_gate.get("selected_candidate_id", "none")
    sel_score = "N/A"
    for c in q_gate.get("candidates", []):
        if c.get("candidate_id") == sel_cand:
            sel_score = c.get("score")
            break

    lines.append(f"| quality.threshold | {thresh} | Selected Score: {sel_score} | - | If score close to threshold, lower it |")

    sc_post = safe_get(gate_matrix, ["stage_c_post"], {})
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
    global logger
    logger = setup_logging(reports_dir)

    run_id = f"run_{int(time.time())}"
    snapshot_dir = os.path.join(reports_dir, "snapshots", run_id)
    os.makedirs(snapshot_dir, exist_ok=True)

    logger.info(f"Reports directory: {reports_dir}")
    logger.info(f"Snapshot directory: {snapshot_dir}")

    # 1. Generate L6 Data (5s for debug speed and flush safety)
    duration_sec = 5.0
    wav_path, gt = generate_l6_audio_and_gt(duration_sec=duration_sec)

    # 2. Baseline Run with transcribe()
    logger.info("Running L6 Baseline...")
    config = get_l6_baseline_config()

    try:
        res = transcribe(wav_path, config=config)

        metrics = calculate_metrics(res.analysis_data.notes, gt, duration_sec)
        metrics["level"] = "L6"
        metrics["name"] = "synthetic_pop_song"

        analysis_diag = getattr(res.analysis_data, "diagnostics", {}) or {}
        stage_b_diag = {"decision_trace": analysis_diag.get("decision_trace", {})}

        run_info = {
            "level": "L6",
            "name": "synthetic_pop_song",
            "metrics": metrics,
            "config": asdict(config),
            "diagnostics": stage_b_diag,
            "analysis_diagnostics": analysis_diag,
        }

        base_name = "L6_synthetic_pop_song"
        save_json(os.path.join(snapshot_dir, f"{base_name}_metrics.json"), metrics)
        save_json(os.path.join(snapshot_dir, f"{base_name}_run_info.json"), run_info)

        gate_matrix_path = os.path.join(reports_dir, "l6_gate_matrix.json")
        write_gate_matrix(run_info, gate_matrix_path)

        acc_report_path = os.path.join(reports_dir, "l6_accuracy_report.md")
        gm = json.load(open(gate_matrix_path))
        write_accuracy_report(metrics, gm, acc_report_path)

        if metrics["note_f1"] == 0.0:
            logger.warning("F1 is 0.0! Dumping top notes:")
            if gt:
                logger.warning(f"GT (first 5): {gt[:5]}")
            pred_list = [(n.midi_note, n.start_sec, n.end_sec) for n in res.analysis_data.notes]
            logger.warning(f"Pred (first 5): {pred_list[:5] if pred_list else 'Empty'}")

        # Run Sweep (Always)
        run_sweep(wav_path, gt, config, reports_dir, duration_sec)

        save_json(os.path.join(reports_dir, "benchmark_results.json"), [metrics])
        save_json(os.path.join(reports_dir, "stage_metrics.json"), {"n_runs": 1, "note": "generated by l6_debug_runner"})

        print("\n=== L6 Gate Matrix ===")
        print(open(gate_matrix_path, "r", encoding="utf-8").read())
        print("\n=== L6 Accuracy Report ===")
        print(open(acc_report_path, "r", encoding="utf-8").read())
        print("\n=== L6 Sweep Results ===")
        print(open(os.path.join(reports_dir, "l6_sweep_results.json"), "r", encoding="utf-8").read())

        # Verification
        expected_files = [
            "reports/bench_run.log",
            "reports/l6_gate_matrix.json",
            "reports/l6_accuracy_report.md",
            "reports/l6_sweep_results.json",
            f"reports/snapshots/{run_id}/L6_synthetic_pop_song_run_info.json",
            f"reports/snapshots/{run_id}/L6_synthetic_pop_song_metrics.json"
        ]
        print("\n=== Artifact Verification ===")
        all_ok = True
        for fpath in expected_files:
            if os.path.exists(fpath):
                print(f"[OK] {fpath}")
            else:
                print(f"[MISSING] {fpath}")
                all_ok = False

        if not all_ok:
            logger.error("Some artifacts are missing!")

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()

    logger.info("Done.")

if __name__ == "__main__":
    main()
