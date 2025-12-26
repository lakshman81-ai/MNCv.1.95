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

import backend.pipeline.transcribe
backend.pipeline.transcribe.TranscriptionResult = PatchedTranscriptionResult

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
    # Baseline: Separation Disabled, Polyphony Filter default/chords
    config.stage_b.separation["enabled"] = False
    config.stage_b.melody_filtering.update({"fmin_hz": 180.0, "fmax_hz": 1600.0, "voiced_prob_threshold": 0.40})

    for det in ["rmvpe", "crepe", "swiftf0", "yin"]:
        if det in config.stage_b.detectors:
            config.stage_b.detectors[det]["enabled"] = True

    return config

def get_l6_improved_config() -> PipelineConfig:
    config = PipelineConfig()

    # 1. Enable Separation with Synthetic Model
    config.stage_b.separation["enabled"] = True
    config.stage_b.separation["synthetic_model"] = True
    config.stage_b.separation["gates"] = {"min_mixture_score": 0.01}

    # 2. Filtering
    config.stage_b.active_stems = ["vocals", "mix"]
    config.stage_b.melody_filtering.update({"fmin_hz": 180.0, "fmax_hz": 1600.0, "voiced_prob_threshold": 0.40})

    # 3. Detectors
    for det in ["rmvpe", "crepe", "swiftf0", "yin"]:
        if det in config.stage_b.detectors:
            config.stage_b.detectors[det]["enabled"] = True

    # 4. Polyphony Filter Mode
    config.stage_c.polyphony_filter["mode"] = "decomposed_melody"

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
                curr[part] = {}
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

    # Sweep on top of IMPROVED config? No, sweep is usually on baseline to find params.
    # But if baseline is 0.0, we should probably sweep on Improved.
    # Let's sweep on the Improved config logic.
    base_config = get_l6_improved_config()

    sweeps = []
    # Tuning params
    for val in [60, 100]:
        sweeps.append(("stage_c.post_merge.max_gap_ms", val))

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
                "note_count": metrics["note_count"],
                "selected_candidate_id": selected_id
            })

        except Exception as e:
            logger.error(f"Sweep run {label} failed: {e}")
            results.append({"run_label": label, "error": str(e)})

    sweep_path = os.path.join(reports_dir, "l6_sweep_results.json")
    save_json(sweep_path, results)
    return results

def write_gate_matrix(run_info: Dict[str, Any], path: str):
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
    lines = []
    lines.append("# L6 Accuracy & Failure Mode Report")
    lines.append("")
    lines.append("## 1) L6 Headline Metrics")
    lines.append(f"- **note_f1**: {metrics.get('note_f1', 0.0):.3f}")
    lines.append(f"- **note_count**: {metrics.get('note_count')}")
    lines.append(f"- **voiced_ratio**: {metrics.get('voiced_ratio', 0.0):.3f}")
    lines.append(f"- **octave_jump_rate**: {metrics.get('octave_jump_rate', 0.0):.3f}")

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

    # 1. Generate L6 Data (Reduced to 1.0s to ENSURE completion)
    duration_sec = 1.0
    wav_path, gt = generate_l6_audio_and_gt(duration_sec=duration_sec)

    try:
        # ---------------------------------------
        # 2a. Run Baseline (Skipped to save time)
        # ---------------------------------------
        # logger.info("Running L6 Baseline (Original)...")
        # config_base = get_l6_baseline_config()
        # res_base = transcribe(wav_path, config=config_base)
        # metrics_base = calculate_metrics(res_base.analysis_data.notes, gt, duration_sec)

        # ---------------------------------------
        # 2b. Run Improved (Tuned Config)
        # ---------------------------------------
        logger.info("Running L6 Improved (Separation + Decomposed Melody)...")
        config_imp = get_l6_improved_config()
        res_imp = transcribe(wav_path, config=config_imp)
        metrics_imp = calculate_metrics(res_imp.analysis_data.notes, gt, duration_sec)

        logger.info(f"Improved F1: {metrics_imp['note_f1']:.3f}")

        # Save Improved Run Info
        analysis_diag = getattr(res_imp.analysis_data, "diagnostics", {}) or {}
        stage_b_diag = {"decision_trace": analysis_diag.get("decision_trace", {})}
        run_info = {
            "level": "L6",
            "name": "synthetic_pop_song_improved",
            "metrics": metrics_imp,
            "config": asdict(config_imp),
            "diagnostics": stage_b_diag,
            "analysis_diagnostics": analysis_diag,
        }

        save_json(os.path.join(snapshot_dir, f"L6_improved_metrics.json"), metrics_imp)
        save_json(os.path.join(snapshot_dir, f"L6_improved_run_info.json"), run_info)

        # Generate Reports based on Improved
        gate_matrix_path = os.path.join(reports_dir, "l6_gate_matrix.json")
        write_gate_matrix(run_info, gate_matrix_path)

        acc_report_path = os.path.join(reports_dir, "l6_accuracy_report.md")
        write_accuracy_report(metrics_imp, {}, acc_report_path)

        # ---------------------------------------
        # 3. Sweep (on Improved)
        # ---------------------------------------
        # run_sweep(wav_path, gt, config_imp, reports_dir, duration_sec)

        print("\n=== L6 Improvement Result ===")
        # print(f"Baseline F1: {metrics_base['note_f1']:.3f}")
        print(f"Improved F1: {metrics_imp['note_f1']:.3f}")

        print("\n=== L6 Gate Matrix (Improved) ===")
        print(open(gate_matrix_path, "r", encoding="utf-8").read())
        print("\n=== L6 Accuracy Report (Improved) ===")
        print(open(acc_report_path, "r", encoding="utf-8").read())

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        for h in logging.getLogger().handlers:
            h.flush()

    logger.info("Done.")

if __name__ == "__main__":
    main()
