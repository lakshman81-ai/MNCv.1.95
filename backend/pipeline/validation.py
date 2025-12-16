"""Pipeline invariant checks for stage outputs."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Optional
import json
import math
import os
import time
import logging

import numpy as np

from .models import AnalysisData, FramePitch, NoteEvent, StageAOutput, StageBOutput, TranscriptionResult

logger = logging.getLogger(__name__)


_DEF_TOL = 1e-3


def _hz_to_midi(pitch_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(pitch_hz / 440.0)


def _validate_timebase_from_frames(frames: Iterable[FramePitch], hop_seconds: float, duration: float) -> None:
    times = [fp.time for fp in frames]
    if len(times) < 2:
        return
    diffs = np.diff(times)
    median_dt = float(np.median(diffs))
    if not math.isclose(median_dt, hop_seconds, rel_tol=1e-2, abs_tol=1e-4):
        raise AssertionError(
            f"Frame spacing {median_dt:.6f}s deviates from hop_seconds {hop_seconds:.6f}s"
        )
    if max(times) - duration > max(_DEF_TOL, hop_seconds * 1.5):
        raise AssertionError("Frame times exceed clip duration")


def validate_invariants(stage_output: Any, config: Any, analysis_data: Optional[AnalysisData] = None) -> None:
    """Validate invariants per stage.

    Raises AssertionError on invariant violations.
    """
    # Stage A
    if isinstance(stage_output, StageAOutput):
        mix = stage_output.stems.get("mix")
        if mix is not None:
            expected_duration = len(mix.audio) / float(mix.sr)
            if not math.isclose(stage_output.meta.duration_sec, expected_duration, rel_tol=1e-2, abs_tol=1e-3):
                raise AssertionError(
                    f"Meta duration {stage_output.meta.duration_sec:.4f}s does not match audio length {expected_duration:.4f}s"
                )
        if stage_output.meta.hop_length <= 0 or stage_output.meta.window_size <= 0:
            raise AssertionError("Stage A hop/window must be positive")
        return

    # Stage B
    if isinstance(stage_output, StageBOutput):
        meta = stage_output.meta
        hop_length = getattr(meta, "hop_length", None)
        sr = getattr(meta, "sample_rate", None) or getattr(meta, "target_sr", None)
        if hop_length and sr and len(stage_output.time_grid) >= 2:
            hop_seconds = float(hop_length) / float(sr)
            grid_diffs = np.diff(stage_output.time_grid)
            median_dt = float(np.median(grid_diffs))
            if not math.isclose(median_dt, hop_seconds, rel_tol=1e-2, abs_tol=1e-4):
                raise AssertionError("Stage B time grid hop does not match resolved hop length")
        if stage_output.f0_main.size != stage_output.time_grid.size:
            raise AssertionError("Stage B f0_main length must align with time_grid")
        if stage_output.timeline:
            _validate_timebase_from_frames(stage_output.timeline, float(meta.hop_length) / float(meta.sample_rate), meta.duration_sec)
        return

    # Stage C outputs (list of NoteEvent)
    if isinstance(stage_output, list) and stage_output and isinstance(stage_output[0], NoteEvent):
        meta = analysis_data.meta if analysis_data else None
        duration = getattr(meta, "duration_sec", None)
        for note in stage_output:
            if note.end_sec <= note.start_sec:
                raise AssertionError("Note end must be after start")
            if duration is not None and (note.start_sec < -_DEF_TOL or note.end_sec - duration > max(_DEF_TOL, 0.01)):
                raise AssertionError("Note timing falls outside audio duration")
            if note.pitch_hz > 0:
                midi_est = _hz_to_midi(note.pitch_hz)
                if abs(midi_est - note.midi_note) > 0.75:
                    raise AssertionError("Note MIDI/pitch_hz mismatch exceeds tolerance")
        return

    # Stage D final result
    if isinstance(stage_output, TranscriptionResult):
        analysis = stage_output.analysis_data
        if analysis and analysis.meta:
            duration = analysis.meta.duration_sec
            hop_seconds = analysis.frame_hop_seconds or (float(analysis.meta.hop_length) / float(analysis.meta.sample_rate))
            if analysis.timeline:
                _validate_timebase_from_frames(analysis.timeline, hop_seconds, duration)
        return


def dump_resolved_config(config: Any, meta: Any, stage_b_out: Optional[StageBOutput] = None, run_dir: str = "results") -> str:
    os.makedirs(run_dir, exist_ok=True)
    run_path = os.path.join(run_dir, f"run_{int(time.time() * 1000)}")
    os.makedirs(run_path, exist_ok=True)

    detectors_enabled = [name for name, det in getattr(getattr(config, "stage_b", {}), "detectors", {}).items() if det.get("enabled", False)]
    detectors_ran = []
    diagnostics: dict = {}
    if stage_b_out is not None:
        detectors_ran = list(stage_b_out.per_detector.get("mix", {}).keys())
        diagnostics = getattr(stage_b_out, "diagnostics", {}) or {}

    payload = {
        "meta": asdict(meta) if meta is not None else {},
        "detectors_enabled": detectors_enabled,
        "detectors_ran": detectors_ran,
        "diagnostics": diagnostics,
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else str(config),
    }

    path = os.path.join(run_path, "resolved_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Resolved config saved", extra={"resolved_config_path": path, "detectors_ran": detectors_ran})
    return path
