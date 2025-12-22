"""
High-level transcription orchestrator.

Pipeline:
    Stage A: load_and_preprocess
    Stage B: extract_features (OR neural_transcription if enabled)
    Stage C: apply_theory (skipped if neural_transcription used)
    Stage D: quantize_and_render

Contract (as used in tests/test_pipeline_flow.py):

    from backend.pipeline import transcribe

    result = transcribe(audio_path, config=PIANO_61KEY_CONFIG)

    assert isinstance(result.musicxml, str)
    analysis = result.analysis_data

    assert analysis.meta.sample_rate == sr_target
    assert analysis.meta.target_sr == sr_target
    assert len(analysis.notes) > 0
"""

import time
import csv
import math
from typing import Optional

from .config import PIANO_61KEY_CONFIG, PipelineConfig
from .stage_a import load_and_preprocess, detect_tempo_and_beats
from .stage_b import extract_features
from .stage_c import apply_theory
from .stage_d import quantize_and_render
from .neural_transcription import transcribe_onsets_frames
from .models import AnalysisData, StageAOutput, TranscriptionResult, StageBOutput, NoteEvent, FramePitch, Stem, AudioType
# Optional dependency: validation.py may not exist in minimal 10-file set.
_VALIDATION_AVAILABLE = True
try:
    from .validation import validate_invariants, dump_resolved_config  # type: ignore
except Exception:
    _VALIDATION_AVAILABLE = False

    def validate_invariants(*_a, **_k):  # best-effort no-op
        return None

    def dump_resolved_config(*_a, **_k):  # best-effort empty
        return {}

from .instrumentation import PipelineLogger
import numpy as np
import os
import json
import copy
from dataclasses import asdict, replace

def _slice_stage_a_output(
    stage_a_out: StageAOutput,
    start_sec: float,
    end_sec: float,
    pad_sec: float = 0.5
) -> StageAOutput:
    """
    Slice StageAOutput into a time window.

    Copies metadata but slices audio arrays in `stems`.
    Does NOT slice 'beats' or other time-global metadata yet,
    as Stage B/C mostly care about audio content.
    """
    sr = stage_a_out.meta.sample_rate
    total_samples = 0

    # Determine safe indices
    # We must know the length of at least one stem to clamp
    mix = stage_a_out.stems.get("mix")
    if mix:
        total_samples = len(mix.audio)
    else:
        # Fallback if no mix stem (unlikely)
        any_stem = next(iter(stage_a_out.stems.values()), None)
        if any_stem:
            total_samples = len(any_stem.audio)

    start_idx = int(max(0, start_sec * sr))
    end_idx = int(min(total_samples, end_sec * sr))

    # Create new stems
    new_stems = {}
    for name, stem in stage_a_out.stems.items():
        # Slice audio
        audio_slice = stem.audio[start_idx:end_idx]
        new_stems[name] = Stem(
            audio=audio_slice,
            sr=stem.sr,
            type=stem.type
        )

    # Create new MetaData (shallow copy + update duration)
    new_meta = replace(stage_a_out.meta)
    new_meta.duration_sec = (end_idx - start_idx) / sr

    # Adjust beats (shift time to slice-relative)
    # Priority: stage_a_out.beats > stage_a_out.meta.beats > []
    src_beats = stage_a_out.beats if stage_a_out.beats else getattr(stage_a_out.meta, "beats", [])
    sliced_beats = []
    if src_beats:
        # Filter beats within [start_sec, end_sec]
        # and shift by -start_sec
        sliced_beats = [
            b - start_sec
            for b in src_beats
            if start_sec <= b <= end_sec
        ]

    # Sync meta beats as well
    new_meta.beats = sliced_beats

    return StageAOutput(
        stems=new_stems,
        meta=new_meta,
        audio_type=stage_a_out.audio_type,
        noise_floor_rms=stage_a_out.noise_floor_rms,
        noise_floor_db=stage_a_out.noise_floor_db,
        beats=sliced_beats
    )

def _score_segment(
    notes: list[NoteEvent],
    duration_sec: float,
    audio_type: AudioType,
    config: PipelineConfig
) -> float:
    """
    Compute quality score [0,1] for a transcribed segment.

    Heuristics:
      1. Inverse Note Density: Penalize if too many notes per second.
      2. Onset Plausibility: Penalize very short notes.
    """
    if duration_sec <= 0:
        return 0.0

    note_count = len(notes)
    if note_count == 0:
        # Silence is valid if the input was silent, but here we assume
        # we want to detect something if audio is present.
        # However, if true silence, density is 0 which is 'stable'.
        return 1.0 # Tentative: low density = high score?
                   # But empty transcription on busy audio is bad.
                   # Since we don't have ground truth, we assume 'stable' = good.

    # 1. Density Score
    # "Too many notes" => chaos/noise.
    density = note_count / duration_sec
    target = config.segmented_transcription.density_target_notes_per_sec
    span = config.segmented_transcription.density_penalty_span

    # Soft clamp: if density <= target, score 1.0.
    # If density >= target + span, score -> 0.0.
    excess = max(0.0, density - target)
    density_score = 1.0 - np.clip(excess / span, 0.0, 1.0)

    # 2. Onset Plausibility
    # Penalize short notes
    min_dur = config.stage_c.min_note_duration_ms / 1000.0
    if audio_type in (AudioType.POLYPHONIC, AudioType.POLYPHONIC_DOMINANT):
         min_dur = config.stage_c.min_note_duration_ms_poly / 1000.0

    short_notes = sum(1 for n in notes if (n.end_sec - n.start_sec) < min_dur)
    plausibility = 1.0 - (short_notes / note_count)

    # Weighted combination
    # Density is a strong indicator of "noise explosion".
    # Plausibility checks for "fragmentation".
    durations = []
    for n in notes or []:
        s = getattr(n, "start_sec", 0.0) or 0.0
        e = getattr(n, "end_sec", 0.0) or 0.0
        if e > s:
            durations.append(e - s)

    median_dur = float(np.median(durations)) if durations else 0.0
    notes_per_sec = (len(notes) / max(1e-6, duration_sec)) if duration_sec else 0.0  # use your existing segment duration var

    penalty = 0.0
    density_target_notes_per_sec = config.segmented_transcription.density_target_notes_per_sec

    if median_dur and median_dur < 0.08:
        penalty += (0.08 - median_dur) * 5.0
    if notes_per_sec > (density_target_notes_per_sec * 2.0):
        penalty += (notes_per_sec - density_target_notes_per_sec * 2.0) * 0.5

    final_score = 0.6 * density_score + 0.4 * plausibility
    score = final_score - penalty

    return float(score)

def _build_candidate_configs(
    base_config: PipelineConfig,
    segment_audio_type: AudioType
) -> list[PipelineConfig]:
    """
    Generate candidate configurations for retry logic.

    C1: Original
    C2: Alternate algorithm priorities
    C3: Relaxed thresholds
    """
    configs = []

    # C1: Primary
    configs.append(base_config)

    # C2: Alternate Algorithms
    # If mono/poly-dominant: enable CREPE + Multi-res YIN
    # If poly: enable ISS adaptive
    c2 = copy.deepcopy(base_config)
    if segment_audio_type == AudioType.POLYPHONIC:
        # Enable adaptive ISS
        c2.stage_b.polyphonic_peeling["iss_adaptive"] = True
    else:
        # Boost CREPE/YIN for melody
        if "crepe" in c2.stage_b.detectors:
            c2.stage_b.detectors["crepe"]["enabled"] = True
        if "yin" in c2.stage_b.detectors:
            c2.stage_b.detectors["yin"]["enable_multires_f0"] = True

    configs.append(c2)

    # C3: Relaxed Thresholds
    # Lower confidence thresholds, increase gap tolerance
    c3 = copy.deepcopy(base_config)
    c3.stage_c.confidence_threshold *= 0.8
    if hasattr(c3.stage_c, "gap_tolerance_s"):
        c3.stage_c.gap_tolerance_s *= 2.0

    # Increase smoothing
    c3.stage_b.voice_tracking["smoothing"] = 0.7

    configs.append(c3)

    return configs

def _stitch_events(
    accumulated_notes: list[NoteEvent],
    new_segment_notes: list[NoteEvent],
    overlap_start: float,
    overlap_end: float,
    pipeline_logger: Optional[PipelineLogger] = None
) -> list[NoteEvent]:
    """
    Merge notes from new segment into accumulated notes, handling overlap.

    Strategy:
    - Notes ending before overlap_start are kept as-is.
    - Notes in new segment starting after overlap_end are kept as-is.
    - In overlap region [overlap_start, overlap_end]:
        - Match notes by pitch.
        - Merge if they overlap in time (within small tolerance).
        - Resolve conflicts (duplicates).
    """
    # 1. Split accumulated notes
    # Keep notes that end cleanly before the merge region or just inside it
    # We define "safe" zone as anything ending before overlap_start
    safe_history = []
    candidates_history = []

    for n in accumulated_notes:
        if n.end_sec < overlap_start:
            safe_history.append(n)
        else:
            candidates_history.append(n)

    # 2. Process new notes
    # Shift time of new notes is NOT needed because Stage B/C inputs
    # were sliced but returned times relative to the slice start?
    # WAIT. Stage B output timestamps are relative to the *slice*.
    # We must assume the caller has already shifted the timestamps
    # of new_segment_notes to global time!
    # Let's verify this assumption in the calling loop.
    # Yes, we will shift them before calling stitch.

    # 3. Merging logic
    # Simple approach:
    # - Start with safe_history.
    # - Try to merge candidates_history with new_segment_notes.

    merged = list(safe_history)

    # Sort by start time
    all_overlap = sorted(candidates_history + new_segment_notes, key=lambda x: x.start_sec)

    # We will use a greedy merge on sorted events
    # If two events have same pitch and overlap in time, merge them.

    if not all_overlap:
        return merged

    current = all_overlap[0]

    for next_note in all_overlap[1:]:
        # Check overlap/adjacency
        # Tolerance for "touching" notes
        TOLERANCE = 0.05

        is_same_pitch = (current.midi_note == next_note.midi_note)
        # Check time overlap: (StartA <= EndB) and (EndA >= StartB)
        # With tolerance for gaps
        is_overlapping = (current.start_sec - TOLERANCE <= next_note.end_sec) and \
                         (current.end_sec + TOLERANCE >= next_note.start_sec)

        if is_same_pitch and is_overlapping:
            # Merge
            # Extend duration
            new_start = min(current.start_sec, next_note.start_sec)
            new_end = max(current.end_sec, next_note.end_sec)

            # Max confidence
            take_next = next_note.confidence > current.confidence
            new_conf = max(current.confidence, next_note.confidence)
            # Max velocity
            new_vel = max(current.velocity, next_note.velocity)

            # Weighted average pitch Hz (if valid)
            w1 = current.confidence
            w2 = next_note.confidence
            if w1 + w2 > 0:
                new_hz = (current.pitch_hz * w1 + next_note.pitch_hz * w2) / (w1 + w2)
            else:
                new_hz = current.pitch_hz

            # Update current
            current.start_sec = new_start
            current.end_sec = new_end
            current.confidence = new_conf
            current.velocity = new_vel
            current.pitch_hz = new_hz
            # Keep voice/staff from winner
            if take_next:
                current.voice = next_note.voice
                current.staff = next_note.staff
        else:
            # No merge, push current and move on
            merged.append(current)
            current = next_note

    merged.append(current)

    stitched = sorted(merged, key=lambda x: (x.start_sec, x.end_sec))
    clean = []
    last_end = -1e9
    for ev in stitched:
        if ev.end_sec is None or ev.start_sec is None or ev.end_sec <= ev.start_sec:
            if pipeline_logger:
                pipeline_logger.log_event("segmented", "stitch_drop_invalid_event",
                                          {"start": getattr(ev, "start_sec", None), "end": getattr(ev, "end_sec", None)})
            continue
        if ev.start_sec < last_end:
            # clamp forward to avoid overlap regressions
            ev.start_sec = last_end
            if ev.end_sec <= ev.start_sec:
                continue
        clean.append(ev)
        last_end = max(last_end, ev.end_sec)
    return clean

def transcribe(
    audio_path: str,
    config: Optional[PipelineConfig] = None,
    pipeline_logger: Optional[PipelineLogger] = None,
) -> TranscriptionResult:
    """
    High-level API: run the full pipeline on an audio file.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file (e.g., .wav, .mp3).
    config : PipelineConfig, optional
        Full pipeline configuration. If None, uses PIANO_61KEY_CONFIG.

    Returns
    -------
    TranscriptionResult
        Object with `.musicxml` (Stage D output) and `.analysis_data`
        (meta + stem timelines + notes).
    """
    if config is None:
        config = PIANO_61KEY_CONFIG

    # Deterministic seeding if requested
    if config.seed is not None:
        import random
        random.seed(config.seed)
        np.random.seed(config.seed)
        try:
            import torch
            torch.manual_seed(config.seed)
        except ImportError:
            pass

    pipeline_logger = pipeline_logger or PipelineLogger()
    stage_metrics: dict[str, dict[str, float]] = {}
    stage_b_out: Optional[StageBOutput] = None

    def _finalize_and_return(d_out: TranscriptionResult, mode: Optional[str] = None) -> TranscriptionResult:
        """Export artifacts/metrics, finalize logging, and return the result."""

        # Save resolved configuration for debugging parity with API path
        dump_resolved_config(config, stage_a_out.meta, stage_b_out)

        # --- Artifact exports (best-effort) ---
        try:
            ad = d_out.analysis_data  # TranscriptionResult.analysis_data
            # timeline.json
            timeline_rows = []
            for fp in getattr(ad, "timeline", []) or []:
                # FramePitch fields per your schema
                timeline_rows.append({
                    "time_sec": getattr(fp, "time", None),
                    "f0_hz": getattr(fp, "pitch_hz", None),
                    "midi": getattr(fp, "midi", None),
                    "confidence": getattr(fp, "confidence", None),
                    "rms": getattr(fp, "rms", None),
                    "active_pitches": getattr(fp, "active_pitches", None),
                })
            pipeline_logger.write_json("timeline.json", timeline_rows)

            # predicted_notes.json
            note_rows = []
            for ne in getattr(ad, "notes", []) or []:
                note_rows.append({
                    "start_sec": getattr(ne, "start_sec", None),
                    "end_sec": getattr(ne, "end_sec", None),
                    "midi_note": getattr(ne, "midi_note", None),
                    "pitch_hz": getattr(ne, "pitch_hz", None),
                    "confidence": getattr(ne, "confidence", None),
                    "velocity": getattr(ne, "velocity", None),
                    "voice": getattr(ne, "voice", None),
                    "staff": getattr(ne, "staff", None),
                })
            pipeline_logger.write_json("predicted_notes.json", note_rows)

            # resolved_config.json (dataclass + runtime meta)
            from dataclasses import asdict as _asdict, is_dataclass as _isdc
            resolved = _asdict(config) if _isdc(config) else {"config": str(config)}
            resolved["runtime"] = {
                "hop_length": getattr(stage_a_out.meta, "hop_length", None),
                "window_size": getattr(stage_a_out.meta, "window_size", None),
                "sample_rate": getattr(stage_a_out.meta, "sample_rate", None),
                "duration_sec": getattr(stage_a_out.meta, "duration_sec", None),
            }
            pipeline_logger.write_json("resolved_config.json", resolved)
            pipeline_logger.write_json("stage_metrics.json", stage_metrics)
            pipeline_logger.write_text("rendered.musicxml", d_out.musicxml or "")

            # summary.csv (append)
            summary_path = os.path.join(pipeline_logger.base_dir, "summary.csv")
            os.makedirs(pipeline_logger.base_dir, exist_ok=True)
            summary_fields = [
                "run_dir",
                "note_count",
                "duration_sec",
                "voiced_ratio",
                "mean_confidence",
            ]
            write_header = not os.path.exists(summary_path)
            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "run_dir": pipeline_logger.run_dir,
                    "note_count": stage_metrics.get("stage_c", {}).get("note_count", 0),
                    "duration_sec": stage_metrics.get("stage_a", {}).get("duration_out_sec", 0.0),
                    "voiced_ratio": stage_metrics.get("stage_b", {}).get("voiced_ratio", 0.0),
                    "mean_confidence": stage_metrics.get("stage_b", {}).get("mean_confidence", 0.0),
                })

            # leaderboard.json (best-effort)
            leaderboard_path = os.path.join(pipeline_logger.base_dir, "leaderboard.json")
            leaderboard_data = {}
            if os.path.exists(leaderboard_path):
                try:
                    with open(leaderboard_path, "r", encoding="utf-8") as f:
                        leaderboard_data = json.load(f) or {}
                except Exception:
                    leaderboard_data = {}
            leaderboard_data[pipeline_logger.run_name] = {
                "note_count": stage_metrics.get("stage_c", {}).get("note_count", 0),
                "voiced_ratio": stage_metrics.get("stage_b", {}).get("voiced_ratio", 0.0),
            }
            with open(leaderboard_path, "w", encoding="utf-8") as f:
                json.dump(leaderboard_data, f, indent=2)

            pipeline_logger.log_event(stage="pipeline", event="artifact_export",
                                      payload={"files": ["timeline.json", "predicted_notes.json", "resolved_config.json", "stage_metrics.json", "rendered.musicxml"]})
        except Exception as e:
            pipeline_logger.log_event(stage="pipeline", event="artifact_export_failed",
                                      payload={"error": str(e)})

        payload = {"notes": len(d_out.analysis_data.notes), "run_dir": pipeline_logger.run_dir}
        if mode:
            payload["mode"] = mode
        pipeline_logger.log_event("pipeline", "complete", payload)
        pipeline_logger.finalize()

        return TranscriptionResult(
            musicxml=d_out.musicxml,
            analysis_data=d_out.analysis_data,
            midi_bytes=d_out.midi_bytes
        )

    # Log missing optional module exactly once per process.
    if not _VALIDATION_AVAILABLE and not getattr(transcribe, "_logged_missing_validation", False):
        pipeline_logger.log_event(
            stage="pipeline",
            event="missing_dependency",
            payload={"module": "validation", "symbols": ["validate_invariants", "dump_resolved_config"]},
        )
        setattr(transcribe, "_logged_missing_validation", True)

    # --------------------------------------------------------
    # Stage A: Signal Conditioning
    # --------------------------------------------------------
    pipeline_logger.log_event(
        "stage_a",
        "start",
        {
            "audio_path": audio_path,
            "detector_preferences": config.stage_b.detectors if hasattr(config, "stage_b") else {},
        },
    )
    t_a = time.perf_counter()
    # Update: Pass full config to allow Stage A to resolve detector-based params
    stage_a_out: StageAOutput = load_and_preprocess(
        audio_path,
        config=config,
    )

    # --------------------------------------------------------
    # BPM Fallback (Gated Safety Net)
    # --------------------------------------------------------
    # Runs only if beat detection was enabled but failed to produce beats in Stage A
    # AND audio is long enough to warrant reliable detection.
    bpm_cfg = getattr(config.stage_a, "bpm_detection", {}) or {}
    bpm_enabled = bool(bpm_cfg.get("enabled", True))
    meta = stage_a_out.meta

    # Check conditions: Enabled AND (No beats or Default+NoBeats) AND Duration >= 6.0
    # Also verify existing beats are empty (do not overwrite)
    needs_fallback = (
        bpm_enabled
        and (len(meta.beats) == 0)
        and (meta.tempo_bpm is None or meta.tempo_bpm <= 0 or (meta.tempo_bpm == 120.0 and len(meta.beats) == 0))
        and (meta.duration_sec >= 6.0)
    )

    if needs_fallback:
        try:
            # Lazy import librosa check
            import importlib.util
            if importlib.util.find_spec("librosa"):
                pipeline_logger.log_event("stage_a", "bpm_fallback_triggered", {"reason": "missing_beats"})

                # Run fallback detection
                # We need the audio from the mix stem
                mix_stem = stage_a_out.stems.get("mix")
                if mix_stem:
                    fb_bpm, fb_beats = detect_tempo_and_beats(
                        mix_stem.audio,
                        sr=mix_stem.sr,
                        enabled=True,
                        tightness=float(bpm_cfg.get("tightness", 100.0)),
                        trim=bool(bpm_cfg.get("trim", True)),
                        hop_length=meta.hop_length,
                        pipeline_logger=pipeline_logger,
                    )

                    if fb_beats:
                        # Sort and dedup with epsilon to preserve order and density
                        fb_beats = sorted([float(b) for b in fb_beats])
                        dedup = []
                        last = None
                        for t in fb_beats:
                            if last is None or t > last + 1e-6:
                                dedup.append(t)
                                last = t
                        fb_beats = dedup

                        # Update in-place
                        meta.beats = fb_beats
                        meta.beat_times = fb_beats # Alias
                        if fb_bpm and fb_bpm > 0:
                            meta.tempo_bpm = fb_bpm

                        pipeline_logger.log_event("stage_a", "bpm_fallback_success", {
                            "bpm": fb_bpm,
                            "n_beats": len(fb_beats)
                        })
                    else:
                        pipeline_logger.log_event("stage_a", "bpm_fallback_no_result")
        except Exception as e:
            pipeline_logger.log_event("stage_a", "bpm_fallback_failed", {"error": str(e)})

    if len(stage_a_out.stems) == 1 and "mix" in stage_a_out.stems:
        pipeline_logger.log_event(
            stage="stage_a",
            event="separation_skipped",
            payload={"reason": "no_real_separation_outputs", "stems": list(stage_a_out.stems.keys())},
        )

    res_a = validate_invariants(stage_a_out, config, logger=pipeline_logger)
    if hasattr(stage_a_out, "diagnostics"):
        stage_a_out.diagnostics["contracts"] = res_a

    pipeline_logger.record_timing(
        "stage_a",
        time.perf_counter() - t_a,
        metadata={
            "sample_rate": stage_a_out.meta.sample_rate,
            "hop_length": stage_a_out.meta.hop_length,
            "window_size": stage_a_out.meta.window_size,
            "audio_type": stage_a_out.audio_type.value,
        },
    )
    original_duration = getattr(stage_a_out.meta, "original_duration_sec", stage_a_out.meta.duration_sec)
    trim_reduction = max(0.0, float(original_duration) - float(stage_a_out.meta.duration_sec))
    pipeline_logger.log_event(
        "pipeline",
        "params_resolved",
        {
            "sample_rate": stage_a_out.meta.sample_rate,
            "hop_length": stage_a_out.meta.hop_length,
            "window_size": stage_a_out.meta.window_size,
            "fmin": getattr(config.stage_b, "fmin", None) if hasattr(config, "stage_b") else None,
            "fmax": getattr(config.stage_b, "fmax", None) if hasattr(config, "stage_b") else None,
            "source": "stage_a.detector_resolution",
        },
    )
    stage_metrics["stage_a"] = {
        "duration_in_sec": float(original_duration),
        "duration_out_sec": float(stage_a_out.meta.duration_sec),
        "trim_reduction_sec": float(trim_reduction),
        "trim_offset_sec": float(trim_reduction) / 2.0,
        "sr_out": float(stage_a_out.meta.sample_rate),
        "loudness_or_rms": float(stage_a_out.meta.loudness_or_rms),
        "loudness_or_rms_post": float(stage_a_out.meta.loudness_post_norm),
        "loudness_measurement": stage_a_out.meta.loudness_measurement,
        "loudness_target_lufs": float(stage_a_out.meta.lufs),
        "normalization_gain_db": float(stage_a_out.meta.normalization_gain_db),
        "noise_floor_db": float(stage_a_out.meta.noise_floor_db),
    }

    # --------------------------------------------------------
    # Pipeline Branch: Onsets & Frames (Feature B)
    # --------------------------------------------------------
    # If O&F is enabled and functional, we skip classical B/C and go straight to D.
    of_notes = []
    of_diag = {}

    # Check if enabled in config
    if config.stage_b.onsets_and_frames.get("enabled", False):
        t_of = time.perf_counter()

        # We need the mix audio. Stage A provides stems["mix"].
        mix_audio = stage_a_out.stems["mix"].audio
        sr = stage_a_out.meta.sample_rate

        of_notes_candidate, of_diag = transcribe_onsets_frames(mix_audio, sr, config)
        note_count = len(of_notes_candidate) if of_notes_candidate else 0

        pipeline_logger.record_timing(
            "onsets_frames",
            time.perf_counter() - t_of,
            metadata=of_diag
        )

        if not of_diag.get("run", False) or note_count == 0:
            pipeline_logger.log_event(
                stage="pipeline",
                event="onsets_frames_fallback",
                payload={"reason": of_diag.get("reason", "zero_notes"), "note_count": note_count},
            )
            # fall through to classical B/C path
        else:
            of_notes = of_notes_candidate
            pipeline_logger.log_event("pipeline", "branch_switch", {"mode": "onsets_frames"})

    if of_notes:
        # --------------------------------------------------------
        # Bypass Stage B/C -> Stage D
        # --------------------------------------------------------
        # We construct AnalysisData with the detected notes.
        # We leave stem_timelines empty or minimal.

        analysis_data = AnalysisData(
            meta=stage_a_out.meta,
            timeline=[], # No FramePitch timeline from O&F
            stem_timelines={},
            notes=of_notes,
            pitch_tracker="onsets_frames"
        )

        # We skip apply_theory, but we must populate analysis_data.events or notes
        # which we did above.

        # --------------------------------------------------------
        # Stage D: Quantization + MusicXML Rendering
        # --------------------------------------------------------
        t_d = time.perf_counter()
        d_out: TranscriptionResult = quantize_and_render(
            of_notes,
            analysis_data,
            config=config,
        )

        res_d = validate_invariants(d_out, config, logger=pipeline_logger)
        if hasattr(d_out.analysis_data, "diagnostics"):
            d_out.analysis_data.diagnostics.setdefault("contracts", {})["stage_d"] = res_d

        pipeline_logger.record_timing(
            "stage_d",
            time.perf_counter() - t_d,
            metadata={"beats_detected": len(d_out.analysis_data.beats), "mode": "onsets_frames"},
        )

        stage_metrics["stage_b"] = {
            "voiced_ratio": 0.0,
            "mean_confidence": 0.0,
            "octave_jump_rate": 0.0,
            "f0_zero_ratio": 0.0,
            "timeline_frames": len(getattr(analysis_data, "timeline", []) or []),
        }

        note_durations = [max(0.0, float(n.end_sec) - float(n.start_sec)) for n in of_notes or [] if getattr(n, "end_sec", 0.0) > getattr(n, "start_sec", 0.0)]
        duration_sec = float(getattr(stage_a_out.meta, "duration_sec", 0.0) or 0.0)
        stage_metrics["stage_c"] = {
            "note_count": len(of_notes or []),
            "note_count_per_10s": (len(of_notes or []) / (duration_sec / 10.0)) if duration_sec else 0.0,
            "median_note_len_ms": float(np.median(note_durations) * 1000.0) if note_durations else 0.0,
            "fragmentation_score": float(sum(1 for d in note_durations if d < 0.08) / max(1, len(of_notes or []))) if of_notes else 0.0,
        }

        stage_metrics["stage_d"] = {
            "quantization_mean_shift_ms": 0.0,
            "quantization_p95_shift_ms": 0.0,
            "rendered_notes": len(getattr(d_out.analysis_data, "notes", []) or []),
        }

        return _finalize_and_return(d_out, mode="onsets_frames")

    # --------------------------------------------------------
    # SEGMENTED TRANSCRIPTION LOGIC (Optional)
    # --------------------------------------------------------
    seg_conf = getattr(config, "segmented_transcription", None)
    use_segmented = (
        seg_conf and
        seg_conf.enabled and
        stage_a_out.meta.duration_sec > seg_conf.segment_sec
    )

    if use_segmented:
        pipeline_logger.log_event("pipeline", "segmented_mode_start", asdict(seg_conf))

        # Determine segments
        duration = stage_a_out.meta.duration_sec
        seg_len = seg_conf.segment_sec
        overlap = seg_conf.overlap_sec
        step = seg_len - overlap
        if step <= 0:
            step = seg_len # safety

        accumulated_notes = []

        # Loop segments
        current_time = 0.0
        seg_idx = 0

        # Keep track of last stitched point to know valid regions
        # Actually stitch logic handles regions.

        while current_time < duration:
            seg_start = current_time
            seg_end = min(duration, current_time + seg_len)

            pipeline_logger.log_event("segment", "start", {"index": seg_idx, "start": seg_start, "end": seg_end})

            # Slice audio (Stage A subset)
            seg_stage_a = _slice_stage_a_output(stage_a_out, seg_start, seg_end)

            # Build candidates (C1, C2, C3)
            # Use global audio type, but allow lightweight check?
            # (User requested lightweight check: silence/low rms -> skip poly)
            # For now we use global + simple candidates.
            candidates = _build_candidate_configs(config, stage_a_out.audio_type)
            candidates = candidates[:seg_conf.retry_max_candidates]

            best_notes = []
            best_score = -1.0
            selected_cand_idx = 0

            for c_idx, cand_config in enumerate(candidates):
                # Run Stage B
                sb_out = extract_features(seg_stage_a, config=cand_config)

                # Run Stage C
                sc_analysis = AnalysisData(
                    meta=seg_stage_a.meta,
                    timeline=[],
                    stem_timelines=sb_out.stem_timelines
                )
                cand_notes = apply_theory(sc_analysis, config=cand_config)

                # Score
                score = _score_segment(cand_notes, seg_end - seg_start, stage_a_out.audio_type, config)

                # Re-calculate stats for logging since _score_segment doesn't return them
                durations = []
                for n in cand_notes or []:
                    s = getattr(n, "start_sec", 0.0) or 0.0
                    e = getattr(n, "end_sec", 0.0) or 0.0
                    if e > s:
                        durations.append(e - s)
                median_dur = float(np.median(durations)) if durations else 0.0
                seg_duration_sec = seg_end - seg_start
                notes_per_sec = (len(cand_notes) / max(1e-6, seg_duration_sec)) if seg_duration_sec else 0.0

                pipeline_logger.log_event("segment", "candidate_evaluated", {
                    "segment_index": seg_idx,
                    "candidate_index": c_idx,
                    "score": score,
                    "note_count": len(cand_notes),
                    "median_dur_s": median_dur,
                    "notes_per_sec": notes_per_sec
                })

                if score > best_score:
                    best_score = score
                    best_notes = cand_notes
                    selected_cand_idx = c_idx

                if score >= seg_conf.retry_quality_threshold:
                    break

            pipeline_logger.log_event("segment", "segment_complete", {
                "index": seg_idx,
                "selected_candidate": selected_cand_idx,
                "final_score": best_score,
                "notes_count": len(best_notes)
            })

            # Time-shift notes to global time
            # Stage B output timelines start at 0.0 relative to slice.
            for n in best_notes:
                n.start_sec += seg_start
                n.end_sec += seg_start

            # Stitch
            overlap_start = current_time + step # Start of overlap region in next iteration
            # But for *this* iteration, we are merging into previous.
            # The overlap region with previous segment was [seg_start, seg_start + overlap]
            # The overlap region with next segment is [seg_end - overlap, seg_end]

            # _stitch_events merges "accumulated" (processed up to now) with "new" (current segment)
            # The overlap happens at `seg_start`.
            # Overlap region is roughly [seg_start, seg_start + overlap]
            # (assuming previous segment ended at seg_start + overlap)

            # Correct logic:
            # We append to accumulated_notes.
            # The "stitch zone" is where the previous segment and this segment overlap.
            # Previous segment covered [prev_start, prev_end].
            # This segment covers [seg_start, seg_end].
            # prev_end was approx seg_start + overlap.
            # So overlap is [seg_start, prev_end].

            # Since we iterate by `step`, seg_start = prev_start + step.
            # prev_end = prev_start + seg_len.
            # So prev_end = seg_start - step + seg_len = seg_start + overlap.
            # Correct.

            merge_start = seg_start
            merge_end = seg_start + overlap

            if seg_idx == 0:
                accumulated_notes = best_notes
            else:
                accumulated_notes = _stitch_events(accumulated_notes, best_notes, merge_start, merge_end, pipeline_logger)

            # Prepare next loop
            current_time += step
            seg_idx += 1

        # Final Analysis Data construction
        # We need to build a dummy StageB output or just populate AnalysisData directly
        # The contract expects `stage_b_out` for `dump_resolved_config` later, but
        # `dump_resolved_config` might handle missing bits?
        # Actually we can just proceed to Stage D with populated AnalysisData.

        # We'll create a synthetic StageBOutput for logging/dumping purposes?
        # Or just reuse the last one?
        # For simplicity, we reuse the last one but warn it's partial.
        # Ideally we aggregate timelines, but that's heavy.
        # We will populate AnalysisData.stem_timelines with empty/minimal data
        # because Stage D mostly consumes `notes` (except for visualization).

        # Note: Stage D `quantize_and_render` takes `notes` as first arg.
        analysis_data = AnalysisData(
            meta=stage_a_out.meta,
            timeline=[],
            stem_timelines={}, # We don't stitch timelines yet, only notes.
            notes=accumulated_notes,
            beats=stage_a_out.beats if stage_a_out.beats else getattr(stage_a_out.meta, "beats", [])
        )

        # Use the last run's Stage B output for invariants/logging
        stage_b_out = StageBOutput(
            time_grid=np.array([]),
            f0_main=np.array([]),
            f0_layers=[],
            per_detector={},
            meta=stage_a_out.meta,
            diagnostics={"mode": "segmented"}
        )

        # Pass the accumulated notes to Stage D
        notes = accumulated_notes

    else:
        # --------------------------------------------------------
        # ORIGINAL SINGLE-PASS PATH
        # --------------------------------------------------------

        # --------------------------------------------------------
        # Stage B: Feature Extraction (Detectors + Ensemble)
        # --------------------------------------------------------
        pipeline_logger.log_event(
            "stage_b",
            "detector_selection",
            {
                "detectors": config.stage_b.detectors if hasattr(config, "stage_b") else {},
                "dependencies": PipelineLogger.dependency_snapshot(["torch", "crepe", "demucs"]),
            },
        )
        t_b = time.perf_counter()
        stage_b_out = extract_features(
            stage_a_out,
            config=config,
            pipeline_logger=pipeline_logger,
        )
        res_b = validate_invariants(stage_b_out, config, logger=pipeline_logger)
        if hasattr(stage_b_out, "diagnostics"):
            stage_b_out.diagnostics["contracts"] = res_b

        pipeline_logger.record_timing(
            "stage_b",
            time.perf_counter() - t_b,
            metadata={
                "resolved_hop": stage_a_out.meta.hop_length,
                "resolved_window": stage_a_out.meta.window_size,
                "detectors_run": list(stage_b_out.per_detector.get("mix", {}).keys()),
                "crepe_used": "crepe" in list(stage_b_out.per_detector.get("mix", {}).keys()),
                "iss_layers": stage_b_out.diagnostics.get("iss", {}).get("layers_found", 0),
            },
        )

        # --------------------------------------------------------
        # Stage C: Note Event Extraction (Theory Application)
        # --------------------------------------------------------
        analysis_data = AnalysisData(
            meta=stage_a_out.meta,
            timeline=[],
            stem_timelines=stage_b_out.stem_timelines,
        )

        pipeline_logger.log_event(
            "stage_c",
            "segmentation",
            {
                "method": config.stage_c.segmentation_method.get("method") if hasattr(config, "stage_c") else None,
                "pitch_tolerance_cents": getattr(config.stage_c, "pitch_tolerance_cents", None) if hasattr(config, "stage_c") else None,
            },
        )
        t_c = time.perf_counter()
        notes = apply_theory(
            analysis_data,
            config=config,
        )
        res_c = validate_invariants(notes, config, analysis_data=analysis_data, logger=pipeline_logger)
        if hasattr(analysis_data, "diagnostics"):
            analysis_data.diagnostics.setdefault("contracts", {})["stage_c"] = res_c
            # Consolidate previous contract statuses into AnalysisData
            if stage_a_out and hasattr(stage_a_out, "diagnostics"):
                analysis_data.diagnostics.setdefault("contracts", {})["stage_a"] = stage_a_out.diagnostics.get("contracts")
            if stage_b_out and hasattr(stage_b_out, "diagnostics"):
                analysis_data.diagnostics.setdefault("contracts", {})["stage_b"] = stage_b_out.diagnostics.get("contracts")

        pipeline_logger.record_timing("stage_c", time.perf_counter() - t_c, metadata={"note_count": len(notes)})

    # --------------------------------------------------------
    # Stage B/C diagnostics (common)
    # --------------------------------------------------------
    timeline_source = list(stage_b_out.timeline or [])
    if not timeline_source and getattr(stage_b_out, "time_grid", None) is not None:
        # Robustly rebuild timeline if missing from stage_b_out
        tg = getattr(stage_b_out, "time_grid", []) or []
        f0m = getattr(stage_b_out, "f0_main", []) or []
        for t, f0 in zip(tg, f0m):
            f0 = float(f0)
            if f0 > 0.0:
                midi = int(round(69 + 12 * math.log2(f0 / 440.0)))
                timeline_source.append(FramePitch(time=float(t), pitch_hz=f0, midi=midi, confidence=0.0, rms=0.0))
            else:
                timeline_source.append(FramePitch(time=float(t), pitch_hz=0.0, midi=None, confidence=0.0, rms=0.0))

    total_frames = len(timeline_source) or int(getattr(stage_b_out, "f0_main", np.array([])).size)
    voiced_frames = [fp for fp in timeline_source if getattr(fp, "pitch_hz", 0.0) > 0]
    mean_conf = float(np.mean([getattr(fp, "confidence", 0.0) for fp in timeline_source])) if timeline_source else 0.0

    # Use consistent denominator
    den = max(1, len(timeline_source))
    voiced_ratio = len(voiced_frames) / den
    zero_ratio = 1.0 - voiced_ratio

    midi_series = [fp.midi for fp in timeline_source if fp.midi is not None]
    octave_jumps = 0
    for a, b in zip(midi_series, midi_series[1:]):
        if a is None or b is None:
            continue
        if abs(a - b) >= 12:
            octave_jumps += 1

    stage_metrics["stage_b"] = {
        "voiced_ratio": float(voiced_ratio),
        "mean_confidence": float(mean_conf),
        "octave_jump_rate": float(octave_jumps / max(1, len(midi_series))) if midi_series else 0.0,
        "f0_zero_ratio": float(zero_ratio),
        "timeline_frames": int(total_frames),
    }

    # Stage C fragmentation statistics
    note_durations = [max(0.0, float(n.end_sec) - float(n.start_sec)) for n in notes or [] if getattr(n, "end_sec", 0.0) > getattr(n, "start_sec", 0.0)]
    short_notes = sum(1 for d in note_durations if d < 0.08)
    duration_sec = float(getattr(stage_a_out.meta, "duration_sec", 0.0) or 0.0)
    stage_metrics["stage_c"] = {
        "note_count": len(notes or []),
        "note_count_per_10s": (len(notes or []) / (duration_sec / 10.0)) if duration_sec else 0.0,
        "median_note_len_ms": float(np.median(note_durations) * 1000.0) if note_durations else 0.0,
        "fragmentation_score": float(short_notes / max(1, len(notes or []))) if notes else 0.0,
    }

    # --------------------------------------------------------
    # Stage D: Quantization + MusicXML Rendering
    # --------------------------------------------------------
    t_d = time.perf_counter()
    d_out: TranscriptionResult = quantize_and_render(
        notes,
        analysis_data,
        config=config,
        pipeline_logger=pipeline_logger,
    )
    res_d = validate_invariants(d_out, config, logger=pipeline_logger)
    if hasattr(d_out.analysis_data, "diagnostics"):
        d_out.analysis_data.diagnostics.setdefault("contracts", {})["stage_d"] = res_d

    pipeline_logger.record_timing(
        "stage_d",
        time.perf_counter() - t_d,
        metadata={"beats_detected": len(d_out.analysis_data.beats)},
    )
    stage_metrics["stage_d"] = {
        "quantization_mean_shift_ms": 0.0,
        "quantization_p95_shift_ms": 0.0,
        "rendered_notes": len(getattr(d_out.analysis_data, "notes", []) or []),
    }

    return _finalize_and_return(d_out)
