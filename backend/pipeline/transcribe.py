# backend/pipeline/transcribe.py
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
from .global_profiles import apply_global_profile
from .config import PIANO_61KEY_CONFIG, PipelineConfig
from .stage_a import load_and_preprocess, detect_tempo_and_beats
from .stage_b import extract_features
from .stage_c import apply_theory
from .stage_d import quantize_and_render
from .neural_transcription import transcribe_onsets_frames
from .models import AnalysisData, StageAOutput, TranscriptionResult, StageBOutput, NoteEvent, FramePitch, Stem, AudioType

# Step 1: Import debug artifact writers
try:
    from .debug import write_frame_timeline_csv
except ImportError:
    write_frame_timeline_csv = None

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

    mix = stage_a_out.stems.get("mix")
    if mix:
        total_samples = len(mix.audio)
    else:
        any_stem = next(iter(stage_a_out.stems.values()), None)
        if any_stem:
            total_samples = len(any_stem.audio)

    start_idx = int(max(0, start_sec * sr))
    end_idx = int(min(total_samples, end_sec * sr))

    new_stems = {}
    for name, stem in stage_a_out.stems.items():
        audio_slice = stem.audio[start_idx:end_idx]
        new_stems[name] = Stem(audio=audio_slice, sr=stem.sr, type=stem.type)

    new_meta = replace(stage_a_out.meta)
    new_meta.duration_sec = (end_idx - start_idx) / sr

    src_beats = stage_a_out.beats if stage_a_out.beats else getattr(stage_a_out.meta, "beats", [])
    sliced_beats = []
    if src_beats:
        sliced_beats = [b - start_sec for b in src_beats if start_sec <= b <= end_sec]

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
    notes_raw: list[NoteEvent],
    duration_sec: float,
    audio_type: AudioType,
    config: PipelineConfig
) -> float:
    """
    Compute quality score [0,1] for a transcribed segment.

    ✅ FIX: Score on RAW segmentation notes (pre-quantization) to avoid
    quantization hiding fragmentation/density issues.
    """
    if duration_sec <= 0:
        return 0.0

    note_count = len(notes_raw)
    if note_count == 0:
        return 1.0

    density = note_count / duration_sec
    target = config.segmented_transcription.density_target_notes_per_sec
    span = config.segmented_transcription.density_penalty_span

    excess = max(0.0, density - target)
    density_score = 1.0 - np.clip(excess / span, 0.0, 1.0)

    # min duration (guard None)
    min_dur = float(config.stage_c.min_note_duration_ms) / 1000.0
    if audio_type in (AudioType.POLYPHONIC, AudioType.POLYPHONIC_DOMINANT):
        poly_ms = getattr(config.stage_c, "min_note_duration_ms_poly", None)
        if poly_ms is not None:
            try:
                min_dur = max(min_dur, float(poly_ms) / 1000.0)
            except Exception:
                pass

    short_notes = sum(1 for n in notes_raw if (n.end_sec - n.start_sec) < min_dur)
    plausibility = 1.0 - (short_notes / max(1, note_count))

    durations = []
    for n in notes_raw:
        s = float(getattr(n, "start_sec", 0.0) or 0.0)
        e = float(getattr(n, "end_sec", 0.0) or 0.0)
        if e > s:
            durations.append(e - s)

    median_dur = float(np.median(durations)) if durations else 0.0
    notes_per_sec = note_count / max(1e-6, duration_sec)

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
    configs.append(base_config)

    c2 = copy.deepcopy(base_config)
    if segment_audio_type == AudioType.POLYPHONIC:
        c2.stage_b.polyphonic_peeling["iss_adaptive"] = True
    else:
        if "crepe" in c2.stage_b.detectors:
            c2.stage_b.detectors["crepe"]["enabled"] = True
        if "yin" in c2.stage_b.detectors:
            c2.stage_b.detectors["yin"]["enable_multires_f0"] = True
    configs.append(c2)

    c3 = copy.deepcopy(base_config)
    c3.stage_c.confidence_threshold *= 0.8
    if hasattr(c3.stage_c, "gap_tolerance_s"):
        c3.stage_c.gap_tolerance_s *= 2.0
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
    """
    safe_history = []
    candidates_history = []

    for n in accumulated_notes:
        if n.end_sec < overlap_start:
            safe_history.append(n)
        else:
            candidates_history.append(n)

    merged = list(safe_history)
    all_overlap = sorted(candidates_history + new_segment_notes, key=lambda x: x.start_sec)

    if not all_overlap:
        return merged

    current = all_overlap[0]

    for next_note in all_overlap[1:]:
        TOLERANCE = 0.05
        is_same_pitch = (current.midi_note == next_note.midi_note)
        is_overlapping = (current.start_sec - TOLERANCE <= next_note.end_sec) and (current.end_sec + TOLERANCE >= next_note.start_sec)

        if is_same_pitch and is_overlapping:
            new_start = min(current.start_sec, next_note.start_sec)
            new_end = max(current.end_sec, next_note.end_sec)

            take_next = next_note.confidence > current.confidence
            new_conf = max(current.confidence, next_note.confidence)
            new_vel = max(current.velocity, next_note.velocity)

            w1 = current.confidence
            w2 = next_note.confidence
            if w1 + w2 > 0:
                new_hz = (current.pitch_hz * w1 + next_note.pitch_hz * w2) / (w1 + w2)
            else:
                new_hz = current.pitch_hz

            current.start_sec = new_start
            current.end_sec = new_end
            current.confidence = new_conf
            current.velocity = new_vel
            current.pitch_hz = new_hz
            if take_next:
                current.voice = next_note.voice
                current.staff = next_note.staff
        else:
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
    if config is None:
        config = PIANO_61KEY_CONFIG

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
        dump_resolved_config(config, stage_a_out.meta, stage_b_out)

        try:
            ad = d_out.analysis_data
            timeline_rows = []
            csv_rows = []

            for fp in getattr(ad, "timeline", []) or []:
                t_row = {
                    "time_sec": getattr(fp, "time", None),
                    "f0_hz": getattr(fp, "pitch_hz", None),
                    "midi": getattr(fp, "midi", None),
                    "confidence": getattr(fp, "confidence", None),
                    "rms": getattr(fp, "rms", None),
                    "active_pitches": getattr(fp, "active_pitches", None),
                }
                timeline_rows.append(t_row)

                if write_frame_timeline_csv:
                    hz = t_row["f0_hz"] or 0.0
                    if hz > 0.0:
                        cents = 1200.0 * math.log2(hz / 440.0) + 6900.0
                    else:
                        cents = float("nan")

                    fused_c = cents
                    smoothed_c = cents
                    idx = len(timeline_rows) - 1

                    if stage_b_out and stage_b_out.diagnostics and "debug_curves" in stage_b_out.diagnostics:
                        curves = stage_b_out.diagnostics["debug_curves"].get("mix") or next(iter(stage_b_out.diagnostics["debug_curves"].values()), None)
                        if curves:
                            fused_arr = curves.get("fused_f0")
                            smoothed_arr = curves.get("smoothed_f0")
                            if fused_arr is not None and idx < len(fused_arr):
                                fval = fused_arr[idx]
                                fused_c = 1200.0 * math.log2(fval / 440.0) + 6900.0 if fval > 0 else float("nan")
                            if smoothed_arr is not None and idx < len(smoothed_arr):
                                sval = smoothed_arr[idx]
                                smoothed_c = 1200.0 * math.log2(sval / 440.0) + 6900.0 if sval > 0 else float("nan")

                    csv_rows.append({
                        "t_sec": t_row["time_sec"],
                        "f0_hz": t_row["f0_hz"],
                        "midi": t_row["midi"],
                        "cents": cents,
                        "confidence": t_row["confidence"],
                        "voiced": (t_row["f0_hz"] or 0) > 0,
                        "detector_name": "fused",
                        "harmonic_rank": 1,
                        "fused_cents": fused_c,
                        "smoothed_cents": smoothed_c,
                    })

            pipeline_logger.write_json("timeline.json", timeline_rows)

            if write_frame_timeline_csv and csv_rows:
                if pipeline_logger.base_dir:
                    csv_path = os.path.join(pipeline_logger.base_dir, "timeline.csv")
                    write_frame_timeline_csv(csv_path, csv_rows)
                    pipeline_logger.log_event("pipeline", "artifact_export", {"files": ["timeline.csv"]})

            # ✅ Quantized notes (post Stage D)
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
                    "measure": getattr(ne, "measure", None),
                    "beat": getattr(ne, "beat", None),
                    "duration_beats": getattr(ne, "duration_beats", None),
                })
            pipeline_logger.write_json("predicted_notes.json", note_rows)

            # ✅ Raw notes (pre-quantization) for benchmarks/diagnostics
            raw_rows = []
            for ne in getattr(ad, "notes_before_quantization", []) or []:
                raw_rows.append({
                    "start_sec": getattr(ne, "start_sec", None),
                    "end_sec": getattr(ne, "end_sec", None),
                    "midi_note": getattr(ne, "midi_note", None),
                    "pitch_hz": getattr(ne, "pitch_hz", None),
                    "confidence": getattr(ne, "confidence", None),
                    "velocity": getattr(ne, "velocity", None),
                    "voice": getattr(ne, "voice", None),
                    "staff": getattr(ne, "staff", None),
                })
            pipeline_logger.write_json("predicted_notes_raw.json", raw_rows)

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

            summary_path = os.path.join(pipeline_logger.base_dir, "summary.csv")
            os.makedirs(pipeline_logger.base_dir, exist_ok=True)
            summary_fields = ["run_dir", "note_count", "duration_sec", "voiced_ratio", "mean_confidence"]
            write_header = not os.path.exists(summary_path)
            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_fields)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "run_dir": pipeline_logger.run_dir,
                    "note_count": stage_metrics.get("stage_d", {}).get("rendered_notes", 0),
                    "duration_sec": stage_metrics.get("stage_a", {}).get("duration_out_sec", 0.0),
                    "voiced_ratio": stage_metrics.get("stage_b", {}).get("voiced_ratio", 0.0),
                    "mean_confidence": stage_metrics.get("stage_b", {}).get("mean_confidence", 0.0),
                })

            leaderboard_path = os.path.join(pipeline_logger.base_dir, "leaderboard.json")
            leaderboard_data = {}
            if os.path.exists(leaderboard_path):
                try:
                    with open(leaderboard_path, "r", encoding="utf-8") as f:
                        leaderboard_data = json.load(f) or {}
                except Exception:
                    leaderboard_data = {}
            leaderboard_data[pipeline_logger.run_name] = {
                "note_count": stage_metrics.get("stage_d", {}).get("rendered_notes", 0),
                "voiced_ratio": stage_metrics.get("stage_b", {}).get("voiced_ratio", 0.0),
            }
            with open(leaderboard_path, "w", encoding="utf-8") as f:
                json.dump(leaderboard_data, f, indent=2)

            pipeline_logger.log_event("pipeline", "artifact_export",
                                      {"files": ["timeline.json", "predicted_notes.json", "predicted_notes_raw.json", "resolved_config.json", "stage_metrics.json", "rendered.musicxml"]})
        except Exception as e:
            pipeline_logger.log_event("pipeline", "artifact_export_failed", {"error": str(e)})

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

    if not _VALIDATION_AVAILABLE and not getattr(transcribe, "_logged_missing_validation", False):
        pipeline_logger.log_event(
            stage="pipeline",
            event="missing_dependency",
            payload={"module": "validation", "symbols": ["validate_invariants", "dump_resolved_config"]},
        )
        setattr(transcribe, "_logged_missing_validation", True)

    # ---------------- Stage A ----------------
    pipeline_logger.log_event(
        "stage_a",
        "start",
        {"audio_path": audio_path, "detector_preferences": config.stage_b.detectors if hasattr(config, "stage_b") else {}},
    )
    t_a = time.perf_counter()
    stage_a_out: StageAOutput = load_and_preprocess(audio_path, config=config)

    apply_global_profile(audio_path=audio_path, stage_a_out=stage_a_out, config=config, pipeline_logger=pipeline_logger)

    # BPM fallback (unchanged)
    bpm_cfg = getattr(config.stage_a, "bpm_detection", {}) or {}
    bpm_enabled = bool(bpm_cfg.get("enabled", True))
    meta = stage_a_out.meta
    needs_fallback = (
        bpm_enabled
        and (len(meta.beats) == 0)
        and (meta.tempo_bpm is None or meta.tempo_bpm <= 0 or (meta.tempo_bpm == 120.0 and len(meta.beats) == 0))
        and (meta.duration_sec >= 6.0)
    )
    if needs_fallback:
        try:
            import importlib.util
            if importlib.util.find_spec("librosa"):
                pipeline_logger.log_event("stage_a", "bpm_fallback_triggered", {"reason": "missing_beats"})
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
                        fb_beats = sorted([float(b) for b in fb_beats])
                        dedup = []
                        last = None
                        for t in fb_beats:
                            if last is None or t > last + 1e-6:
                                dedup.append(t)
                                last = t
                        fb_beats = dedup
                        meta.beats = fb_beats
                        meta.beat_times = fb_beats
                        if fb_bpm and fb_bpm > 0:
                            meta.tempo_bpm = fb_bpm
                        if hasattr(stage_a_out, "diagnostics"):
                            stage_a_out.diagnostics.setdefault("fallbacks", []).append("bpm_detection")
                        pipeline_logger.log_event("stage_a", "bpm_fallback_success", {"bpm": fb_bpm, "n_beats": len(fb_beats)})
                    else:
                        pipeline_logger.log_event("stage_a", "bpm_fallback_no_result")
        except Exception as e:
            pipeline_logger.log_event("stage_a", "bpm_fallback_failed", {"error": str(e)})

    if len(stage_a_out.stems) == 1 and "mix" in stage_a_out.stems:
        pipeline_logger.log_event("stage_a", "separation_skipped",
                                  {"reason": "no_real_separation_outputs", "stems": list(stage_a_out.stems.keys())})

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

    # ---------------- Onsets & Frames branch ----------------
    of_notes = []
    of_diag = {}
    if config.stage_b.onsets_and_frames.get("enabled", False):
        t_of = time.perf_counter()
        mix_audio = stage_a_out.stems["mix"].audio
        sr = stage_a_out.meta.sample_rate
        of_notes_candidate, of_diag = transcribe_onsets_frames(mix_audio, sr, config)
        note_count = len(of_notes_candidate) if of_notes_candidate else 0
        pipeline_logger.record_timing("onsets_frames", time.perf_counter() - t_of, metadata=of_diag)

        if not of_diag.get("run", False) or note_count == 0:
            pipeline_logger.log_event("pipeline", "onsets_frames_fallback",
                                      {"reason": of_diag.get("reason", "zero_notes"), "note_count": note_count})
            if hasattr(stage_a_out, "diagnostics"):
                stage_a_out.diagnostics.setdefault("fallbacks", []).append("onsets_frames")
        else:
            of_notes = of_notes_candidate
            pipeline_logger.log_event("pipeline", "branch_switch", {"mode": "onsets_frames"})

    if of_notes:
        analysis_data = AnalysisData(
            meta=stage_a_out.meta,
            timeline=[],
            stem_timelines={},
            notes=list(of_notes),
            notes_before_quantization=copy.deepcopy(list(of_notes)),  # ✅
            pitch_tracker="onsets_frames"
        )

        t_d = time.perf_counter()
        d_out: TranscriptionResult = quantize_and_render(
            copy.deepcopy(list(of_notes)),  # ✅ pass raw (defensive copy)
            analysis_data,
            config=config,
        )

        res_d = validate_invariants(d_out, config, logger=pipeline_logger)
        if hasattr(d_out.analysis_data, "diagnostics"):
            d_out.analysis_data.diagnostics.setdefault("contracts", {})["stage_d"] = res_d

        pipeline_logger.record_timing("stage_d", time.perf_counter() - t_d,
                                      metadata={"beats_detected": len(d_out.analysis_data.beats), "mode": "onsets_frames"})

        stage_metrics["stage_b"] = {
            "voiced_ratio": 0.0,
            "mean_confidence": 0.0,
            "octave_jump_rate": 0.0,
            "f0_zero_ratio": 0.0,
            "timeline_frames": 0,
        }

        raw_durations = [max(0.0, float(n.end_sec) - float(n.start_sec)) for n in of_notes if n.end_sec > n.start_sec]
        duration_sec = float(getattr(stage_a_out.meta, "duration_sec", 0.0) or 0.0)
        stage_metrics["stage_c"] = {
            "note_count_raw": len(of_notes),
            "note_count_per_10s_raw": (len(of_notes) / (duration_sec / 10.0)) if duration_sec else 0.0,
            "median_note_len_ms_raw": float(np.median(raw_durations) * 1000.0) if raw_durations else 0.0,
            "fragmentation_score_raw": float(sum(1 for d in raw_durations if d < 0.08) / max(1, len(of_notes))),
        }

        stage_metrics["stage_d"] = {
            "quantization_mean_shift_ms": 0.0,
            "quantization_p95_shift_ms": 0.0,
            "rendered_notes": len(getattr(d_out.analysis_data, "notes", []) or []),
        }

        return _finalize_and_return(d_out, mode="onsets_frames")

    # ---------------- Segmented mode ----------------
    seg_conf = getattr(config, "segmented_transcription", None)
    use_segmented = (seg_conf and seg_conf.enabled and stage_a_out.meta.duration_sec > seg_conf.segment_sec)

    if use_segmented:
        pipeline_logger.log_event("pipeline", "segmented_mode_start", asdict(seg_conf))

        duration = stage_a_out.meta.duration_sec
        seg_len = seg_conf.segment_sec
        overlap = seg_conf.overlap_sec
        step = seg_len - overlap
        if step <= 0:
            step = seg_len

        accumulated_raw: list[NoteEvent] = []
        current_time = 0.0
        seg_idx = 0

        while current_time < duration:
            seg_start = current_time
            seg_end = min(duration, current_time + seg_len)

            pipeline_logger.log_event("segment", "start", {"index": seg_idx, "start": seg_start, "end": seg_end})

            seg_stage_a = _slice_stage_a_output(stage_a_out, seg_start, seg_end)

            candidates = _build_candidate_configs(config, stage_a_out.audio_type)
            candidates = candidates[:seg_conf.retry_max_candidates]

            best_notes_raw: list[NoteEvent] = []
            best_score = -1.0
            selected_cand_idx = 0

            for c_idx, cand_config in enumerate(candidates):
                sb_out = extract_features(seg_stage_a, config=cand_config)

                sc_analysis = AnalysisData(meta=seg_stage_a.meta, timeline=[], stem_timelines=sb_out.stem_timelines)
                _ = apply_theory(sc_analysis, config=cand_config)

                cand_raw = list(getattr(sc_analysis, "notes_before_quantization", []) or [])
                score = _score_segment(cand_raw, seg_end - seg_start, stage_a_out.audio_type, cand_config)  # ✅ cand_config

                durations = []
                for n in cand_raw:
                    s = float(getattr(n, "start_sec", 0.0) or 0.0)
                    e = float(getattr(n, "end_sec", 0.0) or 0.0)
                    if e > s:
                        durations.append(e - s)
                median_dur = float(np.median(durations)) if durations else 0.0
                seg_duration_sec = seg_end - seg_start
                notes_per_sec = (len(cand_raw) / max(1e-6, seg_duration_sec)) if seg_duration_sec else 0.0

                pipeline_logger.log_event("segment", "candidate_evaluated", {
                    "segment_index": seg_idx,
                    "candidate_index": c_idx,
                    "score": score,
                    "note_count_raw": len(cand_raw),
                    "median_dur_s_raw": median_dur,
                    "notes_per_sec_raw": notes_per_sec
                })

                if score > best_score:
                    best_score = score
                    best_notes_raw = cand_raw
                    selected_cand_idx = c_idx

                if score >= seg_conf.retry_quality_threshold:
                    break

            pipeline_logger.log_event("segment", "segment_complete", {
                "index": seg_idx,
                "selected_candidate": selected_cand_idx,
                "final_score": best_score,
                "notes_count_raw": len(best_notes_raw)
            })

            # shift to global
            for n in best_notes_raw:
                n.start_sec += seg_start
                n.end_sec += seg_start

            merge_start = seg_start
            merge_end = seg_start + overlap

            if seg_idx == 0:
                accumulated_raw = best_notes_raw
            else:
                accumulated_raw = _stitch_events(accumulated_raw, best_notes_raw, merge_start, merge_end, pipeline_logger)

            current_time += step
            seg_idx += 1

        analysis_data = AnalysisData(
            meta=stage_a_out.meta,
            timeline=[],
            stem_timelines={},
            notes=list(accumulated_raw),
            notes_before_quantization=list(accumulated_raw),
            beats=stage_a_out.beats if stage_a_out.beats else getattr(stage_a_out.meta, "beats", []),
        )

        stage_b_out = StageBOutput(
            time_grid=np.array([]),
            f0_main=np.array([]),
            f0_layers=[],
            per_detector={},
            meta=stage_a_out.meta,
            diagnostics={"mode": "segmented"},
            precalculated_notes=None
        )

        notes_for_stage_d = copy.deepcopy(list(accumulated_raw))  # ✅ pass raw

    else:
        # ---------------- Single-pass B/C ----------------
        pipeline_logger.log_event(
            "stage_b",
            "detector_selection",
            {
                "detectors": config.stage_b.detectors if hasattr(config, "stage_b") else {},
                "dependencies": PipelineLogger.dependency_snapshot(["torch", "crepe", "demucs"]),
            },
        )
        t_b = time.perf_counter()
        stage_b_out = extract_features(stage_a_out, config=config, pipeline_logger=pipeline_logger)

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

        analysis_data = AnalysisData(
            meta=stage_a_out.meta,
            timeline=[],
            stem_timelines=stage_b_out.stem_timelines,
            precalculated_notes=stage_b_out.precalculated_notes
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
        _ = apply_theory(analysis_data, config=config)

        res_c = validate_invariants(analysis_data.notes, config, analysis_data=analysis_data, logger=pipeline_logger)
        if hasattr(analysis_data, "diagnostics"):
            analysis_data.diagnostics.setdefault("contracts", {})["stage_c"] = res_c
            if stage_a_out and hasattr(stage_a_out, "diagnostics"):
                analysis_data.diagnostics.setdefault("contracts", {})["stage_a"] = stage_a_out.diagnostics.get("contracts")
                if "fallbacks" in stage_a_out.diagnostics:
                    analysis_data.diagnostics.setdefault("fallbacks", []).extend(stage_a_out.diagnostics["fallbacks"])
            if stage_b_out and hasattr(stage_b_out, "diagnostics"):
                analysis_data.diagnostics.setdefault("contracts", {})["stage_b"] = stage_b_out.diagnostics.get("contracts")

        pipeline_logger.record_timing("stage_c", time.perf_counter() - t_c,
                                      metadata={"note_count_raw": len(getattr(analysis_data, "notes_before_quantization", []) or [])})

        notes_for_stage_d = copy.deepcopy(list(getattr(analysis_data, "notes_before_quantization", []) or []))  # ✅
        if not notes_for_stage_d:
            # fallback to whatever Stage C produced
            notes_for_stage_d = copy.deepcopy(list(getattr(analysis_data, "notes", []) or []))

    # ---------------- Stage B/C diagnostics ----------------
    timeline_source = list(stage_b_out.timeline or [])
    if not timeline_source and getattr(stage_b_out, "time_grid", None) is not None:
        tg = getattr(stage_b_out, "time_grid", None) or []
        f0m = getattr(stage_b_out, "f0_main", None) or []
        for t, f0 in zip(tg, f0m):
            f0 = float(f0)
            if f0 > 0.0:
                midi = int(round(69 + 12 * math.log2(f0 / 440.0)))
                timeline_source.append(FramePitch(time=float(t), pitch_hz=f0, midi=midi, confidence=0.0, rms=0.0))
            else:
                timeline_source.append(FramePitch(time=float(t), pitch_hz=0.0, midi=None, confidence=0.0, rms=0.0))

    den = max(1, len(timeline_source))
    voiced_frames = [fp for fp in timeline_source if getattr(fp, "pitch_hz", 0.0) > 0]
    voiced_ratio = len(voiced_frames) / den
    zero_ratio = 1.0 - voiced_ratio
    mean_conf = float(np.mean([getattr(fp, "confidence", 0.0) for fp in timeline_source])) if timeline_source else 0.0

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
        "timeline_frames": int(len(timeline_source)),
    }

    # ✅ Stage C metrics must be RAW (pre-quantization)
    raw_notes_for_metrics = list(getattr(analysis_data, "notes_before_quantization", []) or [])
    if not raw_notes_for_metrics:
        raw_notes_for_metrics = list(notes_for_stage_d or [])

    raw_durations = [max(0.0, float(n.end_sec) - float(n.start_sec)) for n in raw_notes_for_metrics if n.end_sec > n.start_sec]
    short_notes = sum(1 for d in raw_durations if d < 0.08)
    duration_sec = float(getattr(stage_a_out.meta, "duration_sec", 0.0) or 0.0)

    stage_metrics["stage_c"] = {
        "note_count_raw": len(raw_notes_for_metrics),
        "note_count_per_10s_raw": (len(raw_notes_for_metrics) / (duration_sec / 10.0)) if duration_sec else 0.0,
        "median_note_len_ms_raw": float(np.median(raw_durations) * 1000.0) if raw_durations else 0.0,
        "fragmentation_score_raw": float(short_notes / max(1, len(raw_notes_for_metrics))) if raw_notes_for_metrics else 0.0,
    }

    # ---------------- Stage D ----------------
    t_d = time.perf_counter()
    d_out: TranscriptionResult = quantize_and_render(
        notes_for_stage_d,      # ✅ raw notes only
        analysis_data,
        config=config,
        pipeline_logger=pipeline_logger,
    )
    res_d = validate_invariants(d_out, config, logger=pipeline_logger)
    if hasattr(d_out.analysis_data, "diagnostics"):
        d_out.analysis_data.diagnostics.setdefault("contracts", {})["stage_d"] = res_d

    pipeline_logger.record_timing("stage_d", time.perf_counter() - t_d,
                                  metadata={"beats_detected": len(d_out.analysis_data.beats)})

    stage_metrics["stage_d"] = {
        "quantization_mean_shift_ms": 0.0,
        "quantization_p95_shift_ms": 0.0,
        "rendered_notes": len(getattr(d_out.analysis_data, "notes", []) or []),
    }

    return _finalize_and_return(d_out)
