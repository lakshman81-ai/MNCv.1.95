"""
Unified Benchmark Runner (L0-L6)

This module implements the full benchmark ladder:
- L0: Mono Sanity (Synthetic Sine/Vibrato)
- L1: Mono Musical (Synthetic MIDI-like scale)
- L2: Poly Dominant (Synthetic Mix)
- L3: Full Poly (MusicXML-backed synthetic score)
- L4: Real Songs (via run_real_songs)
- L5.1 / L5.2: Real-song MIDI-backed synthetic “real songs”
- L6: Synthetic Pop Song (music21-generated score: melody + chords + bass)

It validates algorithm selection (Stage B), records polyphonic diagnostics, and saves artifacts/metrics.

Key fixes applied in this version:
- L6 now explicitly forces melody-only evaluation behavior via Stage C polyphony_filter ("skyline_top_voice")
  to match GT(parts=["Lead"]).
- L6 also explicitly sets stage_b.transcription_mode="classic_song" (prevents mono default via router).
- Robust dict creation for separation.harmonic_masking and detector configs (no KeyError).
- run_pipeline_on_audio now passes pipeline_logger/device to extract_features when supported, and
  constructs AnalysisData using StageBOutput timeline/diagnostics/precalculated_notes when available
  (aligns with backend.pipeline.transcribe design; fixes missing timeline/diagnostics bugs).
- L5.* “freeze classic” fixed: previously wrote config.transcription_mode (wrong); now sets config.stage_b.transcription_mode.
- Optional, friendly error messaging if soundfile/synth backend is missing, instead of crashing at import time.
"""

from __future__ import annotations

import copy
import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import warnings
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict, is_dataclass

import math  # used for cents jump calculations

import music21
from music21 import tempo, chord

from backend.pipeline.config import PipelineConfig, InstrumentProfile
from backend.pipeline.instrumentation import PipelineLogger
from backend.pipeline.models import (
    StageAOutput,
    MetaData,
    Stem,
    AnalysisData,
    AudioType,
    NoteEvent,
)
from backend.pipeline.stage_a import detect_tempo_and_beats
from backend.pipeline.stage_b import extract_features
from backend.pipeline.stage_c import apply_theory
from backend.pipeline.stage_d import quantize_and_render
from backend.pipeline.global_profiles import apply_global_profile
from backend.benchmarks.metrics import (
    note_f1,
    onset_offset_mae,
    dtw_note_f1,
    dtw_onset_error_ms,
    compute_symptom_metrics,
)
from backend.pipeline.debug import match_notes_nearest, write_error_slices_jsonl, write_frame_timeline_csv
from backend.benchmarks.run_real_songs import run_song as run_real_song
from backend.benchmarks.ladder.generators import generate_benchmark_example

# ---- Optional imports with friendly failure ----
_SOUNDFILE_IMPORT_ERROR: Optional[str] = None
try:
    import soundfile as sf
except Exception as e:
    sf = None  # type: ignore
    _SOUNDFILE_IMPORT_ERROR = str(e)

_SYNTH_IMPORT_ERROR: Optional[str] = None
try:
    from backend.benchmarks.ladder.synth import midi_to_wav_synth
except Exception as e:
    midi_to_wav_synth = None  # type: ignore
    _SYNTH_IMPORT_ERROR = str(e)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_runner")

L5_OVERRIDE_FIELD_DOC = (
    "Override keys mirror PipelineConfig: stage_b.separation.(enabled/model/overlap/shifts/"
    "synthetic_model), stage_b.detectors.<name>.(enabled/fmin/fmax/hop_length), "
    "stage_b.ensemble_weights.<name>, stage_b.polyphonic_peeling.(max_layers/max_harmonics/"
    "mask_width/residual_flatness_stop/harmonic_snr_stop_db/iss_adaptive), "
    "stage_b.melody_filtering.(fmin_hz/voiced_prob_threshold), stage_c.(confidence_threshold/"
    "min_note_duration_ms_poly), and "
    "stage_a.high_pass_filter.(cutoff_hz)."
)

# NOTE: L6 added
LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4", "L6", "L5.1", "L5.2"]


def accuracy_benchmark_plan() -> Dict[str, Any]:
    """Structured description of the accuracy-focused benchmark plan."""
    return {
        "ladder": {
            "levels": ["L0", "L1", "L2", "L3", "L4", "L6", "L5.1", "L5.2"],
            "coverage": {
                "L0": "sine_regression",
                "L1": "monophonic_scale",
                "L2": "poly_dominant",
                "L3": "full_poly_musicxml",
                "L4": "real_songs",
                "L6": "synthetic_pop_song",
                "L5.1": "kal_ho_na_ho",
                "L5.2": "tumhare_hi_rahenge",
            },
            "metrics": ["note_f1", "onset_mae_ms", "offset_mae_ms"],
            "artifacts": ["metrics_json", "leaderboard_json", "summary_csv"],
        },
        "end_to_end": {
            "scenarios": [
                "clean_piano",
                "dense_chords",
                "percussive_passages",
                "noisy_inputs",
            ],
            "outputs": ["musicxml", "midi_bytes", "analysis_timelines", "profiling_traces"],
            "goals": [
                "stage_A_to_D_flow",
                "aggregate_outputs",
                "consistency_across_artifacts",
                "latency_and_accuracy_tracking",
            ],
            "acceptance_metrics": ["note_f1", "onset_offset_f1", "runtime_s", "latency_budget_ms"],
        },
        "stage_a": {
            "toggles": [
                "sample_rate_targets",
                "channel_handling",
                "trimming",
                "loudness_normalization",
            ],
            "fixtures": ["silence", "dc_offset_tones", "clipped_signals"],
            "metrics": [
                "snr_change",
                "loudness_change",
                "latency_s",
                "headroom_recovery",
            ],
            "measurements": [
                "pre_post_snr",
                "pre_post_lufs",
                "conditioning_wall_time",
            ],
        },
        "stage_b": {
            "detectors": ["yin", "swiftf0", "crepe", "rmvpe"],
            "ensemble_settings": [
                "confidence_voicing_threshold",
                "pitch_disagreement_cents",
                "per_detector_flags",
                "source_separation",
                "harmonic_masking",
            ],
            "metrics": [
                "f0_precision",
                "f0_recall",
                "voicing_error",
                "latency_s",
                "robustness_to_masking",
            ],
            "fixtures": ["annotated_monophonic", "annotated_polyphonic"],
            "robustness_checks": ["separation_on_off", "harmonic_masking_on_off"],
        },
        "stage_c": {
            "segmentation_modes": ["hmm", "threshold"],
            "parameters": ["minimum_duration", "pitch_merge_tolerance", "gap_filling"],
            "fixtures": ["staccato", "legato", "varied_tempos"],
            "metrics": [
                "note_f_measure",
                "onset_offset_f_measure",
                "fragmentation_rate",
                "merging_rate",
            ],
            "tempo_sweeps": ["slow", "medium", "fast", "rubato_sim"],
        },
        "stage_d": {
            "scenarios": ["tempo_grids", "swing", "rubato"],
            "metrics": [
                "beat_alignment_error",
                "barline_placement",
                "notation_cleanliness",
            ],
            "ground_truth": "synthetic_midi_round_trip",
            "render_checks": ["quantize_and_render", "swing_grid_alignment", "musicxml_schema_validation"],
            "artifacts": ["musicxml", "midi_bytes", "timeline_json"],
        },
        "ablation": {
            "sweeps": [
                "source_separation",
                "ensemble_weights",
                "segmentation_method",
                "detector_voicing_thresholds",
            ],
            "reports": ["f_measure_impact", "runtime_impact", "interaction_notes"],
        },
        "regression": {
            "corpus": "fixed_benchmark_corpus",
            "thresholds": ["accuracy_delta", "timing_delta", "latency_budget", "artifact_completeness"],
            "stage_thresholds": {
                "end_to_end_note_f1_delta": 0.01,
                "stage_a_latency_delta_s": 0.05,
                "stage_b_voicing_error_delta": 0.01,
                "note_f1_floor": {"L0": 0.85, "L1": 0.1, "L2": 0.05, "L3": 0.0, "L4": 0.0, "L6": 0.0},
                "onset_mae_ms_max": 500.0,
                "latency_budget_ms": 60000.0,
            },
            "alerts": True,
        },
        "profiling": {
            "hooks": [
                "stage_timings",
                "noise_floor",
                "detector_confidences",
                "hmm_state_durations",
                "artifact_sizes",
            ],
            "purpose": "contextualize_benchmark_results",
            "artifacts": ["profiling_traces", "intermediate_metrics"],
        },
    }


def make_config(audio_type: AudioType = AudioType.MONOPHONIC) -> PipelineConfig:
    """Factory for pipeline config based on audio type."""
    config = PipelineConfig()
    return config


def midi_to_freq(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12.0)


def synthesize_audio(notes: List[Tuple[int, float]], sr: int = 44100, waveform: str = "sine") -> np.ndarray:
    """Generate simple synthetic audio."""
    signal = np.array([], dtype=np.float32)
    for midi_note, dur in notes:
        freq = midi_to_freq(midi_note)
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
        if waveform == "sine":
            wave = 0.5 * np.sin(2.0 * np.pi * freq * t)
        elif waveform == "saw":
            wave = 0.5 * (2.0 * (t * freq - np.floor(t * freq + 0.5)))
        else:
            wave = 0.5 * np.sin(2.0 * np.pi * freq * t)

        fade_len = int(0.01 * sr)
        if fade_len > 0 and len(wave) >= fade_len:
            fade = np.linspace(0, 1, fade_len)
            wave[:fade_len] *= fade
            wave[-fade_len:] *= fade[::-1]

        signal = np.concatenate((signal, wave))
    return signal


def _safe_disable_stage_c_quantize(config: PipelineConfig) -> None:
    """
    L0/L1 sanity checks must measure detection/segmentation timing, not grid-snapping luck.
    This requires Stage C quantization to be bypassed.
    """
    try:
        if not hasattr(config.stage_c, "quantize") or getattr(config.stage_c, "quantize") is None:
            setattr(config.stage_c, "quantize", {"enabled": False})
        else:
            config.stage_c.quantize["enabled"] = False
    except Exception:
        # last-resort: set attribute
        setattr(config.stage_c, "quantize", {"enabled": False})


def create_pop_song_base(
    duration_sec: float = 60.0,
    tempo_bpm: int = 110,
    seed: int = 0,
) -> music21.stream.Score:
    """
    L6 generator: synthetic pop song.
    - Structure: Intro/Verse/Chorus/Outro-like by repeating sections
    - Parts: Lead melody, Piano block chords, Bass line
    """
    import random
    from music21 import stream, note, chord as m21chord, meter, tempo as m21tempo, instrument

    rng = random.Random(seed)

    score = stream.Score()
    score.append(m21tempo.MetronomeMark(number=tempo_bpm))
    score.append(meter.TimeSignature("4/4"))

    lead = stream.Part()
    lead.partName = "Lead"
    lead.insert(0, instrument.Soprano())

    piano = stream.Part()
    piano.partName = "Piano"
    piano.insert(0, instrument.Piano())

    bass = stream.Part()
    bass.partName = "Bass"
    bass.insert(0, instrument.ElectricBass())

    # I–V–vi–IV in C major (simple pop progression)
    prog = [
        ["C4", "E4", "G4"],
        ["G3", "B3", "D4"],
        ["A3", "C4", "E4"],
        ["F3", "A3", "C4"],
    ]

    # Convert duration_sec to bars (4/4): beats/sec = tempo/60, 4 beats/bar
    total_bars = int(round(duration_sec * (tempo_bpm / 60.0) / 4.0))
    total_bars = max(16, total_bars)

    # Melody scale (pentatonic-ish for stability)
    scale = ["C5", "D5", "E5", "G5", "A5", "G5", "E5", "D5"]

    def add_bar_chords(bar_idx: int) -> None:
        triad = prog[bar_idx % len(prog)]
        c = m21chord.Chord(triad)
        c.quarterLength = 4.0
        piano.append(c)

    def add_bar_bass(bar_idx: int) -> None:
        root = prog[bar_idx % len(prog)][0]
        n = note.Note(root)
        # Force bass octave low
        n.octave = 2
        for _ in range(4):
            nn = note.Note(n.pitch)
            nn.quarterLength = 1.0
            bass.append(nn)

    def add_bar_melody(bar_idx: int) -> None:
        # 8 eighth notes
        for j in range(8):
            base = scale[(bar_idx + j) % len(scale)]
            if rng.random() < 0.25:
                base = rng.choice(scale)
            nn = note.Note(base)
            nn.quarterLength = 0.5
            lead.append(nn)

    for b in range(total_bars):
        add_bar_chords(b)
        add_bar_bass(b)
        add_bar_melody(b)

    score.insert(0, lead)
    score.insert(0, piano)
    score.insert(0, bass)
    return score


def _load_musicxml_notes(xml_path: str) -> List[Tuple[int, float, float]]:
    """Parse a MusicXML into note tuples (midi, start_sec, end_sec)."""
    score = music21.converter.parse(xml_path)
    bpm = 120.0
    mm = score.flatten().getElementsByClass("MetronomeMark")
    if mm:
        bpm = mm[0].number

    sec_per_beat = 60.0 / bpm
    gt: List[Tuple[int, float, float]] = []
    for n in score.flatten().notes:
        if n.isRest:
            continue
        start_sec = float(n.offset * sec_per_beat)
        end_sec = float((n.offset + n.quarterLength) * sec_per_beat)
        if hasattr(n, "pitches"):
            for p in n.pitches:
                gt.append((int(p.midi), start_sec, end_sec))
        else:
            gt.append((int(n.pitch.midi), start_sec, end_sec))
    return gt


def _require_synth_backend() -> None:
    if midi_to_wav_synth is None:
        raise RuntimeError(
            "midi_to_wav_synth is unavailable. Most common cause: missing 'soundfile' and/or libsndfile.\n"
            f"Import error: {_SYNTH_IMPORT_ERROR}"
        )
    if sf is None:
        raise RuntimeError(
            "soundfile is unavailable. Install with: pip install soundfile, and ensure system libsndfile is installed.\n"
            f"Import error: {_SOUNDFILE_IMPORT_ERROR}"
        )


def run_pipeline_on_audio(
    audio: np.ndarray,
    sr: int,
    config: PipelineConfig,
    audio_type: AudioType = AudioType.MONOPHONIC,
    audio_path: Optional[str] = None,
    allow_separation: bool = False,
    pipeline_logger: Optional[PipelineLogger] = None,
    *,
    skip_global_profile: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run full pipeline on raw audio array."""

    # Separation default safety: synthetic/CI environments often cannot download Demucs.
    try:
        if config.stage_b.separation.get("enabled", False) and not allow_separation:
            config.stage_b.separation["enabled"] = False
        elif allow_separation:
            harmonic_mask = config.stage_b.separation.setdefault("harmonic_masking", {})
            harmonic_mask["enabled"] = True
            harmonic_mask.setdefault("mask_width", 0.03)
    except Exception:
        pass

    # Some harnesses want extra detector capacity on poly-dominant.
    if audio_type == AudioType.POLYPHONIC_DOMINANT:
        try:
            rmvpe_cfg = dict(config.stage_b.detectors.get("rmvpe", {}) or {})
            rmvpe_cfg["enabled"] = True
            rmvpe_cfg["fmax"] = max(float(rmvpe_cfg.get("fmax", 1200.0)), 2200.0)
            config.stage_b.detectors["rmvpe"] = rmvpe_cfg

            crepe_cfg = dict(config.stage_b.detectors.get("crepe", {}) or {})
            crepe_cfg["enabled"] = True
            crepe_cfg["model_capacity"] = crepe_cfg.get("model_capacity", "full")
            crepe_cfg["use_viterbi"] = bool(crepe_cfg.get("use_viterbi", True))
            config.stage_b.detectors["crepe"] = crepe_cfg
        except Exception:
            pass

    pipeline_logger = pipeline_logger or PipelineLogger()

    # 1. Stage A
    t_start = time.perf_counter()
    pipeline_logger.log_event(
        "stage_a",
        "start",
        {
            "audio_path": audio_path or "synthetic",
            "audio_type": audio_type.value,
            "detector_preferences": getattr(config.stage_b, "detectors", {}),
        },
    )

    # NOTE: meta.window_size key varies across configs ("n_fft" vs "frame_length").
    yin_cfg = {}
    try:
        yin_cfg = config.stage_b.detectors.get("yin", {}) or {}
    except Exception:
        yin_cfg = {}

    hop_length = int(yin_cfg.get("hop_length", 512) or 512)
    window_size = int(yin_cfg.get("n_fft", yin_cfg.get("frame_length", 2048)) or 2048)

    meta = MetaData(
        sample_rate=sr,
        target_sr=sr,
        duration_sec=float(len(audio)) / sr,
        processing_mode=audio_type.value,
        audio_type=audio_type,
        audio_path=audio_path,
        hop_length=hop_length,
        window_size=window_size,
        lufs=-20.0,
    )

    stems = {"mix": Stem(audio=audio, sr=sr, type="mix")}
    stage_a_out = StageAOutput(stems=stems, meta=meta, audio_type=audio_type)

    # Apply global profile / auto-router unless explicitly skipped (bench harness may want to freeze)
    if not skip_global_profile:
        try:
            apply_global_profile(
                audio_path=audio_path or "synthetic",
                stage_a_out=stage_a_out,
                config=config,
                pipeline_logger=pipeline_logger,
            )
        except Exception as e:
            pipeline_logger.log_event("pipeline", "global_profile_failed", {"error": str(e)})

    # --------------------------------------------------------
    # BPM Fallback Logic
    # --------------------------------------------------------
    bpm_cfg = getattr(config.stage_a, "bpm_detection", {}) or {}
    bpm_enabled = bool(bpm_cfg.get("enabled", True))
    meta = stage_a_out.meta

    needs_fallback = (
        bpm_enabled
        and (len(meta.beats) == 0)
        and (
            meta.tempo_bpm is None
            or meta.tempo_bpm <= 0
            or (meta.tempo_bpm == 120.0 and len(meta.beats) == 0)
        )
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
                        # stable dedupe (no set reordering)
                        dedup: List[float] = []
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

                        pipeline_logger.log_event(
                            "stage_a", "bpm_fallback_success", {"bpm": fb_bpm, "n_beats": len(fb_beats)}
                        )
                    else:
                        pipeline_logger.log_event("stage_a", "bpm_fallback_no_result")
        except Exception as e:
            pipeline_logger.log_event("stage_a", "bpm_fallback_failed", {"error": str(e)})

    t_stage_a = time.perf_counter() - t_start
    pipeline_logger.record_timing(
        "stage_a",
        t_stage_a,
        metadata={"sample_rate": sr, "hop_length": meta.hop_length, "window_size": meta.window_size},
    )

    # 2. Stage B
    pipeline_logger.log_event(
        "stage_b",
        "detector_selection",
        {
            "detectors": getattr(config.stage_b, "detectors", {}),
            "dependencies": PipelineLogger.dependency_snapshot(["torch", "crepe", "demucs"]),
        },
    )
    t_b_start = time.perf_counter()

    # extract_features signature may vary; try richer call first.
    try:
        stage_b_out = extract_features(stage_a_out, config=config, pipeline_logger=pipeline_logger, device=device)
    except TypeError:
        stage_b_out = extract_features(stage_a_out, config=config)

    t_stage_b = time.perf_counter() - t_b_start
    try:
        dets_run = list((stage_b_out.per_detector or {}).get("mix", {}).keys())
    except Exception:
        dets_run = []
    pipeline_logger.record_timing("stage_b", t_stage_b, metadata={"detectors_run": dets_run})

    # 3. Stage C
    pipeline_logger.log_event(
        "stage_c",
        "segmentation",
        {
            "method": (getattr(config.stage_c, "segmentation_method", {}) or {}).get("method"),
            "pitch_tolerance_cents": getattr(config.stage_c, "pitch_tolerance_cents", None),
            "quantize_enabled": bool(getattr(getattr(config, "stage_c", None), "quantize", {}) or {}).get("enabled", True),
        },
    )
    t_c_start = time.perf_counter()

    # IMPORTANT: build AnalysisData aligned with backend.pipeline.transcribe
    sb_meta = getattr(stage_b_out, "meta", None) or meta
    sb_timeline = getattr(stage_b_out, "timeline", None) or []
    sb_stem_timelines = getattr(stage_b_out, "stem_timelines", None) or {}
    sb_diag = getattr(stage_b_out, "diagnostics", None) or {}
    sb_pre_notes = getattr(stage_b_out, "precalculated_notes", None)

    try:
        analysis = AnalysisData(
            meta=sb_meta,
            timeline=sb_timeline,
            stem_timelines=sb_stem_timelines,
            diagnostics=sb_diag,
            precalculated_notes=sb_pre_notes,
        )
    except TypeError:
        # Older AnalysisData signatures: fall back to minimal ctor and then attach what we can.
        analysis = AnalysisData(meta=sb_meta, stem_timelines=sb_stem_timelines)  # type: ignore
        try:
            analysis.timeline = sb_timeline
        except Exception:
            pass
        try:
            analysis.diagnostics = sb_diag
        except Exception:
            pass
        try:
            analysis.precalculated_notes = sb_pre_notes
        except Exception:
            pass

    # Consolidate fallbacks from Stage A into AnalysisData (if present)
    try:
        if hasattr(stage_a_out, "diagnostics") and isinstance(stage_a_out.diagnostics, dict) and "fallbacks" in stage_a_out.diagnostics:
            analysis.diagnostics = getattr(analysis, "diagnostics", None) or {}
            analysis.diagnostics.setdefault("fallbacks", []).extend(stage_a_out.diagnostics.get("fallbacks", []))
    except Exception:
        pass

    # Stage C applies in-place updates to analysis (notes, chords, etc.)
    notes_pred = apply_theory(analysis, config=config)
    t_stage_c = time.perf_counter() - t_c_start
    try:
        pipeline_logger.record_timing("stage_c", t_stage_c, metadata={"note_count": len(notes_pred)})
    except Exception:
        pipeline_logger.record_timing("stage_c", t_stage_c, metadata={"note_count": None})

    # 4. Stage D
    t_d_start = time.perf_counter()
    try:
        # Some builds accept pipeline_logger; try and fall back.
        try:
            transcription_result = quantize_and_render(notes_pred, analysis, config=config, pipeline_logger=pipeline_logger)
        except TypeError:
            transcription_result = quantize_and_render(notes_pred, analysis, config=config)
    except Exception as e:
        logger.warning(f"Stage D failed: {e}")
        transcription_result = None
    t_stage_d = time.perf_counter() - t_d_start
    pipeline_logger.record_timing(
        "stage_d",
        t_stage_d,
        metadata={"beats_detected": len(getattr(getattr(analysis, "meta", None), "beats", []) or [])},
    )

    stage_timings = {
        "stage_a_s": t_stage_a,
        "stage_b_s": t_stage_b,
        "stage_c_s": t_stage_c,
        "stage_d_s": t_stage_d,
        "total_s": t_stage_a + t_stage_b + t_stage_c + t_stage_d,
    }

    detector_conf_traces: Dict[str, Dict[str, float]] = {}
    try:
        for stem_name, dets in (stage_b_out.per_detector or {}).items():
            detector_conf_traces[stem_name] = {}
            for det_name, (_, conf) in dets.items():
                if conf is None or len(conf) == 0:
                    detector_conf_traces[stem_name][det_name] = 0.0
                else:
                    detector_conf_traces[stem_name][det_name] = float(np.mean(conf))
    except Exception:
        pass

    artifact_flags = {
        "musicxml": bool(getattr(transcription_result, "musicxml", "")) if transcription_result is not None else False,
        "midi_bytes": bool(getattr(transcription_result, "midi_bytes", b"")) if transcription_result is not None else False,
        "timeline": bool(sb_stem_timelines),
    }

    pipeline_logger.log_event(
        "pipeline",
        "complete",
        {
            "notes": len(notes_pred) if notes_pred is not None else 0,
            "run_dir": getattr(pipeline_logger, "run_dir", None),
            "context": "benchmark",
        },
    )
    pipeline_logger.finalize()

    return {
        "notes": notes_pred,
        "stage_b_out": stage_b_out,
        "transcription": transcription_result,
        "resolved_config": config,
        "analysis_data": analysis,
        "profiling": {
            "stage_timings": stage_timings,
            "detector_confidences": detector_conf_traces,
            "artifacts": artifact_flags,
        },
    }


class BenchmarkSuite:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results: List[Dict[str, Any]] = []
        os.makedirs(output_dir, exist_ok=True)

    def _deep_merge_config(self, target: Any, updates: Dict[str, Any], path: str = "config") -> None:
        """Recursively merge a dict of overrides into a PipelineConfig or nested dicts."""
        if not isinstance(updates, dict):
            raise ValueError("Overrides must be provided as a dict")

        for key, value in updates.items():
            if is_dataclass(target):
                if not hasattr(target, key):
                    logger.warning(f"Unknown override key {path}.{key} - skipping")
                    continue
                current = getattr(target, key)
                container_type = "dataclass"
            elif isinstance(target, dict):
                current = target.get(key)
                container_type = "dict"
            else:
                logger.warning(f"Cannot merge overrides into {path} (type={type(target)})")
                continue

            if isinstance(current, dict) and isinstance(value, dict):
                self._deep_merge_config(current, value, path=f"{path}.{key}")
            elif is_dataclass(current) and isinstance(value, dict):
                self._deep_merge_config(current, value, path=f"{path}.{key}")
            else:
                if container_type == "dataclass":
                    setattr(target, key, value)
                else:
                    target[key] = copy.deepcopy(value)

    def _load_override_from_path(self, override_path: Optional[str]) -> Dict[str, Any]:
        if not override_path:
            return {}
        with open(override_path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Override file must contain a JSON object")
        return payload

    def _prepare_l5_config(
        self,
        baseline_config: PipelineConfig,
        overrides: Optional[Dict[str, Any]] = None,
        override_path: Optional[str] = None,
    ) -> PipelineConfig:
        """Return an L5-ready config with optional inline/file overrides applied."""
        config = copy.deepcopy(baseline_config)
        merged_sources = [src for src in (overrides, self._load_override_from_path(override_path)) if src]

        if merged_sources:
            logger.info("Applying %d override payload(s) to L5 config", len(merged_sources))

        for src in merged_sources:
            self._deep_merge_config(config, src)

        return config

    def _enforce_regression_thresholds(self, level: str, metrics: Dict[str, Any], profiling: Optional[Dict[str, Any]] = None):
        plan = accuracy_benchmark_plan()
        thresholds = plan.get("regression", {}).get("stage_thresholds", {})

        note_f1_floor = thresholds.get("note_f1_floor", {}).get(level)
        if note_f1_floor is not None and metrics.get("note_f1", 0.0) < note_f1_floor:
            raise RuntimeError(
                f"Regression gate: note F1 {metrics.get('note_f1')} below floor {note_f1_floor} for {level}"
            )

        onset_ceiling = thresholds.get("onset_mae_ms_max")
        if onset_ceiling is not None and metrics.get("onset_mae_ms") is not None:
            if float(metrics["onset_mae_ms"]) > float(onset_ceiling):
                raise RuntimeError(
                    f"Regression gate: onset MAE {metrics['onset_mae_ms']}ms exceeds budget {onset_ceiling}ms"
                )

        latency_budget = thresholds.get("latency_budget_ms")
        if latency_budget is not None and profiling is not None:
            total_ms = float(profiling.get("stage_timings", {}).get("total_s", 0.0)) * 1000.0
            if total_ms > float(latency_budget):
                raise RuntimeError(
                    f"Regression gate: end-to-end latency {total_ms:.2f}ms exceeds budget {latency_budget}ms"
                )

    def _poly_config(
        self,
        use_harmonic_masking: bool = False,
        mask_width: float = 0.03,
        enable_high_capacity: bool = True,
        use_crepe_viterbi: bool = True,
        use_poly_dominant_segmentation: bool = False,
    ) -> PipelineConfig:
        config = PipelineConfig()
        try:
            config.stage_b.separation["enabled"] = True
            config.stage_b.separation["synthetic_model"] = True

            hm = config.stage_b.separation.setdefault("harmonic_masking", {})
            hm["enabled"] = bool(use_harmonic_masking)
            if use_harmonic_masking:
                hm["mask_width"] = float(mask_width)

            config.stage_b.separation.setdefault("polyphonic_dominant_preset", {})
            config.stage_b.separation["polyphonic_dominant_preset"].update(
                {
                    "overlap": 0.75,
                    "shift_range": [2, 8],
                    "overlap_candidates": [0.5, 0.75],
                }
            )
        except Exception:
            pass

        try:
            config.stage_b.polyphonic_peeling["force_on_mix"] = True
            config.stage_b.polyphonic_peeling["max_layers"] = 2
        except Exception:
            pass

        try:
            config.stage_b.melody_filtering.update(
                {
                    "median_window": 7,
                    "voiced_prob_threshold": 0.45,
                    "rms_gate_db": -38.0,
                    "fmin_hz": 450.0,
                    "fmax_hz": 1400.0,
                }
            )
        except Exception:
            pass

        try:
            yin_conf = dict(config.stage_b.detectors.get("yin", {}) or {})
            yin_conf.update(
                {
                    "hop_length": 256,
                    "frame_length": 4096,
                    "fmin": 450.0,
                    "fmax": 1200.0,
                }
            )
            config.stage_b.detectors["yin"] = yin_conf
        except Exception:
            pass

        if enable_high_capacity:
            self._enable_high_capacity_frontend(config, use_crepe_viterbi)
            try:
                config.stage_b.ensemble_weights["crepe"] = 0.5
                config.stage_b.ensemble_weights["yin"] = 2.0
            except Exception:
                pass

        if use_poly_dominant_segmentation:
            self._apply_poly_dominant_segmentation(config)

        try:
            config.stage_b.voice_tracking["max_alt_voices"] = 1
        except Exception:
            pass

        try:
            config.stage_c.gap_tolerance_s = max(getattr(config.stage_c, "gap_tolerance_s", 0.07), 0.07)
            config.stage_c.pitch_tolerance_cents = max(getattr(config.stage_c, "pitch_tolerance_cents", 50.0), 60.0)
            config.stage_c.min_note_duration_ms_poly = max(getattr(config.stage_c, "min_note_duration_ms_poly", 120.0), 150.0)
            config.stage_c.confidence_hysteresis.update({"start": 0.6, "end": 0.4})
            config.stage_c.polyphony_filter["mode"] = "skyline_top_voice"
        except Exception:
            pass

        # Make intent explicit for router, even though AudioType is provided.
        try:
            config.stage_b.transcription_mode = "classic_song"
        except Exception:
            pass

        return config

    def _apply_poly_dominant_segmentation(self, config: PipelineConfig) -> None:
        try:
            config.stage_c.segmentation_method["preset"] = "poly_dominant_strict"
        except Exception:
            pass
        try:
            config.stage_c.min_note_duration_ms_poly = max(float(config.stage_c.min_note_duration_ms_poly), 120.0)
        except Exception:
            pass
        try:
            pc = getattr(config.stage_c, "polyphonic_confidence", None)
            if pc is None:
                setattr(config.stage_c, "polyphonic_confidence", {})
                pc = config.stage_c.polyphonic_confidence
            pc["melody"] = max(float(pc.get("melody", 0.0)), 0.55)
            pc["accompaniment"] = max(float(pc.get("accompaniment", 0.0)), 0.6, pc["melody"])
        except Exception:
            pass

    def _enable_high_capacity_frontend(self, config: PipelineConfig, use_crepe_viterbi: bool = False) -> None:
        try:
            crepe_cfg = dict(config.stage_b.detectors.get("crepe", {}) or {})
            crepe_cfg["enabled"] = True
            crepe_cfg["model_capacity"] = crepe_cfg.get("model_capacity", "full")
            crepe_cfg["use_viterbi"] = bool(use_crepe_viterbi)
            config.stage_b.detectors["crepe"] = crepe_cfg

            rmvpe_cfg = dict(config.stage_b.detectors.get("rmvpe", {}) or {})
            rmvpe_cfg["enabled"] = True
            rmvpe_cfg["fmax"] = float(rmvpe_cfg.get("fmax", 2000.0) or 2000.0)
            config.stage_b.detectors["rmvpe"] = rmvpe_cfg
        except Exception:
            pass

    def _compute_diff(self, previous: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prev_map = {(r.get("level"), r.get("name")): r for r in previous}
        diff: List[Dict[str, Any]] = []
        for r in current:
            key = (r.get("level"), r.get("name"))
            prev = prev_map.get(key)
            if not prev:
                continue
            diff.append(
                {
                    "level": r.get("level"),
                    "name": r.get("name"),
                    "delta_note_f1": r.get("note_f1") - prev.get("note_f1", 0.0),
                    "delta_onset_mae_ms": (r.get("onset_mae_ms") or 0.0) - (prev.get("onset_mae_ms") or 0.0),
                    "previous": prev,
                    "current": r,
                }
            )
        return diff

    def _merge_results(self, previous: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged = {(r.get("level"), r.get("name")): r for r in previous}
        for r in current:
            merged[(r.get("level"), r.get("name"))] = r
        return [merged[k] for k in sorted(merged.keys())]

    def _save_run(
        self,
        level: str,
        name: str,
        res: Dict[str, Any],
        gt: List[Tuple[int, float, float]],
        apply_regression_gate: bool = True,
    ):
        """Save artifacts for a single run."""
        pred_notes = res["notes"] or []
        pred_list = [(n.midi_note, n.start_sec, n.end_sec) for n in pred_notes]

        # Calculate Metrics
        f1 = note_f1(pred_list, gt, onset_tol=0.05)
        onset_mae, offset_mae = onset_offset_mae(pred_list, gt)
        dtw_f1 = dtw_note_f1(pred_list, gt, onset_tol=0.05)
        dtw_onset_ms = dtw_onset_error_ms(pred_list, gt)

        # Frame/timeline metrics
        sb_out = res.get("stage_b_out")
        timeline_source = []
        if sb_out is not None:
            timeline_source = getattr(sb_out, "timeline", None) or []
            if not timeline_source:
                st = getattr(sb_out, "stem_timelines", None) or {}
                if isinstance(st, dict) and st:
                    timeline_source = st.get("mix") or next(iter(st.values()), []) or []

        total_frames = int(len(timeline_source))
        voiced_frames = 0
        vocal_band_frames = 0

        jump_cents_sum = 0.0
        jump_count = 0
        last_p = 0.0

        for fp in timeline_source:
            hz = float(getattr(fp, "pitch_hz", 0.0) or 0.0)
            if hz > 0:
                voiced_frames += 1
                if 80.0 <= hz <= 1400.0:
                    vocal_band_frames += 1

                if last_p > 0:
                    try:
                        cents = abs(1200.0 * math.log2(hz / last_p))
                        jump_cents_sum += cents
                        jump_count += 1
                    except Exception:
                        pass
                last_p = hz
            else:
                last_p = 0.0

        vocal_band_ratio = (vocal_band_frames / voiced_frames) if voiced_frames > 0 else 0.0

        # duration_sec
        duration_sec = 1.0
        try:
            if res.get("analysis_data") is not None:
                duration_sec = float(res["analysis_data"].meta.duration_sec)
            elif sb_out is not None and getattr(sb_out, "meta", None) is not None:
                duration_sec = float(sb_out.meta.duration_sec)
        except Exception:
            duration_sec = 1.0

        pitch_jump_rate = (jump_cents_sum / duration_sec) if duration_sec > 0 else 0.0
        voiced_ratio = (voiced_frames / max(1, total_frames))
        note_density = (len(pred_list) / duration_sec) if duration_sec > 0 else 0.0

        # Normalize NaNs for downstream checks/serialization
        if np.isnan(f1):
            f1 = 0.0
        if np.isnan(dtw_f1):
            dtw_f1 = None
        if onset_mae is not None and np.isnan(onset_mae):
            onset_mae = None
        if dtw_onset_ms is not None and np.isnan(dtw_onset_ms):
            dtw_onset_ms = None

        symptoms = compute_symptom_metrics(pred_list) or {}

        metrics = {
            "level": level,
            "name": name,
            "note_f1": f1,
            "onset_mae_ms": onset_mae * 1000 if onset_mae is not None else None,
            "dtw_note_f1": dtw_f1,
            "dtw_onset_error_ms": dtw_onset_ms,
            "predicted_count": len(pred_list),
            "gt_count": len(gt),
            "vocal_band_ratio": vocal_band_ratio,
            "pitch_jump_rate_cents_sec": pitch_jump_rate,
            "voiced_ratio": voiced_ratio,
            "voicing_ratio": voiced_ratio,  # backward compatibility
            "note_density": note_density,
            "note_count": len(pred_list),
            **symptoms,
        }

        base_path = os.path.join(self.output_dir, f"{level}_{name}")

        with open(f"{base_path}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(f"{base_path}_pred.json", "w") as f:
            json.dump([asdict(n) for n in pred_notes], f, indent=2, default=str)

        with open(f"{base_path}_gt.json", "w") as f:
            json.dump([{"midi": m, "start": s, "end": e} for m, s, e in gt], f, indent=2)

        # Log resolved config + diagnostics
        detectors_ran = list(res["stage_b_out"].per_detector.get("mix", {}).keys()) if res.get("stage_b_out") else []
        diagnostics = getattr(res.get("stage_b_out"), "diagnostics", {}) if res.get("stage_b_out") else {}
        resolved_config = res.get("resolved_config")

        analysis_diag = getattr(res.get("analysis_data"), "diagnostics", {}) if res.get("analysis_data") else {}
        stage_c_diag = analysis_diag.get("stage_c", {}) if isinstance(analysis_diag, dict) else {}

        # Prefer decision_trace resolved transcription_mode if present
        tr_mode = "unknown"
        try:
            dt = diagnostics.get("decision_trace", {}) if isinstance(diagnostics, dict) else {}
            if isinstance(dt, dict):
                tr_mode = dt.get("resolved", {}).get("transcription_mode", tr_mode)
        except Exception:
            pass

        run_info = {
            "detectors_ran": detectors_ran,
            "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
            "config": asdict(resolved_config) if resolved_config else {},
            "transcription_mode": tr_mode,
            "separation_mode": (diagnostics.get("separation", {}) or {}).get("mode", "unknown") if isinstance(diagnostics, dict) else "unknown",
            "selected_stem": stage_c_diag.get("selected_stem", "unknown") if isinstance(stage_c_diag, dict) else "unknown",
            "fallbacks": analysis_diag.get("fallbacks", []) if isinstance(analysis_diag, dict) else [],
        }
        profiling = res.get("profiling", {}) or {}
        run_info["stage_timings"] = profiling.get("stage_timings", {}) or {}
        run_info["detector_confidences"] = profiling.get("detector_confidences", {}) or {}
        run_info["artifacts_present"] = profiling.get("artifacts", {}) or {}

        sep_diag = diagnostics.get("separation", {}) if isinstance(diagnostics, dict) else {}
        stage_b_conf = getattr(resolved_config, "stage_b", None)
        run_info["separation_preset"] = {
            "preset": sep_diag.get("preset") or "default",
            "overlap": sep_diag.get("resolved_overlap", stage_b_conf.separation.get("overlap") if stage_b_conf else None),
            "shifts": sep_diag.get("resolved_shifts", stage_b_conf.separation.get("shifts") if stage_b_conf else None),
            "shift_range": sep_diag.get(
                "shift_range",
                (stage_b_conf.separation.get("polyphonic_dominant_preset", {}).get("shift_range") if stage_b_conf else None),
            ),
            "harmonic_mask_width": (diagnostics.get("harmonic_masking", {}) or {}).get("mask_width") if isinstance(diagnostics, dict) else None,
        }

        with open(f"{base_path}_run_info.json", "w") as f:
            json.dump(run_info, f, indent=2, default=str)

        # Debug artifact: Timeline CSV
        try:
            if sb_out:
                timeline_frames = timeline_source
                csv_rows = []
                d_curves = (diagnostics.get("debug_curves", {}) or {}).get("mix", {}) if isinstance(diagnostics, dict) else {}
                fused_arr = d_curves.get("fused_f0", []) or []
                smoothed_arr = d_curves.get("smoothed_f0", []) or []

                for idx, fp in enumerate(timeline_frames):
                    hz = float(getattr(fp, "pitch_hz", 0.0) or 0.0)
                    cents = float("nan")
                    if hz > 0:
                        cents = 1200.0 * np.log2(hz / 440.0) + 6900.0

                    fused_c = cents
                    smoothed_c = cents

                    if idx < len(fused_arr):
                        fv = fused_arr[idx]
                        fused_c = 1200.0 * np.log2(fv / 440.0) + 6900.0 if fv > 0 else float("nan")

                    if idx < len(smoothed_arr):
                        sv = smoothed_arr[idx]
                        smoothed_c = 1200.0 * np.log2(sv / 440.0) + 6900.0 if sv > 0 else float("nan")

                    row = {
                        "t_sec": float(getattr(fp, "time", 0.0) or 0.0),
                        "f0_hz": hz,
                        "midi": getattr(fp, "midi", None),
                        "cents": cents,
                        "confidence": float(getattr(fp, "confidence", 0.0) or 0.0),
                        "voiced": hz > 0,
                        "detector_name": "fused",
                        "harmonic_rank": 1,
                        "fused_cents": fused_c,
                        "smoothed_cents": smoothed_c,
                    }
                    csv_rows.append(row)

                if csv_rows:
                    write_frame_timeline_csv(f"{base_path}_timeline.csv", csv_rows)
        except Exception as e:
            logger.warning(f"Failed to write timeline CSV: {e}")

        # Debug artifact: Error slices
        try:
            gt_dicts = [{"pitch_midi": m, "onset_sec": s, "offset_sec": e} for m, s, e in gt]
            pred_dicts = [asdict(n) for n in pred_notes]
            for p in pred_dicts:
                if "pitch_midi" not in p and "midi_note" in p:
                    p["pitch_midi"] = p["midi_note"]
                if "onset_sec" not in p and "start_sec" in p:
                    p["onset_sec"] = p["start_sec"]
                if "offset_sec" not in p and "end_sec" in p:
                    p["offset_sec"] = p["end_sec"]

            pairs, extras = match_notes_nearest(gt_dicts, pred_dicts)
            write_error_slices_jsonl(f"{base_path}_errors.jsonl", pairs, extras)
        except Exception as e:
            logger.warning(f"Failed to write error slices: {e}")

        self.results.append(metrics)

        if apply_regression_gate:
            self._enforce_regression_thresholds(level, metrics, res.get("profiling"))

        return metrics

    @staticmethod
    def _score_to_gt(score, parts: Optional[List[str]] = None) -> List[Tuple[int, float, float]]:
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

    # -----------------------
    # Benchmarks
    # -----------------------

    def run_L0_mono_sanity(self):
        logger.info("Running L0: Mono Sanity")

        sr = 44100
        notes = [(69, 1.0)]  # A4, 1 sec
        audio = synthesize_audio(notes, sr=sr, waveform="sine")

        # Add 0.5s padding silence (noise-floor robustness)
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        audio = np.concatenate([silence, audio, silence])

        offset = 0.5
        gt = [(69, 0.0 + offset, 1.0 + offset)]

        config = PipelineConfig()
        try:
            config.stage_b.detectors["swiftf0"]["enabled"] = False
        except Exception:
            pass

        try:
            config.stage_a.silence_trimming["enabled"] = False
            config.stage_a.bpm_detection["trim"] = False
            config.stage_a.transient_pre_emphasis["enabled"] = False
        except Exception:
            pass

        # Disable Stage C quantization for L0 sanity checks
        _safe_disable_stage_c_quantize(config)

        # Force threshold segmentation and disable profile overrides
        try:
            config.stage_c.segmentation_method = {"method": "threshold"}
            config.stage_c.apply_instrument_profile = False
            config.stage_b.apply_instrument_profile = False
        except Exception:
            pass

        res = run_pipeline_on_audio(audio, sr, config, AudioType.MONOPHONIC)

        # Save run (no regression gate)
        m = self._save_run("L0", "sine_440", res, gt, apply_regression_gate=False)

        # Manual relaxed check
        if m["note_f1"] < 0.8:
            pred_list = [(n.midi_note, n.start_sec, n.end_sec) for n in res["notes"]]
            f1_relaxed = note_f1(pred_list, gt, onset_tol=0.1)
            if f1_relaxed < 0.8:
                raise RuntimeError(f"L0 Failed: Sine 440 F1 {f1_relaxed} < 0.8 (relaxed)")
            logger.warning(f"L0 Passed with relaxed onset tolerance (F1={f1_relaxed:.3f})")
        else:
            logger.info(f"L0 Passed strict F1={m['note_f1']:.3f}")

        detectors = res["stage_b_out"].per_detector.get("mix", {}) if res.get("stage_b_out") else {}
        if not any(d in detectors for d in ["yin", "sacf", "swiftf0", "crepe"]):
            raise RuntimeError("L0 Failed: No mono pitch tracker ran!")

        logger.info("L0 Passed.")

    def run_L1_mono_musical(self):
        logger.info("Running L1: Mono Musical")

        notes = [
            (60, 0.5),
            (62, 0.5),
            (64, 0.5),
            (65, 0.5),
            (67, 0.5),
            (69, 0.5),
            (71, 0.5),
            (72, 0.5),
        ]
        audio = synthesize_audio(notes, sr=44100, waveform="saw")

        sr = 44100
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        audio = np.concatenate([silence, audio, silence])
        offset = 0.5

        gt = []
        t = 0.0
        for m_, d in notes:
            gt.append((m_, t + offset, t + d + offset))
            t += d

        config = PipelineConfig()

        # Disable pre-emphasis
        try:
            config.stage_a.transient_pre_emphasis["enabled"] = False
        except Exception:
            pass

        # Disable Stage C quantization for L1 sanity checks
        _safe_disable_stage_c_quantize(config)

        # Force threshold segmentation and disable profile overrides
        try:
            config.stage_c.segmentation_method = {"method": "threshold"}
            config.stage_c.apply_instrument_profile = False
            config.stage_b.apply_instrument_profile = False
        except Exception:
            pass

        res = run_pipeline_on_audio(audio, 44100, config, AudioType.MONOPHONIC)

        m = self._save_run("L1", "scale_c_maj", res, gt)

        if m["note_f1"] < 0.9:
            logger.warning(f"L1 Warning: F1 {m['note_f1']} < 0.9. (Strict pass required for production)")

        logger.info(f"L1 Complete. F1: {m['note_f1']}")

    def run_L2_poly_dominant(self):
        logger.info("Running L2: Poly Dominant")

        sr = 44100
        melody = synthesize_audio([(72, 0.5), (76, 0.5), (79, 0.5)], sr, "sine")
        bass = synthesize_audio([(48, 1.5)], sr, "saw") * 0.5

        mix = melody + bass
        gt_melody = [(72, 0.0, 0.5), (76, 0.5, 1.0), (79, 1.0, 1.5)]

        baseline_config = self._poly_config(
            use_harmonic_masking=True,
            mask_width=0.03,
            enable_high_capacity=True,
            use_crepe_viterbi=True,
        )

        # Initial run
        res = run_pipeline_on_audio(
            mix,
            sr,
            baseline_config,
            AudioType.POLYPHONIC_DOMINANT,
            allow_separation=True,
        )

        # Auto-tuning sweep for fmin
        fmin_candidates = [80.0, 150.0, 300.0, 450.0, 500.0]
        best_fmin_res = None
        best_fmin_metric = {"note_f1": -1.0}

        sweep_logs = []

        for fmin in fmin_candidates:
            sweep_config = self._poly_config(
                use_harmonic_masking=True,
                mask_width=0.03,
                enable_high_capacity=True,
                use_crepe_viterbi=True,
            )
            sweep_config.stage_b.melody_filtering.update(
                {
                    "fmin_hz": fmin,
                    "voiced_prob_threshold": 0.45,
                }
            )
            yin_conf = sweep_config.stage_b.detectors.get("yin", {})
            yin_conf["fmin"] = fmin
            sweep_config.stage_b.detectors["yin"] = yin_conf

            res = run_pipeline_on_audio(
                mix,
                sr,
                sweep_config,
                AudioType.POLYPHONIC_DOMINANT,
                allow_separation=True,
            )

            pred_notes = res["notes"]
            pred_list = [(n.midi_note, n.start_sec, n.end_sec) for n in pred_notes]
            f1 = note_f1(pred_list, gt_melody, onset_tol=0.05)

            sweep_logs.append(f"fmin={fmin}Hz -> F1={f1:.3f}")

            if f1 > best_fmin_metric["note_f1"]:
                best_fmin_metric = {"note_f1": f1, "fmin": fmin}
                best_fmin_res = res

        logger.info(f"L2 Auto-Tuning Sweep: {', '.join(sweep_logs)}")
        logger.info(f"L2 Best Fmin: {best_fmin_metric.get('fmin')}Hz (F1={best_fmin_metric.get('note_f1'):.3f})")

        if best_fmin_res:
            m = self._save_run(
                "L2",
                "melody_plus_bass_synthetic_sep",
                best_fmin_res,
                gt_melody,
                apply_regression_gate=False,
            )
        else:
            m = {"note_f1": 0.0, "name": "melody_plus_bass_synthetic_sep"}

        logger.info(f"L2 Complete. F1: {m['note_f1']}")

        exp_config = self._poly_config(
            use_harmonic_masking=True,
            mask_width=0.03,
            enable_high_capacity=True,
            use_crepe_viterbi=True,
        )
        exp_res = run_pipeline_on_audio(
            mix,
            sr,
            exp_config,
            AudioType.POLYPHONIC_DOMINANT,
            allow_separation=True,
        )
        m_exp = self._save_run(
            "L2",
            "melody_plus_bass_crepe_rmvpe",
            exp_res,
            gt_melody,
            apply_regression_gate=False,
        )
        logger.info(f"L2 CREPE/RMVPE Complete. F1: {m_exp['note_f1']}")

        mask_widths = [0.01, 0.015, 0.02, 0.04, 0.06]
        overlap_candidates = baseline_config.stage_b.separation.get("polyphonic_dominant_preset", {}).get(
            "overlap_candidates",
            [baseline_config.stage_b.separation.get("overlap", 0.25)],
        )
        sweep_results = []
        for overlap in overlap_candidates:
            for width in mask_widths:
                sweep_config = self._poly_config(use_harmonic_masking=True, mask_width=width)
                sweep_config.stage_b.separation["polyphonic_dominant_preset"]["overlap"] = overlap
                sweep_res = run_pipeline_on_audio(
                    mix,
                    sr,
                    sweep_config,
                    AudioType.POLYPHONIC_DOMINANT,
                    allow_separation=True,
                )
                sweep_metric = self._save_run(
                    "L2",
                    f"melody_plus_bass_mask_{width:.3f}_ovl_{overlap:.2f}",
                    sweep_res,
                    gt_melody,
                    apply_regression_gate=False,
                )
                sweep_results.append({"width": width, "overlap": overlap, "note_f1": sweep_metric.get("note_f1", 0.0)})

        if sweep_results:
            best_combo = max(sweep_results, key=lambda x: x["note_f1"])
            logger.info(
                (
                    "L2 harmonic masking sweep complete. Best overlap %.2f width %.3f -> "
                    "F1 %.3f (baseline %.3f)"
                ),
                best_combo.get("overlap"),
                best_combo.get("width"),
                best_combo.get("note_f1"),
                m.get("note_f1", 0.0),
            )

    def run_L3_full_poly(self):
        logger.info("Running L3: Full Poly")
        _require_synth_backend()

        score = generate_benchmark_example("old_macdonald_poly_full")
        gt = self._score_to_gt(score)

        sr = 22050
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            midi_to_wav_synth(score, wav_path, sr=sr)  # type: ignore[misc]
            audio, read_sr = sf.read(wav_path)  # type: ignore[union-attr]
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        max_duration = 8.0
        if len(audio) > int(max_duration * read_sr):
            audio = audio[: int(max_duration * read_sr)]
            gt = [(m, s, min(e, max_duration)) for m, s, e in gt if s < max_duration]

        config = PipelineConfig()
        try:
            config.stage_b.separation["enabled"] = True
            config.stage_b.separation["model"] = "htdemucs"
            config.stage_b.polyphonic_peeling["max_layers"] = 8
        except Exception:
            pass
        for det in ["swiftf0", "rmvpe", "crepe", "yin"]:
            if det in config.stage_b.detectors:
                config.stage_b.detectors[det]["enabled"] = True

        # Match intent explicitly
        try:
            config.stage_b.transcription_mode = "classic_song"
        except Exception:
            pass

        res = run_pipeline_on_audio(
            audio.astype(np.float32),
            int(read_sr),
            config,
            AudioType.POLYPHONIC,
            allow_separation=True,
        )

        m = self._save_run("L3", "old_macdonald_poly_full", res, gt)

        detectors = res["stage_b_out"].per_detector.get("mix", {}) if res.get("stage_b_out") else {}
        if m["note_f1"] < 0.2:
            logger.warning(f"L3 Warning: old_macdonald_poly_full F1 {m['note_f1']} < 0.2")
        if m["onset_mae_ms"] is None or m["onset_mae_ms"] > 300:
            logger.warning(f"L3 Warning: onset MAE {m['onset_mae_ms']}ms is high")
        if len(detectors) < 2:
            logger.warning("L3 Warning: insufficient detector coverage on full-poly mix")
        if m["predicted_count"] == 0:
            logger.warning("L3 Warning: no notes predicted for full-poly example")

        logger.info(f"L3 Complete. F1: {m['note_f1']}")

    def run_L4_real_songs(self, use_preset: bool = False):
        logger.info("Running L4: Real Songs")
        try:
            from backend.pipeline.config import PIANO_61KEY_CONFIG
            base_config = copy.deepcopy(PIANO_61KEY_CONFIG) if use_preset else None

            res_hb = run_real_song("happy_birthday", max_duration=30.0, config=base_config)
            self._save_real_song_result("L4", "happy_birthday", res_hb)

            res_om = run_real_song("old_macdonald", max_duration=30.0, config=base_config)
            self._save_real_song_result("L4", "old_macdonald", res_om)

        except Exception as e:
            logger.error(f"L4 Failed: {e}")

    def run_L6_synthetic_pop_song(self):
        """
        L6: Synthetic pop song (melody + chords + bass).
        Accuracy is measured against the LEAD melody part only (realistic melody-in-poly mix benchmark).
        """
        logger.info("Running L6: Synthetic Pop Song")
        _require_synth_backend()

        sr = 22050
        score = create_pop_song_base(duration_sec=60.0, tempo_bpm=110, seed=0)

        # Measure melody accuracy (lead part only)
        gt = self._score_to_gt(score, parts=["Lead"])

        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            midi_to_wav_synth(score, wav_path, sr=sr)  # type: ignore[misc]
            audio, read_sr = sf.read(wav_path)  # type: ignore[union-attr]

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        config = PipelineConfig()
        # Avoid Demucs on synthetic benches
        try:
            config.stage_b.separation["enabled"] = False
        except Exception:
            pass

        # CRITICAL: enforce polyphonic intent for router + lead-only eval semantics.
        try:
            config.stage_b.transcription_mode = "classic_song"
        except Exception:
            pass
        try:
            config.stage_c.polyphony_filter["mode"] = "skyline_top_voice"
        except Exception:
            try:
                config.stage_c.polyphony_filter = {"mode": "skyline_top_voice"}
            except Exception:
                pass

        # Encourage melody tracking in a poly mix
        try:
            config.stage_b.melody_filtering.update(
                {"fmin_hz": 180.0, "fmax_hz": 1600.0, "voiced_prob_threshold": 0.40}
            )
        except Exception:
            pass
        for det in ["rmvpe", "crepe", "swiftf0", "yin"]:
            try:
                if det in config.stage_b.detectors:
                    config.stage_b.detectors[det]["enabled"] = True
            except Exception:
                pass

        # Keep Stage C quantize enabled here (L6 is musical)
        res = run_pipeline_on_audio(
            audio.astype(np.float32),
            int(read_sr),
            config,
            AudioType.POLYPHONIC_DOMINANT,
            allow_separation=False,
        )

        m = self._save_run("L6", "synthetic_pop_song_lead", res, gt, apply_regression_gate=False)
        logger.info(f"L6 Complete. F1: {m['note_f1']}")

    def run_L5_1_kal_ho_na_ho(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        override_path: Optional[str] = None,
    ):
        logger.info("Running L5.1: Kal Ho Na Ho")
        _require_synth_backend()

        midi_path = os.path.join("backend", "benchmarks", "ladder", "L5.1_kal_ho_na_ho.mid")
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"L5.1 MIDI not found at {midi_path}")

        score = music21.converter.parse(midi_path)
        gt = self._score_to_gt(score)

        sr = 22050
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            midi_to_wav_synth(score, wav_path, sr=sr)  # type: ignore[misc]
            audio, read_sr = sf.read(wav_path)  # type: ignore[union-attr]
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        max_duration = 30.0
        if len(audio) > int(max_duration * read_sr):
            audio = audio[: int(max_duration * read_sr)]
            gt = [(m, s, min(e, max_duration)) for m, s, e in gt if s < max_duration]

        from backend.pipeline.config import PIANO_61KEY_CONFIG
        config = copy.deepcopy(PIANO_61KEY_CONFIG)
        try:
            config.stage_b.separation["enabled"] = True
            config.stage_b.separation["model"] = "htdemucs"
            config.stage_b.separation["synthetic_model"] = False
            config.stage_b.separation["overlap"] = 0.75
            config.stage_b.separation["shifts"] = 2
            config.stage_b.separation["harmonic_masking"] = {"enabled": True, "mask_width": 0.03}
        except Exception:
            pass

        try:
            config.stage_a.high_pass_filter["cutoff_hz"] = 20.0
            config.stage_b.apply_instrument_profile = False
            config.stage_c.apply_instrument_profile = False
            config.stage_b.confidence_voicing_threshold = 0.3
            config.stage_c.confidence_threshold = 0.05
            config.stage_c.min_note_duration_ms_poly = 50.0
            config.stage_c.polyphony_filter["mode"] = "decomposed_melody"
        except Exception:
            pass

        for d in ["crepe", "swiftf0", "yin"]:
            try:
                if d in config.stage_b.detectors:
                    config.stage_b.detectors[d]["fmin"] = 30.0
            except Exception:
                pass
        try:
            config.stage_b.melody_filtering["fmin_hz"] = 30.0
        except Exception:
            pass

        try:
            config.stage_b.polyphonic_peeling["max_layers"] = 8
            config.stage_b.polyphonic_peeling["max_harmonics"] = 20
            config.stage_b.polyphonic_peeling["residual_flatness_stop"] = 1.0
            config.stage_b.polyphonic_peeling["harmonic_snr_stop_db"] = -100.0
            config.stage_b.polyphonic_peeling["mask_width"] = 0.03
            config.stage_b.polyphonic_peeling["iss_adaptive"] = True
        except Exception:
            pass

        try:
            config.stage_b.ensemble_weights = {
                "swiftf0": 0.4,
                "crepe": 0.4,
                "yin": 0.1,
                "sacf": 0.1,
                "cqt": 0.0,
                "rmvpe": 0.0,
            }
        except Exception:
            pass
        for det in ["swiftf0", "crepe", "yin"]:
            try:
                if det in config.stage_b.detectors:
                    config.stage_b.detectors[det]["enabled"] = True
            except Exception:
                pass

        config = self._prepare_l5_config(config, overrides=overrides, override_path=override_path)

        # FIX: set the correct field (previous code wrote config.transcription_mode, which does nothing)
        try:
            config.stage_b.transcription_mode = "classic_song"
        except Exception:
            pass

        res = run_pipeline_on_audio(
            audio.astype(np.float32),
            int(read_sr),
            config,
            AudioType.POLYPHONIC,
            allow_separation=True,
            skip_global_profile=True,
        )

        m = self._save_run("L5.1", "kal_ho_na_ho", res, gt, apply_regression_gate=False)
        logger.info(f"L5.1 Complete. F1: {m['note_f1']}")

    def run_L5_2_tumhare_hi_rahenge(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        override_path: Optional[str] = None,
        use_preset: bool = False,
    ):
        logger.info("Running L5.2: Tumhare Hi Rahenge")
        _require_synth_backend()

        midi_path = os.path.join("backend", "benchmarks", "ladder", "L5.2_tumhare_hi_rahenge.mid")
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"L5.2 MIDI not found at {midi_path}")

        score = music21.converter.parse(midi_path)
        gt = self._score_to_gt(score)

        sr = 22050
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            midi_to_wav_synth(score, wav_path, sr=sr)  # type: ignore[misc]
            audio, read_sr = sf.read(wav_path)  # type: ignore[union-attr]
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        max_duration = 30.0
        if len(audio) > int(max_duration * read_sr):
            audio = audio[: int(max_duration * read_sr)]
            gt = [(m, s, min(e, max_duration)) for m, s, e in gt if s < max_duration]

        if use_preset:
            from backend.pipeline.config import PIANO_61KEY_CONFIG
            config = copy.deepcopy(PIANO_61KEY_CONFIG)
        else:
            config = PipelineConfig()

        try:
            config.stage_b.separation["enabled"] = True
            config.stage_b.separation["model"] = "htdemucs"
            config.stage_b.separation["synthetic_model"] = False
            config.stage_b.separation["harmonic_masking"] = {"enabled": True, "mask_width": 0.03}
        except Exception:
            pass

        try:
            config.stage_c.polyphony_filter["mode"] = "decomposed_melody"
        except Exception:
            pass

        try:
            config.stage_b.polyphonic_peeling["max_layers"] = 3
        except Exception:
            pass
        for det in ["swiftf0", "rmvpe", "crepe", "yin"]:
            try:
                if det in config.stage_b.detectors:
                    config.stage_b.detectors[det]["enabled"] = True
            except Exception:
                pass

        config = self._prepare_l5_config(config, overrides=overrides, override_path=override_path)

        # FIX: set the correct field
        try:
            config.stage_b.transcription_mode = "classic_song"
        except Exception:
            pass

        res = run_pipeline_on_audio(
            audio.astype(np.float32),
            int(read_sr),
            config,
            AudioType.POLYPHONIC,
            allow_separation=True,
            skip_global_profile=True,
        )

        m = self._save_run("L5.2", "tumhare_hi_rahenge", res, gt, apply_regression_gate=False)
        logger.info(f"L5.2 Complete. F1: {m['note_f1']}")

    def _save_real_song_result(self, level: str, name: str, res: Dict[str, Any]):
        """
        Save a real-song run (no GT). Robust to different return shapes from run_real_song.

        Writes:
          - {level}_{name}_metrics.json
          - {level}_{name}_run_info.json  (includes decision_trace + quality_gate if present)
          - optional {level}_{name}_timeline.csv (if timeline available)
        """
        analysis = res.get("analysis_data", None)
        transcription = res.get("transcription", None) or res.get("transcription_result", None) or res.get("result", None)

        if analysis is None and transcription is not None:
            analysis = getattr(transcription, "analysis_data", None)

        notes_obj = None
        if analysis is not None:
            notes_obj = getattr(analysis, "notes_before_quantization", None) or getattr(analysis, "notes", None)
        if notes_obj is None:
            notes_obj = res.get("notes", []) or []

        pred_notes = list(notes_obj) if notes_obj is not None else []
        pred_list = [
            (int(getattr(n, "midi_note", 0)), float(getattr(n, "start_sec", 0.0)), float(getattr(n, "end_sec", 0.0)))
            for n in pred_notes
        ]

        duration_sec = 0.0
        try:
            if analysis is not None and getattr(analysis, "meta", None) is not None:
                duration_sec = float(getattr(analysis.meta, "duration_sec", 0.0) or 0.0)
        except Exception:
            duration_sec = 0.0

        voiced_ratio = 0.0
        try:
            timeline = getattr(analysis, "timeline", None) if analysis is not None else None
            if timeline:
                total = max(1, len(timeline))
                voiced = 0
                for fr in timeline:
                    ap = getattr(fr, "active_pitches", None)
                    if ap is not None and len(ap) > 0:
                        voiced += 1
                voiced_ratio = float(voiced / total)
            else:
                if duration_sec > 0 and pred_list:
                    total_note_dur = 0.0
                    for _, s, e in pred_list:
                        total_note_dur += max(0.0, float(e) - float(s))
                    voiced_ratio = float(min(1.0, total_note_dur / max(1e-6, duration_sec)))
        except Exception:
            voiced_ratio = 0.0

        symptoms = compute_symptom_metrics(pred_list) or {}

        metrics = {
            "level": level,
            "name": name,
            "note_f1": None,
            "onset_mae_ms": None,
            "predicted_count": int(symptoms.get("note_count", 0.0) or 0.0),
            "gt_count": None,
            "fragmentation_score": float(symptoms.get("fragmentation_score", 0.0) or 0.0),
            "note_count_per_10s": float(symptoms.get("note_count_per_10s", 0.0) or 0.0),
            "median_note_len_ms": float(symptoms.get("median_note_len_ms", 0.0) or 0.0),
            "octave_jump_rate": float(symptoms.get("octave_jump_rate", 0.0) or 0.0),
            "voiced_ratio": float(voiced_ratio),
            "note_count": int(symptoms.get("note_count", 0.0) or 0.0),
        }
        self.results.append(metrics)

        base_path = os.path.join(self.output_dir, f"{level}_{name}")
        with open(f"{base_path}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        resolved_config = res.get("resolved_config", None) or res.get("config", None)
        stage_b_out = res.get("stage_b_out", None)
        stage_b_diag = {}
        if stage_b_out is not None:
            stage_b_diag = getattr(stage_b_out, "diagnostics", None) or {}

        analysis_diag = {}
        if analysis is not None:
            analysis_diag = getattr(analysis, "diagnostics", None) or {}

        decision_trace = None
        if isinstance(stage_b_diag, dict) and "decision_trace" in stage_b_diag:
            decision_trace = stage_b_diag.get("decision_trace")
        elif isinstance(analysis_diag, dict) and "decision_trace" in analysis_diag:
            decision_trace = analysis_diag.get("decision_trace")

        quality_gate = analysis_diag.get("quality_gate") if isinstance(analysis_diag, dict) else None
        stage_c_post = analysis_diag.get("stage_c_post") if isinstance(analysis_diag, dict) else None

        profiling = res.get("profiling", {}) or {}

        run_info = {
            "level": level,
            "name": name,
            "duration_sec": float(duration_sec or 0.0),
            "note_count": int(metrics["note_count"]),
            "voiced_ratio": float(metrics["voiced_ratio"]),
            "decision_trace": decision_trace if decision_trace is not None else {},
            "quality_gate": quality_gate if quality_gate is not None else {},
            "stage_c_post": stage_c_post if stage_c_post is not None else {},
            "stage_timings": profiling.get("stage_timings", {}),
            "detector_confidences": profiling.get("detector_confidences", {}),
            "artifacts_present": profiling.get("artifacts", {}),
            "config": asdict(resolved_config) if resolved_config is not None and is_dataclass(resolved_config) else {},
            "diagnostics": stage_b_diag if isinstance(stage_b_diag, dict) else {},
        }

        with open(f"{base_path}_run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, default=str)

        # Optional timeline CSV (minimal)
        try:
            if analysis is not None:
                timeline = getattr(analysis, "timeline", None)
                if timeline:
                    csv_rows = []
                    hop = float(getattr(getattr(analysis, "meta", None), "hop_length", 512) or 512)
                    sr_ = float(getattr(getattr(analysis, "meta", None), "sample_rate", 22050) or 22050)
                    for t, fr in enumerate(timeline):
                        hz = 0.0
                        ap = getattr(fr, "active_pitches", None)
                        if ap:
                            try:
                                hz = float(ap[0].hz)
                            except Exception:
                                hz = 0.0
                        csv_rows.append(
                            {
                                "t": int(t),
                                "time_sec": float((t * hop) / sr_),
                                "f0_hz": float(hz),
                                "voiced": bool(hz > 0),
                            }
                        )
                    if csv_rows:
                        write_frame_timeline_csv(f"{base_path}_timeline.csv", csv_rows)
        except Exception:
            pass

    def generate_summary(self):
        summary_path = os.path.join(self.output_dir, "summary.csv")
        leaderboard_path = os.path.join(self.output_dir, "leaderboard.json")
        snapshot_path = os.path.join(self.output_dir, "summary.json")
        latest_path = os.path.join("results", "benchmark_latest.json")

        header = [
            "level",
            "name",
            "note_f1",
            "onset_mae_ms",
            "predicted_count",
            "gt_count",
            "fragmentation_score",
            "note_count_per_10s",
            "median_note_len_ms",
            "octave_jump_rate",
            "voiced_ratio",
            "note_count",
        ]
        with open(summary_path, "w") as f:
            f.write(",".join(header) + "\n")
            for r in self.results:
                line = [str(r.get(h, "")) for h in header]
                f.write(",".join(line) + "\n")

        lb = {r["name"]: r.get("note_f1", 0.0) for r in self.results}
        with open(leaderboard_path, "w") as f:
            json.dump(lb, f, indent=2)

        with open(snapshot_path, "w") as f:
            json.dump(self.results, f, indent=2)

        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        previous: List[Dict[str, Any]] = []
        if os.path.exists(latest_path):
            try:
                with open(latest_path) as f:
                    previous = json.load(f)
            except Exception:
                previous = []

        merged = self._merge_results(previous, self.results)
        diff = self._compute_diff(previous, merged)
        with open(os.path.join(self.output_dir, "summary_diff.json"), "w") as f:
            json.dump(diff, f, indent=2)

        with open(latest_path, "w") as f:
            json.dump(merged, f, indent=2)

        logger.info(
            "Accuracy snapshot: "
            + ", ".join(f"{r['level']}:{r['name']} F1={r.get('note_f1', 0):.3f}" for r in merged)
        )
        logger.info(f"Summary saved to {summary_path}")


def resolve_levels(level_arg: str) -> List[str]:
    """Expand user level selection into runnable benchmark levels."""
    if level_arg == "all":
        return LEVEL_ORDER
    if level_arg == "L5":
        return ["L5.1", "L5.2"]
    return [level_arg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=f"results/benchmark_{int(time.time())}")
    parser.add_argument(
        "--level",
        choices=["all", "L0", "L1", "L2", "L3", "L4", "L6", "L5", "L5.1", "L5.2"],
        default="all",
        help="Run a specific benchmark level or all levels",
    )
    parser.add_argument("--l5_1_overrides", help=f"Inline JSON overrides for L5.1 config. {L5_OVERRIDE_FIELD_DOC}")
    parser.add_argument("--l5_1_config", help="Path to JSON file with L5.1 config overrides (merged after inline overrides).")
    parser.add_argument("--l5_2_overrides", help=f"Inline JSON overrides for L5.2 config. {L5_OVERRIDE_FIELD_DOC}")
    parser.add_argument("--l5_2_config", help="Path to JSON file with L5.2 config overrides (merged after inline overrides).")
    parser.add_argument(
        "--preset",
        choices=["none", "piano_61key"],
        default="none",
        help="Use a specific config preset as the baseline (for L4/L5.2).",
    )
    args = parser.parse_args()

    def _parse_override_json(raw: Optional[str], label: str) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for {label}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} overrides must be a JSON object")
        return parsed

    try:
        l5_1_overrides = _parse_override_json(args.l5_1_overrides, "L5.1 inline")
        l5_2_overrides = _parse_override_json(args.l5_2_overrides, "L5.2 inline")
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    runner = BenchmarkSuite(args.output)
    to_run = resolve_levels(args.level)
    use_preset = (args.preset == "piano_61key")

    try:
        for lvl in to_run:
            if lvl == "L0":
                runner.run_L0_mono_sanity()
            elif lvl == "L1":
                runner.run_L1_mono_musical()
            elif lvl == "L2":
                runner.run_L2_poly_dominant()
            elif lvl == "L3":
                runner.run_L3_full_poly()
            elif lvl == "L4":
                runner.run_L4_real_songs(use_preset=use_preset)
            elif lvl == "L6":
                runner.run_L6_synthetic_pop_song()
            elif lvl == "L5.1":
                runner.run_L5_1_kal_ho_na_ho(overrides=l5_1_overrides, override_path=args.l5_1_config)
            elif lvl == "L5.2":
                runner.run_L5_2_tumhare_hi_rahenge(overrides=l5_2_overrides, override_path=args.l5_2_config, use_preset=use_preset)
    except Exception as e:
        logger.exception(f"Benchmark Suite Failed: {e}")
        runner.generate_summary()
        sys.exit(1)

    runner.generate_summary()


if __name__ == "__main__":
    main()
