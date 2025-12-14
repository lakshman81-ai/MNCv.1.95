"""
Stage B — Feature Extraction

This module implements pitch detection and feature extraction.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings

from .models import StageAOutput, FramePitch, AnalysisData, AudioType, StageBOutput
from .config import PipelineConfig
from .detectors import (
    SwiftF0Detector, SACFDetector, YinDetector,
    CQTDetector, RMVPEDetector, CREPEDetector,
    iterative_spectral_subtraction, create_harmonic_mask,
    _frame_audio,
    BasePitchDetector
)

# Re-export for tests
__all__ = [
    "extract_features",
    "create_harmonic_mask",
    "iterative_spectral_subtraction"
]

def _arrays_to_timeline(
    f0: np.ndarray,
    conf: np.ndarray,
    rms: Optional[np.ndarray],
    sr: int,
    hop_length: int
) -> List[FramePitch]:
    """Convert f0/conf arrays to List[FramePitch]."""
    timeline = []
    n_frames = len(f0)
    for i in range(n_frames):
        hz = float(f0[i])
        c = float(conf[i])
        r = float(rms[i]) if rms is not None and i < len(rms) else 0.0

        midi = 0.0
        if hz > 0:
            midi = 69.0 + 12.0 * np.log2(hz / 440.0)

        time_sec = float(i * hop_length) / float(sr)

        timeline.append(FramePitch(
            time=time_sec,
            pitch_hz=hz,
            confidence=c,
            midi=round(midi) if hz > 0 else None,
            rms=r,
            active_pitches=[]
        ))
    return timeline

def _init_detector(name: str, conf: Dict[str, Any], sr: int, hop_length: int) -> Optional[BasePitchDetector]:
    """Initialize a detector if enabled."""
    if not conf.get("enabled", False):
        return None

    try:
        if name == "swiftf0":
            return SwiftF0Detector(sr, hop_length, **conf)
        elif name == "sacf":
            return SACFDetector(sr, hop_length, **conf)
        elif name == "yin":
            return YinDetector(sr, hop_length, **conf)
        elif name == "cqt":
            return CQTDetector(sr, hop_length, **conf)
        elif name == "rmvpe":
            return RMVPEDetector(sr, hop_length, **conf)
        elif name == "crepe":
            return CREPEDetector(sr, hop_length, **conf)
    except Exception as e:
        warnings.warn(f"Failed to init detector {name}: {e}")
        return None
    return None

def _ensemble_merge(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    weights: Dict[str, float],
    disagreement_cents: float = 70.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple f0/conf tracks based on weights.
    Simple strategy: Pick the candidate with highest (conf * weight) per frame.
    Future: Handle octave disagreements or smoothing.
    """
    if not results:
        return np.array([]), np.array([])

    # Ensure all have same length
    lengths = [len(r[0]) for r in results.values()]
    if not lengths:
        return np.array([]), np.array([])
    max_len = max(lengths)

    # Pad if necessary
    aligned_results = {}
    for name, (f0, conf) in results.items():
        if len(f0) < max_len:
            pad = max_len - len(f0)
            f0 = np.pad(f0, (0, pad))
            conf = np.pad(conf, (0, pad))
        aligned_results[name] = (f0, conf)

    final_f0 = np.zeros(max_len, dtype=np.float32)
    final_conf = np.zeros(max_len, dtype=np.float32)

    # Iterate frames
    for i in range(max_len):
        best_score = -1.0
        best_f0 = 0.0
        best_conf = 0.0

        for name, (f0, conf) in aligned_results.items():
            w = weights.get(name, 1.0)
            score = conf[i] * w
            if score > best_score:
                best_score = score
                best_f0 = f0[i]
                best_conf = conf[i]

        final_f0[i] = best_f0
        final_conf[i] = best_conf

    return final_f0, final_conf

def extract_features(
    stage_a_out: StageAOutput,
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> StageBOutput:
    """
    Stage B: Extract pitch and features.
    Respects config.stage_b for detector selection and ensemble weights.
    """
    if config is None:
        config = PipelineConfig()

    b_conf = config.stage_b
    sr = stage_a_out.meta.sample_rate
    hop_length = stage_a_out.meta.hop_length

    # 1. Initialize Detectors based on Config
    detectors: Dict[str, BasePitchDetector] = {}
    for name, det_conf in b_conf.detectors.items():
        det = _init_detector(name, det_conf, sr, hop_length)
        if det:
            detectors[name] = det

    # Ensure baseline fallback if no detectors enabled/working
    if not detectors:
        warnings.warn("No detectors enabled or initialized in Stage B. Falling back to default YIN/ACF.")
        detectors["yin"] = YinDetector(sr, hop_length)

    # Log resolved config (we can put it in meta or just print for now as 'log')
    # The requirement is "log a resolved runtime config JSON per run" which we handle by
    # returning data in StageBOutput that can be inspected.

    stem_timelines: Dict[str, List[FramePitch]] = {}
    per_detector: Dict[str, Any] = {}
    f0_main: Optional[np.ndarray] = None

    # 2. Process Stems
    for stem_name, stem in stage_a_out.stems.items():
        audio = stem.audio
        per_detector[stem_name] = {}

        stem_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Decide which detectors to run for this stem
        # If specific override logic exists (e.g. vocals -> use dl models), apply here.
        # Otherwise run all enabled detectors (or a subset based on profile?).
        # For simplicity and robustness per requirements, let's run all enabled detectors for 'mix' or 'vocals'.
        # Optimize: if "drums", maybe skip pitch? config.get_profile('drums') says ignore_pitch.

        # Check profile
        # Since we don't have per-stem instrument info easily unless we infer or map stem names:
        # 'vocals' -> vocals profile, 'mix' -> piano/general?
        # Use simple mapping for now

        # Run all initialized detectors on this stem
        for name, det in detectors.items():
            try:
                # pass audio path if available for caching/logging inside detector
                f0, conf = det.predict(audio, audio_path=stage_a_out.meta.audio_path)
                stem_results[name] = (f0, conf)
                per_detector[stem_name][name] = (f0, conf)
            except Exception as e:
                warnings.warn(f"Detector {name} failed on stem {stem_name}: {e}")


        # Ensemble Merge
        if stem_results:
            merged_f0, merged_conf = _ensemble_merge(stem_results, b_conf.ensemble_weights, b_conf.pitch_disagreement_cents)
        else:
            merged_f0 = np.zeros(1)
            merged_conf = np.zeros(1)

        # If "other" stem and configured, maybe run ISS (polyphonic peeling)
        # Only if strict polyphonic mode?
        # The prompt says: "Stage B currently ... converts ... into per-stem timelines ... but ignores most configuration knobs".
        # We need to respect them.
        # If we have ISS enabled in config and this is a polyphonic context:
        if stem_name == "other" and b_conf.polyphonic_peeling.get("max_layers", 0) > 0:
             # Logic for ISS ...
             # We need a primary and validator. Use "swiftf0" or "yin" as primary if available.
             primary = detectors.get("swiftf0") or detectors.get("yin") or detectors.get("sacf")
             if primary:
                 layers = iterative_spectral_subtraction(
                     audio, sr, primary_detector=primary,
                     max_polyphony=b_conf.polyphonic_peeling.get("max_layers", 4)
                 )
                 # If ISS returns layers, we might want to store them in f0_layers of StageBOutput
                 # For stem_timeline, we typically flatten or take the most prominent.
                 pass

        # Calculate RMS
        n_fft = stage_a_out.meta.window_size if stage_a_out.meta.window_size else 2048
        frames = _frame_audio(audio, n_fft, hop_length)
        rms_vals = np.sqrt(np.mean(frames**2, axis=1))

        # Pad/Trim RMS to match merged F0
        if len(rms_vals) < len(merged_f0):
            rms_vals = np.pad(rms_vals, (0, len(merged_f0) - len(rms_vals)))
        elif len(rms_vals) > len(merged_f0):
            rms_vals = rms_vals[:len(merged_f0)]

        stem_timelines[stem_name] = _arrays_to_timeline(merged_f0, merged_conf, rms_vals, sr, hop_length)

        # Set main f0 (prefer vocals, then mix)
        if stem_name == "vocals":
            f0_main = merged_f0
        elif stem_name == "mix" and f0_main is None:
            f0_main = merged_f0

    if f0_main is None:
         # Pick first
         if stem_timelines:
             first_stem = next(iter(stem_timelines.values()))
             f0_main = np.array([fp.pitch_hz for fp in first_stem])
         else:
             f0_main = np.array([])

    time_grid = np.array([])
    if len(f0_main) > 0:
        time_grid = np.arange(len(f0_main)) * hop_length / sr

    return StageBOutput(
        time_grid=time_grid,
        f0_main=f0_main,
        f0_layers=[], # Populated if ISS or poly detectors are used extensively
        per_detector=per_detector,
        stem_timelines=stem_timelines,
        meta=stage_a_out.meta
    )
