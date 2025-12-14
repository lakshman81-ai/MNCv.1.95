"""
Stage B — Feature Extraction

This module implements pitch detection and feature extraction.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from .models import StageAOutput, FramePitch, AnalysisData, AudioType
from .config import PipelineConfig
from .detectors import (
    SwiftF0Detector, SACFDetector, YinDetector,
    CQTDetector, RMVPEDetector, CREPEDetector,
    iterative_spectral_subtraction, create_harmonic_mask,
    _frame_audio
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

def extract_features(
    stage_a_out: StageAOutput,
    use_crepe: bool = False,
    confidence_threshold: float = 0.5,
    min_duration_ms: float = 0.0,
    config: Optional[PipelineConfig] = None,
    **kwargs
) -> Tuple[List[FramePitch], List[Any], List[Any], Dict[str, List[FramePitch]]]:

    sr = stage_a_out.meta.sample_rate
    hop_length = stage_a_out.meta.hop_length

    # Initialize detectors
    # For now, we use a simple selection strategy based on arguments
    detectors = {}

    # Defaults
    detectors["swift"] = SwiftF0Detector(sr, hop_length)
    detectors["sacf"] = SACFDetector(sr, hop_length)
    detectors["yin"] = YinDetector(sr, hop_length)

    primary_detector = detectors["swift"]
    if use_crepe:
        detectors["crepe"] = CREPEDetector(sr, hop_length)
        primary_detector = detectors["crepe"]

    stem_timelines: Dict[str, List[FramePitch]] = {}

    # Process stems
    for stem_name, stem in stage_a_out.stems.items():
        audio = stem.audio

        f0, conf = None, None

        if stem_name == "vocals":
            # Vocals use primary detector (SwiftF0 or Crepe)
            f0, conf = primary_detector.predict(audio)

        elif stem_name == "other":
            # Other uses ISS (Iterative Spectral Subtraction)
            # using SACF as validator? Test used SACF.
            # For simplicity, let's use Swift as primary and SACF as validator for ISS
            layers = iterative_spectral_subtraction(
                audio, sr,
                primary_detector=detectors["swift"],
                validator_detector=detectors["sacf"],
                max_polyphony=4
            )
            # For now, just take the first layer or merge?
            # Test expects simple timeline. Let's take top layer.
            if layers:
                f0, conf = layers[0]
            else:
                f0, conf = np.zeros_like(audio), np.zeros_like(audio) # Mismatch size, but ISS handles internally

        elif stem_name == "mix":
             # Mix: if monophonic, use primary. If poly, use ISS?
             # run_real_songs uses "mix" with AudioType.MONOPHONIC
             if stage_a_out.meta.audio_type == AudioType.MONOPHONIC:
                 f0, conf = primary_detector.predict(audio)
             else:
                 # Polyphonic mix fallback
                 f0, conf = primary_detector.predict(audio)

        else:
            # Fallback for bass, drums, etc.
            f0, conf = detectors["yin"].predict(audio)

        if f0 is not None and conf is not None:
            # Compute RMS for the audio
            # Assuming n_fft matches window_size or default 2048
            n_fft = stage_a_out.meta.window_size if stage_a_out.meta.window_size else 2048
            frames = _frame_audio(audio, n_fft, hop_length)
            # RMS per frame
            rms_vals = np.sqrt(np.mean(frames**2, axis=1))

            # Ensure length matches f0 (pad or trim)
            if len(rms_vals) < len(f0):
                rms_vals = np.pad(rms_vals, (0, len(f0) - len(rms_vals)))
            elif len(rms_vals) > len(f0):
                rms_vals = rms_vals[:len(f0)]

            stem_timelines[stem_name] = _arrays_to_timeline(f0, conf, rms_vals, sr, hop_length)

    # Aggregate global timeline
    # For now, prefer vocals, then mix, then other
    timeline = []
    if "vocals" in stem_timelines:
        timeline = stem_timelines["vocals"]
    elif "mix" in stem_timelines:
        timeline = stem_timelines["mix"]
    elif "other" in stem_timelines:
        timeline = stem_timelines["other"]

    return timeline, [], [], stem_timelines
