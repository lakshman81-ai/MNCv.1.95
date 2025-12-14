"""
Stage A — Load & Preprocess

This module handles audio loading, resampling, signal conditioning,
and loudness normalization.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np
import math
import warnings

# Optional dependencies
try:
    import librosa
except ImportError:
    librosa = None

try:
    import pyloudnorm
except ImportError:
    pyloudnorm = None

try:
    import scipy.io.wavfile
    import scipy.signal
except ImportError:
    pass

from .models import StageAOutput, MetaData, Stem, AudioType, AudioQuality
from .config import PipelineConfig

# Public constants (exported for tests)
TARGET_LUFS = -23.0
SILENCE_THRESHOLD_DB = 50  # Top-dB relative to peak


def _load_audio_fallback(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Fallback loader using scipy if librosa is missing."""
    try:
        sr, audio = scipy.io.wavfile.read(path)
    except Exception as e:
        raise ImportError(f"librosa missing and scipy failed to load {path}: {e}")

    # Convert int to float -1..1
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0

    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed (simple integer factors only or scipy.signal.resample)
    if sr != target_sr:
        if scipy.signal:
            num_samples = int(len(audio) * float(target_sr) / sr)
            audio = scipy.signal.resample(audio, num_samples)
        else:
            # Poor man's resample (nearest neighbor) if scipy.signal missing?
            # unlikely if scipy.io is there.
            pass

    return audio, target_sr

def _trim_silence(audio: np.ndarray, top_db: float, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Trim leading/trailing silence."""
    if librosa:
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
            return audio_trimmed
        except Exception:
            pass

    # Fallback trim: RMS threshold
    # Simple calculation of RMS per frame
    return audio # TODO: implement manual trim if needed, for now return as is if librosa fails

def _normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float) -> Tuple[np.ndarray, float]:
    """Normalize audio to target LUFS using pyloudnorm or RMS fallback."""
    gain_db = 0.0

    # Method 1: pyloudnorm
    if pyloudnorm:
        try:
            meter = pyloudnorm.Meter(sr)  # create BS.1770 meter
            loudness = meter.integrated_loudness(audio)

            # If loudness is -inf, don't normalize
            if not math.isinf(loudness):
                delta_lufs = target_lufs - loudness
                gain_lin = 10.0 ** (delta_lufs / 20.0)
                audio_norm = audio * gain_lin

                # Check for clipping? (Peak limiter config is in StageAConfig but applied later?)
                # For now just return normalized
                return audio_norm, delta_lufs
        except Exception:
            pass

    # Method 2: RMS-based fallback
    # Assume -23 LUFS is roughly -23 dB RMS (sine wave) but K-weighting differs.
    # Simple RMS normalization to -20 dB RMS as a proxy for -23 LUFS
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-9:
        target_rms = 10.0 ** (-20.0 / 20.0) # -20 dB
        gain_lin = target_rms / rms
        gain_db = 20.0 * np.log10(gain_lin)
        return audio * gain_lin, gain_db

    return audio, 0.0

def _estimate_noise_floor(audio: np.ndarray, percentile: float = 30.0, hop_length: int = 512) -> Tuple[float, float]:
    """Estimate noise floor RMS and dB."""
    if len(audio) == 0:
        return 0.0, -100.0

    # Frame energy
    # Make frames
    if len(audio) < hop_length:
        rms_vals = np.array([np.sqrt(np.mean(audio**2))])
    else:
        # Simple framing
        # Pad to ensure we cover everything?
        n_frames = len(audio) // hop_length
        # We can just reshape roughly to get stats
        # Truncate to multiple of hop
        y = audio[:n_frames * hop_length]
        y_frames = y.reshape((n_frames, hop_length))
        rms_vals = np.sqrt(np.mean(y_frames**2, axis=1))

    noise_rms = float(np.percentile(rms_vals, percentile))
    noise_db = 20.0 * math.log10(noise_rms + 1e-9)
    return noise_rms, noise_db

def load_and_preprocess(
    audio_path: str,
    config: Optional[PipelineConfig] = None
) -> StageAOutput:
    """
    Stage A main entry point.

    1. Load audio (resample to target_sr).
    2. Convert to mono.
    3. Trim silence.
    4. Normalize loudness.
    5. Estimate noise floor.
    """
    if config is None:
        config = PipelineConfig()

    a_conf = config.stage_a
    target_sr = a_conf.target_sample_rate
    target_lufs = float(a_conf.loudness_normalization.get("target_lufs", TARGET_LUFS))
    trim_db = float(a_conf.silence_trimming.get("top_db", SILENCE_THRESHOLD_DB))

    # Resolve window/hop from detectors if possible
    # We look for 'yin' or 'swiftf0' as reference
    hop_length = 512
    window_size = 2048
    if config.stage_b and config.stage_b.detectors:
        # Prefer yin as baseline, or swiftf0
        ref_det = config.stage_b.detectors.get('yin') or config.stage_b.detectors.get('swiftf0')
        if ref_det:
            hop_length = int(ref_det.get('hop_length', 512))
            window_size = int(ref_det.get('n_fft', 2048))

    # 1. Load & Resample
    try:
        if librosa:
            # librosa.load handles resampling and mono conversion (mono=True by default)
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        else:
            audio, sr = _load_audio_fallback(audio_path, target_sr)
    except Exception as e:
        raise RuntimeError(f"Stage A failed to load audio: {e}")

    if len(audio) == 0:
        raise ValueError("Audio too short (empty)")

    original_duration = float(len(audio)) / float(sr)

    # 2. Trim Silence
    if a_conf.silence_trimming.get("enabled", True):
        audio = _trim_silence(audio, top_db=trim_db)

    # 3. Loudness Normalization
    gain_db = 0.0
    if a_conf.loudness_normalization.get("enabled", True):
        audio, gain_db = _normalize_loudness(audio, sr, target_lufs)

    # 4. Noise Floor
    nf_rms, nf_db = _estimate_noise_floor(audio, percentile=a_conf.noise_floor_estimation.get("percentile", 30))

    # 5. Populate Metadata & Output
    # Basic MetaData
    meta = MetaData(
        audio_path=audio_path,
        sample_rate=sr,
        target_sr=target_sr,
        duration_sec=float(len(audio)) / float(sr),
        n_channels=1, # we forced mono
        lufs=target_lufs, # assumed target
        normalization_gain_db=gain_db,
        rms_db=20.0 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9),
        noise_floor_rms=nf_rms,
        noise_floor_db=nf_db,
        pipeline_version="2.0.0",

        # Defaults for downstream (can be updated by detectors)
        hop_length=hop_length,
        window_size=window_size,
        processing_mode="monophonic", # Default assumption, detector may refine
        audio_type=AudioType.MONOPHONIC
    )

    # Stems: for now just 'mix' since we don't do separation in Stage A
    # (Separation is typically start of Stage B or pre-B)
    # The requirement says Stage A output has "stems (mix / vocals ... depending on separation availability)"
    # But usually separation is a heavy process. If explicit separation is not in Stage A logic (it's in Stage B config),
    # we just provide 'mix'.
    stems = {
        "mix": Stem(audio=audio, sr=sr, type="mix")
    }

    return StageAOutput(
        stems=stems,
        meta=meta,
        audio_type=AudioType.MONOPHONIC, # Default
        noise_floor_rms=nf_rms,
        noise_floor_db=nf_db
    )
