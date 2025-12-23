"""
Stage A — Load & Preprocess

This module handles audio loading, resampling, signal conditioning,
and loudness normalization.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Union, Any
import numpy as np
import math
import warnings
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import librosa
except ImportError:
    librosa = None

try:
    import pyloudnorm
except ImportError:
    pyloudnorm = None

try:  # pragma: no cover - optional heavy dependency
    import torch
except Exception:
    torch = None

try:  # pragma: no cover - optional heavy dependency
    from demucs import pretrained
    from demucs.apply import apply_model
except Exception:
    class _DemucsPretrainedStub:
        def get_model(self, *_, **__):
            raise ImportError("demucs not installed")

    def apply_model(*_, **__):  # type: ignore
        raise ImportError("demucs not installed")

    pretrained = _DemucsPretrainedStub()

try:
    import scipy.io.wavfile
    import scipy.signal
except ImportError:
    pass

from .models import StageAOutput, MetaData, Stem, AudioType, AudioQuality
from .config import PipelineConfig, StageAConfig

# Public constants (exported for tests)
TARGET_LUFS = -23.0
SILENCE_THRESHOLD_DB = 50  # Top-dB relative to peak

# -------------------------------------------------------------------------
# New Helpers for 61-Key Preprocessing
# -------------------------------------------------------------------------

try:
    import scipy.signal as _scipy_signal
except Exception:
    _scipy_signal = None

def _remove_dc_offset(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    return (y - float(np.mean(y))).astype(np.float32)

def _high_pass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if _scipy_signal is None or y.size == 0:
        return y.astype(np.float32)
    nyq = 0.5 * sr
    norm = min(max(float(cutoff_hz) / nyq, 1e-5), 0.999)
    try:
        sos = _scipy_signal.butter(int(order), norm, btype="highpass", output="sos")
        return _scipy_signal.sosfiltfilt(sos, y).astype(np.float32)
    except Exception as e:
        logger.warning(f"HPF failed: {e}")
        return y.astype(np.float32)

def _soft_limiter(y: np.ndarray, ceiling_db: float = -1.0, mode: str = "tanh", drive: float = 2.5) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32)
    ceiling = float(10 ** (ceiling_db / 20.0))  # e.g. -1 dB => ~0.891
    x = y.astype(np.float32)

    if mode == "clip":
        return np.clip(x, -ceiling, ceiling).astype(np.float32)

    # tanh soft clip normalized to ceiling
    d = float(max(0.1, drive))
    z = np.tanh(d * x)
    z = z / (np.tanh(d) + 1e-9)
    return (ceiling * z).astype(np.float32)


def detect_audio_type(audio: np.ndarray, sr: int, poly_flatness: float = 0.4) -> AudioType:
    """Lightweight heuristic to infer whether audio is mono/polyphonic."""
    if audio.ndim > 1:
        return AudioType.POLYPHONIC

    if len(audio) == 0:
        return AudioType.MONOPHONIC

    clip = audio[: min(len(audio), sr)]
    spectrum = np.abs(np.fft.rfft(clip))
    if spectrum.size == 0:
        return AudioType.MONOPHONIC

    flatness = float(np.exp(np.mean(np.log(spectrum + 1e-9))) / (np.mean(spectrum) + 1e-9))
    if flatness > poly_flatness:
        return AudioType.POLYPHONIC
    if flatness > poly_flatness * 0.6:
        return AudioType.POLYPHONIC_DOMINANT
    return AudioType.MONOPHONIC


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
    return audio

def _measure_loudness(audio: np.ndarray, sr: int) -> Tuple[Optional[float], str]:
    """Measure loudness using pyloudnorm (LUFS) or RMS fallback."""

    if pyloudnorm:
        try:
            meter = pyloudnorm.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            if not math.isinf(loudness):
                return float(loudness), "lufs"
        except Exception:
            pass

    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-9:
        return float(20.0 * math.log10(rms + 1e-12)), "rms_db"

    return None, "unmeasured"


def _normalize_loudness(
    audio: np.ndarray, sr: int, target_lufs: float
) -> Tuple[np.ndarray, float, Optional[float], Optional[float], str]:
    """Normalize audio to target LUFS using pyloudnorm or RMS fallback."""
    gain_db = 0.0
    measured_before, measurement_type = _measure_loudness(audio, sr)
    audio_out = audio

    if measurement_type == "lufs" and measured_before is not None:
        delta_lufs = target_lufs - measured_before
        gain_db = delta_lufs
        gain_lin = 10.0 ** (gain_db / 20.0)
        audio_out = audio * gain_lin
    elif measurement_type == "rms_db" and measured_before is not None:
        target_rms_db = -20.0
        gain_db = target_rms_db - measured_before
        gain_lin = 10.0 ** (gain_db / 20.0)
        audio_out = audio * gain_lin

    measured_after, post_type = _measure_loudness(audio_out, sr)
    if post_type != "unmeasured":
        measurement_type = post_type

    return audio_out, gain_db, measured_before, measured_after, measurement_type


def warped_linear_prediction(audio: np.ndarray, sr: int, pre_emphasis: float = 0.97) -> np.ndarray:
    """Simple LPC-inspired whitening via pre-emphasis."""
    if len(audio) == 0:
        return audio

    y = np.asarray(audio, dtype=np.float32).reshape(-1)

    # Only attempt LPC-style whitening on short clips to avoid long runtimes
    if len(y) <= max(int(sr * 2), 4096) and librosa is not None:
        try:
            order = max(2, min(16, len(y) // 8))
            coeffs = librosa.lpc(y, order=order)
            if "scipy" in globals() and hasattr(scipy, "signal"):
                return scipy.signal.lfilter(coeffs, [1.0], y).astype(np.float32)
        except Exception:
            pass

    emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    return emphasized.astype(np.float32)

def _estimate_noise_floor(audio: np.ndarray, percentile: float = 30.0, hop_length: int = 512) -> Tuple[float, float]:
    """Estimate noise floor RMS and dB."""
    if len(audio) == 0:
        return 0.0, -100.0

    # Frame energy
    if len(audio) < hop_length:
        rms_vals = np.array([np.sqrt(np.mean(audio**2))])
    else:
        # Simple framing
        n_frames = len(audio) // hop_length
        y = audio[:n_frames * hop_length]
        y_frames = y.reshape((n_frames, hop_length))
        rms_vals = np.sqrt(np.mean(y_frames**2, axis=1))

    noise_rms = float(np.percentile(rms_vals, percentile))
    noise_db = 20.0 * math.log10(noise_rms + 1e-9)
    return noise_rms, noise_db


def detect_tempo_and_beats(
    audio: np.ndarray,
    sr: int,
    enabled: bool,
    tightness: float = 100.0,
    trim: bool = True,
    hop_length: int = 512,
    pipeline_logger: Optional[Any] = None,
) -> Tuple[Optional[float], List[float]]:
    """Run a lightweight tempo/beat estimator if enabled and librosa is available."""

    if not enabled:
        if pipeline_logger:
            pipeline_logger.log_event("stage_a", "bpm_detection_skipped_disabled")
        return None, []

    if librosa is None:
        if pipeline_logger:
            pipeline_logger.log_event("stage_a", "bpm_detection_skipped_missing_librosa")
        return None, []

    try:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return None, []

        # Skip short clips (WI rule: >= 6.0s)
        duration = float(len(y)) / float(sr)
        if duration < 3.0:
            # Silent skip for very short audio to avoid log noise
            return None, []

        if duration < 6.0:
            if pipeline_logger:
                pipeline_logger.log_event("stage_a", "bpm_detection_skipped_short_audio", {"duration": duration})
            return None, []

        # Cap duration to keep beat tracking stable/cheap
        max_seconds = 90.0

        if librosa:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Removed resampling to ensure hop_length alignment with downstream processing
                if y.size > int(sr * max_seconds):
                    y = y[: int(sr * max_seconds)]

                tempo_est, beat_frames = librosa.beat.beat_track(
                    y=y,
                    sr=sr,
                    hop_length=hop_length,
                    tightness=tightness,
                    trim=trim,
                )
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()
        else:  # pragma: no cover - defensive fallback
            beat_times = []
            tempo_est = None

        tempo_val = tempo_est
        if tempo_val is not None and hasattr(tempo_val, "__len__"):
            tempo_val = tempo_val[0] if len(tempo_val) else None
        tempo_val = float(tempo_val) if tempo_val and np.isfinite(tempo_val) and tempo_val > 0 else None

        if pipeline_logger:
            pipeline_logger.log_event("stage_a", "bpm_detection_success", {
                "bpm": tempo_val,
                "n_beats": len(beat_times)
            })

        return tempo_val, beat_times
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Beat tracking failed: {exc}")
        if pipeline_logger:
            pipeline_logger.log_event("stage_a", "bpm_detection_failed", {"error": str(exc)})
        return None, []

def load_and_preprocess(
    audio_path: str,
    config: Optional[Union[PipelineConfig, StageAConfig]] = None,
    target_sr: Optional[int] = None,
    start_offset: float = 0.0,
    max_duration: Optional[float] = None,
    pipeline_logger: Optional[Any] = None,
    **kwargs: Any,
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
        full_conf = PipelineConfig()
        a_conf = full_conf.stage_a
    elif isinstance(config, StageAConfig):
        if logger:
            logger.warning(
                "StageAConfig provided; using default PipelineConfig for other stages. Pass PipelineConfig to configure Stage B/C/D."
            )
        full_conf = PipelineConfig()
        full_conf.stage_a = config
        a_conf = config
    else:
        full_conf = config
        a_conf = config.stage_a

    target_sr = target_sr or a_conf.target_sample_rate
    target_lufs = float(a_conf.loudness_normalization.get("target_lufs", TARGET_LUFS))
    trim_db = float(a_conf.silence_trimming.get("top_db", SILENCE_THRESHOLD_DB))

    # Resolve hop_length / window_size
    hop_length = 512
    window_size = 2048

    if full_conf:
        detectors = full_conf.stage_b.detectors
        yin_conf = detectors.get("yin")
        swift_conf = detectors.get("swiftf0")

        # Priority: YIN > SwiftF0 (if enabled)
        if yin_conf and yin_conf.get("enabled", False):
            hop_length = int(yin_conf.get("hop_length", 512))
            window_size = int(yin_conf.get("frame_length", 2048))
        elif swift_conf and swift_conf.get("enabled", False):
            hop_length = int(swift_conf.get("hop_length", 512))
            window_size = int(swift_conf.get("n_fft", 2048))

    # 1. Load & Resample
    try:
        if librosa:
            # librosa.load handles resampling and mono conversion (mono=True by default)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(
                    audio_path,
                    sr=target_sr,
                    mono=True,
                    offset=max(0.0, float(start_offset or 0.0)),
                    duration=max_duration,
                )
        else:
            audio, sr = _load_audio_fallback(audio_path, target_sr)
            if start_offset or max_duration:
                offset_samples = int(max(0.0, float(start_offset or 0.0)) * sr)
                end = int(offset_samples + (max_duration * sr if max_duration else len(audio)))
                audio = audio[offset_samples:end]
    except Exception as e:
        raise RuntimeError(f"Stage A failed to load audio: {e}")

    if len(audio) == 0:
        raise ValueError("Audio too short (empty)")

    original_duration = float(len(audio)) / float(sr)

    # 1b. Signal conditioning (DC Offset, HPF, Limiter)

    # DC offset removal
    dc_conf = getattr(a_conf, "dc_offset_removal", False)
    if dc_conf is True or (isinstance(dc_conf, dict) and dc_conf.get("enabled", False)):
        audio = _remove_dc_offset(audio)

    # High-pass filter
    hpf_cfg = getattr(a_conf, "high_pass_filter", None) or {}
    legacy_cut = getattr(a_conf, "high_pass_filter_cutoff", None)

    # Fallback to legacy if dict is missing but legacy exists
    if (not hpf_cfg) and (legacy_cut is not None):
        legacy_cut_val = legacy_cut.get("value", 55.0) if isinstance(legacy_cut, dict) else legacy_cut
        hpf_cfg = {"enabled": True, "cutoff_hz": float(legacy_cut_val), "order": 4}

    hpf_enabled = bool(hpf_cfg.get("enabled", False))
    hpf_cutoff = float(hpf_cfg.get("cutoff_hz", 60.0))
    hpf_order = int(hpf_cfg.get("order", 4))

    if hpf_enabled:
        audio = _high_pass(audio, sr=int(sr), cutoff_hz=hpf_cutoff, order=hpf_order)

    # Peak limiter
    lim = getattr(a_conf, "peak_limiter", {}) or {}
    if lim.get("enabled", False):
        audio = _soft_limiter(
            audio,
            ceiling_db=float(lim.get("ceiling_db", -1.0)),
            mode=str(lim.get("mode", "tanh")).lower(),
            drive=float(lim.get("drive", 2.5)),
        )

    # Diagnostics logging
    if pipeline_logger:
        pipeline_logger.log_event("stage_a", "preprocessing_applied", payload={
            "dc_offset": bool(dc_conf),
            "hpf": bool(hpf_enabled),
            "hpf_cutoff": float(hpf_cutoff),
            "limiter": bool(lim.get("enabled", False)),
        })


    # 1c. Transient Emphasis (Optional)
    tpe_conf = a_conf.transient_pre_emphasis
    if tpe_conf.get("enabled", True):
        audio = warped_linear_prediction(audio, sr=sr, pre_emphasis=float(tpe_conf.get("alpha", 0.97)))

    # 2. Trim Silence
    # Allow strict override from config/overrides
    if a_conf.silence_trimming.get("enabled", True):
        audio = _trim_silence(audio, top_db=trim_db, frame_length=window_size, hop_length=hop_length)

    # 3. Loudness Normalization
    gain_db = 0.0
    loudness_before = None
    loudness_after = None
    loudness_measurement = "unmeasured"
    if a_conf.loudness_normalization.get("enabled", True):
        audio, gain_db, loudness_before, loudness_after, loudness_measurement = _normalize_loudness(audio, sr, target_lufs)
    else:
        loudness_before, loudness_measurement = _measure_loudness(audio, sr)
        loudness_after = loudness_before

    # 4. Noise Floor
    nf_rms, nf_db = _estimate_noise_floor(audio, percentile=a_conf.noise_floor_estimation.get("percentile", 30), hop_length=hop_length)

    # 5. Optional tempo / beat detection (lightweight, single pass)
    bpm_cfg = getattr(a_conf, "bpm_detection", None) or {}
    bpm_enabled = bool(bpm_cfg.get("enabled", True))
    bpm_tightness = float(bpm_cfg.get("tightness", 100.0))
    bpm_trim = bool(bpm_cfg.get("trim", True))
    min_bpm = float(bpm_cfg.get("min_bpm", 0.0) or 0.0)
    max_bpm = float(bpm_cfg.get("max_bpm", 1e9) or 1e9)

    # Diagnostics for BPM
    bpm_diag = {"method": "librosa", "enabled": bpm_enabled, "run": False}

    tempo_bpm = None
    beat_times = []

    # Check for short audio gate (Stage A contract/invariants)
    # If audio is very short, librosa might fail or return garbage.
    # We set a gate: e.g., < 6.0s -> skip (too short)
    # Also if user explicitly disabled it.

    is_too_short = (len(audio) / sr) < 6.0 # explicit 6s gate per requirements

    if bpm_enabled and not is_too_short:
        tempo_bpm, beat_times = detect_tempo_and_beats(
            audio,
            sr=target_sr,
            enabled=True,
            tightness=bpm_tightness,
            trim=bpm_trim
        )
        if beat_times:
            beat_times = sorted(list(set(beat_times)))

        # Patch OPT5: BPM Clamping / Octave Correction
        if tempo_bpm and tempo_bpm > 0:
            while tempo_bpm < min_bpm:
                tempo_bpm *= 2.0
            while tempo_bpm > max_bpm:
                tempo_bpm *= 0.5

        bpm_diag["run"] = True
        bpm_diag["result"] = "success" if tempo_bpm else "no_tempo"
        bpm_diag["clamped_bpm"] = tempo_bpm
    else:
        bpm_diag["run"] = False
        bpm_diag["reason"] = "disabled" if not bpm_enabled else "too_short"
        # Fallback default
        tempo_bpm = 120.0
        beat_times = []

    # Ensure tempo_bpm is 120.0 if None
    if tempo_bpm is None:
        tempo_bpm = 120.0

    if pipeline_logger:
        pipeline_logger.log_event("stage_a", "bpm_detection", payload=bpm_diag)
        pipeline_logger.log_event("stage_a", "params_resolved", payload={
            "bpm_detection": {
                "enabled": bpm_enabled,
                "tightness": bpm_tightness,
                "trim": bpm_trim
            },
            "high_pass_filter": {
                "enabled": hpf_enabled,
                "cutoff_hz": hpf_cutoff,
                "order": hpf_order,
                "legacy_fallback_used": (not getattr(a_conf, "high_pass_filter", None)) and (legacy_cut is not None)
            }
        })

    # 6. Detect texture (mono / poly)
    detected_type = detect_audio_type(audio, sr)

    # 7. Populate Metadata & Output
    # Basic MetaData
    meta = MetaData(
        audio_path=audio_path,
        sample_rate=sr,
        target_sr=target_sr,
        duration_sec=float(len(audio)) / float(sr),
        original_duration_sec=float(original_duration),
        n_channels=1, # we forced mono
        lufs=target_lufs, # assumed target
        loudness_measurement=loudness_measurement,
        loudness_or_rms=loudness_before if loudness_before is not None else -float("inf"),
        loudness_post_norm=loudness_after if loudness_after is not None else -float("inf"),
        normalization_gain_db=gain_db,
        rms_db=20.0 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9),
        noise_floor_rms=nf_rms,
        noise_floor_db=nf_db,
        pipeline_version="2.0.0",

        # Instrument (Patch C1)
        # Attempt to set instrument if explicitly provided in kwargs or config, but don't force default.
        instrument=str(kwargs.get("instrument") or getattr(config, "instrument", None) or "") or None,

        # Resolved values
        hop_length=hop_length,
        window_size=window_size,

        processing_mode=detected_type.value,
        audio_type=detected_type,

        tempo_bpm=tempo_bpm,
        beats=beat_times,
        beat_times=beat_times,  # Populate alias as well
    )

    # Output only the mix stem. Separation is handled in Stage B.
    stems = {"mix": Stem(audio=audio, sr=sr, type="mix")}

    return StageAOutput(
        stems=stems,
        meta=meta,
        audio_type=detected_type,
        noise_floor_rms=nf_rms,
        noise_floor_db=nf_db,
        beats=beat_times,
        diagnostics={"bpm": bpm_diag}
    )
