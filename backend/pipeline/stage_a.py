"""
Stage A — Load & Preprocess

This module handles audio loading, resampling, signal conditioning,
and loudness normalization.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Union
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


def _detect_tempo_and_beats(audio: np.ndarray, sr: int, enabled: bool) -> Tuple[Optional[float], List[float]]:
    """Run a lightweight tempo/beat estimator if enabled and librosa is available."""

    if not enabled or librosa is None:
        return None, []

    try:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return None, []

        # Downsample and cap duration to keep beat tracking stable/cheap
        target_sr = 16000 if sr > 16000 else max(11025, sr)
        max_seconds = 90.0
        if librosa:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if sr != target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                if y.size > int(target_sr * max_seconds):
                    y = y[: int(target_sr * max_seconds)]
                tempo_est, beat_frames = librosa.beat.beat_track(
                    y=y,
                    sr=target_sr,
                    hop_length=256,
                    tightness=100,
                )
                beat_times = librosa.frames_to_time(beat_frames, sr=target_sr, hop_length=256).tolist()
        else:  # pragma: no cover - defensive fallback
            beat_times = []
            tempo_est = None

        tempo_val = tempo_est
        if tempo_val is not None and hasattr(tempo_val, "__len__"):
            tempo_val = tempo_val[0] if len(tempo_val) else None
        tempo_val = float(tempo_val) if tempo_val and np.isfinite(tempo_val) and tempo_val > 0 else None
        return tempo_val, beat_times
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Beat tracking failed: {exc}")
        return None, []

def load_and_preprocess(
    audio_path: str,
    config: Optional[Union[PipelineConfig, StageAConfig]] = None,
    target_sr: Optional[int] = None,
    start_offset: float = 0.0,
    max_duration: Optional[float] = None,
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
        # Default empty PipelineConfig which has a default StageAConfig
        full_conf = PipelineConfig()
        a_conf = full_conf.stage_a
    elif isinstance(config, StageAConfig):
        # Wrap StageAConfig in a PipelineConfig so Stage B defaults still
        # influence hop/window selection for consistency across stages.
        full_conf = PipelineConfig()
        full_conf.stage_a = config
        a_conf = config
    else:
        # It's a PipelineConfig
        full_conf = config
        a_conf = config.stage_a

    target_sr = target_sr or a_conf.target_sample_rate
    target_lufs = float(a_conf.loudness_normalization.get("target_lufs", TARGET_LUFS))
    trim_db = float(a_conf.silence_trimming.get("top_db", SILENCE_THRESHOLD_DB))

    # Resolve hop_length / window_size
    hop_length = 512
    window_size = 2048

    if full_conf:
        # Check Stage B detectors for 'yin' or 'swiftf0' preference
        # We use .get() safely
        detectors = full_conf.stage_b.detectors
        yin_conf = detectors.get("yin")
        swift_conf = detectors.get("swiftf0")

        # Priority: YIN > SwiftF0 (if enabled)
        # Note: configs might not have explicit hop_length keys, so we check existence
        if yin_conf and yin_conf.get("enabled", False):
            hop_length = int(yin_conf.get("hop_length", 512))
            window_size = int(yin_conf.get("frame_length", 2048)) # YIN usually calls it frame_length
        elif swift_conf and swift_conf.get("enabled", False):
            hop_length = int(swift_conf.get("hop_length", 512))
            # SwiftF0 might not specify window size same way, default to 2048
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
            # Manual offset/duration handling for fallback path
            if start_offset or max_duration:
                offset_samples = int(max(0.0, float(start_offset or 0.0)) * sr)
                end = int(offset_samples + (max_duration * sr if max_duration else len(audio)))
                audio = audio[offset_samples:end]
    except Exception as e:
        raise RuntimeError(f"Stage A failed to load audio: {e}")

    if len(audio) == 0:
        raise ValueError("Audio too short (empty)")

    original_duration = float(len(audio)) / float(sr)

    # 1b. Transient Emphasis (Optional)
    # Applied after loading but before silence trimming/norm
    tpe_conf = a_conf.transient_pre_emphasis
    if tpe_conf.get("enabled", True):
        audio = warped_linear_prediction(audio, sr=sr, pre_emphasis=float(tpe_conf.get("alpha", 0.97)))

    # 2. Trim Silence
    if a_conf.silence_trimming.get("enabled", True):
        audio = _trim_silence(audio, top_db=trim_db, frame_length=window_size, hop_length=hop_length)

    # 3. Loudness Normalization
    gain_db = 0.0
    if a_conf.loudness_normalization.get("enabled", True):
        audio, gain_db = _normalize_loudness(audio, sr, target_lufs)

    # 4. Noise Floor
    nf_rms, nf_db = _estimate_noise_floor(audio, percentile=a_conf.noise_floor_estimation.get("percentile", 30), hop_length=hop_length)

    # 5. Optional tempo / beat detection (lightweight, single pass)
    tempo_bpm, beat_times = _detect_tempo_and_beats(
        audio,
        sr=target_sr,
        enabled=a_conf.bpm_detection.get("enabled", False),
    )

    # 6. Detect texture (mono / poly) and optionally run separation
    detected_type = detect_audio_type(audio, sr)
    stems = {"mix": Stem(audio=audio, sr=sr, type="mix")}

    sep_conf = getattr(a_conf, "separation", {}) if hasattr(a_conf, "separation") else {}
    separation_enabled = sep_conf.get("enabled", detected_type != AudioType.MONOPHONIC)
    if separation_enabled and detected_type != AudioType.MONOPHONIC and pretrained and apply_model and torch is not None:
        try:
            model = pretrained.get_model(sep_conf.get("model", "htdemucs"))
            overlap = sep_conf.get("overlap", 0.25)
            shifts = sep_conf.get("shifts", 1)
            mix_tensor = torch.tensor(audio, dtype=torch.float32)[None, None, :]
            demucs_out = apply_model(model, mix_tensor, overlap=overlap, shifts=shifts)

            separated: Dict[str, Stem] = {}
            for idx, name in enumerate(getattr(model, "sources", ["vocals", "drums", "bass", "other"])):
                stem_audio = demucs_out[0, idx]
                if hasattr(stem_audio, "detach"):
                    stem_audio = stem_audio.detach().cpu().numpy()
                if stem_audio.ndim > 1:
                    stem_audio = np.mean(stem_audio, axis=0)
                sep_sr = getattr(model, "samplerate", sr)
                separated[name] = Stem(audio=np.asarray(stem_audio, dtype=np.float32), sr=int(sep_sr), type=name)

            if separated:
                stems.update(separated)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"Demucs separation failed: {exc}")

    # 6. Populate Metadata & Output
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

        # Resolved values
        hop_length=hop_length,
        window_size=window_size,

        processing_mode=detected_type.value,
        audio_type=detected_type,

        tempo_bpm=tempo_bpm,
        beats=beat_times,
    )

    # Always provide only the true mix by default.
    stems = {"mix": Stem(audio=audio, sr=sr, type="mix")}

    # If (and only if) real separation produced stems, merge them here.
    # Expect a dict like {"vocals": np.ndarray, "bass": ..., "other": ...} or {"vocals": Stem, ...}
    # In this function, 'separated' (if populated above) already contains Stem objects.
    if locals().get("separated") and isinstance(separated, dict):
        for k, v in separated.items():
            if isinstance(v, Stem):
                stems[k] = v
            elif isinstance(v, np.ndarray):
                stems[k] = Stem(audio=v.astype(np.float32), sr=sr, type=str(k))

    return StageAOutput(
        stems=stems,
        meta=meta,
        audio_type=detected_type,
        noise_floor_rms=nf_rms,
        noise_floor_db=nf_db,
        beats=beat_times,
    )
