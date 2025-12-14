import os
import warnings
from typing import Dict, Optional, Any

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import scipy.signal
import torch
from demucs import pretrained
from demucs.apply import apply_model

from .models import MetaData, AudioType, AudioQuality, Stem, StageAOutput
from .config import PIANO_61KEY_CONFIG, StageAConfig


# ------------------------------------------------------------
# Helper: Audio Type Detection
# ------------------------------------------------------------

def detect_audio_type(y: np.ndarray, sr: int) -> AudioType:
    """
    Heuristic to detect if audio is Monophonic or Polyphonic.
    Uses spectral flatness and spectral contrast.
    """
    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness))

    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    mean_contrast = float(np.mean(contrast))

    if mean_flatness < 0.05 and mean_contrast > 20.0:
        return AudioType.MONOPHONIC
    elif mean_flatness < 0.2:
        return AudioType.POLYPHONIC_DOMINANT
    else:
        return AudioType.POLYPHONIC


# ------------------------------------------------------------
# Helper: Demucs (used by Stage B, not called here)
# ------------------------------------------------------------

def apply_demucs(
    audio: np.ndarray,
    sr: int,
    model_name: str = "htdemucs",
    separation_conf: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply Hybrid Demucs.
    Returns dict of mono stems: {'vocals', 'bass', 'drums', 'other'}.

    separation_conf (from StageBConfig.separation) may contain:
        - 'shifts': int (default 1)
        - 'overlap': float (default 0.25)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = pretrained.get_model(model_name)
        model.to(device)
    except Exception as e:
        warnings.warn(f"Failed to load Demucs model: {e}. Returning original mix as stems.")
        return {"vocals": audio, "bass": audio, "drums": audio, "other": audio}

    # Ensure stereo [2, T]
    if audio.ndim == 1:
        audio_stereo = np.stack([audio, audio])
    elif audio.ndim == 2:
        if audio.shape[0] > 2:
            audio_stereo = audio.T
        else:
            audio_stereo = audio
    else:
        audio_stereo = np.mean(audio, axis=-1)
        audio_stereo = np.stack([audio_stereo, audio_stereo])

    model_sr = model.samplerate
    if sr != model_sr:
        audio_stereo = librosa.resample(audio_stereo, orig_sr=sr, target_sr=model_sr)
        sr = model_sr

    wav = torch.tensor(audio_stereo, dtype=torch.float32).to(device)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    # Config-driven knobs
    shifts = 1
    overlap = 0.25
    if separation_conf is not None:
        shifts = int(separation_conf.get("shifts", shifts))
        overlap = float(separation_conf.get("overlap", overlap))

    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            shifts=shifts,
            split=True,
            overlap=overlap,
            progress=False,
        )[0]

    sources = sources * ref.std() + ref.mean()
    stems: Dict[str, np.ndarray] = {}
    source_names = model.sources
    for i, name in enumerate(source_names):
        stem_audio = sources[i].cpu().numpy()
        stem_mono = np.mean(stem_audio, axis=0)
        stems[name] = stem_mono

    return stems


# ------------------------------------------------------------
# Stage A: load_and_preprocess
# ------------------------------------------------------------

def load_and_preprocess(
    audio_path: str,
    config: StageAConfig = PIANO_61KEY_CONFIG.stage_a,
    fast_mode: bool = False,
) -> StageAOutput:
    """
    Stage A: Signal Conditioning for 61-key piano transcription.

    WI-aligned steps:
    1. Load audio (preserve original SR, then resample to target)
    2. Channel handling (mono sum / left / right)
    3. DC offset removal
    4. Transient pre-emphasis (hammer clicks)
    5. High-pass filter at ~55 Hz
    6. Silence trimming at top_db=50
    7. EBU R128 loudness normalization to -23 LUFS
    8. Optional peak limiter (soft/hard to ceiling)
    9. Noise floor estimation (percentile of RMS)
    10. BPM / beat grid detection
    11. Classify audio type + quality
    12. Return StageAOutput with a single 'mix' stem; separation is handled in Stage B.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # -----------------------------
    # 1. Load audio, then resample
    # -----------------------------
    target_sr = config.target_sample_rate
    try:
        # Load at original SR first (sr=None), then resample
        y, orig_sr = librosa.load(audio_path, sr=None, mono=False)
    except Exception:
        y, orig_sr = sf.read(audio_path)
        if y.ndim == 2:
            y = y.T

    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    sr = target_sr

    # -----------------------------
    # 2. Channel Handling
    # -----------------------------
    n_channels = 1
    if y.ndim > 1:
        n_channels = y.shape[0]
        mode = config.channel_handling
        if mode == "mono_sum":
            y = np.mean(y, axis=0)
        elif mode == "left_only":
            y = y[0]
        elif mode == "right_only":
            y = y[1]

    # Ensure 1D
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # -----------------------------
    # 3. DC Offset Removal
    # -----------------------------
    if config.dc_offset_removal:
        y = y - np.mean(y)

    # -----------------------------
    # 4. Transient Pre-emphasis
    # -----------------------------
    if config.transient_pre_emphasis.get("enabled", False):
        alpha = float(config.transient_pre_emphasis.get("alpha", 0.97))
        # y[t] = x[t] - alpha * x[t-1]
        y = scipy.signal.lfilter([1.0, -alpha], [1.0], y)

    # -----------------------------
    # 5. High-Pass Filter (~55 Hz)
    # -----------------------------
    cutoff = float(config.high_pass_filter_cutoff.get("value", 55.0))
    order = int(config.high_pass_filter_order.get("value", 4))
    sos = scipy.signal.butter(order, cutoff, btype="hp", fs=sr, output="sos")
    y = scipy.signal.sosfilt(sos, y)

    # -----------------------------
    # 6. Silence Trimming
    # -----------------------------
    if config.silence_trimming.get("enabled", True):
        top_db = float(config.silence_trimming.get("top_db", 50))
        y_trimmed, trim_idx = librosa.effects.trim(y, top_db=top_db)
    else:
        y_trimmed = y

    # -----------------------------
    # 7. Loudness Normalization
    # -----------------------------
    norm_gain = 0.0
    y_norm = y_trimmed

    if config.loudness_normalization.get("enabled", True):
        target_lufs = float(config.loudness_normalization.get("target_lufs", -23.0))
        meter = pyln.Meter(sr)
        try:
            if len(y_trimmed) / sr >= 0.4:
                loudness = meter.integrated_loudness(y_trimmed)
                if not (np.isneginf(loudness) or loudness < -70.0):
                    y_norm = pyln.normalize.loudness(y_trimmed, loudness, target_lufs)
                    norm_gain = target_lufs - loudness
        except Exception as e:
            warnings.warn(f"Loudness norm failed: {e}")
            y_norm = y_trimmed

    # -----------------------------
    # 7b. Peak Limiter (optional)
    # -----------------------------
    pl_conf = getattr(config, "peak_limiter", {"enabled": False})
    if pl_conf.get("enabled", False):
        mode = pl_conf.get("mode", "soft")        # "soft" | "hard"
        ceiling_db = float(pl_conf.get("ceiling_db", -1.0))
        ceiling_lin = 10.0 ** (ceiling_db / 20.0)

        if mode == "soft":
            # Simple tanh soft clip
            y_norm = np.tanh(y_norm / (ceiling_lin + 1e-9)) * ceiling_lin
        else:
            # Hard clip
            y_norm = np.clip(y_norm, -ceiling_lin, ceiling_lin)

    # -----------------------------
    # 8. Noise Floor Estimation
    # -----------------------------
    percentile = float(config.noise_floor_estimation.get("percentile", 30))
    frame_len = 2048
    hop = 1024
    if len(y_norm) >= frame_len:
        rms = librosa.feature.rms(y=y_norm, frame_length=frame_len, hop_length=hop)[0]
        noise_floor_rms = float(np.percentile(rms, percentile))
        noise_floor_db = 20.0 * np.log10(noise_floor_rms + 1e-9)
    else:
        noise_floor_rms = 0.0
        noise_floor_db = -80.0

    # -----------------------------
    # 9. BPM / Beat Grid
    # -----------------------------
    bpm = None
    beats: np.ndarray | list[float] = []
    if config.bpm_detection.get("enabled", True):
        try:
            bpm_val, beat_frames = librosa.beat.beat_track(y=y_norm, sr=sr)
            bpm = float(bpm_val)
            beats = librosa.frames_to_time(beat_frames, sr=sr)
        except Exception:
            bpm = None
            beats = []

    # -----------------------------
    # 10. Classification & Quality
    # -----------------------------
    audio_type = detect_audio_type(y_norm, sr)
    ext = os.path.splitext(audio_path)[1].lower()
    quality = AudioQuality.LOSSY if ext in [".mp3", ".ogg", ".m4a"] else AudioQuality.LOSSLESS

    # -----------------------------
    # 11. Build Stems (Stage A)
    # -----------------------------
    stems_output: Dict[str, Stem] = {
        "mix": Stem(audio=y_norm, sr=sr, type="mix")
    }
    # Separation into vocals/bass/drums/other is done in Stage B via apply_demucs()

    # -----------------------------
    # 12. MetaData
    # -----------------------------
    rms_db = -80.0
    if len(y_norm) > 0:
        rms_val = float(np.sqrt(np.mean(y_norm ** 2)))
        if rms_val > 0:
            rms_db = 20.0 * np.log10(rms_val)

    # Default tempo if detection failed
    tempo_bpm = float(bpm) if bpm is not None else 120.0

    meta = MetaData(
        tuning_offset=0.0,
        detected_key="C",  # can be updated later by key detection
        lufs=float(config.loudness_normalization.get("target_lufs", -23.0)),
        processing_mode=audio_type.value,
        audio_type=audio_type,
        audio_quality=quality,
        snr=0.0,  # placeholder if you later compute SNR
        window_size=2048,
        hop_length=512,  # global analysis hop (Stage B uses meta.hop_length)
        sample_rate=sr,
        tempo_bpm=tempo_bpm,
        time_signature="4/4",

        original_sr=orig_sr,
        target_sr=target_sr,
        duration_sec=float(len(y_norm) / sr),

        # beat grid lives both in MetaData and StageAOutput
        beats=[float(b) for b in beats],

        audio_path=audio_path,
        n_channels=n_channels,
        normalization_gain_db=norm_gain,
        rms_db=rms_db,
        noise_floor_rms=noise_floor_rms,
        noise_floor_db=noise_floor_db,
        pipeline_version="2.0.0",
    )

    # -----------------------------
    # 13. StageAOutput
    # -----------------------------
    return StageAOutput(
        stems=stems_output,
        meta=meta,
        audio_type=audio_type,
        noise_floor_rms=noise_floor_rms,
        noise_floor_db=noise_floor_db,
        beats=[float(b) for b in beats],
    )
