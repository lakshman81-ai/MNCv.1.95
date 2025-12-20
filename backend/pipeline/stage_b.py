"""
Stage B — Feature Extraction

This module implements pitch detection and feature extraction.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import logging
import importlib.util
import sys
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None

from .models import StageAOutput, FramePitch, AnalysisData, AudioType, StageBOutput, Stem
from .config import PipelineConfig
from .detectors import (
    SwiftF0Detector, SACFDetector, YinDetector,
    CQTDetector, RMVPEDetector, CREPEDetector,
    iterative_spectral_subtraction, create_harmonic_mask,
    _frame_audio,
    BasePitchDetector
)

# PB0: Fix imports
from copy import deepcopy

logger = logging.getLogger(__name__)

# Re-export for tests
__all__ = [
    "extract_features",
    "create_harmonic_mask",
    "iterative_spectral_subtraction",
    "MultiVoiceTracker",
]

SCIPY_SIGNAL = None
if importlib.util.find_spec("scipy.signal"):
    import scipy.signal as SCIPY_SIGNAL


def _module_available(module_name: str) -> bool:
    """Helper to avoid importing heavy optional deps when missing."""
    mod = sys.modules.get(module_name)
    if mod is not None and getattr(mod, "__spec__", None) is None:
        return False

    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ValueError):
        spec = None

    if spec is None and module_name in sys.modules:
        return True
    return spec is not None


def _butter_filter(audio: np.ndarray, sr: int, cutoff: float, btype: str) -> np.ndarray:
    """Lightweight wrapper for simple Butterworth filtering."""
    if SCIPY_SIGNAL is None or len(audio) == 0:
        return audio.copy()

    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    norm_cutoff = min(max(norm_cutoff, 1e-4), 0.999)
    sos = SCIPY_SIGNAL.butter(4, norm_cutoff, btype=btype, output="sos")
    return SCIPY_SIGNAL.sosfiltfilt(sos, audio)


def _estimate_global_tuning_cents(f0: np.ndarray) -> float:
    """
    Estimate global tuning offset from standard A440 in cents.
    Returns value in [-50, 50].
    """
    # Estimate median fractional MIDI deviation (in cents)
    f = np.asarray(f0, dtype=np.float32)
    f = f[f > 0.0]
    if f.size < 50:
        return 0.0
    midi_float = 69.0 + 12.0 * np.log2(f / 440.0)
    frac = midi_float - np.round(midi_float)
    cents = frac * 100.0
    # wrap to [-50, 50]
    cents = (cents + 50.0) % 100.0 - 50.0
    return float(np.median(cents))


class SyntheticMDXSeparator:
    """
    Lightweight separator tuned on procedurally generated sine/saw/square/FM stems.

    The separator derives template spectral envelopes from analytic waveforms, then
    infers soft weights that are used to project the incoming mix into vocal/bass/
    drums/other stems. This intentionally mirrors a tiny MDX head without external
    weights so it can run in constrained environments.
    """

    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.templates = self._build_templates()

    def _build_templates(self) -> Dict[str, np.ndarray]:
        sr = self.sample_rate
        duration = 0.25
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        base_freqs = [110.0, 220.0, 440.0, 660.0]

        def _normalize_spec(y: np.ndarray) -> np.ndarray:
            window = np.hanning(len(y))
            spec = np.abs(np.fft.rfft(y * window))
            spec = spec / (np.linalg.norm(spec) + 1e-9)
            return spec

        templates: Dict[str, np.ndarray] = {}
        waves = {
            "sine_stack": sum(np.sin(2 * np.pi * f * t) for f in base_freqs),
            "saw": sum(1.0 / (i + 1) * np.sin(2 * np.pi * (i + 1) * base_freqs[1] * t) for i in range(6)),
            "square": sum(
                1.0 / (2 * i + 1) * np.sin(2 * np.pi * (2 * i + 1) * base_freqs[0] * t)
                for i in range(6)
            ),
        }

        # Simple FM voice to emulate vocal richness
        carrier = 220.0
        modulator = 110.0
        waves["fm_voice"] = np.sin(
            2 * np.pi * carrier * t + 5.0 * np.sin(2 * np.pi * modulator * t)
        )

        for name, wave in waves.items():
            templates[name] = _normalize_spec(wave)

        # Broadband template for drums/transients
        broadband = np.hanning(len(t))
        templates["broadband"] = _normalize_spec(broadband)
        return templates

    def _score_mix(self, audio: np.ndarray) -> Dict[str, float]:
        window = np.hanning(len(audio))
        spec = np.abs(np.fft.rfft(audio * window))
        spec = spec / (np.linalg.norm(spec) + 1e-9)

        scores = {}
        for name, tmpl in self.templates.items():
            # cosine similarity with length alignment so template granularity
            # doesn't depend on the input duration
            if len(tmpl) != len(spec):
                x_old = np.linspace(0.0, 1.0, len(tmpl))
                x_new = np.linspace(0.0, 1.0, len(spec))
                tmpl = np.interp(x_new, x_old, tmpl)
                tmpl = tmpl / (np.linalg.norm(tmpl) + 1e-9)

            score = float(np.dot(spec, tmpl))
            scores[name] = score
        return scores

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        if len(audio) == 0:
            return {}

        scores = self._score_mix(audio)
        vocal_score = scores.get("fm_voice", 0.25) + scores.get("sine_stack", 0.25)
        # Assign Sawtooth score to Bass (L2 uses Saw for Bass)
        bass_score = scores.get("square", 0.25) + scores.get("saw", 0.25)

        drum_score = scores.get("broadband", 0.25)

        # Reduce Saw score from 'other' (or remove it from weighted calculation)
        # But we need 4 weights for the tuple unpacking below?
        # Actually 'other_w' comes from 'saw_score' in original code.
        # Let's keep 4 dimensions but repurpose.
        # Original: vocals, bass, drums, other (from saw)
        # New: vocals, bass (square+saw), drums, other (residual or 0?)

        # To keep tuple unpacking safe:
        # We'll just set 'saw_score' to small epsilon if we moved it to bass?
        # Or better: let 'other' be low.
        other_score = 0.01

        raw_weights = np.array([vocal_score, bass_score, drum_score, other_score])
        weights = raw_weights / (np.sum(raw_weights) + 1e-9)
        vocals_w, bass_w, drums_w, other_w = weights

        if SCIPY_SIGNAL is None:
             logger.warning("Scipy signal not available; SyntheticMDXSeparator filtering disabled.")

        vocals = vocals_w * _butter_filter(audio, sr, 12000.0, "low")
        vocals = _butter_filter(vocals, sr, 300.0, "high")

        bass = bass_w * _butter_filter(audio, sr, 180.0, "low")
        drums = drums_w * _butter_filter(audio, sr, 90.0, "high")
        other = audio - (vocals + bass + drums)
        other = other_w * other

        return {
            "vocals": vocals,
            "bass": bass,
            "drums": drums,
            "other": other,
        }


def _run_htdemucs(
    audio: np.ndarray,
    sr: int,
    model_name: str,
    overlap: float,
    shifts: int,
    device: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    if (
        not _module_available("demucs.pretrained")
        or not _module_available("demucs.apply")
        or not _module_available("torch")
    ):
        logger.warning("Demucs not available; skipping neural separation.")
        return None

    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torch

    # Resolve device
    try:
        dev = torch.device(device) if device else torch.device("cpu")
    except Exception:
        dev = torch.device("cpu")

    try:
        model = get_model(model_name)
        model.to(dev)
    except Exception as exc:
        logger.warning(f"HTDemucs unavailable ({exc}); skipping neural separation.")
        return None

    model_sr = getattr(model, "samplerate", sr)

    # Handle Mono vs Stereo input logic
    # Demucs expects (Channels, Time)
    # If input is (Time,), make it (1, Time) first for logic consistency
    if audio.ndim == 1:
        audio = audio[None, :]  # (1, T)

    # Resample if needed
    if model_sr != sr:
        import torch.nn.functional as F
        # Use torch for resampling if available to handle channels correctly
        # or use scipy/numpy carefully.
        # Simple numpy interp works for 1D, but for (C, T) we need loop.
        # Let's stick to numpy for minimal deps, but handle channels.
        ratio = float(model_sr) / float(sr)
        new_len = int(audio.shape[-1] * ratio)
        resampled_channels = []
        for ch in range(audio.shape[0]):
            indices = np.arange(0, new_len) / ratio
            # clamp indices to valid range
            indices = np.clip(indices, 0, audio.shape[-1] - 1)
            res_ch = np.interp(indices, np.arange(audio.shape[-1]), audio[ch])
            resampled_channels.append(res_ch)
        resampled = np.stack(resampled_channels)
    else:
        resampled = audio

    # Demucs expects stereo (2, T)
    # If mono (1, T), duplicate. If > 2, trim.
    C, T = resampled.shape
    if C == 1:
        resampled = np.concatenate([resampled, resampled], axis=0)
    elif C > 2:
        resampled = resampled[:2, :]

    # Prepare tensor: (Batch, Channels, Time) -> (1, 2, T)
    mix_tensor = torch.tensor(resampled, dtype=torch.float32)[None, :, :].to(dev)
    try:
        with torch.no_grad():
            demucs_out = apply_model(model, mix_tensor, overlap=overlap, shifts=shifts, device=dev)
    except Exception as exc:
        logger.warning(f"HTDemucs inference failed ({exc}); skipping neural separation.")
        return None

    sources = getattr(model, "sources", ["vocals", "drums", "bass", "other"])
    separated = {}
    for idx, name in enumerate(sources):
        stem_audio = demucs_out[0, idx].mean(dim=0).cpu().numpy()
        separated[name] = stem_audio

    # Ensure canonical stems exist
    for name in ["vocals", "drums", "bass", "other"]:
        separated.setdefault(name, np.zeros_like(audio))

    return separated


def _resolve_separation(stage_a_out: StageAOutput, b_conf, device: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Resolve separation strategy and track which path actually executed.
    Returns (stems, diagnostics).
    """
    diag = {
        "requested": bool(b_conf.separation.get("enabled", True)),
        "synthetic_requested": bool(b_conf.separation.get("synthetic_model", False)),
        "mode": "disabled",
        "synthetic_ran": False,
        "htdemucs_ran": False,
        "fallback": False,
        "preset": None,
        "resolved_overlap": None,
        "resolved_shifts": None,
        "shift_range": None,
    }

    if not diag["requested"]:
        return stage_a_out.stems, diag

    if len(stage_a_out.stems) > 1 and any(k != "mix" for k in stage_a_out.stems.keys()):
        diag["mode"] = "preseparated"
        return stage_a_out.stems, diag

    mix_stem = stage_a_out.stems.get("mix")
    if mix_stem is None:
        return stage_a_out.stems, diag

    sep_conf = b_conf.separation
    overlap = sep_conf.get("overlap", 0.25)
    shifts = sep_conf.get("shifts", 1)

    preset_conf: Dict[str, Any] = {}
    if getattr(stage_a_out, "audio_type", None) == AudioType.POLYPHONIC_DOMINANT:
        preset_conf = sep_conf.get("polyphonic_dominant_preset", {}) or {}
        diag["preset"] = "polyphonic_dominant"
        overlap = float(preset_conf.get("overlap", overlap))
        if preset_conf.get("shift_range"):
            diag["shift_range"] = list(preset_conf.get("shift_range"))
        if "shifts" in preset_conf:
            shifts = int(preset_conf.get("shifts", shifts))
        else:
            shift_range = preset_conf.get("shift_range")
            if shift_range and isinstance(shift_range, (list, tuple)):
                try:
                    shifts = int(max(shift_range))
                except (TypeError, ValueError):
                    shifts = int(shifts)

    diag["resolved_overlap"] = overlap
    diag["resolved_shifts"] = shifts

    if diag["synthetic_requested"]:
        synthetic = SyntheticMDXSeparator(sample_rate=mix_stem.sr, hop_length=stage_a_out.meta.hop_length)
        try:
            synthetic_stems = synthetic.separate(mix_stem.audio, mix_stem.sr)
            if synthetic_stems:
                diag.update({"mode": "synthetic_mdx", "synthetic_ran": True})
                return {
                    name: type(mix_stem)(audio=audio, sr=mix_stem.sr, type=name)
                    for name, audio in synthetic_stems.items()
                } | {"mix": mix_stem}, diag
        except Exception as exc:
            logger.warning(
                f"Synthetic separator failed; falling back to {sep_conf.get('model', 'htdemucs')}: {exc}"
            )
            diag["fallback"] = True

    separated = _run_htdemucs(
        mix_stem.audio,
        mix_stem.sr,
        sep_conf.get("model", "htdemucs"),
        overlap,
        shifts,
        device=device,
    )

    if separated:
        diag.update({"mode": sep_conf.get("model", "htdemucs"), "htdemucs_ran": True})
        return {
            name: type(mix_stem)(audio=audio, sr=mix_stem.sr, type=name)
            for name, audio in separated.items()
        } | {"mix": mix_stem}, diag

    diag["mode"] = "passthrough"
    return stage_a_out.stems, diag

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


def _median_filter(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return np.asarray(signal, dtype=np.float32)
    k = int(max(1, kernel_size))
    if k % 2 == 0:
        k += 1
    if SCIPY_SIGNAL is not None and hasattr(SCIPY_SIGNAL, "medfilt"):
        return np.asarray(SCIPY_SIGNAL.medfilt(signal, kernel_size=k), dtype=np.float32)

    pad = k // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    filtered = [np.median(padded[i : i + k]) for i in range(len(signal))]
    return np.asarray(filtered, dtype=np.float32)


def _apply_melody_filters(
    f0: np.ndarray,
    conf: np.ndarray,
    rms: Optional[np.ndarray],
    filter_conf: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    f0_out = np.asarray(f0, dtype=np.float32)
    conf_out = np.asarray(conf, dtype=np.float32)
    raw_conf = conf_out.copy()

    if f0_out.size == 0:
        return f0_out, conf_out

    median_win = int(filter_conf.get("median_window", 0) or 0)
    if median_win > 1:
        f0_out = _median_filter(f0_out, median_win)

    voiced_thr = float(filter_conf.get("voiced_prob_threshold", 0.0) or 0.0)
    if voiced_thr > 0.0:
        conf_out = np.where(conf_out >= voiced_thr, conf_out, 0.0)
        if not np.any(conf_out):
            relaxed = voiced_thr * 0.7
            conf_out = np.where(raw_conf >= relaxed, raw_conf, 0.0)

    if rms is not None and rms.size:
        rms_gate_db = float(filter_conf.get("rms_gate_db", -40.0))
        rms_gate = 10 ** (rms_gate_db / 20.0)
        conf_out = np.where(rms >= rms_gate, conf_out, 0.0)

    fmin = float(filter_conf.get("fmin_hz", 0.0) or 0.0)
    fmax = float(filter_conf.get("fmax_hz", 0.0) or 0.0)
    if fmin > 0.0:
        conf_out = np.where(f0_out >= fmin, conf_out, 0.0)
    if fmax > 0.0:
        conf_out = np.where(f0_out <= fmax, conf_out, 0.0)

    if not np.any(conf_out) and raw_conf.size:
        conf_out = raw_conf
        if fmin > 0.0:
            conf_out = np.where(f0_out >= fmin, conf_out, 0.0)
        if fmax > 0.0:
            conf_out = np.where(f0_out <= fmax, conf_out, 0.0)

    f0_out = np.where(conf_out > 0.0, f0_out, 0.0)
    return f0_out, conf_out

def _init_detector(name: str, conf: Dict[str, Any], sr: int, hop_length: int) -> Optional[BasePitchDetector]:
    """Initialize a detector if enabled."""
    if not conf.get("enabled", False):
        return None

    # Remove control/meta keys we already pass positionally
    kwargs = {k: v for k, v in conf.items() if k not in ("enabled", "hop_length")}

    # 61-key fallback defaults if not provided - B1
    kwargs.setdefault("fmin", 60.0)
    kwargs.setdefault("fmax", 2200.0)

    try:
        if name == "swiftf0":
            return SwiftF0Detector(sr, hop_length, **kwargs)
        elif name == "sacf":
            return SACFDetector(sr, hop_length, **kwargs)
        elif name == "yin":
            return YinDetector(sr, hop_length, **kwargs)
        elif name == "cqt":
            return CQTDetector(sr, hop_length, **kwargs)
        elif name == "rmvpe":
            return RMVPEDetector(sr, hop_length, **kwargs)
        elif name == "crepe":
            return CREPEDetector(sr, hop_length, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to init detector {name}: {e}")
        return None
    return None

def _ensemble_merge(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    weights: Dict[str, float],
    disagreement_cents: float = 70.0,
    priority_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple f0/conf tracks based on weights and disagreement.

    Strategy:
      * Align frame counts across detectors
      * Choose the candidate with the highest weighted confidence
      * Down-weight winners that have little consensus (large disagreement)
    """
    if not results:
        return np.array([]), np.array([])

    lengths = [len(r[0]) for r in results.values()]
    if not lengths:
        return np.array([]), np.array([])
    max_len = max(lengths)

    aligned_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, (f0, conf) in results.items():
        if len(f0) < max_len:
            pad = max_len - len(f0)
            f0 = np.pad(f0, (0, pad))
            conf = np.pad(conf, (0, pad))
        aligned_results[name] = (f0, conf)

    final_f0 = np.zeros(max_len, dtype=np.float32)
    final_conf = np.zeros(max_len, dtype=np.float32)

    def _cent_diff(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return float("inf")
        return float(1200.0 * np.log2((a + 1e-9) / (b + 1e-9)))

    for i in range(max_len):
        candidates = []
        for name, (f0, conf) in aligned_results.items():
            w = weights.get(name, 1.0)
            c = float(conf[i])
            f = float(f0[i])
            if c <= 0.0 or f <= 0.0:
                continue

            # Priority floor mostly benefits SwiftF0 on synthetic tones
            eff_conf = max(c, priority_floor if name == "swiftf0" else c)
            candidates.append((name, f, eff_conf, w))

        if not candidates:
            final_f0[i] = 0.0
            final_conf[i] = 0.0
            continue

        # Pick winner by weighted confidence
        best_name, best_f0, best_conf, best_w = max(
            candidates, key=lambda x: x[2] * x[3]
        )

        # Consensus weighting: measure how many other detectors agree
        total_w = sum(c[3] for c in candidates)
        support_w = best_w
        for name, f, c, w in candidates:
            if name == best_name:
                continue
            if abs(_cent_diff(f, best_f0)) <= float(disagreement_cents):
                support_w += w

        consensus = support_w / max(total_w, 1e-6)
        final_f0[i] = best_f0
        final_conf[i] = best_conf * consensus

    return final_f0, final_conf

def _is_polyphonic(audio_type: Any) -> bool:
    """Check if Stage A classified audio as polyphonic."""
    try:
        if isinstance(audio_type, AudioType):
            return audio_type in (AudioType.POLYPHONIC, AudioType.POLYPHONIC_DOMINANT)
        if isinstance(audio_type, str):
            return "poly" in audio_type.lower()
    except Exception:
        pass
    return False


def _augment_with_harmonic_masks(
    stem: Stem,
    prior_detector: BasePitchDetector,
    mask_width: float,
    n_harmonics: int,
    audio_path: Optional[str] = None,
) -> Dict[str, Stem]:
    """
    Derive synthetic melody/accompaniment stems by masking harmonics from a quick f0 prior.
    """
    audio = np.asarray(stem.audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {}

    try:
        f0, conf = prior_detector.predict(audio, audio_path=audio_path)
        hop = getattr(prior_detector, "hop_length", 512)
        n_fft = getattr(prior_detector, "n_fft", 2048)
        n_fft_eff = int(min(n_fft, max(32, len(audio))))
        hop_eff = int(max(1, min(hop, n_fft_eff // 2)))

        f, t, Z = scipy.signal.stft(
            audio,
            fs=stem.sr,
            nperseg=n_fft_eff,
            noverlap=max(0, n_fft_eff - hop_eff),
            boundary="zeros",
            padded=True,
        )

        n_frames = Z.shape[1]
        if f0.shape[0] != n_frames:
            if f0.shape[0] < n_frames:
                pad = n_frames - f0.shape[0]
                f0 = np.pad(f0, (0, pad))
                conf = np.pad(conf, (0, pad))
            else:
                f0 = f0[:n_frames]
                conf = conf[:n_frames]

        mask = create_harmonic_mask(
            f0_hz=f0,
            sr=stem.sr,
            n_fft=n_fft,
            mask_width=mask_width,
            n_harmonics=n_harmonics,
        )

        # Harmonic emphasis keeps bins near f0; residual keeps the rest.
        strength = np.clip(conf, 0.0, 1.0).reshape(1, -1)
        harmonic_keep = np.clip((1.0 - mask) * (0.8 + 0.2 * strength), 0.0, 1.0)
        residual_keep = np.clip(1.0 - harmonic_keep, 0.0, 1.0)

        Z_melody = Z * harmonic_keep
        Z_resid = Z * residual_keep

        _, melody_audio = scipy.signal.istft(
            Z_melody,
            fs=stem.sr,
            nperseg=n_fft_eff,
            noverlap=max(0, n_fft_eff - hop_eff),
            input_onesided=True,
            boundary="zeros",
        )
        _, residual_audio = scipy.signal.istft(
            Z_resid,
            fs=stem.sr,
            nperseg=n_fft_eff,
            noverlap=max(0, n_fft_eff - hop_eff),
            input_onesided=True,
            boundary="zeros",
        )

        melody_audio = np.asarray(melody_audio, dtype=np.float32)
        residual_audio = np.asarray(residual_audio, dtype=np.float32)

        # Match original length
        if melody_audio.size < audio.size:
            melody_audio = np.pad(melody_audio, (0, audio.size - melody_audio.size))
        melody_audio = melody_audio[: audio.size]

        if residual_audio.size < audio.size:
            residual_audio = np.pad(residual_audio, (0, audio.size - residual_audio.size))
        residual_audio = residual_audio[: audio.size]

        return {
            "melody_masked": Stem(audio=melody_audio, sr=stem.sr, type="melody_masked"),
            "residual_masked": Stem(audio=residual_audio, sr=stem.sr, type="residual_masked"),
        }
    except Exception:
        # Do not let masking failure break the pipeline
        return {}


def _resolve_polyphony_filter(config: Optional[PipelineConfig]) -> str:
    try:
        return str(config.stage_c.polyphony_filter.get("mode", "skyline_top_voice"))
    except Exception:
        return "skyline_top_voice"


class MultiVoiceTracker:
    """
    Lightweight multi-voice tracker to keep skyline assignments stable.

    Tracks up to `max_tracks` concurrent voices using a Hungarian assignment on
    pitch proximity with hangover/hysteresis to avoid rapid swapping.
    """

    def __init__(
        self,
        max_tracks: int,
        max_jump_cents: float = 150.0,
        hangover_frames: int = 2,
        smoothing: float = 0.35,
        confidence_bias: float = 5.0,
    ) -> None:
        self.max_tracks = max_tracks
        self.max_jump_cents = float(max_jump_cents)
        self.hangover_frames = int(max(0, hangover_frames))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.confidence_bias = float(confidence_bias)
        self.prev_pitches = np.zeros(max_tracks, dtype=np.float32)
        self.prev_confs = np.zeros(max_tracks, dtype=np.float32)
        self.hold = np.zeros(max_tracks, dtype=np.int32)

    def _pitch_cost(self, prev: float, candidate: float) -> float:
        if prev <= 0.0 or candidate <= 0.0:
            return 0.0
        cents = abs(1200.0 * np.log2((candidate + 1e-6) / (prev + 1e-6)))
        penalty = self.max_jump_cents if cents > self.max_jump_cents else 0.0
        return cents + penalty

    def _assign(self, pitches: np.ndarray, confs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Fallback to greedy ordering when Hungarian solver missing
        if linear_sum_assignment is None or pitches.size == 0:
            ordered = sorted(zip(pitches, confs), key=lambda x: (-x[1], x[0]))
            ordered = ordered[: self.max_tracks]
            new_pitches = np.zeros_like(self.prev_pitches)
            new_confs = np.zeros_like(self.prev_confs)
            for idx, (p, c) in enumerate(ordered):
                new_pitches[idx] = p
                new_confs[idx] = c
            return new_pitches, new_confs

        cost = np.zeros((self.max_tracks, pitches.size), dtype=np.float32)
        for i in range(self.max_tracks):
            for j in range(pitches.size):
                pitch_cost = self._pitch_cost(float(self.prev_pitches[i]), float(pitches[j]))
                cost[i, j] = pitch_cost - float(confs[j]) * self.confidence_bias

        row_idx, col_idx = linear_sum_assignment(cost)
        new_pitches = np.zeros_like(self.prev_pitches)
        new_confs = np.zeros_like(self.prev_confs)
        for r, c in zip(row_idx, col_idx):
            new_pitches[r] = pitches[c]
            new_confs[r] = confs[c]
        return new_pitches, new_confs

    def step(self, candidates: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        if not candidates:
            # Apply hangover to keep tracks alive briefly
            carry_pitches = np.where(self.hold > 0, self.prev_pitches, 0.0)
            carry_confs = np.where(self.hold > 0, self.prev_confs * 0.9, 0.0)
            self.hold = np.maximum(self.hold - 1, 0)
            self.prev_pitches = carry_pitches.astype(np.float32)
            self.prev_confs = carry_confs.astype(np.float32)
            return self.prev_pitches.copy(), self.prev_confs.copy()

        # Keep only the strongest candidates up to track count
        ordered = sorted(candidates, key=lambda x: (-x[1], x[0]))[: self.max_tracks]
        pitches = np.array([c[0] for c in ordered], dtype=np.float32)
        confs = np.array([c[1] for c in ordered], dtype=np.float32)

        assigned_pitches, assigned_confs = self._assign(pitches, confs)

        updated_pitches = np.zeros_like(self.prev_pitches)
        updated_confs = np.zeros_like(self.prev_confs)
        for idx in range(self.max_tracks):
            if assigned_pitches[idx] > 0.0:
                if self.prev_pitches[idx] > 0.0:
                    smoothed = (
                        self.smoothing * float(self.prev_pitches[idx])
                        + (1.0 - self.smoothing) * float(assigned_pitches[idx])
                    )
                else:
                    smoothed = float(assigned_pitches[idx])
                updated_pitches[idx] = smoothed
                updated_confs[idx] = assigned_confs[idx]
                self.hold[idx] = self.hangover_frames
            elif self.hold[idx] > 0:
                updated_pitches[idx] = self.prev_pitches[idx]
                updated_confs[idx] = self.prev_confs[idx] * 0.85
                self.hold[idx] -= 1
            else:
                updated_pitches[idx] = 0.0
                updated_confs[idx] = 0.0

        self.prev_pitches = updated_pitches
        self.prev_confs = updated_confs
        return updated_pitches.copy(), updated_confs.copy()


def extract_features(
    stage_a_out: StageAOutput,
    config: Optional[PipelineConfig] = None,
    pipeline_logger: Optional[Any] = None,
    **kwargs
) -> StageBOutput:
    """
    Stage B: Extract pitch and features.
    Respects config.stage_b for detector selection, ensemble weights, and
    optional polyphonic peeling to expose multiple F0 layers.
    """
    if config is None:
        config = PipelineConfig()

    b_conf = config.stage_b
    sr = stage_a_out.meta.sample_rate
    hop_length = stage_a_out.meta.hop_length

    # Patch D1 / B0: Resolve instrument profile + apply overrides (without mutating config)
    # Priority: kwargs -> config intended instrument -> meta.instrument -> fallback
    instrument = (
        kwargs.get("instrument")
        or (getattr(config, "instrument_override", None) if config else None)
        or (getattr(config, "intended_instrument", None) if config else None)
        or getattr(stage_a_out.meta, "instrument", None)
        or b_conf.instrument
    )

    profile = config.get_profile(str(instrument)) if (instrument and b_conf.apply_instrument_profile) else None
    profile_special = dict(getattr(profile, "special", {}) or {}) if profile else {}
    profile_applied = bool(profile)

    # Work on copies so we don't mutate config dataclasses
    detector_cfgs = deepcopy(b_conf.detectors)
    weights_eff = dict(b_conf.ensemble_weights)
    melody_filter_eff = dict(getattr(b_conf, "melody_filtering", {}) or {})

    # Apply Optional nested overrides first
    nested = profile_special.get("stage_b_detectors") or {}
    for det_name, overrides in (nested.items() if isinstance(nested, dict) else []):
        if det_name in detector_cfgs and isinstance(overrides, dict):
            detector_cfgs[det_name].update(overrides)

    # Apply profile overrides and flat keys
    if profile:
        # Force instrument range everywhere
        for _, dconf in detector_cfgs.items():
            dconf.setdefault("fmin", float(profile.fmin))
            dconf.setdefault("fmax", float(profile.fmax))
            # If profile provides explicit range, we should arguably prioritize it,
            # but usually defaults are broader. Let's enforce profile range if set.
            dconf["fmin"] = float(profile.fmin)
            dconf["fmax"] = float(profile.fmax)

        melody_filter_eff["fmin_hz"] = float(profile.fmin)
        melody_filter_eff["fmax_hz"] = float(profile.fmax)

        # Ensure recommended algo is enabled
        rec = (profile.recommended_algo or "").lower()
        if rec and rec != "none" and rec in detector_cfgs:
            detector_cfgs[rec]["enabled"] = True

        # Map flat keys to structure
        if "yin_trough_threshold" in profile_special and "yin" in detector_cfgs:
             detector_cfgs["yin"]["trough_threshold"] = float(profile_special["yin_trough_threshold"])

        if "yin_conf_threshold" in profile_special and "yin" in detector_cfgs:
             detector_cfgs["yin"]["threshold"] = float(profile_special["yin_conf_threshold"])

        if "yin_frame_length" in profile_special and "yin" in detector_cfgs:
            detector_cfgs["yin"]["frame_length"] = int(profile_special["yin_frame_length"])

        # Violin-style vibrato smoothing
        if "vibrato_smoothing_ms" in profile_special:
            ms = float(profile_special["vibrato_smoothing_ms"])
            frame_ms = 1000.0 * float(hop_length) / float(sr)
            win = int(round(ms / max(frame_ms, 1e-6)))
            if win % 2 == 0:
                win += 1
            current = int(melody_filter_eff.get("median_window", 1) or 1)
            melody_filter_eff["median_window"] = max(current, win)

    # Separation routing happens before harmonic masking/ISS so downstream
    # detectors always see the requested stem layout.
    device = getattr(config, "device", "cpu")
    resolved_stems, separation_diag = _resolve_separation(stage_a_out, b_conf, device=device)

    # Patch OPT6: Apply active_stems filter
    whitelist = getattr(b_conf, "active_stems", None)
    if whitelist is not None:
        # Keep 'mix' always or check if user excluded it? Usually we keep mix for fallback.
        # But if whitelist is explicit, maybe we should respect it fully?
        # Safe bet: always keep mix if it exists, filter others.
        # Requirement says: "active_stems: Optional[List[str]] = None # e.g. ["bass", "vocals"]; None => all"

        filtered_stems = {}
        for sname, sobj in resolved_stems.items():
            if sname == "mix":
                filtered_stems[sname] = sobj
            elif sname in whitelist:
                filtered_stems[sname] = sobj
        resolved_stems = filtered_stems

    # 1. Initialize Detectors based on Config
    detectors: Dict[str, BasePitchDetector] = {}
    for name, det_conf in detector_cfgs.items():
        det = _init_detector(name, det_conf, sr, hop_length)
        if det:
            detectors[name] = det

    # Ensure baseline fallback if no detectors enabled/working
    if not detectors:
        logger.warning("No detectors enabled or initialized in Stage B. Falling back to default YIN/ACF.")
        detectors["yin"] = YinDetector(sr, hop_length)

    stem_timelines: Dict[str, List[FramePitch]] = {}
    per_detector: Dict[str, Any] = {}
    f0_main: Optional[np.ndarray] = None
    all_layers: List[np.ndarray] = []
    iss_total_layers = 0

    # Polyphonic context detection
    polyphonic_context = _is_polyphonic(getattr(stage_a_out, "audio_type", None))
    skyline_mode = _resolve_polyphony_filter(config)
    tracker_cfg = getattr(b_conf, "voice_tracking", {}) or {}

    # Optional harmonic masking to create synthetic melody/bass stems for synthetic material
    augmented_stems = dict(resolved_stems)
    harmonic_mask_applied = False
    harmonic_cfg = b_conf.separation.get("harmonic_masking", {}) if hasattr(b_conf, "separation") else {}
    if harmonic_cfg.get("enabled", False) and "mix" in augmented_stems:
        prior_det = detectors.get("swiftf0")
        if prior_det is None:
            prior_conf = dict(b_conf.detectors.get("swiftf0", {}))
            prior_conf["enabled"] = True
            prior_det = _init_detector("swiftf0", prior_conf, sr, hop_length)
        if prior_det is None:
            prior_det = detectors.get("yin") or _init_detector("yin", {"enabled": True}, sr, hop_length)

        if prior_det is not None:
            synthetic = _augment_with_harmonic_masks(
                augmented_stems["mix"],
                prior_det,
                mask_width=float(harmonic_cfg.get("mask_width", 0.025)),
                n_harmonics=int(harmonic_cfg.get("n_harmonics", 8)),
                audio_path=stage_a_out.meta.audio_path,
            )
            augmented_stems.update(synthetic)
            harmonic_mask_applied = bool(synthetic) or harmonic_cfg.get("enabled", False)

    stems_for_processing = augmented_stems
    polyphonic_context = polyphonic_context or len(stems_for_processing) > 1 or b_conf.polyphonic_peeling.get("force_on_mix", False)

    # 2. Process Stems
    mix_stem_ref = stage_a_out.stems.get("mix") or next(iter(stage_a_out.stems.values()))
    canonical_n_frames = int(np.ceil(len(mix_stem_ref.audio) / hop_length))

    for stem_name, stem in stems_for_processing.items():
        audio = stem.audio
        per_detector[stem_name] = {}

        # Patch D5: Ignore pitch for drums/percussion
        if profile_special.get("ignore_pitch", False):
            merged_f0 = np.zeros(canonical_n_frames, dtype=np.float32)
            merged_conf = np.zeros(canonical_n_frames, dtype=np.float32)
            stem_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        else:
            stem_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

            # Patch B4: Pre-LPF (distorted guitar) before detector predict
            # We filter a copy of audio for detection, preserving original for other uses
            audio_in = audio
            pre_lpf_hz = float(profile_special.get("pre_lpf_hz", 0.0) or 0.0)
            if pre_lpf_hz > 0.0 and SCIPY_SIGNAL is not None:
                audio_in = _butter_filter(np.asarray(audio, dtype=np.float32), sr, pre_lpf_hz, "low")

            # Run all initialized detectors on this stem
            for name, det in detectors.items():
                try:
                    # Prepare audio for this detector
                    audio_for_det = np.asarray(audio_in, dtype=np.float32)

                    f0, conf = det.predict(audio_for_det, audio_path=stage_a_out.meta.audio_path)

                    # B2: Apply stricter SwiftF0 confidence gate if configured
                    if name == "swiftf0":
                        thr = float(detector_cfgs.get("swiftf0", {}).get("confidence_threshold", 0.9))
                        conf = np.where(conf >= thr, conf, 0.0)
                        f0 = np.where(conf > 0.0, f0, 0.0)

                    stem_results[name] = (f0, conf)
                    per_detector[stem_name][name] = (f0, conf)
                except Exception as e:
                    logger.warning(f"Detector {name} failed on stem {stem_name}: {e}")

        # Ensemble Merge with disagreement and SwiftF0 priority floor
        if stem_results:
            merged_f0, merged_conf = _ensemble_merge(
                stem_results,
                b_conf.ensemble_weights,
                b_conf.pitch_disagreement_cents,
                b_conf.confidence_priority_floor,
            )
        else:
            merged_f0 = np.zeros(1, dtype=np.float32)
            merged_conf = np.zeros(1, dtype=np.float32)

        # TP2 Fix: Pad/Trim merged outputs to canonical length before downstream processing
        if len(merged_f0) != canonical_n_frames:
            if pipeline_logger:
                pipeline_logger.log_event(
                    "stage_b",
                    "detector_output_len_mismatch",
                    payload={
                        "stem": stem_name,
                        "expected": canonical_n_frames,
                        "got": len(merged_f0),
                        "action": "pad_or_trim"
                    }
                )

            if len(merged_f0) < canonical_n_frames:
                pad_len = canonical_n_frames - len(merged_f0)
                merged_f0 = np.pad(merged_f0, (0, pad_len), constant_values=0.0)
                merged_conf = np.pad(merged_conf, (0, pad_len), constant_values=0.0)
            else:
                merged_f0 = merged_f0[:canonical_n_frames]
                merged_conf = merged_conf[:canonical_n_frames]

        # B3: Global tuning offset
        # Store tuning offset in diagnostics per stem or for the first significant stem
        tuning_cents = 0.0
        # For simplicity and requirement, compute for current stem (merged_f0)
        # We'll store it in diagnostics later
        tuning_cents = _estimate_global_tuning_cents(merged_f0)

        # Polyphonic peeling (ISS) – optional and gated by config + context
        iss_layers: List[Tuple[np.ndarray, np.ndarray]] = []
        if polyphonic_context and b_conf.polyphonic_peeling.get("max_layers", 0) > 0:
            primary = detectors.get("swiftf0") or detectors.get("yin") or detectors.get("sacf")
            validator = detectors.get("sacf") or detectors.get("yin")
            if primary:
                try:
                    iss_layers = iterative_spectral_subtraction(
                        audio,
                        sr,
                        primary_detector=primary,
                        validator_detector=validator,
                        max_polyphony=b_conf.polyphonic_peeling.get("max_layers", 4),
                        mask_width=b_conf.polyphonic_peeling.get("mask_width", 0.03),
                        min_mask_width=b_conf.polyphonic_peeling.get("min_mask_width", 0.02),
                        max_mask_width=b_conf.polyphonic_peeling.get("max_mask_width", 0.08),
                        mask_growth=b_conf.polyphonic_peeling.get("mask_growth", 1.1),
                        mask_shrink=b_conf.polyphonic_peeling.get("mask_shrink", 0.9),
                        harmonic_snr_stop_db=b_conf.polyphonic_peeling.get("harmonic_snr_stop_db", 3.0),
                        residual_rms_stop_ratio=b_conf.polyphonic_peeling.get("residual_rms_stop_ratio", 0.08),
                        residual_flatness_stop=b_conf.polyphonic_peeling.get("residual_flatness_stop", 0.45),
                        validator_cents_tolerance=b_conf.polyphonic_peeling.get("validator_cents_tolerance", b_conf.pitch_disagreement_cents),
                        validator_agree_window=b_conf.polyphonic_peeling.get("validator_agree_window", 5),
                        validator_disagree_decay=b_conf.polyphonic_peeling.get("validator_disagree_decay", 0.6),
                        validator_min_agree_frames=b_conf.polyphonic_peeling.get("validator_min_agree_frames", 2),
                        validator_min_disagree_frames=b_conf.polyphonic_peeling.get("validator_min_disagree_frames", 2),
                        max_harmonics=b_conf.polyphonic_peeling.get("max_harmonics", 12),
                        audio_path=stage_a_out.meta.audio_path,
                    iss_adaptive=b_conf.polyphonic_peeling.get("iss_adaptive", False),
                    strength_min=b_conf.polyphonic_peeling.get("strength_min", 0.8),
                    strength_max=b_conf.polyphonic_peeling.get("strength_max", 1.2),
                    flatness_thresholds=b_conf.polyphonic_peeling.get("flatness_thresholds", [0.3, 0.6]),
                    )
                    all_layers.extend([f0 for f0, _ in iss_layers])
                    iss_total_layers += len(iss_layers)
                except Exception as e:
                    logger.warning(f"ISS peeling failed for stem {stem_name}: {e}")

        # Calculate RMS (re-calculated here for final alignment, though ideally shared)
        n_fft = stage_a_out.meta.window_size if stage_a_out.meta.window_size else 2048
        frames = _frame_audio(audio, n_fft, hop_length)
        rms_vals = np.sqrt(np.mean(frames**2, axis=1))

        # Pad/Trim RMS to match dominant F0 (which is now canonical length)
        if len(rms_vals) < len(merged_f0):
            rms_vals = np.pad(rms_vals, (0, len(merged_f0) - len(rms_vals)))
        elif len(rms_vals) > len(merged_f0):
            rms_vals = rms_vals[:len(merged_f0)]

        # Patch B3: Vectorized Transient Lockout (RMS-based)
        lockout_ms = float(profile_special.get("transient_lockout_ms", 0.0) or 0.0)
        onset_ratio_thr = float(profile_special.get("onset_ratio_thr", 2.5) or 2.5)

        lockout_frames = int(round((lockout_ms / 1000.0) * sr / max(hop_length, 1))) if lockout_ms > 0 else 0

        if lockout_frames > 0 and len(rms_vals) > 1:
            lock_mask = np.zeros_like(merged_conf, dtype=bool)
            eps = 1e-9
            rms_ratio = np.ones_like(rms_vals, dtype=np.float32)
            rms_ratio[1:] = rms_vals[1:] / (rms_vals[:-1] + eps)

            onset_idx = np.where(rms_ratio >= onset_ratio_thr)[0]
            for idx in onset_idx:
                lock_mask[idx : min(idx + lockout_frames, len(lock_mask))] = True

            # Apply mask to merged
            merged_conf = np.where(lock_mask, 0.0, merged_conf)
            merged_f0   = np.where(lock_mask, 0.0, merged_f0)

            # Apply mask to ISS layers too
            masked_iss_layers = []
            for f0_l, c_l in iss_layers:
                c_l = np.where(lock_mask[:len(c_l)], 0.0, c_l)
                f0_l = np.where(c_l > 0.0, f0_l, 0.0)
                masked_iss_layers.append((f0_l, c_l))
            iss_layers = masked_iss_layers

            # (Optional) Apply mask to per_detector tracks for consistent debugging
            for det_name in per_detector[stem_name]:
                 pf0, pconf = per_detector[stem_name][det_name]
                 if len(pconf) == len(lock_mask):
                     pconf = np.where(lock_mask, 0.0, pconf)
                     pf0 = np.where(lock_mask, 0.0, pf0)
                     per_detector[stem_name][det_name] = (pf0, pconf)


        # Melody stabilization before skyline selection
        # 4F: Use melody_filter_eff
        merged_f0, merged_conf = _apply_melody_filters(
            merged_f0,
            merged_conf,
            rms_vals,
            melody_filter_eff,
        )

        # Build timeline with optional skyline selection from poly layers
        voicing_thr_global = float(b_conf.confidence_voicing_threshold)

        # Apply polyphonic relaxation (if any) to the GLOBAL gate only
        # GUARD: Only apply this relaxation if we are truly in a polyphonic mode (AudioType check).
        # This prevents monophonic paths (L0/L1) from using over-relaxed gates which can leak noise.
        is_true_poly = _is_polyphonic(getattr(stage_a_out, "audio_type", None))

        if is_true_poly:
            poly_relax = float(getattr(b_conf, "polyphonic_voicing_relaxation", 0.0) or 0.0)
        else:
            poly_relax = 0.0

        voicing_thr_effective = voicing_thr_global - poly_relax

        # Melody filter threshold stays a *post-filter* knob (do not silently lower global gate)
        melody_voiced_thr = float(melody_filter_eff.get("voiced_prob_threshold", voicing_thr_effective))

        # Use voicing_thr_effective for deciding voiced/unvoiced frames
        voicing_thr = voicing_thr_effective

        # Record into diagnostics (no schema change if diagnostics already exists)
        diagnostics = locals().get("diagnostics", None)
        # Note: diagnostics dict is created at the end of the function usually,
        # but we need to inject tuning_cents.
        # We'll rely on collecting tuning_cents into a separate dict for now
        # and merging into diagnostics at the end.

        layer_arrays = [(merged_f0, merged_conf)] + iss_layers
        max_frames = max(len(arr[0]) for arr in layer_arrays)

        def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
            if len(arr) < target:
                return np.pad(arr, (0, target - len(arr)))
            return arr[:target]

        padded_layers = [(_pad_to(f0, max_frames), _pad_to(conf, max_frames)) for f0, conf in layer_arrays]
        padded_rms = _pad_to(rms_vals, max_frames)

        timeline: List[FramePitch] = []
        max_alt_voices = int(tracker_cfg.get("max_alt_voices", 4) if polyphonic_context else 0)
        tracker = MultiVoiceTracker(
            max_tracks=1 + max_alt_voices,
            max_jump_cents=tracker_cfg.get("max_jump_cents", 150.0),
            hangover_frames=tracker_cfg.get("hangover_frames", 2),
            smoothing=tracker_cfg.get("smoothing", 0.35),
            confidence_bias=tracker_cfg.get("confidence_bias", 5.0),
        )

        track_buffers = [np.zeros(max_frames, dtype=np.float32) for _ in range(tracker.max_tracks)]
        track_conf_buffers = [np.zeros(max_frames, dtype=np.float32) for _ in range(tracker.max_tracks)]
        primary_track = np.zeros(max_frames, dtype=np.float32)
        # GUARD: Do not use top_voice logic for Monophonic files (L0/L1) even if polyphonic_context=True
        select_top_voice = is_true_poly and polyphonic_context and "top_voice" in str(skyline_mode)

        # B3: Apply tuning offset during MIDI conversion
        tuning_semitones = tuning_cents / 100.0

        for i in range(max_frames):
            candidates: List[Tuple[float, float]] = []
            for f0_arr, conf_arr in padded_layers:
                f = float(f0_arr[i]) if i < len(f0_arr) else 0.0
                c = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                if f > 0.0 and c >= voicing_thr:
                    candidates.append((f, c))

            # Preserve raw detector/layer candidates for skyline selection in Stage C
            active_candidates = list(candidates)

            tracked_pitches, tracked_confs = tracker.step(candidates)
            for voice_idx in range(tracker.max_tracks):
                track_buffers[voice_idx][i] = tracked_pitches[voice_idx]
                track_conf_buffers[voice_idx][i] = tracked_confs[voice_idx]

            primary_idx = 0
            if select_top_voice and tracked_pitches.size:
                primary_idx = int(np.argmax(tracked_pitches))
                if tracked_pitches[primary_idx] <= 0.0:
                    primary_idx = 0

            chosen_pitch = float(tracked_pitches[primary_idx]) if tracked_pitches.size else 0.0
            chosen_conf = float(tracked_confs[primary_idx]) if tracked_confs.size else 0.0
            primary_track[i] = chosen_pitch

            midi = None
            if chosen_pitch > 0.0:
                # Apply global tuning correction
                midi_float = 69.0 + 12.0 * np.log2(chosen_pitch / 440.0)
                midi = int(round(midi_float - tuning_semitones))

            # Use the raw candidates (pre-tracking) to expose all available pitches
            # to downstream skyline selection instead of only the tracked voices.
            active = [(float(p), float(c)) for (p, c) in active_candidates if p > 0.0]

            timeline.append(
                FramePitch(
                    time=float(i * hop_length) / float(sr),
                    pitch_hz=chosen_pitch,
                    confidence=chosen_conf,
                    midi=midi,
                    rms=float(padded_rms[i]) if i < len(padded_rms) else 0.0,
                    active_pitches=active,
                )
            )

        if timeline:
            tail_window = 0.35
            last_pitch = next((fp.pitch_hz for fp in reversed(timeline) if fp.pitch_hz > 0.0), 0.0)
            if last_pitch > 0.0:
                midi_float = 69.0 + 12.0 * np.log2(last_pitch / 440.0)
                last_midi = int(round(midi_float - tuning_semitones))
                tail_start = timeline[-1].time - tail_window
                for fp in reversed(timeline):
                    if fp.time < tail_start:
                        break
                    if fp.pitch_hz <= 0.0:
                        fp.pitch_hz = last_pitch
                        fp.midi = last_midi
                        fp.confidence = max(fp.confidence, voicing_thr)

        stem_timelines[stem_name] = timeline

        if not any(fp.pitch_hz > 0.0 for fp in timeline) and np.any(merged_f0 > 0):
            stem_timelines[stem_name] = _arrays_to_timeline(merged_f0, merged_conf, rms_vals, sr, hop_length)

        # Keep secondary voices as separate layers to aid downstream segmentation/rendering
        for alt in track_buffers[1:]:
            if np.count_nonzero(alt) > 0:
                all_layers.append(alt)

        # Set main f0 (prefer vocals, then mix)
        main_track = primary_track if select_top_voice else track_buffers[0]
        if stem_name == "vocals":
            f0_main = main_track
        elif stem_name == "mix" and f0_main is None:
            f0_main = main_track

    if f0_main is None:
        if stem_timelines:
            first_stem = next(iter(stem_timelines.values()))
            f0_main = np.array([fp.pitch_hz for fp in first_stem], dtype=np.float32)
        else:
            f0_main = np.array([], dtype=np.float32)

    time_grid = np.array([])
    if len(f0_main) > 0:
        time_grid = np.arange(len(f0_main)) * hop_length / sr

    # Populate diagnostics with tuning_cents (using the last calculated one if multiple stems, or default)
    # Ideally should be per-stem or mix.
    # The requirement says "store it in diagnostics".
    tuning_cents_val = locals().get("tuning_cents", 0.0)

    # Patch D6: Diagnostics recording resolved profile
    diagnostics = {
        "instrument": str(instrument),
        "profile": profile.instrument if profile else None,
        "profile_applied": bool(profile_applied),
        "profile_special": dict(profile.special) if profile else {},
        "polyphonic_context": bool(polyphonic_context),
        "detectors_initialized": list(detectors.keys()),
        "separation": separation_diag,
        "harmonic_masking": {
            "enabled": harmonic_cfg.get("enabled", False),
            "applied": harmonic_mask_applied,
            "mask_width": harmonic_cfg.get("mask_width"),
            "n_harmonics": harmonic_cfg.get("n_harmonics"),
        },
        "iss": {
            "enabled": polyphonic_context and b_conf.polyphonic_peeling.get("max_layers", 0) > 0,
            "layers_found": iss_total_layers,
            "max_layers": b_conf.polyphonic_peeling.get("max_layers", 0),
        },
        "skyline_mode": skyline_mode,
        "voice_tracking": {
            "max_alt_voices": int(tracker_cfg.get("max_alt_voices", 4) if polyphonic_context else 0),
            "max_jump_cents": tracker_cfg.get("max_jump_cents", 150.0),
        },
        "global_tuning_cents": tuning_cents_val,
    }

    primary_timeline = (
        stem_timelines.get("vocals")
        or stem_timelines.get("mix")
        or (next(iter(stem_timelines.values())) if stem_timelines else [])
    )

    return StageBOutput(
        time_grid=time_grid,
        f0_main=f0_main,
        f0_layers=all_layers,
        per_detector=per_detector,
        stem_timelines=stem_timelines,
        meta=stage_a_out.meta,
        diagnostics=diagnostics,
        timeline=primary_timeline or [],
    )
