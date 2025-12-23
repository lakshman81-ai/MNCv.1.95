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

import math
from collections import deque

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
from pathlib import Path

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


def sigmoid(x):  # safe-ish
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def weighted_median(values, weights):
    # values: list[float], weights: list[float]
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return float("nan")
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= 0.5 * total:
            return v
    return pairs[-1][0]

# ---- Poly candidate cleanup helpers ----------------------------------------

_LOG2_1200 = 1200.0 / math.log(2.0)
_EPS = 1e-9

def _cents_diff(a_hz: float, b_hz: float) -> float:
    if a_hz <= 0.0 or b_hz <= 0.0:
        return 1e9
    return abs(_LOG2_1200 * math.log((a_hz + _EPS) / (b_hz + _EPS)))

def _maybe_compute_cqt_ctx(audio: np.ndarray, sr: int, hop_length: int,
                           fmin: float, fmax: float,
                           bins_per_octave: int = 36) -> Optional[dict]:
    """
    Returns dict with: mag (float32 [n_bins, n_frames]), freqs (float32 [n_bins])
    or None if librosa missing or computation fails.
    """
    if not _module_available("librosa"):
        return None
    try:
        import librosa  # optional
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return None

        # n_bins spans [fmin, fmax]
        fmin = float(max(20.0, fmin))
        fmax = float(max(fmin * 1.01, fmax))
        n_oct = math.log2(fmax / fmin)
        n_bins = int(max(24, math.ceil(n_oct * bins_per_octave)))

        C = librosa.cqt(y=y, sr=sr, hop_length=hop_length,
                        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
        mag = np.abs(C).astype(np.float32)
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave).astype(np.float32)

        if mag.size == 0 or freqs.size == 0:
            return None
        return {"mag": mag, "freqs": freqs}
    except Exception:
        return None

def _cqt_mag_at(ctx: Optional[dict], frame_idx: int, hz: float) -> float:
    if ctx is None or hz <= 0.0:
        return 0.0
    mag = ctx["mag"]
    freqs = ctx["freqs"]
    if mag.ndim != 2 or freqs.ndim != 1:
        return 0.0
    t = min(max(0, int(frame_idx)), mag.shape[1] - 1)
    # nearest bin
    j = int(np.clip(np.searchsorted(freqs, hz), 0, freqs.size - 1))
    if j > 0 and abs(freqs[j - 1] - hz) < abs(freqs[j] - hz):
        j -= 1
    return float(mag[j, t])

def _cqt_frame_floor(ctx: Optional[dict], frame_idx: int) -> float:
    if ctx is None:
        return 0.0
    mag = ctx["mag"]
    if mag.ndim != 2 or mag.shape[1] == 0:
        return 0.0
    t = min(max(0, int(frame_idx)), mag.shape[1] - 1)
    # robust floor for that frame
    col = mag[:, t]
    return float(np.median(col)) + 1e-9

def _postprocess_candidates(
    candidates: List[Tuple[float, float]],
    frame_idx: int,
    cqt_ctx: Optional[dict],
    max_candidates: int,
    dup_cents: float = 35.0,
    harmonic_cents: float = 35.0,
    cqt_gate_mul: float = 0.25,
    cqt_support_ratio: float = 2.0,
    harmonic_drop_ratio: float = 0.75,
    min_post_conf: float = 0.12,
    harmonic_ratios: Tuple[int, ...] = (2, 3, 4),
    subharmonic_boost_ratio: float = 1.15,
    subharmonic_conf_mul: float = 0.6,
) -> List[Tuple[float, float]]:
    """
    Cleanup for per-frame polyphonic F0 candidates.

    What it does (in order):
    - Soft CQT gate: if a candidate lacks spectral support at its fundamental, reduce conf.
    - Optional subharmonic injection: if f/2 has stronger CQT support than f, add (f/2) as a weak candidate.
      This helps when detectors "lock" onto harmonics instead of fundamentals (common in dense piano).
    - Drop near-duplicates (within dup_cents).
    - Suppress harmonic duplicates for ratios in `harmonic_ratios` (default: 2×, 3×, 4×).
    - Cap count and renormalize confidences to [0, 1] (max=1).

    Notes:
    - `min_post_conf` is applied AFTER the CQT gate so weak, unsupported pitches don't leak downstream.
    """
    if not candidates:
        return []

    floor = _cqt_frame_floor(cqt_ctx, frame_idx)

    gated: List[Tuple[float, float]] = []
    extra: List[Tuple[float, float]] = []

    # clamp conf and soft CQT gate (+ optional subharmonic)
    for f, c in candidates:
        if f <= 0.0 or c <= 0.0:
            continue
        f = float(f)
        c = float(max(0.0, min(1.0, c)))

        m_f = 0.0
        if cqt_ctx is not None and floor > 0.0:
            m_f = _cqt_mag_at(cqt_ctx, frame_idx, f)
            # support if fundamental bin stands out vs median floor
            if m_f < floor * float(cqt_support_ratio):
                c *= float(cqt_gate_mul)

            # Subharmonic injection: if the "half" bin is stronger, we may be tracking a harmonic.
            sub_f = f * 0.5
            if sub_f >= 20.0 and m_f > 0.0:
                m_sub = _cqt_mag_at(cqt_ctx, frame_idx, sub_f)
                if m_sub > m_f * float(subharmonic_boost_ratio):
                    extra.append((sub_f, c * float(subharmonic_conf_mul)))

        if c >= float(min_post_conf):
            gated.append((f, c))

    if extra:
        # Merge extra candidates (they'll be deduped later)
        for f, c in extra:
            if f > 0.0 and c >= float(min_post_conf):
                gated.append((float(f), float(c)))

    if not gated:
        return []

    # sort strongest first
    gated.sort(key=lambda x: x[1], reverse=True)

    kept: List[Tuple[float, float]] = []

    def _looks_like_harmonic(lo: float, hi: float, ratio: int) -> bool:
        if cqt_ctx is None:
            return False
        if ratio <= 1:
            return False
        # ensure hi ~ ratio*lo
        if abs(_cents_diff(hi, float(ratio) * lo)) > harmonic_cents:
            return False
        m_lo = _cqt_mag_at(cqt_ctx, frame_idx, lo)
        m_hi = _cqt_mag_at(cqt_ctx, frame_idx, hi)
        if m_lo <= 0.0:
            return False
        return (m_hi / (m_lo + 1e-9)) < float(harmonic_drop_ratio)

    for f, c in gated:
        # 1) drop near-duplicate pitch (same note repeated across layers)
        dup = False
        for fk, ck in kept:
            if _cents_diff(f, fk) <= dup_cents:
                dup = True
                break
        if dup:
            continue

        # 2) harmonic suppression: if this candidate is ~r× of an already-kept one and looks harmonic, skip it.
        harm = False
        for fk, ck in kept:
            if fk <= 0.0:
                continue
            for r in harmonic_ratios:
                if abs(_cents_diff(f, float(r) * fk)) <= harmonic_cents:
                    if _looks_like_harmonic(fk, f, int(r)) and c <= ck * 1.10:
                        harm = True
                        break
            if harm:
                break
        if harm:
            continue

        kept.append((f, c))
        if len(kept) >= int(max(1, max_candidates)):
            break

    if not kept:
        return []

    # 3) renormalize conf so max=1 (keeps ordering, keeps values bounded)
    max_c = max(c for _, c in kept)
    if max_c > 1e-9:
        kept = [(f, float(c / max_c)) for f, c in kept]

    return kept

class DetectorReliability:
    def __init__(self, base_w: float, pop_penalty_frames=6):
        self.base_w = base_w
        self.pop_penalty = 0
        self.pop_penalty_frames = pop_penalty_frames
        self.prev_cents = None
        self.recent = deque(maxlen=5)

    def update(self, cents, conf, energy_ok=True):
        # stability: small derivative over recent frames
        if cents == cents:  # not NaN
            self.recent.append(cents)
        stability = 1.0
        if len(self.recent) >= 3:
            diffs = [abs(self.recent[i]-self.recent[i-1]) for i in range(1,len(self.recent))]
            stability = 1.0 / (1.0 + (sum(diffs)/len(diffs))/80.0)  # 80 cents scale

        # octave pop heuristic: ~1200 cents jump without energy change
        if self.prev_cents is not None and cents == cents and self.prev_cents == self.prev_cents:
            jump = abs(cents - self.prev_cents)
            if 900.0 <= jump <= 1500.0 and energy_ok:
                self.pop_penalty = self.pop_penalty_frames
        self.prev_cents = cents

        if self.pop_penalty > 0:
            self.pop_penalty -= 1
            pop_factor = 0.2
        else:
            pop_factor = 1.0

        conf_factor = sigmoid((conf - 0.5) * 6.0)  # tune slope
        w = self.base_w * conf_factor * stability * pop_factor
        return w

def viterbi_pitch(fused_cents, fused_conf, midi_states, transition_smoothness=0.5, jump_penalty=0.6):
    # fused_cents: list[float] length T
    T = len(fused_cents)
    S = len(midi_states)
    INF = 1e18

    # precompute state cents
    state_cents = [m*100.0 for m in midi_states]

    # dp[t][s] cost, backpointers
    dp = [[INF]*S for _ in range(T)]
    bp = [[-1]*S for _ in range(T)]

    def emission(t, s):
        c = fused_cents[t]
        if not (c == c) or c <= 0:
            return 5.0  # allow gaps softly
        dist = abs(c - state_cents[s])
        conf = fused_conf[t] if t < len(fused_conf) and fused_conf[t] is not None else 0.5
        # confidence penalty: if conf is high, we trust fused_cents more (higher cost for deviance)
        return (dist/50.0) - 0.8*math.log(max(1e-3, conf))

    def transition(s0, s1):
        step = abs(midi_states[s1] - midi_states[s0])
        # allow small steps (0–2) cheaply, penalize larger jumps
        if step <= 2:
            return transition_smoothness * step # 0.2 * step in snippet
        return transition_smoothness * 2.5 + jump_penalty * (step - 2)

    # init
    for s in range(S):
        dp[0][s] = emission(0, s)

    # forward
    for t in range(1, T):
        # Optimization: prune search space to states near fused pitch if available
        # But Viterbi is globally optimal, so pruning might hurt.
        # Given S ~ 88 (or 61), it's fast enough.
        for s1 in range(S):
            e = emission(t, s1)
            # Find best s0
            # To speed up: inner loop is dense.
            # But python is slow.
            # We assume S is small (<100).
            best_cost = INF
            best_s0 = -1

            for s0 in range(S):
                # Transition matrix is symmetric and simple.
                cost = dp[t-1][s0] + transition(s0, s1)
                if cost < best_cost:
                    best_cost = cost
                    best_s0 = s0

            dp[t][s1] = best_cost + e
            bp[t][s1] = best_s0

    # backtrack
    if T == 0:
        return []

    s = min(range(S), key=lambda k: dp[T-1][k])
    path = [s]*T
    for t in range(T-1, 0, -1):
        s = bp[t][s]
        path[t-1] = s
    smoothed_hz = [440.0 * (2**((midi_states[i]-69)/12.0)) for i in path]
    return smoothed_hz

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
        self._warned_no_scipy = False

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
            if not getattr(self, "_warned_no_scipy", False):
                logger.warning("SyntheticMDX: Scipy missing. Falling back to gain-based separation (no frequency filtering).")
                self._warned_no_scipy = True

            # Normalize to avoid clipping
            s = vocals_w + bass_w
            if s > 1.0:
                vocals_w /= s
                bass_w /= s

            vocals = vocals_w * audio
            bass = bass_w * audio
            drums = drums_w * audio
            other = audio - (vocals + bass + drums)

            return {
                "vocals": vocals,
                "bass": bass,
                "drums": drums,
                "other": other,
            }

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
    # Robust shape handling rules:
    # (T,) -> (1, T) -> (2, T)
    # (1, T) -> (2, T)
    # (2, T) -> OK
    # (T, C) -> (C, T) -> OK/Trim

    # First, handle potentially transposed input (T, C) -> (C, T)
    # Heuristic: if shape[0] > 10 and shape[1] <= 10, assume (T, C)
    if resampled.ndim == 2:
        d0, d1 = resampled.shape
        if d0 > 10 and d1 <= 10:
             resampled = resampled.T

    # Ensure (C, T)
    if resampled.ndim == 1:
        resampled = resampled[None, :]

    C, T = resampled.shape

    # Force stereo
    if C == 1:
        resampled = np.concatenate([resampled, resampled], axis=0)
    elif C > 2:
        resampled = resampled[:2, :]

    # Final guard before inference
    if resampled.shape[0] != 2:
        # Should be unreachable given above logic, but safety first
        logger.warning(f"HTDemucs input shape unexpected: {resampled.shape}. Forcing stereo duplication.")
        if resampled.shape[0] == 1:
             resampled = np.concatenate([resampled, resampled], axis=0)
        else:
             # Fallback: take mean and duplicate
             mono = np.mean(resampled, axis=0, keepdims=True)
             resampled = np.concatenate([mono, mono], axis=0)

    # Prepare tensor: (Batch, Channels, Time) -> (1, 2, T)
    mix_tensor = torch.tensor(resampled, dtype=torch.float32)[None, :, :].to(dev)

    # Hard assert on shape to prevent model crash with cryptic error
    if mix_tensor.shape[1] != 2:
         logger.error(f"HTDemucs tensor shape invalid: {mix_tensor.shape}. Skipping.")
         return None

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
    sep_enabled = b_conf.separation.get("enabled", True)
    # Patch 6: explicit auto handling
    is_auto = (isinstance(sep_enabled, str) and sep_enabled.lower() == "auto")
    should_run = bool(sep_enabled)  # True or "auto" are truthy

    diag = {
        "requested": sep_enabled,
        "is_auto": is_auto,
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

    if not should_run:
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
    if is_auto:
        diag["fallback_reason"] = "htdemucs_failed_or_missing_in_auto"

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
    adaptive_fusion: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge multiple f0/conf tracks based on weights and disagreement.

    Strategy:
      * Align frame counts across detectors
      * Choose the candidate with the highest weighted confidence OR use weighted median (Step 2)
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

    # Initialize reliability trackers if adaptive
    reliabilities = {}
    if adaptive_fusion:
        for name in aligned_results:
            w = weights.get(name, 1.0)
            reliabilities[name] = DetectorReliability(base_w=w)

    for i in range(max_len):
        candidates = []
        detector_outputs = []

        for name, (f0, conf) in aligned_results.items():
            w = weights.get(name, 1.0)
            c = float(conf[i])
            f = float(f0[i])

            # Prepare data for adaptive fusion
            cents = float("nan")
            if f > 0.0:
                 cents = 1200.0 * math.log2(f / 440.0) + 6900.0

            detector_outputs.append({
                "name": name,
                "cents": cents,
                "conf": c,
                "voiced": c > 0.0 and f > 0.0
            })

            if c <= 0.0 or f <= 0.0:
                continue

            # Priority floor mostly benefits SwiftF0 on synthetic tones
            eff_conf = max(c, priority_floor if name == "swiftf0" else c)
            candidates.append((name, f, eff_conf, w))

        if not candidates:
            # Still update reliabilities with unvoiced?
            # Step 2 logic: update only if voiced or valid.
            # Actually we should update tracking state even if unvoiced?
            # The snippet updates if voiced.
            final_f0[i] = 0.0
            final_conf[i] = 0.0
            continue

        if adaptive_fusion:
            # Step 2: Fuse frame
            vals, wts = [], []
            for out in detector_outputs:
                cents = out["cents"]
                if not out.get("voiced", True) or not (cents == cents):
                    continue
                # Assuming energy_ok is True for now, as we don't pass frame rms here
                w = reliabilities[out["name"]].update(cents, out["conf"], energy_ok=True)
                if w > 1e-6:
                    vals.append(cents); wts.append(w)

            fused_cents = weighted_median(vals, wts) if vals else float("nan")

            if fused_cents == fused_cents: # not nan
                fused_hz = 440.0 * (2.0 ** ((fused_cents - 6900.0) / 1200.0))
                final_f0[i] = fused_hz
                # Use max confidence of contributors? Or weighted avg?
                # Weighted avg confidence seems appropriate
                final_conf[i] = sum(c[2] for c in candidates) / len(candidates) # rough approx
            else:
                final_f0[i] = 0.0
                final_conf[i] = 0.0

        else:
            # Original Static Fusion
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
    if SCIPY_SIGNAL is None:
        return {}

    audio = np.asarray(stem.audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {}

    try:
        f0, conf = prior_detector.predict(audio, audio_path=audio_path)
        hop = getattr(prior_detector, "hop_length", 512)
        n_fft = getattr(prior_detector, "n_fft", 2048)
        n_fft_eff = int(min(n_fft, max(32, len(audio))))
        hop_eff = int(max(1, min(hop, n_fft_eff // 2)))

        f, t, Z = SCIPY_SIGNAL.stft(
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
            n_fft=n_fft_eff,
            mask_width=mask_width,
            n_harmonics=n_harmonics,
        )

        # Harmonic emphasis keeps bins near f0; residual keeps the rest.
        strength = np.clip(conf, 0.0, 1.0).reshape(1, -1)
        harmonic_keep = np.clip((1.0 - mask) * (0.8 + 0.2 * strength), 0.0, 1.0)
        residual_keep = np.clip(1.0 - harmonic_keep, 0.0, 1.0)

        Z_melody = Z * harmonic_keep
        Z_resid = Z * residual_keep

        _, melody_audio = SCIPY_SIGNAL.istft(
            Z_melody,
            fs=stem.sr,
            nperseg=n_fft_eff,
            noverlap=max(0, n_fft_eff - hop_eff),
            input_onesided=True,
            boundary="zeros",
        )
        _, residual_audio = SCIPY_SIGNAL.istft(
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


def _validate_config_keys(name: str, cfg: dict, allowed: set[str], pipeline_logger: Optional[Any] = None) -> None:
    unknown = set(cfg.keys()) - allowed
    if unknown:
        msg = f"Config unknown keys in {name}: {sorted(list(unknown))}"
        if pipeline_logger:
            pipeline_logger.log_event("stage_b", "config_unknown_keys", payload={"section": name, "keys": sorted(list(unknown))})
        else:
            logger.warning(msg)


def _curve_summary(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {
        "len": int(x.size),
        "nonzero": int(np.count_nonzero(x)),
        "min": float(x[x > 0].min()) if np.any(x > 0) else 0.0,
        "max": float(x.max()) if x.size else 0.0,
    }


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

    # -------------------------------------------------------------------------
    # E2E / Neural Transcription Routing (Basic Pitch)
    # -------------------------------------------------------------------------
    transcription_mode = getattr(b_conf, "transcription_mode", "classic")

    # Auto logic: try to use E2E if basic_pitch is importable
    if transcription_mode == "auto":
        if _module_available("basic_pitch"):
            transcription_mode = "e2e_basic_pitch"
        else:
            transcription_mode = "classic"

    if transcription_mode == "e2e_basic_pitch":
        temp_dir = kwargs.get("temp_dir", Path("/tmp"))
        if isinstance(temp_dir, str):
            temp_dir = Path(temp_dir)

        try:
            from .neural_transcription import transcribe_basic_pitch_to_notes

            # Use 'mix' stem if available, else first stem
            audio_source = None
            if "mix" in stage_a_out.stems:
                audio_source = stage_a_out.stems["mix"].audio
            elif stage_a_out.stems:
                audio_source = next(iter(stage_a_out.stems.values())).audio

            if audio_source is not None:
                notes = transcribe_basic_pitch_to_notes(
                    audio_source,
                    sr,
                    temp_dir,
                    onset_threshold=getattr(b_conf, "bp_onset_threshold", 0.5),
                    frame_threshold=getattr(b_conf, "bp_frame_threshold", 0.3),
                    minimum_note_length_ms=getattr(b_conf, "bp_minimum_note_length_ms", 127.7),
                    min_hz=getattr(b_conf, "bp_min_hz", 27.5),
                    max_hz=getattr(b_conf, "bp_max_hz", 4186.0),
                    melodia_trick=getattr(b_conf, "bp_melodia_trick", True),
                )

                # Success: return StageBOutput with precalculated_notes
                return StageBOutput(
                    time_grid=np.array([], dtype=np.float32),
                    f0_main=np.array([], dtype=np.float32),
                    f0_layers=[],
                    stem_timelines={},
                    per_detector={},
                    meta=stage_a_out.meta,
                    diagnostics={"stage_b_mode": "e2e_basic_pitch"},
                    precalculated_notes=notes,
                )
        except Exception as e:
            logger.warning(f"Basic Pitch unavailable/failed; falling back to classic: {e}")
            # Fallback to classic

    profile = config.get_profile(str(instrument)) if (instrument and b_conf.apply_instrument_profile) else None
    profile_special = dict(getattr(profile, "special", {}) or {}) if profile else {}
    profile_applied = bool(profile)

    # Work on copies so we don't mutate config dataclasses
    detector_cfgs = deepcopy(b_conf.detectors)
    weights_eff = dict(b_conf.ensemble_weights)
    melody_filter_eff = dict(getattr(b_conf, "melody_filtering", {}) or {})

    # Patch 1C: Filter weights for enabled detectors only
    enabled_dets = {k for k, v in detector_cfgs.items() if v.get("enabled", False)}
    weights_eff = {k: v for k, v in weights_eff.items() if k in enabled_dets}

    # Patch 7: Config Validator
    common_keys = {"enabled", "fmin", "fmax", "hop_length", "frame_length", "threshold"}
    _validate_config_keys("detectors.crepe", detector_cfgs.get("crepe", {}),
                          common_keys | {"model_capacity", "step_ms", "confidence_threshold", "conf_threshold", "use_viterbi"}, pipeline_logger)
    _validate_config_keys("detectors.yin", detector_cfgs.get("yin", {}),
                          common_keys | {"enable_multires_f0", "enable_octave_correction", "octave_jump_penalty", "trough_threshold"}, pipeline_logger)
    _validate_config_keys("detectors.swiftf0", detector_cfgs.get("swiftf0", {}),
                          common_keys | {"confidence_threshold", "n_fft"}, pipeline_logger)

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
        # Check config for adaptive mode
        use_adaptive = getattr(b_conf, "ensemble_mode", "static") == "adaptive"

        if stem_results:
            merged_f0, merged_conf = _ensemble_merge(
                stem_results,
                weights_eff,
                b_conf.pitch_disagreement_cents,
                b_conf.confidence_priority_floor,
                adaptive_fusion=use_adaptive,
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
                    use_freq_aware_masks=bool(config.stage_b.polyphonic_peeling.get("use_freq_aware_masks", True)),
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

        # Capture pre-smoothing (fused) F0 for debug artifacts
        fused_f0_debug = merged_f0.copy()

        # Step 3: Viterbi Smoothing (Optional)
        use_viterbi_smoothing = getattr(b_conf, "smoothing_method", "tracker") == "viterbi"
        if use_viterbi_smoothing and np.any(merged_f0 > 0):
             # Define state space: e.g. MIDI 21..108
             midi_states = list(range(21, 109))
             # Convert f0 to cents for viterbi
             fused_cents = []
             for f in merged_f0:
                 if f > 0:
                     fused_cents.append(1200.0 * math.log2(f/440.0) + 6900.0)
                 else:
                     fused_cents.append(float("nan"))

             # Run Viterbi
             # Note: merged_conf is used as confidence
             smoothed_hz = viterbi_pitch(
                 fused_cents,
                 merged_conf,
                 midi_states,
                 transition_smoothness=getattr(b_conf, "viterbi_transition_smoothness", 0.5),
                 jump_penalty=getattr(b_conf, "viterbi_jump_penalty", 0.6)
             )

             if len(smoothed_hz) == len(merged_f0):
                 # Replace merged_f0 with smoothed (or blend?)
                 # For now replace, but keep confidence?
                 merged_f0 = np.array(smoothed_hz, dtype=np.float32)

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

        # Record fused/smoothed curves for debugging
        # We use fused_f0_debug (pre-smoothing) and merged_f0 (post-smoothing)
        # Note: diagnostics is a local variable shadowing the outer one if we are not careful
        # But here we just want to prepare data to put into the FINAL diagnostics object.
        # We can store it in 'per_detector' temporarily? No, that's typed.
        # Let's add it to a new field in StageBOutput if allowed, or put it in per_detector under reserved names.

        # Actually, let's just use the 'diagnostics' dict that we construct at the end of the function.
        # We need to persist these arrays outside this loop.
        # Let's use a side-channel dict `stem_debug_curves`
        if "stem_debug_curves" not in locals():
            stem_debug_curves = {}

        stem_debug_curves[stem_name] = {
            "fused_f0": _curve_summary(fused_f0_debug),
            "smoothed_f0": _curve_summary(merged_f0),
        }

        layer_arrays = [(merged_f0, merged_conf)] + iss_layers
        max_frames = max(len(arr[0]) for arr in layer_arrays)

        def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
            if len(arr) < target:
                return np.pad(arr, (0, target - len(arr)))
            return arr[:target]

        padded_layers = [(_pad_to(f0, max_frames), _pad_to(conf, max_frames)) for f0, conf in layer_arrays]
        padded_rms = _pad_to(rms_vals, max_frames)

        # ---- Optional CQT context for poly candidate gating (soft gate) ----
        # Only do this for true poly (won't affect L0/L1).
        cqt_ctx = None
        cqt_gate_enabled = bool(b_conf.polyphonic_peeling.get("cqt_gate_enabled", True))
        if is_true_poly and polyphonic_context and cqt_gate_enabled:
            fmin_gate = float(melody_filter_eff.get("fmin_hz", 60.0) or 60.0)
            fmax_gate = float(melody_filter_eff.get("fmax_hz", 2200.0) or 2200.0)
            cqt_ctx = _maybe_compute_cqt_ctx(audio, sr, hop_length, fmin=fmin_gate, fmax=fmax_gate,
                                             bins_per_octave=int(b_conf.polyphonic_peeling.get("cqt_bins_per_octave", 36)))

        # Capture summaries of layer confidences for diagnostics
        if "layer_conf_summaries" not in locals():
            layer_conf_summaries = {}
        try:
            # summaries only, not full arrays
            layer_conf_summaries[stem_name] = [
                _curve_summary(conf_arr) for (_, conf_arr) in padded_layers
            ]
        except Exception:
            pass

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

        # Poly-peeling postprocessing knobs (computed once per stem for speed and consistency)
        tracker_cap = int(tracker.max_tracks)
        max_active_pitches = int(b_conf.polyphonic_peeling.get("max_active_pitches_per_frame", max(tracker_cap, 10)))
        max_candidates_per_frame = int(b_conf.polyphonic_peeling.get("max_candidates_per_frame", max_active_pitches))
        min_post_conf = float(b_conf.polyphonic_peeling.get("min_post_conf", 0.12))

        harm_ratios_raw = b_conf.polyphonic_peeling.get("harmonic_ratios", (2, 3, 4))
        try:
            harmonic_ratios = tuple(int(x) for x in harm_ratios_raw)
            if not harmonic_ratios:
                harmonic_ratios = (2, 3, 4)
        except Exception:
            harmonic_ratios = (2, 3, 4)

        harmonic_cents = float(b_conf.polyphonic_peeling.get("harmonic_cents", b_conf.polyphonic_peeling.get("octave_cents", 35.0)))
        subharmonic_boost_ratio = float(b_conf.polyphonic_peeling.get("subharmonic_boost_ratio", 1.15))
        subharmonic_conf_mul = float(b_conf.polyphonic_peeling.get("subharmonic_conf_mul", 0.6))

        for i in range(max_frames):
            candidates: List[Tuple[float, float]] = []
            for f0_arr, conf_arr in padded_layers:
                f = float(f0_arr[i]) if i < len(f0_arr) else 0.0
                c = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                if f > 0.0 and c >= voicing_thr:
                    candidates.append((f, c))

            # ---- cleanup: CQT soft gate + dedupe + harmonic suppression + cap ----
            # Keep a larger "active" pool for downstream chord detection, but track only up to tracker capacity.
            candidates = _postprocess_candidates(
                candidates=candidates,
                frame_idx=i,
                cqt_ctx=cqt_ctx,
                max_candidates=max_candidates_per_frame,
                dup_cents=float(b_conf.polyphonic_peeling.get("dup_cents", 35.0)),
                harmonic_cents=harmonic_cents,
                cqt_gate_mul=float(b_conf.polyphonic_peeling.get("cqt_gate_mul", 0.25)),
                cqt_support_ratio=float(b_conf.polyphonic_peeling.get("cqt_support_ratio", 2.0)),
                harmonic_drop_ratio=float(b_conf.polyphonic_peeling.get("harmonic_drop_ratio", 0.75)),
                min_post_conf=min_post_conf,
                harmonic_ratios=harmonic_ratios,
                subharmonic_boost_ratio=subharmonic_boost_ratio,
                subharmonic_conf_mul=subharmonic_conf_mul,
            )

            # Preserve raw candidates for skyline selection / chord extraction downstream
            active_candidates = list(candidates)

            # Track only up to tracker capacity (stability), but expose all active candidates.
            track_candidates = active_candidates[:tracker_cap]
            tracked_pitches, tracked_confs = tracker.step(track_candidates)
            for voice_idx in range(tracker.max_tracks):
                track_buffers[voice_idx][i] = tracked_pitches[voice_idx]
                track_conf_buffers[voice_idx][i] = tracked_confs[voice_idx]

            primary_idx = 0
            if select_top_voice and tracked_pitches.size:
                # Fix L5 failure mode: Select by confidence, not highest pitch
                # Improved: Argmax of (conf - jump_penalty) to maintain continuity
                prev_p = primary_track[i-1] if i > 0 else 0.0

                best_score = -999.0
                best_idx = 0
                found_valid = False

                for idx in range(len(tracked_confs)):
                    p = tracked_pitches[idx]
                    c = tracked_confs[idx]

                    if p <= 0.0:
                        continue

                    found_valid = True
                    score = float(c)

                    # Apply penalty for jumps from previous frame's selected pitch
                    if prev_p > 0.0:
                         cents = abs(1200.0 * np.log2(p / prev_p))
                         # Penalty: 0.0005 per cent -> 0.05 per semitone
                         # Reduced from 0.001 to strictly punish octave jumps but allow melodic steps
                         penalty = cents * 0.0005
                         score -= penalty

                    if score > best_score:
                        best_score = score
                        best_idx = idx

                if found_valid:
                    primary_idx = best_idx
                else:
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

            # Step 5 prep: Calculate Onset Strength for this frame/stem?
            # Ideally compute global onset strength once per stem and attach to FramePitch
            # We don't have it here yet. We can do it in Stage C or compute here.
            # Let's leave it to Stage C or future.

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

    # Retrieve debug curves if any
    stem_debug_curves = locals().get("stem_debug_curves", {})

    # Patch D6: Diagnostics recording resolved profile
    diagnostics = {
        "stage_b_mode": "classic",
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
        "cqt_gate": {
            "requested": cqt_gate_enabled,
            "active": cqt_ctx is not None,
            "librosa_available": _module_available("librosa"),
        },
        "skyline_mode": skyline_mode,
        "voice_tracking": {
            "max_alt_voices": int(tracker_cfg.get("max_alt_voices", 4) if polyphonic_context else 0),
            "max_jump_cents": tracker_cfg.get("max_jump_cents", 150.0),
        },
        "global_tuning_cents": tuning_cents_val,
        "debug_curves": stem_debug_curves,
        "layer_conf_summaries": locals().get("layer_conf_summaries", {}),
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
