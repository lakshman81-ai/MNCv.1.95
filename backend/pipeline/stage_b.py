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

from copy import deepcopy
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "extract_features",
    "create_harmonic_mask",
    "iterative_spectral_subtraction",
    "MultiVoiceTracker",
]

SCIPY_SIGNAL = None
if importlib.util.find_spec("scipy.signal"):
    import scipy.signal as SCIPY_SIGNAL


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
            continue
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue
        return default
    return default if cur is None else cur


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


# -----------------------------------------------------------------------------
# Stage B Routing Policy (WI Task 1.1–1.4)
# -----------------------------------------------------------------------------
ROUTING_POLICY_VERSION = "routing_v1"


def _safe_import_librosa():
    try:
        import librosa  # type: ignore
        return librosa
    except Exception:
        return None


_LIBROSA = _safe_import_librosa()


def _infer_audio_type_str(stage_a_out: StageAOutput, routing_features: Dict[str, Any]) -> str:
    """Best-effort audio_type string for diagnostics."""
    try:
        at = getattr(stage_a_out, "audio_type", None)
        if at is not None:
            return str(getattr(at, "value", at))
    except Exception:
        pass

    pm = float(routing_features.get("polyphony_mean", 0.0) or 0.0)
    if pm >= 2.0:
        return "polyphonic"
    if pm >= 1.3:
        return "polyphonic_dominant"
    if pm > 0.0:
        return "monophonic"
    return "unknown"


def _is_piano_like(profile: Optional[str]) -> bool:
    s = str(profile or "").lower()
    return any(k in s for k in ("piano", "keys", "keyboard", "rhodes"))


def _compute_routing_features(mix_audio: np.ndarray, sr: int, duration_sec: float) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """
    Cheap routing features (optional-dep safe).
    Deterministic; bounded compute by truncating analysis window.
    """
    feats: Dict[str, Any] = {
        "duration_sec": float(duration_sec or 0.0),
        "sr": int(sr or 0),
        "rms_mean": 0.0,
        "spectral_centroid_mean": 0.0,
        "spectral_centroid_std": 0.0,
        "spectral_flatness_mean": 0.0,
        "hpss_percussive_ratio": 0.0,
        "polyphony_mean": 0.0,
        "active_pitches_mean": 0.0,
        "active_pitches_p95": 0.0,
        "synthetic_like": False,
        "mixture_score": 0.0,
    }
    missing = {"librosa": False}

    if mix_audio is None or sr <= 0:
        return feats, missing

    y = np.asarray(mix_audio, dtype=np.float32).flatten()
    if y.size == 0:
        return feats, missing

    feats["rms_mean"] = float(np.sqrt(np.mean(y * y) + 1e-12))

    if _LIBROSA is None:
        missing["librosa"] = True
        return feats, missing

    try:
        max_sec = 20.0
        n = int(min(float(duration_sec or 0.0), max_sec) * float(sr))
        y_use = y[:n] if (n > 0 and y.size > n) else y

        centroid = _LIBROSA.feature.spectral_centroid(y=y_use, sr=sr)
        flat = _LIBROSA.feature.spectral_flatness(y=y_use)

        feats["spectral_centroid_mean"] = float(np.mean(centroid)) if centroid.size else 0.0
        feats["spectral_centroid_std"] = float(np.std(centroid)) if centroid.size else 0.0
        feats["spectral_flatness_mean"] = float(np.mean(flat)) if flat.size else 0.0

        try:
            harm, perc = _LIBROSA.effects.hpss(y_use)
            eh = float(np.sum(harm * harm) + 1e-12)
            ep = float(np.sum(perc * perc) + 1e-12)
            feats["hpss_percussive_ratio"] = float(ep / (eh + ep))
        except Exception:
            feats["hpss_percussive_ratio"] = 0.0

        try:
            pitches, mags = _LIBROSA.piptrack(
                y=y_use, sr=sr, hop_length=512, fmin=80.0, fmax=min(2000.0, sr / 2.2)
            )
            if mags.size:
                thr = float(np.percentile(mags, 95)) * 0.10
                active = mags > max(1e-8, thr)
                counts = np.sum(active, axis=0).astype(np.float32)
                feats["active_pitches_mean"] = float(np.mean(counts)) if counts.size else 0.0
                feats["active_pitches_p95"] = float(np.percentile(counts, 95)) if counts.size else 0.0
                feats["polyphony_mean"] = float(feats["active_pitches_mean"])
        except Exception:
            pass

        centroid_mean = float(feats["spectral_centroid_mean"] or 0.0)
        centroid_std = float(feats["spectral_centroid_std"] or 0.0)
        centroid_cv = float(centroid_std / max(1e-6, centroid_mean))
        percr = float(feats["hpss_percussive_ratio"] or 0.0)

        feats["synthetic_like"] = bool(
            (centroid_cv < 0.03) and (percr < 0.08) and (float(feats["spectral_flatness_mean"]) < 0.10)
        )

        mix_score = 0.60 * percr + 0.25 * centroid_cv + 0.15 * float(feats["spectral_flatness_mean"] or 0.0)
        feats["mixture_score"] = float(max(0.0, min(1.0, mix_score)))

    except Exception:
        pass

    return feats, missing


def _normalize_separation_mode(b_conf) -> str:
    sep = getattr(b_conf, "separation", {}) or {}
    mode = sep.get("mode", None)
    enabled = sep.get("enabled", True)

    if isinstance(mode, str):
        m = mode.lower().strip()
        if m in ("off", "disabled", "none", "false", "0"):
            return "off"
        if m in ("auto", "demucs"):
            return m

    if isinstance(enabled, str) and enabled.lower().strip() == "auto":
        return "auto"
    if not bool(enabled):
        return "off"
    return "auto"


def compute_decision_trace(
    stage_a_out: StageAOutput,
    config: Optional[PipelineConfig],
    *,
    requested_mode: Optional[str] = None,
    requested_profile: Optional[str] = None,
    requested_separation_mode: Optional[str] = None,
    pipeline_logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Schema-stable routing decision trace (deterministic).
    Stored in StageBOutput.diagnostics["decision_trace"].
    """
    if config is None:
        config = PipelineConfig()
    b_conf = config.stage_b

    req_mode = str(requested_mode or getattr(b_conf, "transcription_mode", "classic"))
    req_profile = str(
        requested_profile
        or getattr(b_conf, "instrument", None)
        or getattr(getattr(stage_a_out, "meta", None), "instrument", None)
        or "unknown"
    )
    req_sep_mode = str(requested_separation_mode or _normalize_separation_mode(b_conf))

    sr = int(getattr(getattr(stage_a_out, "meta", None), "sample_rate", 0) or 0)
    duration_sec = float(getattr(getattr(stage_a_out, "meta", None), "duration_sec", 0.0) or 0.0)

    mix_stem = (getattr(stage_a_out, "stems", {}) or {}).get("mix", None)
    mix_audio = getattr(mix_stem, "audio", None) if mix_stem is not None else None

    routing_features, missing = _compute_routing_features(
        mix_audio if mix_audio is not None else np.array([], dtype=np.float32),
        sr,
        duration_sec,
    )

    dense_thr = float(_get(config, "stage_b.routing.dense_poly_threshold", 1.8))
    song_mix_thr = float(_get(config, "stage_b.routing.song_like_mixture_score", 0.55))
    song_perc_thr = float(_get(config, "stage_b.routing.song_like_percussive_ratio", 0.35))

    dense_poly = bool(
        (float(routing_features.get("polyphony_mean", 0.0)) >= dense_thr)
        or (float(routing_features.get("active_pitches_mean", 0.0)) >= dense_thr)
    )
    song_like = bool(
        (float(routing_features.get("hpss_percussive_ratio", 0.0)) >= song_perc_thr)
        or (float(routing_features.get("mixture_score", 0.0)) >= song_mix_thr)
    )
    piano_like = _is_piano_like(req_profile)

    oaf_enabled = bool(_get(config, "stage_b.onsets_and_frames.enabled", False))
    oaf_available = bool(oaf_enabled and _module_available("torch"))
    bp_available = bool(_module_available("basic_pitch"))

    manual_override = (str(req_mode).lower().strip() != "auto")

    rule_hits: List[Dict[str, Any]] = []
    rule_hits.append({"rule_id": "R0_manual_override", "passed": bool(manual_override), "score": 1.0 if manual_override else 0.0})
    rule_hits.append({
        "rule_id": "R1_dense_poly_piano_e2e_oaf",
        "passed": bool((not manual_override) and dense_poly and piano_like),
        "score": float(max(float(routing_features.get("polyphony_mean", 0.0)), float(routing_features.get("active_pitches_mean", 0.0))))
        if ((not manual_override) and dense_poly and piano_like) else 0.0,
    })
    rule_hits.append({
        "rule_id": "R2_dense_poly_harmonic_e2e_basic_pitch",
        "passed": bool((not manual_override) and dense_poly and (not piano_like)),
        "score": float(max(float(routing_features.get("polyphony_mean", 0.0)), float(routing_features.get("active_pitches_mean", 0.0))))
        if ((not manual_override) and dense_poly and (not piano_like)) else 0.0,
    })
    rule_hits.append({
        "rule_id": "R3_mixture_song_classic_song",
        "passed": bool((not manual_override) and (not dense_poly) and song_like),
        "score": float(max(float(routing_features.get("hpss_percussive_ratio", 0.0)), float(routing_features.get("mixture_score", 0.0))))
        if ((not manual_override) and (not dense_poly) and song_like) else 0.0,
    })
    rule_hits.append({"rule_id": "R4_default_classic_melody", "passed": bool(not manual_override), "score": 1.0 if (not manual_override) else 0.0})

    resolved_mode = req_mode
    if manual_override:
        if resolved_mode == "e2e_onsets_frames" and (not oaf_available):
            resolved_mode = "classic_piano_poly" if dense_poly else "classic_melody"
        if resolved_mode == "e2e_basic_pitch" and (not bp_available):
            resolved_mode = "classic_piano_poly" if dense_poly else "classic_melody"
    else:
        if dense_poly and piano_like:
            resolved_mode = "e2e_onsets_frames" if oaf_available else "classic_piano_poly"
        elif dense_poly:
            resolved_mode = "e2e_basic_pitch" if bp_available else "classic_piano_poly"
        elif song_like:
            resolved_mode = "classic_song"
        else:
            resolved_mode = "classic_melody"

    resolved_sep = req_sep_mode if req_sep_mode in ("off", "auto", "demucs") else _normalize_separation_mode(b_conf)
    if str(resolved_mode).startswith("e2e_"):
        resolved_sep = "off"

    audio_type_str = _infer_audio_type_str(stage_a_out, routing_features)

    decision_trace: Dict[str, Any] = {
        "policy_version": ROUTING_POLICY_VERSION,
        # FIX: requested should reflect what was requested (not resolved)
        "requested": {
            "transcription_mode": str(req_mode),
            "profile": str(req_profile),
            "separation_mode": str(req_sep_mode),
        },
        "resolved": {
            "transcription_mode": str(resolved_mode),
            "profile": str(req_profile),
            "separation_mode": str(resolved_sep),
            "audio_type": str(audio_type_str),
        },
        "routing_features": dict(routing_features),
        "rule_hits": list(rule_hits),
        "separation": {
            "ran": False,
            "backend": "none",
            "skip_reasons": [],
            "gates": {
                "min_duration_sec": float(_get(config, "stage_b.separation.gates.min_duration_sec", 8.0)),
                "min_mixture_score": float(_get(config, "stage_b.separation.gates.min_mixture_score", 0.45)),
                "bypass_if_synthetic_like": bool(_get(config, "stage_b.separation.gates.bypass_if_synthetic_like", True)),
            },
            "outputs": {"stems": ["mix"], "selected_primary_stem": "mix"},
        },
        "missing_deps": dict(missing),
        "availability": {"onsets_frames": bool(oaf_available), "basic_pitch": bool(bp_available)},
    }

    try:
        if pipeline_logger is not None and hasattr(pipeline_logger, "log_event"):
            pipeline_logger.log_event("stage_b_decision_trace", decision_trace=decision_trace)
    except Exception:
        pass

    return decision_trace


def sigmoid(x):
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def weighted_median(values, weights):
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


_LOG2_1200 = 1200.0 / math.log(2.0)
_EPS = 1e-9


def _cents_diff(a_hz: float, b_hz: float) -> float:
    if a_hz <= 0.0 or b_hz <= 0.0:
        return 1e9
    return abs(_LOG2_1200 * math.log((a_hz + _EPS) / (b_hz + _EPS)))


def _maybe_compute_cqt_ctx(audio: np.ndarray, sr: int, hop_length: int,
                           fmin: float, fmax: float,
                           bins_per_octave: int = 36) -> Optional[dict]:
    if not _module_available("librosa"):
        return None
    try:
        import librosa
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return None

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
    col = mag[:, t]
    return float(np.median(col)) + 1e-9


def _postprocess_candidates(
    candidates: List[Tuple[float, float]],
    frame_idx: int,
    cqt_ctx: Optional[dict],
    max_candidates: int,
    dup_cents: float = 35.0,
    octave_cents: float = 35.0,
    cqt_gate_mul: float = 0.25,
    cqt_support_ratio: float = 2.0,
    harmonic_drop_ratio: float = 0.75,
) -> List[Tuple[float, float]]:
    if not candidates:
        return []

    floor = _cqt_frame_floor(cqt_ctx, frame_idx)
    gated: List[Tuple[float, float]] = []
    for f, c in candidates:
        if f <= 0.0 or c <= 0.0:
            continue
        c = float(max(0.0, min(1.0, c)))

        if cqt_ctx is not None and floor > 0.0:
            m = _cqt_mag_at(cqt_ctx, frame_idx, float(f))
            if m < floor * float(cqt_support_ratio):
                c *= float(cqt_gate_mul)

        if c > 0.0:
            gated.append((float(f), float(c)))

    if not gated:
        return []

    gated.sort(key=lambda x: x[1], reverse=True)

    kept: List[Tuple[float, float]] = []

    def _looks_like_harmonic(lo: float, hi: float) -> bool:
        if cqt_ctx is None:
            return False
        if abs(_cents_diff(hi, 2.0 * lo)) > octave_cents:
            return False
        m_lo = _cqt_mag_at(cqt_ctx, frame_idx, lo)
        m_hi = _cqt_mag_at(cqt_ctx, frame_idx, hi)
        if m_lo <= 0.0:
            return False
        return (m_hi / (m_lo + 1e-9)) < float(harmonic_drop_ratio)

    for f, c in gated:
        dup = False
        for fk, _ in kept:
            if _cents_diff(f, fk) <= dup_cents:
                dup = True
                break
        if dup:
            continue

        harm = False
        for fk, ck in kept:
            if fk > 0.0 and f > 0.0 and abs(_cents_diff(f, 2.0 * fk)) <= octave_cents:
                if _looks_like_harmonic(fk, f) and c <= ck * 1.10:
                    harm = True
                    break
        if harm:
            continue

        kept.append((f, c))
        if len(kept) >= int(max(1, max_candidates)):
            break

    if not kept:
        return []

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
        if cents == cents:
            self.recent.append(cents)
        stability = 1.0
        if len(self.recent) >= 3:
            diffs = [abs(self.recent[i] - self.recent[i - 1]) for i in range(1, len(self.recent))]
            stability = 1.0 / (1.0 + (sum(diffs) / len(diffs)) / 80.0)

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

        conf_factor = sigmoid((conf - 0.5) * 6.0)
        w = self.base_w * conf_factor * stability * pop_factor
        return w


def viterbi_pitch(fused_cents, fused_conf, midi_states, transition_smoothness=0.5, jump_penalty=0.6):
    T = len(fused_cents)
    S = len(midi_states)
    INF = 1e18
    state_cents = [m * 100.0 for m in midi_states]
    dp = [[INF] * S for _ in range(T)]
    bp = [[-1] * S for _ in range(T)]

    def emission(t, s):
        c = fused_cents[t]
        if not (c == c) or c <= 0:
            return 5.0
        dist = abs(c - state_cents[s])
        conf = fused_conf[t] if t < len(fused_conf) and fused_conf[t] is not None else 0.5
        return (dist / 50.0) - 0.8 * math.log(max(1e-3, conf))

    def transition(s0, s1):
        step = abs(midi_states[s1] - midi_states[s0])
        if step <= 2:
            return transition_smoothness * step
        return transition_smoothness * 2.5 + jump_penalty * (step - 2)

    if T == 0:
        return []

    for s in range(S):
        dp[0][s] = emission(0, s)

    for t in range(1, T):
        for s1 in range(S):
            e = emission(t, s1)
            best_cost = INF
            best_s0 = -1
            for s0 in range(S):
                cost = dp[t - 1][s0] + transition(s0, s1)
                if cost < best_cost:
                    best_cost = cost
                    best_s0 = s0
            dp[t][s1] = best_cost + e
            bp[t][s1] = best_s0

    s = min(range(S), key=lambda k: dp[T - 1][k])
    path = [s] * T
    for t in range(T - 1, 0, -1):
        s = bp[t][s]
        path[t - 1] = s
    smoothed_hz = [440.0 * (2 ** ((midi_states[i] - 69) / 12.0)) for i in path]
    return smoothed_hz


def _butter_filter(audio: np.ndarray, sr: int, cutoff: float, btype: str) -> np.ndarray:
    if SCIPY_SIGNAL is None or len(audio) == 0:
        return audio.copy()
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    norm_cutoff = min(max(norm_cutoff, 1e-4), 0.999)
    sos = SCIPY_SIGNAL.butter(4, norm_cutoff, btype=btype, output="sos")
    return SCIPY_SIGNAL.sosfiltfilt(sos, audio)


def _estimate_global_tuning_cents(f0: np.ndarray) -> float:
    f = np.asarray(f0, dtype=np.float32)
    f = f[f > 0.0]
    if f.size < 50:
        return 0.0
    midi_float = 69.0 + 12.0 * np.log2(f / 440.0)
    frac = midi_float - np.round(midi_float)
    cents = frac * 100.0
    cents = (cents + 50.0) % 100.0 - 50.0
    return float(np.median(cents))


class SyntheticMDXSeparator:
    """
    Lightweight separator tuned on procedurally generated sine/saw/square/FM stems.
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

        carrier = 220.0
        modulator = 110.0
        waves["fm_voice"] = np.sin(
            2 * np.pi * carrier * t + 5.0 * np.sin(2 * np.pi * modulator * t)
        )

        for name, wave in waves.items():
            templates[name] = _normalize_spec(wave)

        broadband = np.hanning(len(t))
        templates["broadband"] = _normalize_spec(broadband)
        return templates

    def _score_mix(self, audio: np.ndarray) -> Dict[str, float]:
        window = np.hanning(len(audio))
        spec = np.abs(np.fft.rfft(audio * window))
        spec = spec / (np.linalg.norm(spec) + 1e-9)

        scores = {}
        for name, tmpl in self.templates.items():
            if len(tmpl) != len(spec):
                x_old = np.linspace(0.0, 1.0, len(tmpl))
                x_new = np.linspace(0.0, 1.0, len(spec))
                tmpl2 = np.interp(x_new, x_old, tmpl)
                tmpl2 = tmpl2 / (np.linalg.norm(tmpl2) + 1e-9)
            else:
                tmpl2 = tmpl
            scores[name] = float(np.dot(spec, tmpl2))
        return scores

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        if len(audio) == 0:
            return {}

        scores = self._score_mix(audio)
        vocal_score = scores.get("fm_voice", 0.25) + scores.get("sine_stack", 0.25)
        bass_score = scores.get("square", 0.25) + scores.get("saw", 0.25)
        drum_score = scores.get("broadband", 0.25)
        other_score = 0.01

        raw_weights = np.array([vocal_score, bass_score, drum_score, other_score], dtype=np.float32)
        weights = raw_weights / (np.sum(raw_weights) + 1e-9)
        vocals_w, bass_w, drums_w, other_w = weights

        if SCIPY_SIGNAL is None:
            if not getattr(self, "_warned_no_scipy", False):
                logger.warning("SyntheticMDX: Scipy missing. Falling back to gain-based separation (no frequency filtering).")
                self._warned_no_scipy = True

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

    if audio.ndim == 1:
        audio = audio[None, :]

    if model_sr != sr:
        ratio = float(model_sr) / float(sr)
        new_len = int(audio.shape[-1] * ratio)
        resampled_channels = []
        for ch in range(audio.shape[0]):
            indices = np.arange(0, new_len) / ratio
            indices = np.clip(indices, 0, audio.shape[-1] - 1)
            res_ch = np.interp(indices, np.arange(audio.shape[-1]), audio[ch])
            resampled_channels.append(res_ch)
        resampled = np.stack(resampled_channels)
    else:
        resampled = audio

    if resampled.ndim == 2:
        d0, d1 = resampled.shape
        if d0 > 10 and d1 <= 10:
            resampled = resampled.T

    if resampled.ndim == 1:
        resampled = resampled[None, :]

    C, _T = resampled.shape

    if C == 1:
        resampled = np.concatenate([resampled, resampled], axis=0)
    elif C > 2:
        resampled = resampled[:2, :]

    if resampled.shape[0] != 2:
        logger.warning(f"HTDemucs input shape unexpected: {resampled.shape}. Forcing stereo duplication.")
        if resampled.shape[0] == 1:
            resampled = np.concatenate([resampled, resampled], axis=0)
        else:
            mono = np.mean(resampled, axis=0, keepdims=True)
            resampled = np.concatenate([mono, mono], axis=0)

    mix_tensor = torch.tensor(resampled, dtype=torch.float32)[None, :, :].to(dev)

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

    for name in ["vocals", "drums", "bass", "other"]:
        separated.setdefault(name, np.zeros_like(audio.reshape(-1)))

    return separated


def _resolve_separation(
    stage_a_out: StageAOutput,
    b_conf,
    device: str = "cpu",
    *,
    routing_features: Optional[Dict[str, Any]] = None,
    resolved_transcription_mode: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Benchmark-safe separation routing.

    Key fix:
      - If synthetic_model is enabled, DO NOT skip just because synthetic_like=True or mixture_score is low.
        Those gates are appropriate for Demucs, not for SyntheticMDX.
    """
    sep = getattr(b_conf, "separation", {}) or {}

    mode = sep.get("mode", None)
    enabled = sep.get("enabled", True)
    sep_enabled = enabled

    if isinstance(mode, str):
        sep_mode = mode.lower().strip()
    else:
        sep_mode = "auto" if bool(enabled) else "off"
        if isinstance(enabled, str) and enabled.lower().strip() == "auto":
            sep_mode = "auto"
    if sep_mode not in ("off", "auto", "demucs"):
        sep_mode = "auto"

    rf = routing_features or {}
    duration_sec = float(getattr(getattr(stage_a_out, "meta", None), "duration_sec", 0.0) or 0.0)
    synthetic_like = bool(rf.get("synthetic_like", False))
    mixture_score = float(rf.get("mixture_score", 0.0) or 0.0)

    gates = sep.get("gates", {}) or {}
    min_duration_sec = float(gates.get("min_duration_sec", 8.0))
    min_mixture_score = float(gates.get("min_mixture_score", 0.45))
    bypass_if_synth = bool(gates.get("bypass_if_synthetic_like", True))

    # Determine backend intent early so gating can be correct
    use_synth = bool(sep.get("synthetic_model", False))
    backend = "synthetic_mdx" if use_synth else "demucs"

    def _skip(reason_list: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        stems = getattr(stage_a_out, "stems", {}) or {}
        mix_only = {"mix": stems.get("mix", None)} if "mix" in stems else stems
        return (
            mix_only,
            {
                "ran": False,
                "requested": bool(sep_enabled) if "sep_enabled" in locals() else False,
                "backend": "none",
                "skip_reasons": list(reason_list),
                "gates": {
                    "min_duration_sec": float(min_duration_sec),
                    "min_mixture_score": float(min_mixture_score),
                    "bypass_if_synthetic_like": bool(bypass_if_synth),
                },
                "outputs": {"stems": ["mix"], "selected_primary_stem": "mix"},
                "mode": "disabled",
                "synthetic_ran": False,
                "htdemucs_ran": False,
                "fallback": False,
            },
        )

    if str(resolved_transcription_mode or "").startswith("e2e_"):
        return _skip(["e2e_mode"])

    if sep_mode == "off":
        return _skip(["off"])

    skip_reasons: List[str] = []

    # Duration gate applies to both backends
    if duration_sec > 0.0 and duration_sec < min_duration_sec:
        skip_reasons.append("short_duration")

    # Demucs-specific gates (do NOT apply to synthetic backend by default)
    if not use_synth:
        if bypass_if_synth and synthetic_like:
            skip_reasons.append("synthetic_like")
        if mixture_score < min_mixture_score:
            skip_reasons.append("low_mixture_score")

    if skip_reasons:
        return _skip(skip_reasons)

    mix_stem = (getattr(stage_a_out, "stems", {}) or {}).get("mix", None)
    if mix_stem is None:
        return _skip(["no_mix_stem"])
    mix_audio = getattr(mix_stem, "audio", None)
    if mix_audio is None:
        return _skip(["no_audio"])

    sep_conf = getattr(b_conf, "separation", {}) or {}
    overlap = float(sep_conf.get("overlap", 0.25))
    shifts = int(sep_conf.get("shifts", 1))

    preset_conf: Dict[str, Any] = {}
    preset_name = None
    if getattr(stage_a_out, "audio_type", None) == AudioType.POLYPHONIC_DOMINANT:
        preset_conf = sep_conf.get("polyphonic_dominant_preset", {}) or {}
        preset_name = "polyphonic_dominant"
        overlap = float(preset_conf.get("overlap", overlap))
        if "shifts" in preset_conf:
            shifts = int(preset_conf.get("shifts", shifts))
        else:
            shift_range = preset_conf.get("shift_range")
            if shift_range and isinstance(shift_range, (list, tuple)):
                try:
                    shifts = int(max(shift_range))
                except (TypeError, ValueError):
                    shifts = int(shifts)

    diag_extras = {
        "preset": preset_name,
        "resolved_overlap": overlap,
        "resolved_shifts": shifts,
        "shift_range": preset_conf.get("shift_range", None),
    }

    try:
        raw_stems = {}
        if use_synth:
            model = SyntheticMDXSeparator(
                sample_rate=mix_stem.sr,
                hop_length=getattr(getattr(stage_a_out, "meta", None), "hop_length", 512),
            )
            raw_stems = model.separate(np.asarray(mix_audio, dtype=np.float32), mix_stem.sr)
        else:
            raw_stems = _run_htdemucs(
                np.asarray(mix_audio, dtype=np.float32),
                mix_stem.sr,
                str(sep_conf.get("model_name", "htdemucs")),
                overlap,
                shifts,
                device,
            )

        if not raw_stems:
            return _skip(["backend_returned_empty"])

        target_sr = mix_stem.sr
        target_len = len(mix_audio)

        final_stems: Dict[str, Stem] = {}
        final_stems["mix"] = mix_stem

        for k, v in raw_stems.items():
            if not isinstance(v, np.ndarray):
                continue
            out_audio = np.asarray(v, dtype=np.float32).reshape(-1)

            if len(out_audio) != target_len and target_len > 0 and len(out_audio) > 0:
                x_old = np.linspace(0, 1, len(out_audio))
                x_new = np.linspace(0, 1, target_len)
                out_audio = np.interp(x_new, x_old, out_audio).astype(np.float32)

            final_stems[k] = type(mix_stem)(audio=out_audio, sr=target_sr, type=k)

        stem_names = sorted(list(final_stems.keys()))
        selected = "mix"
        if "vocals" in final_stems:
            selected = "vocals"
        elif "other" in final_stems:
            selected = "other"
        elif stem_names:
            selected = stem_names[0]

        return final_stems, {
            "ran": True,
            "requested": bool(sep_enabled),
            "backend": backend,
            "skip_reasons": [],
            "gates": {
                "min_duration_sec": float(min_duration_sec),
                "min_mixture_score": float(min_mixture_score),
                "bypass_if_synthetic_like": bool(bypass_if_synth),
            },
            "outputs": {"stems": stem_names, "selected_primary_stem": str(selected)},
            "mode": "synthetic_mdx" if use_synth else sep_conf.get("model_name", "htdemucs"),
            "synthetic_ran": bool(use_synth),
            "htdemucs_ran": not bool(use_synth),
            "fallback": False,
            **diag_extras
        }

    except Exception:
        d = _skip(["runtime_error"])[1]
        d["fallback"] = True
        if sep_mode == "auto":
            d["fallback_reason"] = "htdemucs_failed_or_missing_in_auto"
        return _skip(["runtime_error"])[0], d


def _arrays_to_timeline(
    f0: np.ndarray,
    conf: np.ndarray,
    rms: Optional[np.ndarray],
    sr: int,
    hop_length: int
) -> List[FramePitch]:
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
    filtered = [np.median(padded[i: i + k]) for i in range(len(signal))]
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
    if not conf.get("enabled", False):
        return None

    kwargs = {k: v for k, v in conf.items() if k not in ("enabled", "hop_length")}
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

            eff_conf = max(c, priority_floor if name == "swiftf0" else c)
            candidates.append((name, f, eff_conf, w))

        if not candidates:
            final_f0[i] = 0.0
            final_conf[i] = 0.0
            continue

        if adaptive_fusion:
            vals, wts = [], []
            for out in detector_outputs:
                cents = out["cents"]
                if not out.get("voiced", True) or not (cents == cents):
                    continue
                w = reliabilities[out["name"]].update(cents, out["conf"], energy_ok=True)
                if w > 1e-6:
                    vals.append(cents)
                    wts.append(w)

            fused_cents = weighted_median(vals, wts) if vals else float("nan")

            if fused_cents == fused_cents:
                fused_hz = 440.0 * (2.0 ** ((fused_cents - 6900.0) / 1200.0))
                final_f0[i] = fused_hz
                final_conf[i] = sum(c[2] for c in candidates) / len(candidates)
            else:
                final_f0[i] = 0.0
                final_conf[i] = 0.0

        else:
            best_name, best_f0, best_conf, best_w = max(candidates, key=lambda x: x[2] * x[3])

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
        return {}


def _resolve_polyphony_filter(config: Optional[PipelineConfig]) -> str:
    try:
        return str(config.stage_c.polyphony_filter.get("mode", "skyline_top_voice"))
    except Exception:
        return "skyline_top_voice"


class MultiVoiceTracker:
    """
    Lightweight multi-voice tracker to keep skyline assignments stable.
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
            carry_pitches = np.where(self.hold > 0, self.prev_pitches, 0.0)
            carry_confs = np.where(self.hold > 0, self.prev_confs * 0.9, 0.0)
            self.hold = np.maximum(self.hold - 1, 0)
            self.prev_pitches = carry_pitches.astype(np.float32)
            self.prev_confs = carry_confs.astype(np.float32)
            return self.prev_pitches.copy(), self.prev_confs.copy()

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
        try:
            if pipeline_logger and hasattr(pipeline_logger, "log_event"):
                pipeline_logger.log_event(
                    "stage_b_config_unknown_keys",
                    section=name,
                    keys=sorted(list(unknown)),
                )
            else:
                logger.warning(msg)
        except Exception:
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
    if config is None:
        config = PipelineConfig()

    b_conf = config.stage_b
    sr = stage_a_out.meta.sample_rate
    hop_length = stage_a_out.meta.hop_length

    instrument = (
        kwargs.get("instrument")
        or (getattr(config, "instrument_override", None) if config else None)
        or (getattr(config, "intended_instrument", None) if config else None)
        or getattr(stage_a_out.meta, "instrument", None)
        or b_conf.instrument
    )

    decision_trace = compute_decision_trace(
        stage_a_out,
        config,
        requested_mode=getattr(b_conf, "transcription_mode", "classic"),
        requested_profile=str(instrument),
        requested_separation_mode=_normalize_separation_mode(b_conf),
        pipeline_logger=pipeline_logger,
    )
    resolved_mode = str(decision_trace.get("resolved", {}).get("transcription_mode", getattr(b_conf, "transcription_mode", "classic")))
    routing_features = dict(decision_trace.get("routing_features", {}) or {})

    transcription_mode = resolved_mode

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

                return StageBOutput(
                    time_grid=np.array([], dtype=np.float32),
                    f0_main=np.array([], dtype=np.float32),
                    f0_layers=[],
                    stem_timelines={},
                    per_detector={},
                    meta=stage_a_out.meta,
                    diagnostics={"stage_b_mode": "e2e_basic_pitch", "decision_trace": decision_trace},
                    precalculated_notes=notes,
                )
        except Exception as e:
            logger.warning(f"Basic Pitch unavailable/failed; falling back to classic: {e}")

    profile = config.get_profile(str(instrument)) if (instrument and b_conf.apply_instrument_profile) else None
    profile_special = dict(getattr(profile, "special", {}) or {}) if profile else {}
    profile_applied = bool(profile)

    detector_cfgs = deepcopy(b_conf.detectors)
    weights_eff = dict(b_conf.ensemble_weights)
    melody_filter_eff = dict(getattr(b_conf, "melody_filtering", {}) or {})

    enabled_dets = {k for k, v in detector_cfgs.items() if v.get("enabled", False)}
    weights_eff = {k: v for k, v in weights_eff.items() if k in enabled_dets}

    common_keys = {"enabled", "fmin", "fmax", "hop_length", "frame_length", "threshold"}
    _validate_config_keys("detectors.crepe", detector_cfgs.get("crepe", {}),
                          common_keys | {"model_capacity", "step_ms", "confidence_threshold", "conf_threshold", "use_viterbi"}, pipeline_logger)
    _validate_config_keys("detectors.yin", detector_cfgs.get("yin", {}),
                          common_keys | {"enable_multires_f0", "enable_octave_correction", "octave_jump_penalty", "trough_threshold"}, pipeline_logger)
    _validate_config_keys("detectors.swiftf0", detector_cfgs.get("swiftf0", {}),
                          common_keys | {"confidence_threshold", "n_fft"}, pipeline_logger)

    nested = profile_special.get("stage_b_detectors") or {}
    for det_name, overrides in (nested.items() if isinstance(nested, dict) else []):
        if det_name in detector_cfgs and isinstance(overrides, dict):
            detector_cfgs[det_name].update(overrides)

    if profile:
        for _, dconf in detector_cfgs.items():
            dconf.setdefault("fmin", float(profile.fmin))
            dconf.setdefault("fmax", float(profile.fmax))
            dconf["fmin"] = float(profile.fmin)
            dconf["fmax"] = float(profile.fmax)

        melody_filter_eff["fmin_hz"] = float(profile.fmin)
        melody_filter_eff["fmax_hz"] = float(profile.fmax)

        rec = (profile.recommended_algo or "").lower()
        if rec and rec != "none" and rec in detector_cfgs:
            detector_cfgs[rec]["enabled"] = True

        if "yin_trough_threshold" in profile_special and "yin" in detector_cfgs:
            detector_cfgs["yin"]["trough_threshold"] = float(profile_special["yin_trough_threshold"])
        if "yin_conf_threshold" in profile_special and "yin" in detector_cfgs:
            detector_cfgs["yin"]["threshold"] = float(profile_special["yin_conf_threshold"])
        if "yin_frame_length" in profile_special and "yin" in detector_cfgs:
            detector_cfgs["yin"]["frame_length"] = int(profile_special["yin_frame_length"])

        if "vibrato_smoothing_ms" in profile_special:
            ms = float(profile_special["vibrato_smoothing_ms"])
            frame_ms = 1000.0 * float(hop_length) / float(sr)
            win = int(round(ms / max(frame_ms, 1e-6)))
            if win % 2 == 0:
                win += 1
            current = int(melody_filter_eff.get("median_window", 1) or 1)
            melody_filter_eff["median_window"] = max(current, win)

    device = getattr(config, "device", "cpu")
    resolved_stems, separation_diag = _resolve_separation(
        stage_a_out,
        b_conf,
        device=device,
        routing_features=routing_features,
        resolved_transcription_mode=resolved_mode,
    )
    try:
        decision_trace["separation"] = separation_diag
    except Exception:
        pass

    whitelist = getattr(b_conf, "active_stems", None)
    if whitelist is not None:
        filtered_stems = {}
        for sname, sobj in resolved_stems.items():
            if sname == "mix":
                filtered_stems[sname] = sobj
            elif sname in whitelist:
                filtered_stems[sname] = sobj
        resolved_stems = filtered_stems

    detectors: Dict[str, BasePitchDetector] = {}
    for name, det_conf in detector_cfgs.items():
        det = _init_detector(name, det_conf, sr, hop_length)
        if det:
            detectors[name] = det

    if not detectors:
        logger.warning("No detectors enabled or initialized in Stage B. Falling back to default YIN.")
        detectors["yin"] = YinDetector(sr, hop_length)

    stem_timelines: Dict[str, List[FramePitch]] = {}
    per_detector: Dict[str, Any] = {}
    f0_main: Optional[np.ndarray] = None
    all_layers: List[np.ndarray] = []
    iss_total_layers = 0

    # debug containers (avoid locals() hacks)
    stem_debug_curves: Dict[str, Any] = {}
    layer_conf_summaries: Dict[str, Any] = {}
    tuning_cents_by_stem: Dict[str, float] = {}

    polyphonic_context = _is_polyphonic(getattr(stage_a_out, "audio_type", None))
    skyline_mode = _resolve_polyphony_filter(config)
    tracker_cfg = getattr(b_conf, "voice_tracking", {}) or {}

    # Harmonic masking config
    sep_conf = getattr(b_conf, "separation", {}) or {}
    harmonic_cfg = sep_conf.get("harmonic_masking", {}) or {}

    augmented_stems = dict(resolved_stems)
    harmonic_mask_applied = False
    if bool(harmonic_cfg.get("enabled", False)) and "mix" in augmented_stems:
        prior_det = detectors.get("swiftf0")
        if prior_det is None:
            prior_conf = dict(getattr(b_conf, "detectors", {}).get("swiftf0", {}) or {})
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
            harmonic_mask_applied = bool(synthetic)

    stems_for_processing = augmented_stems
    polyphonic_context = polyphonic_context or (len(stems_for_processing) > 1) or bool(getattr(b_conf, "polyphonic_peeling", {}).get("force_on_mix", False))

    # Canonical n_frames aligned to framing logic (reduces mismatch churn)
    mix_stem_ref = stage_a_out.stems.get("mix") or next(iter(stage_a_out.stems.values()))
    n_fft_ref = int(stage_a_out.meta.window_size or 2048)
    frames_ref = _frame_audio(np.asarray(mix_stem_ref.audio, dtype=np.float32), n_fft_ref, hop_length)
    canonical_n_frames = int(frames_ref.shape[0])

    for stem_name, stem in stems_for_processing.items():
        audio = np.asarray(stem.audio, dtype=np.float32).reshape(-1)
        per_detector[stem_name] = {}

        if profile_special.get("ignore_pitch", False):
            merged_f0 = np.zeros(canonical_n_frames, dtype=np.float32)
            merged_conf = np.zeros(canonical_n_frames, dtype=np.float32)
            stem_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        else:
            stem_results = {}

            audio_in = audio
            pre_lpf_hz = float(profile_special.get("pre_lpf_hz", 0.0) or 0.0)
            if pre_lpf_hz > 0.0 and SCIPY_SIGNAL is not None:
                audio_in = _butter_filter(audio_in, sr, pre_lpf_hz, "low")

            for name, det in detectors.items():
                try:
                    f0, conf = det.predict(audio_in, audio_path=stage_a_out.meta.audio_path)

                    if name == "swiftf0":
                        thr = float(detector_cfgs.get("swiftf0", {}).get("confidence_threshold", 0.9))
                        conf = np.where(conf >= thr, conf, 0.0)
                        f0 = np.where(conf > 0.0, f0, 0.0)

                    stem_results[name] = (f0, conf)
                    per_detector[stem_name][name] = (f0, conf)
                except Exception as e:
                    logger.warning(f"Detector {name} failed on stem {stem_name}: {e}")

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
            merged_f0 = np.zeros(canonical_n_frames, dtype=np.float32)
            merged_conf = np.zeros(canonical_n_frames, dtype=np.float32)

        if len(merged_f0) != canonical_n_frames:
            if len(merged_f0) < canonical_n_frames:
                pad_len = canonical_n_frames - len(merged_f0)
                merged_f0 = np.pad(merged_f0, (0, pad_len), constant_values=0.0)
                merged_conf = np.pad(merged_conf, (0, pad_len), constant_values=0.0)
            else:
                merged_f0 = merged_f0[:canonical_n_frames]
                merged_conf = merged_conf[:canonical_n_frames]

        tuning_cents = _estimate_global_tuning_cents(merged_f0)
        tuning_cents_by_stem[stem_name] = float(tuning_cents)

        # RMS aligned to canonical length
        n_fft = int(stage_a_out.meta.window_size or 2048)
        frames = _frame_audio(audio, n_fft, hop_length)
        rms_vals = np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)
        if len(rms_vals) < canonical_n_frames:
            rms_vals = np.pad(rms_vals, (0, canonical_n_frames - len(rms_vals)))
        elif len(rms_vals) > canonical_n_frames:
            rms_vals = rms_vals[:canonical_n_frames]

        # ISS peeling (FIXED: correct call signature / indentation)
        iss_layers: List[Tuple[np.ndarray, np.ndarray]] = []
        poly_peel = getattr(b_conf, "polyphonic_peeling", {}) or {}
        if polyphonic_context and int(poly_peel.get("max_layers", 0) or 0) > 0:
            primary = detectors.get("swiftf0") or detectors.get("yin") or detectors.get("sacf")
            validator = detectors.get("sacf") or detectors.get("yin")
            if primary:
                try:
                    iss_layers = iterative_spectral_subtraction(
                        audio,
                        sr,
                        primary_detector=primary,
                        validator_detector=validator,
                        max_polyphony=int(poly_peel.get("max_layers", 4)),
                        mask_width=float(poly_peel.get("mask_width", 0.03)),
                        min_mask_width=float(poly_peel.get("min_mask_width", 0.02)),
                        max_mask_width=float(poly_peel.get("max_mask_width", 0.08)),
                        mask_growth=float(poly_peel.get("mask_growth", 1.1)),
                        mask_shrink=float(poly_peel.get("mask_shrink", 0.9)),
                        harmonic_snr_stop_db=float(poly_peel.get("harmonic_snr_stop_db", 3.0)),
                        residual_rms_stop_ratio=float(poly_peel.get("residual_rms_stop_ratio", 0.08)),
                        residual_flatness_stop=float(poly_peel.get("residual_flatness_stop", 0.45)),
                        validator_cents_tolerance=float(poly_peel.get("validator_cents_tolerance", b_conf.pitch_disagreement_cents)),
                        validator_agree_window=int(poly_peel.get("validator_agree_window", 5)),
                        validator_disagree_decay=float(poly_peel.get("validator_disagree_decay", 0.6)),
                        validator_min_agree_frames=int(poly_peel.get("validator_min_agree_frames", 2)),
                        validator_min_disagree_frames=int(poly_peel.get("validator_min_disagree_frames", 2)),
                        max_harmonics=int(poly_peel.get("max_harmonics", 12)),
                        audio_path=stage_a_out.meta.audio_path,
                        iss_adaptive=bool(poly_peel.get("iss_adaptive", False)),
                        strength_min=float(poly_peel.get("strength_min", 0.8)),
                        strength_max=float(poly_peel.get("strength_max", 1.2)),
                        flatness_thresholds=poly_peel.get("flatness_thresholds", [0.3, 0.6]),
                        use_freq_aware_masks=bool(poly_peel.get("use_freq_aware_masks", True)),
                    )
                    for f0_l, _c_l in iss_layers:
                        all_layers.append(np.asarray(f0_l, dtype=np.float32))
                    iss_total_layers += len(iss_layers)
                except Exception as e:
                    logger.warning(f"ISS peeling failed for stem {stem_name}: {e}")

        # Transient lockout
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
                lock_mask[idx: min(idx + lockout_frames, len(lock_mask))] = True

            merged_conf = np.where(lock_mask, 0.0, merged_conf)
            merged_f0 = np.where(lock_mask, 0.0, merged_f0)

            masked_iss_layers = []
            for f0_l, c_l in iss_layers:
                c_l2 = np.asarray(c_l, dtype=np.float32)
                f0_l2 = np.asarray(f0_l, dtype=np.float32)
                L = min(len(lock_mask), len(c_l2))
                if L > 0:
                    c_l2[:L] = np.where(lock_mask[:L], 0.0, c_l2[:L])
                    f0_l2[:L] = np.where(c_l2[:L] > 0.0, f0_l2[:L], 0.0)
                masked_iss_layers.append((f0_l2, c_l2))
            iss_layers = masked_iss_layers

            for det_name in list(per_detector[stem_name].keys()):
                pf0, pconf = per_detector[stem_name][det_name]
                if len(pconf) == len(lock_mask):
                    pconf2 = np.where(lock_mask, 0.0, pconf)
                    pf02 = np.where(lock_mask, 0.0, pf0)
                    per_detector[stem_name][det_name] = (pf02, pconf2)

        merged_f0, merged_conf = _apply_melody_filters(
            merged_f0,
            merged_conf,
            rms_vals,
            melody_filter_eff,
        )

        fused_f0_debug = merged_f0.copy()

        use_viterbi_smoothing = getattr(b_conf, "smoothing_method", "tracker") == "viterbi"
        if use_viterbi_smoothing and np.any(merged_f0 > 0):
            midi_states = list(range(21, 109))
            fused_cents = []
            for f in merged_f0:
                if f > 0:
                    fused_cents.append(1200.0 * math.log2(f / 440.0) + 6900.0)
                else:
                    fused_cents.append(float("nan"))

            smoothed_hz = viterbi_pitch(
                fused_cents,
                merged_conf,
                midi_states,
                transition_smoothness=getattr(b_conf, "viterbi_transition_smoothness", 0.5),
                jump_penalty=getattr(b_conf, "viterbi_jump_penalty", 0.6)
            )
            if len(smoothed_hz) == len(merged_f0):
                merged_f0 = np.array(smoothed_hz, dtype=np.float32)

        voicing_thr_global = float(getattr(b_conf, "confidence_voicing_threshold", 0.0) or 0.0)
        is_true_poly = _is_polyphonic(getattr(stage_a_out, "audio_type", None))
        poly_relax = float(getattr(b_conf, "polyphonic_voicing_relaxation", 0.0) or 0.0) if is_true_poly else 0.0
        voicing_thr = voicing_thr_global - poly_relax

        stem_debug_curves[stem_name] = {
            "fused_f0": _curve_summary(fused_f0_debug),
            "smoothed_f0": _curve_summary(merged_f0),
        }

        layer_arrays = [(merged_f0, merged_conf)] + iss_layers
        max_frames = max(len(arr[0]) for arr in layer_arrays) if layer_arrays else canonical_n_frames

        def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
            arr = np.asarray(arr, dtype=np.float32)
            if len(arr) < target:
                return np.pad(arr, (0, target - len(arr)))
            return arr[:target]

        padded_layers = [(_pad_to(f0, max_frames), _pad_to(conf, max_frames)) for f0, conf in layer_arrays]
        padded_rms = _pad_to(rms_vals, max_frames)

        cqt_ctx = None
        cqt_gate_enabled = bool((getattr(b_conf, "polyphonic_peeling", {}) or {}).get("cqt_gate_enabled", True))
        if is_true_poly and polyphonic_context and cqt_gate_enabled:
            fmin_gate = float(melody_filter_eff.get("fmin_hz", 60.0) or 60.0)
            fmax_gate = float(melody_filter_eff.get("fmax_hz", 2200.0) or 2200.0)
            cqt_ctx = _maybe_compute_cqt_ctx(
                audio, sr, hop_length,
                fmin=fmin_gate, fmax=fmax_gate,
                bins_per_octave=int((getattr(b_conf, "polyphonic_peeling", {}) or {}).get("cqt_bins_per_octave", 36)),
            )

        try:
            layer_conf_summaries[stem_name] = [_curve_summary(conf_arr) for (_, conf_arr) in padded_layers]
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

        select_top_voice = is_true_poly and polyphonic_context and "top_voice" in str(skyline_mode)
        tuning_semitones = float(tuning_cents) / 100.0

        poly_peel2 = getattr(b_conf, "polyphonic_peeling", {}) or {}
        max_cands = int(poly_peel2.get("max_candidates_per_frame", tracker.max_tracks))

        for i in range(max_frames):
            candidates: List[Tuple[float, float]] = []
            for f0_arr, conf_arr in padded_layers:
                f = float(f0_arr[i]) if i < len(f0_arr) else 0.0
                c = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                if f > 0.0 and c >= voicing_thr:
                    candidates.append((f, c))

            candidates = _postprocess_candidates(
                candidates=candidates,
                frame_idx=i,
                cqt_ctx=cqt_ctx,
                max_candidates=max_cands,
                dup_cents=float(poly_peel2.get("dup_cents", 35.0)),
                octave_cents=float(poly_peel2.get("octave_cents", 35.0)),
                cqt_gate_mul=float(poly_peel2.get("cqt_gate_mul", 0.25)),
                cqt_support_ratio=float(poly_peel2.get("cqt_support_ratio", 2.0)),
                harmonic_drop_ratio=float(poly_peel2.get("harmonic_drop_ratio", 0.75)),
            )

            active_candidates = list(candidates)

            tracked_pitches, tracked_confs = tracker.step(candidates)
            for voice_idx in range(tracker.max_tracks):
                track_buffers[voice_idx][i] = tracked_pitches[voice_idx]
                track_conf_buffers[voice_idx][i] = tracked_confs[voice_idx]

            primary_idx = 0
            if select_top_voice and tracked_pitches.size:
                prev_p = primary_track[i - 1] if i > 0 else 0.0
                best_score = -999.0
                best_idx = 0
                found_valid = False

                for idx in range(len(tracked_confs)):
                    p = float(tracked_pitches[idx])
                    c = float(tracked_confs[idx])
                    if p <= 0.0:
                        continue
                    found_valid = True
                    score = c
                    if prev_p > 0.0:
                        cents = abs(1200.0 * np.log2(p / prev_p))
                        score -= cents * 0.0005
                    if score > best_score:
                        best_score = score
                        best_idx = idx

                if found_valid:
                    primary_idx = best_idx

            chosen_pitch = float(tracked_pitches[primary_idx]) if tracked_pitches.size else 0.0
            chosen_conf = float(tracked_confs[primary_idx]) if tracked_confs.size else 0.0
            primary_track[i] = chosen_pitch

            midi = None
            if chosen_pitch > 0.0:
                midi_float = 69.0 + 12.0 * np.log2(chosen_pitch / 440.0)
                midi = int(round(midi_float - tuning_semitones))

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

        stem_timelines[stem_name] = timeline

        for alt in track_buffers[1:]:
            if np.count_nonzero(alt) > 0:
                all_layers.append(alt)

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

    diagnostics = {
        "transcription_mode": resolved_mode,
        "stage_b_mode": "classic",
        "instrument": str(instrument),
        "profile": profile.instrument if profile else None,
        "profile_applied": bool(profile_applied),
        "profile_special": dict(getattr(profile, "special", {}) or {}) if profile else {},
        "polyphonic_context": bool(polyphonic_context),
        "detectors_initialized": list(detectors.keys()),
        "separation": separation_diag,
        "harmonic_masking": {
            "enabled": bool(harmonic_cfg.get("enabled", False)),
            "applied": bool(harmonic_mask_applied),
            "mask_width": harmonic_cfg.get("mask_width"),
            "n_harmonics": harmonic_cfg.get("n_harmonics"),
        },
        "iss": {
            "enabled": bool(polyphonic_context and int((getattr(b_conf, "polyphonic_peeling", {}) or {}).get("max_layers", 0) or 0) > 0),
            "layers_found": int(iss_total_layers),
            "max_layers": int((getattr(b_conf, "polyphonic_peeling", {}) or {}).get("max_layers", 0) or 0),
        },
        "cqt_gate": {
            "requested": bool(cqt_gate_enabled),
            "active": bool(cqt_ctx is not None),
            "librosa_available": bool(_module_available("librosa")),
        },
        "skyline_mode": skyline_mode,
        "voice_tracking": {
            "max_alt_voices": int(tracker_cfg.get("max_alt_voices", 4) if polyphonic_context else 0),
            "max_jump_cents": tracker_cfg.get("max_jump_cents", 150.0),
        },
        "global_tuning_cents_by_stem": tuning_cents_by_stem,
        "debug_curves": stem_debug_curves,
        "layer_conf_summaries": layer_conf_summaries,
        "decision_trace": decision_trace,
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
