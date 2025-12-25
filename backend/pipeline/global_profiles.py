"""Global transcription profile auto-router.

Goal
----
Provide a *single* global knob that can automatically pick a sensible
transcription strategy (and tune Stage B/C knobs) based on the imported audio.

Supported user-facing modes
---------------------------
You can set any of these via:
- PipelineConfig.transcription_mode (preferred), OR
- PipelineConfig.global_transcription_mode (legacy), OR
- env var MUSICNOTE_MODE

Values:
- "auto"             : choose among the modes below
- "melody"           : classic detectors tuned for monophonic / lead melody
- "piano_poly"       : classic detectors tuned for dense harmonic/polyphonic instruments
- "song"             : classic detectors tuned for full mixes (often percussive + wideband)
- "classic"          : do not override (use whatever your config already says)
- "e2e_basic_pitch"  : force end-to-end Basic Pitch (if available; else falls back)

Back-compat aliases:
- "piano" -> "piano_poly"
"""

from __future__ import annotations

import os
import importlib.util
from typing import Any, Optional, Tuple


# --------------------------
# Small helpers
# --------------------------

def _get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _set_attr(obj: Any, key: str, value: Any) -> None:
    if obj is None:
        return
    if isinstance(obj, dict):
        obj[key] = value
        return
    try:
        setattr(obj, key, value)
    except Exception:
        # Some configs may be frozen/slots; best-effort.
        return


def _basic_pitch_available() -> bool:
    # Prefer the project wrapper if present; otherwise check import spec.
    try:
        spec = importlib.util.find_spec("basic_pitch")
        if spec is not None:
            return True
    except Exception:
        pass
    # Some installs expose only basic_pitch.inference
    try:
        spec = importlib.util.find_spec("basic_pitch.inference")
        if spec is not None:
            return True
    except Exception:
        pass
    return False


def _safe_audio_summary(stage_a_out: Any) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Return (duration_sec, sr, audio_type_str) best-effort."""
    dur = _get(stage_a_out, "meta.duration_sec", None)
    sr = _get(stage_a_out, "meta.sample_rate", None)
    at = _get(stage_a_out, "audio_type.value", None) or _get(stage_a_out, "audio_type", None)
    if isinstance(at, str):
        at_str = at
    else:
        at_str = str(at) if at is not None else None
    try:
        dur = float(dur) if dur is not None else None
    except Exception:
        dur = None
    try:
        sr = int(sr) if sr is not None else None
    except Exception:
        sr = None
    return dur, sr, at_str


def _compute_percussive_ratio(y, sr: int) -> Optional[float]:
    """Return percussive energy ratio in [0,1] using HPSS (if librosa available)."""
    try:
        import numpy as np
        import librosa
    except Exception:
        return None

    if y is None or sr is None:
        return None

    try:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # cap for speed
        max_samp = int(sr * 30.0)
        if y.size > max_samp:
            y = y[:max_samp]

        # robust stft
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
        H, P = librosa.decompose.hpss(S)
        e_h = float(np.sum(H))
        e_p = float(np.sum(P))
        tot = e_h + e_p + 1e-9
        return float(e_p / tot)
    except Exception:
        return None


def _is_song_like(audio_path: str, stage_a_out: Any, percussive_ratio: Optional[float]) -> bool:
    ext = os.path.splitext(audio_path or "")[1].lower()
    if ext in (".mp3", ".m4a", ".aac", ".ogg", ".opus"):
        return True

    stems = _get(stage_a_out, "stems", {}) or {}
    # If Stage A already produced Demucs-like stems, treat as song-ish.
    if isinstance(stems, dict) and any(k in stems for k in ("vocals", "bass", "drums", "other")):
        return True

    if percussive_ratio is not None and percussive_ratio >= 0.45:
        return True

    return False


# --------------------------
# Profiles (classic tuning)
# --------------------------

PROFILES = {
    "melody": {
        "stage_b": {
            "transcription_mode": "classic",
            "separation": {"enabled": False},
            "voice_tracking": {"max_alt_voices": 1, "skyline_mode": "top_voice"},
        },
        "stage_c": {
            "polyphony_filter": {"mode": "skyline_top_voice"},
            "stem_selection": {"prefer_order": ["mix", "other", "vocals", "melody_masked"], "mix_margin": 0.02},
        },
    },
    "piano_poly": {
        "stage_b": {
            "transcription_mode": "classic",
            "separation": {"enabled": False},
            "polyphonic_peeling": {"iss_adaptive": True},
            "voice_tracking": {"max_alt_voices": 6, "skyline_mode": "multi_voice"},
        },
        "stage_c": {
            "polyphony_filter": {"mode": "pianoroll_chords", "pianoroll_max_notes_per_frame": 10},
            "stem_selection": {"prefer_order": ["mix", "other", "vocals", "melody_masked"], "mix_margin": 0.0},
        },
    },
    "song": {
        "stage_b": {
            "transcription_mode": "classic",
            # run Demucs when available, otherwise stay on mix (auto is safe for synthetic too)
            "separation": {"enabled": "auto"},
            "voice_tracking": {"max_alt_voices": 4, "skyline_mode": "top_voice"},
        },
        "stage_c": {
            "polyphony_filter": {"mode": "skyline_top_voice"},
            "stem_selection": {"prefer_order": ["mix", "other", "vocals", "melody_masked"], "mix_margin": 0.02},
        },
    },
}



def _apply_profile(config: Any, profile_name: str) -> None:
    profile = PROFILES.get(profile_name)
    if not profile:
        return

    sb = _get(config, "stage_b", None)
    sc = _get(config, "stage_c", None)

    # stage_b
    for k, v in (profile.get("stage_b") or {}).items():
        cur = _get(sb, k, None)
        if isinstance(v, dict) and isinstance(cur, dict):
            cur.update(v)
        elif isinstance(v, dict) and cur is not None and not isinstance(cur, dict):
            # if sb has a sub-object, set its attrs
            for kk, vv in v.items():
                _set_attr(cur, kk, vv)
        else:
            _set_attr(sb, k, v)

    # stage_c
    for k, v in (profile.get("stage_c") or {}).items():
        cur = _get(sc, k, None)
        if isinstance(v, dict) and isinstance(cur, dict):
            cur.update(v)
        elif isinstance(v, dict) and cur is not None and not isinstance(cur, dict):
            for kk, vv in v.items():
                _set_attr(cur, kk, vv)
        else:
            _set_attr(sc, k, v)


def _normalize_mode(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    m = str(raw).strip().lower()
    if m == "piano":
        m = "piano_poly"
    return m


def _infer_mode(audio_path: str, stage_a_out: Any) -> Tuple[str, dict]:
    """Infer a high-level mode and return (mode, diagnostics)."""
    dur, sr, at_str = _safe_audio_summary(stage_a_out)
    mix_audio = _get(stage_a_out, "stems.mix.audio", None)

    percussive_ratio = None
    if isinstance(sr, int) and sr > 0 and mix_audio is not None:
        percussive_ratio = _compute_percussive_ratio(mix_audio, sr)

    is_song = _is_song_like(audio_path, stage_a_out, percussive_ratio)

    # Poly hints
    at_str_l = (at_str or "").lower()
    is_poly = any(x in at_str_l for x in ("polyphonic", "poly"))
    dense_poly = ("dominant" in at_str_l) or (is_poly and (percussive_ratio is not None and percussive_ratio < 0.25))

    e2e_ok = _basic_pitch_available()

    # Policy (matches your description)
    if dense_poly and (percussive_ratio is None or percussive_ratio < 0.25):
        mode = "e2e_basic_pitch" if e2e_ok else "piano_poly"
    elif is_song:
        mode = "song"
    elif is_poly:
        mode = "piano_poly"
    else:
        mode = "melody"

    diag = {
        "duration_sec": dur,
        "sr": sr,
        "audio_type": at_str,
        "percussive_ratio": percussive_ratio,
        "dense_poly": dense_poly,
        "song_like": is_song,
        "basic_pitch_available": e2e_ok,
    }
    return mode, diag


# --------------------------
# Public entry point
# --------------------------

def apply_global_profile(
    *,
    audio_path: str,
    stage_a_out: Any,
    config: Any,
    pipeline_logger: Any = None,
) -> str:
    """Mutate config in-place based on global mode. Returns the applied mode string."""
    # Priority: new key -> legacy key -> env -> default
    forced = (
        _normalize_mode(_get(config, "transcription_mode", None))
        or _normalize_mode(_get(config, "global_transcription_mode", None))
        or _normalize_mode(os.getenv("MUSICNOTE_MODE"))
    )
    if forced is None:
        forced = "auto"

    # Resolve auto
    diag = {}
    mode = forced
    if forced == "auto":
        mode, diag = _infer_mode(audio_path, stage_a_out)

    # Apply resolved mode
    sb = _get(config, "stage_b", None)
    if mode in ("classic", "e2e_basic_pitch"):
        # Force only the Stage B top-level switch, do not touch other knobs.
        if sb is not None:
            _set_attr(sb, "transcription_mode", mode)
        applied = mode
    elif mode in ("melody", "piano_poly", "song"):
        _apply_profile(config, mode)
        applied = mode
    else:
        # Unknown -> treat as classic/no-op
        applied = "classic"

    # Also mirror applied mode back into config if possible (for debug exports)
    _set_attr(config, "transcription_mode", applied)

    if pipeline_logger is not None:
        try:
            pipeline_logger.log_event(
                stage="pipeline",
                event="global_profile_applied",
                payload={"forced": forced, "applied": applied, **(diag or {})},
            )
        except Exception:
            pass

    return applied
