"""
Stage B — Feature Extraction

This module implements pitch detection and feature extraction.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import warnings
import importlib.util

from .models import StageAOutput, FramePitch, AnalysisData, AudioType, StageBOutput, Stem
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

SCIPY_SIGNAL = None
if importlib.util.find_spec("scipy.signal"):
    import scipy.signal as SCIPY_SIGNAL


def _module_available(module_name: str) -> bool:
    """Helper to avoid importing heavy optional deps when missing."""
    return importlib.util.find_spec(module_name) is not None


def _butter_filter(audio: np.ndarray, sr: int, cutoff: float, btype: str) -> np.ndarray:
    """Lightweight wrapper for simple Butterworth filtering."""
    if SCIPY_SIGNAL is None or len(audio) == 0:
        return audio.copy()

    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    norm_cutoff = min(max(norm_cutoff, 1e-4), 0.999)
    sos = SCIPY_SIGNAL.butter(4, norm_cutoff, btype=btype, output="sos")
    return SCIPY_SIGNAL.sosfiltfilt(sos, audio)


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
            # cosine similarity
            score = float(np.dot(spec, tmpl))
            scores[name] = score
        return scores

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        if len(audio) == 0:
            return {}

        scores = self._score_mix(audio)
        vocal_score = scores.get("fm_voice", 0.25) + scores.get("sine_stack", 0.25)
        bass_score = scores.get("square", 0.25)
        saw_score = scores.get("saw", 0.25)
        drum_score = scores.get("broadband", 0.25)

        raw_weights = np.array([vocal_score, bass_score, drum_score, saw_score])
        weights = raw_weights / (np.sum(raw_weights) + 1e-9)
        vocals_w, bass_w, drums_w, other_w = weights

        vocals = vocals_w * _butter_filter(audio, sr, 12000.0, "low")
        vocals = _butter_filter(vocals, sr, 120.0, "high")

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


def _run_htdemucs(audio: np.ndarray, sr: int, model_name: str, overlap: float, shifts: int) -> Optional[Dict[str, Any]]:
    if (
        not _module_available("demucs.pretrained")
        or not _module_available("demucs.apply")
        or not _module_available("torch")
    ):
        warnings.warn("Demucs not available; skipping neural separation.")
        return None

    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import torch

    model = get_model(model_name)
    model_sr = getattr(model, "samplerate", sr)

    if model_sr != sr:
        # simple linear resample
        ratio = float(model_sr) / float(sr)
        indices = np.arange(0, len(audio) * ratio) / ratio
        resampled = np.interp(indices, np.arange(len(audio)), audio)
    else:
        resampled = audio

    mix_tensor = torch.tensor(resampled, dtype=torch.float32)[None, None, :]
    with torch.no_grad():
        demucs_out = apply_model(model, mix_tensor, overlap=overlap, shifts=shifts)

    sources = getattr(model, "sources", ["vocals", "drums", "bass", "other"])
    separated = {}
    for idx, name in enumerate(sources):
        stem_audio = demucs_out[0, idx].mean(dim=0).cpu().numpy()
        separated[name] = stem_audio

    # Ensure canonical stems exist
    for name in ["vocals", "drums", "bass", "other"]:
        separated.setdefault(name, np.zeros_like(audio))

    return separated


def _resolve_separation(stage_a_out: StageAOutput, b_conf) -> Dict[str, Any]:
    if not b_conf.separation.get("enabled", True):
        return stage_a_out.stems

    if len(stage_a_out.stems) > 1 and any(k != "mix" for k in stage_a_out.stems.keys()):
        return stage_a_out.stems

    mix_stem = stage_a_out.stems.get("mix")
    if mix_stem is None:
        return stage_a_out.stems

    sep_conf = b_conf.separation
    synthetic_requested = sep_conf.get("synthetic_model", False)
    overlap = sep_conf.get("overlap", 0.25)
    shifts = sep_conf.get("shifts", 1)

    if synthetic_requested:
        synthetic = SyntheticMDXSeparator(sample_rate=mix_stem.sr, hop_length=stage_a_out.meta.hop_length)
        try:
            synthetic_stems = synthetic.separate(mix_stem.audio, mix_stem.sr)
            if synthetic_stems:
                return {
                    name: type(mix_stem)(audio=audio, sr=mix_stem.sr, type=name)
                    for name, audio in synthetic_stems.items()
                } | {"mix": mix_stem}
        except Exception as exc:
            warnings.warn(f"Synthetic separator failed; falling back to {sep_conf.get('model', 'htdemucs')}: {exc}")

    separated = _run_htdemucs(
        mix_stem.audio,
        mix_stem.sr,
        sep_conf.get("model", "htdemucs"),
        overlap,
        shifts,
    )

    if separated:
        return {
            name: type(mix_stem)(audio=audio, sr=mix_stem.sr, type=name)
            for name, audio in separated.items()
        } | {"mix": mix_stem}

    return stage_a_out.stems

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

        f, t, Z = scipy.signal.stft(
            audio,
            fs=stem.sr,
            nperseg=n_fft,
            noverlap=max(0, n_fft - hop),
            boundary=None,
            padded=False,
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
            nperseg=n_fft,
            noverlap=max(0, n_fft - hop),
            input_onesided=True,
            boundary=None,
        )
        _, residual_audio = scipy.signal.istft(
            Z_resid,
            fs=stem.sr,
            nperseg=n_fft,
            noverlap=max(0, n_fft - hop),
            input_onesided=True,
            boundary=None,
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


def extract_features(
    stage_a_out: StageAOutput,
    config: Optional[PipelineConfig] = None,
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

    # Separation routing happens before harmonic masking/ISS so downstream
    # detectors always see the requested stem layout.
    resolved_stems = _resolve_separation(stage_a_out, b_conf)

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

    stem_timelines: Dict[str, List[FramePitch]] = {}
    per_detector: Dict[str, Any] = {}
    f0_main: Optional[np.ndarray] = None
    all_layers: List[np.ndarray] = []

    # Polyphonic context detection
    polyphonic_context = _is_polyphonic(getattr(stage_a_out, "audio_type", None))
    skyline_mode = _resolve_polyphony_filter(config)

    # Optional harmonic masking to create synthetic melody/bass stems for synthetic material
    augmented_stems = dict(stage_a_out.stems)
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

    stems_for_processing = augmented_stems
    polyphonic_context = polyphonic_context or len(stems_for_processing) > 1 or b_conf.polyphonic_peeling.get("force_on_mix", False)

    # 2. Process Stems
    for stem_name, stem in resolved_stems.items():
        audio = stem.audio
        per_detector[stem_name] = {}

        stem_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Run all initialized detectors on this stem
        for name, det in detectors.items():
            try:
                f0, conf = det.predict(audio, audio_path=stage_a_out.meta.audio_path)
                stem_results[name] = (f0, conf)
                per_detector[stem_name][name] = (f0, conf)
            except Exception as e:
                warnings.warn(f"Detector {name} failed on stem {stem_name}: {e}")

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
                        audio_path=stage_a_out.meta.audio_path,
                    )
                    all_layers.extend([f0 for f0, _ in iss_layers])
                except Exception as e:
                    warnings.warn(f"ISS peeling failed for stem {stem_name}: {e}")

        # Calculate RMS
        n_fft = stage_a_out.meta.window_size if stage_a_out.meta.window_size else 2048
        frames = _frame_audio(audio, n_fft, hop_length)
        rms_vals = np.sqrt(np.mean(frames**2, axis=1))

        # Pad/Trim RMS to match dominant F0
        if len(rms_vals) < len(merged_f0):
            rms_vals = np.pad(rms_vals, (0, len(merged_f0) - len(rms_vals)))
        elif len(rms_vals) > len(merged_f0):
            rms_vals = rms_vals[:len(merged_f0)]

        # Build timeline with optional skyline selection from poly layers
        voicing_thr = float(b_conf.confidence_voicing_threshold)
        layer_arrays = [(merged_f0, merged_conf)] + iss_layers
        max_frames = max(len(arr[0]) for arr in layer_arrays)

        def _pad_to(arr: np.ndarray, target: int) -> np.ndarray:
            if len(arr) < target:
                return np.pad(arr, (0, target - len(arr)))
            return arr[:target]

        padded_layers = [(_pad_to(f0, max_frames), _pad_to(conf, max_frames)) for f0, conf in layer_arrays]
        padded_rms = _pad_to(rms_vals, max_frames)

        timeline: List[FramePitch] = []
        main_track = np.zeros(max_frames, dtype=np.float32)

        for i in range(max_frames):
            candidates: List[Tuple[float, float]] = []
            for f0_arr, conf_arr in padded_layers:
                f = float(f0_arr[i]) if i < len(f0_arr) else 0.0
                c = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                if f > 0.0 and c >= voicing_thr:
                    candidates.append((f, c))

            chosen_pitch = 0.0
            chosen_conf = 0.0
            active = sorted(candidates, key=lambda x: x[0], reverse=True)

            if active:
                if skyline_mode == "skyline_top_voice":
                    chosen_pitch, chosen_conf = active[0]
                else:
                    # default to highest confidence
                    chosen_pitch, chosen_conf = max(active, key=lambda x: x[1])

            main_track[i] = chosen_pitch

            midi = None
            if chosen_pitch > 0.0:
                midi = int(round(69.0 + 12.0 * np.log2(chosen_pitch / 440.0)))

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

        # Set main f0 (prefer vocals, then mix)
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

    return StageBOutput(
        time_grid=time_grid,
        f0_main=f0_main,
        f0_layers=all_layers,
        per_detector=per_detector,
        stem_timelines=stem_timelines,
        meta=stage_a_out.meta
    )
