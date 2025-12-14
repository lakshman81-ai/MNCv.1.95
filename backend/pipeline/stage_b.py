from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import librosa
import scipy.signal

from .models import (
    MetaData,
    FramePitch,
    StageAOutput,
    Stem,
    StageBOutput,
)
from .detectors import (
    YinDetector,
    CQTDetector,
    SACFDetector,
    SwiftF0Detector,
    RMVPEDetector,
    CREPEDetector,
)
from .config import (
    PIANO_61KEY_CONFIG,
    PipelineConfig,
    StageBConfig,
    InstrumentProfile,
)
from .stage_a import apply_demucs


# ------------------------------------------------------------
# Utility: MIDI / Hz
# ------------------------------------------------------------

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)


# ------------------------------------------------------------
# Polyphonic “Peeling” Helpers (ISS-style)
# ------------------------------------------------------------

def create_harmonic_mask(
    stft: np.ndarray,
    f0_curve: np.ndarray,
    sr: int,
    width: float = 0.03,
) -> np.ndarray:
    """
    Build a time–frequency mask around harmonics of f0_curve.
    width: fractional bandwidth around each harmonic (e.g. 0.03 = ±3%).
    """
    n_freqs, n_frames = stft.shape
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=(n_freqs - 1) * 2)
    mask = np.zeros_like(stft, dtype=np.float32)

    for t in range(min(n_frames, len(f0_curve))):
        f0 = f0_curve[t]
        if f0 <= 0:
            continue

        # Up to 20 harmonics, clipped at Nyquist
        harmonics = np.arange(1, 20) * f0
        harmonics = harmonics[harmonics < sr / 2.0]

        for h_freq in harmonics:
            bw = h_freq * width
            f_low = h_freq - bw / 2.0
            f_high = h_freq + bw / 2.0

            idx_start = np.searchsorted(fft_freqs, f_low)
            idx_end = np.searchsorted(fft_freqs, f_high)
            idx_start = max(0, idx_start)
            idx_end = min(n_freqs, idx_end)
            if idx_start < idx_end:
                mask[idx_start:idx_end, t] = 1.0

    return mask


def iterative_spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    primary_detector,
    validator_detector,
    max_polyphony: int = 4,
    mask_width: float = 0.03,
    audio_path: Optional[str] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    ISS-style polyphonic peeling:

    1. Estimate dominant F0 with primary_detector.
    2. Validate trajectory with SACF (validator_detector).
    3. Build a harmonic mask and subtract it from the spectrogram.
    4. Repeat up to max_polyphony layers.

    Returns: list of (f0_layer, conf_layer).
    """
    extracted: List[Tuple[np.ndarray, np.ndarray]] = []

    y_resid = audio.copy()
    n_fft = 2048
    hop_length = primary_detector.hop_length

    for _ in range(max_polyphony):
        # 1) Proposal
        f0, conf = primary_detector.predict(y_resid, audio_path=audio_path)
        if len(f0) == 0:
            break

        # Termination condition: too weak
        if float(np.mean(conf)) < 0.15 and float(np.max(conf)) < 0.25:
            break

        # 2) Validation
        validation_score = validator_detector.validate_curve(
            f0, y_resid, threshold=0.2
        )
        if validation_score < 0.2 and len(extracted) > 0:
            # Only stop early if we already have at least one good layer
            break

        extracted.append((f0, conf))

        # 3) Subtraction
        S = librosa.stft(y_resid, n_fft=n_fft, hop_length=hop_length)
        mask = create_harmonic_mask(S, f0, sr, width=mask_width)
        S_clean = S * (1.0 - mask)
        y_resid = librosa.istft(
            S_clean, hop_length=hop_length, length=len(y_resid)
        )

    return extracted


# ------------------------------------------------------------
# Stage B: Feature Extraction
# ------------------------------------------------------------

def extract_features(
    stage_a_output: StageAOutput,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
) -> StageBOutput:
    """
    Stage B: Detectors + Ensemble + Polyphonic Peeling (ISS).

    WI Alignment:
    - 2.2.1: Source separation (HTDemucs) to get piano-focused stems.
    - 2.2.2: Instrument profiles (SwiftF0/RMVPE/CREPE/YIN/SACF/CQT).
    - 2.2.3: Ensemble logic with SwiftF0 priority and disagreement threshold.
    - Piano: ISS-based polyphonic peeling for up to N layers.
    - Output: StageBOutput with time_grid, f0_main, f0_layers, per_detector, stem_timelines.
    """
    sb_config: StageBConfig = config.stage_b
    meta: MetaData = stage_a_output.meta

    # --------------------------------------------------------
    # 1. Source Separation (Demucs) – from Stage A "mix" stem
    # --------------------------------------------------------
    stems_in = stage_a_output.stems
    mix_stem = stems_in.get("mix", None)

    final_stems: Dict[str, Stem] = {}

    if sb_config.separation.get("enabled", True) and mix_stem is not None:
        # Run Demucs on the conditioned mix, using config knobs
        demucs_stems = apply_demucs(
            mix_stem.audio,
            mix_stem.sr,
            model_name=sb_config.separation.get("model", "htdemucs"),
            separation_conf=sb_config.separation,
        )

        # Map Demucs outputs to Stem objects
        for name, audio in demucs_stems.items():
            final_stems[name] = Stem(audio=audio, sr=mix_stem.sr, type=name)
    else:
        # No separation; pass through Stage A stems (usually just 'mix')
        final_stems = dict(stems_in)

    # --------------------------------------------------------
    # 2. Global Time Grid
    # --------------------------------------------------------
    global_sr = meta.sample_rate
    global_hop = meta.hop_length

    # Guard: if hop_length accidentally zero, fall back to 512
    if global_hop <= 0:
        global_hop = 512

    duration_sec = float(meta.duration_sec)

    if duration_sec <= 0.0:
        # Fallback if meta.duration_sec is missing
        if mix_stem is not None:
            duration_sec = len(mix_stem.audio) / float(mix_stem.sr)
        else:
            duration_sec = 0.0

    if duration_sec <= 0.0:
        # No audio, empty output
        return StageBOutput(
            time_grid=np.zeros(0),
            f0_main=np.zeros(0),
            f0_layers=[],
            per_detector={},
            stem_timelines={},
            meta=meta,
        )

    n_frames_global = int(duration_sec * global_sr / global_hop) + 1
    time_grid = librosa.frames_to_time(
        np.arange(n_frames_global),
        sr=global_sr,
        hop_length=global_hop,
    )

    # --------------------------------------------------------
    # 3. Stem → Instrument Profile Mapping
    # --------------------------------------------------------
    stem_profile_map: Dict[str, str] = {
        "vocals": "vocals",
        "bass": "bass_guitar",
        "other": "piano_61key",  # typical HTDemucs mapping
        "piano": "piano_61key",  # if 6-stem model used
        "mix": "piano_61key",    # fallback when no sep
    }

    def get_profile_for_stem(stem_name: str) -> InstrumentProfile:
        profile_name = stem_profile_map.get(stem_name, "piano_61key")
        p = config.get_profile(profile_name)
        if p is None:
            p = config.get_profile("piano_61key")
        return p

    # --------------------------------------------------------
    # 4. Process Each Stem: Detectors + Ensemble + ISS
    # --------------------------------------------------------
    per_detector_results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    stem_timelines: Dict[str, List[FramePitch]] = {}

    # Global aggregates
    f0_main_global = np.zeros(n_frames_global, dtype=float)
    f0_layers_global: List[np.ndarray] = []

    for stem_name, stem in final_stems.items():
        if stem_name == "drums":
            # Ignore percussive stem for pitch
            continue

        profile = get_profile_for_stem(stem_name)

        # If this is a pure percussive profile, skip
        if profile.recommended_algo == "none":
            continue

        # Ensure audio at global_sr
        audio = stem.audio
        if stem.sr != global_sr:
            audio = librosa.resample(audio, orig_sr=stem.sr, target_sr=global_sr)

        # ----------------------------------------------------
        # 4.0 Precompute per-frame RMS for this stem
        # ----------------------------------------------------
        # This feeds into Stage C via FramePitch.rms → NoteEvent.rms_value
        frame_rms = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=global_hop,
        )[0]  # shape: (n_frames_rms,)

        if len(frame_rms) != n_frames_global:
            if len(frame_rms) > n_frames_global:
                frame_rms = frame_rms[:n_frames_global]
            else:
                pad = n_frames_global - len(frame_rms)
                frame_rms = np.pad(frame_rms, (0, pad))

        # ----------------------------------------------------
        # 4.1 Instantiate detectors according to config/profile
        # ----------------------------------------------------
        detectors: List[Tuple[str, Any]] = []

        def add_det(cls, name: str, **kwargs):
            try:
                d = cls(
                    sr=global_sr,
                    hop_length=global_hop,
                    **kwargs,
                )
                detectors.append((name, d))
            except Exception as e:
                print(f"[Stage B] Failed to init {name} for stem '{stem_name}': {e}")

        dconf = sb_config.detectors

        # SwiftF0
        if dconf.get("swiftf0", {}).get("enabled", False):
            add_det(
                SwiftF0Detector,
                "swiftf0",
                fmin=profile.fmin,
                fmax=profile.fmax,
            )

        # YIN
        if dconf.get("yin", {}).get("enabled", False):
            add_det(
                YinDetector,
                "yin",
                fmin=profile.fmin,
                fmax=profile.fmax,
            )

        # SACF
        if dconf.get("sacf", {}).get("enabled", False):
            add_det(
                SACFDetector,
                "sacf",
                fmin=profile.fmin,
                fmax=profile.fmax,
            )

        # CQT
        if dconf.get("cqt", {}).get("enabled", False):
            add_det(
                CQTDetector,
                "cqt",
                fmin=profile.fmin,
                fmax=profile.fmax,
            )

        # RMVPE (if recommended or globally enabled)
        if profile.recommended_algo == "rmvpe" or dconf.get("rmvpe", {}).get("enabled", False):
            rm_conf = dconf.get("rmvpe", {})
            sil_thresh = profile.special.get(
                "silence_threshold", rm_conf.get("silence_threshold", 0.04)
            )
            fmin_rm = rm_conf.get("fmin", profile.fmin)
            fmax_rm = rm_conf.get("fmax", profile.fmax)
            add_det(
                RMVPEDetector,
                "rmvpe",
                silence_threshold=sil_thresh,
                fmin=fmin_rm,
                fmax=fmax_rm,
            )

        # CREPE (if recommended or globally enabled)
        if profile.recommended_algo == "crepe" or dconf.get("crepe", {}).get("enabled", False):
            cr_conf = dconf.get("crepe", {})
            viterbi = profile.special.get(
                "viterbi", cr_conf.get("use_viterbi", False)
            )
            model_capacity = cr_conf.get("model_capacity", "full")
            fmin_cr = cr_conf.get("fmin", profile.fmin)
            fmax_cr = cr_conf.get("fmax", profile.fmax)
            add_det(
                CREPEDetector,
                "crepe",
                model_capacity=model_capacity,
                use_viterbi=viterbi,
                fmin=fmin_cr,
                fmax=fmax_cr,
            )

        # If no detectors, skip stem
        if not detectors:
            continue

        # --------------------------------------------
        # 4.2 Run All Detectors for This Stem
        # --------------------------------------------
        candidates_per_frame: List[List[Dict[str, float]]] = [
            [] for _ in range(n_frames_global)
        ]
        stem_raw_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        primary_det = None  # For ISS (usually SwiftF0)
        validator_det = None  # For ISS (SACF)

        for d_name, det in detectors:
            if d_name == "swiftf0":
                primary_det = det
            if d_name == "sacf":
                validator_det = det

            f0, conf = det.predict(audio, audio_path=meta.audio_path)

            # Normalize length to global frame count
            if len(f0) != n_frames_global:
                if len(f0) > n_frames_global:
                    f0 = f0[:n_frames_global]
                    conf = conf[:n_frames_global]
                else:
                    pad = n_frames_global - len(f0)
                    f0 = np.pad(f0, (0, pad))
                    conf = np.pad(conf, (0, pad))

            stem_raw_results[d_name] = (f0, conf)

            for i in range(n_frames_global):
                candidates_per_frame[i].append(
                    {"name": d_name, "f0": float(f0[i]), "conf": float(conf[i])}
                )

        per_detector_results[stem_name] = stem_raw_results

        # --------------------------------------------
        # 4.3 Ensemble Logic (Monophonic main track)
        # --------------------------------------------
        weights = sb_config.ensemble_weights
        priority_floor = sb_config.confidence_priority_floor
        disagreement_cents = sb_config.pitch_disagreement_cents
        voicing_thresh = sb_config.confidence_voicing_threshold

        stem_f0 = np.zeros(n_frames_global, dtype=float)
        stem_conf = np.zeros(n_frames_global, dtype=float)

        for i in range(n_frames_global):
            cands = candidates_per_frame[i]
            if not cands:
                continue

            # SwiftF0 candidate (for priority rule)
            swift = next((c for c in cands if c["name"] == "swiftf0"), None)

            # Check cross-detector disagreement
            valid_pitches = [
                c["f0"] for c in cands if c["f0"] > 10.0 and c["conf"] > 0.1
            ]
            disagreement = False
            if len(valid_pitches) >= 2:
                min_p = min(valid_pitches)
                max_p = max(valid_pitches)
                if min_p > 0:
                    diff_cents = 1200.0 * np.log2(max_p / min_p)
                    if diff_cents > disagreement_cents:
                        disagreement = True

            # WI: If detectors strongly disagree, treat frame as unvoiced
            if disagreement:
                stem_f0[i] = 0.0
                stem_conf[i] = 0.0
                continue

            final_f0 = 0.0
            final_conf = 0.0

            # Priority override (SwiftF0 dominates if confident)
            if swift and swift["conf"] >= priority_floor:
                final_f0 = swift["f0"]
                final_conf = swift["conf"]
            else:
                # Weighted average across detectors
                num = 0.0
                den = 0.0
                max_c = 0.0
                for c in cands:
                    w = weights.get(c["name"], 0.0)
                    if c["f0"] > 10.0:
                        num += c["f0"] * c["conf"] * w
                        den += c["conf"] * w
                        if c["conf"] > max_c:
                            max_c = c["conf"]

                if den > 0.0:
                    final_f0 = num / den
                    final_conf = max_c

            # Voicing threshold
            if final_conf < voicing_thresh:
                final_f0 = 0.0

            stem_f0[i] = final_f0
            stem_conf[i] = final_conf

        # --------------------------------------------
        # 4.4 Ensemble Smoothing (per instrument)
        # --------------------------------------------
        kernel = int(profile.special.get("ensemble_smoothing_frames", 3))
        if kernel > 1 and kernel % 2 == 1:
            # median filter kernel must be odd
            stem_f0 = scipy.signal.medfilt(stem_f0, kernel_size=kernel)

        # --------------------------------------------
        # 4.5 Polyphonic Peeling for Piano (ISS)
        # --------------------------------------------
        stem_layers: List[np.ndarray] = []
        if (
            profile.instrument == "piano_61key"
            and primary_det is not None
            and validator_det is not None
        ):
            if hasattr(primary_det, "reset_state"):
                primary_det.reset_state()

            iss_audio = audio
            extracted = iterative_spectral_subtraction(
                iss_audio,
                global_sr,
                primary_detector=primary_det,
                validator_detector=validator_det,
                max_polyphony=sb_config.polyphonic_peeling.get("max_layers", 8),
                mask_width=sb_config.polyphonic_peeling.get("mask_width", 0.03),
                audio_path=meta.audio_path,
            )

            if extracted:
                # Use first layer as main (dominant) for this stem
                main_f0, main_conf = extracted[0]
                if len(main_f0) != n_frames_global:
                    pad = n_frames_global - len(main_f0)
                    main_f0 = np.pad(main_f0, (0, max(0, pad)))[:n_frames_global]
                    main_conf = np.pad(main_conf, (0, max(0, pad)))[:n_frames_global]

                stem_f0 = main_f0
                stem_conf = main_conf

                # Remaining layers as polyphony
                for (lf0, _lconf) in extracted[1:]:
                    if len(lf0) != n_frames_global:
                        pad2 = n_frames_global - len(lf0)
                        lf0 = np.pad(lf0, (0, max(0, pad2)))[:n_frames_global]
                    stem_layers.append(lf0)

        # --------------------------------------------
        # 4.6 Build FramePitch timeline for this stem
        # --------------------------------------------
        timeline: List[FramePitch] = []
        for i in range(n_frames_global):
            active: List[Tuple[float, float]] = []

            if stem_f0[i] > 0.0:
                active.append((float(stem_f0[i]), float(stem_conf[i])))

            for layer_f0 in stem_layers:
                if layer_f0[i] > 0.0:
                    # ISS layers currently lack per-layer confidence;
                    # we assign a nominal 0.8 for downstream processing.
                    active.append((float(layer_f0[i]), 0.8))

            active.sort(key=lambda x: x[1], reverse=True)

            if active:
                main_pitch_hz = active[0][0]
                main_conf = active[0][1]
                main_midi = int(round(hz_to_midi(main_pitch_hz)))
            else:
                main_pitch_hz = 0.0
                main_conf = 0.0
                main_midi = None

            rms_val = float(frame_rms[i]) if i < len(frame_rms) else 0.0

            timeline.append(
                FramePitch(
                    time=float(time_grid[i]),
                    pitch_hz=main_pitch_hz,
                    midi=main_midi,
                    confidence=main_conf,
                    rms=rms_val,
                    active_pitches=active,
                )
            )

        stem_timelines[stem_name] = timeline

        # --------------------------------------------
        # 4.7 Aggregate to Global Tracks
        # --------------------------------------------
        if len(stem_timelines) == 1:
            # First processed stem defines initial global main/layers
            f0_main_global = stem_f0
            f0_layers_global = list(stem_layers)
        else:
            # If we have vocals + piano, treat vocals as main melody
            if stem_name == "vocals":
                f0_main_global = stem_f0
            elif profile.instrument == "piano_61key":
                f0_layers_global.append(stem_f0)
                f0_layers_global.extend(stem_layers)
            else:
                # For now, non-piano, non-vocal can be added as layers
                f0_layers_global.append(stem_f0)

    # --------------------------------------------------------
    # 5. Build StageBOutput
    # --------------------------------------------------------
    return StageBOutput(
        time_grid=time_grid,
        f0_main=f0_main_global,
        f0_layers=f0_layers_global,
        per_detector=per_detector_results,
        stem_timelines=stem_timelines,
        meta=meta,
    )
