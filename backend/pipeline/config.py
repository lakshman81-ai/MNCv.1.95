from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


# ------------------------------------------------------------
# Stage A Config (Signal Conditioning)
# ------------------------------------------------------------

@dataclass
class StageAConfig:
    target_sample_rate: int = 44100
    channel_handling: str = "mono_sum"  # "mono_sum", "left_only", "right_only"

    # Preprocessing flags/dicts (updated for 61-key flexibility)
    dc_offset_removal: Union[bool, Dict[str, Any]] = True

    # Transient emphasis (hammer clicks)
    transient_pre_emphasis: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "alpha": 0.97}
    )

    # High-pass filter (can be used as dict in PIANO_61KEY_CONFIG or legacy separate fields)
    high_pass_filter: Dict[str, Any] = field(default_factory=dict)

    # Legacy fields kept for backward compatibility (defaults align with legacy behavior)
    high_pass_filter_cutoff: Dict[str, Any] = field(
        default_factory=lambda: {"value": 55.0}
    )
    high_pass_filter_order: Dict[str, Any] = field(
        default_factory=lambda: {"value": 4}
    )

    # Silence trimming (keep decay tails)
    silence_trimming: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "top_db": 50}
    )

    # Loudness normalization (EBU R128)
    loudness_normalization: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "target_lufs": -23.0}
    )

    # Peak limiter (Soft clip or -1 dB ceiling)
    peak_limiter: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,          # set True to use it
            "mode": "soft",            # "soft" | "hard"
            "ceiling_db": -1.0,        # max peak level
        }
    )

    # Noise floor estimation (percentile of RMS)
    noise_floor_estimation: Dict[str, Any] = field(
        default_factory=lambda: {"percentile": 30}
    )

    # BPM / beat grid detection
    bpm_detection: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "min_bpm": 55.0,
            "max_bpm": 215.0,
        }
    )


# ------------------------------------------------------------
# Stage B Config (Detectors + Ensemble + ISS)
# ------------------------------------------------------------

@dataclass
class StageBConfig:
    # Instrument selection (for Stage B tuning)
    instrument: str = "piano_61key"

    # Transcription mode: "classic" (default) | "e2e_basic_pitch" | "auto"
    transcription_mode: str = "classic"

    # Basic Pitch parameters
    bp_onset_threshold: float = 0.5
    bp_frame_threshold: float = 0.3
    bp_minimum_note_length_ms: float = 127.7
    bp_min_hz: float = 27.5      # A0
    bp_max_hz: float = 4186.0    # C8
    bp_melodia_trick: bool = True

    # Active stems whitelist (None = all)
    active_stems: Optional[List[str]] = None  # e.g. ["bass", "vocals"]
    # Flag to enable instrument-specific profile overrides in Stage B
    apply_instrument_profile: bool = True

    # Source separation (HTDemucs)
    separation: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "model": "htdemucs",
            # If True, prefer the synthetic model fine-tuned on procedurally
            # generated L2 sine/saw/square/FM stems. Falls back to the
            # configured "model" when unavailable.
            "synthetic_model": False,
            "overlap": 0.25,  # Demucs overlap
            "shifts": 1,      # number of shifts (test-time augmentation)
            # Polyphonic-dominant preset: heavier overlap + more TTA to peel melody
            "polyphonic_dominant_preset": {
                "overlap": 0.75,
                "shift_range": [2, 5],
                "overlap_candidates": [0.5, 0.75],
            },
            # Optional harmonic masking guided by a fast F0 prior
            "harmonic_masking": {
                "enabled": True,
                "mask_width": 0.02,
                "n_harmonics": 12,
            },
        }
    )

    # Global voicing threshold for ensemble F0
    confidence_voicing_threshold: float = 0.58

    # Additional relaxation applied when polyphonic context is detected
    polyphonic_voicing_relaxation: float = 0.07

    # SwiftF0 priority floor
    confidence_priority_floor: float = 0.5

    # Cross-detector disagreement tolerance (cents)
    pitch_disagreement_cents: float = 70.0  # WI: 70 cents

    # Ensemble weights (WI-aligned core)
    #   Piano: SwiftF0 dominates, SACF/CQT support.
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "swiftf0": 0.5,
            "sacf": 0.3,
            "cqt": 0.2,
            "yin": 0.3,
            "rmvpe": 0.5,
            "crepe": 1.0,   # Boost Crepe as it is now installed and generally superior
        }
    )

    # Ensemble fusion mode: "static" (weighted sum) | "adaptive" (reliability-gated weighted median)
    ensemble_mode: str = "static"

    # Smoothing method: "tracker" (Hungarian assignment) | "viterbi" (HMM-based)
    smoothing_method: str = "tracker"

    # Viterbi smoothing parameters
    viterbi_transition_smoothness: float = 0.5
    viterbi_jump_penalty: float = 0.6

    # Polyphonic peeling (ISS) settings
    polyphonic_peeling: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_layers": 8,
            "mask_width": 0.03,  # Fractional bandwidth around harmonics
            "min_mask_width": 0.02,
            "max_mask_width": 0.08,
            "mask_growth": 1.1,
            "mask_shrink": 0.9,
            "harmonic_snr_stop_db": 3.0,
            "residual_rms_stop_ratio": 0.08,
            "residual_flatness_stop": 0.45,
            "validator_cents_tolerance": 50.0,
            "validator_agree_window": 5,
            "validator_disagree_decay": 0.6,
            "validator_min_agree_frames": 2,
            "validator_min_disagree_frames": 2,
            "max_harmonics": 12,
            "force_on_mix": True,
            # Adaptive ISS (WI Feature E) - ENABLED by default for better L4/L5
            "iss_adaptive": True,
            "strength_min": 0.8,
            "strength_max": 1.2,
            "flatness_thresholds": [0.3, 0.6],  # [low, high]
            "use_freq_aware_masks": True,
        }
    )

    # Voice tracking / skyline settings
    voice_tracking: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_alt_voices": 4,
            "max_jump_cents": 150.0,
            "hangover_frames": 4,
            "smoothing": 0.35,
            "confidence_bias": 5.0,
        }
    )

    # Global detector enable flags + defaults
    detectors: Dict[str, Any] = field(
        default_factory=lambda: {
            "rmvpe": {
                "enabled": False,
                "fmin": 50.0,
                "fmax": 1200.0,
                "hop_length": 160,
                "silence_threshold": 0.04,
            },
            "crepe": {
                "enabled": False,
                "model_capacity": "small",  # "tiny"|"small"|"medium"|"large"|"full"
                "fmin": 80.0,
                "fmax": 1000.0,
                "use_viterbi": False,
                "step_ms": 10,
                "conf_threshold": 0.5,
            },
            "swiftf0": {"enabled": True},
            "yin": {
                "enabled": True,
                "fmin": 80.0,
                "fmax": 1000.0,
                "hop_length": 256,
                "frame_length": 4096,
                "threshold": 0.08,
                # Multi-resolution + Octave Correction (WI Feature C) - ENABLED for L5 accuracy
                "enable_multires_f0": True,
                "enable_octave_correction": False,
                "octave_jump_penalty": 0.35,
            },
            "sacf": {"enabled": True},
            "cqt": {
                "enabled": True,
                # Morphological filtering (WI Feature D)
                "enable_salience_morphology": False,
                "morph_kernel": 3,
            },
        }
    )

    # Melody post-filtering (applied after detector merge)
    melody_filtering: Dict[str, Any] = field(
        default_factory=lambda: {
            "median_window": 5,
            "voiced_prob_threshold": 0.4,
            "rms_gate_db": -40.0,
            "fmin_hz": 80.0,
            "fmax_hz": 1000.0,
        }
    )

    # Onsets & Frames (WI Feature B)
    onsets_and_frames: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "onset_threshold": 0.5,
            "frame_threshold": 0.5,
            "min_note_duration_ms": 30,
        }
    )


# ------------------------------------------------------------
# Stage C Config (Note Segmentation)
# ------------------------------------------------------------

@dataclass
class StageCConfig:
    # Flag to enable instrument-specific profile overrides in Stage C
    apply_instrument_profile: bool = True

    # Segmentation method selection + HMM defaults
    segmentation_method: Dict[str, Any] = field(
        default_factory=lambda: {
            "method": "hmm",  # "hmm" | "viterbi" | "threshold" | "rms_gate"
            "states": ["attack", "sustain", "silence"],
            # Optional state smoothing; defaults to off to preserve historical behaviour
            "use_state_smoothing": False,
            "transition_penalty": 0.8,
            "stay_bonus": 0.05,
            "silence_bias": 0.1,
            "energy_weight": 0.35,
            # Validation defaults (Patch 1B)
            "min_onset_frames": 2,
            "release_frames": 2,
            "time_merge_frames": 1,
            "split_semitone": 0.7,
        }
    )

    # Minimum note duration in milliseconds
    # WI: 30 ms for piano/guitar; captures fast grace notes/trills.
    min_note_duration_ms: float = 30.0

    # Polyphonic-specific minimum duration - REDUCED for fast ornaments in L5
    min_note_duration_ms_poly: float = 45.0

    # HMM frame stability (used in HMMProcessor)
    frame_stability: Dict[str, Any] = field(
        default_factory=lambda: {"stable_frames_required": 2}
    )

    # Pitch tolerance for merging (cents)
    pitch_tolerance_cents: float = 50.0

    # Allow bridging micro-gaps within sustained notes (seconds)
    gap_tolerance_s: float = 0.05

    # Base confidence threshold and hysteresis for note activation
    confidence_threshold: float = 0.20
    confidence_hysteresis: Dict[str, float] = field(
        default_factory=lambda: {"start": 0.6, "end": 0.4}
    )

    # Gap filling (legato) in ms
    gap_filling: Dict[str, Any] = field(
        default_factory=lambda: {"max_gap_ms": 100.0}
    )

    # --- CONSOLIDATED FIX (L6 runner + stage_c.py dotted-path knobs) ---
    # stage_c.py reads:
    #   stage_c.chord_onset_snap_ms (default 25ms)
    #   stage_c.post_merge.max_gap_ms (preferred) else stage_c.gap_filling.max_gap_ms
    #
    # To preserve existing behavior, default post_merge to None
    # so merge_gap_ms continues to come from gap_filling.max_gap_ms unless overridden.
    chord_onset_snap_ms: float = 25.0
    post_merge: Optional[Dict[str, Any]] = None
    # ---------------------------------------------------------------

    # Confidence gates for polyphonic timelines (melody vs accompaniment)
    # Relaxed accompaniment threshold to 0.40 to capture quiet inner voices
    polyphonic_confidence: Dict[str, float] = field(
        default_factory=lambda: {"melody": 0.55, "accompaniment": 0.40}
    )

    # RMS → MIDI velocity mapping
    velocity_map: Dict[str, float] = field(
        default_factory=lambda: {
            "min_db": -40.0,
            "max_db": -4.0,
            "min_vel": 20.0,
            "max_vel": 105.0,
            "noise_floor_db_margin": 6.0,  # (Patch 5B)
        }
    )

    # Feature Flags for robustness upgrades
    use_onset_refinement: bool = True
    use_repeated_note_splitter: bool = True

    # Polyphony filter mode ("skyline_top_voice" used as a hint to Stage D)
    polyphony_filter: Dict[str, str] = field(
        default_factory=lambda: {"mode": "skyline_top_voice"}
    )


# ------------------------------------------------------------
# Stage D Config (Rendering / MusicXML)
# ------------------------------------------------------------

@dataclass
class StageDConfig:
    # Forced key signature (overrides detection)
    forced_key: Optional[str] = None

    # MusicXML divisions per quarter note (24 = 1/24 quarter)
    divisions_per_quarter: int = 24

    # Quantization grid (e.g. 16 for 1/16th)
    quantization_grid: int = 16

    # Staff split point (MIDI pitch)
    staff_split_point: Dict[str, Any] = field(
        default_factory=lambda: {"pitch": 60}  # C4
    )

    # Staccato marking threshold (in beats)
    staccato_marking: Dict[str, Any] = field(
        default_factory=lambda: {"threshold_beats": 0.25}
    )

    # Quantization mode: "grid" (strict) | "light_rubato" (snap only if close)
    quantization_mode: str = "grid"
    light_rubato_snap_ms: float = 30.0

    # General glissando detection (disabled for piano by WI)
    glissando_threshold_general: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "min_semitones": 2.0,
            "max_time_ms": 500.0,
        }
    )

    # Piano-specific glissando handling (always discrete)
    glissando_handling_piano: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False}
    )

    # Tempo/Time signature defaults
    tempo_bpm: float = 120.0
    time_signature: str = "4/4"


# ------------------------------------------------------------
# Segmented Transcription Config (Auto-Retry)
# ------------------------------------------------------------

@dataclass
class SegmentedTranscriptionConfig:
    enabled: bool = False
    segment_sec: float = 10.0
    overlap_sec: float = 2.0
    retry_quality_threshold: float = 0.9
    retry_max_candidates: int = 3
    # Density heuristic parameters - INCREASED for complex L5 pieces
    density_target_notes_per_sec: float = 8.0
    density_penalty_span: float = 6.0


# ------------------------------------------------------------
# Instrument Profiles
# ------------------------------------------------------------

@dataclass
class InstrumentProfile:
    instrument: str
    recommended_algo: str
    fmin: float
    fmax: float
    # Arbitrary extra keys; Stage B currently uses:
    #   - "ensemble_smoothing_frames"
    #   - "viterbi"
    #   - "silence_threshold"
    special: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# Pipeline Config
# ------------------------------------------------------------

@dataclass
class PipelineConfig:
    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    seed: Optional[int] = None  # deterministic runs when set
    instrument_override: Optional[str] = None  # e.g. "bass_guitar", "guitar_distorted"
    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b: StageBConfig = field(default_factory=StageBConfig)
    stage_c: StageCConfig = field(default_factory=StageCConfig)
    stage_d: StageDConfig = field(default_factory=StageDConfig)
    segmented_transcription: SegmentedTranscriptionConfig = field(default_factory=SegmentedTranscriptionConfig)
    instrument_profiles: List[InstrumentProfile] = field(default_factory=list)

    def get_profile(self, instrument_name: str) -> Optional[InstrumentProfile]:
        """
        Robust instrument profile lookup with simple aliasing.
        """
        name = instrument_name.lower()

        # Simple aliases; extend as needed
        aliases = {
            "piano": "piano_61key",
            "keys": "piano_61key",
            # Guitar aliases
            "electric_guitar": "electric_guitar_clean",
            "electric-guitar": "electric_guitar_clean",
            "distorted_guitar": "electric_guitar_distorted",
            "guitar_distorted": "electric_guitar_distorted",
            "electric_guitar_distorted": "electric_guitar_distorted",
            "electric_guitar_overdrive": "electric_guitar_distorted",
            "electric_guitar_distortion": "electric_guitar_distorted",

            "drums": "drums_percussive",
            "percussion": "drums_percussive",
        }
        canonical = aliases.get(name, name)

        for p in self.instrument_profiles:
            if p.instrument.lower() == canonical:
                return p
        return None


# ------------------------------------------------------------
# Default Instrument Profiles (WI-based)
# ------------------------------------------------------------

_profiles: List[InstrumentProfile] = [
    # 61-key piano (C2–C7), main target
    InstrumentProfile(
        instrument="piano_61key",
        recommended_algo="swiftf0",
        fmin=60.0,
        fmax=2200.0,
        special={
            # Piano: light ensemble smoothing per WI/master table
            "ensemble_smoothing_frames": 3,
            "viterbi": False,
        },
    ),

    # Vocals (singing) – RMVPE primary
    InstrumentProfile(
        instrument="vocals",
        recommended_algo="rmvpe",
        fmin=50.0,
        fmax=1200.0,
        special={
            "viterbi": True,
            "silence_threshold": 0.04,
        },
    ),

    # Violin – CREPE with Viterbi
    InstrumentProfile(
        instrument="violin",
        recommended_algo="crepe",
        fmin=190.0,
        fmax=3500.0,
        special={
            "viterbi": True,
        },
    ),

    # Bass Guitar – YIN, wide windows (low frequencies)
    InstrumentProfile(
        instrument="bass_guitar",
        recommended_algo="yin",
        fmin=30.0,
        fmax=400.0,
        special={
            "yin_frame_length": 8192,
            # Backwards compat alias (optional but safe)
            "frame_length": 8192,
        },
    ),

    # 5-String Bass - Lower range + backtracking
    InstrumentProfile(
        instrument="bass_5string",
        recommended_algo="yin",
        fmin=30.0,
        fmax=400.0,
        special={
            "yin_frame_length": 8192,
            "stage_c_backtrack_ms": 100.0,
        },
    ),

    # Cello – RMVPE with smoothing
    InstrumentProfile(
        instrument="cello",
        recommended_algo="rmvpe",
        fmin=65.0,
        fmax=880.0,
        special={
            "viterbi": True,
            "silence_threshold": 0.04,
        },
    ),

    # Flute – CREPE, smaller windows (fast attacks)
    InstrumentProfile(
        instrument="flute",
        recommended_algo="crepe",
        fmin=261.0,
        fmax=3349.0,
        special={
            "small_window": True,
        },
    ),

    # Acoustic Guitar – SwiftF0, light smoothing
    InstrumentProfile(
        instrument="acoustic_guitar",
        recommended_algo="swiftf0",
        fmin=82.0,
        fmax=880.0,
        special={
            "ensemble_smoothing_frames": 3,
        },
    ),

    # Electric Guitar (clean) – YIN, low threshold
    InstrumentProfile(
        instrument="electric_guitar_clean",
        recommended_algo="yin",
        fmin=80.0,
        fmax=1200.0,
        special={
            "yin_conf_threshold": 0.05,
            # Backwards compat
            "threshold": 0.05,
        },
    ),

    # Electric Guitar (distorted) – YIN tuned for rough signals + pre-LPF
    InstrumentProfile(
        instrument="electric_guitar_distorted",
        recommended_algo="yin",
        fmin=80.0,
        fmax=1200.0,
        special={
            # Report: LPF ~800–1000 Hz to remove distortion "fizz"
            "pre_lpf_hz": 1000.0,
            # Report: raise YIN trough threshold for rough/distorted waveforms
            "yin_trough_threshold": 0.20,
            # Report: Transient Lockout
            "transient_lockout_ms": 10.0,
            "viterbi": False,
        },
    ),

    # Drums / percussive – no pitch
    InstrumentProfile(
        instrument="drums_percussive",
        recommended_algo="none",
        fmin=0.0,
        fmax=0.0,
        special={
            "ignore_pitch": True,
            "high_conf_threshold": 0.15,
        },
    ),
]


# ------------------------------------------------------------
# Default Pipeline Config instance
# ------------------------------------------------------------

# 61-key preset with all enabled preprocessing and detector defaults
PIANO_61KEY_CONFIG = PipelineConfig(
    stage_a=StageAConfig(
        target_sample_rate=22050,
        silence_trimming={"enabled": True, "top_db": 40.0},
        high_pass_filter={"enabled": True, "cutoff_hz": 50.0, "order": 2},
        dc_offset_removal=True,
        peak_limiter={"enabled": True, "ceiling_db": -1.0, "mode": "soft", "drive": 1.2},
        bpm_detection={"enabled": True, "tightness": 100},
    ),
    stage_b=StageBConfig(
        confidence_voicing_threshold=0.5,
        confidence_priority_floor=0.5,
        pitch_disagreement_cents=50.0,
        ensemble_weights={"swiftf0": 0.6, "sacf": 0.2, "cqt": 0.3, "yin": 0.1, "crepe": 0.2},
        separation={
            "enabled": "auto",
            "model": "htdemucs",
            "synthetic_model": False,
            "overlap": 0.25,
            "shifts": 1,
            "harmonic_masking": {"enabled": True, "mask_width": 0.02, "n_harmonics": 12}
        },
        detectors={
            "swiftf0": {"enabled": True, "fmin": 60.0, "fmax": 2000.0, "confidence_threshold": 0.9},
            "sacf":    {"enabled": True, "fmin": 60.0, "fmax": 2200.0, "window_size": 4096, "threshold": 0.3},
            "cqt":     {"enabled": True, "fmin": 60.0, "fmax": 4000.0, "bins_per_octave": 48, "n_bins": 240, "max_peaks": 8},
            "yin":     {"enabled": True, "fmin": 60.0, "fmax": 2200.0, "frame_length": 4096, "threshold": 0.1, "enable_multires_f0": True, "enable_octave_correction": True, "octave_jump_penalty": 0.6},
            "rmvpe":   {"enabled": False},
            "crepe":   {"enabled": True, "model_capacity": "small", "confidence_threshold": 0.5},
        },
        melody_filtering={
             "median_window": 7,
             "voiced_prob_threshold": 0.45,
             "rms_gate_db": -45.0,
        },
    ),
    stage_c=StageCConfig(
        min_note_duration_ms=35.0,
        min_note_duration_ms_poly=50.0,
        segmentation_method={
            "method": "hmm",
            "min_onset_frames": 2,
            "release_frames": 2,
            "time_merge_frames": 1,
            "split_semitone": 0.7,
        },
        pitch_tolerance_cents=50.0,
        # NOTE: leaving chord_onset_snap_ms/post_merge at defaults here preserves existing behavior.
        # You can override in L6 runner or profiles if desired.
    ),
    stage_d=StageDConfig(
        quantization_grid=16,
        tempo_bpm=120.0,
        time_signature="4/4",
    ),
    instrument_profiles=_profiles
)
