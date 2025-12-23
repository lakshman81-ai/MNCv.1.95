# Pipeline Workflow & Contracts (MNC v2.0)

This document outlines the detailed call graph, data flow, algorithms, and configuration triggers for the music transcription pipeline (Stages A-D).

## Detailed Pipeline Flowchart

This chart explicitly details the Algorithm Selection Logic (Profile $\rightarrow$ Weights $\rightarrow$ Fallback) and expands the Detector Bank to include every algorithm mentioned in Section 2.

```mermaid
flowchart TD
    %% Main Entry Point
    Start(["Start: Audio Input"]) --> StageA["Stage A: Signal Conditioning"]

    %% Stage A Summary
    subgraph StageA_Process [Stage A: Preprocessing]
        SA1["Resample / Mono / Trim"]
        SA2["Loudness Norm EBU R128"]
        SA1 --> SA2
    end
    StageA --> SA1

    %% Decision: Neural E2E vs Standard
    SA2 --> CheckE2E{"E2E Mode?<br>Basic Pitch / Auto"}
    CheckE2E -- Yes --> NeuralTrans["Neural Transcription<br>Basic Pitch / O&F"]
    NeuralTrans --> StageD

    CheckE2E -- No --> LogicStart

    %% --- STAGE B: FEATURE EXTRACTION ---
    subgraph StageB_Process [Stage B: Feature Extraction]
        direction TB

        %% --- SECTION 3: SELECTION LOGIC ---
        subgraph Selection_Logic [Selection Logic]
            direction TB
            LogicStart(Input Context)

            %% Logic 1: Instrument Profile
            SL1{"1. Instrument<br>Profile?"}
            LogicStart --> SL1

            SL1 -- Yes --> SL1_Load["Load Preset Config<br>e.g. Violin=CREPE, Bass=YIN"]
            SL1 -- No --> SL1_Def["Load Default Config"]
        end

        %% --- SECTION 2: NOTE EXTRACTION ALGORITHMS ---
        subgraph Detectors [Detector Bank]
            direction TB
            %% Setup Inputs
            SL1_Load & SL1_Def --> D_Input("Run Detectors Parallel")

            %% The 6 Algorithms from Section 2
            D_Input --> Alg_YIN
            D_Input --> Alg_Swift
            D_Input --> Alg_SACF
            D_Input --> Alg_CREPE
            D_Input --> Alg_RMVPE
            D_Input --> Alg_CQT

            Alg_YIN["**YIN**<br>Time-domain Autocorr<br><i>Best for: Bass/Clean</i>"]
            Alg_Swift["**SwiftF0**<br>Learning-based Est.<br><i>Priority: High</i>"]
            Alg_SACF["**SACF**<br>Simple Autocorr<br><i>Legacy/Fast</i>"]
            Alg_CREPE["**CREPE**<br>Neural Network<br><i>Best for: Violin/Flute</i>"]
            Alg_RMVPE["**RMVPE**<br>Vocal Extraction<br><i>Best for: Vocals</i>"]
            Alg_CQT["**CQT**<br>Constant-Q Transform<br><i>Validation/Spec</i>"]
        end

        %% --- SECTION 3: ENSEMBLE WEIGHTS (Parallel Path 1) ---
        Alg_YIN & Alg_Swift & Alg_SACF & Alg_CREPE & Alg_RMVPE & Alg_CQT --> Ensemble

        subgraph Ensemble_Logic [Ensemble & Smoothing]
            Ensemble["**2. Ensemble Fusion**<br>Merge Outputs"]

            %% Logic 2: Fusion Mode
            Ensemble --> Fusion{"Mode?"}
            Fusion -- Static --> W_Avg["Weighted Average"]
            Fusion -- Adaptive --> W_Med["Reliability-Gated<br>Weighted Median"]

            W_Avg & W_Med --> Smoothing{"Smoothing?"}
            Smoothing -- Tracker --> S_Track["Hungarian Tracker"]
            Smoothing -- Viterbi --> S_Vit["Viterbi Path"]

            S_Track & S_Vit --> ResultB(["Main Stage B Output"])
        end

        %% --- ISS / POLYPHONY (Parallel Path 2) ---
        %% ISS runs independently using a primary detector to peel layers
        LogicStart -.-> CheckPoly{"Polyphonic?"}
        CheckPoly -- Yes --> ISS["**Iterative Spectral Subtraction**<br>Peel multiple layers<br>(Adaptive / Freq-Aware)"]
        ISS --> PolyLayers(["Polyphonic Layers"])
        CheckPoly -- No --> NoPoly["No Peeling"]
    end

    %% Merge Paths
    ResultB & PolyLayers --> StageC["Stage C: Apply Theory"]
    ResultB & NoPoly --> StageC

    subgraph StageC_Process [Stage C]
        SC1["Segmentation<br>(Threshold / HMM / Decomposed)"] --> SC2["Duration/Velocity Filter"]
    end

    subgraph StageD_Process [Stage D: Quantize & Render]
        SC2 --> StageD_Quant["Quantization"]
        StageD_Quant --> StageD_Render["Export XML/MIDI"]
    end
    StageD_Render --> End(["Final Output"])
```

## Flowchart Explainer

### Stage A: Preprocessing
*   **Signal Conditioning**: The input audio is resampled to a consistent rate (44.1kHz or 22.05kHz), mixed to mono, and trimmed of silence.
*   **Normalization**: Loudness is normalized to a target (e.g., -23 LUFS) to ensure consistent detector response.
*   **E2E Bypass**: If `transcription_mode` is set to "e2e_basic_pitch" or "onsets_and_frames", the classic detector pipeline is bypassed, sending neural transcription results directly to Stage C/D.

### Stage B: Selection & Detectors (Sections 2 & 3)
*   **Selection Logic**:
    *   **Profile**: Checks for specific instrument profiles (e.g., "Violin") to prioritize recommended algorithms (e.g., CREPE).
    *   **Fallback**: Automatically falls back to DSP methods (YIN/SACF) if neural dependencies are missing or fail.
*   **Detector Bank**: Up to six algorithms run in parallel:
    *   **YIN/SACF**: Robust DSP methods for bass and clean signals.
    *   **SwiftF0/RMVPE/CREPE**: High-accuracy neural estimators for melody/vocals/instruments.
    *   **CQT**: Spectral analysis validator.

### Stage B: Polyphony & Ensemble
*   **Ensemble Fusion**: Outputs are fused into a main timeline.
    *   **Static**: Traditional weighted averaging.
    *   **Adaptive**: Reliability-gated weighted median (robust against outliers).
*   **Smoothing**:
    *   **Tracker**: Hungarian assignment for voice continuity.
    *   **Viterbi**: HMM-based pathfinding for optimal global pitch contour.
*   **Polyphony (ISS)**: **Iterative Spectral Subtraction** peels accompaniment voices. Now supports **Adaptive** strength and **Frequency-Aware Masks** (wider for bass).
*   **Output**: The Main Voice and Polyphonic Layers are combined and sent to Stage C.

### Stage C: Segmentation
*   **Note Extraction**:
    *   **Skyline**: Selects the best candidate per frame (prioritizing confidence & vocal band).
    *   **Decomposed Melody**: Fully decomposes polyphony and picks the strongest melodic track.
*   **Refinement**:
    *   **Onset Snapping**: Aligns starts to spectral flux peaks.
    *   **Repeated Note Splitter**: Splits long notes on re-articulation (energy bumps).
*   **Filtering**: Notes below minimum duration or velocity thresholds are discarded.

### Stage D: Quantization & Rendering
*   **Quantization**:
    *   **Grid Mode**: Hard snap to nearest grid unit (e.g., 1/16th).
    *   **Light Rubato**: Snaps only notes close to the grid (within ~30ms), preserving expressive timing elsewhere.
*   **Rendering**: Formats notes into MusicXML (Grand Staff with braces) and MIDI (including glissando).

## Tunable Parameters (Tuner/Audit)

The following parameters are exposed for iterative tuning and audit verification.

### Stage A: Conditioning
*   `target_sample_rate`: Working sample rate (default 44100Hz, or 22050Hz for piano).
*   `loudness_normalization.target_lufs`: Target integrated loudness (e.g., -23.0).
*   `high_pass_filter.cutoff_hz`: HPF cutoff (20-60Hz).
*   `bpm_detection.min_bpm` / `max_bpm`: Allowed tempo range.
*   `peak_limiter.mode`: "soft" or "hard" clipping.

### Stage B: Features
*   **Mode**:
    *   `transcription_mode`: "classic", "e2e_basic_pitch", "auto".
    *   `active_stems`: Whitelist of stems to process (e.g., ["vocals"]).
*   **Separation**:
    *   `separation.enabled`: Auto/True/False.
    *   `separation.model`: "htdemucs" or "synthetic" (L2).
*   **Detectors**:
    *   `detectors.<name>.enabled`: Toggle SwiftF0, YIN, CREPE, RMVPE.
    *   `ensemble_weights.<name>`: Contribution of each detector.
*   **Fusion & Smoothing**:
    *   `ensemble_mode`: "static" (weighted average) or "adaptive" (reliability-gated).
    *   `smoothing_method`: "tracker" (Hungarian) or "viterbi" (HMM).
    *   `viterbi_transition_smoothness`: Smoothness cost for Viterbi.
*   **Polyphony**:
    *   `polyphonic_peeling.max_layers`: Max voices to extract (ISS).
    *   `polyphonic_peeling.iss_adaptive`: Enable adaptive strength.
    *   `polyphonic_peeling.use_freq_aware_masks`: Wider masks for bass frequencies.
    *   `polyphonic_peeling.cqt_gate_enabled`: Soft spectral gating for candidates.

### Stage C: Segmentation
*   **Logic**:
    *   `segmentation_method.method`: "hmm", "threshold", "viterbi".
    *   `polyphony_filter.mode`: "skyline_top_voice" (default) or "decomposed_melody" (L5).
    *   `confidence_threshold`: Minimum confidence for note activation.
    *   `min_note_duration_ms`: Minimum duration (Mono).
    *   `min_note_duration_ms_poly`: Minimum duration (Poly).
    *   `polyphonic_confidence`: Thresholds for Melody vs Accompaniment.
*   **Refinement**:
    *   `use_onset_refinement`: Snap start times to flux peaks.
    *   `use_repeated_note_splitter`: Split long notes on energy re-articulation.
    *   `chord_onset_snap_ms`: Tolerance for snapping simultaneous notes.
    *   `gap_filling.max_gap_ms`: Max gap to bridge for legato.

### Stage D: Quantization
*   `quantization_mode`: "grid" or "light_rubato".
*   `light_rubato_snap_ms`: Window for snapping in rubato mode (e.g., 30ms).
*   `quantization_grid`: Grid resolution (16 = 1/16th).
*   `forced_key`: Override detected key signature.
*   `glissando_threshold_general`: Config for glissando detection.

---

## Stage A: Load & Preprocess (`backend/pipeline/stage_a.py`)

**Goal:** Normalize audio into a consistent, analysis-ready format (Mono, Fixed Sample Rate, Normalized Gain) and perform initial global analysis (BPM, Texture).

### Strategies & Algorithms

1.  **Loading & Resampling**:
    *   **Algorithm**: `librosa.load` (with fallback to `scipy.io.wavfile`).
    *   **Why**: Detectors (especially Neural ones) require fixed sample rates.
2.  **Mono Conversion**:
    *   **Algorithm**: Average channels `(L+R)/2` or select specific channel.
    *   **Why**: Pitch detection is inherently monophonic in the time domain.
3.  **DC Offset Removal**:
    *   **Algorithm**: `y = y - mean(y)`.
    *   **Why**: Removes 0Hz energy that biases RMS calculations.
4.  **High-Pass Filter (HPF)**:
    *   **Algorithm**: Butterworth filter (default order 4, ~55-60Hz).
    *   **Why**: Removes sub-bass rumble and mic handling noise.
5.  **Peak Limiting (Optional)**:
    *   **Algorithm**: Tanh soft-clipping or hard clipping.
    *   **Why**: Tames transients to prevent clipping during normalization.
6.  **Loudness Normalization**:
    *   **Algorithm**: EBU R128 (via `pyloudnorm`) or RMS-based gain.
    *   **Why**: Ensures consistent energy levels for detector confidence thresholds.
7.  **BPM Detection**:
    *   **Algorithm**: `librosa.beat.beat_track` (tightness=100) or Fallback loop.
    *   **Why**: Provides the rhythmic grid for quantization.
8.  **Texture Detection**:
    *   **Algorithm**: Spectral flatness analysis (`detect_audio_type`).
    *   **Why**: Sets `AudioType` (Mono/Poly) to guide Stage B/C algorithm selection.

### Output Contract: `StageAOutput`

```python
@dataclass
class StageAOutput:
    stems: Dict[str, Stem]      # "mix" stem always present
    meta: MetaData              # SR, Duration, BPM, Key, AudioType
    audio_type: AudioType       # MONOPHONIC | POLYPHONIC | POLYPHONIC_DOMINANT
    noise_floor_rms: float      # Estimated noise floor
    beats: List[float]          # Detected beat timestamps (seconds)
    diagnostics: Dict[str, Any] # "bpm_method", "preprocessing_applied", etc.
```

---

## Stage B: Feature Extraction (`backend/pipeline/stage_b.py`)

**Goal:** Extract fundamental frequency (f0) contours, confidence scores, and perform source separation if necessary.

### Strategies & Algorithms

1.  **Instrument Profile Resolution**:
    *   **Strategy**: Resolves `InstrumentProfile` (e.g., "piano_61key") to override detector params.
2.  **Source Separation (Optional)**:
    *   **Trigger**: `config.stage_b.separation.enabled` (True/Auto) AND Polyphonic.
    *   **Algorithm**: `HTDemucs` (Hybrid Transformer Demucs) or `SyntheticMDX`.
    *   **Why**: Isolates instruments (Vocals, Bass, Drums) for dense mixes.
3.  **Adaptive Fusion (Ensemble)**:
    *   **Trigger**: `ensemble_mode="adaptive"`.
    *   **Algorithm**: Weighted median of candidates, gated by signal stability and detector reliability.
    *   **Why**: Robust against single-detector errors; median is more stable than mean for outliers.
4.  **Viterbi Smoothing**:
    *   **Trigger**: `smoothing_method="viterbi"`.
    *   **Algorithm**: HMM-based pathfinding on the pitch curve to minimize octave jumps.
    *   **Why**: Produces smoother, more musical pitch contours than simple frame-by-frame selection.
5.  **Polyphonic Peeling (ISS)**:
    *   **Trigger**: `polyphonic_peeling.max_layers > 0` AND Polyphonic context.
    *   **Algorithm**: Iterative Spectral Subtraction with **Frequency-Aware Masking** (wider masks for bass) and **Adaptive Strength**.
    *   **Why**: Recovers secondary voices (accompaniment) hidden by the melody.

### Configuration & Thresholds

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `confidence_voicing_threshold` | 0.58 | 0.3-0.9 | Minimum confidence for voicing. |
| `ensemble_mode` | "static" | "static","adaptive" | Fusion strategy. |
| `smoothing_method` | "tracker" | "tracker","viterbi" | Smoothing algorithm. |
| `polyphonic_peeling.max_layers` | 8 | 0-16 | Max number of ISS layers. |
| `polyphonic_peeling.iss_adaptive` | True | Bool | Adapt subtraction strength. |

### Output Contract: `StageBOutput`

```python
@dataclass
class StageBOutput:
    time_grid: np.ndarray           # Time values for frames
    f0_main: np.ndarray             # Dominant pitch track (Hz)
    f0_layers: List[np.ndarray]     # Secondary pitch tracks (polyphony)
    stem_timelines: Dict[str, List[FramePitch]] # Per-stem timelines
    per_detector: Dict[str, Any]    # Raw detector outputs
    meta: MetaData                  # Passed through
    diagnostics: Dict[str, Any]     # "fused_f0", "cqt_gate", "iss"
    precalculated_notes: Optional[List[NoteEvent]] # E2E bypass notes
```

---

## Stage C: Theory & Segmentation (`backend/pipeline/stage_c.py`)

**Goal:** Convert continuous frame data into discrete musical `NoteEvents` (Start, End, Pitch, Velocity).

### Strategies & Algorithms

1.  **Skyline Selection**:
    *   **Trigger**: `polyphony_filter.mode = "skyline_top_voice"`.
    *   **Algorithm**: Selects "best" candidate per frame based on Confidence, Continuity, and Vocal Range (80-1400Hz).
2.  **Onset Refinement (`snap_onset`)**:
    *   **Trigger**: `use_onset_refinement=True`.
    *   **Algorithm**: Aligns note start times to local peaks in spectral flux (onset strength).
    *   **Why**: Improves timing accuracy for percussive or distinct attacks.
3.  **Repeated Note Splitter**:
    *   **Trigger**: `use_repeated_note_splitter=True`.
    *   **Algorithm**: Detects re-articulations (energy bumps) within sustained pitch segments to split them.
    *   **Why**: Handles repeated notes (e.g., piano) that pitch detectors see as a continuous line.
4.  **Polyphonic Decomposition**:
    *   **Trigger**: Polyphonic AudioType + `polyphony_filter.mode != "skyline_top_voice"`.
    *   **Algorithm**: Greedily assigns concurrent pitches to stable voice tracks.
    *   **Mode "decomposed_melody"**: Decomposes fully but selects only the single best track (for difficult melody extraction).

### Configuration & Thresholds

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `use_onset_refinement` | True | Bool | Snap to flux peaks. |
| `use_repeated_note_splitter` | True | Bool | Split re-articulations. |
| `min_note_duration_ms` | 30.0 | 10-100 | Min duration (Mono). |
| `min_note_duration_ms_poly` | 45.0 | 30-150 | Min duration (Poly). |
| `confidence_threshold` | 0.20 | 0.1-0.9 | Base activation threshold. |

### Output Contract: `AnalysisData`

```python
@dataclass
class AnalysisData:
    meta: MetaData
    notes: List[NoteEvent]          # The final list of notes
    stem_timelines: Dict[str, List[FramePitch]]
    beats: List[float]              # Beat grid
    diagnostics: Dict[str, Any]     # "segmentation_method", "note_count"
```

---

## Stage D: Rendering (`backend/pipeline/stage_d.py`)

**Goal:** Align notes to the musical grid (Quantization) and export standard formats (MusicXML, MIDI).

### Strategies & Algorithms

1.  **Quantization**:
    *   **Mode: Grid**: Snaps all notes to nearest grid (e.g., 1/16th).
    *   **Mode: Light Rubato**: Only snaps notes that are within `light_rubato_snap_ms` (e.g., 30ms) of a grid line.
    *   **Why**: "Light Rubato" preserves expressive timing for solo performances while fixing obvious errors.
2.  **Voice Assignment**:
    *   **Strategy**: Uses `NoteEvent.voice` ID. Groups events by (Staff, Voice).
    *   **Mapping**: Treble/Bass Staff split (default C4/60) + Grand Staff grouping.
3.  **Music21 Rendering**:
    *   **Objects**: `Note`, `Chord`, `Part`, `Score`, `Glissando`.
    *   **Layout**: Grand Staff group with brace.
4.  **Export**:
    *   **MusicXML**: String dump.
    *   **MIDI**: Binary write (via temp file).

### Configuration & Thresholds

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| `quantization_mode` | "grid" | "grid","light_rubato" | Quantization strategy. |
| `light_rubato_snap_ms` | 30.0 | 10-100 | Snap window for rubato. |
| `quantization_grid` | 16 | 4,8,16,32 | Grid resolution. |
| `forced_key` | None | Str | Override key signature. |

### Output Contract: `TranscriptionResult`

```python
@dataclass
class TranscriptionResult:
    musicxml: str           # Full MusicXML content
    midi_bytes: bytes       # Standard MIDI file content
    analysis_data: AnalysisData # Ref to source data
```
