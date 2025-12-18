# Music Transcription Pipeline Architecture

## 1. Workflow
The system processes audio files through a sequential pipeline of four stages, orchestrated by `backend/pipeline/transcribe.py`:

*   **Stage A (`load_and_preprocess`):** Loads audio, resamples to target rate (default 22.05kHz or 44.1kHz), converts to mono, trims silence, normalizes loudness (EBU R128 or RMS), and estimates the noise floor. It produces a `StageAOutput` object containing the preprocessed audio and metadata.
*   **Stage B (`extract_features`):** The core extraction phase. It resolves the "Instrument Profile" to tune parameters, runs pitch detection algorithms (detectors), merges their outputs into an ensemble result, performs optional source separation (using models like Demucs if enabled), and handles polyphonic "peeling" (Iterative Spectral Subtraction) if required. It outputs a `StageBOutput` with pitch time-grids and confidence scores.
*   **Stage C (`apply_theory`):** Converts the frame-wise pitch data from Stage B into discrete musical notes (`NoteEvent` objects). It applies segmentation (HMM or Threshold-based), enforces minimum note durations, handles confidence hysteresis, and assigns velocity/dynamics based on RMS energy.
*   **Stage D (`quantize_and_render`):** Takes the raw `NoteEvent` objects and aligns them to a musical grid (quantization). It groups notes into staves (Grand Staff for piano), handles voice assignment, and renders the final output as MusicXML and MIDI files.

## 2. Note Extraction Algorithms
The system supports multiple pitch detection algorithms (in `backend/pipeline/detectors.py`), which can run in parallel:

*   **YIN (`YinDetector`):** A time-domain autocorrelation method. Robust for bass and clean signals. Used as a fallback and primary for bass/guitar profiles.
*   **SwiftF0:** A placeholder for a learning-based estimator (requires external model). High priority when enabled.
*   **SACF (`SACFDetector`):** A simplified autocorrelation function detector.
*   **CREPE (`CREPEDetector`):** A neural network-based pitch tracker (requires `crepe` library). High accuracy for monophonic signals like violin/flute.
*   **RMVPE (`RMVPEDetector`):** A robust model for vocal pitch extraction (requires `torch`).
*   **CQT (`CQTDetector`):** Frequency-domain method using Constant-Q Transform. Good for spectral visualization and validation.

## 3. Selection Logic
The algorithm selection is dynamic and hierarchical:

1.  **Instrument Profile:** The highest priority. If an input instrument is specified (e.g., "piano", "violin"), the system loads a preset profile from `config.py`. This profile forces a "recommended algorithm" (e.g., CREPE for violin, YIN for bass) and applies specific tuning overrides.
2.  **Ensemble Weights:** If multiple detectors run, their outputs are merged using a weighted average defined in `config.stage_b.ensemble_weights` (e.g., SwiftF0: 0.5, SACF: 0.3).
3.  **Fallback:** If neural models (SwiftF0/CREPE) are unavailable or fail, the system falls back to DSP-based methods (YIN/SACF) to ensure a result is always produced.

## 4. Key Parameters
Configuration is managed via `backend/pipeline/config.py`. Key parameters include:

*   **`stage_b.confidence_voicing_threshold` (default 0.5):** The confidence score required to consider a frame "voiced" (containing a note).
*   **`stage_b.polyphonic_peeling`:** Controls the Iterative Spectral Subtraction (ISS) for extracting multiple simultaneous notes (polyphony).
*   **`stage_c.min_note_duration_ms` (default 50ms):** Notes shorter than this are discarded as noise.
*   **`stage_c.segmentation_method`:** Chooses between "hmm" (Hidden Markov Model, best for real audio) and "threshold" (best for synthetic/clean audio).
*   **`stage_d.quantization_grid` (default 16):** The grid resolution for rhythm (1/16th notes).

## 5. Key If-Then Logic
*   **Polyphony Detection:** `if audio_type == POLYPHONIC`: The system enables "Polyphonic Peeling" (ISS) in Stage B to extract multiple layers of pitch, and Stage C uses stricter confidence gates for secondary voices.
*   **Instrument Tuning:**
    *   `if instrument == "bass_guitar"`: Stage B switches to larger FFT window sizes (8192) to capture low frequencies.
    *   `if instrument == "distorted_guitar"`: A Low-Pass Filter (LPF) is applied to remove high-frequency distortion before pitch detection.
*   **Synthetic Fallback:** `if segmentation_method == "threshold"`: Stage C bypasses the complex HMM logic and uses simple confidence gating, critical for verifying the pipeline with pure sine waves.

## 6. Benchmark Methodology
Benchmarks are located in `backend/benchmarks/`. The methodology typically involves:
*   **Dataset:** A collection of audio files with known ground truth (MIDI/MusicXML).
*   **Execution:** The pipeline processes these files.
*   **Metrics:** The output notes are compared to ground truth using:
    *   **Precision/Recall/F1-Score:** For note detection accuracy.
    *   **COnion / Mean Deviation:** For pitch accuracy in cents.
    *   **Duration Accuracy:** Comparing start/end times.
*   **Artifacts:** Results are exported to `summary.csv` and `metrics.json` for analysis.

## 7. Segmented Transcription (10s Chunks)
This feature handles long audio files by processing them in chunks.

*   **Invocation:** Set `SegmentedTranscriptionConfig.enabled = True` in your `PipelineConfig` and ensure the audio duration exceeds `segment_sec` (default 10s).
*   **Logic (`transcribe.py`):**
    1.  **Slicing:** The audio is sliced into overlapping 10-second windows (`_slice_stage_a_output`).
    2.  **Processing:** Each slice acts as a mini-Stage A input and goes through Stage B and C.
    3.  **Scoring:** Each segment is scored based on note density and plausibility (`_score_segment`). If a segment scores poorly, the system can retry with different parameters (candidates).
    4.  **Stitching:** The resulting notes from overlapping segments are merged (`_stitch_events`) to form a continuous timeline before being sent to Stage D.
