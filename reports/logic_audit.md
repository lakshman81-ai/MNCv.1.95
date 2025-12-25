# Logic Audit: MusicNote Pipeline

## Top 10 Risk Points

1. **Mutable Default Argument in `AnalysisData`**
   - **File**: `backend/pipeline/models.py:127`
   - **Risk**: `beat_times: List[float] = field(default_factory=list)` is correct, but `stage_a.py:detect_tempo_and_beats` returns `beat_times = []` on failure. However, in `stage_b.py`, `MultiVoiceTracker.__init__` has no mutable defaults, but `_postprocess_candidates` uses `candidates: List[Tuple[float, float]]` which is fine.
   - **Correction**: `backend/pipeline/stage_c.py:833`: `_dedupe_overlapping_notes(notes: List[NoteEvent], overlap_thr: float = 0.85)` is safe.
   - **Found**: `backend/pipeline/stage_c.py:465`: `_segment_monophonic` uses `seg_cfg: Dict[str, Any] = {}`. This is a classic mutable default argument bug. If `seg_cfg` is modified in place, it persists across calls.
   - **Severity**: High.

2. **Unit Mismatch in `min_note_duration`**
   - **File**: `backend/pipeline/transcribe.py:108` vs `backend/pipeline/stage_c.py:1125`
   - **Risk**: `transcribe.py` calculates score using `config.stage_c.min_note_duration_ms / 1000.0`. `stage_c.py` resolves `min_note_dur_ms` and converts to seconds: `min_note_dur_s = float(min_note_dur_ms) / 1000.0`.
   - **Issue**: `_segment_monophonic` takes `min_note_dur_s`. `quantize_notes` takes `min_steps` (grid units).
   - **Observation**: In `stage_d.py`, `quantize_and_render` calculates `dur_beats`. If `use_beat_grid` is False, it uses `quarter_dur`.
   - **Risk**: `snap_chord_starts` takes `tol_ms` but converts to seconds `tol = tol_ms / 1000.0`.
   - **Found**: `backend/pipeline/stage_c.py:476`: `time_merge_frames = int(gap_tolerance_s / hop_s)`. `gap_tolerance_s` defaults to 0.05. `hop_s` is estimated from timeline. If `hop_s` is 0.01 (10ms), merge is 5 frames. But `stage_c.py:442` sets `gap_tolerance_s` default to 0.05.
   - **Severity**: Medium (Consistency checked, seems OK but fragile).

3. **None Handling in `StageBOutput` timeline**
   - **File**: `backend/pipeline/transcribe.py:441`
   - **Risk**: `timeline_source = list(stage_b_out.timeline or [])`. `stage_b.py` constructs `StageBOutput` with `timeline=primary_timeline or []`. `primary_timeline` comes from `stem_timelines`.
   - **Issue**: If `f0_main` is populated but `stem_timelines` is empty (e.g., `_arrays_to_timeline` failed or wasn't called properly in fallback), `timeline` is empty. `transcribe.py` attempts to rebuild it from `time_grid` and `f0_main` ONLY if `timeline_source` is empty.
   - **Logic**: `stage_b.py:1159` sets `f0_main` if None.
   - **Severity**: Low (Fallback exists).

4. **Silent Failure in `detect_tempo_and_beats`**
   - **File**: `backend/pipeline/stage_a.py:326`
   - **Risk**: Catches `Exception as exc` and returns `None, []`.
   - **Impact**: If `librosa` fails or audio is weird, BPM defaults to 120.0 silently (logged only).
   - **Severity**: Medium (Data quality).

5. **Potential IndexOutOfBounds in `_cqt_mag_at`**
   - **File**: `backend/pipeline/stage_b.py:84`
   - **Risk**: `j = int(np.clip(np.searchsorted(freqs, hz), 0, freqs.size - 1))`. `freqs` comes from `librosa.cqt_frequencies`.
   - **Issue**: If `freqs` is empty (caught in `_maybe_compute_cqt_ctx`), returns None. But `_cqt_mag_at` checks `ctx is None`.
   - **Logic**: `mag[j, t]`. `t` is clamped. `j` is clamped. Safe.
   - **Severity**: Low (Guarded).

6. **Inconsistent Audio Type Checks**
   - **File**: `backend/pipeline/stage_b.py:873` vs `backend/pipeline/transcribe.py`
   - **Risk**: `_is_polyphonic` checks `AudioType` enum OR string "poly".
   - **Issue**: `stage_a.py` returns `AudioType` enum. `transcribe.py` passes `AudioType` to `_score_segment`.
   - **Risk**: String check "poly" in `audio_type.lower()` might match "polyphonic_dominant" correctly, but relies on stringiness if object is not Enum. `models.py` defines `AudioType(str, Enum)`.
   - **Severity**: Low.

7. **Loop Variable Leak in `apply_theory`**
   - **File**: `backend/pipeline/stage_c.py:1101`
   - **Risk**: `for vidx, (vname, timeline) in enumerate(timelines_to_process):`. Inside loop, `seg_cfg_local` is created. `notes` list is appended.
   - **Issue**: `quantized_notes = quantize_notes(notes, ...)`. `notes` accumulates across stems. `_dedupe_overlapping_notes` is called on the full list `notes`.
   - **Risk**: `_dedupe_overlapping_notes` sorts by voice. If multiple stems produce same voice ID (logic: `voice_id = (vidx * 16) + ...`), they are distinct.
   - **Severity**: Low (Logic seems sound).

8. **Hardcoded Sample Rates / Magic Numbers**
   - **File**: `backend/pipeline/stage_b.py:537`
   - **Risk**: `SyntheticMDXSeparator` uses `base_freqs = [110.0, 220.0, ...]` and `sr=44100` default.
   - **Issue**: If input audio is 22050 (Stage A default for Piano), `SyntheticMDXSeparator` (if used) might mismatch if it assumes 44100 generation but receives 22050.
   - **Check**: `SyntheticMDXSeparator.__init__` takes `sample_rate`. Called in `_resolve_separation` with `mix_stem.sr`. Correct.
   - **Severity**: Low.

9. **Heavy dependency imports inside functions**
   - **File**: `backend/pipeline/stage_d.py`
   - **Risk**: `import music21`. Wrapped in try/except at module level.
   - **Issue**: `quantize_and_render` checks `MUSIC21_AVAILABLE`.
   - **Risk**: `stage_b.py` imports `torch`, `demucs`.
   - **Issue**: `_run_htdemucs` imports inside function. Good.
   - **Severity**: Low (Best practice followed).

10. **Unbounded Growth in `AnalysisData.diagnostics`**
    - **File**: `backend/pipeline/transcribe.py`
    - **Risk**: `analysis_data.diagnostics` accumulates dictionaries from all stages.
    - **Issue**: If processing very long files or loops (segmented), this could grow. But segmentation creates new `AnalysisData` per segment? No, `AnalysisData` is per transcription result.
    - **Severity**: Low.

## Other Findings

- **Mutable Default**: `backend/pipeline/stage_c.py:465`: `seg_cfg: Dict[str, Any] = {}`.
- **Logic**: `backend/pipeline/stage_d.py:270`: `if voice_idx is None: voice_idx = 1`. Later `voice_idx` used as key.
- **Contract**: `StageBOutput.per_detector` is `Dict[str, Any]`. Schema says `per_detector[stem_name][det_name] = (f0, conf)`. Verified in `stage_b.py`.
- **Contract**: `StageAOutput.stems` is `Dict[str, Stem]`. Verified.
