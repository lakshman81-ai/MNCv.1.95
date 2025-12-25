# Task 2: Minor Logic Changes

## List of Fixes

1.  **Kill Mutable Default Args (Task 2.1)**
    - **File:** `backend/pipeline/stage_c.py`
    - **Fix:** Changed `seg_cfg: Dict[str, Any] = {}` to `seg_cfg: Optional[Dict[str, Any]] = None` and initialized inside the function.

2.  **Use `math.log2` (Task 2.2)**
    - **Files:** `backend/pipeline/stage_c.py`, `backend/benchmarks/benchmark_runner.py`
    - **Fix:** Replaced scalar usage of `np.log2` with `math.log2` for performance and consistency in pitch-to-cents calculations.

3.  **Normalize Onset Proxy & Clamp (Task 2.3)**
    - **File:** `backend/pipeline/stage_c.py`
    - **Fix:**
        - Updated `snap_onset` to bias towards the original frame index on flat signals (zeros) to prevent unwarranted backward shifts.
        - Added clamp in `_segment_monophonic`:
            - `refined_start = min(refined_start, i)` to ensure onset does not exceed current frame (safety).
            - `refined_start = max(refined_start, current_end + 1)` to prevent overlap with previous note.

4.  **Benchmark Runner Logging (Task 2.4)**
    - **File:** `backend/benchmarks/benchmark_runner.py`
    - **Fix:**
        - Updated `run_pipeline_on_audio` to include `BPM Fallback` logic parity with `transcribe.py`.
        - Updated `_save_run` to log `transcription_mode`, `separation_mode`, `selected_stem`, and `fallbacks`.
        - Ensured `L0` disables `bpm_detection["trim"]` to avoid beat grid shifts on synthetic silence-padded audio.
