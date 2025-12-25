# Task 1: Polyphonics Changes

## Diagnostics & Logging (Task 2.4 / 1)
- **Stage C:** Added `selected_stem` to `analysis_data.diagnostics["stage_c"]` to track which audio stem was prioritized for transcription (e.g., 'vocals' vs 'mix').
- **Stage B:** Updated `diagnostics` to explicitly include `transcription_mode`, `separation_mode`, and profile details.
- **Transcribe:** Added logic to record fallback events (BPM fallback, Onsets & Frames fallback) into `analysis_data.diagnostics["fallbacks"]`.
- **Benchmark Runner:** Updated `_save_run` to log `transcription_mode`, `separation_mode`, `selected_stem`, and `fallbacks` into `_run_info.json` and metric summaries.

## Core Logic
- **Mutable Default Fix:** Fixed mutable default argument `seg_cfg` in `stage_c.py`.
- **Snap Onset Robustness:** Implemented clamping in `_segment_monophonic` to ensure `refined_start` respects causal boundaries (`>= prev_end + 1`) and does not drift into invalid regions due to detector noise.
- **Bias Fix:** Updated `snap_onset` to break ties in favor of the original frame index, preventing systematic backward shifts on flat/silent signals.
