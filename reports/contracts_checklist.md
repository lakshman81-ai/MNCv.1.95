# Contracts Checklist

## Data Models vs Stage Usage

### StageAOutput
| Field | Producer (Stage A) | Consumer (Stage B / Transcribe) | Status |
| :--- | :--- | :--- | :--- |
| `stems` | Populated (`mix` mandatory, others optional) | Stage B (`extract_features`) iterates stems. | ✅ |
| `meta` | Populated `MetaData` | Passed through to `StageBOutput`. Used for SR/Hop. | ✅ |
| `audio_type` | Detected via `detect_audio_type` | Used in Stage B for `_is_polyphonic`. | ✅ |
| `noise_floor_rms` | Calculated | Used in Stage C velocity mapping (if passed via meta?). | ⚠️ (AnalysisData gets it from Meta) |
| `beats` | Populated via `detect_tempo_and_beats` | Used in `transcribe.py` for fallback and Stage D quantization. | ✅ |
| `diagnostics` | Populated | Logged/Ignored. | ✅ |

### StageBOutput
| Field | Producer (Stage B) | Consumer (Stage C / Transcribe) | Status |
| :--- | :--- | :--- | :--- |
| `time_grid` | Calculated from `hop_length` | Used for fallback timeline construction in `transcribe.py`. | ✅ |
| `f0_main` | Merged F0 array | Used for fallback. | ✅ |
| `f0_layers` | Populated if polyphonic peeling runs | Not explicitly consumed by Stage C logic yet (only `stem_timelines`). | ⚠️ (Info only) |
| `stem_timelines` | Populated `Dict[str, List[FramePitch]]` | **Critical**. Stage C iterates this to create notes. | ✅ |
| `per_detector` | Populated `Dict[str, Dict]` | Used for diagnostics/metrics. | ✅ |
| `meta` | Passed through | Passed to AnalysisData. | ✅ |
| `precalculated_notes`| Populated if E2E (Basic Pitch) | Checked in Stage C to bypass theory. | ✅ |

### AnalysisData
| Field | Producer (Transcribe / Stage C) | Consumer (Stage D / Output) | Status |
| :--- | :--- | :--- | :--- |
| `meta` | Inherited from Stage A | Used for Tempo, Key, TimeSig in Stage D. | ✅ |
| `timeline` | Inherited from Stage B (Primary) | Used for metrics. | ✅ |
| `notes` | Produced by Stage C (`apply_theory`) | **Critical**. Input to Stage D. | ✅ |
| `stem_timelines` | Inherited | Used for visualizers/metrics. | ✅ |
| `beats` | From Meta or Stage A | Used in Stage D for grid quantization. | ✅ |

### NoteEvent
| Field | Stage C (Creation) | Stage D (Rendering) | Invariant Check |
| :--- | :--- | :--- | :--- |
| `start_sec` | Set (float) | Used for ordering/quantization. | `start < end` |
| `end_sec` | Set (float) | Used for duration. | `start < end` |
| `midi_note` | Set (int) | Used for pitch. | `0 < midi < 128` |
| `pitch_hz` | Set (float) | Info only. | `> 0` |
| `confidence` | Set (float) | Info only. | `0 <= c <= 1` |
| `velocity` | Mapped from RMS (0-1 or 20-105) | Used for MIDI velocity. | Stage D handles >1 check. |
| `voice` | Set (1-based index) | Used for Music21 Voices. | Stable? |
| `staff` | Default 'treble'/'bass' | Used for Part assignment. | Valid Enum? |

## Inter-Stage Invariants

1. **Time Grid Monotonicity**
   - Stage B `time_grid` is `arange * hop / sr`. Monotonic by definition.
   - Stage B `timeline` appends sequential frames. Monotonic.

2. **Timeline Length Alignment**
   - Stage B enforces `canonical_n_frames` based on `mix` stem length.
   - Pad/Trim logic exists in `stage_b.py` for detectors.
   - **Risk**: If stems have different lengths coming out of Separation? `_run_htdemucs` returns same-length stems. `SyntheticMDX` returns same-length. Stage A slice returns same-length.

3. **Note Start < End**
   - Stage C `_segment_monophonic` produces segments `(s, e)` where `s <= e`.
   - `start_sec = tl[s].time`, `end_sec = tl[e].time + hop`.
   - `hop > 0`. So `end > start`.
   - `_sanitize_notes` enforces `end > start`.

4. **Beat Arrays Sane**
   - Stage A `detect_tempo_and_beats` returns `sorted(list(set(beat_times)))`.
   - `transcribe.py` fallback sorts and dedups.
   - `stage_d.py` sorts and uniques beat times.
   - **Risk**: Empty beat array handled? Yes (`use_beat_grid` flag).

5. **Voice/Staff Stability**
   - Stage C assigns `voice_id = vidx * 16 + sub_idx + 1`. Deterministic.
   - Stage D handles `voice_idx`.
   - Stage D infers staff from split point if not set. Stage C sets default? No, Stage C `NoteEvent` defaults `staff="treble"`. `stage_c.py` does `staff=n.staff` (copy). `_segment_monophonic` creates NoteEvent with explicit staff? No, `NoteEvent` constructor defaults used?
   - **Correction**: `stage_c.py:1260` creates `NoteEvent(..., staff=n.staff)`? No, `stage_c.py:1260` creates `NoteEvent(..., staff="treble")` implicitly?
   - **Check**: `stage_c.py:1255`: `NoteEvent(..., voice=voice_id)`. `staff` arg is missing? No, `NoteEvent` definition has `staff: str = "treble"`.
   - **Issue**: Stage C does NOT set staff logic based on pitch? `quantize_and_render` in Stage D (line 263) recalculates staff: `if staff_name not in ("treble", "bass"): staff_name = "treble" if ...`.
   - **Verdict**: Safe (Stage D handles assignment).
