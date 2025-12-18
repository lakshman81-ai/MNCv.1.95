# backend/pipeline/stage_c.py
"""
Stage C — Theory / Note segmentation

This module converts frame-wise pitch timelines into discrete NoteEvent objects.

Unit-test compatibility
-----------------------
backend/tests/test_stage_c.py expects:
  - apply_theory
  - quantize_notes

Important model constraints (from backend/pipeline/models.py)
-----------------------------------------------------------
NoteEvent fields:
  start_sec, end_sec, midi_note, pitch_hz, confidence, velocity
(no source_stem/source_detector fields)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Support both package and top‑level imports for models
try:
    from .models import AnalysisData, FramePitch, NoteEvent, AudioType  # type: ignore
except Exception:
    from models import AnalysisData, FramePitch, NoteEvent, AudioType  # type: ignore

from copy import deepcopy

def _get(obj: Any, path: str, default: Any = None) -> Any:
    if obj is None:
        return default
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return cur


def _velocity_from_rms(rms_list: List[float], vmin: int = 20, vmax: int = 105) -> int:
    # C4: Velocity mapping from RMS (20–105)
    if not rms_list:
        return 64

    rms = float(np.mean(rms_list)) # Use mean of note RMS

    # Default range -40dB to -4dB
    min_rms = 10**(-40/20)
    max_rms = 10**(-4/20)

    x = (rms - min_rms) / max(max_rms - min_rms, 1e-9)
    x = float(np.clip(x, 0.0, 1.0))
    x = x ** 0.6
    v = 20 + int(round(x * (105 - 20)))
    return int(np.clip(v, 20, 105))

def _velocity_to_dynamic(v: int) -> str:
    # C4: Dynamic label assignment
    if v < 30: return "pp"
    if v < 45: return "p"
    if v < 60: return "mp"
    if v < 75: return "mf"
    if v < 90: return "f"
    return "ff"

def _cents_diff_hz(a, b):
    if a <= 0 or b <= 0:
        return 1e9
    return abs(1200.0 * np.log2((a + 1e-9) / (b + 1e-9)))

def _estimate_hop_seconds(timeline: List[FramePitch]) -> float:
    if len(timeline) < 2:
        return 0.01
    dt = [timeline[i].time - timeline[i - 1].time for i in range(1, min(len(timeline), 50))]
    if not dt:
        return 0.01
    hop_s = float(np.median(dt))
    return max(1e-4, hop_s)


def _has_distinct_poly_layers(timeline: List[FramePitch], cents_tolerance: float = 35.0) -> bool:
    """Return True when active_pitches include clearly different layers."""
    for fp in timeline:
        if not getattr(fp, "active_pitches", None) or len(fp.active_pitches) < 2:
            continue

        pitches = [p for (p, _) in fp.active_pitches if p > 0.0]
        if len(pitches) < 2:
            continue

        ref = pitches[0]
        for other in pitches[1:]:
            if ref <= 0 or other <= 0:
                continue
            cents = abs(1200.0 * np.log2(other / ref))
            if cents > cents_tolerance:
                return True

    return False


def _segment_monophonic(
    timeline: List[FramePitch],
    conf_thr: float,
    min_note_dur_s: float,
    gap_tolerance_s: float,
    semitone_stability: float = 0.60,
    min_rms: float = 0.01,
    conf_start: float | None = None,
    conf_end: float | None = None,
    seg_cfg: Dict[str, Any] = {},
    hop_s: float = 0.01,
) -> List[Tuple[int, int]]:
    """
    Segment monophonic FramePitch into (start_idx, end_idx) segments.
    Updated for C1 (hysteresis) and C2 (gap merge/vibrato).
    """
    if len(timeline) < 2:
        return []

    # C1: Frame stability onset (>= 3 frames) + release hysteresis (2-3 frames)
    min_on = int(seg_cfg.get("min_onset_frames", 3))
    rel = int(seg_cfg.get("release_frames", 2))

    # C2 Params
    split_semi = float(seg_cfg.get("split_semitone", 0.7))
    split_cents = split_semi * 100.0

    # P4: Prioritize time_merge_frames logic
    tmf_cfg = seg_cfg.get("time_merge_frames")
    if tmf_cfg is not None:
        time_merge_frames = int(tmf_cfg)
    else:
        # P4: Default fallback
        time_merge_frames = int(gap_tolerance_s / hop_s)
    time_merge_frames = max(0, time_merge_frames)

    segs: List[Tuple[int, int]] = []

    stable = 0
    silent = 0
    active = False

    current_start = -1
    current_end = -1
    current_pitch_hz = 0.0

    # P5: Pitch reference buffer for vibrato stability
    pitch_buffer: List[float] = []
    pitch_buffer_size = 7

    # P2: Apply confidence hysteresis
    c_start = conf_start if conf_start is not None else conf_thr
    c_end = conf_end if conf_end is not None else conf_thr

    for i, fp in enumerate(timeline):
        # P2 Hysteresis logic
        if not active:
             is_voiced_frame = (fp.pitch_hz > 0.0 and fp.confidence >= c_start)
        else:
             is_voiced_frame = (fp.pitch_hz > 0.0 and fp.confidence >= c_end)

        if is_voiced_frame:
            silent = 0
            if not active:
                stable += 1
                if stable >= min_on:
                    active = True
                    current_start = i - (min_on - 1)
                    current_end = i
                    current_pitch_hz = fp.pitch_hz
                    pitch_buffer = [fp.pitch_hz]
            else:
                # Check vibrato / pitch jump - C2
                diff = _cents_diff_hz(fp.pitch_hz, current_pitch_hz)
                if diff <= split_cents:
                    # Extend note
                    current_end = i
                    # P5: Update pitch reference to track drift/vibrato
                    pitch_buffer.append(fp.pitch_hz)
                    if len(pitch_buffer) > pitch_buffer_size:
                        pitch_buffer.pop(0)
                    current_pitch_hz = float(np.median(pitch_buffer))
                else:
                    # Split note
                    segs.append((current_start, current_end))

                    # Reset for new note
                    current_start = i
                    current_end = i
                    current_pitch_hz = fp.pitch_hz
                    pitch_buffer = [fp.pitch_hz]
                    stable = min_on

        else:
            stable = 0
            if active:
                silent += 1
                if silent >= rel:
                    # Finalize
                    segs.append((current_start, current_end))
                    active = False
                    current_start = -1
                    current_end = -1

    # Close pending
    if active and current_start != -1:
        segs.append((current_start, current_end))

    # C2: Gap merge (1-frame)
    if len(segs) < 2:
        pass # Optimization: fall through to return merged_segs if empty

    merged_segs = []
    if segs:
        curr_s, curr_e = segs[0]
        # Calculate pitch for current seg to check against next
        def get_seg_pitch(s, e):
            p = [timeline[x].pitch_hz for x in range(s, e+1) if timeline[x].pitch_hz > 0]
            if not p: return 0.0
            return np.median(p)

        curr_p = get_seg_pitch(curr_s, curr_e)

        for i in range(1, len(segs)):
            next_s, next_e = segs[i]
            next_p = get_seg_pitch(next_s, next_e)

            # Gap in frames
            gap = next_s - curr_e - 1

            if gap <= time_merge_frames and _cents_diff_hz(curr_p, next_p) <= split_cents:
                # Merge
                curr_e = next_e
                # Keep curr_p stable
            else:
                # P3: Enforce min_note_duration_in_threshold_mode
                dur = (curr_e - curr_s + 1) * hop_s
                if dur >= min_note_dur_s:
                    merged_segs.append((curr_s, curr_e))

                curr_s, curr_e = next_s, next_e
                curr_p = next_p

        # Check last
        dur = (curr_e - curr_s + 1) * hop_s
        if dur >= min_note_dur_s:
            merged_segs.append((curr_s, curr_e))

    return merged_segs


def _viterbi_voicing_mask(
    timeline: List[FramePitch],
    conf_weight: float,
    energy_weight: float,
    transition_penalty: float,
    stay_bonus: float,
    silence_bias: float,
) -> np.ndarray:
    if len(timeline) == 0:
        return np.zeros(0, dtype=bool)

    mids = np.array([fp.midi if fp.midi is not None else -1 for fp in timeline], dtype=np.float64)
    conf = np.clip(np.array([fp.confidence for fp in timeline], dtype=np.float64), 0.0, 1.0)
    rms = np.array([fp.rms for fp in timeline], dtype=np.float64)

    # Normalize RMS to [0,1] range to support use as a confidence prior
    rms_norm = rms.copy()
    if np.any(rms_norm > 0):
        rms_norm /= float(np.percentile(rms_norm[rms_norm > 0], 95))
    rms_norm = np.clip(rms_norm, 0.0, 1.0)

    voiced_score = conf_weight * conf + energy_weight * rms_norm
    silence_score = (1.0 - conf_weight) * (1.0 - conf) + (1.0 - energy_weight) * (1.0 - rms_norm) + silence_bias

    n = len(timeline)
    voiced_cost = np.zeros(n, dtype=np.float64)
    silence_cost = np.zeros(n, dtype=np.float64)
    backpointer = np.zeros((n, 2), dtype=np.int8)

    voiced_cost[0] = -voiced_score[0]
    silence_cost[0] = -silence_score[0]

    for i in range(1, n):
        # Transition into voiced
        stay_voiced = voiced_cost[i - 1] - stay_bonus
        switch_to_voiced = silence_cost[i - 1] + transition_penalty
        if stay_voiced <= switch_to_voiced:
            voiced_cost[i] = stay_voiced
            backpointer[i, 1] = 1  # came from voiced
        else:
            voiced_cost[i] = switch_to_voiced
            backpointer[i, 1] = 0  # came from silence
        voiced_cost[i] -= voiced_score[i]

        # Transition into silence
        stay_silence = silence_cost[i - 1] - stay_bonus
        switch_to_silence = voiced_cost[i - 1] + transition_penalty
        if stay_silence <= switch_to_silence:
            silence_cost[i] = stay_silence
            backpointer[i, 0] = 0
        else:
            silence_cost[i] = switch_to_silence
            backpointer[i, 0] = 1
        silence_cost[i] -= silence_score[i]

    # Backtrack
    state = 1 if voiced_cost[-1] <= silence_cost[-1] else 0
    mask = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        mask[i] = state == 1 and mids[i] > 0
        state = int(backpointer[i, state])

    return mask


def _segments_from_mask(
    timeline: List[FramePitch],
    mask: np.ndarray,
    hop_s: float,
    min_note_dur_s: float,
    min_conf: float,
    min_rms: float,
) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    i = 0
    n = len(timeline)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        e = i

        times = [timeline[j].time for j in range(s, e + 1)]
        confs = [timeline[j].confidence for j in range(s, e + 1)]
        rms_vals = [timeline[j].rms for j in range(s, e + 1)]

        dur = float(times[-1] - times[0] + hop_s)
        if dur >= min_note_dur_s and np.mean(confs) >= min_conf and np.mean(rms_vals) >= min_rms:
            segs.append((s, e))

        i += 1

    return segs


def apply_theory(analysis_data: AnalysisData, config: Any = None) -> List[NoteEvent]:
    """
    Convert FramePitch timelines into NoteEvent list.

    - Uses analysis_data.stem_timelines if analysis_data.timeline is empty.
    - Prefers stem order: mix -> vocals -> first available.
    - Applies default rhythmic quantization (1/16 grid at detected tempo).
    """
    # Legacy call signature support: apply_theory(timeline, analysis_data)
    if isinstance(analysis_data, list) and isinstance(config, AnalysisData):
        legacy_timeline = analysis_data or getattr(config, "timeline", [])
        analysis_data = config
        if not analysis_data.stem_timelines:
            analysis_data.stem_timelines = {"mix": legacy_timeline}
    elif not isinstance(analysis_data, AnalysisData):
        return []

    # Patch F0: Apply instrument profile overrides inside Stage C
    # Resolve instrument from metadata if available, else config
    meta_instr = getattr(analysis_data.meta, "instrument", None)
    config_instr = _get(config, "stage_b.instrument", "piano_61key")

    # Priority: MetaData (resolved in Stage B) -> Config -> Default
    instrument_name = meta_instr if meta_instr else config_instr

    profile = None
    profile_special = {}

    # Check if overrides are enabled
    apply_profile = _get(config, "stage_c.apply_instrument_profile", True)

    if apply_profile and config and hasattr(config, "get_profile"):
        profile = config.get_profile(str(instrument_name))
        if profile:
            profile_special = dict(getattr(profile, "special", {}) or {})

    # Helper to resolve a config value with override priority:
    # 1. profile.special['stage_c_X'] (if exists)
    # 2. config.stage_c.X
    # 3. default

    def resolve_val(key, default):
        # Try profile special first (e.g. stage_c_min_note_duration_ms)
        special_key = f"stage_c_{key}"
        if special_key in profile_special:
            return profile_special[special_key]
        # Then config
        return _get(config, f"stage_c.{key}", default)

    stem_timelines: Dict[str, List[FramePitch]] = analysis_data.stem_timelines or {}

    if not stem_timelines:
        analysis_data.notes = []
        return []

    if "mix" in stem_timelines:
        stem_name = "mix"
    elif "vocals" in stem_timelines:
        stem_name = "vocals"
    else:
        stem_name = next(iter(stem_timelines.keys()))

    primary_timeline = stem_timelines.get(stem_name, [])
    if len(primary_timeline) < 2:
        analysis_data.notes = []
        return []

    # Thresholds (read from config if available)
    base_conf = float(resolve_val("confidence_threshold", _get(config, "stage_c.special.high_conf_threshold", 0.15)))
    hyst_conf = resolve_val("confidence_hysteresis", {}) or {}
    start_conf = float(hyst_conf.get("start", base_conf))
    end_conf = float(hyst_conf.get("end", base_conf))

    poly_conf = float(_get(config, "stage_c.polyphonic_confidence.melody", base_conf))
    accomp_conf = float(_get(config, "stage_c.polyphonic_confidence.accompaniment", poly_conf))
    conf_thr = base_conf

    min_note_dur_ms = resolve_val("min_note_duration_ms", 50.0)
    min_note_dur_s = float(min_note_dur_ms) / 1000.0

    min_note_dur_ms_poly = resolve_val("min_note_duration_ms_poly", None)
    gap_tolerance_s = float(resolve_val("gap_tolerance_s", 0.05))

    # Calculate min_rms
    min_db = float(_get(config, "stage_c.velocity_map.min_db", -40.0))
    min_rms = 10 ** (min_db / 20.0)

    # Use noise floor from Stage A if available
    nf = 0.0
    try:
        nf = float(getattr(getattr(analysis_data, "meta", None), "noise_floor_rms", 0.0) or 0.0)
    except Exception:
        nf = 0.0

    if nf > 0.0:
        # require ~+6 dB above estimated noise floor
        min_rms = max(min_rms, nf * (10 ** (6.0 / 20.0)))

    # Build list of timelines to process.
    timelines_to_process: List[Tuple[str, List[FramePitch]]] = [(stem_name, primary_timeline)]
    audio_type = getattr(analysis_data.meta, "audio_type", None)
    allow_secondary = audio_type in (getattr(AudioType, "POLYPHONIC", None), getattr(AudioType, "POLYPHONIC_DOMINANT", None))
    if allow_secondary and len(stem_timelines) > 1:
        for other_name, other_timeline in stem_timelines.items():
            if other_name == stem_name or len(other_timeline) < 2:
                continue
            timelines_to_process.append((other_name, other_timeline))

    notes: List[NoteEvent] = []

    seg_cfg = resolve_val("segmentation_method", {}) or {}
    seg_method = str(seg_cfg.get("method", "threshold")).lower()

    # P1: Auto-enable smoothing for HMM
    smoothing_enabled = bool(seg_cfg.get("use_state_smoothing", False))
    if seg_method == "hmm":
        smoothing_enabled = True

    transition_penalty = float(seg_cfg.get("transition_penalty", 0.8))
    stay_bonus = float(seg_cfg.get("stay_bonus", 0.05))
    silence_bias = float(seg_cfg.get("silence_bias", 0.1))
    energy_weight = float(seg_cfg.get("energy_weight", 0.35))
    conf_weight = max(0.0, min(1.0, 1.0 - energy_weight))

    for vidx, (vname, timeline) in enumerate(timelines_to_process):
        # Detect polyphonic context based on active pitch annotations
        poly_frames = [fp for fp in timeline if getattr(fp, "active_pitches", []) and len(fp.active_pitches) > 1]
        has_distinct_poly = _has_distinct_poly_layers(timeline)

        voice_conf_gate = conf_thr
        voice_min_dur_s = min_note_dur_s

        if poly_frames:
            # PC7 (Part 1): Update assignment tolerances indirectly by using correct gates
            voice_conf_gate = poly_conf if vidx == 0 else accomp_conf
            try:
                if min_note_dur_ms_poly is not None:
                    voice_min_dur_s = max(voice_min_dur_s, float(min_note_dur_ms_poly) / 1000.0)
            except Exception:
                pass

            if vidx > 0 and not has_distinct_poly:
                voice_conf_gate = max(voice_conf_gate, accomp_conf)
                try:
                    if min_note_dur_ms_poly is not None:
                        voice_min_dur_s = max(voice_min_dur_s, float(min_note_dur_ms_poly) / 1000.0)
                except Exception:
                    pass
        elif vidx > 0:
            voice_conf_gate = max(voice_conf_gate, accomp_conf)

        hop_s = _estimate_hop_seconds(timeline)

        use_viterbi = smoothing_enabled and seg_method in ("viterbi", "hmm")
        if use_viterbi:
            mask = _viterbi_voicing_mask(
                timeline,
                conf_weight=conf_weight,
                energy_weight=energy_weight,
                transition_penalty=transition_penalty,
                stay_bonus=stay_bonus,
                silence_bias=silence_bias,
            )
            segs = _segments_from_mask(
                timeline=timeline,
                mask=mask,
                hop_s=hop_s,
                min_note_dur_s=voice_min_dur_s,
                min_conf=voice_conf_gate,
                min_rms=min_rms,
            )
        else:
            segs = _segment_monophonic(
                timeline=timeline,
                conf_thr=voice_conf_gate,
                min_note_dur_s=voice_min_dur_s,
                gap_tolerance_s=gap_tolerance_s,
                min_rms=min_rms,
                conf_start=max(start_conf, voice_conf_gate),
                conf_end=min(max(end_conf, 0.0), max(start_conf, voice_conf_gate)),
                seg_cfg=seg_cfg,
                hop_s=hop_s,
            )

        for (s, e) in segs:
            mids = [timeline[i].midi for i in range(s, e + 1) if timeline[i].midi is not None and timeline[i].midi > 0]
            hzs = [timeline[i].pitch_hz for i in range(s, e + 1) if timeline[i].pitch_hz > 0]
            confs = [timeline[i].confidence for i in range(s, e + 1)]
            rmss = [timeline[i].rms for i in range(s, e + 1)]

            if not mids:
                continue

            # Enforce RMS gate (important for noise floor handling)
            if rmss and np.mean(rmss) < min_rms:
                continue

            midi_note = int(round(float(np.median(mids))))
            pitch_hz = float(np.median(hzs)) if hzs else 0.0
            confidence = float(np.mean(confs)) if confs else 0.0

            # P6: Velocity semantics + RMS value
            midi_vel = _velocity_from_rms(rmss)
            velocity_norm = float(midi_vel) / 127.0
            rms_val = float(np.mean(rmss)) if rmss else 0.0

            dynamic_label = _velocity_to_dynamic(midi_vel)

            start_sec = float(timeline[s].time)
            end_sec = float(timeline[e].time + hop_s)
            if end_sec <= start_sec:
                end_sec = start_sec + hop_s

            notes.append(
                NoteEvent(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    midi_note=midi_note,
                    pitch_hz=pitch_hz,
                    confidence=confidence,
                    velocity=velocity_norm, # Normalized 0..1
                    rms_value=rms_val,
                    dynamic=dynamic_label,
                    voice=int(vidx + 1),
                )
            )

    # Patch F8: Optional Bass Backtracking
    # Improves timing of bass notes which often have slow attacks or pre-transient noise.
    bass_backtrack_ms = float(profile_special.get("bass_backtrack_ms", 0.0))
    if bass_backtrack_ms > 0.0 and notes:
        backtrack_sec = bass_backtrack_ms / 1000.0
        # Sort to ensure safe lookback
        notes.sort(key=lambda n: n.start_sec)
        for i, n in enumerate(notes):
            # Only apply if not overlapping previous note too much
            prev_end = notes[i-1].end_sec if i > 0 else 0.0
            new_start = n.start_sec - backtrack_sec

            # Allow slight overlap or butt up against previous
            if new_start >= prev_end - 0.05:
                n.start_sec = max(0.0, new_start)

    quantized_notes = quantize_notes(notes, analysis_data=analysis_data)
    analysis_data.notes = quantized_notes
    return quantized_notes


def _sec_to_beat_index(t: float, beat_times: list[float]) -> float:
    n = len(beat_times)
    if n < 2:
        return 0.0

    bt = np.asarray(beat_times, dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)

    # interior mapping
    if bt[0] <= t <= bt[-1]:
        return float(np.interp(t, bt, idx))

    # edge intervals (robust)
    dt0 = float(bt[1] - bt[0])
    dt1 = float(bt[-1] - bt[-2])

    # guard against bad beat arrays
    if dt0 <= 1e-6 or dt1 <= 1e-6:
        # fallback: clamp rather than explode
        return 0.0 if t < bt[0] else float(n - 1)

    if t < bt[0]:
        return float((t - bt[0]) / dt0)  # beat 0 + negative offset
    else:
        return float((n - 1) + (t - bt[-1]) / dt1)  # extend beyond last beat


def _beat_index_to_sec(b: float, beat_times: list[float]) -> float:
    n = len(beat_times)
    if n < 2:
        return 0.0

    bt = np.asarray(beat_times, dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)

    if 0.0 <= b <= float(n - 1):
        return float(np.interp(b, idx, bt))

    dt0 = float(bt[1] - bt[0])
    dt1 = float(bt[-1] - bt[-2])
    if dt0 <= 1e-6 or dt1 <= 1e-6:
        return float(bt[0] if b < 0 else bt[-1])

    if b < 0.0:
        return float(bt[0] + b * dt0)
    else:
        return float(bt[-1] + (b - float(n - 1)) * dt1)


def quantize_notes(
    notes: List[NoteEvent],
    tempo_bpm: float = 120.0,
    grid: str = "1/16",
    min_steps: int = 1,
    analysis_data: AnalysisData | None = None,
) -> List[NoteEvent]:
    """
    Quantize note start/end times to a rhythmic grid.

    Parameters
    ----------
    notes : List[NoteEvent]
    tempo_bpm : float
        Tempo used to convert beats to seconds when no analysis_data is provided.
    analysis_data : AnalysisData, optional
        Preferred source for tempo and time signature; if provided, meta fields
        take precedence over the tempo_bpm argument.
    grid : str
        Grid like "1/16", "1/8", "1/4". Interpreted as fraction of a whole note.
        In 4/4: one beat = quarter note, so step_beats = 4/denom.
    min_steps : int
        Minimum duration in grid steps.

    Returns
    -------
    List[NoteEvent]
        New list with quantized timing.
    """
    if not notes:
        return []

    analysis: Optional[AnalysisData] = analysis_data

    beat_times = []
    if analysis is not None:
        beat_times = list(getattr(analysis, "beats", []) or [])
        if not beat_times:
            meta = getattr(analysis, "meta", None)
            if meta is not None:
                beat_times = list(getattr(meta, "beat_times", []) or getattr(meta, "beats", []) or [])

    use_beat_times = len(beat_times) >= 2

    bpm_source = None
    if analysis is not None:
        bpm_source = _get(analysis, "meta.tempo_bpm", None)
        if bpm_source is None:
            beats_seq = beat_times
            if beats_seq:
                diffs = np.diff(sorted(beats_seq))
                if diffs.size:
                    median_diff = float(np.median(diffs))
                    if median_diff > 0:
                        bpm_source = 60.0 / median_diff

    if bpm_source is None:
        bpm_source = tempo_bpm

    bpm = float(bpm_source) if bpm_source and bpm_source > 0 else None
    use_soft_snap = bpm is None or not np.isfinite(bpm)
    effective_bpm = bpm if bpm and np.isfinite(bpm) else 100.0
    sec_per_beat = 60.0 / effective_bpm

    # Parse grid
    denom = 16
    if not use_soft_snap:
        try:
            m = grid.strip().split("/")
            if len(m) == 2:
                denom = int(m[1])
            else:
                denom = int(grid)
        except Exception:
            denom = 16
        denom = max(1, denom)
        # Adaptive grid: slower tempi -> coarser grid
        if effective_bpm < 75:
            denom = min(denom, 8)
        elif effective_bpm > 140:
            denom = max(denom, 32)
    else:
        denom = 8

    step_beats = 4.0 / float(max(1, denom))  # in 4/4, quarter note = 1 beat
    step_sec = sec_per_beat * step_beats
    step_sec = max(1e-4, step_sec)

    if use_soft_snap:
        durations = [float(n.end_sec - n.start_sec) for n in notes if n.end_sec > n.start_sec]
        median_dur = float(np.median(durations)) if durations else step_sec
        soft_step = max(0.08, min(step_sec * 1.5, median_dur * 0.5))
        step_sec = max(step_sec, soft_step)

    beats_per_measure = 4
    if analysis is not None:
        ts = _get(analysis, "meta.time_signature", "4/4") or "4/4"
        try:
            num, _den = ts.split("/")
            beats_per_measure = max(1, int(num))
        except Exception:
            beats_per_measure = 4

    out: List[NoteEvent] = []
    for n in notes:
        # Quantize Logic
        if use_beat_times:
            # Beat-space quantization
            bs = _sec_to_beat_index(float(n.start_sec), beat_times)
            be = _sec_to_beat_index(float(n.end_sec), beat_times)

            qbs = round(bs / step_beats) * step_beats
            qbe = round(be / step_beats) * step_beats

            if qbe <= qbs:
                qbe = qbs + max(int(min_steps), 1) * step_beats

            # Convert back to seconds
            qs = _beat_index_to_sec(qbs, beat_times)
            qe = _beat_index_to_sec(qbe, beat_times)

            # Clamp negative times from backward extrapolation
            qs = max(0.0, qs)
            qe = max(0.0, qe)

            # Re-enforce duration in seconds after clamping
            if qe <= qs:
                # Fallback step sec estimation if needed, but we can try to respect grid
                # Just bump qe
                qe = qs + 0.05

            # For NoteEvent metrics
            beat_idx = qbs
            duration_beats = qbe - qbs

        else:
            # Constant BPM fallback
            s = float(n.start_sec)
            e = float(n.end_sec)
            qs = round(s / step_sec) * step_sec
            qe = round(e / step_sec) * step_sec

            if qe <= qs:
                qe = qs + max(int(min_steps), 1) * step_sec
            if (qe - qs) < max(int(min_steps), 1) * step_sec:
                qe = qs + max(int(min_steps), 1) * step_sec

            beat_idx = qs / sec_per_beat
            duration_beats = (qe - qs) / sec_per_beat

        measure = int(beat_idx // beats_per_measure) + 1
        beat_in_measure = (beat_idx % beats_per_measure) + 1

        # Clamp to audio duration if known
        if analysis is not None:
             dur = getattr(analysis.meta, "duration_sec", 0.0)
             if dur > 0.0 and qe > dur:
                 qe = dur
                 # Recalculate duration_beats if possible, but complex in beat space
                 # if we clamp qe, duration_beats might be wrong relative to grid
                 # but strict clamp is safer for validation.
                 pass

        out.append(
            NoteEvent(
                start_sec=float(qs),
                end_sec=float(qe),
                midi_note=int(n.midi_note),
                pitch_hz=float(n.pitch_hz),
                confidence=float(n.confidence),
                velocity=float(n.velocity),
                dynamic=n.dynamic,
                measure=measure,
                beat=float(beat_in_measure),
                duration_beats=float(duration_beats),
                voice=n.voice,
                staff=n.staff,
            )
        )

    return out
