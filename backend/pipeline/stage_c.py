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
  start_sec, end_sec, midi_note, pitch_hz, confidence, velocity,
  rms_value, dynamic, voice, staff, measure, beat, duration_beats
(no source_stem/source_detector fields)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

# Support both package and top-level imports for models
try:
    from .models import AnalysisData, FramePitch, NoteEvent, AudioType  # type: ignore
except Exception:
    from models import AnalysisData, FramePitch, NoteEvent, AudioType  # type: ignore


def snap_onset(frame_idx: int, onset_strength: List[float], radius: int = 2) -> int:
    """Refine frame_idx to local maximum of onset_strength within radius."""
    if not onset_strength:
        return frame_idx
    n = len(onset_strength)
    if frame_idx < 0 or frame_idx >= n:
        return frame_idx

    lo = max(0, frame_idx - radius)
    hi = min(n - 1, frame_idx + radius)

    best_i = frame_idx
    best_val = onset_strength[frame_idx]

    for i in range(lo, hi + 1):
        if onset_strength[i] > best_val:
            best_val = onset_strength[i]
            best_i = i
    return best_i


def should_split_same_pitch(
    i: int,
    onset_strength: List[float],
    band_energy: List[float],
    thr_onset: float = 0.7,
    thr_bump: float = 0.15,
) -> bool:
    """
    Check if a repeated note split is warranted at index i.
    i: current frame index
    """
    if not onset_strength or i >= len(onset_strength) or i < 2:
        return False

    if onset_strength[i] < thr_onset:
        return False

    if not band_energy:
        # conservative: without per-band energy, avoid over-splitting
        return False

    prev = band_energy[max(0, i - 2) : i]
    if not prev:
        return False

    bump = band_energy[i] - (sum(prev) / len(prev))
    return bump >= thr_bump


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


def _midi_to_hz(midi_note: int) -> float:
    # A4 = 440Hz at MIDI 69
    try:
        return float(440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0)))
    except Exception:
        return 0.0


def _velocity_from_rms(rms_list: List[float], vmin: int = 20, vmax: int = 105) -> int:
    # Velocity mapping from RMS (20–105)
    if not rms_list:
        return 64

    rms = float(np.mean(rms_list))

    # Default range -40dB to -4dB
    min_rms = 10 ** (-40 / 20)
    max_rms = 10 ** (-4 / 20)

    x = (rms - min_rms) / max(max_rms - min_rms, 1e-9)
    x = float(np.clip(x, 0.0, 1.0))
    x = x**0.6
    v = 20 + int(round(x * (105 - 20)))
    return int(np.clip(v, 20, 105))


def _velocity_to_dynamic(v: int) -> str:
    if v < 30:
        return "pp"
    if v < 45:
        return "p"
    if v < 60:
        return "mp"
    if v < 75:
        return "mf"
    if v < 90:
        return "f"
    return "ff"


def _cents_diff_hz(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 1e9
    return abs(1200.0 * math.log2((a + 1e-9) / (b + 1e-9)))


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
            cents = abs(1200.0 * math.log2(other / ref))
            if cents > cents_tolerance:
                return True

    return False


def _decompose_polyphonic_timeline(
    timeline: List[FramePitch],
    pitch_tolerance_cents: float = 50.0,
    max_tracks: int = 5,
    *,
    hangover_frames: int = 3,
    new_track_penalty: float = 80.0,
    crossing_penalty: float = 60.0,
    min_cand_conf: float = 0.05,
) -> List[List[FramePitch]]:
    if not timeline:
        return []

    tracks: List[List[FramePitch]] = [[] for _ in range(max_tracks)]
    track_heads: List[float] = [0.0] * max_tracks
    track_age: List[int] = [10**9] * max_tracks

    LOG2_1200 = 1200.0 / math.log(2.0)

    def cents(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return 1e9
        return abs(LOG2_1200 * math.log((a + 1e-9) / (b + 1e-9)))

    def assign_cost(track_i: int, p: float) -> float:
        head = track_heads[track_i]
        age = track_age[track_i]
        if head > 0.0:
            d = cents(p, head)
            if d > pitch_tolerance_cents:
                return 1e9
            return d + 5.0 * min(age, 10)
        return new_track_penalty + 5.0 * min(age, 10)

    import itertools

    for fp in timeline:
        if getattr(fp, "active_pitches", None):
            candidates = [(p, c) for (p, c) in fp.active_pitches if p > 0.0 and c >= min_cand_conf]
            candidates.sort(key=lambda x: x[1], reverse=True)
        elif fp.pitch_hz > 0.0 and fp.confidence >= min_cand_conf:
            candidates = [(fp.pitch_hz, fp.confidence)]
        else:
            candidates = []

        if candidates:
            candidates = candidates[:max_tracks]

            best_map = None
            best_cost = 1e18

            m = len(candidates)
            track_ids = list(range(max_tracks))

            for perm in itertools.permutations(track_ids, m):
                cost_sum = 0.0
                assigned_pitch = [0.0] * max_tracks

                ok = True
                for j, ti in enumerate(perm):
                    p, conf = candidates[j]
                    cst = assign_cost(ti, p) - 80.0 * float(conf)
                    if cst >= 1e8:
                        ok = False
                        break
                    cost_sum += cst
                    assigned_pitch[ti] = p

                if not ok:
                    continue

                # soft penalty for inversions/crossings (very rough)
                for i in range(max_tracks):
                    for k in range(i + 1, max_tracks):
                        pi = assigned_pitch[i]
                        pk = assigned_pitch[k]
                        if pi > 0.0 and pk > 0.0 and pi < pk:
                            cost_sum += crossing_penalty

                if cost_sum < best_cost:
                    best_cost = cost_sum
                    best_map = perm

            assigned_tracks = set()
            if best_map is not None:
                for j, ti in enumerate(best_map):
                    p_hz, conf = candidates[j]
                    tracks[ti].append(
                        FramePitch(
                            time=fp.time,
                            pitch_hz=p_hz,
                            midi=int(round(69 + 12 * math.log2(p_hz / 440.0))) if p_hz > 0 else None,
                            confidence=float(conf),
                            rms=fp.rms,
                        )
                    )
                    track_heads[ti] = p_hz
                    track_age[ti] = 0
                    assigned_tracks.add(ti)

            for i in range(max_tracks):
                if i not in assigned_tracks:
                    tracks[i].append(FramePitch(time=fp.time, pitch_hz=0.0, midi=None, confidence=0.0, rms=fp.rms))
                    track_age[i] = min(track_age[i] + 1, 10**9)
                    if track_age[i] > hangover_frames:
                        track_heads[i] = 0.0

        else:
            for i in range(max_tracks):
                tracks[i].append(FramePitch(time=fp.time, pitch_hz=0.0, midi=None, confidence=0.0, rms=fp.rms))
                track_age[i] = min(track_age[i] + 1, 10**9)
                if track_age[i] > hangover_frames:
                    track_heads[i] = 0.0

    return tracks


def _segment_monophonic(
    timeline: List[FramePitch],
    conf_thr: float,
    min_note_dur_s: float,
    gap_tolerance_s: float,
    semitone_stability: float = 0.60,
    min_rms: float = 0.01,
    conf_start: float | None = None,
    conf_end: float | None = None,
    seg_cfg: Optional[Dict[str, Any]] = None,
    hop_s: float = 0.01,
) -> List[Tuple[int, int]]:
    """
    Segment monophonic FramePitch into (start_idx, end_idx) segments.
    Updated for hysteresis, gap merge/vibrato, and robustness/glitch tolerance.
    """
    seg_cfg = seg_cfg or {}

    if len(timeline) < 2:
        return []

    min_on = int(seg_cfg.get("min_onset_frames", 3))
    rel = int(seg_cfg.get("release_frames", 2))

    split_semi = float(seg_cfg.get("split_semitone", 0.7))
    split_cents = split_semi * 100.0

    tmf_cfg = seg_cfg.get("time_merge_frames")
    if tmf_cfg is not None:
        time_merge_frames = int(tmf_cfg)
    else:
        time_merge_frames = int(gap_tolerance_s / hop_s)
    time_merge_frames = max(0, time_merge_frames)

    segs: List[Tuple[int, int]] = []

    stable = 0
    silent = 0
    active = False

    current_start = -1
    current_end = -1
    current_pitch_hz = 0.0

    pitch_buffer: List[float] = []
    pitch_buffer_size = int(seg_cfg.get("pitch_ref_window_frames", 7))
    pitch_buffer_size = max(3, min(21, pitch_buffer_size))

    glitch_counter = 0
    MAX_GLITCH_FRAMES = 2

    c_start = conf_start if conf_start is not None else conf_thr
    c_end = conf_end if conf_end is not None else conf_thr

    onset_strength: List[float] = []
    rms_values = [fp.rms for fp in timeline]
    if rms_values:
        onset_strength = [0.0] * len(rms_values)
        for k in range(1, len(rms_values)):
            d = rms_values[k] - rms_values[k - 1]
            if d > 0:
                onset_strength[k] = d
        m = max(onset_strength) if onset_strength else 0
        if m > 0:
            onset_strength = [x / m for x in onset_strength]

    for i, fp in enumerate(timeline):
        if not active:
            is_voiced_frame = (fp.pitch_hz > 0.0 and fp.confidence >= c_start)
        else:
            is_voiced_frame = (fp.pitch_hz > 0.0 and fp.confidence >= c_end)

        if is_voiced_frame:
            is_glitch = False
            if active:
                diff = _cents_diff_hz(fp.pitch_hz, current_pitch_hz)
                if diff > split_cents:
                    is_glitch = True

            is_repeated_split = False
            use_splitter = seg_cfg.get("use_repeated_note_splitter", True)
            if active and not is_glitch and use_splitter:
                if should_split_same_pitch(i, onset_strength, rms_values, thr_onset=0.6, thr_bump=0.1):
                    is_repeated_split = True

            if is_glitch or is_repeated_split:
                if glitch_counter < MAX_GLITCH_FRAMES and is_glitch:
                    current_end = i
                    glitch_counter += 1
                    continue
                else:
                    glitch_counter = 0
                    segs.append((current_start, current_end))

                    use_refinement = seg_cfg.get("use_onset_refinement", True)
                    refined_start = snap_onset(i, onset_strength) if (use_refinement and onset_strength) else i
                    refined_start = min(refined_start, i)
                    refined_start = max(refined_start, current_end + 1)

                    current_start = refined_start
                    current_end = i
                    current_pitch_hz = fp.pitch_hz
                    pitch_buffer = [fp.pitch_hz]
                    stable = min_on
                    active = True
            else:
                glitch_counter = 0

            silent = 0
            if not active:
                stable += 1
                if stable >= min_on:
                    active = True
                    use_refinement = seg_cfg.get("use_onset_refinement", True)

                    start_idx = i - (min_on - 1)
                    refined_start = (
                        snap_onset(start_idx, onset_strength) if (use_refinement and onset_strength) else start_idx
                    )
                    refined_start = min(refined_start, i)
                    refined_start = max(refined_start, current_end + 1)

                    current_start = refined_start
                    current_end = i
                    current_pitch_hz = fp.pitch_hz
                    pitch_buffer = [fp.pitch_hz]
            else:
                current_end = i
                pitch_buffer.append(fp.pitch_hz)
                if len(pitch_buffer) > pitch_buffer_size:
                    pitch_buffer.pop(0)
                current_pitch_hz = float(np.median(pitch_buffer))

        else:
            stable = 0
            if active:
                silent += 1
                if silent >= rel:
                    segs.append((current_start, current_end))
                    active = False
                    current_start = -1
                    current_end = -1

    if active and current_start != -1:
        segs.append((current_start, current_end))

    merged_segs: List[Tuple[int, int]] = []
    if segs:
        curr_s, curr_e = segs[0]

        def get_seg_pitch(s: int, e: int) -> float:
            p = [timeline[x].pitch_hz for x in range(s, e + 1) if timeline[x].pitch_hz > 0]
            if not p:
                return 0.0
            return float(np.median(p))

        curr_p = get_seg_pitch(curr_s, curr_e)

        for i in range(1, len(segs)):
            next_s, next_e = segs[i]
            next_p = get_seg_pitch(next_s, next_e)
            gap = next_s - curr_e - 1

            if gap <= time_merge_frames and _cents_diff_hz(curr_p, next_p) <= split_cents:
                curr_e = next_e
            else:
                dur = (curr_e - curr_s + 1) * hop_s
                if dur >= min_note_dur_s:
                    merged_segs.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e
                curr_p = next_p

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
        stay_voiced = voiced_cost[i - 1] - stay_bonus
        switch_to_voiced = silence_cost[i - 1] + transition_penalty
        if stay_voiced <= switch_to_voiced:
            voiced_cost[i] = stay_voiced
            backpointer[i, 1] = 1
        else:
            voiced_cost[i] = switch_to_voiced
            backpointer[i, 1] = 0
        voiced_cost[i] -= voiced_score[i]

        stay_silence = silence_cost[i - 1] - stay_bonus
        switch_to_silence = voiced_cost[i - 1] + transition_penalty
        if stay_silence <= switch_to_silence:
            silence_cost[i] = stay_silence
            backpointer[i, 0] = 0
        else:
            silence_cost[i] = switch_to_silence
            backpointer[i, 0] = 1
        silence_cost[i] -= silence_score[i]

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


def _sanitize_notes(notes: List[NoteEvent]) -> List[NoteEvent]:
    """
    Ensure pitch_hz is valid. If pitch_hz missing/<=0 but midi_note exists, derive pitch_hz.
    Then drop obviously invalid notes.
    """
    clean: List[NoteEvent] = []
    for n in notes:
        if n is None:
            continue
        if n.end_sec <= n.start_sec:
            continue

        # E2E hygiene: pitch_hz fallback from midi_note
        phz = getattr(n, "pitch_hz", None)
        if phz is None or float(phz) <= 0.0:
            mn = getattr(n, "midi_note", None)
            if mn is not None:
                try:
                    n.pitch_hz = _midi_to_hz(int(mn))
                except Exception:
                    n.pitch_hz = 0.0

        if n.pitch_hz is None or float(n.pitch_hz) <= 0.0:
            continue

        clean.append(n)

    clean.sort(key=lambda x: (x.start_sec, x.end_sec, x.pitch_hz))
    return clean


def _dedupe_overlapping_notes(notes: List[NoteEvent], overlap_thr: float = 0.85) -> List[NoteEvent]:
    if not notes:
        return notes
    notes = sorted(
        notes,
        key=lambda n: (n.voice or 0, n.midi_note or 0, n.start_sec, n.end_sec, -float(getattr(n, "confidence", 0.0))),
    )
    out: List[NoteEvent] = []
    for n in notes:
        if not out:
            out.append(n)
            continue
        p = out[-1]
        if (p.voice == n.voice) and (p.midi_note == n.midi_note):
            a0, a1 = p.start_sec, p.end_sec
            b0, b1 = n.start_sec, n.end_sec
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0) + 1e-9
            if inter / union >= overlap_thr:
                if float(getattr(n, "confidence", 0.0)) > float(getattr(p, "confidence", 0.0)):
                    out[-1] = n
                continue
        out.append(n)
    return out


def _snap_chord_starts(notes: List[NoteEvent], tol_ms: float = 25.0) -> List[NoteEvent]:
    if not notes:
        return notes
    tol = tol_ms / 1000.0
    notes = sorted(notes, key=lambda n: n.start_sec)
    i = 0
    out: List[NoteEvent] = []
    while i < len(notes):
        j = i + 1
        group = [notes[i]]
        while j < len(notes) and abs(notes[j].start_sec - notes[i].start_sec) <= tol:
            group.append(notes[j])
            j += 1

        if len(group) >= 2:
            s0 = min(n.start_sec for n in group)
            for n in group:
                n.start_sec = s0

        out.extend(group)
        i = j
    return out


def _snap_chord_starts_with_count(notes: List[NoteEvent], tol_ms: float = 25.0) -> Tuple[List[NoteEvent], int]:
    """Chord snap + count of moved notes."""
    if not notes:
        return notes, 0
    tol = tol_ms / 1000.0
    notes = sorted(notes, key=lambda n: (float(n.start_sec), int(getattr(n, "midi_note", 0))))
    moved = 0
    i = 0
    out: List[NoteEvent] = []
    while i < len(notes):
        j = i + 1
        group = [notes[i]]
        while j < len(notes) and abs(float(notes[j].start_sec) - float(notes[i].start_sec)) <= tol:
            group.append(notes[j])
            j += 1
        if len(group) >= 2:
            s0 = min(float(n.start_sec) for n in group)
            for n in group:
                if abs(float(n.start_sec) - s0) > 1e-9:
                    moved += 1
                n.start_sec = s0
        out.extend(group)
        i = j
    return out, moved


def _merge_same_pitch_gaps(
    notes: List[NoteEvent],
    max_gap_ms: float = 60.0,
    *,
    max_merge_dur_s: float = 8.0,
    clamp_factor: float = 4.0,
) -> Tuple[List[NoteEvent], int]:
    """
    Merge same-pitch notes separated by tiny gaps to reduce fragmentation.
    Guardrails:
      - same midi_note
      - same voice (when present)
      - gap <= max_gap_ms
      - prevents absurd merges (max_merge_dur_s, clamp_factor)
    """
    if not notes:
        return notes, 0

    gap_s = float(max_gap_ms) / 1000.0
    notes_sorted = sorted(
        notes,
        key=lambda n: (
            int(getattr(n, "voice", -1) if getattr(n, "voice", None) is not None else -1),
            int(getattr(n, "midi_note", 0)),
            float(n.start_sec),
            float(n.end_sec),
        ),
    )

    merged = 0
    out: List[NoteEvent] = []
    i = 0
    while i < len(notes_sorted):
        cur = notes_sorted[i]
        j = i + 1
        while j < len(notes_sorted):
            nxt = notes_sorted[j]
            if int(getattr(nxt, "midi_note", -999)) != int(getattr(cur, "midi_note", -999)):
                break
            if getattr(cur, "voice", None) is not None or getattr(nxt, "voice", None) is not None:
                if getattr(cur, "voice", None) != getattr(nxt, "voice", None):
                    break

            gap = float(nxt.start_sec) - float(cur.end_sec)
            if gap < -1e-6:
                break
            if gap > gap_s:
                break

            new_end = max(float(cur.end_sec), float(nxt.end_sec))
            new_dur = new_end - float(cur.start_sec)
            cur_dur = float(cur.end_sec) - float(cur.start_sec)
            nxt_dur = float(nxt.end_sec) - float(nxt.start_sec)
            ref = max(cur_dur, nxt_dur, 1e-6)

            if new_dur > float(max_merge_dur_s):
                break
            if new_dur > float(clamp_factor) * ref:
                break

            cur.end_sec = new_end
            try:
                cur.confidence = float(max(float(getattr(cur, "confidence", 0.0) or 0.0), float(getattr(nxt, "confidence", 0.0) or 0.0)))
            except Exception:
                pass
            if getattr(cur, "velocity", None) is not None and getattr(nxt, "velocity", None) is not None:
                try:
                    cur.velocity = float(max(float(cur.velocity), float(nxt.velocity)))
                except Exception:
                    pass

            merged += 1
            j += 1

        out.append(cur)
        i = j

    out = sorted(out, key=lambda n: (float(n.start_sec), float(n.end_sec), int(getattr(n, "midi_note", 0))))
    return out, merged


def _timeline_score(timeline: list) -> tuple[float, float]:
    """Return (voiced_ratio, mean_confidence) for a FramePitch timeline."""
    if not timeline:
        return 0.0, 0.0
    voiced = 0
    conf_sum = 0.0
    conf_n = 0
    for fp in timeline:
        try:
            hz = float(getattr(fp, "pitch_hz", 0.0) or 0.0)
            if hz > 0.0:
                voiced += 1
                c = getattr(fp, "confidence", None)
                if c is not None:
                    conf_sum += float(c)
                    conf_n += 1
        except Exception:
            continue
    total = max(1, len(timeline))
    voiced_ratio = voiced / total
    mean_conf = (conf_sum / conf_n) if conf_n else 0.0
    return voiced_ratio, mean_conf


def _select_best_stem_timeline(stem_timelines: dict, config: Any) -> tuple[str, list]:
    """Pick the best stem timeline based on voiced_ratio/conf within a prefer order."""
    if not stem_timelines:
        return "timeline", []

    prefer_order = _get(config, "stem_selection.prefer_order", None)
    if not prefer_order:
        prefer_order = ["vocals", "other", "melody_masked", "mix"]

    mix_margin = float(_get(config, "stem_selection.mix_margin", 0.02) or 0.0)

    scores = {}
    for stem in prefer_order:
        tl = stem_timelines.get(stem)
        if tl is None:
            continue
        vr, mc = _timeline_score(tl)
        score = vr * (0.5 + mc)
        scores[stem] = (score, vr, mc)

    if not scores:
        stem, tl = next(iter(stem_timelines.items()))
        return stem, tl

    best_stem = max(scores.items(), key=lambda kv: kv[1][0])[0]
    best_score = scores[best_stem][0]

    if "mix" in scores and best_stem != "mix":
        mix_score = scores["mix"][0]
        if mix_score >= best_score - mix_margin:
            best_stem = "mix"

    return best_stem, stem_timelines[best_stem]


def apply_theory(analysis_data: AnalysisData, config: Any = None) -> List[NoteEvent]:
    """
    Convert FramePitch timelines into NoteEvent list.

    Quantization gate:
      If config.stage_c.quantize["enabled"] is False, returns RAW notes (no grid snap).
      Default assumed enabled when config missing.
    """
    # Legacy call signature support: apply_theory(timeline, analysis_data)
    if isinstance(analysis_data, list) and isinstance(config, AnalysisData):
        legacy_timeline = analysis_data or getattr(config, "timeline", [])
        analysis_data = config
        if not analysis_data.stem_timelines:
            analysis_data.stem_timelines = {"mix": legacy_timeline}
    elif not isinstance(analysis_data, AnalysisData):
        return []

    quantize_enabled = bool(_get(config, "stage_c.quantize.enabled", True))

    # Short-circuit if precalculated notes exist (E2E path)
    if analysis_data.precalculated_notes is not None:
        clean = _sanitize_notes(list(analysis_data.precalculated_notes or []))

        # Always store raw
        analysis_data.notes_before_quantization = deepcopy(clean)

        if not quantize_enabled:
            analysis_data.notes = list(clean)
            if hasattr(analysis_data, "diagnostics"):
                analysis_data.diagnostics["stage_c_mode"] = "precalculated_raw"
                analysis_data.diagnostics["stage_c_quantize_enabled"] = False
            return list(clean)

        quantized = quantize_notes(clean, analysis_data=analysis_data)
        analysis_data.notes = quantized

        if hasattr(analysis_data, "diagnostics"):
            analysis_data.diagnostics["stage_c_mode"] = "precalculated"
            analysis_data.diagnostics["stage_c_quantize_enabled"] = True
        return quantized

    meta_instr = getattr(analysis_data.meta, "instrument", None)
    config_instr = _get(config, "stage_b.instrument", "piano_61key")
    instrument_name = meta_instr if meta_instr else config_instr

    profile_special: Dict[str, Any] = {}
    apply_profile = _get(config, "stage_c.apply_instrument_profile", True)

    if apply_profile and config and hasattr(config, "get_profile"):
        profile = config.get_profile(str(instrument_name))
        if profile:
            profile_special = dict(getattr(profile, "special", {}) or {})

    def resolve_val(key, default):
        special_key = f"stage_c_{key}"
        if special_key in profile_special:
            return profile_special[special_key]
        nested_c = profile_special.get("stage_c", {})
        if isinstance(nested_c, dict) and key in nested_c:
            return nested_c[key]
        return _get(config, f"stage_c.{key}", default)

    seg_cfg = dict(resolve_val("segmentation_method", {}) or {})
    seg_cfg["use_onset_refinement"] = _get(config, "stage_c.use_onset_refinement", True)
    seg_cfg["use_repeated_note_splitter"] = _get(config, "stage_c.use_repeated_note_splitter", True)

    if "stage_c_pitch_ref_window_frames" in profile_special:
        seg_cfg["pitch_ref_window_frames"] = int(profile_special["stage_c_pitch_ref_window_frames"])

    stem_timelines: Dict[str, List[FramePitch]] = analysis_data.stem_timelines or {}
    if not stem_timelines:
        analysis_data.notes_before_quantization = []
        analysis_data.notes = []
        return []

    stem_name, primary_timeline = _select_best_stem_timeline(stem_timelines, config)

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

    min_db = float(_get(config, "stage_c.velocity_map.min_db", -40.0))
    min_rms = 10 ** (min_db / 20.0)

    nf = 0.0
    try:
        nf = float(getattr(getattr(analysis_data, "meta", None), "noise_floor_rms", 0.0) or 0.0)
    except Exception:
        nf = 0.0

    if nf > 0.0:
        margin = float(_get(config, "stage_c.velocity_map.noise_floor_db_margin", 6.0))
        min_rms = max(min_rms, nf * (10 ** (margin / 20.0)))

    timelines_to_process: List[Tuple[str, List[FramePitch]]] = [(stem_name, primary_timeline)]
    audio_type = getattr(analysis_data.meta, "audio_type", None)
    allow_secondary = audio_type in (
        getattr(AudioType, "POLYPHONIC", None),
        getattr(AudioType, "POLYPHONIC_DOMINANT", None),
    )

    if allow_secondary and len(stem_timelines) > 1:
        other_keys = sorted([k for k in stem_timelines.keys() if k != stem_name])
        for other_name in other_keys:
            timelines_to_process.append((other_name, stem_timelines[other_name]))

    notes: List[NoteEvent] = []

    seg_method = str(seg_cfg.get("method") or _get(config, "stage_c.segmentation_method", "threshold")).lower()
    smoothing_enabled = bool(seg_cfg.get("use_state_smoothing", _get(config, "stage_c.use_state_smoothing", False)))
    if seg_method == "hmm":
        smoothing_enabled = True

    transition_penalty = float(seg_cfg.get("transition_penalty", 0.8))
    stay_bonus = float(seg_cfg.get("stay_bonus", 0.05))
    silence_bias = float(seg_cfg.get("silence_bias", 0.1))
    energy_weight = float(seg_cfg.get("energy_weight", 0.35))
    conf_weight = max(0.0, min(1.0, 1.0 - energy_weight))

    poly_filter_mode = _get(config, "stage_c.polyphony_filter.mode", "skyline_top_voice")
    max_alt_voices = int(_get(config, "stage_b.voice_tracking.max_alt_voices", 4))
    max_tracks = 1 + max_alt_voices
    poly_pitch_tolerance = float(_get(config, "stage_c.pitch_tolerance_cents", 50.0))

    poly_used = False

    for vidx, (_vname, timeline) in enumerate(timelines_to_process):
        try:
            if not timeline or len(timeline) < 2:
                continue

            if poly_filter_mode == "skyline_top_voice":
                new_tl: List[FramePitch] = []
                skyline_conf_thr = max(0.05, min(conf_thr * 0.5, 0.2))
                for fp in timeline:
                    ap = getattr(fp, "active_pitches", []) or []
                    cand = [(p, c) for (p, c) in ap if p > 0.0 and c >= skyline_conf_thr]
                    if cand:
                        cand.sort(key=lambda x: x[1], reverse=True)
                        top_conf = cand[0][1]
                        contestants = [x for x in cand if x[1] >= top_conf * 0.9]
                        best_cand = contestants[0]
                        prev_pitch = new_tl[-1].pitch_hz if new_tl else 0.0

                        if len(contestants) > 1:
                            scored_candidates = []
                            for p, c in contestants:
                                score = c
                                if 80.0 <= p <= 1400.0:
                                    score += 0.05
                                if prev_pitch > 0.0:
                                    cents_diff = abs(1200.0 * math.log2(p / prev_pitch))
                                    if cents_diff < 100:
                                        score += 0.1
                                    elif cents_diff < 1200:
                                        score += 0.05
                                    else:
                                        score -= 0.05
                                scored_candidates.append(((p, c), score))
                            best_cand = max(scored_candidates, key=lambda x: x[1])[0]

                        p_best, c_best = best_cand
                        midi_new = int(round(69 + 12 * math.log2(p_best / 440.0)))
                        fp2 = FramePitch(
                            time=fp.time,
                            pitch_hz=p_best,
                            midi=midi_new,
                            confidence=c_best,
                            rms=fp.rms,
                            active_pitches=getattr(fp, "active_pitches", None),
                        )
                        new_tl.append(fp2)
                    else:
                        new_tl.append(fp)
                timeline = new_tl

            poly_frames = [fp for fp in timeline if getattr(fp, "active_pitches", []) and len(fp.active_pitches) > 1]
            enable_polyphony = (len(poly_frames) > 0) and (poly_filter_mode != "skyline_top_voice")
            if enable_polyphony:
                poly_used = True

            if enable_polyphony:
                voice_timelines = _decompose_polyphonic_timeline(
                    timeline,
                    pitch_tolerance_cents=poly_pitch_tolerance,
                    max_tracks=max_tracks,
                )

                if poly_filter_mode == "decomposed_melody" and len(voice_timelines) > 1:
                    best_idx = 0
                    best_score = -1.0
                    for i, tl in enumerate(voice_timelines):
                        score = 0.0
                        for fp in tl:
                            if fp.pitch_hz > 0:
                                s = fp.confidence
                                if 80.0 <= fp.pitch_hz <= 1400.0:
                                    s *= 1.2
                                score += s
                        if score > best_score:
                            best_score = score
                            best_idx = i
                    voice_timelines = [voice_timelines[best_idx]]
            else:
                voice_timelines = [timeline]

            voice_conf_gate = conf_thr
            voice_min_dur_s = min_note_dur_s
            has_distinct_poly = _has_distinct_poly_layers(timeline)

            if poly_frames or enable_polyphony:
                voice_conf_gate = poly_conf if vidx == 0 else accomp_conf
                try:
                    if min_note_dur_ms_poly is not None:
                        voice_min_dur_s = max(voice_min_dur_s, float(min_note_dur_ms_poly) / 1000.0)
                except Exception:
                    pass

                if vidx > 0 and not has_distinct_poly:
                    voice_conf_gate = max(voice_conf_gate, accomp_conf)
            elif vidx > 0:
                voice_conf_gate = max(voice_conf_gate, accomp_conf)

            hop_s = _estimate_hop_seconds(timeline)

            for sub_idx, sub_tl in enumerate(voice_timelines):
                if not any(fp.pitch_hz > 0 for fp in sub_tl):
                    continue

                use_viterbi = smoothing_enabled and seg_method in ("viterbi", "hmm")
                if poly_filter_mode == "skyline_top_voice":
                    use_viterbi = False

                if use_viterbi:
                    mask = _viterbi_voicing_mask(
                        sub_tl,
                        conf_weight=conf_weight,
                        energy_weight=energy_weight,
                        transition_penalty=transition_penalty,
                        stay_bonus=stay_bonus,
                        silence_bias=silence_bias,
                    )
                    segs = _segments_from_mask(
                        timeline=sub_tl,
                        mask=mask,
                        hop_s=hop_s,
                        min_note_dur_s=voice_min_dur_s,
                        min_conf=voice_conf_gate,
                        min_rms=min_rms,
                    )
                else:
                    seg_cfg_local = dict(seg_cfg)
                    if enable_polyphony and sub_idx > 0:
                        seg_cfg_local["use_repeated_note_splitter"] = False

                    segs = _segment_monophonic(
                        timeline=sub_tl,
                        conf_thr=voice_conf_gate,
                        min_note_dur_s=voice_min_dur_s,
                        gap_tolerance_s=gap_tolerance_s,
                        min_rms=min_rms,
                        conf_start=max(start_conf, voice_conf_gate),
                        conf_end=min(max(end_conf, 0.0), max(start_conf, voice_conf_gate)),
                        seg_cfg=seg_cfg_local,
                        hop_s=hop_s,
                    )

                for (s, e) in segs:
                    mids = [
                        sub_tl[i].midi
                        for i in range(s, e + 1)
                        if sub_tl[i].midi is not None and sub_tl[i].midi > 0
                    ]
                    hzs = [sub_tl[i].pitch_hz for i in range(s, e + 1) if sub_tl[i].pitch_hz > 0]
                    confs = [sub_tl[i].confidence for i in range(s, e + 1)]
                    rmss = [sub_tl[i].rms for i in range(s, e + 1)]

                    if not mids:
                        continue
                    if rmss and float(np.mean(rmss)) < min_rms:
                        continue

                    midi_note = int(round(float(np.median(mids))))
                    pitch_hz = float(np.median(hzs)) if hzs else 0.0
                    confidence = float(np.mean(confs)) if confs else 0.0

                    midi_vel = _velocity_from_rms(rmss)
                    velocity_norm = float(midi_vel) / 127.0
                    rms_val = float(np.mean(rmss)) if rmss else 0.0
                    dynamic_label = _velocity_to_dynamic(midi_vel)

                    start_sec = float(sub_tl[s].time)
                    end_sec = float(sub_tl[e].time + hop_s)
                    if end_sec <= start_sec:
                        end_sec = start_sec + hop_s

                    voice_id = (vidx * 16) + (sub_idx + 1)

                    notes.append(
                        NoteEvent(
                            start_sec=start_sec,
                            end_sec=end_sec,
                            midi_note=midi_note,
                            pitch_hz=pitch_hz,
                            confidence=confidence,
                            velocity=velocity_norm,
                            rms_value=rms_val,
                            dynamic=dynamic_label,
                            voice=voice_id,
                        )
                    )

        except Exception as e:
            logger.warning(f"Onset/Theory error for stem {_vname}: {e}")
            if hasattr(analysis_data, 'diagnostics'):
                analysis_data.diagnostics.setdefault("onset_errors", {})[_vname] = repr(e)
    # Bass backtracking (optional)
    bass_backtrack_ms = float(
        profile_special.get("stage_c_backtrack_ms", 0.0) or profile_special.get("bass_backtrack_ms", 0.0)
    )
    if bass_backtrack_ms > 0.0 and notes:
        backtrack_sec = bass_backtrack_ms / 1000.0
        min_dur_s = float(min_note_dur_s)

        notes.sort(key=lambda n: n.start_sec)

        by_voice: Dict[Any, List[NoteEvent]] = {}
        for n in notes:
            by_voice.setdefault(n.voice, []).append(n)

        for _v, v_notes in by_voice.items():
            for i, n in enumerate(v_notes):
                prev_end = v_notes[i - 1].end_sec if i > 0 else 0.0

                new_start = max(0.0, float(n.start_sec) - backtrack_sec)
                new_start = max(new_start, float(prev_end))

                latest_ok_start = float(n.end_sec) - min_dur_s
                new_start = min(new_start, latest_ok_start)

                if new_start < float(n.start_sec):
                    n.start_sec = float(new_start)

    # ---------------------------------------------------------------------
    # Stage C Post-processing (poly stability)
    # ---------------------------------------------------------------------
    snap_ms = float(_get(config, "stage_c.chord_onset_snap_ms", 25.0))
    merge_gap_ms = float(
        _get(config, "stage_c.post_merge.max_gap_ms", None)
        or _get(config, "stage_c.gap_filling.max_gap_ms", None)
        or 60.0
    )
    merge_gap_ms = float(min(max(merge_gap_ms, 0.0), 200.0))

    gap_merges = 0
    chord_snaps = 0

    do_poly_post = bool(poly_used) or len(notes) >= 2
    if do_poly_post:
        notes = _dedupe_overlapping_notes(notes)

        if merge_gap_ms > 0:
            notes, gap_merges = _merge_same_pitch_gaps(
                notes,
                max_gap_ms=merge_gap_ms,
                max_merge_dur_s=float(_get(config, "stage_c.post_merge.max_merge_dur_s", 8.0) or 8.0),
                clamp_factor=float(_get(config, "stage_c.post_merge.clamp_factor", 4.0) or 4.0),
            )

        if snap_ms > 0:
            notes, chord_snaps = _snap_chord_starts_with_count(notes, tol_ms=snap_ms)

    raw_notes = _sanitize_notes(notes)
    analysis_data.notes_before_quantization = list(raw_notes)

    if hasattr(analysis_data, "diagnostics"):
        analysis_data.diagnostics["stage_c"] = {
            "segmentation_method": seg_method,
            "timelines_processed": len(timelines_to_process),
            "note_count_raw": len(raw_notes),
            "selected_stem": stem_name,
            "quantize_enabled": quantize_enabled,
        }
        analysis_data.diagnostics["stage_c_post"] = {
            "gap_merges": int(locals().get("gap_merges", 0) or 0),
            "chord_snaps": int(locals().get("chord_snaps", 0) or 0),
            "snap_tol_ms": float(locals().get("snap_ms", 0.0) or 0.0),
            "merge_gap_ms": float(locals().get("merge_gap_ms", 0.0) or 0.0),
        }

    if not quantize_enabled:
        analysis_data.notes = list(raw_notes)
        return list(raw_notes)

    quantized_notes = quantize_notes(raw_notes, analysis_data=analysis_data)
    analysis_data.notes = quantized_notes
    return quantized_notes


def _sec_to_beat_index(t: float, beat_times: list[float]) -> float:
    n = len(beat_times)
    if n < 2:
        return 0.0

    bt = np.asarray(beat_times, dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)

    if bt[0] <= t <= bt[-1]:
        return float(np.interp(t, bt, idx))

    dt0 = float(bt[1] - bt[0])
    dt1 = float(bt[-1] - bt[-2])

    if dt0 <= 1e-6 or dt1 <= 1e-6:
        return 0.0 if t < bt[0] else float(n - 1)

    if t < bt[0]:
        return float((t - bt[0]) / dt0)
    else:
        return float((n - 1) + (t - bt[-1]) / dt1)


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
    """
    if not notes:
        return []

    analysis: Optional[AnalysisData] = analysis_data

    beat_times: List[float] = []
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
        if effective_bpm < 75:
            denom = min(denom, 8)
        elif effective_bpm > 140:
            denom = max(denom, 32)
    else:
        denom = 8

    step_beats = 4.0 / float(max(1, denom))
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
        # ensure pitch_hz valid WITHOUT mutating input notes
        phz = getattr(n, "pitch_hz", None)
        if phz is None or float(phz) <= 0.0:
            phz = _midi_to_hz(int(getattr(n, "midi_note", 0) or 0))

        if use_beat_times:
            bs = _sec_to_beat_index(float(n.start_sec), beat_times)
            be = _sec_to_beat_index(float(n.end_sec), beat_times)

            qbs = round(bs / step_beats) * step_beats
            qbe = round(be / step_beats) * step_beats

            if qbe <= qbs:
                qbe = qbs + max(int(min_steps), 1) * step_beats

            qs = _beat_index_to_sec(qbs, beat_times)
            qe = _beat_index_to_sec(qbe, beat_times)

            qs = max(0.0, qs)
            qe = max(0.0, qe)
            if qe <= qs:
                qe = qs + 0.05

            beat_idx = qbs
            duration_beats = qbe - qbs

        else:
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

        if analysis is not None:
            dur = getattr(analysis.meta, "duration_sec", 0.0)
            if dur > 0.0 and qe > dur:
                qe = dur

        out.append(
            NoteEvent(
                start_sec=float(qs),
                end_sec=float(qe),
                midi_note=int(n.midi_note),
                pitch_hz=float(phz),
                confidence=float(getattr(n, "confidence", 0.0) or 0.0),
                velocity=float(getattr(n, "velocity", 0.0) or 0.0),
                dynamic=getattr(n, "dynamic", None),
                measure=measure,
                beat=float(beat_in_measure),
                duration_beats=float(duration_beats),
                voice=getattr(n, "voice", 1),
                staff=getattr(n, "staff", "treble"),
            )
        )

    return out
