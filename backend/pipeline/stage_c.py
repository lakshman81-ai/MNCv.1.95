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

import numpy as np
import math

# Support both package and top‑level imports for models
try:
    from .models import AnalysisData, FramePitch, NoteEvent, AudioType  # type: ignore
except Exception:
    from models import AnalysisData, FramePitch, NoteEvent, AudioType  # type: ignore

from copy import deepcopy

def snap_onset(frame_idx: int, onset_strength: List[float], radius: int = 2) -> int:
    """Refine frame_idx to local maximum of onset_strength within radius."""
    if not onset_strength:
        return frame_idx
    lo = max(0, frame_idx - radius)
    hi = min(len(onset_strength) - 1, frame_idx + radius)
    # Find index of max value in range [lo, hi]
    # Note: onset_strength is list[float]
    best_i = frame_idx
    best_val = -1.0
    for i in range(lo, hi + 1):
        if onset_strength[i] > best_val:
            best_val = onset_strength[i]
            best_i = i
    return best_i

def should_split_same_pitch(i: int, onset_strength: List[float], band_energy: List[float], thr_onset: float = 0.7, thr_bump: float = 0.15) -> bool:
    """
    Check if a repeated note split is warranted at index i.
    i: current frame index
    """
    if not onset_strength or i >= len(onset_strength) or i < 2:
        return False

    if onset_strength[i] < thr_onset:
        return False

    # Check local energy bump in specific band
    # band_energy should be energy of the specific pitch bin
    # We might not have per-bin energy here easily unless we computed it.
    # Fallback: just use onset strength peak + pitch stability
    # But prompt says: "short-term pitch-band energy bump exists"
    # If we don't have band_energy, we rely on onset_strength.

    if not band_energy:
         # Without band energy, we should be conservative.
         # Only split if onset strength is very high?
         return False # False to avoid over-splitting on L0/L1 without per-band data

    prev = band_energy[max(0, i - 2):i]
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
    hangover_frames: int = 3,          # keep head through short dropouts
    new_track_penalty: float = 80.0,   # discourage spawning new tracks
    crossing_penalty: float = 60.0,    # discourage voice crossing
    min_cand_conf: float = 0.05,       # ignore very weak candidates
) -> List[List[FramePitch]]:
    if not timeline:
        return []

    tracks: List[List[FramePitch]] = [[] for _ in range(max_tracks)]
    track_heads: List[float] = [0.0] * max_tracks
    track_age: List[int] = [10**9] * max_tracks  # frames since last voiced

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
            # small penalty grows with age (prefer continuity)
            return d + 5.0 * min(age, 10)
        # empty track: discourage spawning unless needed
        return new_track_penalty + 5.0 * min(age, 10)

    for fp in timeline:
        # candidates (top by confidence, clipped to max_tracks)
        if getattr(fp, "active_pitches", None):
            candidates = [(p, c) for (p, c) in fp.active_pitches if p > 0.0 and c >= min_cand_conf]
            candidates.sort(key=lambda x: x[1], reverse=True)
        elif fp.pitch_hz > 0.0 and fp.confidence >= min_cand_conf:
            candidates = [(fp.pitch_hz, fp.confidence)]
        else:
            candidates = []

        if candidates:
            candidates = candidates[:max_tracks]

            # brute-force best assignment (small max_tracks => fast)
            best_map = None
            best_cost = 1e18

            m = len(candidates)
            track_ids = list(range(max_tracks))

            # try subsets of tracks of size m (simple: try all tracks permutations and take first m)
            # because max_tracks is small, brute-forcing permutations is okay
            import itertools
            for perm in itertools.permutations(track_ids, m):
                cost_sum = 0.0
                assigned_pitch = [0.0] * max_tracks

                ok = True
                for j, ti in enumerate(perm):
                    p, conf = candidates[j]
                    cst = assign_cost(ti, p) - 80.0 * float(conf)  # confidence bonus
                    if cst >= 1e8:
                        ok = False
                        break
                    cost_sum += cst
                    assigned_pitch[ti] = p

                if not ok:
                    continue

                # crossing penalty: enforce track index ~ pitch order (track0 highest, trackN lowest)
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
                    tracks[ti].append(FramePitch(
                        time=fp.time,
                        pitch_hz=p_hz,
                        midi=int(round(69 + 12 * math.log2(p_hz / 440.0))) if p_hz > 0 else None,
                        confidence=float(conf),
                        rms=fp.rms
                    ))
                    track_heads[ti] = p_hz
                    track_age[ti] = 0
                    assigned_tracks.add(ti)

            # fill others with silence; DO NOT instantly reset head (hangover)
            for i in range(max_tracks):
                if i not in assigned_tracks:
                    tracks[i].append(FramePitch(time=fp.time, pitch_hz=0.0, midi=None, confidence=0.0, rms=fp.rms))
                    track_age[i] = min(track_age[i] + 1, 10**9)
                    if track_age[i] > hangover_frames:
                        track_heads[i] = 0.0

        else:
            # no candidates: keep heads for hangover_frames instead of wiping immediately
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
    seg_cfg: Dict[str, Any] = {},
    hop_s: float = 0.01,
) -> List[Tuple[int, int]]:
    """
    Segment monophonic FramePitch into (start_idx, end_idx) segments.
    Updated for C1 (hysteresis), C2 (gap merge/vibrato), and robustness/glitch tolerance.
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
    pitch_buffer_size = int(seg_cfg.get("pitch_ref_window_frames", 7))
    pitch_buffer_size = max(3, min(21, pitch_buffer_size))

    # Glitch tolerance
    glitch_counter = 0
    MAX_GLITCH_FRAMES = 2

    # P2: Apply confidence hysteresis
    c_start = conf_start if conf_start is not None else conf_thr
    c_end = conf_end if conf_end is not None else conf_thr

    # Step 5: Onset Refinement
    # If we have onset_strength in timeline (FramePitch.rms used as proxy or new field?)
    # FramePitch doesn't have onset_strength.
    # We can try to infer it from RMS changes or if we added it (we added it to debug CSV but not FramePitch struct yet).
    # Let's assume we use rms delta as weak proxy if no explicit data.
    onset_strength = []
    # Compute simple onset strength from RMS if not present
    # Or check if we can add it to FramePitch (we didn't modify models.py yet).
    # We can assume FramePitch has no extra field, so we calculate locally.
    rms_values = [fp.rms for fp in timeline]
    if rms_values:
        # Simple Flux: diff(rms) > 0
        onset_strength = [0.0] * len(rms_values)
        for k in range(1, len(rms_values)):
            d = rms_values[k] - rms_values[k-1]
            if d > 0:
                onset_strength[k] = d
        # Normalize
        m = max(onset_strength) if onset_strength else 0
        if m > 0:
            onset_strength = [x/m for x in onset_strength]

    # Step 5.1: Snap onset
    # We apply this when detecting 'active' transition.

    for i, fp in enumerate(timeline):
        # P2 Hysteresis logic
        if not active:
             is_voiced_frame = (fp.pitch_hz > 0.0 and fp.confidence >= c_start)
        else:
             is_voiced_frame = (fp.pitch_hz > 0.0 and fp.confidence >= c_end)

        if is_voiced_frame:
            # Check for glitch if active
            is_glitch = False
            if active:
                diff = _cents_diff_hz(fp.pitch_hz, current_pitch_hz)
                if diff > split_cents:
                     is_glitch = True

            # Step 5.2: Check for repeated note split (same pitch, new onset)
            is_repeated_split = False

            # Feature Flag: Repeated Note Splitter
            use_splitter = seg_cfg.get("use_repeated_note_splitter", True)
            if active and not is_glitch and use_splitter:
                # Use should_split_same_pitch logic
                # Need band energy -> use fp.rms as proxy for now
                if should_split_same_pitch(i, onset_strength, rms_values, thr_onset=0.6, thr_bump=0.1):
                     is_repeated_split = True

            if is_glitch or is_repeated_split:
                if glitch_counter < MAX_GLITCH_FRAMES and is_glitch: # Only absorb pitch glitches
                    # Absorb glitch: extend current note logically but don't update pitch ref
                    # Treat as part of note
                    current_end = i
                    glitch_counter += 1
                    continue
                else:
                    # Confirmed split (pitch jump or repeated note)
                    glitch_counter = 0

                    # Close current
                    segs.append((current_start, current_end))

                    # Start new
                    active = True

                    # Feature Flag: Onset Refinement
                    use_refinement = seg_cfg.get("use_onset_refinement", True)

                    # Snap onset?
                    if use_refinement:
                        refined_start = snap_onset(i, onset_strength) if onset_strength else i
                    else:
                        refined_start = i

                    # Ensure refined start is causally valid (>= current_end + 1?)
                    refined_start = max(refined_start, current_end + 1)

                    current_start = refined_start
                    current_end = i
                    current_pitch_hz = fp.pitch_hz
                    pitch_buffer = [fp.pitch_hz]
                    stable = min_on # Assume stability resets? Or we trust this split?
            else:
                glitch_counter = 0

            silent = 0
            if not active:
                stable += 1
                if stable >= min_on:
                    active = True

                    # Feature Flag: Onset Refinement
                    use_refinement = seg_cfg.get("use_onset_refinement", True)

                    # Snap start
                    start_idx = i - (min_on - 1)
                    if use_refinement:
                        refined_start = snap_onset(start_idx, onset_strength) if onset_strength else start_idx
                    else:
                        refined_start = start_idx

                    current_start = refined_start

                    current_end = i
                    current_pitch_hz = fp.pitch_hz
                    pitch_buffer = [fp.pitch_hz]
            else:
                # Extend note
                # Check vibrato / pitch jump - C2
                # (We already checked is_glitch above, so here diff <= split_cents)
                current_end = i
                # P5: Update pitch reference
                pitch_buffer.append(fp.pitch_hz)
                if len(pitch_buffer) > pitch_buffer_size:
                    pitch_buffer.pop(0)
                current_pitch_hz = float(np.median(pitch_buffer))


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
                # P3: Enforce min_note_duration
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


def _sanitize_notes(notes: List[NoteEvent]) -> List[NoteEvent]:
    clean = []
    for n in notes:
        if n is None:
            continue
        if n.end_sec <= n.start_sec:
            continue
        if n.pitch_hz is None or n.pitch_hz <= 0:
            continue
        clean.append(n)
    clean.sort(key=lambda x: (x.start_sec, x.end_sec, x.pitch_hz))
    return clean

def _dedupe_overlapping_notes(notes: List[NoteEvent], overlap_thr: float = 0.85) -> List[NoteEvent]:
    if not notes:
        return notes
    # Sort primarily by voice, then pitch, then time
    notes = sorted(notes, key=lambda n: (n.voice or 0, n.midi_note or 0, n.start_sec, n.end_sec, -n.confidence))
    out: List[NoteEvent] = []
    for n in notes:
        if not out:
            out.append(n); continue
        p = out[-1]
        # Only check if same voice and same MIDI pitch
        if (p.voice == n.voice) and (p.midi_note == n.midi_note):
            # overlap ratio
            a0, a1 = p.start_sec, p.end_sec
            b0, b1 = n.start_sec, n.end_sec
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0) + 1e-9
            if inter / union >= overlap_thr:
                # keep the higher-confidence one
                if n.confidence > p.confidence:
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
    out = []
    while i < len(notes):
        j = i + 1
        group = [notes[i]]
        # Collect group within tolerance window
        while j < len(notes) and abs(notes[j].start_sec - notes[i].start_sec) <= tol:
            group.append(notes[j])
            j += 1

        if len(group) >= 2:
            # Snap all to the earliest start in the group
            s0 = min(n.start_sec for n in group)
            for n in group:
                n.start_sec = s0

        out.extend(group)
        i = j
    return out



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
        # Default favors vocals when present (songs), but falls back safely.
        prefer_order = ["vocals", "other", "melody_masked", "mix"]

    mix_margin = float(_get(config, "stem_selection.mix_margin", 0.02) or 0.0)

    scores = {}
    for stem in prefer_order:
        tl = stem_timelines.get(stem)
        if tl is None:
            continue
        vr, mc = _timeline_score(tl)
        # Simple scalar score: prioritize being voiced, then confidence.
        score = vr * (0.5 + mc)
        scores[stem] = (score, vr, mc)

    if not scores:
        # fallback: first available
        stem, tl = next(iter(stem_timelines.items()))
        return stem, tl

    best_stem = max(scores.items(), key=lambda kv: kv[1][0])[0]
    best_score = scores[best_stem][0]

    # Optional safety: prefer mix if it's close to best (avoid separation artifacts)
    if "mix" in scores and best_stem != "mix":
        mix_score = scores["mix"][0]
        if mix_score >= best_score - mix_margin:
            best_stem = "mix"

    return best_stem, stem_timelines[best_stem]


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

    # Short-circuit if precalculated notes exist (E2E path)
    if analysis_data.precalculated_notes is not None:
        clean = _sanitize_notes(analysis_data.precalculated_notes)
        analysis_data.notes = clean
        analysis_data.diagnostics["stage_c_mode"] = "precalculated"
        return clean

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
        # Also check flat key without stage_c_ prefix if it was in the report?
        # But report says "stage_c_min_note_duration_ms"
        # However, for nested override "stage_c" -> key
        # We can also check explicit nested overrides.
        nested_c = profile_special.get("stage_c", {})
        if isinstance(nested_c, dict) and key in nested_c:
            return nested_c[key]

        # Then config
        return _get(config, f"stage_c.{key}", default)

    # Patch C0: Resolve overrides for segmentation
    # Specifically map: stage_c_pitch_ref_window_frames, stage_c_conf_start/end

    seg_cfg = dict(resolve_val("segmentation_method", {}) or {})

    # Inject robustness flags into seg_cfg
    # Now config.stage_c has these fields directly, so we can use _get properly
    seg_cfg["use_onset_refinement"] = _get(config, "stage_c.use_onset_refinement", True)
    seg_cfg["use_repeated_note_splitter"] = _get(config, "stage_c.use_repeated_note_splitter", True)

    if "stage_c_pitch_ref_window_frames" in profile_special:
         # Note: stage_c_pitch_ref_window_frames doesn't exist in segmentation_method usually,
         # it's usually a local param or logic. But let's check.
         # Actually it's P5 in _segment_monophonic which currently hardcodes pitch_buffer_size=7.
         # We need to pass it down.
         pass

    # We'll pass seg_cfg into _segment_monophonic, so let's update it if needed
    if "stage_c_pitch_ref_window_frames" in profile_special:
        seg_cfg["pitch_ref_window_frames"] = int(profile_special["stage_c_pitch_ref_window_frames"])

    stem_timelines: Dict[str, List[FramePitch]] = analysis_data.stem_timelines or {}

    if not stem_timelines:
        analysis_data.notes = []
        return []
    stem_name, primary_timeline = _select_best_stem_timeline(stem_timelines, config)
    # Removed strict length check here to allow robust fallback handling loop below

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
        # require ~+6 dB above estimated noise floor (Patch 5B)
        margin = float(_get(config, "stage_c.velocity_map.noise_floor_db_margin", 6.0))
        min_rms = max(min_rms, nf * (10 ** (margin / 20.0)))

    # Build list of timelines to process.
    timelines_to_process: List[Tuple[str, List[FramePitch]]] = [(stem_name, primary_timeline)]
    audio_type = getattr(analysis_data.meta, "audio_type", None)
    allow_secondary = audio_type in (getattr(AudioType, "POLYPHONIC", None), getattr(AudioType, "POLYPHONIC_DOMINANT", None))

    if allow_secondary and len(stem_timelines) > 1:
        # Add others in deterministic order
        other_keys = sorted([k for k in stem_timelines.keys() if k != stem_name])
        for other_name in other_keys:
             timelines_to_process.append((other_name, stem_timelines[other_name]))

    notes: List[NoteEvent] = []

    # seg_cfg is already built above
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

    # Polyphony config
    poly_filter_mode = _get(config, "stage_c.polyphony_filter.mode", "skyline_top_voice")
    max_alt_voices = int(_get(config, "stage_b.voice_tracking.max_alt_voices", 4))
    max_tracks = 1 + max_alt_voices
    poly_pitch_tolerance = float(_get(config, "stage_c.pitch_tolerance_cents", 50.0))

    for vidx, (vname, timeline) in enumerate(timelines_to_process):
        if not timeline or len(timeline) < 2:
            continue

        # Patch: Skyline Top Voice Selection
        # If active, derive frame pitch from the highest confident active pitch
        if poly_filter_mode == "skyline_top_voice":
            new_tl = []
            # Use a modest floor to keep weak melody candidates in play
            skyline_conf_thr = max(0.05, min(conf_thr * 0.5, 0.2))
            for fp in timeline:
                ap = getattr(fp, "active_pitches", []) or []
                # Filter by confidence floor
                cand = [(p, c) for (p, c) in ap if p > 0.0 and c >= skyline_conf_thr]
                if cand:
                    # Fix L5 failure mode: Pick by confidence + continuity + vocal band preference
                    # Previous choice: highest pitch (caused overtones/flute capture)

                    # 1. Sort by confidence descending
                    cand.sort(key=lambda x: x[1], reverse=True)

                    # 2. Prefer vocal band (80-1400Hz) if confidence is close
                    # We take the top few candidates within 10% confidence of the winner
                    top_conf = cand[0][1]
                    contestants = [x for x in cand if x[1] >= top_conf * 0.9]

                    best_cand = contestants[0]

                    # If we have a previous selected pitch (from previous frame in new_tl), use it for continuity
                    prev_pitch = new_tl[-1].pitch_hz if new_tl else 0.0

                    if len(contestants) > 1:
                        # Tie-breaker logic
                        scored_candidates = []
                        for p, c in contestants:
                            score = c
                            # Bonus for vocal band - Expanded to 1400Hz (approx F6) for female vocals/flutes
                            if 80.0 <= p <= 1400.0:
                                score += 0.05

                            # Bonus for continuity
                            if prev_pitch > 0.0:
                                cents_diff = abs(1200.0 * math.log2(p / prev_pitch))
                                if cents_diff < 100: # Within semitone
                                    score += 0.1
                                elif cents_diff < 1200: # Within octave
                                    score += 0.05
                                else:
                                    score -= 0.05 # Large jumps penalized

                            scored_candidates.append(((p, c), score))

                        best_cand = max(scored_candidates, key=lambda x: x[1])[0]

                    p_best, c_best = best_cand

                    # Recompute MIDI
                    midi_new = int(round(69 + 12 * math.log2(p_best / 440.0)))
                    # Create updated frame (assuming immutable-ish usage, creating new is safer)
                    fp2 = FramePitch(
                        time=fp.time,
                        pitch_hz=p_best,
                        midi=midi_new,
                        confidence=c_best,
                        rms=fp.rms,
                        active_pitches=fp.active_pitches
                    )
                    new_tl.append(fp2)
                else:
                    # Fallback to original Stage B choice
                    new_tl.append(fp)
            # Replace the timeline for processing
            timeline = new_tl

        # Detect polyphonic context based on active pitch annotations
        poly_frames = [fp for fp in timeline if getattr(fp, "active_pitches", []) and len(fp.active_pitches) > 1]

        # Should we enable polyphonic segmentation?
        # Gate: config != "skyline_top_voice" (implied "process_all") or just presence of poly_frames?
        # WI implies we should use active_pitches if present, unless explicitly disabled?
        # WI says: "Detect polyphony... OR config signal"
        enable_polyphony = (len(poly_frames) > 0) and (poly_filter_mode != "skyline_top_voice")

        # Decompose into tracks if polyphony active
        voice_timelines = []
        if enable_polyphony:
             voice_timelines = _decompose_polyphonic_timeline(
                 timeline,
                 pitch_tolerance_cents=poly_pitch_tolerance,
                 max_tracks=max_tracks
             )

             # Feature: Decomposed Melody Mode (L5.2)
             # If mode is "decomposed_melody", pick only the best track as the "melody"
             # This avoids outputting accompaniment tracks for monophonic benchmarks
             if poly_filter_mode == "decomposed_melody" and len(voice_timelines) > 1:
                # Score tracks by total confidence mass in vocal range
                best_idx = 0
                best_score = -1.0

                for i, tl in enumerate(voice_timelines):
                    score = 0.0
                    for fp in tl:
                        if fp.pitch_hz > 0:
                            s = fp.confidence
                            # Boost vocal range (80-1400Hz)
                            if 80.0 <= fp.pitch_hz <= 1400.0:
                                 s *= 1.2
                            score += s

                    if score > best_score:
                        best_score = score
                        best_idx = i

                # Keep only the best track
                voice_timelines = [voice_timelines[best_idx]]

        else:
             voice_timelines = [timeline]

        # Determine thresholds
        voice_conf_gate = conf_thr
        voice_min_dur_s = min_note_dur_s
        has_distinct_poly = _has_distinct_poly_layers(timeline)

        # Tune thresholds for context
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

        # Process each decomposed voice track
        for sub_idx, sub_tl in enumerate(voice_timelines):
             # Skip empty tracks
             if not any(fp.pitch_hz > 0 for fp in sub_tl):
                 continue

             use_viterbi = smoothing_enabled and seg_method in ("viterbi", "hmm")

             # WI Patch: If using Skyline Top Voice, we must rely on pitch changes for segmentation,
             # so we disable Viterbi (which only tracks voicing state) to use the pitch-sensitive segmenter.
             if poly_filter_mode == "skyline_top_voice":
                 use_viterbi = False

             segs = []

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
                 # For poly decomposition: only allow repeated-note splitter for the strongest track (sub_idx==0)
                 # to prevent false splits in accompaniment
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
                 # Extract data from sub_tl
                 mids = [sub_tl[i].midi for i in range(s, e + 1) if sub_tl[i].midi is not None and sub_tl[i].midi > 0]
                 hzs = [sub_tl[i].pitch_hz for i in range(s, e + 1) if sub_tl[i].pitch_hz > 0]
                 confs = [sub_tl[i].confidence for i in range(s, e + 1)]
                 rmss = [sub_tl[i].rms for i in range(s, e + 1)]

                 if not mids:
                     continue

                 if rmss and np.mean(rmss) < min_rms:
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

                 # Stable Voice ID
                 # vidx = stem index
                 # sub_idx = local voice index (0..4)
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

    # Patch C8: Optional Bass Backtracking with Strict Clamp
    bass_backtrack_ms = float(profile_special.get("stage_c_backtrack_ms", 0.0) or profile_special.get("bass_backtrack_ms", 0.0))
    if bass_backtrack_ms > 0.0 and notes:
        backtrack_sec = bass_backtrack_ms / 1000.0
        min_dur_s = min_note_dur_s # Use the resolved min dur

        # Sort to ensure safe lookback order
        notes.sort(key=lambda n: n.start_sec)

        # We need to do this per-voice to respect monophonic lines correctly.
        # Group by voice first.
        by_voice = {}
        for n in notes:
            v = n.voice
            if v not in by_voice: by_voice[v] = []
            by_voice[v].append(n)

        for v in by_voice:
            v_notes = by_voice[v]
            # Assumed sorted by start_sec

            for i, n in enumerate(v_notes):
                prev_end = v_notes[i-1].end_sec if i > 0 else 0.0

                # Desired new start
                new_start = max(0.0, float(n.start_sec) - backtrack_sec)

                # Strict clamp: No overlap with previous note in same voice
                new_start = max(new_start, prev_end)

                # Safety clamp: Ensure min duration remains
                # If backtracking eats into end, we must stop before end - min_dur
                max_start_limit = max(new_start, float(n.end_sec) - min_dur_s)
                # Ideally we don't move start later than it was, only earlier.
                # But here we are checking if new_start is valid w.r.t end.
                if new_start < max_start_limit:
                    n.start_sec = float(new_start)

    if enable_polyphony:
        notes = _dedupe_overlapping_notes(notes)
        snap_ms = float(_get(config, "stage_c.chord_onset_snap_ms", 25.0))
        if snap_ms > 0:
            notes = _snap_chord_starts(notes, tol_ms=snap_ms)

    # Populate diagnostics
    if hasattr(analysis_data, "diagnostics"):
         analysis_data.diagnostics["stage_c"] = {
             "segmentation_method": seg_method,
             "timelines_processed": len(timelines_to_process),
             "note_count_raw": len(notes)
         }

    quantized_notes = quantize_notes(notes, analysis_data=analysis_data)

    # Gap tolerance post-processing (C1 - Glitch tolerance)
    # Merge notes that are same pitch and extremely close in time, even after quantization if needed,
    # or before quantization. Merging before is better.
    # Currently _segment_monophonic handles gap_tolerance in frames.
    # But if notes are split due to other reasons and land close, we might want to merge.
    # However, existing logic is mostly sufficient.

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
