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
    from .models import AnalysisData, FramePitch, NoteEvent  # type: ignore
except Exception:
    from models import AnalysisData, FramePitch, NoteEvent  # type: ignore


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
    if not rms_list:
        return 64
    r = float(np.median(np.asarray(rms_list, dtype=np.float64)))
    # Map RMS roughly [0..0.2] -> [vmin..vmax]
    x = max(0.0, min(1.0, r / 0.2))
    v = int(round(vmin + x * (vmax - vmin)))
    return int(max(vmin, min(vmax, v)))


def _segment_monophonic(
    timeline: List[FramePitch],
    conf_thr: float,
    min_note_dur_s: float,
    gap_tolerance_s: float,
    semitone_stability: float = 0.60,
    min_rms: float = 0.01,
) -> List[Tuple[int, int]]:
    """
    Segment monophonic FramePitch into (start_idx, end_idx) segments.

    Strategy:
      - active if midi present, confidence >= conf_thr, AND rms >= min_rms
      - keep segment through short gaps (<= gap_tolerance_s)
      - split if pitch jumps too much (in semitones)
      - split if gap was caused by low RMS (energy drop -> repeated note)
    """
    if len(timeline) < 2:
        return []

    times = np.array([fp.time for fp in timeline], dtype=np.float64)
    mids = np.array([fp.midi if fp.midi is not None else -1 for fp in timeline], dtype=np.int32)
    conf = np.array([fp.confidence for fp in timeline], dtype=np.float64)
    rmss = np.array([fp.rms for fp in timeline], dtype=np.float64)

    # estimate hop
    dt = np.diff(times)
    hop_s = float(np.median(dt)) if dt.size else 0.01
    hop_s = max(1e-4, hop_s)

    active = (mids > 0) & (conf >= float(conf_thr)) & (rmss >= min_rms)

    segs: List[Tuple[int, int]] = []
    i = 0
    while i < len(timeline):
        if not active[i]:
            i += 1
            continue

        s = i
        e = i
        last_m = float(mids[i])
        peak_rms = rmss[i]
        gap = 0.0
        i += 1

        while i < len(timeline):
            if active[i]:
                m = float(mids[i])
                r = rmss[i]
                peak_rms = max(peak_rms, r)

                # Check 1: Pitch jump
                if abs(m - last_m) > semitone_stability * 12.0:
                    break

                # Check 2: Gap analysis (Re-attack)
                if gap > 0:
                    gap_indices = range(e + 1, i)
                    if gap_indices:
                        gap_rms = rmss[gap_indices]
                        if np.mean(gap_rms) < min_rms:
                             break

                # Check 3: Intra-note re-attack (Valley detection)
                # If we are well inside the note (some frames passed)
                # And we detect a rise from a valley
                if i > s + 2:
                    prev_r = rmss[i-1]
                    # Valley: previous RMS significantly lower than peak
                    is_valley = prev_r < peak_rms * 0.85
                    # Attack: current RMS rising significantly
                    is_rise = r > prev_r * 1.05

                    if is_valley and is_rise:
                        break

                last_m = m
                e = i
                gap = 0.0
            else:
                gap += hop_s
                if gap > float(gap_tolerance_s):
                    break
            i += 1

        dur = float(times[e] - times[s] + hop_s)
        if dur >= float(min_note_dur_s):
            segs.append((s, e))

    return segs


def apply_theory(analysis_data: AnalysisData, config: Any = None) -> List[NoteEvent]:
    """
    Convert FramePitch timelines into NoteEvent list.

    - Uses analysis_data.stem_timelines if analysis_data.timeline is empty.
    - Prefers stem order: mix -> vocals -> first available.
    """
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

    timeline = stem_timelines.get(stem_name, [])
    if len(timeline) < 2:
        analysis_data.notes = []
        return []

    # Thresholds (read from config if available)
    conf_thr = float(_get(config, "stage_c.confidence_threshold", _get(config, "stage_c.special.high_conf_threshold", 0.15)))
    min_note_dur_s = float(_get(config, "stage_c.min_note_duration_s", 0.05))
    min_note_dur_ms = _get(config, "stage_c.min_note_duration_ms", None)
    if min_note_dur_ms is not None:
        try:
            min_note_dur_s = float(min_note_dur_ms) / 1000.0
        except Exception:
            pass
    gap_tolerance_s = float(_get(config, "stage_c.gap_tolerance_s", 0.03))

    # Derive semitone stability from configuration.  The config may specify
    # stage_c.pitch_tolerance_cents (e.g. 50), which we convert to a semitone
    # stability factor.  This factor is used to determine how many semitones
    # difference triggers a new note segment.  Default fallback aligns with
    # previous behaviour (~0.60 semitones * 12 = 7.2 semitones).
    pitch_tol_cents = float(_get(config, "stage_c.pitch_tolerance_cents", 50.0))
    try:
        semitone_stability = max(0.01, pitch_tol_cents / 100.0 / 12.0)
    except Exception:
        semitone_stability = 0.60  # fallback

    # Calculate min_rms
    min_db = float(_get(config, "stage_c.velocity_map.min_db", -40.0))
    # RMS gate usually needs to be lower than min velocity threshold to allow decay?
    # Use -40dB or slightly lower as gate.
    min_rms = 10 ** (min_db / 20.0)

    segs = _segment_monophonic(
        timeline=timeline,
        conf_thr=conf_thr,
        min_note_dur_s=min_note_dur_s,
        gap_tolerance_s=gap_tolerance_s,
        semitone_stability=semitone_stability,
        min_rms=min_rms
    )

    # hop estimate for end time
    dt = [timeline[i].time - timeline[i - 1].time for i in range(1, min(len(timeline), 50))]
    hop_s = float(np.median(dt)) if dt else 0.01

    notes: List[NoteEvent] = []
    for (s, e) in segs:
        mids = [timeline[i].midi for i in range(s, e + 1) if timeline[i].midi is not None and timeline[i].midi > 0]
        hzs = [timeline[i].pitch_hz for i in range(s, e + 1) if timeline[i].pitch_hz > 0]
        confs = [timeline[i].confidence for i in range(s, e + 1)]
        rmss = [timeline[i].rms for i in range(s, e + 1)]

        if not mids:
            continue

        midi_note = int(round(float(np.median(mids))))
        pitch_hz = float(np.median(hzs)) if hzs else 0.0
        confidence = float(np.mean(confs)) if confs else 0.0
        velocity = _velocity_from_rms(rmss)

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
                velocity=velocity,
            )
        )

    analysis_data.notes = notes
    return notes


def quantize_notes(
    notes: List[NoteEvent],
    tempo_bpm: float = 120.0,
    grid: str = "1/16",
    min_steps: int = 1,
) -> List[NoteEvent]:
    """
    Quantize note start/end times to a rhythmic grid.

    Parameters
    ----------
    notes : List[NoteEvent]
    tempo_bpm : float
        Tempo used to convert beats to seconds.
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

    bpm = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else 120.0
    sec_per_beat = 60.0 / bpm

    # Parse grid
    denom = 16
    try:
        m = grid.strip().split("/")
        if len(m) == 2:
            denom = int(m[1])
        else:
            denom = int(grid)
    except Exception:
        denom = 16
    denom = max(1, denom)

    step_beats = 4.0 / float(denom)  # in 4/4, quarter note = 1 beat
    step_sec = sec_per_beat * step_beats
    step_sec = max(1e-4, step_sec)

    out: List[NoteEvent] = []
    for n in notes:
        s = float(n.start_sec)
        e = float(n.end_sec)

        qs = round(s / step_sec) * step_sec
        qe = round(e / step_sec) * step_sec

        if qe <= qs:
            qe = qs + max(int(min_steps), 1) * step_sec

        # enforce minimum duration
        if (qe - qs) < max(int(min_steps), 1) * step_sec:
            qe = qs + max(int(min_steps), 1) * step_sec

        out.append(
            NoteEvent(
                start_sec=float(qs),
                end_sec=float(qe),
                midi_note=int(n.midi_note),
                pitch_hz=float(n.pitch_hz),
                confidence=float(n.confidence),
                velocity=int(n.velocity),
            )
        )

    return out
