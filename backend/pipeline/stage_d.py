from typing import List, Optional, Dict, Tuple
import numpy as np
import music21
from music21 import (
    stream,
    note,
    chord,
    tempo,
    meter,
    key,
    dynamics,
    articulations,
    layout,
    clef,
)

from .models import NoteEvent, AnalysisData
from .config import PIANO_61KEY_CONFIG, PipelineConfig


def quantize_and_render(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
) -> str:
    """
    Stage D: Render Sheet Music (MusicXML) using music21.

    WI alignment:
    - Staff split at C4 (configurable).
    - Uses beat-based durations and onsets from Stage C (duration_beats, start_beats).
    - Chord representation via music21.chord.Chord when multiple notes share onset.
    - Ties handled via music21.makeMeasures() + makeTies().
    - Rests filled via makeRests() for complete measures.
    - Dynamics from NoteEvent.velocity (0–1) and NoteEvent.dynamic ('p', 'mf', 'f').
    - Staccato marking for short notes based on threshold_beats.
    - Glissando config is read but not yet applied (explicit TODO).
    """

    d_conf = config.stage_d
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0
    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else "4/4"
    split_pitch = d_conf.staff_split_point.get("pitch", 60)  # C4

    # NOTE: divisions_per_quarter is currently not used.
    # music21 chooses internal divisions automatically; we
    # keep this knob in config for potential future grid forcing.
    divisions_per_quarter = getattr(d_conf, "divisions_per_quarter", None)

    # Glissando config (currently not applied in v1)
    gliss_conf = d_conf.glissando_threshold_general
    piano_gliss_conf = d_conf.glissando_handling_piano

    # --------------------------------------------------------
    # 1. Setup Score and Parts (treble + bass)
    # --------------------------------------------------------
    s = stream.Score()

    part_treble = stream.Part()
    part_treble.id = "P1"
    part_bass = stream.Part()
    part_bass.id = "P2"

    # Clefs
    part_treble.append(clef.TrebleClef())
    part_bass.append(clef.BassClef())

    # Time Signature
    ts_obj = meter.TimeSignature(ts_str)
    part_treble.append(ts_obj)
    part_bass.append(ts_obj)

    # Tempo
    mm = tempo.MetronomeMark(number=float(bpm))
    part_treble.append(mm)
    part_bass.append(mm)

    # Key (if detected)
    if analysis_data.meta.detected_key:
        try:
            k = key.Key(analysis_data.meta.detected_key)
            part_treble.append(k)
            part_bass.append(k)
        except Exception:
            # If key parsing fails, we simply skip key signature
            pass

    # --------------------------------------------------------
    # 2. Prepare Events: group by staff + onset in beats
    # --------------------------------------------------------

    # Helper: get start beat and duration in beats for an event.
    # Prefer Stage C's quantized fields; fallback to seconds.
    quarter_dur = 60.0 / float(bpm)

    def get_event_beats(e: NoteEvent) -> Tuple[float, float]:
        # duration_beats (from Stage C) if present
        dur_beats = getattr(e, "duration_beats", None)
        if dur_beats is None:
            dur_beats = (e.end_sec - e.start_sec) / quarter_dur

        # start_beats (from Stage C) if present
        start_beats = getattr(e, "start_beats", None)
        if start_beats is None:
            start_beats = e.start_sec / quarter_dur

        return float(start_beats), float(dur_beats)

    # Sort events by start time first to stabilize grouping
    events_sorted = sorted(events, key=lambda e: (e.start_sec, e.midi_note))

    # Group key: (staff_name, start_beat_quantized) -> list[NoteEvent]
    # We assume Stage C already quantized to a grid; we use a small rounding
    # to avoid FP noise.
    #
    # NOTE:
    # All notes on the same staff and onset beat are rendered as a single chord.
    # Multi-voice notation on the same staff is not yet supported (future option:
    # use NoteEvent.voice to split into separate Voices instead of one Chord).
    staff_groups: Dict[Tuple[str, float], List[NoteEvent]] = {}

    for e in events_sorted:
        # Staff from NoteEvent if set; otherwise use split_pitch rule
        staff_name = getattr(e, "staff", None)
        if staff_name not in ("treble", "bass"):
            staff_name = "treble" if e.midi_note >= split_pitch else "bass"

        start_beats, dur_beats = get_event_beats(e)
        # Slight rounding to a grid of 1/64 beats to group "simultaneous" onsets
        start_key = round(start_beats * 64.0) / 64.0

        key_tuple = (staff_name, start_key)
        if key_tuple not in staff_groups:
            staff_groups[key_tuple] = []
        staff_groups[key_tuple].append(e)

    # --------------------------------------------------------
    # 3. Create music21 Notes / Chords from grouped events
    # --------------------------------------------------------

    staccato_thresh = d_conf.staccato_marking.get("threshold_beats", 0.25)

    def build_m21_from_group(group: List[NoteEvent]):
        """
        Build a music21 Note or Chord for a group of NoteEvents
        sharing the same staff and onset beat.
        """
        # Use quantized beats from Stage C if present, else fallback to seconds
        start_beats, dur_beats_first = get_event_beats(group[0])

        # If durations differ in group, we take the max (block chord assumption)
        dur_beats_candidates: List[float] = []
        for e in group:
            _, dur_b = get_event_beats(e)
            dur_beats_candidates.append(dur_b)
        dur_beats = max(dur_beats_candidates) if dur_beats_candidates else dur_beats_first

        # Ensure positive non-zero duration
        if dur_beats <= 0.0:
            dur_beats = staccato_thresh

        # Pitches
        midi_pitches = sorted(list({e.midi_note for e in group}))

        if len(midi_pitches) > 1:
            m21_obj: music21.note.NotRest = chord.Chord(midi_pitches)
        else:
            m21_obj = note.Note(midi_pitches[0])

        # Duration: quarterLength is beats (1 beat = quarter note)
        q_len = float(dur_beats)
        try:
            m21_obj.duration = music21.duration.Duration(q_len)
        except Exception:
            m21_obj.duration = music21.duration.Duration(1.0)

        # Velocity: average from NoteEvent.velocity (0..1)
        velocities = [getattr(e, "velocity", 0.7) for e in group]
        avg_vel_norm = float(np.mean(velocities)) if velocities else 0.7
        midi_velocity = int(max(1, min(127, round(avg_vel_norm * 127.0))))
        m21_obj.volume.velocity = midi_velocity

        # Dynamics: from NoteEvent.dynamic if present (p/mf/f)
        # We pick the most "intense" dynamic in the group
        dyn_priority = {"p": 1, "mp": 2, "mf": 3, "f": 4}
        chosen_dyn = None
        best_score = 0
        for e in group:
            dyn = getattr(e, "dynamic", None)
            if dyn is None:
                continue
            score = dyn_priority.get(dyn, 0)
            if score > best_score:
                chosen_dyn = dyn
                best_score = score

        if chosen_dyn:
            dyn_obj = dynamics.Dynamic(chosen_dyn)
            m21_obj.expressions.append(dyn_obj)

        # Staccato articulation for very short notes
        if q_len < float(staccato_thresh):
            m21_obj.articulations.append(articulations.Staccato())

        return m21_obj, float(start_beats)

    # Insert groups into parts
    for (staff_name, start_key), group in sorted(staff_groups.items(), key=lambda x: x[0]):
        m21_obj, start_beats = build_m21_from_group(group)
        offset = float(start_beats)  # beats; music21 uses quarterLength offsets

        if staff_name == "bass":
            part_bass.insert(offset, m21_obj)
        else:
            part_treble.insert(offset, m21_obj)

    s.append(part_treble)
    s.append(part_bass)

    # --------------------------------------------------------
    # 4. Make Measures, Rests, Ties, and layout
    # --------------------------------------------------------

    try:
        # makeMeasures splits notes across bars and sets <measure> structure.
        s_quant = s.makeMeasures(inPlace=False)

        # Fill gaps with rests so measures are complete
        s_quant.makeRests(inPlace=True)

        # Create ties for split notes
        s_quant.makeTies(inPlace=True)

        # TODO (future): Implement glissando spanners based on gliss_conf and piano_gliss_conf:
        # - For non-piano profiles, detect adjacent notes on same staff/voice where:
        #       |Δpitch_semitones| >= gliss_conf["min_semitones"]
        #       and Δtime_beats <= gliss_conf["max_time_ms"] / (60000.0 / bpm)
        #   and insert music21.spanner.Glissando(start_note, end_note).
        # - For piano when piano_gliss_conf["enabled"] is False, skip glissando encoding.

    except Exception as e:
        print(f"[Stage D] makeMeasures/makeRests/makeTies failed: {e}")
        s_quant = s  # fallback: raw score (export might still work if durations are sane)

    # --------------------------------------------------------
    # 5. Export to MusicXML string
    # --------------------------------------------------------

    from music21.musicxml import m21ToXml
    exporter = m21ToXml.GeneralObjectExporter(s_quant)
    musicxml_bytes = exporter.parse()
    musicxml_str = musicxml_bytes.decode("utf-8")

    return musicxml_str
