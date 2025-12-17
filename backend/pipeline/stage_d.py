from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import tempfile
import os
import math
try:
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
        midi,
    )
    MUSIC21_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    music21 = None  # type: ignore
    MUSIC21_AVAILABLE = False

# Import models and config from the package or the top level.  This makes stage_d
# usable both as ``backend.pipeline.stage_d`` and as a top‑level module ``stage_d``.
try:
    from .models import NoteEvent, AnalysisData, TranscriptionResult  # type: ignore
    from .config import PIANO_61KEY_CONFIG, PipelineConfig  # type: ignore
except Exception:
    from models import NoteEvent, AnalysisData, TranscriptionResult  # type: ignore
    from config import PIANO_61KEY_CONFIG, PipelineConfig  # type: ignore


def _lcm(a, b): return abs(a*b) // math.gcd(a, b)

def quantize_and_render(
    events: List[NoteEvent],
    analysis_data: AnalysisData,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
    pipeline_logger: Optional[Any] = None,
) -> TranscriptionResult:
    """
    Stage D: Render Sheet Music (MusicXML) and MIDI using music21.

    Returns TranscriptionResult containing musicxml string and midi bytes.
    """

    if not MUSIC21_AVAILABLE:
        if pipeline_logger:
            pipeline_logger.log_event(
                "stage_d",
                "feature_disabled",
                {"feature": "music21", "reason": "missing"},
            )
        return TranscriptionResult(
            musicxml="",
            analysis_data=analysis_data,
            midi_bytes=b"",
        )

    d_conf = config.stage_d

    # D1: Tempo/time signature export
    bpm = analysis_data.meta.tempo_bpm
    # Fallback to beat diffs
    if bpm is None:
        beats_seq = getattr(analysis_data, "beats", []) or []
        if beats_seq:
            diffs = np.diff(sorted(beats_seq))
            if diffs.size:
                med = float(np.median(diffs))
                if med > 0:
                    bpm = 60.0 / med

    # Fallback to config or default
    if bpm is None or not np.isfinite(bpm):
        bpm = float(getattr(d_conf, "tempo_bpm", 120.0))

    ts_str = analysis_data.meta.time_signature if analysis_data.meta.time_signature else getattr(d_conf, "time_signature", "4/4")
    split_pitch = d_conf.staff_split_point.get("pitch", 60)  # C4

    # D2: Divisions/measure integrity aligned to quantization grid
    # Compute divisions based on grid
    grid_val = int(getattr(d_conf, "quantization_grid", 16))
    divisions_per_quarter = _lcm(4, grid_val)
    # Note: we don't explicitly force music21 divisions often, but we can try setting it on stream if needed.
    # More importantly, we quantize durations to the grid.

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

    # Time Signature / Tempo / Key: use distinct elements per staff to avoid shared state
    try:
        ts_treble = meter.TimeSignature(ts_str)
        ts_bass = meter.TimeSignature(ts_str)
    except Exception:
        ts_treble = meter.TimeSignature("4/4")
        ts_bass = meter.TimeSignature("4/4")

    part_treble.append(ts_treble)
    part_bass.append(ts_bass)

    beats_per_measure = float(ts_treble.numerator) * (4.0 / float(ts_treble.denominator))

    tempo_treble = tempo.MetronomeMark(number=float(bpm))
    tempo_bass = tempo.MetronomeMark(number=float(bpm))
    part_treble.append(tempo_treble)
    part_bass.append(tempo_bass)

    # Key (if detected)
    if analysis_data.meta.detected_key:
        try:
            key_treble = key.Key(analysis_data.meta.detected_key)
            key_bass = key.Key(analysis_data.meta.detected_key)
            part_treble.append(key_treble)
            part_bass.append(key_bass)
        except Exception:
            pass

    # --------------------------------------------------------
    # 2. Prepare Events: group by staff + onset in beats
    # --------------------------------------------------------
    quarter_dur = 60.0 / float(bpm)
    grid_res_beats = 4.0 / float(grid_val) # e.g. 4/16 = 0.25 beats for 16th note

    # Use beat grid if available
    beat_times = getattr(analysis_data, "beats", []) or []
    use_beat_grid = len(beat_times) > 1

    def get_event_beats(e: NoteEvent) -> Tuple[float, float]:
        start_beats = 0.0
        dur_beats = 0.0

        # Calculate start_beats
        if use_beat_grid:
            # Interpolate beat position from time
            # beat_times are timestamps of beats 0, 1, 2...
            # We assume linear tempo between beat markers
            # np.interp returns extrapolated values for times outside [min, max]
            start_beats = float(np.interp(e.start_sec, beat_times, np.arange(len(beat_times))))

            # For duration, we calculate end_beat and subtract
            end_beats = float(np.interp(e.end_sec, beat_times, np.arange(len(beat_times))))
            dur_beats = end_beats - start_beats
        else:
            # Fallback to constant tempo
            dur_beats = getattr(e, "duration_beats", None)
            if dur_beats is None or not np.isfinite(dur_beats):
                dur_beats = (e.end_sec - e.start_sec) / quarter_dur

            start_beats_val = getattr(e, "start_beats", None)
            if start_beats_val is not None and np.isfinite(start_beats_val):
                start_beats = float(start_beats_val)
            else:
                measure = getattr(e, "measure", None)
                beat = getattr(e, "beat", None)
                if (
                    measure is not None
                    and beat is not None
                    and np.isfinite(measure)
                    and np.isfinite(beat)
                ):
                    start_beats = (float(measure) - 1.0) * beats_per_measure + (float(beat) - 1.0)
                else:
                    start_beats = e.start_sec / quarter_dur

        # D2: Quantize to grid
        # Round start and duration to nearest grid unit
        start_beats = round(start_beats / grid_res_beats) * grid_res_beats
        dur_beats = max(grid_res_beats, round(dur_beats / grid_res_beats) * grid_res_beats)

        return float(start_beats), float(dur_beats)

    events_sorted = sorted(events, key=lambda e: (e.start_sec, e.midi_note))

    # Group by (staff, voice_idx, start_beats)
    # D3: Multi-voice support - we will group by voice index too
    staff_voice_groups: Dict[Tuple[str, int, float], Dict[str, Any]] = {}

    for e in events_sorted:
        staff_name = getattr(e, "staff", None)
        if staff_name not in ("treble", "bass"):
            staff_name = "treble" if e.midi_note >= split_pitch else "bass"

        # Determine voice. If not set, infer from pitch relative to split?
        # Assuming e.voice is populated (1-based index)
        voice_idx = getattr(e, "voice", 1)
        if voice_idx is None: voice_idx = 1

        # Simple split: If e.voice wasn't set smartly upstream, we might want to override here?
        # But requirement D3 says: "rely on music21 Voices if used".
        # And "Rule for assigning voices (minimal, usable now): voice=1 for higher pitches..."
        # We rely on stage C to have set e.voice if polyphonic.
        # If stage C didn't set meaningful voices, they all default to 1.

        start_beats, dur_beats = get_event_beats(e)

        # Key for grouping: staff, voice, start_time
        # Round start time for key to avoid float jitter
        start_key = round(start_beats * 1024.0) / 1024.0

        key_tuple = (staff_name, voice_idx, start_key)
        if key_tuple not in staff_voice_groups:
            staff_voice_groups[key_tuple] = {"events": [], "start_beats": start_beats}
        staff_voice_groups[key_tuple]["events"].append(e)

    # --------------------------------------------------------
    # 3. Create music21 Notes / Chords from grouped events
    # --------------------------------------------------------

    staccato_thresh = d_conf.staccato_marking.get("threshold_beats", 0.25)

    def build_m21_from_group(group: List[NoteEvent], start_beats_value: float):
        _, dur_beats_first = get_event_beats(group[0])

        dur_beats_candidates: List[float] = []
        for e in group:
            _, dur_b = get_event_beats(e)
            dur_beats_candidates.append(dur_b)
        dur_beats = max(dur_beats_candidates) if dur_beats_candidates else dur_beats_first

        if dur_beats <= 0.0:
            dur_beats = staccato_thresh

        midi_pitches = sorted(list({e.midi_note for e in group}))

        if len(midi_pitches) > 1:
            m21_obj: music21.note.NotRest = chord.Chord(midi_pitches)
        else:
            m21_obj = note.Note(midi_pitches[0])

        # D2: Ensure duration is compatible with grid
        # We already quantized dur_beats in get_event_beats
        q_len = dur_beats

        try:
            m21_obj.duration = music21.duration.Duration(q_len)
        except Exception:
            m21_obj.duration = music21.duration.Duration(1.0)

        velocities = [getattr(e, "velocity", 0.7) for e in group]
        # velocity is now 20-105 range from stage C, but might be 0.0-1.0 from legacy?
        # NoteEvent definition says float 0.8 default (legacy) but Stage C uses int 20-105.
        # We should handle both.

        vel_vals = []
        for v in velocities:
            if v > 1.0: # assume MIDI velocity
                vel_vals.append(v)
            else: # assume normalized
                vel_vals.append(v * 127.0)

        avg_vel = float(np.mean(vel_vals)) if vel_vals else 64.0
        midi_velocity = int(max(1, min(127, round(avg_vel))))
        m21_obj.volume.velocity = midi_velocity

        dyn_priority = {"pp": 1, "p": 2, "mp": 3, "mf": 4, "f": 5, "ff": 6}
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

        if q_len < float(staccato_thresh):
            m21_obj.articulations.append(articulations.Staccato())

        return m21_obj, float(start_beats_value)

    # We need to construct voices within parts.
    # Structure: Part -> [Voice1, Voice2, ...]
    # We collect all m21 objects for each (staff, voice)

    staff_voice_content: Dict[str, Dict[int, List[Tuple[float, Any]]]] = {
        "treble": {}, "bass": {}
    }

    for (staff_name, voice_idx, _start_key), group_dict in sorted(staff_voice_groups.items(), key=lambda x: x[0]):
        m21_obj, start_beats_value = build_m21_from_group(
            group_dict["events"], group_dict["start_beats"]
        )
        if voice_idx not in staff_voice_content[staff_name]:
            staff_voice_content[staff_name][voice_idx] = []
        staff_voice_content[staff_name][voice_idx].append((start_beats_value, m21_obj))

    # Helper to build a Part from voice contents
    def _populate_part_with_voices(p: stream.Part, voice_data: Dict[int, List[Tuple[float, Any]]], total_dur: float):
        # If only one voice (voice 1), and no others, we can just insert directly into Part?
        # Actually standard practice is if single voice, just put notes in part.
        # But if multiple voices, use Voice objects.
        # Let's see how many voices we have.

        voice_indices = sorted(voice_data.keys())
        if not voice_indices:
             return

        # Use voices if more than one voice index is present OR if the voice index is > 1
        use_voices = len(voice_indices) > 1 or (voice_indices and voice_indices[0] > 1)

        if use_voices:
            for v_idx in voice_indices:
                v_obj = stream.Voice()
                v_obj.id = str(v_idx)
                for offset, obj in voice_data[v_idx]:
                    v_obj.insert(offset, obj)
                p.insert(0, v_obj)
        else:
            # Single voice 1
            v_idx = voice_indices[0]
            for offset, obj in voice_data[v_idx]:
                p.insert(offset, obj)

    # Determine max duration to pad
    treble_max = 0.0
    for v_idx, items in staff_voice_content["treble"].items():
        for off, obj in items:
            treble_max = max(treble_max, off + obj.duration.quarterLength)

    bass_max = 0.0
    for v_idx, items in staff_voice_content["bass"].items():
        for off, obj in items:
            bass_max = max(bass_max, off + obj.duration.quarterLength)

    target_duration = max(treble_max, bass_max)

    _populate_part_with_voices(part_treble, staff_voice_content["treble"], target_duration)
    _populate_part_with_voices(part_bass, staff_voice_content["bass"], target_duration)

    # Pad to ensure each staff can produce measures - rudimentary padding
    # If using Voices, padding rests need to be inside voices too or in the part?
    # M21 makeMeasures usually handles incomplete measures by padding with rests if we set `makeRests=True`.
    # But let's ensure we at least cover the duration.

    # 3b. Glissando (Optional)
    if gliss_conf.get("enabled", False):
        min_semitones = float(gliss_conf.get("min_semitones", 2.0))
        max_time_ms = float(gliss_conf.get("max_time_ms", 500.0))
        from music21 import spanner

        for p in (part_treble, part_bass):
            # Sort by offset to ensure sequence
            # Filter only Note objects (skip chords)
            p_notes = [n for n in p.flat.notes if isinstance(n, note.Note)]
            p_notes.sort(key=lambda n: n.offset)

            for i in range(len(p_notes) - 1):
                n1 = p_notes[i]
                n2 = p_notes[i+1]

                # Semitone check
                if abs(n2.pitch.ps - n1.pitch.ps) < min_semitones:
                    continue

                # Time gap check
                # n1_end in beats
                n1_end = n1.offset + n1.quarterLength
                n2_start = n2.offset
                gap_beats = n2_start - n1_end

                # Convert to ms
                # bpm is defined earlier
                gap_sec = gap_beats * (60.0 / float(bpm))
                gap_ms = max(0.0, gap_sec * 1000.0)

                if gap_ms <= max_time_ms:
                    gliss = spanner.Glissando(n1, n2)
                    p.insert(n1.offset, gliss)

    # --------------------------------------------------------
    # 4. Make Measures, Rests, Ties, and layout
    # --------------------------------------------------------

    quantized_parts = []
    for p in (part_treble, part_bass):
        try:
            # makeMeasures might fail with complex voice overlaps, so we use best effort
            p_quant = p.makeMeasures(inPlace=False)
            p_quant.makeRests(inPlace=True)
            p_quant.makeTies(inPlace=True)
        except Exception as e:
            print(f"[Stage D] makeMeasures/makeRests/makeTies failed for part {p.id}: {e}")
            p_quant = p
        quantized_parts.append(p_quant)

    s_quant = stream.Score()
    for qp in quantized_parts:
        s_quant.append(qp)

    # --------------------------------------------------------
    # 5. Export to MusicXML string and MIDI bytes
    # --------------------------------------------------------

    # MusicXML
    from music21.musicxml import m21ToXml
    exporter = m21ToXml.GeneralObjectExporter(s_quant)
    musicxml_bytes = exporter.parse()
    musicxml_str = musicxml_bytes.decode("utf-8")

    # MIDI
    midi_bytes = b""
    try:
        mf = midi.translate.music21ObjectToMidiFile(s_quant)
        # We need to write to a temp file because music21.midi.MidiFile doesn't easily write to bytes directly
        # but supports .write() to a file handle or path.
        # Actually .writestr() works if available? Check music21 docs... typically .writestr() produces bytes.
        if hasattr(mf, 'writestr'):
            midi_bytes = mf.writestr()
        else:
            # Fallback to temp file (Windows-safe approach)
            fd, tmp_path = tempfile.mkstemp(suffix=".mid")
            os.close(fd) # Close handle immediately so music21 can open it

            mf.open(tmp_path, 'wb')
            mf.write()
            mf.close()

            with open(tmp_path, 'rb') as f:
                midi_bytes = f.read()
            os.unlink(tmp_path)
    except Exception as e:
        print(f"[Stage D] MIDI export failed: {e}")
    finally:
        if not isinstance(midi_bytes, (bytes, bytearray)):
            midi_bytes = bytes(midi_bytes or b"")

    return TranscriptionResult(
        musicxml=musicxml_str,
        analysis_data=analysis_data,
        midi_bytes=midi_bytes
    )


def _snap_ql(x: float, eps: float = 0.02) -> float:
    """Snap a quarterLength to MusicXML-friendly values."""
    if x is None or not np.isfinite(x):
        return 0.0
    if isinstance(x, (list, tuple)):
        try:
            x = x[0]
        except Exception:
            return 0.0
    x = float(x)
    for denom in (1, 2, 4, 8, 16, 32):
        y = round(x * denom) / denom
        if abs(x - y) <= eps:
            return float(y)
    y = float(round(x * 32) / 32)
    common = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    best = min(common, key=lambda v: abs(y - v))
    if abs(best - y) <= 0.15:
        return float(best)
    return y
