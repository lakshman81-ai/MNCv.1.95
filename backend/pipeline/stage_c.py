from typing import List, Dict, Any
import numpy as np
import librosa
import scipy.signal

from .models import NoteEvent, AnalysisData, FramePitch
from .config import PIANO_61KEY_CONFIG, StageCConfig, PipelineConfig

# HMM states
HMM_STATE_SILENCE = 0
HMM_STATE_ATTACK = 1
HMM_STATE_STABLE = 2
HMM_NUM_STATES = 3


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


# ------------------------------------------------------------
# HMM Processor (monophonic stems: vocals / bass)
# ------------------------------------------------------------

class HMMProcessor:
    def __init__(self, config: StageCConfig):
        self.config = config
        # Transition Probabilities (Silence -> Attack -> Stable)
        self.trans_mat = np.array([
            [0.99, 0.01, 0.00],  # From Silence
            [0.10, 0.00, 0.90],  # From Attack
            [0.05, 0.00, 0.95],  # From Stable
        ])
        self.trans_mat = self.trans_mat / self.trans_mat.sum(axis=1, keepdims=True)
        self.start_prob = np.array([0.9, 0.1, 0.0])

    def _compute_emissions(self, timeline: List[FramePitch]) -> np.ndarray:
        """
        Build emission probabilities for Silence / Attack / Stable using
        confidence + energy (RMS) as per WI: Attack–Sustain–Silence.
        """
        n_frames = len(timeline)
        emission = np.zeros((HMM_NUM_STATES, n_frames), dtype=float)

        v_map = self.config.velocity_map
        min_db, max_db = v_map["min_db"], v_map["max_db"]

        for t, fp in enumerate(timeline):
            conf = fp.confidence
            rms = fp.rms

            # Normalize rms into [0,1] roughly based on velocity_map
            if rms <= 0.0:
                db = -80.0
            else:
                db = 20.0 * np.log10(rms)
            clamped = max(min_db, min(max_db, db))
            energy_norm = (clamped - min_db) / (max_db - min_db + 1e-6)

            # Silence: high when both confidence and energy are low
            emission[HMM_STATE_SILENCE, t] = np.exp(-4.0 * (conf + energy_norm))

            # Attack: high when energy is high but confidence is still ramping up
            emission[HMM_STATE_ATTACK, t] = energy_norm * (1.0 - conf)

            # Stable: high when confidence is high and energy is moderate–high
            emission[HMM_STATE_STABLE, t] = conf * (0.5 + 0.5 * energy_norm)

        # Ensure no zero columns to keep Viterbi stable
        emission += 1e-6
        return emission

    def decode(self, timeline: List[FramePitch]) -> List[int]:
        n_frames = len(timeline)
        if n_frames == 0:
            return []

        emission = self._compute_emissions(timeline)

        try:
            state_seq = librosa.sequence.viterbi(
                emission, self.trans_mat, p_init=self.start_prob
            )
            return state_seq
        except Exception:
            # Fallback: mark all as silence
            return [HMM_STATE_SILENCE] * n_frames

    def segment_notes(self, timeline: List[FramePitch], states: List[int]) -> List[NoteEvent]:
        """
        Convert HMM state sequence + FramePitch into NoteEvents.
        Uses:
        - min_note_duration_ms
        - gap_filling.max_gap_ms
        - pitch_tolerance_cents
        - frame_stability.stable_frames_required
        """
        notes: List[NoteEvent] = []
        current_note_start = -1
        accum_data = {"midi": [], "hz": [], "conf": [], "rms": []}

        min_dur_sec = self.config.min_note_duration_ms / 1000.0
        stable_required = int(self.config.frame_stability.get("stable_frames_required", 2))

        pending_start = -1
        consecutive_note_frames = 0

        for t, state in enumerate(states):
            is_note = (state == HMM_STATE_STABLE) or (state == HMM_STATE_ATTACK)

            if is_note:
                # Track pending region before we promote to a full note
                if pending_start == -1:
                    pending_start = t
                    consecutive_note_frames = 1
                else:
                    consecutive_note_frames += 1

                if current_note_start == -1 and consecutive_note_frames >= stable_required:
                    # Promote pending_start to an active note start
                    current_note_start = pending_start

                # Only accumulate data once a note is active
                if current_note_start != -1:
                    fp = timeline[t]
                    if fp.pitch_hz > 0:
                        m = hz_to_midi(fp.pitch_hz)
                        accum_data["midi"].append(m)
                        accum_data["hz"].append(fp.pitch_hz)
                        accum_data["conf"].append(fp.confidence)
                        accum_data["rms"].append(fp.rms)

                        # Legato / pitch jump check (avoid glissando as multiple notes)
                        if len(accum_data["midi"]) > 5:
                            curr_avg = float(np.mean(accum_data["midi"][-5:]))
                            last_avg = float(np.mean(accum_data["midi"][:-1]))
                            if abs(curr_avg - last_avg) > 0.8:  # ~0.8 semitone jump
                                # Split note here
                                self._close_note(
                                    notes,
                                    timeline,
                                    current_note_start,
                                    t,
                                    accum_data,
                                    min_dur_sec,
                                )
                                current_note_start = t
                                pending_start = t
                                consecutive_note_frames = 1
                                accum_data = {
                                    "midi": [m],
                                    "hz": [fp.pitch_hz],
                                    "conf": [fp.confidence],
                                    "rms": [fp.rms],
                                }
            else:
                # Reset pending region
                pending_start = -1
                consecutive_note_frames = 0

                # Note ended
                if current_note_start != -1:
                    self._close_note(
                        notes, timeline, current_note_start, t, accum_data, min_dur_sec
                    )
                    current_note_start = -1
                    accum_data = {"midi": [], "hz": [], "conf": [], "rms": []}

        # Close trailing note
        if current_note_start != -1:
            self._close_note(
                notes, timeline, current_note_start, len(states), accum_data, min_dur_sec
            )

        # Gap Filling (Legato)
        max_gap_sec = self.config.gap_filling.get("max_gap_ms", 100.0) / 1000.0
        merged_notes: List[NoteEvent] = []
        if notes:
            merged_notes.append(notes[0])
            for i in range(1, len(notes)):
                curr = notes[i]
                prev = merged_notes[-1]
                gap = curr.start_sec - prev.end_sec
                if gap <= max_gap_sec and abs(curr.midi_note - prev.midi_note) < (
                    self.config.pitch_tolerance_cents / 100.0
                ):
                    # Merge into previous
                    prev.end_sec = curr.end_sec
                else:
                    merged_notes.append(curr)

        return merged_notes

    def _close_note(
        self,
        notes: List[NoteEvent],
        timeline: List[FramePitch],
        start: int,
        end: int,
        data: Dict[str, List[float]],
        min_dur_sec: float,
    ):
        if end <= start:
            return
        dur = timeline[end - 1].time - timeline[start].time
        if dur < min_dur_sec:
            return

        if not data["hz"]:
            return

        avg_hz = float(np.mean(data["hz"]))
        avg_conf = float(np.mean(data["conf"])) if data["conf"] else 0.0
        avg_rms = float(np.mean(data["rms"])) if data["rms"] else 0.0
        midi_note = int(round(hz_to_midi(avg_hz)))

        notes.append(
            NoteEvent(
                start_sec=timeline[start].time,
                end_sec=timeline[end - 1].time,
                midi_note=midi_note,
                pitch_hz=avg_hz,
                confidence=avg_conf,
                rms_value=avg_rms,
            )
        )


# ------------------------------------------------------------
# Greedy Polyphonic Processor (piano stem)
# ------------------------------------------------------------

class GreedyPolyProcessor:
    def __init__(self, config: StageCConfig):
        self.config = config
        # Convert cents to semitones for matching tolerance
        self.match_tolerance = float(config.pitch_tolerance_cents) / 100.0

    def process(self, timeline: List[FramePitch]) -> List[NoteEvent]:
        notes: List[NoteEvent] = []
        active_tracks: List[Dict[str, Any]] = []
        min_dur_sec = self.config.min_note_duration_ms / 1000.0

        for t, frame in enumerate(timeline):
            curr_time = frame.time
            candidates = frame.active_pitches  # List[(hz, conf)]
            cand_midi = []
            for hz, conf in candidates:
                if hz > 40.0:
                    cand_midi.append(
                        {"midi": hz_to_midi(hz), "hz": hz, "conf": conf, "used": False}
                    )

            next_tracks: List[Dict[str, Any]] = []

            # Update existing tracks
            for track in active_tracks:
                best_match = None
                best_dist = float("inf")

                for c in cand_midi:
                    if c["used"]:
                        continue
                    dist = abs(c["midi"] - track["midi"])
                    if dist < self.match_tolerance and dist < best_dist:
                        best_dist = dist
                        best_match = c

                if best_match is not None:
                    best_match["used"] = True
                    # Smooth pitch
                    track["midi"] = 0.8 * track["midi"] + 0.2 * best_match["midi"]
                    track["hz_accum"].append(best_match["hz"])
                    track["conf_accum"].append(best_match["conf"])
                    track["rms_accum"].append(frame.rms)
                    track["end_t"] = curr_time
                    track["last_seen"] = t
                    next_tracks.append(track)
                else:
                    # Lost track: allow brief gaps, otherwise finalize
                    if t - track["last_seen"] > 3:  # frames tolerance
                        self._finalize_track(notes, track, min_dur_sec)
                    else:
                        next_tracks.append(track)

            # Spawn new tracks
            for c in cand_midi:
                if not c["used"] and c["conf"] > 0.15:
                    new_track: Dict[str, Any] = {
                        "midi": c["midi"],
                        "start_t": curr_time,
                        "end_t": curr_time,
                        "hz_accum": [c["hz"]],
                        "conf_accum": [c["conf"]],
                        "rms_accum": [frame.rms],
                        "last_seen": t,
                    }
                    next_tracks.append(new_track)

            active_tracks = next_tracks

        # Finalize remaining tracks
        for track in active_tracks:
            self._finalize_track(notes, track, min_dur_sec)

        return notes

    def _finalize_track(
        self,
        notes: List[NoteEvent],
        track: Dict[str, Any],
        min_dur_sec: float,
    ):
        dur = track["end_t"] - track["start_t"]
        if dur < min_dur_sec:
            return

        avg_hz = float(np.mean(track["hz_accum"]))
        avg_conf = float(np.mean(track["conf_accum"]))
        avg_rms = float(np.mean(track.get("rms_accum", [0.0])))
        midi_note = int(round(track["midi"]))

        notes.append(
            NoteEvent(
                start_sec=track["start_t"],
                end_sec=track["end_t"],
                midi_note=midi_note,
                pitch_hz=avg_hz,
                confidence=avg_conf,
                rms_value=avg_rms,
            )
        )


# ------------------------------------------------------------
# Tuning, Velocity, and Main Stage C
# ------------------------------------------------------------

def estimate_global_tuning_offset(notes: List[NoteEvent]) -> float:
    """
    Estimate global tuning offset in semitones (fractional) based on pitch_hz.
    """
    if not notes:
        return 0.0
    deviations = []
    for n in notes:
        if n.pitch_hz <= 0:
            continue
        raw_midi = hz_to_midi(n.pitch_hz)
        dev = raw_midi - round(raw_midi)
        deviations.append(dev)

    if not deviations:
        return 0.0

    hist, bins = np.histogram(deviations, bins=50, range=(-0.5, 0.5))
    peak_idx = int(np.argmax(hist))
    center_val = (bins[peak_idx] + bins[peak_idx + 1]) / 2.0
    return float(center_val)


def apply_tuning(notes: List[NoteEvent], offset: float) -> None:
    """
    Apply global tuning offset (in semitones) to notes' MIDI numbers.
    """
    for n in notes:
        if n.pitch_hz <= 0:
            continue
        raw_midi = hz_to_midi(n.pitch_hz)
        corrected_midi = raw_midi - offset
        n.midi_note = int(round(corrected_midi))


def map_velocity(notes: List[NoteEvent], config: StageCConfig) -> None:
    """
    Map raw RMS values to MIDI velocity & dynamic markings.
    Config velocity_map:
        min_db, max_db: loudness bounds
        min_vel, max_vel: velocity range in MIDI (0–127)
    """
    v_map = config.velocity_map
    min_db, max_db = v_map["min_db"], v_map["max_db"]
    min_v, max_v = v_map["min_vel"], v_map["max_vel"]

    for n in notes:
        if n.rms_value <= 0:
            db = -80.0
        else:
            db = 20.0 * np.log10(n.rms_value)

        clamped = max(min_db, min(max_db, db))
        ratio = (clamped - min_db) / (max_db - min_db + 1e-6)

        vel = min_v + ratio * (max_v - min_v)
        n.velocity = vel / 127.0  # normalize to 0–1

        # Dynamic marking (simple bands)
        if vel < 45:
            n.dynamic = "p"
        elif vel < 80:
            n.dynamic = "mf"
        else:
            n.dynamic = "f"


# ------------------------------------------------------------
# Simple threshold-based monophonic segmentation (fallback)
# ------------------------------------------------------------

def _threshold_segment_monophonic(
    timeline: List[FramePitch],
    config: StageCConfig,
) -> List[NoteEvent]:
    """
    Fallback segmentation when segmentation_method.method != 'hmm'.

    Simple logic:
    - Treat frames with pitch_hz > 0 and confidence > 0.5 as voiced.
    - Group contiguous voiced frames into notes.
    - Apply min_note_duration_ms.
    - Use per-frame RMS for velocity mapping later.
    """
    notes: List[NoteEvent] = []
    min_dur_sec = config.min_note_duration_ms / 1000.0
    voiced_thresh = 0.5

    start_idx = None
    hz_accum: List[float] = []
    conf_accum: List[float] = []
    rms_accum: List[float] = []

    def close_note(end_idx: int):
        nonlocal start_idx, hz_accum, conf_accum, rms_accum
        if start_idx is None:
            return
        start_t = timeline[start_idx].time
        end_t = timeline[end_idx - 1].time
        dur = end_t - start_t
        if dur < min_dur_sec:
            # Reset accumulators
            start_idx = None
            hz_accum = []
            conf_accum = []
            rms_accum = []
            return
        if not hz_accum:
            start_idx = None
            hz_accum = []
            conf_accum = []
            rms_accum = []
            return

        avg_hz = float(np.mean(hz_accum))
        avg_conf = float(np.mean(conf_accum)) if conf_accum else 0.0
        avg_rms = float(np.mean(rms_accum)) if rms_accum else 0.0
        midi_note = int(round(hz_to_midi(avg_hz)))

        notes.append(
            NoteEvent(
                start_sec=start_t,
                end_sec=end_t,
                midi_note=midi_note,
                pitch_hz=avg_hz,
                confidence=avg_conf,
                rms_value=avg_rms,
            )
        )
        # Reset accumulators
        start_idx = None
        hz_accum = []
        conf_accum = []
        rms_accum = []

    for idx, fp in enumerate(timeline):
        voiced = fp.pitch_hz > 0 and fp.confidence > voiced_thresh
        if voiced:
            if start_idx is None:
                start_idx = idx
            hz_accum.append(fp.pitch_hz)
            conf_accum.append(fp.confidence)
            rms_accum.append(fp.rms)
        else:
            if start_idx is not None:
                close_note(idx)

    # Close trailing note
    if start_idx is not None:
        close_note(len(timeline))

    return notes


# ------------------------------------------------------------
# Main Stage C entry point
# ------------------------------------------------------------

def apply_theory(
    analysis_data: AnalysisData,
    config: PipelineConfig = PIANO_61KEY_CONFIG,
) -> List[NoteEvent]:
    """
    Stage C: convert continuous F0 timelines into discrete NoteEvent objects.

    - HMM for monophonic stems ('vocals', 'bass') when segmentation_method.method == "hmm".
    - Threshold-based fallback for monophonic if other methods are selected.
    - Greedy polyphonic tracker for 'other'/'piano'/'mix'.
    - Global tuning offset estimation.
    - Velocity mapping.
    - Basic beat-duration assignment (duration_beats).
    """
    stage_c_conf = config.stage_c
    stem_timelines = analysis_data.stem_timelines
    all_notes: List[NoteEvent] = []

    method = stage_c_conf.segmentation_method.get("method", "hmm").lower()

    # 1. Process Vocals/Bass with HMM or threshold-based segmentation
    hmm_proc = HMMProcessor(stage_c_conf) if method == "hmm" else None

    for stem in ["vocals", "bass"]:
        if stem in stem_timelines:
            tl = stem_timelines[stem]
            if hmm_proc is not None:
                states = hmm_proc.decode(tl)
                notes = hmm_proc.segment_notes(tl, states)
            else:
                notes = _threshold_segment_monophonic(tl, stage_c_conf)

            for n in notes:
                n.voice = 1  # Lead
                n.staff = "treble" if stem == "vocals" else "bass"
            all_notes.extend(notes)

    # 2. Process Other/Piano/Mix with Greedy Polyphonic
    greedy_proc = GreedyPolyProcessor(stage_c_conf)
    for stem in ["other", "piano", "mix"]:
        if stem in stem_timelines:
            tl = stem_timelines[stem]
            notes = greedy_proc.process(tl)
            for n in notes:
                n.voice = 1
                # staff split refined in Stage D based on MIDI
                n.staff = "treble"
            all_notes.extend(notes)

    # 3. Adaptive Tuning
    offset = estimate_global_tuning_offset(all_notes)
    analysis_data.meta.tuning_offset = offset
    apply_tuning(all_notes, offset)

    # 4. Velocity Mapping
    map_velocity(all_notes, stage_c_conf)

    # 5. Quantize duration to beats (simple)
    bpm = analysis_data.meta.tempo_bpm if analysis_data.meta.tempo_bpm else 120.0

    # Ensure bpm is plain float (librosa can return numpy types)
    if isinstance(bpm, np.ndarray):
        bpm = float(bpm)

    quarter_dur = 60.0 / float(bpm)

    for n in all_notes:
        n.duration_beats = (n.end_sec - n.start_sec) / quarter_dur
        # measure/beat placement is handled later in Stage D via MusicXML layer

    analysis_data.notes = all_notes
    return all_notes
