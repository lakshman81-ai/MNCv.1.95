import argparse
import json
import logging
import os
import sys
import numpy as np
import librosa
import music21

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio to music notation")
    parser.add_argument("--audio_path", required=True, help="Path to input audio file")
    parser.add_argument("--audio_start_offset_sec", type=float, default=10.0, help="Start offset in seconds")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate")
    parser.add_argument("--output_musicxml", default="output.musicxml", help="Output MusicXML path")
    parser.add_argument("--output_midi", default="output.mid", help="Output MIDI path")
    parser.add_argument("--output_png", default="output.png", help="Output PNG path")
    parser.add_argument("--output_log", default="transcription_log.json", help="Output log path")
    parser.add_argument(
        "--quantization_strategy",
        choices=["nearest", "classifier"],
        default="nearest",
        help="Quantization strategy: snap to nearest denominator or use a learned classifier",
    )
    return parser.parse_args()

def freq_to_midi(freq):
    if freq is None or freq <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq / 440.0)))

class DurationClassifier:
    """
    Lightweight probabilistic classifier that maps continuous beat durations to
    the most likely rhythmic category. The class stores Gaussian prototypes in
    log-beat space that can be refined offline and shipped as parameters.
    """

    def __init__(self, categories, mus=None, sigmas=None):
        self.categories = np.array(categories, dtype=float)
        # Default prototypes center around the provided categories in log space
        self.mus = np.array(mus) if mus is not None else np.log(self.categories)
        # Default spread loosely models human timing variability
        self.sigmas = (
            np.array(sigmas)
            if sigmas is not None
            else np.full_like(self.mus, 0.20, dtype=float)
        )

    def predict(self, beat_duration):
        beat_duration = max(beat_duration, 1e-6)
        log_val = np.log(beat_duration)
        log_probs = -0.5 * ((log_val - self.mus) / self.sigmas) ** 2
        idx = int(np.argmax(log_probs))
        return float(self.categories[idx])


def quantize_duration(
    seconds,
    start_time,
    tempo_times,
    tempo_curve,
    denominators,
    classifier=None,
):
    local_bpm = float(np.interp(start_time, tempo_times, tempo_curve))
    beats = seconds * (local_bpm / 60.0)

    if classifier is not None:
        quantized = classifier.predict(beats)
    else:
        quantized = min(denominators, key=lambda x: abs(x - beats))

    return quantized, beats, local_bpm


def compute_tempo_curve(y, sr, hop_length):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo_curve = librosa.beat.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None
    )
    tempo_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    return tempo_times, tempo_curve


def cumulative_beats(tempo_times, tempo_curve):
    if len(tempo_times) == 0:
        return np.array([])

    beats = [0.0]
    for i in range(1, len(tempo_times)):
        dt = tempo_times[i] - tempo_times[i - 1]
        beats.append(beats[-1] + (tempo_curve[i - 1] / 60.0) * dt)
    return np.array(beats)

def main():
    args = parse_args()

    # Parameters from WI
    params = {
        "f0_fmin": librosa.note_to_hz('C2'),
        "f0_fmax": librosa.note_to_hz('C6'),
        "frame_length": 2048,
        "hop_length": 512,
        "pitch_smoothing_ms": 75,
        "min_note_duration_sec": 0.06,
        "merge_gap_threshold_sec": 0.15,
        "quantization_tolerance": 0.20,
        "rhythmic_denominators": [
            4.0,
            3.0,
            2.0,
            1.5,
            1.0,
            0.75,
            0.6666666667,
            0.5,
            0.3333333333,
            0.25,
            0.1666666667,
            0.125,
        ],
        "split_midi_threshold": 60
    }

    logger.info(f"Starting transcription for {args.audio_path}")

    if not os.path.exists(args.audio_path):
        logger.error(f"Audio file not found: {args.audio_path}")
        sys.exit(1)

    # 1. Load Audio
    logger.info("Step 1: Loading Audio...")
    try:
        y, sr = librosa.load(args.audio_path, sr=args.sample_rate, offset=args.audio_start_offset_sec)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        sys.exit(1)

    # 2. Preprocess (Normalization)
    logger.info("Step 2: Preprocessing...")
    y = librosa.util.normalize(y)

    # 3. Tempo and Beats
    logger.info("Step 3: Estimating Tempo Curve...")
    tempo_times, tempo_curve = compute_tempo_curve(y, sr, params["hop_length"])
    if tempo_curve.size == 0:
        logger.error("Failed to compute tempo curve")
        sys.exit(1)

    global_tempo = float(np.median(tempo_curve))
    logger.info(
        f"Detected tempo curve with median tempo {global_tempo:.2f} BPM and {len(tempo_curve)} windows"
    )

    beat_positions = cumulative_beats(tempo_times, tempo_curve)

    # 4. Pitch Tracking (pyin)
    logger.info("Step 4: Pitch Tracking...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=params["f0_fmin"],
        fmax=params["f0_fmax"],
        sr=sr,
        frame_length=params["frame_length"],
        hop_length=params["hop_length"]
    )

    times = librosa.times_like(f0, sr=sr, hop_length=params["hop_length"])

    # 7. Segment Notes (Simplified logic merging Steps 5-8)
    logger.info("Step 7: Segmenting Notes...")

    current_midi = None
    start_time = None

    # Convert f0 sequence to MIDI sequence (handling None/unvoiced)
    midi_sequence = [freq_to_midi(f) if v else None for f, v in zip(f0, voiced_flag)]

    note_events = []

    for i, midi in enumerate(midi_sequence):
        t = times[i]

        if midi is None:
            if current_midi is not None:
                # End note
                duration = t - start_time
                if duration >= params["min_note_duration_sec"]:
                    note_events.append({
                        "start_sec": start_time,
                        "end_sec": t,
                        "midi": current_midi
                    })
                current_midi = None
                start_time = None
        else:
            if current_midi is None:
                # Start note
                current_midi = midi
                start_time = t
            elif midi != current_midi:
                # Pitch change -> End current, start new
                duration = t - start_time
                if duration >= params["min_note_duration_sec"]:
                    note_events.append({
                        "start_sec": start_time,
                        "end_sec": t,
                        "midi": current_midi
                    })
                current_midi = midi
                start_time = t

    # Close last note if active
    if current_midi is not None:
        note_events.append({
            "start_sec": start_time,
            "end_sec": times[-1],
            "midi": current_midi
        })

    # 8. Merge Adjacent Same-Pitch Notes
    logger.info("Step 8: Merging Notes...")
    merged_notes = []
    if note_events:
        merged_notes.append(note_events[0])
        for n in note_events[1:]:
            last = merged_notes[-1]
            gap = n["start_sec"] - last["end_sec"]
            if n["midi"] == last["midi"] and gap < params["merge_gap_threshold_sec"]:
                last["end_sec"] = n["end_sec"]
            else:
                merged_notes.append(n)

    logger.info(f"Total notes extracted: {len(merged_notes)}")

    # 9-14. Quantization, Voice Assignment, MusicXML
    logger.info("Step 9-14: Building Score...")

    s = music21.stream.Score()
    p_treble = music21.stream.Part()
    p_bass = music21.stream.Part()

    p_treble.insert(0, music21.clef.TrebleClef())
    p_bass.insert(0, music21.clef.BassClef())

    # Tempo
    mm = music21.tempo.MetronomeMark(number=global_tempo)
    p_treble.insert(0, mm)

    duration_classifier = None
    if args.quantization_strategy == "classifier":
        duration_classifier = DurationClassifier(params["rhythmic_denominators"])

    log_entries = []

    for n in merged_notes:
        dur_sec = n["end_sec"] - n["start_sec"]
        q_dur, raw_beats, local_bpm = quantize_duration(
            dur_sec,
            n["start_sec"],
            tempo_times,
            tempo_curve,
            params["rhythmic_denominators"],
            classifier=duration_classifier,
        )

        m21_note = music21.note.Note(n["midi"])
        # Snap written duration to the quantized value instead of the raw beat length
        m21_note.quarterLength = q_dur

        # Calculate start beat using integrated tempo curve
        start_beat = float(np.interp(n["start_sec"], tempo_times, beat_positions))

        # Determine staff
        if n["midi"] >= params["split_midi_threshold"]:
            p_treble.insert(start_beat, m21_note)
            staff = "treble"
        else:
            p_bass.insert(start_beat, m21_note)
            staff = "bass"

        log_entries.append({
            "start_sec": n["start_sec"],
            "end_sec": n["end_sec"],
            "midi": n["midi"],
            "quantized_rhythm": q_dur,
            "start_beat": start_beat,
            "local_bpm": local_bpm,
            "staff": staff
        })

    s.insert(0, p_treble)
    s.insert(0, p_bass)

    # 13. Key Detection
    try:
        key = s.analyze('key')
        p_treble.insert(0, key)
        logger.info(f"Detected key: {key}")
    except Exception as e:
        logger.warning(f"Key detection failed: {e}")

    # Make Measures and Ties (Crucial for notation)
    logger.info("Structuring measures and ties...")
    try:
        s.makeMeasures(inPlace=True)
        s.makeTies(inPlace=True)
    except Exception as e:
        logger.error(f"Failed to make measures/ties: {e}")

    # 14. Render Output
    logger.info("Step 14: Writing Output Files...")
    try:
        s.write('musicxml', fp=args.output_musicxml)
        logger.info(f"Written MusicXML to {args.output_musicxml}")
    except Exception as e:
        logger.error(f"Failed to write MusicXML: {e}")

    try:
        s.write('midi', fp=args.output_midi)
        logger.info(f"Written MIDI to {args.output_midi}")
    except Exception as e:
        logger.error(f"Failed to write MIDI: {e}")

    # 15. Render PNG
    try:
        # Attempts to use external helper (MuseScore/LilyPond)
        s.write('musicxml.png', fp=args.output_png)
        logger.info(f"Written PNG to {args.output_png}")
    except Exception as e:
        logger.warning(f"PNG generation failed (environment dependencies likely missing): {e}")

    # 16. Logging
    with open(args.output_log, 'w') as f:
        json.dump(log_entries, f, indent=2)
    logger.info(f"Written log to {args.output_log}")

if __name__ == "__main__":
    main()
