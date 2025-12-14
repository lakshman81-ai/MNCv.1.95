import os
import json
import time
import argparse
import pandas as pd
import pretty_midi
import mir_eval
import numpy as np
import sys
from typing import List, Tuple

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend')))

from transcription import transcribe_audio_pipeline
from pipeline.models import NoteEvent

def note_events_to_pretty_midi(note_events: List[NoteEvent]) -> pretty_midi.PrettyMIDI:
    """Converts a list of NoteEvent objects to a pretty_midi.PrettyMIDI object."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    for event in note_events:
        note = pretty_midi.Note(
            velocity=int(event.velocity * 127) if event.velocity else 100,
            pitch=event.midi_note,
            start=event.start_sec,
            end=event.end_sec
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi

def get_mir_eval_data(midi_obj: pretty_midi.PrettyMIDI) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts intervals and pitches from a PrettyMIDI object for mir_eval."""
    intervals = []
    pitches = []
    for instrument in midi_obj.instruments:
        for note in instrument.notes:
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)

    if not intervals:
        return np.empty((0, 2)), np.array([])

    return np.array(intervals), np.array(pitches)

def run_benchmark(dataset_path: str, reports_path: str) -> Tuple[list, float]:
    """Runs the benchmarking script on the dataset and returns results and average F1."""
    audio_dir = os.path.join(dataset_path, 'audio')
    midi_dir = os.path.join(dataset_path, 'midi')

    if not os.path.exists(audio_dir) or not os.listdir(audio_dir):
        print(f"Audio directory not found or is empty: {audio_dir}")
        return [], 0.0

    if not os.path.exists(midi_dir):
        print(f"MIDI directory not found: {midi_dir}")
        return [], 0.0

    results = []

    for filename in sorted(os.listdir(audio_dir)):
        if not filename.endswith(('.wav', '.mp3', '.flac')):
            continue

        audio_path = os.path.join(audio_dir, filename)
        midi_filename = os.path.splitext(filename)[0] + '.mid'
        midi_path = os.path.join(midi_dir, midi_filename)

        if not os.path.exists(midi_path):
            print(f"Warning: Corresponding MIDI file not found for {filename}. Skipping.")
            continue

        print(f"Processing {filename}...")

        # 1. Run transcription pipeline
        start_time = time.time()
        transcription_result = transcribe_audio_pipeline(audio_path, use_mock=False)
        processing_time = time.time() - start_time

        # 2. Convert result to pretty_midi
        estimated_midi = note_events_to_pretty_midi(transcription_result.notes)

        # 3. Load ground truth MIDI
        reference_midi = pretty_midi.PrettyMIDI(midi_path)

        # 4. Evaluate using mir_eval
        ref_intervals, ref_pitches = get_mir_eval_data(reference_midi)
        est_intervals, est_pitches = get_mir_eval_data(estimated_midi)

        scores = mir_eval.transcription.evaluate(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches
        )

        # 5. Reporting
        report = {
            'track_name': filename,
            'processing_time': processing_time,
            'scores': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in scores.items()}
        }
        results.append(report)

        report_filename = os.path.splitext(filename)[0] + '.json'
        with open(os.path.join(reports_path, report_filename), 'w') as f:
            json.dump(report, f, indent=2)

    if not results:
        print("No audio files were processed.")
        return [], 0.0

    # 6. Prepare and print summary
    summary_data = []
    total_f1 = 0
    for result in results:
        f1_score = result['scores'].get('F-measure', 0.0)
        total_f1 += f1_score
        summary_data.append({
            'Track Name': result['track_name'],
            'F1 Score': f1_score,
            'Processing Time': f"{result['processing_time']:.2f}s"
        })

    avg_f1_score = total_f1 / len(results)

    df = pd.DataFrame(summary_data)
    print("\n--- Benchmark Summary ---")
    print(df.to_string(index=False))
    print(f"\nAverage F1 Score: {avg_f1_score:.4f}")

    return summary_data, avg_f1_score

def main():
    parser = argparse.ArgumentParser(description="Run transcription benchmark.")
    parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help='Exit with status 1 if F1 score regresses by more than 1%.'
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(base_dir, 'dataset')
    reports_path = os.path.join(base_dir, 'reports')
    summary_path = os.path.join(reports_path, 'latest_summary.json')

    os.makedirs(reports_path, exist_ok=True)

    summary_data, avg_f1_score = run_benchmark(dataset_path, reports_path)

    if not summary_data:
        print("Benchmark run did not produce any results. Exiting.")
        return

    # Handle regression checking
    if args.fail_on_regression:
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                previous_summary = json.load(f)
                previous_avg_f1 = previous_summary.get('average_f1_score', 0)

            if previous_avg_f1 > 0:
                f1_change_ratio = (avg_f1_score - previous_avg_f1) / previous_avg_f1
                if f1_change_ratio < -0.01:
                    print(f"\n--- REGRESSION DETECTED ---")
                    print(f"Average F1 score dropped from {previous_avg_f1:.4f} to {avg_f1_score:.4f} (a {f1_change_ratio:.2%} change).")
                    sys.exit(1)
                else:
                    print(f"\n--- No Regression Detected ---")
                    print(f"Average F1 score changed from {previous_avg_f1:.4f} to {avg_f1_score:.4f} (a {f1_change_ratio:+.2%} change).")
            else:
                print("\nPrevious F1 score was zero. Cannot compute relative change.")
        else:
            print("\nWarning: No previous benchmark summary found. Cannot check for regression.")

    # Save the current summary
    current_summary = {'average_f1_score': avg_f1_score, 'results': summary_data}
    with open(summary_path, 'w') as f:
        json.dump(current_summary, f, indent=2)
    print(f"\nBenchmark summary saved to {summary_path}")

if __name__ == '__main__':
    main()
