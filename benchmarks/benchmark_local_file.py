import argparse
import sys
import os
import glob
import music21
from typing import List, Tuple

# Ensure backend is in path
sys.path.append(os.getcwd())

from backend.transcription import transcribe_audio_pipeline

def parse_musicxml_to_notes(xml_path_or_str: str) -> List[Tuple[int, float, float]]:
    """
    Parses MusicXML (file path or string) and returns list of (midi, onset_beat, duration_beat).
    Sorts by onset.
    """
    try:
        # Check if it's a file path
        if os.path.exists(xml_path_or_str[:500]): # simple check
             s = music21.converter.parse(xml_path_or_str)
        elif xml_path_or_str.strip().startswith("<?xml"):
             s = music21.converter.parse(xml_path_or_str, format='musicxml')
        else:
             # Assume it's a path that doesn't exist?
             s = music21.converter.parse(xml_path_or_str)

        notes = []
        # Flatten and extract notes
        # We need absolute timing in beats.
        # music21's flat representation gives offsets from start.
        flat_s = s.flat.notes

        for n in flat_s:
            if n.isNote:
                midi = int(n.pitch.midi)
                onset = float(n.offset)
                duration = float(n.quarterLength)
                notes.append((midi, onset, duration))
            elif n.isChord:
                for p in n.pitches:
                    midi = int(p.midi)
                    onset = float(n.offset)
                    duration = float(n.quarterLength)
                    notes.append((midi, onset, duration))

        notes.sort(key=lambda x: x[1])
        return notes

    except Exception as e:
        print(f"Error parsing MusicXML: {e}")
        return []

def match_notes(
    ref_notes: List[Tuple[int, float, float]],
    pred_notes: List[Tuple[int, float, float]],
    onset_tol: float = 0.25,
    duration_tol: float = 0.25
) -> Tuple[float, float]:
    """
    Returns (pitch_accuracy, rhythm_accuracy) percentages.

    Greedy matching:
    For each ref note, find a pred note that matches midi and onset.
    If multiple match, pick the closest onset.
    """
    if not ref_notes:
        return 0.0, 0.0

    matched_indices = set()
    pitch_matches = 0
    rhythm_matches = 0

    for r_midi, r_onset, r_dur in ref_notes:
        best_match_idx = None
        best_onset_diff = float('inf')

        # Look for a match in predicted notes
        for i, (p_midi, p_onset, p_dur) in enumerate(pred_notes):
            if i in matched_indices:
                continue

            if p_midi == r_midi:
                onset_diff = abs(p_onset - r_onset)
                if onset_diff <= onset_tol:
                    if onset_diff < best_onset_diff:
                        best_onset_diff = onset_diff
                        best_match_idx = i

        if best_match_idx is not None:
            matched_indices.add(best_match_idx)
            pitch_matches += 1

            # Check duration for rhythm accuracy
            # Rhythm accuracy: Pitch matched AND duration matched?
            # Requirement says: "Pitch accuracy: percentage of reference notes for which there exists a predicted note with the SAME midi_note and onset..."
            # "Rhythm accuracy: same but considering durations as well"
            # This implies Rhythm accuracy is a subset of Pitch accuracy.

            p_midi, p_onset, p_dur = pred_notes[best_match_idx]
            dur_diff = abs(p_dur - r_dur)
            if dur_diff <= duration_tol:
                rhythm_matches += 1

    pitch_acc = (pitch_matches / len(ref_notes)) * 100.0
    rhythm_acc = (rhythm_matches / len(ref_notes)) * 100.0

    return pitch_acc, rhythm_acc

def main():
    parser = argparse.ArgumentParser(description="Benchmark Music-Note-Creator pipeline.")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("reference_xml", help="Path to reference MusicXML file")

    args = parser.parse_args()

    audio_path = args.audio_path
    ref_xml_path = args.reference_xml

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)

    if not os.path.exists(ref_xml_path):
        print(f"Error: Reference XML file not found at {ref_xml_path}")
        sys.exit(1)

    print(f"Benchmarking: {audio_path}")
    print(f"Reference: {ref_xml_path}")

    # Run pipeline
    try:
        result = transcribe_audio_pipeline(audio_path, use_crepe=False)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

    # Save outputs
    out_dir = "benchmarks_results"
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    pred_xml_path = os.path.join(out_dir, f"{base_name}_pred.musicxml")
    pred_midi_path = os.path.join(out_dir, f"{base_name}_pred.mid")

    with open(pred_xml_path, "w", encoding="utf-8") as f:
        f.write(result.musicxml)

    with open(pred_midi_path, "wb") as f:
        f.write(result.midi_bytes)

    # Evaluate
    ref_notes = parse_musicxml_to_notes(ref_xml_path)
    # Parse predicted XML string
    pred_notes = parse_musicxml_to_notes(result.musicxml)

    pitch_acc, rhythm_acc = match_notes(ref_notes, pred_notes)

    print("\n=== BENCHMARK RESULT ===")
    print(f"Reference notes: {len(ref_notes)}")
    print(f"Predicted notes: {len(pred_notes)}")
    print(f"Pitch accuracy:  {pitch_acc:.1f}%")
    print(f"Rhythm accuracy: {rhythm_acc:.1f}% (±0.25 beats)")
    print(f"Predicted MusicXML saved to: {pred_xml_path}")
    print(f"Predicted MIDI saved to: {pred_midi_path}")

if __name__ == "__main__":
    main()
