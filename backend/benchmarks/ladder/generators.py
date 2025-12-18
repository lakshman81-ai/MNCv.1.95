from music21 import stream, note, chord, tempo, key, meter, dynamics
import random
import copy

def create_sine_wave_score(freq=440, duration=10.0):
    """
    L0: Creates a score with a single long note.
    """
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=60)) # 1 beat per second

    n = note.Note()
    n.pitch.frequency = freq
    n.quarterLength = duration # 10 seconds at 60 bpm
    n.volume.velocity = 90

    p.append(n)
    s.append(p)
    return s

def create_c_major_scale():
    """
    L1: C Major Scale (Up and Down).
    """
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=120))
    p.append(key.Key('C'))
    p.append(meter.TimeSignature('4/4'))

    # C4 to C5 and back
    pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
               "B4", "A4", "G4", "F4", "E4", "D4", "C4"]

    for pi in pitches:
        n = note.Note(pi)
        n.quarterLength = 1.0 # Quarter notes
        p.append(n)

    s.append(p)
    return s

def create_melody_bass_2voice():
    """
    L2: Melody (Treble) + Bass (2 octaves below).
    """
    s = stream.Score()

    # Melody Part
    p_melody = stream.Part()
    p_melody.id = "Melody"
    p_melody.append(tempo.MetronomeMark(number=100))

    melody_notes = ["C5", "C5", "G5", "G5", "A5", "A5", "G5"] # Twinkle start
    for pi in melody_notes:
        n = note.Note(pi)
        n.quarterLength = 1.0
        if pi == "G5" and melody_notes.index(pi) == 6: # Last one long
            n.quarterLength = 2.0
        p_melody.append(n)

    # Bass Part (C3 range)
    p_bass = stream.Part()
    p_bass.id = "Bass"

    # Simple root notes
    bass_notes = ["C3", "C3", "E3", "C3", "F3", "F3", "C3"]
    for pi in bass_notes:
        n = note.Note(pi)
        n.quarterLength = 1.0
        if pi == "C3" and bass_notes.index(pi) == 6:
            n.quarterLength = 2.0
        p_bass.append(n)

    s.insert(0, p_melody)
    s.insert(0, p_bass)
    return s

def create_melody_chords():
    """
    L3: Melody + Block Chords.
    """
    s = stream.Score()

    # Melody
    p_melody = stream.Part()
    p_melody.append(tempo.MetronomeMark(number=100))
    melody_notes = ["E5", "D5", "C5", "D5", "E5", "E5", "E5"] # Mary had a little lamb
    for i, pi in enumerate(melody_notes):
        n = note.Note(pi)
        n.quarterLength = 1.0
        if i == len(melody_notes) - 1:
             n.quarterLength = 2.0
        p_melody.append(n)

    # Chords
    p_chords = stream.Part()

    # C Major (C-E-G), G Major (G-B-D)
    # Bar 1: C Major (4 beats)
    c_maj = chord.Chord(["C4", "E4", "G4"])
    c_maj.quarterLength = 4.0
    p_chords.insert(0, c_maj)

    # Bar 2: G Major (4 beats)
    g_maj = chord.Chord(["G3", "B3", "D4"])
    g_maj.quarterLength = 4.0
    p_chords.insert(4.0, g_maj)

    s.insert(0, p_melody)
    s.insert(0, p_chords)
    return s

def create_happy_birthday_base():
    """Generates the base monophonic Happy Birthday theme in C Major."""
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=100))
    p.append(key.Key('C'))
    p.append(meter.TimeSignature('3/4'))

    melody_data = [
        ("G4", 1), ("G4", 1), ("A4", 2),
        ("G4", 2), ("C5", 2), ("B4", 3),
        ("G4", 1), ("G4", 1), ("A4", 2),
        ("G4", 2), ("D5", 2), ("C5", 3),
        ("G4", 1), ("G4", 1), ("G5", 2),
        ("E5", 2), ("C5", 2), ("B4", 2), ("A4", 3),
        ("F5", 1), ("F5", 1), ("E5", 2),
        ("C5", 2), ("D5", 2), ("C5", 3),
    ]

    for pitch_name, dur in melody_data:
        n = note.Note(pitch_name)
        n.quarterLength = dur
        p.append(n)

    s.append(p)
    return s

def create_old_macdonald_base():
    """Generates the base monophonic Old MacDonald in C Major."""
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=100))
    p.append(key.Key('C'))
    p.append(meter.TimeSignature('4/4'))

    melody_data = [
        ("C4", 1), ("C4", 1), ("C4", 1), ("G4", 1),
        ("A4", 1), ("A4", 1), ("G4", 2),
        ("E4", 1), ("E4", 1), ("D4", 1), ("D4", 1),
        ("C4", 2),
        ("D4", 1), ("D4", 1), ("C4", 1), ("C4", 2),
    ]

    for pitch_name, dur in melody_data:
        n = note.Note(pitch_name)
        n.quarterLength = dur
        p.append(n)

    s.append(p)
    return s

def apply_expressive_performance(score_in, intensity=1.0):
    s = copy.deepcopy(score_in)
    return s # Placeholder for now

def apply_accompaniment(score_in, song_name, style="block"):
    s = copy.deepcopy(score_in)
    return s # Placeholder for now, simplistic L4 handling

def generate_benchmark_example(example_id: str):
    """
    Dispatcher to create specific benchmark examples.
    """
    if example_id == "sine_440":
        return create_sine_wave_score(freq=440)
    elif example_id == "sawtooth_440":
        return create_sine_wave_score(freq=440)

    elif example_id == "c_major_scale":
        return create_c_major_scale()

    elif example_id == "melody_bass_2voice":
        return create_melody_bass_2voice()

    elif example_id == "melody_chords":
        return create_melody_chords()

    elif "happy_birthday" in example_id:
        # Fallback for L4 if requested but not in mock_data
        # The runner checks is_real_audio first, but if we call this directly:
        return create_happy_birthday_base()

    elif "old_macdonald" in example_id:
        return create_old_macdonald_base()

    else:
        # Fallback for old ones if needed or error
        raise ValueError(f"Unknown example_id: {example_id}")
