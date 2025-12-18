BENCHMARK_LEVELS = [
    {
        "id": "L0_SIGNAL",
        "name": "Signal Primitives",
        "description": "10s of pure Sine/Sawtooth waves at fixed freq (e.g., 440Hz).",
        "examples": ["sine_440", "sawtooth_440"],
        "expected_metrics": {"stage_a_bpm_sane": False, "stage_b_fer": 0.05} # Example expectations
    },
    {
        "id": "L1_MONO",
        "name": "Monophonic Scales",
        "description": "Clean C Major scale, constant velocity, synthetic instrument.",
        "examples": ["c_major_scale"],
        "expected_metrics": {"stage_c_recall": 1.0}
    },
    {
        "id": "L2_POLY_SIMPLE",
        "name": "Simple Polyphony",
        "description": "Melody + Bass (2 voices), distinct frequency ranges.",
        "examples": ["melody_bass_2voice"],
        "expected_metrics": {"stage_c_voice_assignment": 0.9}
    },
    {
        "id": "L3_HOMOPHONIC",
        "name": "Homophonic Texture",
        "description": "Melody + Chords (Block chords), synthetic piano.",
        "examples": ["melody_chords"],
        "expected_metrics": {"stage_c_chord_f1": 0.85}
    },
    {
        "id": "L4_REAL",
        "name": "Real World",
        "description": "Happy Birthday / Old MacDonald (Real Instruments).",
        "examples": ["happy_birthday_real", "old_macdonald_real"],
        "is_real_audio": True # Marker to skip synthesis and look in mock_data
    }
]
