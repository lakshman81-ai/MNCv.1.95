
import numpy as np
import pytest
from unittest.mock import MagicMock
from backend.pipeline.stage_c import _sec_to_beat_index, _beat_index_to_sec, quantize_notes, apply_theory
from backend.pipeline.stage_a import load_and_preprocess
from backend.pipeline.stage_d import quantize_and_render
from backend.pipeline.config import StageAConfig, StageCConfig
from backend.pipeline.models import NoteEvent, AnalysisData, MetaData, FramePitch, AudioType

# --- STAGE A TESTS ---

def test_stage_a_config_defaults():
    """Test that Stage A handles missing config safely."""
    import soundfile as sf
    import tempfile
    import os

    # Create dummy audio
    sr = 22050
    audio = np.zeros(int(sr * 0.1), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    try:
        # Pass None as config
        out = load_and_preprocess(tmp_path, config=None)
        assert out is not None
        assert out.meta.target_sr == 44100 # Default default
    finally:
        os.unlink(tmp_path)

def test_stage_a_legacy_hpf_fallback():
    """Test legacy HPF config mapping."""
    import soundfile as sf
    import tempfile
    import os

    sr = 22050
    audio = np.zeros(int(sr * 0.1), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    conf = StageAConfig(
        high_pass_filter={},
        high_pass_filter_cutoff={"value": 100.0},
        high_pass_filter_order={"value": 4},
    )

    try:
        # Mock logger to verify params
        mock_logger = MagicMock()
        load_and_preprocess(tmp_path, config=conf, pipeline_logger=mock_logger)

        # Verify logger call
        found = False
        for call in mock_logger.log_event.call_args_list:
            args, kwargs = call
            if args[1] == "params_resolved":
                payload = kwargs['payload']
                hpf = payload['high_pass_filter']
                assert hpf['cutoff_hz'] == 100.0
                assert hpf['legacy_fallback_used'] is True
                found = True
        assert found
    finally:
        os.unlink(tmp_path)

# --- STAGE C TESTS ---

def test_quantization_extrapolation():
    # Beats: 0.0, 1.0, 2.0 (60 BPM)
    beats = [0.0, 1.0, 2.0]

    # Negative time (before first beat)
    # interval 1.0. t=-0.5 -> beat -0.5
    idx = _sec_to_beat_index(-0.5, beats)
    assert np.isclose(idx, -0.5)

    # Round trip
    t = _beat_index_to_sec(idx, beats)
    assert np.isclose(t, -0.5)

    # Far future
    # last interval 1.0. last beat 2.0.
    # t=5.0 -> 3.0s past last beat -> +3.0 beats -> beat 5.0 (index 2 + 3)
    idx = _sec_to_beat_index(5.0, beats)
    assert np.isclose(idx, 5.0)

def test_quantize_notes_clamping():
    # Beats: 1.0, 2.0 (start late)
    beats = [1.0, 2.0]

    # Note starts at 0.0, ends at 0.5. Before first beat.
    # interval is 1.0.
    # 0.0 is 1.0s before beat 1.0. -> beat index -1.0.
    # Quantized beat: round(-1.0) = -1.0.
    # Converted back: beat -1.0 -> time 0.0 (extrapolated).
    # Wait, 1.0 + (-1.0 - 0) * 1.0 = 0.0.
    # If we had note at -0.5 -> beat -1.5 -> round to -2.0.
    # Time: 1.0 + (-2.0) = -1.0.
    # Should clamp to 0.0.

    n = NoteEvent(start_sec=-0.5, end_sec=-0.1, midi_note=60, pitch_hz=261.6, confidence=0.9)
    # Provide analysis with beats
    analysis = AnalysisData()
    analysis.beats = beats

    out = quantize_notes([n], analysis_data=analysis)

    assert out[0].start_sec >= 0.0
    assert out[0].end_sec >= 0.0
    assert out[0].end_sec > out[0].start_sec

def test_stage_c_noise_floor_gating():
    # If noise floor is high, notes with low RMS should be dropped

    timeline = [
        FramePitch(time=i*0.01, pitch_hz=440.0, midi=69, confidence=0.9, rms=0.005)
        for i in range(10)
    ] # rms 0.005 is approx -46 dB

    analysis = AnalysisData()
    analysis.stem_timelines = {"mix": timeline}
    analysis.meta.noise_floor_rms = 0.01 # -40 dB. Gate should be +6dB => -34 dB => 0.02

    # Config default min_db is -40.

    notes = apply_theory(analysis)
    # RMS 0.005 < 0.02. Should get 0 notes.
    assert len(notes) == 0

    # Now lower noise floor
    analysis.meta.noise_floor_rms = 0.001
    # Adjust config min_db to -60 (0.001) so hard limit doesn't kill it
    conf = {"stage_c": {"velocity_map": {"min_db": -60.0}}}

    notes = apply_theory(analysis, config=conf)
    assert len(notes) == 1

# --- STAGE D TESTS ---

def test_stage_d_beat_fallback():
    events = [NoteEvent(start_sec=0.0, end_sec=1.0, midi_note=60, pitch_hz=261.6)]
    analysis = AnalysisData(events=events)
    # Empty beats
    analysis.beats = []
    # Meta beats
    analysis.meta.beat_times = [0.0, 1.0, 2.0]

    mock_logger = MagicMock()

    # We expect quantize_and_render to use meta.beat_times
    # We can check logger
    quantize_and_render(events, analysis, pipeline_logger=mock_logger)

    found = False
    for call in mock_logger.log_event.call_args_list:
        args, kwargs = call
        if args[1] == "beat_grid_selected":
            assert kwargs['payload']['source'] == "meta.beat_times"
            found = True
    assert found
