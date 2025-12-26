import numpy as np
import pytest

from backend.transcription import transcribe_audio_pipeline
from backend.pipeline.models import (
    StageAOutput,
    StageBOutput,
    MetaData,
    Stem,
    TranscriptionResult,
    AudioType,
)


@pytest.fixture()
def mock_stage_a_output():
    meta = MetaData(
        processing_mode="polyphonic",
        target_sr=44100,
        hop_length=256,
        sample_rate=44100,
    )
    return StageAOutput(
        stems={"mix": Stem(audio=np.ones(128, dtype=np.float32), sr=44100, type="mix")},
        meta=meta,
        audio_type=AudioType.POLYPHONIC,
    )


@pytest.fixture()
def mock_stage_b_output():
    time_grid = np.array([0.0, 0.01, 0.02], dtype=np.float32)
    f0_main = np.array([440.0, 441.0, 0.0], dtype=np.float32)
    return StageBOutput(
        time_grid=time_grid,
        f0_main=f0_main,
        f0_layers=[],
        per_detector={},
        stem_timelines={},
    )


def test_processing_mode_and_timeline_fallback(monkeypatch, mock_stage_a_output, mock_stage_b_output):
    # Patch pipeline stages to avoid heavy processing
    monkeypatch.setattr("backend.transcription.load_and_preprocess", lambda *a, **k: mock_stage_a_output)
    monkeypatch.setattr("backend.transcription.extract_features", lambda *a, **k: mock_stage_b_output)
    monkeypatch.setattr("backend.transcription.apply_theory", lambda analysis_data, config: analysis_data)
    monkeypatch.setattr(
        "backend.transcription.quantize_and_render",
        lambda events_with_theory, analysis_data, config: TranscriptionResult(
            musicxml="<score-partwise/>", analysis_data=analysis_data, midi_bytes=b""
        ),
    )
    monkeypatch.setattr("backend.transcription.validate_invariants", lambda *a, **k: None)

    # Patch beat/onset helpers to deterministic outputs
    monkeypatch.setattr("librosa.beat.beat_track", lambda y, sr: (120.0, np.array([0, 1], dtype=int)))
    monkeypatch.setattr("librosa.frames_to_time", lambda frames, sr: np.array([0.0, 1.0]))
    monkeypatch.setattr("librosa.onset.onset_detect", lambda *a, **k: np.array([], dtype=float))

    result = transcribe_audio_pipeline("dummy.wav", trim_silence=False)

    analysis = result.analysis_data
    assert analysis.meta.processing_mode == "polyphonic"  # preserved from Stage A
    assert analysis.n_frames == 3
    assert len(analysis.timeline) == 3
    assert pytest.approx(analysis.frame_hop_seconds, rel=1e-6) == 0.01
    assert analysis.meta.beats == [0.0, 1.0]
    # Ensure frame-sized arrays were normalized to timeline length
    assert len(mock_stage_b_output.time_grid) == analysis.n_frames
    assert len(mock_stage_b_output.f0_main) == analysis.n_frames
