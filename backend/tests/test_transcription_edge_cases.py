import numpy as np
import soundfile as sf
import pytest

from backend.transcription import transcribe_audio_pipeline


def _write_audio(tmp_path, name, audio, sr=22050):
    path = tmp_path / name
    sf.write(path, audio.astype(np.float32), sr)
    return str(path)


def test_transcribe_handles_silence(tmp_path):
    audio = np.zeros(22050, dtype=np.float32)
    path = _write_audio(tmp_path, "silence.wav", audio)

    result = transcribe_audio_pipeline(path, trim_silence=False)
    meta = result.analysis_data.meta

    assert meta.duration_sec > 0
    assert meta.sample_rate == meta.target_sr
    assert all(fp.pitch_hz == 0 for fp in result.analysis_data.timeline)


def test_transcribe_clipped_audio_preserves_metadata(tmp_path):
    sr = 16000
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    audio = np.clip(1.5 * np.sin(2 * np.pi * 330.0 * t), -1.0, 1.0)
    path = _write_audio(tmp_path, "clipped.wav", audio, sr=sr)

    result = transcribe_audio_pipeline(path, trim_silence=False, target_sample_rate=sr)
    meta = result.analysis_data.meta

    assert pytest.approx(meta.duration_sec, rel=0.05) == 1.0
    assert meta.sample_rate == sr
    musicxml_str = str(result.musicxml)
    assert musicxml_str.strip().startswith("<?xml")


def test_transcribe_short_bursts_tracks_boundaries(tmp_path):
    sr = 22050
    burst = 0.1
    silence = 0.1
    t_note = np.linspace(0, burst, int(sr * burst), endpoint=False)
    tone = 0.8 * np.sin(2 * np.pi * 523.25 * t_note)

    audio = np.concatenate([
        np.zeros(int(sr * silence)),
        tone,
        np.zeros(int(sr * silence)),
        tone,
    ]).astype(np.float32)
    path = _write_audio(tmp_path, "bursts.wav", audio, sr=sr)

    result = transcribe_audio_pipeline(path, trim_silence=False, hop_length=256)
    timeline = result.analysis_data.timeline
    assert timeline, "Timeline should retain burst detections"

    times = [fp.time for fp in timeline if fp.pitch_hz > 0]
    assert times, "Expected voiced frames for bursts"
    assert min(times) >= 0.0
    assert max(times) <= (len(audio) / sr) * 1.05


def test_transcribe_respects_hop_length_and_timebase(tmp_path):
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    path = _write_audio(tmp_path, "hop_check.wav", tone, sr=sr)

    hop = 256
    window = 1024

    result = transcribe_audio_pipeline(
        path,
        trim_silence=False,
        target_sample_rate=sr,
        hop_length=hop,
        window_size=window,
    )

    meta = result.analysis_data.meta
    assert meta.hop_length == hop
    assert meta.window_size == window

    timeline = result.analysis_data.timeline
    assert timeline, "Expected timeline entries for hop/timebase validation"

    dts = [timeline[i].time - timeline[i - 1].time for i in range(1, len(timeline))]
    assert pytest.approx(np.median(dts), rel=0.05, abs=1e-4) == hop / sr

    # Ensure note timing stays within audio duration
    if result.analysis_data.notes:
        for note in result.analysis_data.notes:
            assert note.start_sec >= -1e-3
            assert note.end_sec <= (len(tone) / sr) * 1.05
