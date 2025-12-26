import pytest

from backend.pipeline.models import FramePitch, NoteEvent
from backend.pipeline.transcribe import _quality_metrics


def test_quality_metrics_falls_back_when_timeline_has_no_active_pitches():
    notes = [
        NoteEvent(start_sec=0.0, end_sec=0.8, midi_note=60, pitch_hz=261.63, confidence=0.9),
        NoteEvent(start_sec=0.9, end_sec=1.6, midi_note=62, pitch_hz=293.66, confidence=0.85),
    ]

    # Simulate a timeline lacking active pitch annotations (e.g., when diagnostics are stripped)
    timeline = [
        FramePitch(time=i * 0.1, pitch_hz=0.0, midi=None, confidence=0.0, active_pitches=None)
        for i in range(20)
    ]

    metrics = _quality_metrics(notes, duration_sec=2.0, timeline_source=timeline)

    assert metrics["note_count"] == 2
    assert metrics["voiced_ratio"] == pytest.approx(0.75)
