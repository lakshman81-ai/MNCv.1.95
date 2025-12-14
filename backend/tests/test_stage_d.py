
import pytest
import numpy as np
from backend.pipeline.stage_d import quantize_and_render
from backend.pipeline.models import NoteEvent, AnalysisData, MetaData

def test_gap_merging():
    # Tempo 120. 1/32 note = 0.0625s.
    # Gap < 0.0625 should be merged (previous note extended).

    meta = MetaData(tempo_bpm=120.0)
    analysis = AnalysisData(meta=meta)

    # Note 1 ends at 1.0
    # Note 2 starts at 1.05 (Gap 0.05 < 0.0625)
    n1 = NoteEvent(start_sec=0.0, end_sec=1.0, midi_note=60, pitch_hz=261.6, rms_value=0.1)
    n2 = NoteEvent(start_sec=1.05, end_sec=2.0, midi_note=60, pitch_hz=261.6, rms_value=0.1) # Same pitch for merge

    quantize_and_render([n1, n2], analysis)

    # n1 end should be extended to n2.end_sec (merged)
    assert n1.end_sec == 2.0

def test_velocity_mapping():
    meta = MetaData()
    analysis = AnalysisData(meta=meta)

    # Low RMS (-60dB -> 0.001) -> Velocity 20
    # High RMS (0dB -> 1.0) -> Velocity 105
    # Mid RMS (-30dB -> ~0.0316) -> Mid Velocity

    n_quiet = NoteEvent(start_sec=0, end_sec=1, midi_note=60, pitch_hz=261.6, rms_value=0.001)
    n_loud = NoteEvent(start_sec=1, end_sec=2, midi_note=62, pitch_hz=293.6, rms_value=1.0) # Diff pitch to avoid merge

    xml_out = quantize_and_render([n_quiet, n_loud], analysis)

    # Check updated velocity fields (0-1 range stored in float)
    # 20/127 ~= 0.157
    # 105/127 ~= 0.826

    assert 0.15 <= n_quiet.velocity <= 0.16
    assert 0.82 <= n_loud.velocity <= 0.83
