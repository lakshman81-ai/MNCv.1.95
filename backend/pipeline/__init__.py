# backend/pipeline/__init__.py

from __future__ import annotations

from typing import Optional

from .config import PIANO_61KEY_CONFIG, PipelineConfig
from .stage_a import load_and_preprocess
from .stage_b import extract_features
from .stage_c import apply_theory
from .stage_d import quantize_and_render
from .models import (
    AnalysisData,
    TranscriptionResult,
    StageAOutput,
    StageBOutput,
    NoteEvent,
)

__all__ = [
    "PIANO_61KEY_CONFIG",
    "PipelineConfig",
    "load_and_preprocess",
    "extract_features",
    "apply_theory",
    "quantize_and_render",
    "AnalysisData",
    "TranscriptionResult",
    "StageAOutput",
    "StageBOutput",
    "NoteEvent",
    "transcribe",
]


def transcribe(
    audio_path: str,
    config: Optional[PipelineConfig] = None,
) -> TranscriptionResult:
    """
    High-level orchestration:
        Stage A → Stage B → Stage C → Stage D

    Parameters
    ----------
    audio_path : str
        Path to the input audio file (e.g. WAV/MP3).
    config : PipelineConfig, optional
        Full pipeline configuration. If None, uses PIANO_61KEY_CONFIG.

    Returns
    -------
    TranscriptionResult
        TranscriptionResult(musicxml, analysis_data, midi_bytes)
    """
    if config is None:
        config = PIANO_61KEY_CONFIG

    # -----------------------------
    # Stage A — Load & Preprocess
    # -----------------------------
    stage_a_out: StageAOutput = load_and_preprocess(
        audio_path=audio_path,
        config=config.stage_a,
        fast_mode=False,
    )

    # Initialize AnalysisData with MetaData from Stage A
    analysis = AnalysisData(meta=stage_a_out.meta)
    # Attach beat grid & any other Stage A diagnostics
    analysis.beats = stage_a_out.beats

    # -----------------------------
    # Stage B — Pitch / F0 Tracking
    # -----------------------------
    stage_b_out: StageBOutput = extract_features(
        stage_a_output=stage_a_out,
        config=config,
    )

    analysis.stem_timelines = stage_b_out.stem_timelines

    # -----------------------------
    # Stage C — Note Segmentation
    # -----------------------------
    notes = apply_theory(
        analysis_data=analysis,
        config=config,
    )

    analysis.notes = notes
    analysis.notes_before_quantization = list(notes)

    # -----------------------------
    # Stage D — MusicXML Rendering
    # -----------------------------
    musicxml_str: str = quantize_and_render(
        events=notes,
        analysis_data=analysis,
        config=config,
    )

    # -----------------------------
    # Optional: MIDI Export
    # -----------------------------
    midi_bytes = b""
    try:
        # Optional helper module; if you have it, use it.
        from .midi_export import notes_to_midi_bytes  # type: ignore

        midi_bytes = notes_to_midi_bytes(
            notes=notes,
            meta=analysis.meta,
        )
    except Exception:
        # Safe fallback: no MIDI export available
        midi_bytes = b""

    return TranscriptionResult(
        musicxml=musicxml_str,
        analysis_data=analysis,
        midi_bytes=midi_bytes,
    )
