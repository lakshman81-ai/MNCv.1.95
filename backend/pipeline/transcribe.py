"""
High-level transcription orchestrator.

Pipeline:
    Stage A: load_and_preprocess
    Stage B: extract_features
    Stage C: apply_theory
    Stage D: quantize_and_render

Contract (as used in tests/test_pipeline_flow.py):

    from backend.pipeline import transcribe

    result = transcribe(audio_path, config=PIANO_61KEY_CONFIG)

    assert isinstance(result.musicxml, str)
    analysis = result.analysis_data

    assert analysis.meta.sample_rate == sr_target
    assert analysis.meta.target_sr == sr_target
    assert len(analysis.notes) > 0
"""

import time
from typing import Optional

from .config import PIANO_61KEY_CONFIG, PipelineConfig
from .stage_a import load_and_preprocess
from .stage_b import extract_features
from .stage_c import apply_theory
from .stage_d import quantize_and_render
from .models import AnalysisData, StageAOutput, TranscriptionResult
from .validation import validate_invariants, dump_resolved_config
from .instrumentation import PipelineLogger


def transcribe(
    audio_path: str,
    config: Optional[PipelineConfig] = None,
    pipeline_logger: Optional[PipelineLogger] = None,
) -> TranscriptionResult:
    """
    High-level API: run the full pipeline on an audio file.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file (e.g., .wav, .mp3).
    config : PipelineConfig, optional
        Full pipeline configuration. If None, uses PIANO_61KEY_CONFIG.

    Returns
    -------
    TranscriptionResult
        Object with `.musicxml` (Stage D output) and `.analysis_data`
        (meta + stem timelines + notes).
    """
    if config is None:
        config = PIANO_61KEY_CONFIG

    pipeline_logger = pipeline_logger or PipelineLogger()

    # --------------------------------------------------------
    # Stage A: Signal Conditioning
    # --------------------------------------------------------
    pipeline_logger.log_event(
        "stage_a",
        "start",
        {
            "audio_path": audio_path,
            "detector_preferences": config.stage_b.detectors if hasattr(config, "stage_b") else {},
        },
    )
    t_a = time.perf_counter()
    # Update: Pass full config to allow Stage A to resolve detector-based params
    stage_a_out: StageAOutput = load_and_preprocess(
        audio_path,
        config=config,
    )
    validate_invariants(stage_a_out, config)
    pipeline_logger.record_timing(
        "stage_a",
        time.perf_counter() - t_a,
        metadata={
            "sample_rate": stage_a_out.meta.sample_rate,
            "hop_length": stage_a_out.meta.hop_length,
            "window_size": stage_a_out.meta.window_size,
            "audio_type": stage_a_out.audio_type.value,
        },
    )

    # --------------------------------------------------------
    # Stage B: Feature Extraction (Detectors + Ensemble)
    # --------------------------------------------------------
    pipeline_logger.log_event(
        "stage_b",
        "detector_selection",
        {
            "detectors": config.stage_b.detectors if hasattr(config, "stage_b") else {},
            "dependencies": PipelineLogger.dependency_snapshot(["torch", "crepe", "demucs"]),
        },
    )
    t_b = time.perf_counter()
    stage_b_out = extract_features(
        stage_a_out,
        config=config,
    )
    validate_invariants(stage_b_out, config)
    pipeline_logger.record_timing(
        "stage_b",
        time.perf_counter() - t_b,
        metadata={
            "resolved_hop": stage_a_out.meta.hop_length,
            "resolved_window": stage_a_out.meta.window_size,
            "detectors_run": list(stage_b_out.per_detector.get("mix", {}).keys()),
        },
    )

    # --------------------------------------------------------
    # Stage C: Note Event Extraction (Theory Application)
    # --------------------------------------------------------
    analysis_data = AnalysisData(
        meta=stage_a_out.meta,
        timeline=[],
        stem_timelines=stage_b_out.stem_timelines,
    )

    pipeline_logger.log_event(
        "stage_c",
        "segmentation",
        {
            "method": config.stage_c.segmentation_method.get("method") if hasattr(config, "stage_c") else None,
            "pitch_tolerance_cents": getattr(config.stage_c, "pitch_tolerance_cents", None) if hasattr(config, "stage_c") else None,
        },
    )
    t_c = time.perf_counter()
    notes = apply_theory(
        analysis_data,
        config=config,
    )
    validate_invariants(notes, config, analysis_data=analysis_data)
    pipeline_logger.record_timing("stage_c", time.perf_counter() - t_c, metadata={"note_count": len(notes)})

    # --------------------------------------------------------
    # Stage D: Quantization + MusicXML Rendering
    # --------------------------------------------------------
    t_d = time.perf_counter()
    d_out: TranscriptionResult = quantize_and_render(
        notes,
        analysis_data,
        config=config,
    )
    validate_invariants(d_out, config)
    pipeline_logger.record_timing(
        "stage_d",
        time.perf_counter() - t_d,
        metadata={"beats_detected": len(d_out.analysis_data.beats)},
    )

    # Save resolved configuration for debugging parity with API path
    dump_resolved_config(config, stage_a_out.meta, stage_b_out)

    pipeline_logger.log_event(
        "pipeline",
        "complete",
        {"notes": len(d_out.analysis_data.notes), "run_dir": pipeline_logger.run_dir},
    )
    pipeline_logger.finalize()

    # d_out contains musicxml, analysis_data, and midi_bytes.
    # Ensure we return a TranscriptionResult with all fields.
    # analysis_data was updated in-place by quantize_and_render (it adds beats, etc.)
    # but d_out.analysis_data is safer to use.

    return TranscriptionResult(
        musicxml=d_out.musicxml,
        analysis_data=d_out.analysis_data,
        midi_bytes=d_out.midi_bytes
    )
