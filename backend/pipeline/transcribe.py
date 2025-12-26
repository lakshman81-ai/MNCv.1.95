from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .instrumentation import PipelineLogger
from .models import AnalysisData, NoteEvent, TranscriptionResult
from .stage_a import load_and_preprocess
from .stage_b import compute_decision_trace, extract_features
from .stage_c import apply_theory
from .stage_d import quantize_and_render
from .neural_transcription import transcribe_onsets_frames


def _quality_metrics(
    notes: List[NoteEvent],
    duration_sec: float,
    *,
    timeline_source: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    duration_sec = float(max(1e-6, duration_sec or 0.0))
    note_count = int(len(notes or []))
    if note_count == 0:
        return {
            "voiced_ratio": 0.0,
            "note_count": 0,
            "notes_per_sec": 0.0,
            "median_note_dur_ms": 0.0,
            "fragmentation_lt_80ms": 0.0,
        }

    durs = []
    tiny = 0
    total_note_dur = 0.0
    for n in notes:
        dur = float(getattr(n, "end_sec", 0.0)) - float(getattr(n, "start_sec", 0.0))
        dur = max(0.0, dur)
        durs.append(dur)
        total_note_dur += dur
        if dur < 0.08:
            tiny += 1

    durs_sorted = sorted(durs)
    mid = len(durs_sorted) // 2
    median_dur = durs_sorted[mid] if durs_sorted else 0.0

    # voiced_ratio:
    # - prefer detector timeline if available (classic path)
    # - else fallback to coverage ratio (E2E path)
    voiced_ratio: Optional[float] = None
    if timeline_source is not None:
        total_frames = max(1, len(timeline_source))
        voiced_frames = 0
        frames_with_active_attr = 0
        for fr in timeline_source:
            ap = getattr(fr, "active_pitches", None)
            if ap is None:
                continue
            frames_with_active_attr += 1
            if len(ap) > 0:
                voiced_frames += 1
        if frames_with_active_attr > 0:
            voiced_ratio = float(voiced_frames / total_frames)

    if voiced_ratio is None:
        voiced_ratio = float(min(1.0, total_note_dur / duration_sec))

    return {
        "voiced_ratio": float(max(0.0, min(1.0, voiced_ratio))),
        "note_count": note_count,
        "notes_per_sec": float(note_count / duration_sec),
        "median_note_dur_ms": float(1000.0 * median_dur),
        "fragmentation_lt_80ms": float(tiny / max(1, note_count)),
    }


def _quality_score(metrics: Dict[str, Any]) -> float:
    """
    Deterministic scalar score in [0, 1].
    Intended to reject: (a) empty output, (b) extreme fragmentation, (c) crazy note rate.
    """
    if int(metrics.get("note_count", 0) or 0) <= 0:
        return 0.0

    voiced = float(metrics.get("voiced_ratio", 0.0) or 0.0)
    nps = float(metrics.get("notes_per_sec", 0.0) or 0.0)
    med_ms = float(metrics.get("median_note_dur_ms", 0.0) or 0.0)
    frag = float(metrics.get("fragmentation_lt_80ms", 0.0) or 0.0)

    # normalize median duration: 0ms..200ms -> 0..1 (cap)
    dur_term = max(0.0, min(1.0, med_ms / 200.0))

    # note rate penalty (prefer <= ~12 notes/sec)
    rate_term = 1.0 - max(0.0, min(1.0, (nps - 6.0) / 12.0))  # 6->1.0, 18->0.0

    # fragmentation penalty
    frag_term = 1.0 - max(0.0, min(1.0, frag))

    score = 0.45 * voiced + 0.25 * dur_term + 0.20 * rate_term + 0.10 * frag_term
    return float(max(0.0, min(1.0, score)))


def _candidate_order(routed_mode: str) -> List[str]:
    """
    Deterministic fallback chain.
    (We don’t run everything; we run until we find an accepted candidate.)

    NOTE: These mode ids must match your routing + Stage B logic.
    """
    all_modes = [
        "e2e_onsets_frames",
        "e2e_basic_pitch",
        "classic_piano_poly",
        "classic_song",
        "classic_melody",
    ]
    routed_mode = routed_mode if routed_mode in all_modes else "classic_melody"
    rest = [m for m in all_modes if m != routed_mode]
    return [routed_mode] + rest


def transcribe(
    audio_path: str,
    config: Optional[PipelineConfig] = None,
    pipeline_logger: Optional[PipelineLogger] = None,
    device: str = "cpu",
    *,
    # ---- NEW: explicit caller overrides (fixes L6 debug runner relying on "auto") ----
    requested_mode: Optional[str] = None,
    requested_profile: Optional[str] = None,
    requested_separation_mode: Optional[str] = None,
) -> TranscriptionResult:
    """
    High-level transcription entry point with unified quality gate.

    Why the new args:
    - Some benchmarks/debug runners generate synthetic audio with weak/unknown metadata.
      In those cases, auto-routing can pick a mono-biased mode and tank L6 results.
    - Callers (like tools/l6_debug_runner.py) can now force the intended mode explicitly:
        transcribe(..., requested_mode="classic_song")
    """
    if config is None:
        config = PipelineConfig()
    if pipeline_logger is None:
        pipeline_logger = PipelineLogger()

    t0 = time.time()

    # ---------------- Stage A ----------------
    stage_a_out = load_and_preprocess(audio_path, config, pipeline_logger=pipeline_logger)
    duration_sec = float(getattr(getattr(stage_a_out, "meta", None), "duration_sec", 0.0) or 0.0)

    # Resolve caller intent (defaults preserved)
    cfg_mode = getattr(config.stage_b, "transcription_mode", "auto")
    req_mode = str(requested_mode) if requested_mode is not None else str(cfg_mode)

    meta_profile = str(getattr(getattr(stage_a_out, "meta", None), "instrument", None) or "unknown")
    req_profile = str(requested_profile) if requested_profile is not None else meta_profile

    # ---------------- Routing (Stage B trace only) ----------------
    base_trace = compute_decision_trace(
        stage_a_out,
        config,
        requested_mode=req_mode,
        requested_profile=req_profile,
        requested_separation_mode=requested_separation_mode,
        pipeline_logger=pipeline_logger,
    )

    # If caller forced a specific mode (not auto), respect it for the candidate chain start.
    if req_mode and req_mode != "auto":
        routed_mode = req_mode
    else:
        routed_mode = str(base_trace.get("resolved", {}).get("transcription_mode", "classic_melody"))

    candidate_ids = _candidate_order(routed_mode)

    # ---------------- Unified Quality Gate ----------------
    qcfg = getattr(config, "quality_gate", None) or {}
    q_enabled = bool(qcfg.get("enabled", True))
    q_threshold = float(qcfg.get("threshold", 0.45))
    q_max_candidates = int(qcfg.get("max_candidates", 3))

    candidates: List[Dict[str, Any]] = []
    fallbacks_triggered: List[str] = []

    best: Optional[Tuple[str, float, AnalysisData]] = None

    # Mix audio for E2E paths
    mix_audio = None
    try:
        mix_audio = stage_a_out.stems["mix"].audio
    except Exception:
        mix_audio = None

    for idx, cand_id in enumerate(candidate_ids):
        if idx >= q_max_candidates:
            break

        cand_decision = "evaluated"
        cand_score = 0.0
        cand_metrics: Dict[str, Any] = {}
        cand_analysis: Optional[AnalysisData] = None
        cand_trace: Dict[str, Any] = {}

        try:
            cand_cfg = copy.deepcopy(config)

            # Force requested mode for this candidate (decision trace must reflect it)
            cand_cfg.stage_b.transcription_mode = cand_id

            if cand_id == "e2e_onsets_frames":
                enabled = bool(getattr(cand_cfg.stage_b, "onsets_and_frames", {}).get("enabled", False))
                if not enabled:
                    raise RuntimeError("onsets_frames_disabled")

                if mix_audio is None:
                    raise RuntimeError("missing_mix_audio")

                notes = transcribe_onsets_frames(
                    mix_audio,
                    int(getattr(stage_a_out.meta, "sample_rate", 44100)),
                    cand_cfg.stage_b.onsets_and_frames,
                    device=device,
                )

                cand_trace = compute_decision_trace(
                    stage_a_out,
                    cand_cfg,
                    requested_mode="e2e_onsets_frames",
                    requested_profile=req_profile,
                    requested_separation_mode=requested_separation_mode,
                    pipeline_logger=pipeline_logger,
                )

                cand_analysis = AnalysisData(
                    meta=stage_a_out.meta,
                    timeline=[],
                    stem_timelines={},
                    notes=list(notes),
                    notes_before_quantization=list(notes),
                    chords=[],
                    diagnostics={"decision_trace": cand_trace},
                )

            else:
                # Classic + BasicPitch both go through Stage B -> Stage C
                stage_b_out = extract_features(
                    stage_a_out, config=cand_cfg, pipeline_logger=pipeline_logger, device=device
                )
                cand_trace = (
                    dict((stage_b_out.diagnostics or {}).get("decision_trace", {}))
                    if hasattr(stage_b_out, "diagnostics")
                    else {}
                )

                # Convert StageBOutput to AnalysisData for Stage C
                cand_analysis = AnalysisData(
                    meta=stage_b_out.meta,
                    timeline=stage_b_out.timeline,
                    stem_timelines=stage_b_out.stem_timelines,
                    diagnostics=stage_b_out.diagnostics,
                    precalculated_notes=stage_b_out.precalculated_notes,
                )

                # Stage C (updates cand_analysis.notes in-place)
                apply_theory(cand_analysis, cand_cfg)

                # Ensure decision trace is visible at top-level too
                try:
                    cand_analysis.diagnostics = cand_analysis.diagnostics or {}
                    cand_analysis.diagnostics["decision_trace"] = cand_trace
                except Exception:
                    pass

            # Score candidate
            notes_raw = list(cand_analysis.notes_before_quantization or cand_analysis.notes or [])
            timeline_src = cand_analysis.timeline if (cand_analysis.timeline and len(cand_analysis.timeline) > 0) else None
            cand_metrics = _quality_metrics(notes_raw, duration_sec, timeline_source=timeline_src)
            cand_score = _quality_score(cand_metrics)

            accepted = (not q_enabled) or (
                int(cand_metrics.get("note_count", 0)) > 0 and cand_score >= q_threshold
            )
            cand_decision = "accepted" if accepted else "rejected"
            cand_reason = "ok" if accepted else "below_threshold"

            # Track best even if rejected
            if best is None or cand_score > best[1]:
                best = (cand_id, cand_score, cand_analysis)

            candidates.append(
                {
                    "candidate_id": cand_id,
                    "score": float(cand_score),
                    "metrics": dict(cand_metrics),
                    "decision": cand_decision,
                    "reason": cand_reason,
                }
            )

            if accepted:
                # stop at first accepted (deterministic fallback chain)
                break

            fallbacks_triggered.append(f"fallback_from_{cand_id}")

        except Exception as e:
            fallbacks_triggered.append(f"error_{cand_id}")
            candidates.append(
                {
                    "candidate_id": cand_id,
                    "score": 0.0,
                    "metrics": {
                        "voiced_ratio": 0.0,
                        "note_count": 0,
                        "notes_per_sec": 0.0,
                        "median_note_dur_ms": 0.0,
                        "fragmentation_lt_80ms": 0.0,
                    },
                    "decision": "rejected",
                    "reason": f"error:{type(e).__name__}",
                }
            )

    if best is None:
        # ultimate fallback: empty analysis
        analysis_data = AnalysisData(
            meta=stage_a_out.meta,
            timeline=[],
            stem_timelines={},
            notes=[],
            notes_before_quantization=[],
            chords=[],
            diagnostics={},
        )
        selected_id = "none"
        selected_score = 0.0
    else:
        selected_id, selected_score, analysis_data = best

    # Mark selected in candidate trace
    for c in candidates:
        if c.get("candidate_id") == selected_id:
            c["decision"] = "selected"

    analysis_data.diagnostics = analysis_data.diagnostics or {}
    analysis_data.diagnostics["quality_gate"] = {
        "enabled": bool(q_enabled),
        "threshold": float(q_threshold),
        "candidates": list(candidates),
        "selected_candidate_id": str(selected_id),
        "fallbacks_triggered": list(fallbacks_triggered),
    }

    # If Stage B trace wasn’t set (E2E path), keep base_trace
    if "decision_trace" not in analysis_data.diagnostics:
        analysis_data.diagnostics["decision_trace"] = base_trace

    # ---------------- Stage D ----------------
    tr = quantize_and_render(analysis_data.notes, analysis_data, config, pipeline_logger=pipeline_logger)
    musicxml_str = tr.musicxml
    midi_bytes = tr.midi_bytes

    analysis_data.diagnostics["timing"] = {
        "total_sec": float(time.time() - t0),
        "selected_candidate_score": float(selected_score),
    }

    return TranscriptionResult(musicxml=musicxml_str, analysis_data=analysis_data, midi_bytes=midi_bytes)
