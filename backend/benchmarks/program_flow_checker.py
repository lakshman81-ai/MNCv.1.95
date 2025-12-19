# backend/benchmarks/program_flow_checker.py
# Purpose: "Program Flow Checker" — runs synthetic benchmark cases and writes a JSON report that
# includes (a) accuracy-ish metrics and (b) which major pipeline branches executed.
#
# Philosophy:
# - We do NOT attempt full Python branch coverage (that would require coverage tooling).
# - Instead we track *pipeline-relevant* decision points: which detectors ran, polyphony appeared,
#   beat grid existed, etc. using timings/profiling + output structures.
#
# Usage:
#   python -m backend.benchmarks.program_flow_checker --levels L0 L1 L2 L3
#
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.benchmarks.benchmark_runner import (
    AudioType,
    make_config,
    run_pipeline_on_audio,
)


# -----------------------------
# Small utilities
# -----------------------------

def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _midi_to_hz(midi: int) -> float:
    return float(440.0 * (2.0 ** ((midi - 69) / 12.0)))


def _adsr_env(n: int, sr: int, attack_s: float = 0.01, release_s: float = 0.03) -> np.ndarray:
    a = int(max(1, round(attack_s * sr)))
    r = int(max(1, round(release_s * sr)))
    env = np.ones(n, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _tone(midi: int, dur_s: float, sr: int, kind: str = "sine") -> np.ndarray:
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    f = _midi_to_hz(midi)
    if kind == "saw":
        y = 2.0 * (t * f - np.floor(0.5 + t * f))
    else:
        y = np.sin(2.0 * np.pi * f * t)
    y = y.astype(np.float32) * _adsr_env(n, sr)
    return y


def _click(dur_s: float, sr: int) -> np.ndarray:
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    y = np.sin(2.0 * np.pi * 3000.0 * t) * np.exp(-t / 0.01)
    return y.astype(np.float32)


def _mix(chunks: List[Tuple[float, np.ndarray]], total_dur_s: float, sr: int) -> np.ndarray:
    total_n = int(round(total_dur_s * sr))
    y = np.zeros(total_n, dtype=np.float32)
    for start_s, seg in chunks:
        i0 = int(round(start_s * sr))
        i1 = min(total_n, i0 + len(seg))
        if i0 < total_n:
            y[i0:i1] += seg[: i1 - i0]
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 1e-6:
        y = y / max(1.0, peak) * 0.8
    return y


@dataclass(frozen=True)
class GTNote:
    start_sec: float
    end_sec: float
    midi: int


@dataclass(frozen=True)
class CaseDef:
    case_id: str
    audio_type: AudioType
    sr: int
    audio: np.ndarray
    gt_notes: List[GTNote]
    expected: Dict[str, Any]


def _simple_note_metrics(pred_notes: Sequence[Any], gt_notes: Sequence[GTNote], tol_onset_s: float = 0.08) -> Dict[str, Any]:
    preds: List[GTNote] = []
    for n in pred_notes:
        try:
            preds.append(GTNote(float(getattr(n, "start_sec")), float(getattr(n, "end_sec")), int(getattr(n, "midi"))))
        except Exception:
            continue

    gt = list(gt_notes)
    matched_gt = set()
    matched_pred = set()
    onset_errors: List[float] = []

    for pi, p in enumerate(preds):
        best = None
        best_err = 1e9
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            if p.midi != g.midi:
                continue
            err = abs(p.start_sec - g.start_sec)
            if err <= tol_onset_s and err < best_err:
                best = gi
                best_err = err
        if best is not None:
            matched_pred.add(pi)
            matched_gt.add(best)
            onset_errors.append(best_err)

    tp = len(matched_pred)
    fp = max(0, len(preds) - tp)
    fn = max(0, len(gt) - tp)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    onset_mae_ms = (float(np.mean(onset_errors)) * 1000.0) if onset_errors else None

    return {
        "pred_count": len(preds),
        "gt_count": len(gt),
        "note_precision": prec,
        "note_recall": rec,
        "note_f1": f1,
        "onset_mae_ms": onset_mae_ms,
    }


def _extract_feature_flags(res: Dict[str, Any]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}

    timings = res.get("timings") or {}
    dets = timings.get("detectors_run", [])
    flags["detectors_run"] = dets
    flags["has_crepe"] = any("crepe" in str(d).lower() for d in dets)
    flags["has_yin"] = any("yin" in str(d).lower() for d in dets)
    flags["has_swiftf0"] = any("swift" in str(d).lower() for d in dets)

    notes = res.get("notes") or []
    voices = []
    for n in notes:
        v = getattr(n, "voice", None)
        if v is not None:
            try:
                voices.append(int(v))
            except Exception:
                pass
    flags["max_voice_id"] = max(voices) if voices else None

    overlaps = 0
    try:
        sorted_notes = sorted(notes, key=lambda n: (float(n.start_sec), float(n.end_sec)))
        last_end = -1.0
        for n in sorted_notes:
            s = float(n.start_sec)
            e = float(n.end_sec)
            if s < last_end - 1e-6:
                overlaps += 1
            last_end = max(last_end, e)
    except Exception:
        overlaps = 0
    flags["note_overlaps_count"] = overlaps
    flags["polyphonic_proxy"] = bool((flags["max_voice_id"] and flags["max_voice_id"] > 1) or overlaps > 0)

    stage_a = timings.get("stage_a", {}) if isinstance(timings.get("stage_a", {}), dict) else {}
    beat_count = stage_a.get("beat_count")
    flags["beat_grid_proxy"] = bool(beat_count and int(beat_count) > 0) if beat_count is not None else None

    prof = res.get("profiling") or {}
    events = prof.get("events") if isinstance(prof, dict) else None
    if isinstance(events, list):
        names = [e.get("name") for e in events if isinstance(e, dict)]
        flags["event_names_sample"] = sorted({n for n in names if isinstance(n, str)})[:80]
    else:
        flags["event_names_sample"] = []

    return flags


def build_cases() -> Dict[str, List[CaseDef]]:
    sr = 22050
    cases: Dict[str, List[CaseDef]] = {}

    # L0: primitives
    dur = 10.0
    cases["L0"] = [
        CaseDef(
            case_id="sine_440",
            audio_type=AudioType.MONOPHONIC,
            sr=sr,
            audio=_tone(69, dur, sr, "sine"),
            gt_notes=[GTNote(0.0, dur, 69)],
            expected={"note_f1_min": 0.98},
        ),
        CaseDef(
            case_id="saw_440",
            audio_type=AudioType.MONOPHONIC,
            sr=sr,
            audio=_tone(69, dur, sr, "saw"),
            gt_notes=[GTNote(0.0, dur, 69)],
            expected={"note_f1_min": 0.90},
        ),
    ]

    # L1: mono scale + transient stress
    note_dur = 0.6
    gap = 0.08
    mids = [60, 62, 64, 65, 67, 69, 71, 72]  # C4..C5
    t = 0.0
    chunks = []
    gt = []
    for m in mids:
        seg = _tone(m, note_dur, sr, "sine")
        chunks.append((t, seg))
        gt.append(GTNote(t, t + note_dur, m))
        t += note_dur + gap
    total = t
    y_clean = _mix(chunks, total, sr)
    y_clicks = _mix(chunks + [(g.start_sec, _click(0.03, sr)) for g in gt], total, sr)

    cases["L1"] = [
        CaseDef("mono_scale_clean", AudioType.MONOPHONIC, sr, y_clean, gt, {"note_f1_min": 0.70}),
        CaseDef("mono_scale_with_clicks", AudioType.MONOPHONIC, sr, y_clicks, gt, {"note_f1_min": 0.60}),
    ]

    # L2: 2-voice melody + bass
    melody = [72, 74, 76, 77, 79, 77, 76, 74]
    bass = [48, 43, 45, 47]
    chunks = []
    gt = []
    t = 0.0
    for m in melody:
        seg = _tone(m, 0.5, sr, "sine")
        chunks.append((t, seg))
        gt.append(GTNote(t, t + 0.5, m))
        t += 0.55
    bt = 0.0
    for b in bass:
        seg = _tone(b, 1.4, sr, "sine")
        chunks.append((bt, seg))
        gt.append(GTNote(bt, bt + 1.4, b))
        bt += 1.4

    total = max(t, bt + 0.5)
    cases["L2"] = [
        CaseDef("melody_bass_2voice", AudioType.POLYPHONIC_DOMINANT, sr, _mix(chunks, total, sr), gt, {"note_f1_min": 0.35}),
    ]

    # L3: triad block chords
    chords = [([60, 64, 67], 0.8), ([65, 69, 72], 0.8), ([67, 71, 74], 0.8), ([60, 64, 67], 1.0)]
    chunks = []
    gt = []
    t = 0.0
    for mids, dur_s in chords:
        for m in mids:
            chunks.append((t, _tone(m, dur_s, sr, "sine")))
            gt.append(GTNote(t, t + dur_s, m))
        t += dur_s + 0.1
    cases["L3"] = [
        CaseDef("block_chords_triad", AudioType.POLYPHONIC, sr, _mix(chunks, t, sr), gt, {"note_f1_min": 0.25}),
    ]
    return cases


def _expectation_pass(metrics: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    passed = True
    details: List[Dict[str, Any]] = []
    if "note_f1_min" in expected:
        floor = float(expected["note_f1_min"])
        val = float(metrics.get("note_f1") or 0.0)
        ok = val >= floor
        passed = passed and ok
        details.append({"metric": "note_f1", "op": ">=", "target": floor, "value": val, "ok": ok})
    return passed, details


def run_flow_check(levels: Sequence[str], output_dir: Path) -> Dict[str, Any]:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / f"flow_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    ladder = build_cases()
    report: Dict[str, Any] = {"created_utc": stamp, "levels": list(levels), "cases": [], "coverage": {}}

    feature_counts: Dict[str, int] = {
        "has_yin": 0, "has_swiftf0": 0, "has_crepe": 0,
        "polyphonic_proxy": 0, "beat_grid_proxy_true": 0,
    }

    for lvl in levels:
        for case in ladder.get(lvl, []):
            config = make_config(audio_type=case.audio_type)
            res = run_pipeline_on_audio(audio=case.audio, sr=case.sr, config=config, audio_type=case.audio_type)

            flags = _extract_feature_flags(res)
            metrics = _simple_note_metrics(res.get("notes") or [], case.gt_notes)

            passed, exp_details = _expectation_pass(metrics, case.expected)

            for k in ("has_yin", "has_swiftf0", "has_crepe", "polyphonic_proxy"):
                if flags.get(k):
                    feature_counts[k] += 1
            if flags.get("beat_grid_proxy") is True:
                feature_counts["beat_grid_proxy_true"] += 1

            row = {
                "level": lvl, "case_id": case.case_id, "audio_type": str(case.audio_type),
                "expected": case.expected, "metrics": metrics,
                "expectations": {"passed": passed, "details": exp_details},
                "flags": flags,
            }
            report["cases"].append(row)
            _write_json(report_dir / f"{lvl}_{case.case_id}_flow.json", row)

    report["coverage"] = feature_counts
    _write_json(report_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Program Flow Checker: synthetic cases + flow/accuracy report.")
    p.add_argument("--levels", nargs="*", default=["L0", "L1", "L2", "L3"], help="Levels to run.")
    p.add_argument("--output-dir", default=str(Path("results") / "flow_checks"), help="Where to write reports.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = run_flow_check(levels=list(args.levels), output_dir=Path(args.output_dir))
    failed = [c for c in report.get("cases", []) if not (c.get("expectations", {}) or {}).get("passed", True)]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
