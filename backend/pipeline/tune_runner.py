"""
BENCH_TUNE_MODE runner.

Runs benchmark levels (default: L4, L5.1, L5.2) and auto-tunes config when note_f1 < threshold.

Design goals:
- No regex editing of config.py
- Uses in-process execution of benchmark_runner with monkeypatched PIANO_61KEY_CONFIG
- Isolated output per level/iteration under results/tuning/<date>/...
- Robust metric discovery (metrics.json or fallback)
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import io
import json
import os
import runpy
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Small utilities
# -----------------------------

def now_stamp() -> Tuple[str, str]:
    dt = datetime.now()
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


# -----------------------------
# Dotted-path override helpers
# -----------------------------

def _get_path(root: Any, path: str) -> Tuple[bool, Any]:
    """
    Return (exists, value) for dotted path.
    Supports objects (attrs) and dicts (keys).
    """
    cur = root
    for key in path.split("."):
        if cur is None:
            return False, None
        if isinstance(cur, dict):
            if key not in cur:
                return False, None
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return False, None
            cur = getattr(cur, key)
    return True, cur


def _set_path(root: Any, path: str, value: Any) -> bool:
    """
    Set dotted path if it exists (or if intermediate container is dict).
    Returns True if set, False if path can't be resolved safely.
    """
    parts = path.split(".")
    cur = root
    for i, key in enumerate(parts[:-1]):
        if cur is None:
            return False
        if isinstance(cur, dict):
            if key not in cur or cur[key] is None:
                # allow creation for dict intermediate
                cur[key] = {}
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return False
            cur = getattr(cur, key)

    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
        return True
    if hasattr(cur, last):
        setattr(cur, last, value)
        return True
    return False


def apply_overrides(cfg: Any, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply overrides that can be set; return dict of actually-applied overrides.
    """
    applied: Dict[str, Any] = {}
    for k, v in overrides.items():
        ok = _set_path(cfg, k, v)
        if ok:
            applied[k] = v
    return applied


# -----------------------------
# Metrics discovery / extraction
# -----------------------------

def _find_metrics_files(outdir: Path) -> List[Path]:
    if not outdir.exists():
        return []
    hits: List[Path] = []
    direct = outdir / "metrics.json"
    if direct.exists():
        hits.append(direct)

    for p in outdir.rglob("metrics.json"):
        if p not in hits:
            hits.append(p)

    # Sort by mtime newest first
    hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return hits


def _extract_note_f1(obj: Any) -> Optional[float]:
    """
    Search nested dicts/lists for a key 'note_f1' containing a number.
    """
    if isinstance(obj, dict):
        if "note_f1" in obj and is_number(obj["note_f1"]):
            return float(obj["note_f1"])
        for v in obj.values():
            got = _extract_note_f1(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = _extract_note_f1(it)
            if got is not None:
                return got
    return None


def load_metrics(outdir: Path, level: str) -> Dict[str, Any]:
    """
    Load best-effort metrics payload.
    """
    metrics: Dict[str, Any] = {
        "level": level,
        "note_f1": 0.0,
        "missing_metric": True,
        "metrics_path": None,
        "raw": None,
    }

    candidates = _find_metrics_files(outdir)
    for mp in candidates:
        data = read_json(mp)
        if data is None:
            continue

        # Try direct level indexing first
        note_f1: Optional[float] = None
        if isinstance(data, dict) and level in data and isinstance(data[level], dict):
            if "note_f1" in data[level] and is_number(data[level]["note_f1"]):
                note_f1 = float(data[level]["note_f1"])

        if note_f1 is None:
            note_f1 = _extract_note_f1(data)

        if note_f1 is not None:
            metrics["note_f1"] = float(note_f1)
            metrics["missing_metric"] = False
            metrics["metrics_path"] = str(mp)
            metrics["raw"] = data
            return metrics

        # keep last seen raw for debugging
        metrics["metrics_path"] = str(mp)
        metrics["raw"] = data

    # fallback: summary.csv last row if present
    summary = outdir / "summary.csv"
    if summary.exists():
        try:
            import csv
            rows = []
            with summary.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                for k in ("note_f1", "f1", "Note F1"):
                    if k in last and is_number(last[k]):
                        metrics["note_f1"] = float(last[k])
                        metrics["missing_metric"] = False
                        metrics["metrics_path"] = str(summary)
                        metrics["raw"] = {"summary_last_row": last}
                        return metrics
        except Exception:
            pass

    return metrics


def extract_symptoms(metrics_raw: Any) -> Dict[str, Any]:
    """
    Try to pull helpful knobs from metrics if they exist.
    Works even if keys are missing (returns {} defaults).
    """
    out: Dict[str, Any] = {}

    def find_key(obj: Any, key: str) -> Optional[Any]:
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                got = find_key(v, key)
                if got is not None:
                    return got
        elif isinstance(obj, list):
            for it in obj:
                got = find_key(it, key)
                if got is not None:
                    return got
        return None

    for k in ("fragmentation_score", "note_count_per_10s", "median_note_len_ms",
              "octave_jump_rate", "voiced_ratio", "note_count"):
        v = find_key(metrics_raw, k)
        if v is not None and (is_number(v) or isinstance(v, (int, float))):
            out[k] = float(v)

    return out


# -----------------------------
# Candidate override generation
# -----------------------------

def propose_candidate_overrides(
    base_overrides: Dict[str, Any],
    symptoms: Dict[str, Any],
    level: str,
    iter_idx: int,
) -> List[Dict[str, Any]]:
    """
    Guided search: generate a small set of candidate override dicts.
    Returns ordered candidates (best-first heuristically).
    """
    cands: List[Dict[str, Any]] = []

    frag = float(symptoms.get("fragmentation_score", 0.0))
    nps10 = float(symptoms.get("note_count_per_10s", 0.0))
    med_ms = float(symptoms.get("median_note_len_ms", 0.0))
    octave_jumps = float(symptoms.get("octave_jump_rate", 0.0))

    # Heuristic flags
    fragmentation_high = (frag >= 0.30) or (med_ms > 0.0 and med_ms < 90.0) or (nps10 > 0.0 and nps10 > 40.0)
    recall_low = (nps10 > 0.0 and nps10 < 6.0)
    octave_high = (octave_jumps >= 0.25)

    def bump(d: Dict[str, Any], path: str, delta: float, floor: Optional[float] = None, ceil: Optional[float] = None):
        cur = d.get(path, None)
        if cur is None or not is_number(cur):
            return
        v = float(cur) + float(delta)
        if floor is not None:
            v = max(float(floor), v)
        if ceil is not None:
            v = min(float(ceil), v)
        d[path] = v

    # Start from base
    base = dict(base_overrides)

    # Ensure baseline has reasonable defaults if not already present
    base.setdefault("stage_b.voice_tracking.smoothing", 0.0)
    base.setdefault("stage_c.confidence_threshold", None)  # leave None unless we can infer
    base.setdefault("stage_c.gap_tolerance_s", None)
    base.setdefault("stage_c.pitch_tolerance_cents", None)
    base.setdefault("stage_c.min_note_duration_ms_poly", None)
    base.setdefault("stage_c.polyphony_filter.mode", None)
    base.setdefault("stage_b.polyphonic_peeling.iss_adaptive", None)
    base.setdefault("stage_c.segmentation_method.transition_penalty", None)

    # Candidate 1: Stabilize fragmentation (most common for L5.*)
    if fragmentation_high:
        c1 = dict(base_overrides)
        # poly min duration bump (if present)
        if "stage_c.min_note_duration_ms_poly" in c1 and is_number(c1["stage_c.min_note_duration_ms_poly"]):
            bump(c1, "stage_c.min_note_duration_ms_poly", +20.0, floor=40.0, ceil=140.0)
        else:
            c1["stage_c.min_note_duration_ms_poly"] = 70.0

        # confidence up a touch
        if "stage_c.confidence_threshold" in c1 and is_number(c1["stage_c.confidence_threshold"]):
            bump(c1, "stage_c.confidence_threshold", +0.03, floor=0.05, ceil=0.60)
        else:
            c1["stage_c.confidence_threshold"] = 0.18

        # gap tolerance up a touch
        if "stage_c.gap_tolerance_s" in c1 and is_number(c1["stage_c.gap_tolerance_s"]):
            bump(c1, "stage_c.gap_tolerance_s", +0.02, floor=0.01, ceil=0.20)
        else:
            c1["stage_c.gap_tolerance_s"] = 0.07

        # smoothing up
        if "stage_b.voice_tracking.smoothing" in c1 and is_number(c1["stage_b.voice_tracking.smoothing"]):
            bump(c1, "stage_b.voice_tracking.smoothing", +0.10, floor=0.0, ceil=0.95)
        else:
            c1["stage_b.voice_tracking.smoothing"] = 0.4

        # poly filter: try decomposed melody first for L5.*
        if level.startswith("L5"):
            c1["stage_c.polyphony_filter.mode"] = "decomposed_melody"

        cands.append(c1)

    # Candidate 2: Improve recall (if too sparse)
    if recall_low:
        c2 = dict(base_overrides)
        # confidence down a touch
        if "stage_c.confidence_threshold" in c2 and is_number(c2["stage_c.confidence_threshold"]):
            bump(c2, "stage_c.confidence_threshold", -0.03, floor=0.02, ceil=0.60)
        else:
            c2["stage_c.confidence_threshold"] = 0.10

        # poly min duration down a touch
        if "stage_c.min_note_duration_ms_poly" in c2 and is_number(c2["stage_c.min_note_duration_ms_poly"]):
            bump(c2, "stage_c.min_note_duration_ms_poly", -10.0, floor=20.0, ceil=140.0)
        else:
            c2["stage_c.min_note_duration_ms_poly"] = 50.0

        # pitch tolerance up
        if "stage_c.pitch_tolerance_cents" in c2 and is_number(c2["stage_c.pitch_tolerance_cents"]):
            bump(c2, "stage_c.pitch_tolerance_cents", +10.0, floor=20.0, ceil=120.0)
        else:
            c2["stage_c.pitch_tolerance_cents"] = 60.0

        cands.append(c2)

    # Candidate 3: Reduce octave chaos
    if octave_high:
        c3 = dict(base_overrides)
        if "stage_b.voice_tracking.smoothing" in c3 and is_number(c3["stage_b.voice_tracking.smoothing"]):
            bump(c3, "stage_b.voice_tracking.smoothing", +0.15, floor=0.0, ceil=0.95)
        else:
            c3["stage_b.voice_tracking.smoothing"] = 0.55
        cands.append(c3)

    # Candidate 4+: Explore polyphony filter modes (L5.* only)
    if level.startswith("L5"):
        for mode in ("skyline_top_voice", "process_all", "decomposed_melody"):
            c = dict(base_overrides)
            c["stage_c.polyphony_filter.mode"] = mode
            cands.append(c)

    # If we generated nothing, make a gentle default exploration
    if not cands:
        c = dict(base_overrides)
        c["stage_b.voice_tracking.smoothing"] = float(c.get("stage_b.voice_tracking.smoothing", 0.0) or 0.0) + 0.1
        cands.append(c)

    # Deduplicate candidates (by JSON repr)
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for c in cands:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    # Keep candidates bounded
    return uniq[:8]


# -----------------------------
# Benchmark runner execution (in-process)
# -----------------------------

def _clear_modules(prefixes: Tuple[str, ...]) -> None:
    kill = [m for m in list(sys.modules.keys()) if any(m == p or m.startswith(p + ".") for p in prefixes)]
    for m in kill:
        sys.modules.pop(m, None)


def run_benchmark_inprocess(
    level: str,
    outdir: Path,
    overrides: Dict[str, Any],
    benchmark_module: str = "backend.benchmarks.benchmark_runner",
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run benchmark_runner as if `python -m ...` but in-process, with monkeypatched config.

    Returns dict containing returncode, stdout_path, stderr_path, applied_overrides.
    """
    extra_args = extra_args or []
    ensure_dir(outdir)

    stdout_path = outdir / "benchmark.stdout.txt"
    stderr_path = outdir / "benchmark.stderr.txt"

    # Import config and monkeypatch PIANO_61KEY_CONFIG
    import backend.pipeline.config as cfgmod  # type: ignore

    original_cfg = cfgmod.PIANO_61KEY_CONFIG
    patched_cfg = copy.deepcopy(original_cfg)
    applied = apply_overrides(patched_cfg, overrides)

    cfgmod.PIANO_61KEY_CONFIG = patched_cfg  # monkeypatch

    # Clear modules that might have cached old config references
    _clear_modules((
        "backend.benchmarks.benchmark_runner",
        "backend.pipeline.transcribe",
        "backend.pipeline.stage_a",
        "backend.pipeline.stage_b",
        "backend.pipeline.stage_c",
        "backend.pipeline.stage_d",
        "backend.pipeline.neural_transcription",
    ))

    # Run the module with redirected stdout/stderr
    argv_old = sys.argv[:]
    # FORCE preset=piano_61key so the tuning affects L4/L5.2
    sys.argv = [sys.executable, "-m", benchmark_module, "--level", level, "--output", str(outdir), "--preset", "piano_61key"] + extra_args

    rc = 0
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            try:
                runpy.run_module(benchmark_module, run_name="__main__")
            except SystemExit as e:
                # benchmark_runner may call sys.exit()
                rc = int(e.code) if isinstance(e.code, int) else 1
    except Exception:
        rc = 1
        err_buf.write("\n=== EXCEPTION ===\n")
        err_buf.write(traceback.format_exc())
    finally:
        stdout_path.write_text(out_buf.getvalue(), encoding="utf-8")
        stderr_path.write_text(err_buf.getvalue(), encoding="utf-8")
        sys.argv = argv_old

        # Restore original config object
        cfgmod.PIANO_61KEY_CONFIG = original_cfg

    return {
        "returncode": rc,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "applied_overrides": applied,
    }


# -----------------------------
# Main tuning loop
# -----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="BENCH_TUNE_MODE: benchmark + autotune runner")
    p.add_argument("--levels", nargs="*", default=["L4", "L5.1", "L5.2"])
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument("--output-root", default=str(Path("results") / "tuning"))
    p.add_argument("--benchmark-module", default="backend.benchmarks.benchmark_runner")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra arg to pass to benchmark_runner (repeatable).")
    args = p.parse_args()

    date_s, time_s = now_stamp()
    root = ensure_dir(Path(args.output_root) / date_s / f"run_{time_s}")

    manifest = {
        "date": date_s,
        "time": time_s,
        "levels": args.levels,
        "threshold": args.threshold,
        "max_iters": args.max_iters,
        "benchmark_module": args.benchmark_module,
        "cwd": str(Path.cwd()),
        "python": sys.version,
    }
    write_json(root / "manifest.json", manifest)

    global_best_overrides: Dict[str, Any] = {}
    report: Dict[str, Any] = {"manifest": manifest, "levels": []}

    for level in args.levels:
        level_dir = ensure_dir(root / level.replace("/", "_"))
        level_report: Dict[str, Any] = {"level": level, "baseline": None, "best": None, "iterations": []}

        # --- Baseline run ---
        base_out = ensure_dir(level_dir / "iter_00_baseline")
        run_info = run_benchmark_inprocess(
            level=level,
            outdir=base_out,
            overrides=global_best_overrides,
            benchmark_module=args.benchmark_module,
            extra_args=list(args.extra_arg) if args.extra_arg else None,
        )
        metrics = load_metrics(base_out, level)
        symptoms = extract_symptoms(metrics.get("raw"))

        baseline = {
            "outdir": str(base_out),
            "returncode": run_info["returncode"],
            "note_f1": metrics["note_f1"],
            "missing_metric": metrics["missing_metric"],
            "metrics_path": metrics["metrics_path"],
            "symptoms": symptoms,
            "applied_overrides": run_info["applied_overrides"],
        }
        write_json(base_out / "run_result.json", {"run_info": run_info, "metrics": metrics, "symptoms": symptoms})
        level_report["baseline"] = baseline

        best_overrides = dict(global_best_overrides)
        best_f1 = float(metrics["note_f1"])
        best_iter_dir = str(base_out)

        # --- Tune loop ---
        if best_f1 < float(args.threshold):
            no_improve_streak = 0
            last_best = best_f1

            for it in range(1, int(args.max_iters) + 1):
                iter_dir = ensure_dir(level_dir / f"iter_{it:02d}")
                candidates = propose_candidate_overrides(best_overrides, symptoms, level, it)

                iter_best_local = {"note_f1": -1.0, "overrides": None, "dir": None, "metrics": None, "symptoms": None}

                # Evaluate candidates
                for ci, cand_overrides in enumerate(candidates):
                    cand_dir = ensure_dir(iter_dir / f"cand_{ci:02d}")
                    ri = run_benchmark_inprocess(
                        level=level,
                        outdir=cand_dir,
                        overrides=cand_overrides,
                        benchmark_module=args.benchmark_module,
                        extra_args=list(args.extra_arg) if args.extra_arg else None,
                    )
                    m = load_metrics(cand_dir, level)
                    s = extract_symptoms(m.get("raw"))

                    write_json(cand_dir / "run_result.json", {"run_info": ri, "metrics": m, "symptoms": s, "overrides": cand_overrides})

                    if float(m["note_f1"]) > float(iter_best_local["note_f1"]):
                        iter_best_local = {
                            "note_f1": float(m["note_f1"]),
                            "overrides": cand_overrides,
                            "dir": str(cand_dir),
                            "metrics": m,
                            "symptoms": s,
                        }

                    # Early break if threshold reached
                    if float(m["note_f1"]) >= float(args.threshold):
                        break

                # Adopt best candidate from this iteration
                if iter_best_local["overrides"] is not None:
                    best_overrides = dict(iter_best_local["overrides"])
                    best_f1 = max(best_f1, float(iter_best_local["note_f1"]))
                    best_iter_dir = str(iter_best_local["dir"])
                    symptoms = dict(iter_best_local["symptoms"] or {})

                level_report["iterations"].append({
                    "iter": it,
                    "best_note_f1": float(iter_best_local["note_f1"]),
                    "best_dir": iter_best_local["dir"],
                    "best_overrides": iter_best_local["overrides"],
                })
                write_json(iter_dir / "iter_best.json", level_report["iterations"][-1])

                # Stop if threshold reached
                if best_f1 >= float(args.threshold):
                    break

                # Stagnation stop
                if best_f1 - last_best < 0.01:
                    no_improve_streak += 1
                else:
                    no_improve_streak = 0
                    last_best = best_f1

                if no_improve_streak >= 3:
                    break

        # Finalize level result
        level_report["best"] = {
            "note_f1": float(best_f1),
            "best_dir": best_iter_dir,
            "best_overrides": best_overrides,
        }
        write_json(level_dir / "level_report.json", level_report)
        report["levels"].append(level_report)

        # Carry best overrides forward to next level
        global_best_overrides = dict(best_overrides)

    # Final report + best_overrides.json
    write_json(root / "final_report.json", report)
    write_json(root / "best_overrides.json", global_best_overrides)

    # Return non-zero if any level failed threshold (soft signal)
    any_fail = any((lvl.get("best", {}).get("note_f1", 0.0) < float(args.threshold)) for lvl in report["levels"])
    return 2 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
