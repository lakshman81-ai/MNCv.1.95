"""
BENCH_TUNE_MODE runner.

Runs benchmark levels (default: L4, L5.1, L5.2) and auto-tunes config when note_f1 < threshold.

Design goals:
- No regex editing of config.py
- Uses in-process execution of benchmark_runner with monkeypatched PIANO_61KEY_CONFIG
- Isolated output per level/iteration under results/tuning/<date>/...
- Robust metric discovery (metrics.json or fallback)
- Strict dependency management
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
import subprocess
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Dependency Management
# -----------------------------

REQUIRED_PACKAGES = [
    "numpy", "scipy", "librosa", "soundfile", "torch", "torchaudio",
    "demucs", "crepe", "music21"
]

def check_dependencies(strict: bool = False, auto_install: bool = True) -> None:
    """
    Check for required packages.
    If strict=True:
      - Attempt auto-install if missing.
      - Raise RuntimeError if still missing.
    Else:
      - Log warning if missing.
    """
    missing = []
    for pkg in REQUIRED_PACKAGES:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)

    if not missing:
        return

    print(f"[{'STRICT' if strict else 'WARN'}] Missing packages: {missing}")

    if strict:
        if auto_install:
            print("Attempting auto-install via pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install missing packages {missing}: {e}")

            # Re-verify
            still_missing = []
            for pkg in missing:
                if importlib.util.find_spec(pkg) is None:
                    still_missing.append(pkg)

            if still_missing:
                raise RuntimeError(f"Critical: Packages still missing after install: {still_missing}")
            print("Dependencies installed successfully.")
        else:
            raise RuntimeError(f"Missing required packages: {missing}. Auto-install disabled.")
    else:
        print("Proceeding despite missing dependencies (non-strict mode).")


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


def append_jsonl(path: Path, obj: Any) -> None:
    """Append a single JSON line to a file."""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


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
    STRICT CONSTRAINT: Max 3 parameter changes per candidate vs base_overrides.
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
            # Treat missing as 0.0 or sensible default for purpose of bump if needed,
            # but usually we want to modify existing. If missing, we might assume a default.
            # For this simple tuner, we'll try to find it in base or assume reasonable default
            return
        v = float(cur) + float(delta)
        if floor is not None:
            v = max(float(floor), v)
        if ceil is not None:
            v = min(float(ceil), v)
        d[path] = v

    # --- Baseline defaults for reference (to allow bumping if not in base) ---
    # We populate these in a temp dict to perform logic, but valid candidate
    # MUST contain the values.
    # The 'base' here is strictly what we carry over.

    # Helper to create a candidate with limited changes
    def make_cand():
        # returns a copy of base
        return dict(base_overrides)

    # Candidate 1: Fragmentation A (Durations) - Max 2 knobs
    if fragmentation_high:
        c = make_cand()
        # 1. Min duration
        c.setdefault("stage_c.min_note_duration_ms_poly", 70.0)
        bump(c, "stage_c.min_note_duration_ms_poly", +20.0, floor=40.0, ceil=140.0)

        # 2. Gap tolerance
        c.setdefault("stage_c.gap_tolerance_s", 0.07)
        bump(c, "stage_c.gap_tolerance_s", +0.02, floor=0.01, ceil=0.20)

        cands.append(c)

    # Candidate 2: Fragmentation B (Confidence/Smoothing) - Max 2 knobs
    if fragmentation_high:
        c = make_cand()
        # 1. Confidence
        c.setdefault("stage_c.confidence_threshold", 0.18)
        bump(c, "stage_c.confidence_threshold", +0.03, floor=0.05, ceil=0.60)

        # 2. Smoothing
        c.setdefault("stage_b.voice_tracking.smoothing", 0.4)
        bump(c, "stage_b.voice_tracking.smoothing", +0.10, floor=0.0, ceil=0.95)

        cands.append(c)

    # Candidate 3: Recall (Confidence/PitchTol) - Max 3 knobs
    if recall_low:
        c = make_cand()
        # 1. Confidence
        c.setdefault("stage_c.confidence_threshold", 0.10)
        bump(c, "stage_c.confidence_threshold", -0.03, floor=0.02, ceil=0.60)

        # 2. Min duration
        c.setdefault("stage_c.min_note_duration_ms_poly", 50.0)
        bump(c, "stage_c.min_note_duration_ms_poly", -10.0, floor=20.0, ceil=140.0)

        # 3. Pitch Tolerance
        c.setdefault("stage_c.pitch_tolerance_cents", 60.0)
        bump(c, "stage_c.pitch_tolerance_cents", +10.0, floor=20.0, ceil=120.0)

        cands.append(c)

    # Candidate 4: Octave Stability - Max 1 knob
    if octave_high:
        c = make_cand()
        c.setdefault("stage_b.voice_tracking.smoothing", 0.55)
        bump(c, "stage_b.voice_tracking.smoothing", +0.15, floor=0.0, ceil=0.95)
        cands.append(c)

    # Candidate 5: L5 Poly Modes & Peeling - Max 2 knobs
    if level.startswith("L5"):
        # 1. Poly Mode
        for mode in ("skyline_top_voice", "decomposed_melody"):
            current_mode = base_overrides.get("stage_c.polyphony_filter.mode", None)
            if current_mode != mode:
                c = make_cand()
                c["stage_c.polyphony_filter.mode"] = mode
                cands.append(c)

        # 2. Harmonics
        c = make_cand()
        c.setdefault("stage_b.polyphonic_peeling.max_harmonics", 1)
        bump(c, "stage_b.polyphonic_peeling.max_harmonics", +1, floor=1, ceil=12)
        cands.append(c)

        # 3. Layers
        c = make_cand()
        c.setdefault("stage_b.polyphonic_peeling.max_layers", 4)
        bump(c, "stage_b.polyphonic_peeling.max_layers", +1, floor=1, ceil=8)
        cands.append(c)

    # Candidate 6: Octave Errors (Fmin/Fmax)
    if octave_high:
        c = make_cand()
        # Try adjusting frequency bounds slightly
        c.setdefault("stage_b.detectors.yin.fmin", 60.0)
        c.setdefault("stage_b.detectors.crepe.fmin", 60.0)
        bump(c, "stage_b.detectors.yin.fmin", -10.0, floor=20.0, ceil=100.0)
        bump(c, "stage_b.detectors.crepe.fmin", -10.0, floor=20.0, ceil=100.0)
        cands.append(c)

    # Candidate 7: Onset Sensitivity
    if nps10 > 20.0: # high density
        c = make_cand()
        c.setdefault("stage_a.diff_threshold", 0.5)
        bump(c, "stage_a.diff_threshold", +0.1, floor=0.1, ceil=2.0)
        cands.append(c)

    # Fallback exploration if no specific symptoms triggered
    if not cands:
        c = make_cand()
        c.setdefault("stage_b.voice_tracking.smoothing", 0.0)
        bump(c, "stage_b.voice_tracking.smoothing", +0.1, floor=0.0, ceil=0.95)
        cands.append(c)

    # Deduplicate candidates (by JSON repr)
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for c in cands:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    # Enforce strict 3-knob limit check (just in case)
    final_cands = []
    for c in uniq:
        diff_count = 0
        for k, v in c.items():
            if k not in base_overrides or base_overrides[k] != v:
                diff_count += 1

        # New keys count as changes
        # Also keys removed count (but we don't remove here)

        if diff_count <= 3:
            final_cands.append(c)
        else:
            # If we somehow exceeded, we drop it to be strict.
            pass

    return final_cands[:8]


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
    use_preset: bool = False,
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

    # Construct args
    # Note: --output must be passed.
    cmd_args = [benchmark_module, "--level", level, "--output", str(outdir)]
    if use_preset:
        cmd_args.extend(["--preset", "piano_61key"])

    cmd_args.extend(extra_args)

    sys.argv = cmd_args

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
        "command_args": cmd_args[2:], # capture effective args
    }


# -----------------------------
# Main tuning loop
# -----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="BENCH_TUNE_MODE: benchmark + autotune runner")
    p.add_argument("--levels", nargs="*", default=["L4", "L5.1", "L5.2"])
    p.add_argument("--threshold", type=float, default=0.75, help="Legacy alias for --target-f1")
    p.add_argument("--target-f1", type=float, default=None, help="Target Note F1 score (default: 0.75)")
    p.add_argument("--max-iters", type=int, default=12)
    p.add_argument("--strict-deps", action="store_true", help="Enforce required dependencies and fail if missing")
    p.add_argument("--use-preset", action="store_true", help="Use --preset piano_61key for supported levels")
    p.add_argument("--output-root", default=str(Path("results") / "tuning"))
    p.add_argument("--benchmark-module", default="backend.benchmarks.benchmark_runner")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra arg to pass to benchmark_runner (repeatable).")
    args = p.parse_args()

    # Resolve target F1
    target_f1 = args.target_f1 if args.target_f1 is not None else args.threshold

    # 1. Dependency Preflight
    check_dependencies(strict=args.strict_deps, auto_install=True)

    date_s, time_s = now_stamp()
    root = ensure_dir(Path(args.output_root) / date_s / f"run_{time_s}")
    trials_log_path = root / "tuning_trials.jsonl"

    manifest = {
        "date": date_s,
        "time": time_s,
        "levels": args.levels,
        "target_f1": target_f1,
        "max_iters": args.max_iters,
        "benchmark_module": args.benchmark_module,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "strict_deps": args.strict_deps,
        "use_preset": args.use_preset,
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
            use_preset=args.use_preset,
        )
        metrics = load_metrics(base_out, level)
        symptoms = extract_symptoms(metrics.get("raw"))
        note_f1_val = metrics.get("note_f1", 0.0)

        # Log Baseline Trial
        append_jsonl(trials_log_path, {
            "level": level,
            "iter_idx": 0,
            "params_changed": {}, # Baseline has no changes relative to global best at start
            "score": note_f1_val,
            "pass": note_f1_val >= target_f1,
            "run_dir": str(base_out),
            "timestamp": datetime.now().isoformat(),
            "command": run_info.get("command_args"),
        })

        baseline = {
            "outdir": str(base_out),
            "returncode": run_info["returncode"],
            "note_f1": note_f1_val,
            "missing_metric": metrics["missing_metric"],
            "metrics_path": metrics["metrics_path"],
            "symptoms": symptoms,
            "applied_overrides": run_info["applied_overrides"],
        }
        write_json(base_out / "run_result.json", {"run_info": run_info, "metrics": metrics, "symptoms": symptoms})
        level_report["baseline"] = baseline

        best_overrides = dict(global_best_overrides)
        best_f1 = float(note_f1_val)
        best_iter_dir = str(base_out)

        # --- Tune loop ---
        if best_f1 < target_f1:
            no_improve_streak = 0
            last_best = best_f1

            for it in range(1, int(args.max_iters) + 1):
                iter_dir = ensure_dir(level_dir / f"iter_{it:02d}")
                candidates = propose_candidate_overrides(best_overrides, symptoms, level, it)

                iter_best_local = {"note_f1": -1.0, "overrides": None, "dir": None, "metrics": None, "symptoms": None}

                # Evaluate candidates
                for ci, cand_overrides in enumerate(candidates):
                    cand_dir = ensure_dir(iter_dir / f"cand_{ci:02d}")

                    # Calculate params changed for logging
                    diff = {}
                    for k, v in cand_overrides.items():
                        if k not in best_overrides or best_overrides[k] != v:
                            diff[k] = v

                    ri = run_benchmark_inprocess(
                        level=level,
                        outdir=cand_dir,
                        overrides=cand_overrides,
                        benchmark_module=args.benchmark_module,
                        extra_args=list(args.extra_arg) if args.extra_arg else None,
                        use_preset=args.use_preset,
                    )
                    m = load_metrics(cand_dir, level)
                    s = extract_symptoms(m.get("raw"))
                    score = float(m["note_f1"])

                    # Log Candidate Trial
                    append_jsonl(trials_log_path, {
                        "level": level,
                        "iter_idx": it,
                        "candidate_idx": ci,
                        "params_changed": diff,
                        "score": score,
                        "pass": score >= target_f1,
                        "run_dir": str(cand_dir),
                        "timestamp": datetime.now().isoformat(),
                        "command": ri.get("command_args"),
                    })

                    write_json(cand_dir / "run_result.json", {"run_info": ri, "metrics": m, "symptoms": s, "overrides": cand_overrides})

                    if score > float(iter_best_local["note_f1"]):
                        iter_best_local = {
                            "note_f1": score,
                            "overrides": cand_overrides,
                            "dir": str(cand_dir),
                            "metrics": m,
                            "symptoms": s,
                        }

                    # Early break if threshold reached
                    if score >= target_f1:
                        break

                # Adopt best candidate from this iteration
                if iter_best_local["overrides"] is not None:
                    # Only adopt if it improved or we are exploring?
                    # Greedy: always take the best of this iteration if it's better than nothing.
                    # But we should compare to `best_f1` of the Level.

                    # However, typical hill climbing moves to the best neighbor.
                    # We will update `best_overrides` to `iter_best_local` even if it's not better than absolute best?
                    # No, we should ensure strict improvement or at least non-regression if we want to drift.
                    # But for simple tuning, let's track the absolute best.

                    if iter_best_local["note_f1"] > best_f1:
                        best_overrides = dict(iter_best_local["overrides"])
                        best_f1 = float(iter_best_local["note_f1"])
                        best_iter_dir = str(iter_best_local["dir"])
                        symptoms = dict(iter_best_local["symptoms"] or {})
                    else:
                        # If no candidate improved on global best, do we keep the old best?
                        # Yes. But we might want to take the best of *this* iteration to continue searching?
                        # This script logic: "best_overrides" is the starting point for the next iteration.
                        # If we didn't improve, we stick with the old best_overrides for the next generation.
                        pass

                level_report["iterations"].append({
                    "iter": it,
                    "best_note_f1": float(iter_best_local["note_f1"]),
                    "best_dir": iter_best_local["dir"],
                    "best_overrides": iter_best_local["overrides"],
                })
                write_json(iter_dir / "iter_best.json", level_report["iterations"][-1])

                # Stop if threshold reached
                if best_f1 >= target_f1:
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
    any_fail = any((lvl.get("best", {}).get("note_f1", 0.0) < target_f1) for lvl in report["levels"])
    return 2 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
