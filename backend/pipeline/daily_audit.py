# backend/pipeline/daily_audit.py
"""
Daily Pipeline Audit (Benchmark Ladder Snapshotter)

What this script does
- Runs benchmark levels sequentially (ladder gating: stop on first failure).
- Captures stdout/stderr per level.
- Writes a per-run manifest (git SHA, Python, platform, args).
- Stores benchmark outputs inside the audit run folder (preferred).
- Optionally snapshots the repository's results/ folder (filtered to avoid recursion).

Key improvements vs. earlier version
- Avoids copying results/audit into itself (prevents exponential growth).
- Uses benchmark_runner's --output so each level writes into a clean folder.
- Does not rely solely on subprocess return code (benchmark_runner may swallow exceptions).
- Performs defensive "pass/fail" evaluation from produced metrics + thresholds.
- Provides stable level name mapping: L0_MONO_SANITY -> L0, etc.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Small utilities
# -----------------------------

def now_stamps() -> Tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_rmtree(p: Path) -> None:
    try:
        if p.exists():
            shutil.rmtree(p)
    except Exception:
        pass


def copytree_overwrite(src: Path, dst: Path, ignore: Optional[callable] = None) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def run_subprocess(
    cmd: Sequence[str],
    cwd: Optional[Path],
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, float]:
    """
    Run subprocess and persist stdout/stderr to files.

    Returns:
      (returncode, wall_time_s)

    Notes:
      - If timeout triggers, returncode=124 (like GNU timeout).
      - If runner crashes before starting, returncode=125.
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        rc = int(proc.returncode)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        stderr += f"\n[daily_audit] TIMEOUT after {timeout_sec}s\n"
    except Exception as exc:
        rc = 125
        stdout = ""
        stderr = f"[daily_audit] SUBPROCESS ERROR: {exc}\n"

    dt = time.perf_counter() - t0

    ensure_dir(stdout_path.parent)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    return rc, float(dt)


def try_get_git_head(repo_root: Path) -> Optional[str]:
    try:
        rc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if rc.returncode == 0:
            head = (rc.stdout or "").strip()
            return head or None
    except Exception:
        pass
    return None


def try_get_git_status_short(repo_root: Path) -> Optional[str]:
    try:
        rc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if rc.returncode == 0:
            return (rc.stdout or "").strip() or None
    except Exception:
        pass
    return None


def read_summary_csv(summary_csv: Path) -> List[Dict[str, str]]:
    if not summary_csv.exists():
        return []
    try:
        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def tail(rows: List[Any], n: int = 20) -> List[Any]:
    return rows[-n:] if len(rows) > n else rows


def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


# -----------------------------
# Level mapping + evaluation
# -----------------------------

LEVEL_MAP: Dict[str, str] = {
    # legacy-ish names
    "L0_MONO_SANITY": "L0",
    "L1_MONO_MUSIC": "L1",
    "L2_POLY_DOMINANT": "L2",
    "L3_FULL_POLY": "L3",
    "L4_REAL_SONGS": "L4",
    # already compact
    "L0": "L0",
    "L1": "L1",
    "L2": "L2",
    "L3": "L3",
    "L4": "L4",
    "ALL": "all",
    "all": "all",
}


def resolve_level_token(level: str) -> str:
    key = str(level).strip()
    if key in LEVEL_MAP:
        return LEVEL_MAP[key]
    # Accept "L2_something" -> "L2"
    m = re.match(r"^(L[0-4])\b", key.upper())
    if m:
        return m.group(1)
    return key


def _try_load_benchmark_thresholds(benchmark_module: str) -> Dict[str, Any]:
    """
    Attempt to import benchmark_runner and fetch accuracy_benchmark_plan thresholds.
    Falls back to a minimal default if import fails.
    """
    fallback = {
        "note_f1_floor": {"L0": 0.85, "L1": 0.10, "L2": 0.05, "L3": 0.00, "L4": 0.00},
        "onset_mae_ms_max": 500.0,
    }
    try:
        mod = __import__(benchmark_module, fromlist=["accuracy_benchmark_plan"])
        plan = getattr(mod, "accuracy_benchmark_plan", None)
        if callable(plan):
            payload = plan()
            thresholds = payload.get("regression", {}).get("stage_thresholds", {}) or {}
            note_f1_floor = thresholds.get("note_f1_floor") or fallback["note_f1_floor"]
            onset_mae_ms_max = thresholds.get("onset_mae_ms_max", fallback["onset_mae_ms_max"])
            return {
                "note_f1_floor": dict(note_f1_floor),
                "onset_mae_ms_max": float(onset_mae_ms_max) if onset_mae_ms_max is not None else None,
            }
    except Exception:
        pass
    return fallback


def extract_accuracy_from_output(output_dir: Path, expected_level: str) -> Dict[str, Any]:
    """
    Defensive extraction of "accuracy-ish" data from the benchmark output directory.
    Works with the current benchmark_runner.py behavior (per-scenario *_metrics.json + summary.csv).
    """
    out: Dict[str, Any] = {"output_dir": str(output_dir), "expected_level": expected_level}
    out["exists"] = output_dir.exists()

    # summary.csv (if present)
    summary_csv = output_dir / "summary.csv"
    rows = read_summary_csv(summary_csv)
    if rows:
        out["summary_last_row"] = rows[-1]
        out["summary_tail"] = tail(rows, 10)
        out["summary_columns"] = list(rows[-1].keys())
        out["summary_rows_for_expected_level"] = [r for r in rows if (r.get("level") == expected_level)]
    else:
        out["summary_last_row"] = None
        out["summary_tail"] = []
        out["summary_columns"] = []
        out["summary_rows_for_expected_level"] = []

    # metrics files for this level
    metrics_files = sorted(output_dir.glob(f"{expected_level}_*_metrics.json"))
    metrics_objs: List[Dict[str, Any]] = []
    for p in metrics_files:
        obj = read_json(p)
        if isinstance(obj, dict):
            obj["_path"] = str(p)
            metrics_objs.append(obj)

    out["metrics_files"] = [str(p) for p in metrics_files]
    out["metrics"] = metrics_objs

    # leaderboard.json, summary.json, summary_diff.json (if present)
    for name in ["leaderboard.json", "summary.json", "summary_diff.json"]:
        p = output_dir / name
        out[name.replace(".", "_")] = read_json(p) if p.exists() else None

    # Derive quick aggregates for gating
    note_f1_vals = [m.get("note_f1") for m in metrics_objs]
    note_f1_vals_f = [v for v in (_coerce_float(x) for x in note_f1_vals) if v is not None]
    out["note_f1_min"] = min(note_f1_vals_f) if note_f1_vals_f else None
    out["note_f1_max"] = max(note_f1_vals_f) if note_f1_vals_f else None
    out["note_f1_mean"] = (sum(note_f1_vals_f) / len(note_f1_vals_f)) if note_f1_vals_f else None

    # onset_mae_ms (may be None)
    onset_vals = [m.get("onset_mae_ms") for m in metrics_objs]
    onset_vals_f = [v for v in (_coerce_float(x) for x in onset_vals) if v is not None]
    out["onset_mae_ms_min"] = min(onset_vals_f) if onset_vals_f else None
    out["onset_mae_ms_max"] = max(onset_vals_f) if onset_vals_f else None

    return out


def evaluate_level_passfail(
    expected_level: str,
    rc: int,
    stdout_text: str,
    stderr_text: str,
    accuracy: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine whether the level passed.

    Important: benchmark_runner may swallow exceptions and still exit 0.
    So we evaluate based on produced artifacts + conservative thresholds + log hints.
    """
    info: Dict[str, Any] = {
        "expected_level": expected_level,
        "subprocess_returncode": rc,
        "reasons": [],
    }

    # 1) Hard fail if subprocess timed out or crashed
    if rc in (124, 125):
        info["reasons"].append(f"subprocess_rc_{rc}")
        return False, info

    # 2) Detect obvious failure strings (best-effort)
    combined = (stdout_text or "") + "\n" + (stderr_text or "")
    if "Benchmark Suite Failed" in combined or "Traceback (most recent call last)" in combined:
        info["reasons"].append("runner_reported_failure")

    # 3) Require at least one metrics file for expected level
    metrics = accuracy.get("metrics", []) or []
    if not metrics:
        info["reasons"].append("no_metrics_files")
        return False, info

    # 4) Apply note_f1 floor if present
    floor = (thresholds.get("note_f1_floor") or {}).get(expected_level)
    note_f1_min = accuracy.get("note_f1_min")
    if floor is not None and note_f1_min is not None:
        if float(note_f1_min) < float(floor):
            info["reasons"].append(f"note_f1_below_floor(min={note_f1_min}, floor={floor})")

    # 5) Optional onset ceiling
    onset_ceiling = thresholds.get("onset_mae_ms_max")
    onset_max = accuracy.get("onset_mae_ms_max")
    if onset_ceiling is not None and onset_max is not None:
        if float(onset_max) > float(onset_ceiling):
            info["reasons"].append(f"onset_mae_above_ceiling(max={onset_max}, ceiling={onset_ceiling})")

    ok = len(info["reasons"]) == 0
    if ok:
        info["reasons"].append("ok")
    return ok, info


# -----------------------------
# Benchmark execution
# -----------------------------

@dataclass
class LevelRun:
    level_requested: str
    level_resolved: str
    cmd: List[str]
    returncode: int
    wall_time_s: float
    ok: bool
    eval: Dict[str, Any]
    stdout_path: str
    stderr_path: str
    output_dir: str
    accuracy: Dict[str, Any]
    results_root_snapshot: Optional[str] = None


class DailyPipelineAudit:
    def __init__(
        self,
        levels: List[str],
        audit_root: Path = Path("results") / "audit",
        results_root: Path = Path("results"),
        benchmark_module: str = "backend.benchmarks.benchmark_runner",
        repo_root: Optional[Path] = None,
        snapshot_results_root: bool = False,
        clean_results_root: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.levels = levels
        self.audit_root = audit_root
        self.results_root = results_root
        self.benchmark_module = benchmark_module
        self.repo_root = repo_root or Path.cwd()
        self.snapshot_results_root = bool(snapshot_results_root)
        self.clean_results_root = bool(clean_results_root)
        self.env = env or {}

        date_s, time_s = now_stamps()
        self.date_s = date_s
        self.time_s = time_s
        self.run_dir = ensure_dir(self.audit_root / date_s / f"run_{time_s}")

        self.thresholds = _try_load_benchmark_thresholds(self.benchmark_module)

    def _write_manifest(self, args_payload: Optional[Dict[str, Any]] = None) -> None:
        manifest = {
            "date": self.date_s,
            "time": self.time_s,
            "run_dir": str(self.run_dir),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "repo_root": str(self.repo_root),
            "git_head": try_get_git_head(self.repo_root),
            "git_status_porcelain": try_get_git_status_short(self.repo_root),
            "levels_requested": self.levels,
            "benchmark_module": self.benchmark_module,
            "thresholds": self.thresholds,
            "env_overrides": self.env,
            "args": args_payload or {},
        }
        write_json(self.run_dir / "manifest.json", manifest)

    def _ignore_results_snapshot(self, root: str, names: List[str]) -> set:
        # Prevent recursion: never copy results/audit into snapshots.
        ignore = set()
        if Path(root).resolve() == self.results_root.resolve():
            if "audit" in names:
                ignore.add("audit")
        # Optionally ignore cache-like dirs
        for n in names:
            if n in (".pytest_cache", "__pycache__"):
                ignore.add(n)
        return ignore

    def _snapshot_results_root(self, dst: Path) -> None:
        ignore_fn = lambda root, names: self._ignore_results_snapshot(root, names)
        copytree_overwrite(self.results_root, dst, ignore=ignore_fn)

    def _clean_results_root_safely(self) -> None:
        """
        Clean results_root while preserving results/audit (and only that).
        Useful if other tooling writes to results/ and you want a clean baseline.
        """
        if not self.results_root.exists():
            return
        for child in self.results_root.iterdir():
            if child.name == "audit":
                continue
            safe_rmtree(child)

    def run_level(
        self,
        level: str,
        extra_args: Optional[List[str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> LevelRun:
        extra_args = extra_args or []

        level_resolved = resolve_level_token(level)

        level_dir = ensure_dir(self.run_dir / "bench" / str(level))
        stdout_path = level_dir / f"benchmark_{level_resolved}.stdout.txt"
        stderr_path = level_dir / f"benchmark_{level_resolved}.stderr.txt"

        # Prefer writing benchmark outputs into the audit folder (no copying needed).
        output_dir = ensure_dir(level_dir / "benchmark_output")

        # Clean results_root before running (optional)
        if self.clean_results_root:
            self._clean_results_root_safely()

        # Run benchmark runner as a module (stable import path)
        # benchmark_runner supports: --output <dir> --level <L0|L1|...|all>
        cmd = [
            sys.executable,
            "-m",
            self.benchmark_module,
            "--level",
            level_resolved,
            "--output",
            str(output_dir),
        ] + list(extra_args)

        rc, wall_s = run_subprocess(
            cmd=cmd,
            cwd=self.repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_sec=timeout_sec,
            env=self.env,
        )

        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""

        # Extract accuracy/metrics from the benchmark output folder
        accuracy = extract_accuracy_from_output(output_dir, expected_level=level_resolved)

        ok, eval_info = evaluate_level_passfail(
            expected_level=level_resolved,
            rc=rc,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            accuracy=accuracy,
            thresholds=self.thresholds,
        )

        results_root_snapshot_path = None
        if self.snapshot_results_root:
            snap_dir = ensure_dir(level_dir / "results_root_snapshot")
            self._snapshot_results_root(snap_dir)
            results_root_snapshot_path = str(snap_dir)

        level_run = LevelRun(
            level_requested=level,
            level_resolved=level_resolved,
            cmd=cmd,
            returncode=rc,
            wall_time_s=wall_s,
            ok=ok,
            eval=eval_info,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            output_dir=str(output_dir),
            accuracy=accuracy,
            results_root_snapshot=results_root_snapshot_path,
        )

        write_json(level_dir / "level_result.json", {
            "level_requested": level_run.level_requested,
            "level_resolved": level_run.level_resolved,
            "cmd": level_run.cmd,
            "returncode": level_run.returncode,
            "wall_time_s": level_run.wall_time_s,
            "ok": level_run.ok,
            "eval": level_run.eval,
            "stdout": level_run.stdout_path,
            "stderr": level_run.stderr_path,
            "output_dir": level_run.output_dir,
            "accuracy": level_run.accuracy,
            "results_root_snapshot": level_run.results_root_snapshot,
        })

        return level_run

    def run(self, extra_args: Optional[List[str]] = None, timeout_sec: Optional[int] = None) -> int:
        index: Dict[str, Any] = {
            "status": "running",
            "run_dir": str(self.run_dir),
            "date": self.date_s,
            "time": self.time_s,
            "levels_requested": self.levels,
            "bench": [],
        }
        write_json(self.run_dir / "audit_index.json", index)

        all_ok = True
        for lvl in self.levels:
            res = self.run_level(lvl, extra_args=extra_args, timeout_sec=timeout_sec)

            index["bench"].append({
                "level_requested": res.level_requested,
                "level_resolved": res.level_resolved,
                "ok": res.ok,
                "returncode": res.returncode,
                "wall_time_s": res.wall_time_s,
                "level_result_json": str((self.run_dir / "bench" / str(lvl) / "level_result.json").resolve()),
                "output_dir": res.output_dir,
                "results_root_snapshot": res.results_root_snapshot,
            })
            write_json(self.run_dir / "audit_index.json", index)

            if not res.ok:
                all_ok = False
                # Gate behavior: stop on first failure (benchmark ladder principle)
                break

        index["status"] = "ok" if all_ok else "failed"
        write_json(self.run_dir / "audit_index.json", index)

        # Convenience pointer: results/audit/<date>/latest -> current run
        latest_dir = self.audit_root / self.date_s / "latest"
        safe_rmtree(latest_dir)
        copytree_overwrite(self.run_dir, latest_dir)

        return 0 if all_ok else 2


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily pipeline audit: run benchmarks and snapshot outputs per level."
    )
    p.add_argument(
        "--levels",
        nargs="*",
        default=["L0_MONO_SANITY", "L1_MONO_MUSIC"],
        help="Benchmark levels to run in order (L0/L1/L2/L3/L4 or legacy names). Stops on first failure.",
    )
    p.add_argument(
        "--benchmark-module",
        default="backend.benchmarks.benchmark_runner",
        help="Python module path for benchmark runner (must support --output and --level).",
    )
    p.add_argument(
        "--results-root",
        default="results",
        help="Repo results folder (used only for optional snapshot + latest pointer).",
    )
    p.add_argument(
        "--audit-root",
        default=str(Path("results") / "audit"),
        help="Where to store audit runs (default: results/audit).",
    )
    p.add_argument(
        "--repo-root",
        default=str(Path.cwd()),
        help="Repository root (cwd for benchmark subprocess).",
    )
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=0,
        help="Optional timeout per level (0 = no timeout).",
    )
    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg to pass to benchmark_runner (repeatable). Example: --extra-arg --device --extra-arg cpu",
    )
    p.add_argument(
        "--snapshot-results-root",
        action="store_true",
        help="Also snapshot the repo results/ folder after each level (audit folder excluded to avoid recursion).",
    )
    p.add_argument(
        "--clean-results-root",
        action="store_true",
        help="Before each level, delete everything under results/ except results/audit. Useful to avoid contamination.",
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        help='Environment override KEY=VALUE (repeatable). Example: --env CUDA_VISIBLE_DEVICES=0',
    )
    return p.parse_args()


def _parse_env_pairs(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def main() -> int:
    args = parse_args()

    env = _parse_env_pairs(args.env)

    audit = DailyPipelineAudit(
        levels=list(args.levels),
        audit_root=Path(args.audit_root),
        results_root=Path(args.results_root),
        benchmark_module=str(args.benchmark_module),
        repo_root=Path(args.repo_root),
        snapshot_results_root=bool(args.snapshot_results_root),
        clean_results_root=bool(args.clean_results_root),
        env=env,
    )

    timeout = None if int(args.timeout_sec) <= 0 else int(args.timeout_sec)

    # Record args in manifest early
    audit._write_manifest(args_payload={
        "levels": args.levels,
        "benchmark_module": args.benchmark_module,
        "results_root": args.results_root,
        "audit_root": args.audit_root,
        "repo_root": args.repo_root,
        "timeout_sec": args.timeout_sec,
        "extra_args": args.extra_arg,
        "snapshot_results_root": args.snapshot_results_root,
        "clean_results_root": args.clean_results_root,
        "env": args.env,
    })

    return audit.run(extra_args=list(args.extra_arg) if args.extra_arg else None, timeout_sec=timeout)


if __name__ == "__main__":
    raise SystemExit(main())
