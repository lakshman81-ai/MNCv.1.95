# backend/pipeline/daily_audit.py
# Purpose: Run benchmark levels and SAVE ALL RESULTS/ACCURACY into a dated run folder.
# Works even if benchmark_runner only writes to ./results (this script snapshots it per level).

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
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
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def copytree_overwrite(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def safe_rmtree(p: Path) -> None:
    try:
        if p.exists():
            shutil.rmtree(p)
    except Exception:
        pass


def run_subprocess(
    cmd: Sequence[str],
    cwd: Optional[Path],
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )
    ensure_dir(stdout_path.parent)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return int(proc.returncode)


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


def extract_accuracy_from_results(results_snapshot: Path) -> Dict[str, Any]:
    """
    Defensive extraction of "accuracy-ish" data.
    - summary.csv: returns last row + header keys
    - leaderboard.json: stores the whole object (schema may vary)
    - metrics.json inside results/run_*/ if present: stores the newest one
    """
    out: Dict[str, Any] = {}

    # summary.csv
    summary_csv = results_snapshot / "summary.csv"
    rows = read_summary_csv(summary_csv)
    if rows:
        out["summary_last_row"] = rows[-1]
        out["summary_tail"] = tail(rows, 10)
        out["summary_columns"] = list(rows[-1].keys())
    else:
        out["summary_last_row"] = None
        out["summary_tail"] = []
        out["summary_columns"] = []

    # leaderboard.json
    leaderboard_json = results_snapshot / "leaderboard.json"
    if leaderboard_json.exists():
        out["leaderboard"] = read_json(leaderboard_json)
    else:
        out["leaderboard"] = None

    # newest metrics.json under results/run_*/
    run_dirs = sorted([p for p in results_snapshot.glob("run_*") if p.is_dir()])
    newest_metrics: Optional[Path] = None
    newest_dir: Optional[Path] = None
    if run_dirs:
        newest_dir = run_dirs[-1]
        cand = newest_dir / "metrics.json"
        if cand.exists():
            newest_metrics = cand

    out["newest_run_dir"] = str(newest_dir) if newest_dir else None
    out["newest_metrics"] = read_json(newest_metrics) if newest_metrics else None

    return out


# -----------------------------
# Benchmark execution
# -----------------------------

@dataclass
class LevelRun:
    level: str
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str
    results_snapshot: str
    accuracy: Dict[str, Any]


class DailyPipelineAudit:
    def __init__(
        self,
        levels: List[str],
        audit_root: Path = Path("results") / "audit",
        results_root: Path = Path("results"),
        benchmark_module: str = "backend.benchmarks.benchmark_runner",
        repo_root: Optional[Path] = None,
    ) -> None:
        self.levels = levels
        self.audit_root = audit_root
        self.results_root = results_root
        self.benchmark_module = benchmark_module
        self.repo_root = repo_root or Path.cwd()

        date_s, time_s = now_stamps()
        self.date_s = date_s
        self.time_s = time_s
        self.run_dir = ensure_dir(self.audit_root / date_s / f"run_{time_s}")

    def _write_manifest(self) -> None:
        manifest = {
            "date": self.date_s,
            "time": self.time_s,
            "run_dir": str(self.run_dir),
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "git_head": try_get_git_head(self.repo_root),
            "levels": self.levels,
            "benchmark_module": self.benchmark_module,
        }
        write_json(self.run_dir / "manifest.json", manifest)

    def _snapshot_results(self, dst: Path) -> None:
        # Snapshot EVERYTHING under ./results into this per-level folder
        copytree_overwrite(self.results_root, dst)

    def run_level(self, level: str, extra_args: Optional[List[str]] = None, timeout_sec: Optional[int] = None) -> LevelRun:
        extra_args = extra_args or []

        level_dir = ensure_dir(self.run_dir / "bench" / level)
        stdout_path = level_dir / f"benchmark_{level}.stdout.txt"
        stderr_path = level_dir / f"benchmark_{level}.stderr.txt"

        # Run benchmark runner as a module (most stable)
        cmd = [sys.executable, "-m", self.benchmark_module, "--level", level] + extra_args

        rc = run_subprocess(
            cmd=cmd,
            cwd=self.repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_sec=timeout_sec,
        )

        # Snapshot results after the run so each level gets its own preserved outputs
        snapshot_dir = level_dir / "results_snapshot"
        self._snapshot_results(snapshot_dir)

        # Extract accuracy/metrics defensively
        accuracy = extract_accuracy_from_results(snapshot_dir)

        # Store a single JSON summary for this level
        level_run = LevelRun(
            level=level,
            cmd=cmd,
            returncode=rc,
            stdout=str(stdout_path),
            stderr=str(stderr_path),
            results_snapshot=str(snapshot_dir),
            accuracy=accuracy,
        )
        write_json(level_dir / "level_result.json", {
            "level": level_run.level,
            "cmd": level_run.cmd,
            "returncode": level_run.returncode,
            "stdout": level_run.stdout,
            "stderr": level_run.stderr,
            "results_snapshot": level_run.results_snapshot,
            "accuracy": level_run.accuracy,
        })

        return level_run

    def run(self, extra_args: Optional[List[str]] = None, timeout_sec: Optional[int] = None) -> int:
        self._write_manifest()

        index: Dict[str, Any] = {
            "status": "running",
            "run_dir": str(self.run_dir),
            "date": self.date_s,
            "time": self.time_s,
            "levels": self.levels,
            "bench": [],
        }
        write_json(self.run_dir / "audit_index.json", index)

        all_ok = True
        for lvl in self.levels:
            res = self.run_level(lvl, extra_args=extra_args, timeout_sec=timeout_sec)
            index["bench"].append({
                "level": res.level,
                "returncode": res.returncode,
                "level_result_json": str((self.run_dir / "bench" / lvl / "level_result.json").resolve()),
                "results_snapshot": res.results_snapshot,
            })
            write_json(self.run_dir / "audit_index.json", index)

            if res.returncode != 0:
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
    p = argparse.ArgumentParser(description="Daily pipeline audit: run benchmarks and snapshot results per level.")
    p.add_argument(
        "--levels",
        nargs="*",
        default=["L0_MONO_SANITY", "L1_MONO_MUSIC"],
        help="Benchmark levels to run in order. Stops on first failure.",
    )
    p.add_argument(
        "--benchmark-module",
        default="backend.benchmarks.benchmark_runner",
        help="Python module path for benchmark runner.",
    )
    p.add_argument(
        "--results-root",
        default="results",
        help="Where benchmark_runner writes outputs (default: results).",
    )
    p.add_argument(
        "--audit-root",
        default=str(Path("results") / "audit"),
        help="Where to store audit runs (default: results/audit).",
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
    return p.parse_args()


def main() -> int:
    args = parse_args()

    audit = DailyPipelineAudit(
        levels=list(args.levels),
        audit_root=Path(args.audit_root),
        results_root=Path(args.results_root),
        benchmark_module=str(args.benchmark_module),
        repo_root=Path.cwd(),
    )

    timeout = None if int(args.timeout_sec) <= 0 else int(args.timeout_sec)

    return audit.run(extra_args=list(args.extra_arg) if args.extra_arg else None, timeout_sec=timeout)


if __name__ == "__main__":
    raise SystemExit(main())
