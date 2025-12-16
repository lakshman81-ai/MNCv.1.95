"""CI-friendly regression gate for benchmark note F1."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Tuple


def _load_metrics(path: str) -> Dict[Tuple[str, str], float]:
    metrics: Dict[Tuple[str, str], float] = {}
    if not os.path.exists(path):
        return metrics

    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                level = row.get("level") or ""
                name = row.get("name") or ""
                key = (level, name)
                try:
                    metrics[key] = float(row.get("note_f1", 0.0))
                except Exception:
                    metrics[key] = 0.0
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Accept list-of-dicts or dict keyed by name
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                level = str(entry.get("level", ""))
                name = str(entry.get("name", ""))
                key = (level, name)
                try:
                    metrics[key] = float(entry.get("note_f1", 0.0))
                except Exception:
                    metrics[key] = 0.0
        elif isinstance(data, dict):
            for name, value in data.items():
                key = ("", str(name))
                try:
                    metrics[key] = float(value)
                except Exception:
                    metrics[key] = 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if note_f1 regresses beyond a threshold.")
    parser.add_argument("--baseline", required=False, default="results/benchmark_latest.json", help="Baseline metrics JSON/CSV")
    parser.add_argument("--current", required=True, help="Current metrics JSON/CSV (e.g., summary.csv)")
    parser.add_argument("--max_regress_pct", type=float, default=5.0, help="Maximum allowed percentage regression")
    args = parser.parse_args()

    baseline = _load_metrics(args.baseline)
    current = _load_metrics(args.current)

    if not baseline or not current:
        # Nothing to compare; be lenient for first runs
        return

    threshold = max(float(args.max_regress_pct), 0.0) / 100.0
    regressions = []
    for key, current_f1 in current.items():
        base_f1 = baseline.get(key)
        if base_f1 is None or base_f1 <= 0:
            continue
        if current_f1 < base_f1 * (1.0 - threshold):
            regressions.append((key, base_f1, current_f1))

    if regressions:
        messages = [
            f"{lvl}/{name}: {cur:.3f} < baseline {base:.3f} by more than {threshold*100:.1f}%"
            for (lvl, name), base, cur in regressions
        ]
        raise SystemExit("Regression gate failed: " + "; ".join(messages))


if __name__ == "__main__":
    main()
