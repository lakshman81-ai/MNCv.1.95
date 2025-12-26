#!/usr/bin/env python3
"""
Compare two benchmark snapshot directories (JSON files) and flag regressions.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

def load_snapshot(path: Path) -> Dict[str, Any]:
    """Load benchmark results from a folder or single JSON file."""
    results = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            # If it's a list (summary.json), index by level+name
            if isinstance(data, list):
                for item in data:
                    key = f"{item.get('level')}_{item.get('name')}"
                    results[key] = item
            else:
                # Single metric file
                key = f"{data.get('level')}_{data.get('name')}"
                results[key] = data
        except Exception as e:
            print(f"Error loading {path}: {e}")
    elif path.is_dir():
        for f in path.glob("*_metrics.json"):
            try:
                data = json.loads(f.read_text())
                key = f"{data.get('level')}_{data.get('name')}"
                results[key] = data
            except Exception:
                pass
    return results

def compare(baseline: Dict[str, Any], candidate: Dict[str, Any], threshold_f1: float = -0.05) -> List[str]:
    regressions = []

    all_keys = set(baseline.keys()) | set(candidate.keys())

    for key in sorted(all_keys):
        b = baseline.get(key)
        c = candidate.get(key)

        if not b:
            # New benchmark?
            continue
        if not c:
            regressions.append(f"❌ {key}: Missing in candidate run")
            continue

        # Compare Note F1
        b_f1 = b.get("note_f1")
        c_f1 = c.get("note_f1")

        if b_f1 is not None and c_f1 is not None:
            delta = c_f1 - b_f1
            if delta < threshold_f1:
                regressions.append(f"📉 {key}: F1 dropped by {delta:.3f} ({b_f1:.3f} -> {c_f1:.3f})")

        # Compare Note Count (sanity)
        b_cnt = b.get("predicted_count", 0)
        c_cnt = c.get("predicted_count", 0)
        if b_cnt > 0 and c_cnt == 0:
             regressions.append(f"🔴 {key}: Zero notes (was {b_cnt})")

        # Compare Voiced Ratio
        b_vr = b.get("voiced_ratio", 0.0)
        c_vr = c.get("voiced_ratio", 0.0)
        if b_vr > 0.1 and c_vr < 0.05:
            regressions.append(f"🔇 {key}: Voiced ratio collapse ({b_vr:.2f} -> {c_vr:.2f})")

    return regressions

def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark snapshots.")
    parser.add_argument("baseline", type=Path, help="Baseline folder or JSON")
    parser.add_argument("candidate", type=Path, help="Candidate folder or JSON")
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"Baseline path not found: {args.baseline}")
        sys.exit(1)
    if not args.candidate.exists():
        print(f"Candidate path not found: {args.candidate}")
        sys.exit(1)

    b_data = load_snapshot(args.baseline)
    c_data = load_snapshot(args.candidate)

    print(f"Loaded {len(b_data)} baseline items, {len(c_data)} candidate items.")

    regressions = compare(b_data, c_data)

    if regressions:
        print("\nRegressions Detected:")
        for r in regressions:
            print(r)
        sys.exit(1)
    else:
        print("\n✅ No significant regressions found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
