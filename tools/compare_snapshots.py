#!/usr/bin/env python3
"""
tools/compare_snapshots.py

Compare two snapshot folders (each containing *_metrics.json written by benchmark_runner)
and flag regressions.

Usage:
  python tools/compare_snapshots.py --baseline reports/snapshots/<runA> --current reports/snapshots/<runB>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_metrics(dir_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fn in os.listdir(dir_path):
        if not fn.endswith("_metrics.json"):
            continue
        case_id = fn[:-len("_metrics.json")]
        try:
            out[case_id] = _load_json(os.path.join(dir_path, fn))
        except Exception:
            continue
    return out


def compare(baseline_dir: str, current_dir: str) -> Tuple[str, bool]:
    a = _collect_metrics(baseline_dir)
    b = _collect_metrics(current_dir)

    common = sorted(set(a.keys()) & set(b.keys()))
    md: List[str] = []
    md.append("# Regression Flags")
    md.append("")
    md.append(f"Baseline: `{baseline_dir}`")
    md.append(f"Current: `{current_dir}`")
    md.append("")

    if not common:
        md.append("No comparable cases found.")
        return "\n".join(md) + "\n", False

    regressions: List[Tuple[str, List[str]]] = []

    for cid in common:
        am = a[cid]
        bm = b[cid]

        a_nc = float(am.get("note_count", am.get("predicted_count", 0.0)) or 0.0)
        b_nc = float(bm.get("note_count", bm.get("predicted_count", 0.0)) or 0.0)

        a_vr = float(am.get("voiced_ratio", 0.0) or 0.0)
        b_vr = float(bm.get("voiced_ratio", 0.0) or 0.0)

        a_fr = float(am.get("fragmentation_score", 0.0) or 0.0)
        b_fr = float(bm.get("fragmentation_score", 0.0) or 0.0)

        a_f1 = am.get("note_f1", None)
        b_f1 = bm.get("note_f1", None)

        flags: List[str] = []
        if a_nc > 0 and (b_nc < 0.8 * a_nc) and ((a_nc - b_nc) >= 10):
            flags.append(f"note_count drop {a_nc:.0f}→{b_nc:.0f}")
        if (a_vr - b_vr) > 0.05:
            flags.append(f"voiced_ratio drop {a_vr:.3f}→{b_vr:.3f}")
        if (b_fr - a_fr) > 0.05:
            flags.append(f"fragmentation increase {a_fr:.3f}→{b_fr:.3f}")
        if isinstance(a_f1, (int, float)) and isinstance(b_f1, (int, float)) and (a_f1 - b_f1) > 0.02:
            flags.append(f"note_f1 drop {a_f1:.3f}→{b_f1:.3f}")

        if flags:
            regressions.append((cid, flags))

    if not regressions:
        md.append("✅ No regressions detected by heuristic gates.")
        return "\n".join(md) + "\n", False

    md.append("## Regressions")
    for cid, flags in regressions:
        md.append(f"- **{cid}**: " + "; ".join(flags))
    md.append("")
    md.append("> Note: These are heuristic gates. Inspect the per-case *_run_info.json and timeline CSVs to diagnose.")
    return "\n".join(md) + "\n", True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline snapshot directory")
    ap.add_argument("--current", required=True, help="Current snapshot directory")
    ap.add_argument("--out", default="", help="Write markdown to this path (optional)")
    args = ap.parse_args()

    baseline_dir = os.path.abspath(args.baseline)
    current_dir = os.path.abspath(args.current)

    md, has_reg = compare(baseline_dir, current_dir)

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        print(md)

    return 2 if has_reg else 0


if __name__ == "__main__":
    raise SystemExit(main())
