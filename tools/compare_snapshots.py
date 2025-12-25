#!/usr/bin/env python3
import sys
import json
import argparse
import os

def load_snapshot(path):
    with open(path, 'r') as f:
        return json.load(f)

def compare_benchmarks(prev_bench, curr_bench, thresholds):
    # Convert list to dict keyed by (level, name)
    prev_map = {(x['level'], x['name']): x for x in prev_bench}
    curr_map = {(x['level'], x['name']): x for x in curr_bench}

    all_keys = sorted(list(set(prev_map.keys()) | set(curr_map.keys())))

    print("\n## Benchmark Comparison")
    print(f"{'Level':<5} {'Name':<25} {'F1 (Prev)':<10} {'F1 (Curr)':<10} {'Diff':<10} {'Status'}")
    print("-" * 80)

    regressions = []

    for key in all_keys:
        p = prev_map.get(key, {})
        c = curr_map.get(key, {})

        p_f1 = p.get('note_f1', 0.0) or 0.0
        c_f1 = c.get('note_f1', 0.0) or 0.0

        diff = c_f1 - p_f1
        status = "OK"

        # Check regression
        # Tolerance: e.g., 0.01 drop allowed?
        tol = thresholds.get('f1_drop_tolerance', 0.02)
        if diff < -tol:
            status = "REGRESSION"
            regressions.append(f"{key}: F1 dropped by {abs(diff):.4f}")
        elif diff > tol:
            status = "IMPROVED"

        print(f"{key[0]:<5} {key[1]:<25} {p_f1:<10.4f} {c_f1:<10.4f} {diff:<+10.4f} {status}")

    return regressions

def compare_trace_metrics(prev_metrics, curr_metrics):
    print("\n## Trace Metrics Comparison")

    # Flatten specific keys we care about
    def get_val(m, path):
        curr = m
        for k in path:
            curr = curr.get(k, {})
        return curr if isinstance(curr, (int, float)) else 0

    keys_to_compare = [
        ("stage_b", "voiced_ratio"),
        ("stage_b", "mean_confidence"),
        ("stage_c", "note_count"),
        ("stage_c", "fragmentation_score"),
        ("stage_d", "rendered_notes")
    ]

    print(f"{'Metric':<30} {'Prev':<10} {'Curr':<10} {'Diff':<10}")
    print("-" * 65)

    for path in keys_to_compare:
        name = ".".join(path)
        p_val = get_val(prev_metrics, path)
        c_val = get_val(curr_metrics, path)
        diff = c_val - p_val
        print(f"{name:<30} {p_val:<10.4f} {c_val:<10.4f} {diff:<+10.4f}")

def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark snapshots.")
    parser.add_argument("prev", help="Path to previous snapshot JSON")
    parser.add_argument("curr", help="Path to current snapshot JSON")
    parser.add_argument("--tolerance", type=float, default=0.05, help="F1 drop tolerance")

    args = parser.parse_args()

    if not os.path.exists(args.prev):
        print(f"Error: Previous snapshot not found: {args.prev}")
        sys.exit(1)
    if not os.path.exists(args.curr):
        print(f"Error: Current snapshot not found: {args.curr}")
        sys.exit(1)

    prev_snap = load_snapshot(args.prev)
    curr_snap = load_snapshot(args.curr)

    print(f"Comparing Run {prev_snap.get('run_id')} vs Run {curr_snap.get('run_id')}")

    regs = compare_benchmarks(
        prev_snap.get('benchmark_results', []),
        curr_snap.get('benchmark_results', []),
        {'f1_drop_tolerance': args.tolerance}
    )

    compare_trace_metrics(
        prev_snap.get('trace_metrics', {}),
        curr_snap.get('trace_metrics', {})
    )

    if regs:
        print("\nREGRESSIONS DETECTED:")
        for r in regs:
            print(f"- {r}")
        sys.exit(1)
    else:
        print("\nNo regressions detected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
