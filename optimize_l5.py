
import sys
import os
import itertools
import subprocess
import json
import re
from typing import Dict, Any, List

def run_benchmark(level: str) -> float:
    """Run benchmark for a specific level and return F1 score."""
    cmd = [
        "python", "-m", "backend.benchmarks.benchmark_runner",
        "--level", level
    ]
    print(f"Running benchmark {level}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})

        if result.returncode != 0:
            print(f"Benchmark failed: {result.stderr}")
            return 0.0

        # Try to find JSON first
        results_dir = "results"
        if os.path.exists(results_dir):
            bench_dirs = [d for d in os.listdir(results_dir) if d.startswith("benchmark_")]
            if bench_dirs:
                latest_bench = max(bench_dirs, key=lambda d: os.path.getmtime(os.path.join(results_dir, d)))
                metrics_path = os.path.join(results_dir, latest_bench, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                        if level in data:
                            return float(data[level].get("note_f1", 0.0))
                        if "note_f1" in data:
                             return float(data["note_f1"])

        # Fallback to stdout regex
        match = re.search(r"Note F1:\s*([\d\.]+)", result.stdout)
        if match:
            return float(match.group(1))

        match = re.search(r"F1:\s*([\d\.]+)", result.stdout)
        if match:
            return float(match.group(1))

    except Exception as e:
        print(f"Error running benchmark: {e}")
        return 0.0

    return 0.0

def patch_config(params: Dict[str, Any]):
    """
    Dynamically patch backend/pipeline/config.py with new params.
    """
    with open("backend/pipeline/config.py", "r") as f:
        content = f.read()

    new_content = content

    for key, value in params.items():
        if key == "stage_c.min_note_duration_ms_poly":
            new_content = re.sub(
                r"(min_note_duration_ms_poly:\s*float\s*=\s*)([\d\.]+)",
                f"\\g<1>{value}",
                new_content
            )
        elif key == "stage_b.polyphonic_peeling.mask_width":
            # Use \g<1> to prevent ambiguity with digits in value
            new_content = re.sub(
                r'("mask_width":\s*)([\d\.]+)(,\s*# Fractional bandwidth)',
                f'\\g<1>{value}\\g<3>',
                new_content
            )
        elif key == "stage_c.segmentation_method.transition_penalty":
            new_content = re.sub(
                r'("transition_penalty":\s*)([\d\.]+)',
                f'\\g<1>{value}',
                new_content
            )
        elif key == "stage_c.polyphonic_confidence.accompaniment":
            new_content = re.sub(
                r'("accompaniment":\s*)([\d\.]+)',
                f'\\g<1>{value}',
                new_content
            )

    with open("backend/pipeline/config.py", "w") as f:
        f.write(new_content)

def main():
    grid = {
        "stage_c.min_note_duration_ms_poly": [40.0, 45.0, 50.0],
        "stage_b.polyphonic_peeling.mask_width": [0.025, 0.035],
        "stage_c.segmentation_method.transition_penalty": [0.7, 0.9],
        "stage_c.polyphonic_confidence.accompaniment": [0.35, 0.45],
        "stage_b.detectors.yin.frame_length": [4096]
    }

    keys = list(grid.keys())
    values = list(grid.values())

    combinations = list(itertools.product(*values))
    print(f"Total combinations: {len(combinations)}")

    results = []

    with open("backend/pipeline/config.py", "r") as f:
        original_config = f.read()

    try:
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\n--- Run {i+1}/{len(combinations)} ---")
            print(f"Params: {params}")

            patch_config(params)

            f1 = run_benchmark("L5.1")
            print(f"L5.1 F1: {f1}")

            results.append({
                "params": params,
                "f1": f1
            })

    finally:
        with open("backend/pipeline/config.py", "w") as f:
            f.write(original_config)

    results.sort(key=lambda x: x["f1"], reverse=True)

    print("\n\n=== Optimization Results ===")
    print(f"{'F1':<10} | Params")
    print("-" * 100)
    for r in results:
        p_str = ", ".join([f"{k.split('.')[-1]}={v}" for k, v in r["params"].items()])
        print(f"{r['f1']:<10.4f} | {p_str}")

    best = results[0]
    print("\nBest Parameters:")
    print(json.dumps(best["params"], indent=2))

if __name__ == "__main__":
    main()
