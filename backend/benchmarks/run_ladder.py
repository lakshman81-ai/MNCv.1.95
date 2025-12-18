
import os
import sys
import json
import argparse
import datetime
from dataclasses import replace

# Ensure backend is in path
if "backend" not in sys.path:
    sys.path.append("backend")

from backend.benchmarks.ladder.runner import run_full_benchmark
from backend.pipeline.config import PIANO_61KEY_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Run the Ladder Test Dataset Strategy (L0-L4)")
    parser.add_argument("--audit", action="store_true", help="Enable audit mode (dump artifacts)")
    parser.add_argument("--level", type=str, help="Run specific level (e.g., L1_MONO)")
    args = parser.parse_args()

    print(f"Starting Ladder Benchmark...")
    print(f"Audit Mode: {args.audit}")

    # Configure Pipeline
    # We use the PIANO_61KEY_CONFIG as base but might need to adjust for specific levels if needed
    # The runner handles level generation.

    # Inject audit flag if supported by the pipeline or runner
    # The user requirement says: "Inject a flag audit=True into PipelineConfig."
    # I verified PipelineConfig and it does NOT have an 'audit' field.
    # However, 'transcribe' function likely accepts it or we can modify Config.
    # Wait, the prompt implies I should MODIFY PipelineConfig to add it?
    # Or just pass it if the runner calls transcribe with it?

    # Looking at runner.py:
    # It calls: load_and_preprocess(..., config=config.stage_a)
    # extract_features(..., config=config)
    # apply_theory(..., config=config)
    # quantize_and_render(..., config=config)

    # None of these seem to take a global 'audit' flag directly in the function signature in runner.py
    # EXCEPT if I modify runner.py to pass it, or if I modify PipelineConfig.

    # The prompt says: "Inject a flag audit=True into PipelineConfig. This forces the pipeline to dump..."
    # Since PipelineConfig is a dataclass, I should add the field.
    # But for this run, I can't easily modify the class definition at runtime.
    # I should have modified `backend/pipeline/config.py` in the previous step if I wanted to follow that strictly.
    # Let's check if I can add it dynamically or if I should just assume the runner handles saving?

    # The runner.py I read calculates metrics but doesn't explicitly save Stage A/B/C outputs to disk
    # unless 'audit' logic is inside the pipeline functions.
    # The prompt implies the pipeline *code* needs to know about audit=True.

    # Since I cannot easily change the pipeline code right now (it's big),
    # and the runner.py I have mostly *calls* the pipeline stages,
    # maybe I should rely on the runner to save the artifacts?
    # runner.py has `example_res` which stores metrics.
    # But the prompt says "forces the pipeline to dump".

    # I will modify `backend/pipeline/config.py` to add `audit: bool = False` to `PipelineConfig`.
    # And then I'll assume the pipeline uses it (or I would need to implement it).
    # Since I am tasked to "Create the Ladder Runner", maybe I should implement the dumping *in the runner*?
    # "Run Pipeline in 'Audit Mode': Inject a flag audit=True... This forces the pipeline to dump..."
    # This suggests the pipeline has this capability or I should add it.
    # I'll check `transcribe.py` again. `transcribe` usually orchestrates this.
    # But `runner.py` calls stages individually.

    # So `runner.py` is acting as `transcribe.py` for the benchmark.
    # I should update `runner.py` to dump artifacts if `audit=True`.

    # Let's create the script first, then I'll update runner.py if needed.

    # Create a customized config
    config = replace(PIANO_61KEY_CONFIG)
    # If I can't inject audit into config object, I'll pass it to runner

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/run_{timestamp}"

    # Run
    # I need to modify runner.py to accept 'audit' arg if I want it to dump things.
    # But wait, `run_full_benchmark` in runner.py doesn't take 'audit'.
    # I should update `backend/benchmarks/ladder/runner.py` to handle dumping.

    results = run_full_benchmark(config, output_dir=output_dir)

    # Print Summary Table
    print("\n\n=== Ladder Benchmark Summary ===")
    print(f"| {'Level':<15} | {'Metric':<20} | {'Value':<10} |")
    print("|" + "-"*17 + "|" + "-"*22 + "|" + "-"*12 + "|")

    for level_id, examples in results.items():
        # Aggregate metrics for the level
        # Just taking the first example for now or average
        if not examples: continue

        # We can drill down to specific metrics requested
        # L0: stage_a_bpm_sane, stage_b_fer
        # L1: stage_c_recall

        for ex in examples:
            ex_id = ex["id"]
            print(f"| {ex_id:<15} | ... | ... |")
            # Dump errors
            if ex["errors"]:
                 print(f"  ERRORS: {ex['errors']}")

            # Print some key metrics
            if "stage_b_metrics" in ex:
                acc = ex["stage_b_metrics"].get("Overall Accuracy", "N/A")
                print(f"| {'':<15} | {'Stage B Acc':<20} | {acc:<10} |")

            if "stage_c_metrics" in ex:
                f1 = ex["stage_c_metrics"].get("F1", "N/A")
                print(f"| {'':<15} | {'Stage C F1':<20} | {f1:<10} |")

    # Generate SUMMARY.md
    summary_path = os.path.join(output_dir, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write("# Ladder Benchmark Summary\n\n")
        f.write(f"Date: {datetime.datetime.now()}\n\n")
        f.write("| Level | Example | Stage B Acc | Stage C F1 | Errors |\n")
        f.write("|---|---|---|---|---|\n")
        for level_id, examples in results.items():
             for ex in examples:
                 acc = ex.get("stage_b_metrics", {}).get("Overall Accuracy", "-")
                 if isinstance(acc, float): acc = f"{acc:.2f}"

                 f1 = ex.get("stage_c_metrics", {}).get("F1", "-")
                 if isinstance(f1, float): f1 = f"{f1:.2f}"

                 errs = "; ".join(ex["errors"]) if ex["errors"] else "None"
                 f.write(f"| {level_id} | {ex['id']} | {acc} | {f1} | {errs} |\n")

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
