"""
Pipeline Audit Runner

Runs canonical scenarios to verify pipeline continuity and contracts.
Generates synthetic assets on the fly.
"""

import argparse
import sys
import os
import numpy as np
import soundfile as sf
import json
import dataclasses
import copy
from typing import Dict, Any

from backend.pipeline.config import PIANO_61KEY_CONFIG, PipelineConfig
from backend.pipeline.transcribe import transcribe
from backend.pipeline.models import AudioType, TranscriptionResult
from backend.pipeline.instrumentation import PipelineLogger

def generate_sine_wave(freq=440.0, duration=2.0, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    return y, sr

def generate_poly_chord(freqs=[440.0, 554.37, 659.25], duration=2.0, sr=44100):
    # C major (C4, E4, G4)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.zeros_like(t)
    for f in freqs:
        y += 0.3 * np.sin(2 * np.pi * f * t)
    return y, sr

def generate_silence(duration=5.0, sr=44100):
    # Just a tiny bit of noise to avoid absolute zero issues if needed
    y = np.random.normal(0, 1e-6, int(sr * duration))
    return y, sr

def run_scenario(name: str, audio_path: str, config: PipelineConfig) -> Dict[str, Any]:
    print(f"\n--- Scenario: {name} ---")

    # Run pipeline
    # We use a throwaway logger to capture timing/events if needed,
    # but transcribe returns the main result.
    logger = PipelineLogger(run_name=f"audit_{name}")

    try:
        result = transcribe(audio_path, config=config, pipeline_logger=logger)

        # Check diagnostics
        analysis = result.analysis_data

        contracts = analysis.diagnostics.get("contracts", {}) if hasattr(analysis, "diagnostics") else {}

        summary = {
            "name": name,
            "status": "success",
            "notes": len(analysis.notes),
            "beats": len(analysis.beats),
            "contracts": contracts,
            "pipeline_version": analysis.meta.pipeline_version
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        summary = {
            "name": name,
            "status": "crash",
            "error": str(e)
        }

    return summary

def main():
    parser = argparse.ArgumentParser(description="Pipeline Audit Runner")
    parser.add_argument("--output", default="audit_report.json")
    parser.add_argument("--output-root", default="results/audit_assets", help="Directory for synthetic assets")
    args = parser.parse_args()

    # Create synthetic assets
    os.makedirs(args.output_root, exist_ok=True)

    # 1. Mono Clean
    path_mono = os.path.join(args.output_root, "mono_sine.wav")
    y, sr = generate_sine_wave()
    sf.write(path_mono, y, sr)

    # 2. Poly Chord
    path_poly = os.path.join(args.output_root, "poly_chord.wav")
    y, sr = generate_poly_chord()
    sf.write(path_poly, y, sr)

    # 3. Short Clip (BPM gate)
    path_short = os.path.join(args.output_root, "short_clip.wav")
    y, sr = generate_sine_wave(duration=4.0) # < 6s
    sf.write(path_short, y, sr)

    results = []

    # Scenario 1: Mono Clean
    cfg_mono = copy.deepcopy(PIANO_61KEY_CONFIG)
    cfg_mono.stage_b.detectors["yin"]["enabled"] = True # Force robust detector
    # Ensure sine wave isn't gated by noise floor estimation
    cfg_mono.stage_a.noise_floor_estimation = {"method": "percentile", "percentile": 1}
    # Also we might need to lower velocity threshold in Stage C if sine is quiet
    cfg_mono.stage_c.velocity_map["min_db"] = -80.0
    # Use threshold segmentation for synthetic sine (HMM expects ADSR)
    cfg_mono.stage_c.segmentation_method = {"method": "threshold"}
    # Disable separation to keep audit fast (contract verification only)
    if not cfg_mono.stage_b.separation:
        cfg_mono.stage_b.separation = {}
    cfg_mono.stage_b.separation["enabled"] = False
    results.append(run_scenario("Mono Clean", path_mono, cfg_mono))

    # Scenario 2: Poly Chord
    cfg_poly = copy.deepcopy(PIANO_61KEY_CONFIG)
    # Use threshold segmentation for synthetic sine
    cfg_poly.stage_c.segmentation_method = {"method": "threshold"}
    cfg_poly.stage_a.noise_floor_estimation = {"method": "percentile", "percentile": 1}
    cfg_poly.stage_c.velocity_map["min_db"] = -80.0
    # Disable separation to keep audit fast
    if not cfg_poly.stage_b.separation:
        cfg_poly.stage_b.separation = {}
    cfg_poly.stage_b.separation["enabled"] = False
    results.append(run_scenario("Poly Chord", path_poly, cfg_poly))

    # Scenario 3: Short Clip (Verify BPM gate)
    # Default config has 6s limit? Check code.
    # Actually code uses librosa defaults or overrides.
    # We just run it and check beat count.
    results.append(run_scenario("Short Clip", path_short, PIANO_61KEY_CONFIG))

    # Scenario 4: No Deps (Simulation)
    # We can't easily unload modules in a running process without hacking sys.modules
    # So we skip this or mock it if we had a mocking framework here.
    # For this script, we'll skip actual dependency unloading to avoid breaking other tests.

    # Print Dashboard
    print("\n\n=== PIPELINE AUDIT DASHBOARD ===")
    print(f"{'SCENARIO':<20} | {'STATUS':<10} | {'NOTES':<6} | {'BEATS':<6} | {'CONTRACTS'}")
    print("-" * 80)
    for r in results:
        contracts_str = "OK"
        if r.get("contracts"):
            # v is {"status": "pass"/"fail", ...}
            fails = []
            for k, v in r["contracts"].items():
                if isinstance(v, dict) and v.get("status") != "pass":
                    fails.append(f"{k}({len(v.get('violations', []))})")
                elif isinstance(v, str) and v != "pass": # Fallback if I returned string somewhere
                    fails.append(k)

            if fails:
                contracts_str = f"FAIL: {fails}"

        print(f"{r['name']:<20} | {r['status']:<10} | {r.get('notes', '-'):<6} | {r.get('beats', '-'):<6} | {contracts_str}")

    # Dump JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
