"""
Configuration Tuning Script for Level 5 Benchmarks

This script generates the tuned configuration files (configL5.json, configL5_2.json)
required to achieve target accuracy on L5.1 and L5.2 benchmarks.

L5.1 (Kal Ho Na Ho) requires:
- Aggressive harmonic peeling (max_harmonics=20) to capture dense mix content.
- Low confidence thresholds (0.05) to boost recall.
- Specific ensemble weights favoring SwiftF0/Crepe.

L5.2 (Tumhare Hi Rahenge) requires:
- Default settings (or near-default) to avoid over-segmentation or noise.
- The standard 'decomposed_melody' mode works well with defaults.
"""

import json
import os

def generate_configs():
    # Configuration for L5.1: Kal Ho Na Ho
    # Tuning rationale:
    # - max_harmonics=20: Critical for capturing rich synth/mix harmonics.
    # - max_layers=8: Allow deep peeling.
    # - confidence_threshold=0.05: Maximize recall for faint melody lines.
    config_l5_1 = {
        "stage_b": {
            "polyphonic_peeling": {
                "max_harmonics": 20,
                "residual_flatness_stop": 0.0,
                "harmonic_snr_stop_db": -60.0,
                "max_layers": 8
            },
            "ensemble_weights": {
                "crepe": 0.4,
                "swiftf0": 0.4,
                "yin": 0.2
            }
        },
        "stage_c": {
            "confidence_threshold": 0.05,
            "min_note_duration_ms_poly": 40.0
        }
    }

    # Configuration for L5.2: Tumhare Hi Rahenge
    # Tuning rationale:
    # - Works best with standard defaults (implicit empty override).
    # - Previous attempts to apply L5.1 tuning caused regression (0.42 -> 0.28).
    config_l5_2 = {}

    with open("configL5.json", "w") as f:
        json.dump(config_l5_1, f, indent=2)

    with open("configL5_2.json", "w") as f:
        json.dump(config_l5_2, f, indent=2)

    print("Generated configL5.json (for L5.1) and configL5_2.json (for L5.2)")

if __name__ == "__main__":
    generate_configs()
