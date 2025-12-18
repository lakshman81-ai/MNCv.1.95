import os
import json
import traceback
import numpy as np
import sys
from typing import Dict, Any, List

# Ensure backend is in path
if "backend" not in sys.path:
    sys.path.append("backend")

from .levels import BENCHMARK_LEVELS
from .generators import generate_benchmark_example
from .synth import midi_to_wav_synth
from .metrics.stage_a import calculate_stage_a_metrics
from .metrics.stage_b import calculate_stage_b_metrics
from .metrics.stage_c import calculate_stage_c_metrics
from .metrics.stage_d import calculate_stage_d_metrics

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def run_full_benchmark(config: Any, output_dir: str = "benchmark_results") -> Dict[str, Any]:
    """
    Runs the full benchmark ladder.
    config: PipelineConfig object.
    """
    # Import pipeline stages inside function to avoid circular deps or init issues
    try:
        from backend.pipeline.stage_a import load_and_preprocess
        from backend.pipeline.stage_b import extract_features
        from backend.pipeline.stage_c import apply_theory
        from backend.pipeline.stage_d import quantize_and_render
        from backend.pipeline.models import AnalysisData
    except ImportError:
        # Fallback to local import if backend not in path correctly
        from pipeline.stage_a import load_and_preprocess
        from pipeline.stage_b import extract_features
        from pipeline.stage_c import apply_theory
        from pipeline.stage_d import quantize_and_render
        from pipeline.models import AnalysisData

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for level in BENCHMARK_LEVELS:
        level_id = level["id"]
        print(f"Running Level: {level_id}")
        level_results = []

        for example_id in level["examples"]:
            print(f"  Example: {example_id}")
            example_res = {"id": example_id, "errors": []}

            try:
                wav_path = ""
                midi_path = ""

                # Check if it is real audio
                if level.get("is_real_audio", False):
                    # Look in mock_data
                    mock_data_dir = os.path.join("backend", "mock_data")
                    potential_wav = os.path.join(mock_data_dir, f"{example_id}.wav")
                    potential_midi = os.path.join(mock_data_dir, f"{example_id}.mid")

                    if os.path.exists(potential_wav):
                        wav_path = potential_wav
                        # Check for GT MIDI
                        if os.path.exists(potential_midi):
                            midi_path = potential_midi
                        else:
                            # Try _gt.mid
                            potential_midi_gt = os.path.join(mock_data_dir, f"{example_id}_gt.mid")
                            if os.path.exists(potential_midi_gt):
                                midi_path = potential_midi_gt
                            else:
                                print(f"    Warning: No GT MIDI found for {example_id}")
                    else:
                        print(f"    Warning: Real audio file not found: {potential_wav}")
                        example_res["errors"].append("File not found")
                        level_results.append(example_res)
                        continue
                else:
                    # Synthetic Generation
                    score = generate_benchmark_example(example_id)
                    midi_path = os.path.join(output_dir, f"{example_id}.mid")
                    score.write("midi", fp=midi_path)

                    wav_path = os.path.join(output_dir, f"{example_id}.wav")
                    midi_to_wav_synth(score, wav_path)

                # 3. Stage A
                stage_a_out = load_and_preprocess(wav_path, config=config.stage_a)

                # Audit Dump Stage A
                # We can manually dump here since we are the runner
                with open(os.path.join(output_dir, f"{example_id}_stage_a.json"), "w") as f:
                    # Basic metadata dump
                    json.dump(stage_a_out.meta.__dict__, f, indent=2, cls=NumpyEncoder)

                ma = calculate_stage_a_metrics(
                    stage_a_out.stems["mix"].audio if "mix" in stage_a_out.stems else list(stage_a_out.stems.values())[0].audio,
                    stage_a_out.meta.sample_rate,
                    stage_a_out.meta
                )
                example_res["stage_a_metrics"] = ma

                # 4. Stage B
                stage_b_out = extract_features(stage_a_out, config=config)

                # Audit Dump Stage B
                # Dump time grid and f0 main for inspection
                np.savetxt(os.path.join(output_dir, f"{example_id}_stage_b_f0.csv"),
                           np.column_stack([stage_b_out.time_grid, stage_b_out.f0_main]),
                           delimiter=",", header="time,frequency")

                if midi_path:
                    mb = calculate_stage_b_metrics(stage_b_out, midi_path)
                    example_res["stage_b_metrics"] = mb

                # 5. Stage C
                analysis_data = AnalysisData(
                    meta=stage_b_out.meta,
                    stem_timelines=stage_b_out.stem_timelines,
                )

                notes = apply_theory(analysis_data, config=config)

                # Audit Dump Stage C
                with open(os.path.join(output_dir, f"{example_id}_stage_c_notes.json"), "w") as f:
                    json.dump([n.__dict__ for n in notes], f, indent=2, cls=NumpyEncoder)

                if midi_path:
                    mc = calculate_stage_c_metrics(notes, midi_path)
                    example_res["stage_c_metrics"] = mc

                # 6. Stage D
                # update analysis_data with notes for Stage D
                analysis_data.notes = notes

                xml_content = quantize_and_render(notes, analysis_data, config=config)
                xml_path = os.path.join(output_dir, f"{example_id}.musicxml")
                with open(xml_path, "w") as f:
                    f.write(xml_content)

                if midi_path:
                    md = calculate_stage_d_metrics(xml_path, midi_path)
                    example_res["stage_d_metrics"] = md

            except Exception as e:
                print(f"    Error: {e}")
                traceback.print_exc()
                example_res["errors"].append(str(e))

            level_results.append(example_res)

        results[level_id] = level_results

    # Write summary
    with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    return results
