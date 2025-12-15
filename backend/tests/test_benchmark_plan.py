import pytest

from backend.benchmarks.benchmark_runner import accuracy_benchmark_plan


@pytest.fixture(scope="module")
def plan():
    return accuracy_benchmark_plan()


def test_plan_has_all_sections(plan):
    expected_sections = {
        "end_to_end",
        "stage_a",
        "stage_b",
        "stage_c",
        "stage_d",
        "ablation",
        "regression",
        "profiling",
    }
    assert expected_sections.issubset(plan.keys())


def test_end_to_end_expectations(plan):
    end_to_end = plan["end_to_end"]
    assert set(end_to_end["scenarios"]) == {
        "clean_piano",
        "dense_chords",
        "percussive_passages",
        "noisy_inputs",
    }
    assert set(end_to_end["outputs"]) == {"musicxml", "midi_bytes", "analysis_timelines"}
    assert "stage_A_to_D_flow" in end_to_end["goals"]


def test_stage_specific_expectations(plan):
    assert {"sample_rate_targets", "loudness_normalization"}.issubset(set(plan["stage_a"]["toggles"]))
    assert {"f0_precision", "f0_recall", "voicing_error"}.issubset(set(plan["stage_b"]["metrics"]))
    assert {"hmm", "threshold"}.issubset(set(plan["stage_c"]["segmentation_modes"]))
    assert "beat_alignment_error" in plan["stage_d"]["metrics"]


def test_regression_and_profiling_expectations(plan):
    assert plan["regression"]["alerts"] is True
    assert "accuracy_delta" in plan["regression"]["thresholds"]
    assert "stage_timings" in plan["profiling"]["hooks"]
    assert plan["profiling"]["purpose"] == "contextualize_benchmark_results"
