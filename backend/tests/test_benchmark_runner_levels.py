import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.benchmarks import benchmark_runner


def test_resolve_levels_all_returns_full_order():
    assert benchmark_runner.resolve_levels("all") == benchmark_runner.LEVEL_ORDER


def test_resolve_levels_maps_l5_to_sublevels():
    assert benchmark_runner.resolve_levels("L5") == ["L5.1", "L5.2"]


def test_resolve_levels_passthrough_for_other_levels():
    assert benchmark_runner.resolve_levels("L3") == ["L3"]
