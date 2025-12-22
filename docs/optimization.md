# Optimization Guide

## Iterative Tuning

The `optimize_l5.py` script has been removed. Use `backend/pipeline/tune_runner.py` for iterative optimization of L5.1 and L5.2 benchmarks.

### Usage

```bash
# Tune L5.1 (Kal Ho Na Ho)
python -m backend.pipeline.tune_runner --level L5.1 --target-f1 0.65

# Tune L5.2 (Tumhare Hi Rahenge)
python -m backend.pipeline.tune_runner --level L5.2 --target-f1 0.50
```

This runner will iteratively tweak parameters (like thresholds, harmonic peeling settings, etc.) to maximize the F1 score.

## Benchmarking

To run the standard benchmark suite (L0-L4), use the unified runner:

```bash
python -m backend.benchmarks.benchmark_runner --level all
```

Or specific levels:

```bash
python -m backend.benchmarks.benchmark_runner --level L4
```
