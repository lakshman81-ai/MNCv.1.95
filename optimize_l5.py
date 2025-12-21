"""
Optimize L5 (Deprecated)

This script is deprecated. Please use backend/pipeline/tune_runner.py instead.
It will forward all arguments to tune_runner.py.
"""

import sys
import runpy
import warnings

def main():
    warnings.warn(
        "optimize_l5.py is deprecated and will be removed. "
        "Please use 'python -m backend.pipeline.tune_runner' instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Forward execution to tune_runner
    # We must ensure we run it as a module
    sys.argv[0] = "backend/pipeline/tune_runner.py"
    runpy.run_module("backend.pipeline.tune_runner", run_name="__main__", alter_sys=True)

if __name__ == "__main__":
    main()
