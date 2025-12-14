"""Proxy module for Stage A.

This module exists within the ``backend.pipeline`` namespace so that tests
importing ``backend.pipeline.stage_a`` find the Stage A implementation.  It
simply re-exports everything from the top-level ``stage_a`` module.
"""

# Import all public attributes from the root-level stage_a module.  We use a
# wildcard import here to propagate variables like TARGET_LUFS and functions
# such as load_and_preprocess.  The noqa directives silence style warnings
# about wildcard imports in this stub.
from stage_a import *  # type: ignore  # noqa: F401,F403

__all__ = [
    name for name in globals().keys() if not name.startswith("_")
]