"""Proxy module for Stage B.

This module exists within the ``backend.pipeline`` namespace so that tests
importing ``backend.pipeline.stage_b`` find the Stage B implementation.  It
re-exports everything from the top-level ``stage_b`` module.
"""

from stage_b import *  # type: ignore  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]