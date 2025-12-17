"""Lightweight structured logging for pipeline stages.

This module centralizes timing and JSONL logging so benchmarks and
API calls can emit consistent diagnostics without pulling in heavy
logging frameworks. All writes are best-effort and should never raise
inference-breaking exceptions.
"""
from __future__ import annotations

import json
import os
import time
import importlib.util
from dataclasses import asdict
from typing import Any, Dict, Optional


class PipelineLogger:
    """Structured logger that emits JSONL events and timing summaries."""

    def __init__(self, base_dir: str = "results", run_name: Optional[str] = None):
        self.base_dir = base_dir
        self.run_name = run_name or f"run_{int(time.time())}"
        self.run_dir = os.path.join(self.base_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.logs_path = os.path.join(self.run_dir, "logs.jsonl")
        self.timing_path = os.path.join(self.run_dir, "timing.json")
        self._timing: Dict[str, float] = {}
        self._start_time = time.perf_counter()
        self.log_event(
            "pipeline",
            "start",
            {
                "run_dir": self.run_dir,
                "dependencies": self.dependency_snapshot(
                    ["torch", "crepe", "demucs", "librosa", "pyloudnorm"]
                ),
            },
        )

    @staticmethod
    def dependency_snapshot(modules: Optional[list[str]] = None) -> Dict[str, bool]:
        """Return availability flags for the requested modules."""
        snapshot: Dict[str, bool] = {}
        for name in modules or []:
            try:
                snapshot[name] = importlib.util.find_spec(name) is not None
            except Exception:
                snapshot[name] = False
        return snapshot

    def log_event(self, stage: str, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "stage": stage,
            "event": event,
            "timestamp": time.time(),
        }
        if payload:
            for key, value in payload.items():
                try:
                    json.dumps(value, default=str)  # validate serializable
                    entry[key] = value
                except Exception:
                    entry[key] = str(value)
        try:
            with open(self.logs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            # Never break inference due to logging failures
            pass

    def record_timing(self, stage: str, duration_s: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._timing[stage] = float(duration_s)
        payload = {"duration_s": float(duration_s)}
        if metadata:
            payload.update(metadata)
        self.log_event(stage, "timing", payload)

    def finalize(self) -> None:
        if "total" not in self._timing:
            self._timing["total"] = float(time.perf_counter() - self._start_time)
        try:
            with open(self.timing_path, "w", encoding="utf-8") as f:
                json.dump(self._timing, f, indent=2)
        except Exception:
            pass

    @property
    def timing(self) -> Dict[str, float]:
        return dict(self._timing)

    def emit_config(self, stage: str, config_obj: Any, extras: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"config": {}}
        try:
            payload["config"] = asdict(config_obj)
        except Exception:
            payload["config"] = str(config_obj)
        if extras:
            payload.update(extras)
        self.log_event(stage, "config", payload)

    def _safe_json_default(self, o):
        try:
            import numpy as _np
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
        except Exception:
            pass

        # dataclasses
        try:
            from dataclasses import asdict as _asdict, is_dataclass as _isdc
            if _isdc(o):
                return _asdict(o)
        except Exception:
            pass

        # enums
        v = getattr(o, "value", None)
        if v is not None:
            return v

        return str(o)

    def write_json(self, filename: str, obj) -> None:
        try:
            path = os.path.join(self.run_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2, default=self._safe_json_default)
        except Exception as e:
            self.log_event(stage="logger", event="artifact_write_failed",
                           payload={"filename": filename, "error": str(e)})

    def write_text(self, filename: str, text: str) -> None:
        try:
            path = os.path.join(self.run_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            self.log_event(stage="logger", event="artifact_write_failed",
                           payload={"filename": filename, "error": str(e)})

    def write_bytes(self, filename: str, data: bytes) -> None:
        try:
            path = os.path.join(self.run_dir, filename)
            with open(path, "wb") as f:
                f.write(data)
        except Exception as e:
            self.log_event(stage="logger", event="artifact_write_failed",
                           payload={"filename": filename, "error": str(e)})
