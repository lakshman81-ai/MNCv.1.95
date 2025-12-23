"""
Stage B — Optional Neural Transcription Path (Onsets & Frames)

This module provides a direct audio-to-notes path using an Onsets & Frames style model,
bypassing the traditional Stage B pitch tracking + Stage C segmentation pipeline.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple, Any

import numpy as np

from .models import NoteEvent, AnalysisData
from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _safe_import_torch() -> bool:
    try:
        import torch
        return True
    except Exception:
        return False


class OnsetsFramesModel:
    """
    Placeholder for the actual PyTorch model.
    In a real deployment, this would load a checkpoint.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Dimensions: 88 piano keys usually.
        # This is a stub.

    def infer(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference.
        Returns:
            onsets: (T, 88) probabilities
            frames: (T, 88) probabilities
            velocities: (T, 88) or None
        """
        # STUB: Return zeros
        # In real impl: mel_spectrogram -> model -> logits -> sigmoid
        n_frames = int(len(audio) / 512) # rough guess
        return (
            np.zeros((n_frames, 88), dtype=np.float32),
            np.zeros((n_frames, 88), dtype=np.float32),
            np.zeros((n_frames, 88), dtype=np.float32)
        )


_MODEL_CACHE: Optional[OnsetsFramesModel] = None


def load_model(config: PipelineConfig):
    conf = getattr(config.stage_b, "onsets_and_frames", {}) or {}
    ckpt = conf.get("checkpoint_path")
    if not ckpt:
        return None, {"run": False, "reason": "no_checkpoint"}

    # If you later add checkpoint loading, do it here.
    # For now, still return None until real loading exists.
    return None, {"run": False, "reason": "checkpoint_loading_not_implemented"}


def decode_notes(
    onsets: np.ndarray,
    frames: np.ndarray,
    velocities: Optional[np.ndarray],
    onset_thresh: float = 0.5,
    frame_thresh: float = 0.5,
    min_dur_frames: int = 1,
    hop_length: int = 512,
    sr: int = 16000,
    midi_offset: int = 21, # A0 for piano
) -> List[NoteEvent]:
    """
    Decode onset/frame probabilities into NoteEvents.
    Simple greedy decoding:
      1. Find onset peaks > onset_thresh.
      2. Extend forward while frame > frame_thresh.
    """
    notes = []
    n_frames, n_pitches = onsets.shape

    # Time conversion
    frame_time = hop_length / float(sr)

    for p in range(n_pitches):
        # Find peaks in onset curve
        # Simple thresholding for now (scipy.signal.find_peaks would be better)
        onset_col = onsets[:, p]
        frame_col = frames[:, p]

        # Identify onset candidates
        # (val > thresh) AND (val > prev) AND (val > next)
        is_peak = (onset_col > onset_thresh)
        is_peak[1:-1] &= (onset_col[1:-1] > onset_col[:-2]) & (onset_col[1:-1] > onset_col[2:])

        peak_indices = np.where(is_peak)[0]

        for onset_idx in peak_indices:
            # Determine end index
            # Walk forward from onset_idx
            end_idx = onset_idx + 1
            while end_idx < n_frames and frame_col[end_idx] > frame_thresh:
                end_idx += 1

            # Check duration
            if (end_idx - onset_idx) < min_dur_frames:
                continue

            # If velocity head exists, use it, else default
            vel = 0.8
            if velocities is not None:
                vel = float(velocities[onset_idx, p])

            midi_note = p + midi_offset
            pitch_hz = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

            notes.append(NoteEvent(
                start_sec=float(onset_idx * frame_time),
                end_sec=float(end_idx * frame_time),
                midi_note=midi_note,
                pitch_hz=pitch_hz,
                confidence=float(onset_col[onset_idx]),
                velocity=vel,
                voice=1
            ))

    # Sort by start time
    notes.sort(key=lambda n: n.start_sec)
    return notes


def transcribe_onsets_frames(
    audio: np.ndarray,
    sr: int,
    config: PipelineConfig
) -> Tuple[List[NoteEvent], Dict[str, Any]]:
    """
    Main entry point for O&F transcription.
    """
    diag = {"run": False, "reason": "disabled"}

    if not config.stage_b.onsets_and_frames.get("enabled", False):
        return [], diag

    if not _safe_import_torch():
        logger.warning("Onsets & Frames enabled but torch not available.")
        diag["reason"] = "no_torch"
        return [], diag

    model, load_diag = load_model(config)
    if model is None:
        return [], load_diag

    # Run inference
    # Note: Model usually expects 16k mono. We should resample if needed.
    # For now, pass as is (assuming model handles or stub).
    try:
        onsets, frames, velocities = model.infer(audio, sr)
    except Exception as e:
        logger.error(f"O&F Inference failed: {e}")
        diag["reason"] = f"inference_error: {e}"
        return [], diag

    # Decode
    cfg = config.stage_b.onsets_and_frames

    # Calculate min frames
    # hop ~ 32ms for 512/16k.
    # min_ms = 30 -> ~1 frame.
    min_ms = cfg.get("min_note_duration_ms", 30)
    hop = 512 # Assumption for the model
    # Recalculate based on SR if model is strict, but usually fixed hop in ms.

    notes = decode_notes(
        onsets,
        frames,
        velocities,
        onset_thresh=float(cfg.get("onset_threshold", 0.5)),
        frame_thresh=float(cfg.get("frame_threshold", 0.5)),
        min_dur_frames=1, # simplified
        hop_length=hop,
        sr=sr
    )

    diag["run"] = True
    diag["note_count"] = len(notes)
    return notes, diag


def transcribe_basic_pitch_to_notes(
    audio: np.ndarray,
    sr: int,
    tmp_dir: Any,  # pathlib.Path
    *,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length_ms: float,
    min_hz: float,
    max_hz: float,
    melodia_trick: bool,
) -> List[NoteEvent]:
    """
    Wrapper for Basic Pitch inference.
    Isolates the heavy dependency inside this function.
    """
    # Import inside function so classic mode has zero TF/basic-pitch import cost
    try:
        from basic_pitch.inference import predict
    except Exception as e:
        raise RuntimeError("basic-pitch not installed") from e

    import soundfile as sf
    from pathlib import Path

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / "basic_pitch_input.wav"

    # Basic Pitch reads from file path; it will load mono at 22.05k internally.
    sf.write(str(wav_path), audio.astype(np.float32), sr)

    model_output, midi_data, note_events = predict(
        str(wav_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length_ms,
        minimum_frequency=min_hz,
        maximum_frequency=max_hz,
        melodia_trick=melodia_trick,
    )

    # note_events tuples: (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bend_list)
    notes: List[NoteEvent] = []
    for start_s, end_s, pitch_midi, amplitude, _pitch_bends in note_events:
        vel = int(np.clip(round(float(amplitude) * 127.0), 1, 127))
        # Frequency from MIDI
        pitch_hz = 440.0 * (2.0 ** ((pitch_midi - 69.0) / 12.0))

        notes.append(NoteEvent(
            start_sec=float(start_s),
            end_sec=float(end_s),
            midi_note=int(pitch_midi),
            pitch_hz=pitch_hz,
            confidence=1.0, # Basic Pitch implies high confidence if emitted
            velocity=float(vel) / 127.0, # NoteEvent expects 0-1
            voice=1,
            dynamic="mf" # Default
        ))

    # Clean up temp file?
    # Usually safer to leave for OS cleanup or explicit cleanup context,
    # but here we can try to remove to avoid clutter.
    try:
        if wav_path.exists():
            wav_path.unlink()
    except Exception:
        pass

    return notes
