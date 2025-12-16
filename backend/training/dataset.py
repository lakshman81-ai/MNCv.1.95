"""
Dataset loaders for MIDI/MusicXML aligned audio.
Example: MAESTRO, MusicNet.
Offline only.
"""

from typing import List, Tuple, Any
import numpy as np

class AudioMidiDataset:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        # Placeholder
        return np.zeros(16000), {}
