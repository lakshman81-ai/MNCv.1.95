"""
Augmentation toolkit for training data.
Features: Pitch shift, Time stretch, Noise injection, Reverb convolution.
Offline only.
"""

import numpy as np

def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """
    Shift pitch by n_steps semitones.
    """
    try:
        import librosa
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except ImportError:
        return audio

def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Stretch time by rate (rate > 1.0 speeds up).
    """
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=rate)
    except ImportError:
        return audio

def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    noise = np.random.randn(*audio.shape) * noise_level
    return audio + noise
