
import numpy as np
import pytest
from unittest.mock import patch
from backend.pipeline.detectors import _autocorr_pitch_per_frame

class TestDetectorsFFTConsistency:
    @pytest.fixture
    def random_frames(self):
        # 100 frames of 2048 samples, random noise
        return np.random.randn(100, 2048).astype(np.float32)

    def test_autocorr_backend_consistency(self, random_frames):
        """
        Verify that _autocorr_pitch_per_frame produces consistent results
        regardless of whether scipy.fft or numpy.fft is used.
        """
        sr = 44100
        fmin = 60
        fmax = 2000

        # 1. Run with default (likely SciPy if installed)
        f0_default, conf_default = _autocorr_pitch_per_frame(random_frames, sr, fmin, fmax)

        # 2. Force NumPy backend by mocking _FFT_LIB in detectors module
        # We need to patch the module-level variable _FFT_LIB
        with patch("backend.pipeline.detectors._FFT_LIB", np.fft):
            f0_numpy, conf_numpy = _autocorr_pitch_per_frame(random_frames, sr, fmin, fmax)

        # 3. Compare results
        # F0 should be very close (peak index shouldn't change for strong peaks, but for noise it might jump)
        # For random noise, peaks are random, so index might shift if values differ slightly.
        # But we want to ensure the logic holds.
        # Let's check similarity.

        # Exact match for float32 might be too strict due to 1e-3 diff in FFT.
        # But peak picking `argmax` is discrete.
        # Let's count mismatches.

        mismatches = np.sum(f0_default != f0_numpy)
        print(f"F0 Mismatches: {mismatches} / {len(f0_default)}")

        # Tolerable mismatch rate for noise (chaos)
        # If input was a sine wave, it should be 0.
        # But for noise, slight numerical diffs can swap nearby peaks.
        # Assert mismatches < 10% ?
        assert mismatches < len(f0_default) * 0.1, "Too many F0 mismatches between backends"

        # Check confidence correlation
        assert np.allclose(conf_default, conf_numpy, atol=1e-2), "Confidence values diverge too much"

    def test_autocorr_sine_wave_consistency(self):
        """Verify consistency on a stable signal (sine wave)."""
        sr = 44100
        fmin = 60
        fmax = 2000
        t = np.linspace(0, 2048/sr, 2048)
        # 440 Hz sine
        sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        frames = np.tile(sine, (10, 1)) # 10 identical frames

        f0_default, conf_default = _autocorr_pitch_per_frame(frames, sr, fmin, fmax)

        with patch("backend.pipeline.detectors._FFT_LIB", np.fft):
            f0_numpy, conf_numpy = _autocorr_pitch_per_frame(frames, sr, fmin, fmax)

        # Should be identical for strong signal
        np.testing.assert_allclose(f0_default, f0_numpy, rtol=1e-5)
        np.testing.assert_allclose(conf_default, conf_numpy, atol=1e-4)
