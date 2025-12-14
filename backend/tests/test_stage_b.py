import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from backend.pipeline.stage_b import extract_features, create_harmonic_mask, iterative_spectral_subtraction
from backend.pipeline.models import StageAOutput, MetaData, Stem, AudioQuality, FramePitch
from backend.pipeline.detectors import SwiftF0Detector, SACFDetector

class TestStageB:
    @pytest.fixture
    def sr(self):
        return 22050

    @pytest.fixture
    def hop_length(self):
        return 256

    @pytest.fixture
    def mock_stage_a_output(self, sr, hop_length):
        meta = MetaData(
            duration_sec=2.0,
            sample_rate=sr,
            hop_length=hop_length,
            audio_quality=AudioQuality.LOSSLESS,
            audio_path="test_audio.wav"
        )

        # Create dummy audio
        audio = np.zeros(int(2.0 * sr))
        stems = {
            "vocals": Stem(audio=audio, sr=sr, type="vocals"),
            "other": Stem(audio=audio, sr=sr, type="other")
        }

        return StageAOutput(
            audio_type="POLYPHONIC",
            meta=meta,
            stems=stems
        )

    def test_create_harmonic_mask(self, sr):
        # Create a dummy STFT (freqs x frames)
        n_fft = 2048
        n_frames = 10
        stft = np.ones((1025, n_frames), dtype=np.complex64)

        # Create a constant f0 curve at 440Hz
        f0_curve = np.full(n_frames, 440.0)

        # Updated signature: f0_hz, sr, n_fft, mask_width
        mask = create_harmonic_mask(f0_curve, sr, n_fft, mask_width=0.03)

        # Check if bins corresponding to 440Hz are masked
        fft_freqs = np.linspace(0, sr/2, 1025)

        # Find index for 440Hz
        idx_440 = np.argmin(np.abs(fft_freqs - 440.0))

        # The mask should be 0.0 at this index (masked out)
        assert mask[idx_440, 0] == 0.0, "Fundamental frequency should be masked"

        # Check harmonic (880Hz)
        idx_880 = np.argmin(np.abs(fft_freqs - 880.0))
        assert mask[idx_880, 0] == 0.0, "First harmonic should be masked"

    @patch("backend.pipeline.stage_b.SwiftF0Detector")
    @patch("backend.pipeline.stage_b.SACFDetector")
    def test_iterative_spectral_subtraction_flow(self, MockSACF, MockSwiftF0, sr, hop_length):
        """
        Verify the loop runs multiple times and stops when confidence is low.
        """
        # Setup Mocks
        primary = MockSwiftF0.return_value
        validator = MockSACF.return_value

        # Mock responses
        # Iteration 0: Strong note (440Hz)
        # Iteration 1: Weak note (330Hz) -> Should trigger stop if conf < termination

        n_frames = 100

        # Call 1: High confidence
        f0_1 = np.full(n_frames, 440.0)
        conf_1 = np.full(n_frames, 0.9)

        # Call 2: Low confidence (triggers stop in ISS if voiced_ratio < 0.05)
        # We need conf to be > 0.0 for voiced check.
        # ISS threshold is voiced_ratio < 0.05.
        # If conf_2 is 0.05 everywhere, voiced_ratio depends on (conf > 0.1).
        # 0.05 is not > 0.1. So voiced_ratio will be 0.
        # So it stops.

        f0_2 = np.full(n_frames, 330.0)
        conf_2 = np.full(n_frames, 0.05)

        primary.predict.side_effect = [(f0_1, conf_1), (f0_2, conf_2)]
        primary.hop_length = hop_length
        primary.n_fft = 2048 # needed for ISS stft?

        # validator.validate_curve.return_value = 0.8
        # ISS uses validator.predict
        validator.predict.return_value = (f0_1, conf_1) # Just pass validation

        audio = np.zeros(n_frames * hop_length)

        extracted = iterative_spectral_subtraction(
            audio, sr, primary, validator, max_polyphony=4
        )

        # Should have extracted 1 note (the second one failed check)
        assert len(extracted) == 1
        assert extracted[0][0][0] == 440.0

    @patch("backend.pipeline.stage_b.SwiftF0Detector")
    @patch("backend.pipeline.stage_b.SACFDetector")
    def test_extract_features_routing(self, MockSACF, MockSwiftF0, mock_stage_a_output):
        """
        Verify that Vocals go to SwiftF0 direct, and Other goes to ISS.
        """
        # Setup Mocks
        # SwiftF0 instance 1 (Vocals)
        # SwiftF0 instance 2 (Other - Primary)

        # We need to distinguish instances or just count calls.
        # extract_features instantiates detectors inside.

        mock_swift_instance = MockSwiftF0.return_value
        mock_sacf_instance = MockSACF.return_value

        # Mock predict to return zeros so it doesn't crash
        n_frames = int(mock_stage_a_output.meta.duration_sec * 22050 / 256) + 1
        mock_swift_instance.predict.return_value = (np.zeros(n_frames), np.zeros(n_frames))

        stage_b_out = extract_features(mock_stage_a_output)
        stems = stage_b_out.stem_timelines

        # Check that SwiftF0 was initialized
        # In optimized implementation, detectors are reused, so we expect at least 1 init.
        assert MockSwiftF0.call_count >= 1

        # Check that SACF was initialized
        assert MockSACF.call_count >= 1

        # Check that stems contains keys
        assert "vocals" in stems
        assert "other" in stems
