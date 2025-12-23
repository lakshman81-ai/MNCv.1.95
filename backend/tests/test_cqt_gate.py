
import pytest
import numpy as np
from backend.pipeline.stage_b import _postprocess_candidates

def test_postprocess_no_librosa():
    """
    Verify behavior when cqt_ctx is None (Librosa missing).
    - Deduplication should still work.
    - CQT Gating and Harmonic Suppression should be disabled (candidates kept).
    """
    candidates = [
        (440.0, 0.9),
        (440.5, 0.8), # Near duplicate of 440.0
        (880.0, 0.85), # Octave (potential harmonic)
        (220.0, 0.1), # Weak (would be gated if noise floor was checked)
    ]

    processed = _postprocess_candidates(
        candidates,
        frame_idx=0,
        cqt_ctx=None,
        max_candidates=5,
        dup_cents=35.0,
        octave_cents=35.0
    )

    pitches = [p for p, c in processed]
    confs = [c for p, c in processed]

    # 1. Deduplication
    assert 440.0 in pitches
    assert 440.5 not in pitches

    # 2. Harmonic Suppression Disabled
    # Without CQT, we assume 880 is real to avoid killing valid octave intervals.
    assert 880.0 in pitches

    # 3. CQT Gating Disabled
    # Weak 220.0 is kept.
    assert 220.0 in pitches

    # 4. Normalization
    # Max conf was 0.9 (440.0).
    # 440.0 -> 1.0
    # 880.0 -> 0.85 / 0.9 = 0.944
    assert confs[0] == 1.0

def test_postprocess_with_mock_cqt():
    """
    Verify behavior when cqt_ctx is present.
    - Harmonic suppression should kill weak upper octaves.
    - Gating should kill weak fundamentals below noise floor.
    """
    # Mock context
    # Frequencies: 220, 440, 880
    freqs = np.array([220.0, 440.0, 880.0], dtype=np.float32)

    # Magnitudes (1 frame):
    # 220: 0.01 (Noise)
    # 440: 1.00 (Strong Fundamental)
    # 880: 0.10 (Weak Harmonic)
    mag = np.array([[0.01], [1.0], [0.1]], dtype=np.float32)

    ctx = {"mag": mag, "freqs": freqs}

    candidates = [
        (440.0, 0.9),
        (880.0, 0.8), # Candidate is strong-ish confidence, but weak CQT energy relative to 440
        (220.0, 0.5), # Medium confidence, but very low CQT energy (below median)
    ]

    # Median floor of [0.01, 1.0, 0.1] is 0.1.
    # Thresholds:
    # Gating: val < floor * ratio?
    #   220: 0.01 < 0.1 * 2.0 (0.2)? Yes. Gated.
    # Harmonic: hi/lo < drop_ratio?
    #   880/440: 0.1 / 1.0 = 0.1. 0.1 < 0.75? Yes. Suppressed.

    processed = _postprocess_candidates(
        candidates,
        frame_idx=0,
        cqt_ctx=ctx,
        max_candidates=5,
        cqt_support_ratio=2.0,
        cqt_gate_mul=0.1,
        harmonic_drop_ratio=0.75
    )

    pitches = [p for p, c in processed]

    # 440 kept
    assert 440.0 in pitches

    # 880 dropped (Harmonic suppression)
    assert 880.0 not in pitches

    # 220 gated (kept but heavily downweighted)
    # 220 conf -> 0.5 * 0.1 = 0.05
    # Renormalized: 0.05 / 0.9 = 0.055
    assert 220.0 in pitches
    p220_conf = next(c for p, c in processed if p == 220.0)
    assert p220_conf < 0.1

if __name__ == "__main__":
    test_postprocess_no_librosa()
    test_postprocess_with_mock_cqt()
