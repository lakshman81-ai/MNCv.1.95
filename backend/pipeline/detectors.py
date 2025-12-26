# backend/pipeline/detectors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import warnings
import numpy as np
import scipy.signal

# Bolt Optimization: Prefer scipy.fft over numpy.fft for speed (1.5-2x faster)
try:
    import scipy.fft
    _FFT_LIB = scipy.fft
except ImportError:
    _FFT_LIB = np.fft


# --------------------------------------------------------------------------------------
# Optional dependencies (never fail import of this module)
# --------------------------------------------------------------------------------------
try:
    import librosa  # type: ignore
except Exception as e:  # pragma: no cover
    librosa = None  # type: ignore
    _LIBROSA_IMPORT_ERR = e  # type: ignore


# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------
def hz_to_midi(hz: float) -> float:
    if hz <= 0.0:
        return 0.0
    return 69.0 + 12.0 * float(np.log2(hz / 440.0))


def midi_to_hz(m: int) -> float:
    """Convert MIDI pitch to frequency in Hz."""
    return 440.0 * 2 ** ((float(m) - 69.0) / 12.0)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _frame_audio(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) <= 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    if len(y) < frame_length:
        pad = frame_length - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    n_frames = 1 + (len(y) - frame_length) // hop_length
    if n_frames <= 0:
        n_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, frame_length),
        strides=(y.strides[0] * hop_length, y.strides[0]),
        writeable=False,
    )
    return np.asarray(frames, dtype=np.float32)


def _autocorr_pitch_per_frame(
    frames: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight ACF pitch estimator: returns (f0, conf) per frame.
    conf ~ normalized ACF peak (0..1).
    Optimized: Uses vectorized FFT-based autocorrelation for >3x speedup.
    """
    n_frames, frame_length = frames.shape
    f0 = np.zeros((n_frames,), dtype=np.float32)
    conf = np.zeros((n_frames,), dtype=np.float32)

    if n_frames == 0:
        return f0, conf

    # Lags corresponding to frequency bounds
    lag_min = max(1, int(sr / max(fmax, 1e-6)))
    lag_max = max(lag_min + 1, int(sr / max(fmin, 1e-6)))
    lag_max = min(lag_max, frame_length - 2)

    if lag_min >= lag_max:
        return f0, conf

    # 1. Apply window
    win = np.hanning(frame_length).astype(np.float32)
    x = frames * win

    # 2. Subtract mean (per frame)
    x = x - np.mean(x, axis=1, keepdims=True)

    # 3. Compute energy (denom) for later validation
    denom = np.sum(x**2, axis=1) + 1e-12

    # 4. Vectorized Autocorrelation using FFT (Batched to manage memory)
    # Pad to >= 2*L - 1 to get linear convolution (Wiener-Khinchin)
    n_fft = 2 ** int(np.ceil(np.log2(2 * frame_length - 1)))

    # Process in chunks to avoid OOM on long audio files
    BATCH_SIZE = 2000

    for start_idx in range(0, n_frames, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, n_frames)
        x_batch = x[start_idx:end_idx]
        denom_batch = denom[start_idx:end_idx]

        # FFT based autocorrelation
        # Use scipy.fft if available (via _FFT_LIB)
        # Use next_fast_len for optimal FFT size if using SciPy
        fft_len = n_fft
        if _FFT_LIB is not np.fft and hasattr(_FFT_LIB, "next_fast_len"):
             fft_len = _FFT_LIB.next_fast_len(n_fft)

        X = _FFT_LIB.rfft(x_batch, n=fft_len, axis=1)
        P = X * np.conj(X)
        ac = _FFT_LIB.irfft(P, n=fft_len, axis=1)

        # 5. Extract non-negative lags (0 to frame_length-1)
        # The first 'frame_length' samples of irfft result correspond to lags 0..L-1
        ac = ac[:, :frame_length]
        ac0 = ac[:, 0] + 1e-12

        # 6. Peak picking in the valid lag range
        seg = ac[:, lag_min:lag_max]
        if seg.shape[1] == 0:
            continue

        k_seg = np.argmax(seg, axis=1)
        k = k_seg + lag_min
        # Advanced indexing relative to the batch
        peaks = ac[np.arange(len(x_batch)), k]

        # 7. Compute confidence and filter
        peaks_norm = np.clip(peaks / ac0, 0.0, 1.0)
        valid = (denom_batch > 1e-10) & (peaks_norm > 0.0)

        # 8. Assign to global arrays
        # Map batch indices to global indices
        f0_batch = np.zeros_like(peaks_norm)
        conf_batch = np.zeros_like(peaks_norm)

        f0_batch[valid] = sr / k[valid]
        conf_batch[valid] = peaks_norm[valid]

        f0[start_idx:end_idx] = f0_batch
        conf[start_idx:end_idx] = conf_batch

    return f0, conf


def _autocorr_pitch_per_frame_safe(
    frames: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterative 'safe' ACF fallback for robustness.
    Includes explicit mean removal, zero-lag energy normalization,
    and strict neighbor checks for peak picking.
    """
    n_frames, frame_length = frames.shape
    f0_out = np.zeros((n_frames,), dtype=np.float32)
    conf_out = np.zeros((n_frames,), dtype=np.float32)

    if n_frames == 0:
        return f0_out, conf_out

    lag_min = max(1, int(sr / max(fmax, 1e-6)))
    # Ensure lag_max + 1 < frame_length for safe indexing
    # because we need norm_corr[lag+1]
    lag_max = min(int(sr / max(fmin, 1e-6)), frame_length - 3)

    if lag_min >= lag_max:
        return f0_out, conf_out

    # Numba optimization hook (if available, otherwise pure Python)
    # Since we can't easily rely on numba being present, we write pure python optimized
    # loop or accept it is slow (fallback).

    for i in range(n_frames):
        frame = frames[i].astype(np.float32)
        # DC removal
        frame = frame - np.mean(frame)

        # ACF
        # mode='full' gives 2*L-1 output. lag 0 is at index L-1.
        # We only need positive lags.
        # This is slow per frame but robust.
        corr = np.correlate(frame, frame, mode="full")[len(frame)-1:]

        c0 = float(corr[0])
        if c0 <= 1e-12:
            continue

        # Normalize by zero-lag energy
        norm_corr = corr / (c0 + 1e-12)

        best_lag = 0
        max_val = 0.0

        # Iterative peak picking with bounds checks
        # range(lag_min, lag_max + 1) covers lags we want to check.
        # We need neighbor access lag-1 and lag+1.
        # Since lag_min >= 1, lag-1 >= 0.
        # Since lag_max <= len-3, lag+1 <= len-2.

        for lag in range(lag_min, lag_max + 1):
            val = float(norm_corr[lag])
            if val > threshold and val > max_val:
                # Safe neighbor check (local maximum)
                if norm_corr[lag - 1] < val and norm_corr[lag + 1] < val:
                    max_val = val
                    best_lag = lag

        if best_lag > 0:
            f0_out[i] = sr / best_lag
            conf_out[i] = max_val

    return f0_out, conf_out


# --------------------------------------------------------------------------------------
# Public polyphonic helpers (imported by tests / used by Stage B)
# --------------------------------------------------------------------------------------
def create_harmonic_mask(
    f0_hz: np.ndarray,
    sr: int,
    n_fft: int,
    mask_width: float = 0.03,
    n_harmonics: int = 8,
    min_band_hz: float = 6.0,
    frequency_aware_width: bool = False,
) -> np.ndarray:
    """
    Create a time-frequency mask that zeros bins around harmonics of f0.
    Returns mask shape: (n_fft//2 + 1, n_frames). 1.0 = keep, 0.0 = remove.

    If frequency_aware_width is True, low frequencies get wider masks.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float32).reshape(-1)
    n_frames = int(f0_hz.shape[0])
    n_bins = n_fft // 2 + 1

    # Optimization: Work in (n_frames, n_bins) to allow contiguous row updates
    # which is significantly faster than strided column updates.
    # We transpose back to (n_bins, n_frames) at the end.
    mask_T = np.ones((n_frames, n_bins), dtype=np.float32)

    # Frequency per bin
    bin_hz = float(sr) / float(n_fft)

    for t in range(n_frames):
        f0 = float(f0_hz[t])
        if f0 <= 0.0 or not np.isfinite(f0):
            continue

        for h in range(1, n_harmonics + 1):
            fh = f0 * h
            if fh >= float(sr) / 2.0:
                break

            if frequency_aware_width:
                 # Wider for low notes, narrower for high notes
                 # fh is scalar here (float), so comparison is safe.
                 # Let's boost mask width for low freqs (<200Hz)
                 width_factor = 1.0
                 if fh < 200.0:
                     width_factor = 1.0 + (200.0 - fh) / 100.0
                 bw = max(float(min_band_hz), abs(mask_width * width_factor) * fh)
            else:
                bw = max(float(min_band_hz), abs(mask_width) * fh)

            lo = fh - bw
            hi = fh + bw

            # Vectorized index calculation
            # lo <= i * bin_hz <= hi  =>  lo/bin_hz <= i <= hi/bin_hz
            idx_lo = int(np.ceil(lo / bin_hz))
            idx_hi = int(np.floor(hi / bin_hz))

            # Clamp to valid range
            idx_lo = max(0, idx_lo)
            idx_hi = min(n_bins - 1, idx_hi)

            if idx_lo <= idx_hi:
                # Contiguous slice update on the transposed array
                mask_T[t, idx_lo : idx_hi + 1] = 0.0

    return mask_T.T


def iterative_spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    primary_detector: "BasePitchDetector",
    validator_detector: Optional["BasePitchDetector"] = None,
    max_polyphony: int = 8,
    mask_width: float = 0.03,
    min_mask_width: float = 0.02,
    max_mask_width: float = 0.08,
    mask_growth: float = 1.1,
    mask_shrink: float = 0.9,
    harmonic_snr_stop_db: float = 3.0,
    residual_rms_stop_ratio: float = 0.08,
    residual_flatness_stop: float = 0.45,
    validator_cents_tolerance: float = 50.0,
    validator_agree_window: int = 5,
    validator_disagree_decay: float = 0.6,
    validator_min_agree_frames: int = 2,
    validator_min_disagree_frames: int = 2,
    max_harmonics: int = 12,
    audio_path: Optional[str] = None,
    # Adaptive ISS params (Feature E)
    iss_adaptive: bool = False,
    strength_min: float = 0.8,
    strength_max: float = 1.2,
    flatness_thresholds: Optional[List[float]] = None,
    # Frequency-aware masking override
    use_freq_aware_masks: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Iterative spectral subtraction ("peeling") with optional adaptive scheduling.
    """
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return []

    if flatness_thresholds is None:
        flatness_thresholds = [0.3, 0.6]

    hop = _safe_int(getattr(primary_detector, "hop_length", 512), 512)
    n_fft = _safe_int(getattr(primary_detector, "n_fft", 2048), 2048)
    bin_hz = float(sr) / float(n_fft)

    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    residual = y.copy()
    base_rms = float(np.sqrt(np.mean(residual**2)) + 1e-9)
    current_mask_width = float(mask_width)

    def _spectral_flatness(magnitude: np.ndarray) -> float:
        mag = np.asarray(magnitude, dtype=np.float32)
        if mag.size == 0:
            return 0.0
        mag = np.maximum(mag, 1e-9)
        log_mean = float(np.mean(np.log(mag)))
        geo = float(np.exp(log_mean))
        arith = float(np.mean(mag)) + 1e-12
        return float(geo / arith)

    def _rolling_mean(arr: np.ndarray, win: int) -> np.ndarray:
        if win <= 1:
            return arr.astype(np.float32)
        kernel = np.ones(win, dtype=np.float32) / float(win)
        return np.convolve(arr.astype(np.float32), kernel, mode="same")

    def _consecutive_lengths(flags: np.ndarray) -> np.ndarray:
        lengths = np.zeros_like(flags, dtype=np.int32)
        run = 0
        for i, v in enumerate(flags):
            run = run + 1 if v else 0
            lengths[i] = run
        return lengths

    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * np.clip(t, 0.0, 1.0)

    def _layer_similarity(
        f0_a: np.ndarray,
        conf_a: np.ndarray,
        f0_b: np.ndarray,
        conf_b: np.ndarray,
        conf_thr: float = 0.1,
        cents_tol: float = 35.0,
    ) -> float:
        # Check similarity on voiced overlap
        mask = (conf_a > conf_thr) & (conf_b > conf_thr) & (f0_a > 0.0) & (f0_b > 0.0)
        overlap_count = np.count_nonzero(mask)
        if overlap_count < 10:
            return 0.0

        diffs = np.abs(1200.0 * np.log2((f0_a[mask] + 1e-9) / (f0_b[mask] + 1e-9)))
        matches = np.count_nonzero(diffs <= cents_tol)
        return float(matches) / float(overlap_count)

    for _layer in range(int(max_polyphony)):
        f0, conf = primary_detector.predict(residual, audio_path=audio_path)

        if f0 is None or conf is None:
            break
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)

        # Optional validator gate
        if validator_detector is not None:
            try:
                vf0, vconf = validator_detector.predict(residual, audio_path=audio_path)
                vf0 = np.asarray(vf0, dtype=np.float32).reshape(-1)
                vconf = np.asarray(vconf, dtype=np.float32).reshape(-1)

                if np.mean((vf0 > 0.0).astype(np.float32)) < 0.05:
                    break

                with np.errstate(divide="ignore", invalid="ignore"):
                    cents = 1200.0 * np.log2((f0 + 1e-9) / (vf0 + 1e-9))
                agree_raw = (np.abs(cents) <= float(validator_cents_tolerance)) & (f0 > 0.0) & (vf0 > 0.0)

                agree_smooth = _rolling_mean(agree_raw.astype(np.float32), int(max(1, validator_agree_window)))
                agree_runs = _consecutive_lengths(agree_raw)
                disagree_runs = _consecutive_lengths(~agree_raw & (f0 > 0.0) & (vf0 > 0.0))

                stable_agree = agree_runs >= int(max(1, validator_min_agree_frames))
                stable_disagree = disagree_runs >= int(max(1, validator_min_disagree_frames))

                gate = agree_smooth + (1.0 - agree_smooth) * float(validator_disagree_decay)
                gate = np.clip(gate, 0.0, 1.0)

                gate = np.where(stable_disagree, gate * float(validator_disagree_decay), gate)
                gate = np.where(~stable_agree & ~stable_disagree, gate * 0.8, gate)

                conf = conf * gate.astype(np.float32)
            except Exception:
                pass

        voiced_ratio = float(np.mean((conf > 0.1).astype(np.float32)))
        if voiced_ratio < 0.05:
            break

        # Check similarity to previous layer to prevent duplicates
        if layers:
            prev_f0, prev_conf = layers[-1]
            n_cmp = min(len(f0), len(prev_f0))
            sim = _layer_similarity(
                f0[:n_cmp], conf[:n_cmp], prev_f0[:n_cmp], prev_conf[:n_cmp],
                conf_thr=0.1, cents_tol=35.0
            )
            if sim >= 0.85:
                # Stop if new layer is too similar to the previous one (duplicate extraction)
                break

        # STFT -> apply harmonic mask -> iSTFT
        try:
            n_fft_eff = int(min(n_fft, max(32, len(residual))))
            hop_eff = int(max(1, min(hop, n_fft_eff // 2)))

            f, t, Z = scipy.signal.stft(
                residual,
                fs=sr,
                nperseg=n_fft_eff,
                noverlap=max(0, n_fft_eff - hop_eff),
                boundary="zeros",
                padded=True,
            )
            n_frames = Z.shape[1]
            if f0.shape[0] != n_frames:
                if f0.shape[0] < n_frames:
                    pad = n_frames - f0.shape[0]
                    f0 = np.pad(f0, (0, pad))
                    conf = np.pad(conf, (0, pad))
                else:
                    f0 = f0[:n_frames]
                    conf = conf[:n_frames]

            # Adaptive Schedule (Feature E)
            magnitude = np.abs(Z).astype(np.float32)
            flatness = _spectral_flatness(magnitude)

            eff_mask_width = current_mask_width
            eff_strength_scale = 1.0

            if iss_adaptive:
                # E2) Adapt mask/subtraction schedule
                # If flatness high (noisy): widen mask and increase subtraction slowly
                # If harmonic energy high (stable): narrow mask and reduce subtraction (protect melody)

                # Normalize flatness to [0, 1] relative to thresholds
                f_norm = (flatness - flatness_thresholds[0]) / (flatness_thresholds[1] - flatness_thresholds[0] + 1e-9)
                f_norm = np.clip(f_norm, 0.0, 1.0)

                # Widen mask for noisy content
                eff_mask_width = _lerp(float(min_mask_width), float(max_mask_width), f_norm)

                # Strength adaptation
                # (Logic inverted from flatness: low flatness = high harmonicity)
                harmonicity = 1.0 - f_norm
                eff_strength_scale = _lerp(float(strength_max), float(strength_min), harmonicity)
            else:
                eff_mask_width = np.clip(current_mask_width, min_mask_width, max_mask_width)

            f0_valid = f0[f0 > 0.0]
            median_f0 = float(np.median(f0_valid)) if f0_valid.size else 0.0
            max_possible_h = int(float(sr) / 2.0 / max(median_f0, 1e-6)) if median_f0 > 0 else 1
            adaptive_harmonics = int(max(1, min(int(max_harmonics), max_possible_h)))

            min_band = max(bin_hz, float(eff_mask_width) * max(median_f0, 1.0) * 0.5)

            mask = create_harmonic_mask(
                f0_hz=f0,
                sr=sr,
                n_fft=n_fft,
                mask_width=float(eff_mask_width),
                n_harmonics=adaptive_harmonics,
                min_band_hz=min_band,
                frequency_aware_width=use_freq_aware_masks,
            )

            harmonic_energy = float(np.mean(magnitude * (1.0 - mask)))
            residual_energy = float(np.mean(magnitude * mask))
            harmonic_snr = 10.0 * np.log10((harmonic_energy + 1e-9) / (residual_energy + 1e-9))

            if harmonic_snr < float(harmonic_snr_stop_db):
                break

            layers.append((f0, conf))

            # Apply mask
            strength = np.clip(conf, 0.0, 1.0).reshape(1, -1) * eff_strength_scale

            # Soft subtraction with decay (Step 4)
            # We assume eff_strength_scale handles the soft nature.
            soft_mask = 1.0 - (1.0 - mask) * strength
            Z2 = Z * soft_mask

            _, residual2 = scipy.signal.istft(
                Z2,
                fs=sr,
                nperseg=n_fft_eff,
                noverlap=max(0, n_fft_eff - hop_eff),
                input_onesided=True,
                boundary="zeros",
            )
            residual = np.asarray(residual2, dtype=np.float32).reshape(-1)
            if residual.size < y.size:
                residual = np.pad(residual, (0, y.size - residual.size))
            residual = residual[: y.size]

            residual_rms = float(np.sqrt(np.mean(residual**2)) + 1e-9)
            if residual_rms / base_rms < float(residual_rms_stop_ratio):
                break

            if flatness > float(residual_flatness_stop):
                break

            # Adapt base mask width for next iter if standard schedule used (fallback/concurrent)
            if not iss_adaptive:
                if harmonic_snr < float(harmonic_snr_stop_db) + 3.0:
                    current_mask_width = min(float(max_mask_width), float(current_mask_width) * float(mask_growth))
                else:
                    current_mask_width = max(float(min_mask_width), float(current_mask_width) * float(mask_shrink))
        except Exception:
            break

    return layers


# --------------------------------------------------------------------------------------
# Detector base + implementations
# --------------------------------------------------------------------------------------
@dataclass
class DetectorOutput:
    f0_hz: np.ndarray
    confidence: np.ndarray


class BasePitchDetector:
    """
    Base class used by Stage B.
    Must implement: predict(audio, audio_path=None) -> (f0, conf)
    """

    def __init__(
        self,
        sr: int,
        hop_length: int,
        n_fft: int = 2048,
        fmin: float = 50.0,
        fmax: float = 1200.0,
        threshold: float = 0.10,
        **kwargs: Any,  # absorb unknown config keys safely
    ):
        self.sr = int(sr)
        self.hop_length = int(hop_length)

        # PB1: Map frame_length -> n_fft if provided
        if "frame_length" in kwargs and n_fft == 2048:
             # Only override default n_fft if frame_length is explicit
             # Note: caller usually passes n_fft=default.
             # If we see frame_length, use it.
             n_fft = int(kwargs["frame_length"])
             # Clean up to avoid confusion? Or keep it?
             # kwargs.pop("frame_length")
             # Leaving it in kwargs is safer for now.

        self.n_fft = int(n_fft)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.threshold = float(threshold)
        self._warned: Dict[str, bool] = {}
        self.kwargs = kwargs # Store extra config

    def _warn_once(self, key: str, msg: str) -> None:
        if not self._warned.get(key, False):
            warnings.warn(msg)
            self._warned[key] = True

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class YinDetector(BasePitchDetector):
    """
    Prefers librosa.pyin when available; otherwise falls back to lightweight ACF tracker.
    Supports Multi-Resolution F0 and Octave Error Correction (Feature C).
    """

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Config options (Feature C)
        enable_multires = self.kwargs.get("enable_multires_f0", False)
        enable_octave_corr = self.kwargs.get("enable_octave_correction", False)
        octave_penalty = float(self.kwargs.get("octave_jump_penalty", 0.35))

        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        # Helper to run one pass
        def _run_pass(frame_len: int) -> Tuple[np.ndarray, np.ndarray]:
            if librosa is not None:
                try:
                    # PB2: Support trough_threshold (Patch E1)
                    # Pass trough_threshold only if explicitly requested (or configured in profile),
                    # to be safe across librosa versions that might not support it (though 0.8+ does).
                    extra_kwargs = {}

                    # Also check flat key in kwargs just in case it wasn't nested
                    trough = self.kwargs.get("trough_threshold") or self.kwargs.get("yin_trough_threshold")
                    if trough is not None:
                         extra_kwargs["trough_threshold"] = float(trough)

                    try:
                        f0, _, voiced_prob = librosa.pyin(
                            y=y,
                            fmin=float(self.fmin),
                            fmax=float(self.fmax),
                            sr=int(self.sr),
                            frame_length=int(frame_len),
                            hop_length=int(self.hop_length),
                            fill_na=0.0,
                            **extra_kwargs
                        )
                    except TypeError:
                        # Fallback if argument not supported
                        f0, _, voiced_prob = librosa.pyin(
                            y=y,
                            fmin=float(self.fmin),
                            fmax=float(self.fmax),
                            sr=int(self.sr),
                            frame_length=int(frame_len),
                            hop_length=int(self.hop_length),
                            fill_na=0.0,
                        )
                    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
                    conf = np.asarray(voiced_prob, dtype=np.float32).reshape(-1)
                    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
                    conf = np.where(f0 > 0.0, conf, 0.0).astype(np.float32)
                    return f0, conf
                except Exception:
                    pass
            # Fallback
            frames = _frame_audio(y, frame_length=frame_len, hop_length=self.hop_length)
            # Use safe iterative fallback
            f0, conf = _autocorr_pitch_per_frame_safe(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax, threshold=self.threshold)
            return f0, conf

        if enable_multires:
            # Short + Long windows
            win_short = int(self.n_fft)
            win_long = int(self.n_fft * 2)

            f0_s, conf_s = _run_pass(win_short)
            f0_l, conf_l = _run_pass(win_long)

            # Fuse per frame
            n = min(len(f0_s), len(f0_l))
            f0_s, conf_s = f0_s[:n], conf_s[:n]
            f0_l, conf_l = f0_l[:n], conf_l[:n]

            final_f0 = np.zeros(n, dtype=np.float32)
            final_conf = np.zeros(n, dtype=np.float32)

            for i in range(n):
                # Prefer candidate with higher confidence
                # Tie-break: short window usually better for timing, long for bass pitch
                if conf_s[i] >= conf_l[i]:
                    final_f0[i] = f0_s[i]
                    final_conf[i] = conf_s[i]
                else:
                    final_f0[i] = f0_l[i]
                    final_conf[i] = conf_l[i]

            f0, conf = final_f0, final_conf
        else:
            f0, conf = _run_pass(self.n_fft)

        # Octave Error Correction
        if enable_octave_corr:
            f0 = self._apply_octave_correction(f0, conf, octave_penalty)

        # Final thresholding
        conf = np.clip(conf, 0.0, 1.0)
        conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
        f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
        return f0, conf

    def _apply_octave_correction(self, f0: np.ndarray, conf: np.ndarray, penalty: float) -> np.ndarray:
        # Simple heuristic: for each frame, consider 0.5*f, 1.0*f, 2.0*f.
        # Choose the one that minimizes jump from previous frame.
        # This acts like a greedy Viterbi with a lookback of 1.
        out_f0 = f0.copy()
        prev = 0.0

        for i in range(len(f0)):
            curr = f0[i]
            if curr <= 0:
                prev = 0.0
                continue

            if prev <= 0:
                prev = curr
                continue

            candidates = [curr, curr * 0.5, curr * 2.0]
            best_c = curr
            min_cost = float("inf")

            for cand in candidates:
                if cand < self.fmin or cand > self.fmax:
                    continue
                # Continuity cost: cents diff
                diff_cents = abs(1200.0 * np.log2(cand / prev))
                # Bias towards original detection (0 penalty for 1.0*f)
                cand_penalty = 0.0
                if cand != curr:
                    cand_penalty = penalty * 1200.0 # moderate penalty for switching octave

                cost = diff_cents + cand_penalty
                if cost < min_cost:
                    min_cost = cost
                    best_c = cand

            out_f0[i] = best_c
            prev = best_c

        return out_f0


class SACFDetector(BasePitchDetector):
    """
    Summary autocorrelation style: currently returns a dominant f0 + confidence using ACF.
    (Designed to be stable without extra dependencies.)
    """

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)

        # SACF tends to be noisier; apply threshold a bit more strictly
        thr = max(self.threshold, 0.12)
        conf = np.where(conf >= thr, conf, 0.0).astype(np.float32)
        f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
        return f0, conf

    def validate_curve(self, f0_curve: np.ndarray, audio: np.ndarray) -> float:
        """Compare a proposed f0 curve against SACF's internal estimate."""
        f0_curve = np.asarray(f0_curve, dtype=np.float32)
        pred_f0, pred_conf = self.predict(audio)
        n = min(len(f0_curve), len(pred_f0))
        if n == 0:
            return 0.0

        agreement = np.abs(f0_curve[:n] - pred_f0[:n])
        weights = np.where(pred_conf[:n] > 0.0, pred_conf[:n], 0.1)
        matches = (agreement < 5.0).astype(np.float32)
        weighted = float(np.average(matches, weights=weights)) if np.sum(weights) > 0 else float(np.mean(matches))
        return weighted


class CQTDetector(BasePitchDetector):
    """
    CQT-based dominant pitch (requires librosa). If librosa missing, returns zeros with warning.
    Supports Morphological Filtering (Feature D).
    """

    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048, **kwargs: Any):
        super().__init__(sr=sr, hop_length=hop_length, n_fft=n_fft, **kwargs)
        self.bins_per_octave = _safe_int(kwargs.get("bins_per_octave", 36), 36)
        self.n_bins = _safe_int(kwargs.get("n_bins", 7 * self.bins_per_octave), 7 * self.bins_per_octave)
        self.enable_morphology = kwargs.get("enable_salience_morphology", False)
        self.morph_kernel = int(kwargs.get("morph_kernel", 3))

    def predict(
        self,
        audio: np.ndarray,
        audio_path: Optional[str] = None,
        polyphony: bool = False,
        max_peaks: int = 5,
    ) -> Tuple[Any, Any]:
        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)

        def _fft_peaks(frame_mag: np.ndarray) -> Tuple[List[float], List[float]]:
            if frame_mag.size == 0:
                return [], []
            mean_val = float(np.mean(frame_mag)) + 1e-9
            ordered_bins = np.argsort(frame_mag)[::-1]
            freqs = np.fft.rfftfreq(frame_mag.size * 2 - 2, 1.0 / float(self.sr))
            found_pitch: List[float] = []
            found_conf: List[float] = []
            for b in ordered_bins:
                freq_val = float(freqs[b])
                if any(abs(freq_val - existing) < 5.0 for existing in found_pitch):
                    continue
                found_pitch.append(freq_val)
                found_conf.append(float(np.clip((frame_mag[b] / mean_val - 1.0) / 4.0, 0.0, 1.0)))
                if len(found_pitch) >= max_peaks:
                    break
            return found_pitch, found_conf

        if librosa is None:
            self._warn_once("no_librosa", "CQTDetector disabled: librosa not available.")
            if polyphony:
                pitches_list: List[List[float]] = []
                confs_list: List[List[float]] = []
                for frame in frames:
                    spectrum = np.abs(np.fft.rfft(frame))
                    p, c = _fft_peaks(spectrum)
                    pitches_list.append(p)
                    confs_list.append(c)
                return pitches_list, confs_list

            n = frames.shape[0]
            return np.zeros((n,), dtype=np.float32), np.zeros((n,), dtype=np.float32)

        try:
            C = librosa.cqt(
                y=y,
                sr=int(self.sr),
                hop_length=int(self.hop_length),
                fmin=float(self.fmin),
                n_bins=int(self.n_bins),
                bins_per_octave=int(self.bins_per_octave),
            )
            M = np.abs(C).astype(np.float32)  # (bins, frames)
            if M.size == 0:
                return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

            # Feature D: Morphological Filtering
            if self.enable_morphology:
                try:
                    import scipy.ndimage as ndimage
                except Exception:
                    ndimage = None

                if ndimage is not None:
                    # Apply grey_closing (dilation then erosion) along time axis primarily, or 2D?
                    # Usually we want to connect ridges in time.
                    # Kernel shape: (freq_bins, time_frames) -> (1, k)
                    k = max(1, self.morph_kernel)
                    M = ndimage.grey_closing(M, size=(1, k))

            freqs = librosa.cqt_frequencies(
                n_bins=int(self.n_bins), fmin=float(self.fmin), bins_per_octave=int(self.bins_per_octave)
            )

            if polyphony:
                # Return top-N peaks per frame for integration tests that expect poly lists
                pitches_list: List[List[float]] = []
                confs_list: List[List[float]] = []
                for frame_idx in range(M.shape[1]):
                    ordered_bins = np.argsort(M[:, frame_idx])[::-1]
                    frame_conf = []
                    frame_pitch = []
                    frame_mean = np.mean(M[:, frame_idx]) + 1e-9
                    for b in ordered_bins:
                        freq_val = float(freqs[b])
                        if any(abs(freq_val - existing) < 5.0 for existing in frame_pitch):
                            continue
                        frame_pitch.append(freq_val)
                        frame_conf.append(float(np.clip((M[b, frame_idx] / frame_mean - 1.0) / 4.0, 0.0, 1.0)))
                        if len(frame_pitch) >= max_peaks:
                            break
                    pitches_list.append(frame_pitch)
                    confs_list.append(frame_conf)
                return pitches_list, confs_list

            # dominant bin per frame
            idx = np.argmax(M, axis=0)
            f0 = freqs[idx].astype(np.float32)

            # confidence from peak-to-mean ratio
            peak = M[idx, np.arange(M.shape[1])]
            mean = np.mean(M, axis=0) + 1e-9
            conf = (peak / mean).astype(np.float32)
            conf = np.clip((conf - 1.0) / 4.0, 0.0, 1.0)  # squash
            conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
            f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
            return f0, conf
        except Exception:
            # fallback to FFT-based peak picking or ACF
            if polyphony:
                pitches_list = []
                confs_list = []
                for frame in frames:
                    spectrum = np.abs(np.fft.rfft(frame))
                    p, c = _fft_peaks(spectrum)
                    pitches_list.append(p)
                    confs_list.append(c)
                return pitches_list, confs_list

            f0, conf = _autocorr_pitch_per_frame_safe(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax, threshold=self.threshold)
            return f0, conf


class SwiftF0Detector(BasePitchDetector):
    """
    Placeholder wrapper: requires torch model in your project.
    If torch not available, returns zeros (and Stage B will warn).
    """

    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048, **kwargs: Any):
        super().__init__(sr=sr, hop_length=hop_length, n_fft=n_fft, **kwargs)
        # Check enabled lazily in predict to avoid import here?
        # But we need to know if enabled to register it.
        # Original code used 'import torch' top level.
        # We can check availability via a util or just try import in predict.
        self.enabled = True # Assume enabled config-wise, check dep later.

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import torch
        except Exception:
            torch = None

        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        n = frames.shape[0]

        if torch is None:
            self._warn_once("no_torch", "SwiftF0 disabled: torch not available. Falling back to ACF.")
            # Fall through to ACF fallback

        # If you later add real SwiftF0 inference, replace this block.
        # For now: stable fallback to ACF so pipeline still works deterministically.
        # Use fast ACF here as it's not strictly a "fallback" but the placeholder impl?
        # Requirement says "Only patch the naïve / fallback loops (YIN fallback / placeholder brute ACF)".
        # SwiftF0 here IS the placeholder. So safe is better?
        # But SwiftF0 is primary. If we make it slow, defaults become slow.
        # Let's keep fast for SwiftF0 placeholder, only change YIN/SACF/CREPE-fallback.
        f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)
        conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)
        f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
        return f0, conf


class RMVPEDetector(BasePitchDetector):
    """
    Placeholder wrapper for RMVPE. If torch not available, returns zeros.
    """

    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048, **kwargs: Any):
        super().__init__(sr=sr, hop_length=hop_length, n_fft=n_fft, **kwargs)
        self.enabled = True

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import torch
        except Exception:
            torch = None

        y = np.asarray(audio, dtype=np.float32).reshape(-1)
        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        n = frames.shape[0]

        if torch is None:
            self._warn_once("no_torch", "RMVPE disabled: torch not available.")
            return np.zeros((n,), dtype=np.float32), np.zeros((n,), dtype=np.float32)

        # Replace with actual RMVPE inference later.
        # Keep fast ACF for placeholder
        f0, conf = _autocorr_pitch_per_frame(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax)
        conf = np.where(conf >= self.threshold, conf, 0.0).astype(np.float32)

        try:
            # placeholder post-filter: keep only frames with confidence
            f0 = np.where(conf > 0.0, f0, 0.0).astype(np.float32)
            conf = conf.astype(np.float32)
            return f0, conf
        except Exception as e:
            # Never crash: return zeros
            n = int(len(getattr(conf, "__len__", lambda: [])()) or 0)
            if n == 0:
                # best-effort length inference
                n = int(len(f0)) if "f0" in locals() and hasattr(f0, "__len__") else 0
            if hasattr(self, "warn_once"):
                self.warn_once("rmvpe_error", f"RMVPE placeholder failed: {e}")
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)


class CREPEDetector(BasePitchDetector):
    """
    CREPE wrapper. (Feature A)
    """

    def __init__(self, sr: int, hop_length: int, n_fft: int = 2048, **kwargs: Any):
        super().__init__(sr=sr, hop_length=hop_length, n_fft=n_fft, **kwargs)
        self.model_capacity = str(kwargs.get("model_capacity", "small"))
        self.step_ms = int(kwargs.get("step_ms", 10))
        # Support both keys (Patch 1A)
        self.conf_threshold = float(kwargs.get("confidence_threshold", kwargs.get("conf_threshold", 0.5)))

    def predict(self, audio: np.ndarray, audio_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import crepe
        except Exception:
            crepe = None

        y = np.asarray(audio, dtype=np.float32).reshape(-1)

        # Determine number of frames expected by pipeline
        frames = _frame_audio(y, frame_length=self.n_fft, hop_length=self.hop_length)
        n_frames = frames.shape[0]

        if crepe is None:
            self._warn_once("no_crepe", "CREPE disabled: crepe not available.")
            return np.zeros((n_frames,), dtype=np.float32), np.zeros((n_frames,), dtype=np.float32)

        try:
            # CREPE runs at 16k usually, handle internally?
            # crepe.predict automatically resamples if needed.
            # step_size in ms.

            # Note: crepe.predict outputs (time, frequency, confidence, activation)
            # We need to suppress print output from crepe if possible, but it's verbose.
            # verbose=0 is default in newer versions but let's check.

            time, frequency, confidence, _ = crepe.predict(
                y,
                self.sr,
                viterbi=False, # We do our own smoothing or allow pipeline to handle it
                step_size=self.step_ms,
                model_capacity=self.model_capacity,
                verbose=0
            )

            # Resample/Interp to match our pipeline grid (hop_length)
            # Pipeline grid: t = i * hop_length / sr
            pipeline_times = np.arange(n_frames) * self.hop_length / self.sr

            # CREPE output might be shorter/longer. Interpolate.
            f0_interp = np.interp(pipeline_times, time, frequency, left=0.0, right=0.0)
            conf_interp = np.interp(pipeline_times, time, confidence, left=0.0, right=0.0)

            # Apply threshold
            conf_out = np.where(conf_interp >= self.conf_threshold, conf_interp, 0.0).astype(np.float32)
            f0_out = np.where(conf_out > 0.0, f0_interp, 0.0).astype(np.float32)

            # Filter range
            f0_out = np.where((f0_out >= self.fmin) & (f0_out <= self.fmax), f0_out, 0.0)
            conf_out = np.where(f0_out > 0.0, conf_out, 0.0)

            return f0_out, conf_out

        except Exception as e:
            self._warn_once("crepe_error", f"CREPE inference failed: {e}")
            # Fallback to ACF if CREPE fails (e.g. missing tensorflow or shape error)
            # This ensures we always return *some* pitch data rather than silence.
            # Use SAFE fallback here
            f0, conf = _autocorr_pitch_per_frame_safe(frames, sr=self.sr, fmin=self.fmin, fmax=self.fmax, threshold=self.threshold)
            return f0, conf


# Alias for backwards compatibility
CQTPeaksDetector = CQTDetector  # type: ignore
