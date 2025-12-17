
import sys
import os
import time
import numpy as np

# Add root to sys.path so we can import backend
sys.path.append(os.getcwd())

from backend.pipeline.detectors import create_harmonic_mask

def benchmark():
    # Setup
    sr = 44100
    n_fft = 2048
    duration = 5.0 # seconds
    n_frames = int(duration * sr / 512)

    # Generate random f0 between 100 and 1000 Hz
    f0_hz = np.random.uniform(100.0, 1000.0, n_frames).astype(np.float32)

    # Mask parameters
    mask_width = 0.03
    n_harmonics = 16

    print(f"Benchmarking create_harmonic_mask with {n_frames} frames, {n_harmonics} harmonics...")

    # Warmup
    create_harmonic_mask(f0_hz, sr, n_fft, mask_width, n_harmonics)

    # Run
    start_time = time.perf_counter()
    n_runs = 100
    for _ in range(n_runs):
        create_harmonic_mask(f0_hz, sr, n_fft, mask_width, n_harmonics)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / n_runs
    print(f"Average time per run: {avg_time*1000:.3f} ms")

if __name__ == "__main__":
    benchmark()
