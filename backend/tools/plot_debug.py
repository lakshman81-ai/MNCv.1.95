
import argparse
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os
import pandas as pd

def plot_stage_b_debug(wav_path, f0_path, output_path=None):
    """
    Plots Spectrogram with F0 overlay.
    """
    print(f"Loading audio: {wav_path}")
    y, sr = librosa.load(wav_path, sr=None)

    print(f"Loading F0: {f0_path}")
    # Load F0
    # Assuming CSV with header time,frequency
    df = pd.read_csv(f0_path)
    times = df['time'].values
    freqs = df['frequency'].values

    # Filter out zeros for plotting
    mask = freqs > 0
    times_voiced = times[mask]
    freqs_voiced = freqs[mask]

    plt.figure(figsize=(12, 8))

    # Plot Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')

    # Overlay F0
    plt.plot(times_voiced, freqs_voiced, label='Detected F0', color='cyan', linewidth=2, alpha=0.8)

    plt.title(f'Stage B Debug: {os.path.basename(wav_path)}')
    plt.legend(loc='upper right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Stage B Output")
    parser.add_argument("wav_path", help="Path to input WAV file")
    parser.add_argument("f0_path", help="Path to Stage B F0 CSV (time,frequency)")
    parser.add_argument("--output", "-o", help="Output image path", default="debug_plot.png")

    args = parser.parse_args()

    plot_stage_b_debug(args.wav_path, args.f0_path, args.output)
