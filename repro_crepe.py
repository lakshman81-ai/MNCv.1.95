
import numpy as np
import crepe
import warnings
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_crepe():
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 440.0
    # Sine wave
    audio = 0.5 * np.sin(2 * np.pi * f0 * t)

    print(f"Running Crepe (Full, Viterbi) on {duration}s sine wave at {f0}Hz, SR={sr}...")

    try:
        # Match L2 config: model_capacity='full', viterbi=True
        time, frequency, confidence, activation = crepe.predict(
            audio,
            sr,
            viterbi=True,
            step_size=10,
            model_capacity='full',
            verbose=1
        )

        avg_conf = np.mean(confidence)
        # Filter low confidence for freq calc
        valid_indices = confidence > 0.5
        avg_freq = np.mean(frequency[valid_indices]) if np.any(valid_indices) else 0.0

        print(f"Mean Confidence: {avg_conf:.4f}")
        print(f"Mean Frequency (where conf>0.5): {avg_freq:.2f} Hz")

        if avg_conf < 0.1:
            print("FAILURE: Confidence too low.")
        elif abs(avg_freq - f0) > 10.0:
            print(f"FAILURE: Frequency mismatch. Expected {f0}, got {avg_freq}")
        else:
            print("SUCCESS: Detection looks correct.")

    except Exception as e:
        print(f"CRITICAL EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crepe()
