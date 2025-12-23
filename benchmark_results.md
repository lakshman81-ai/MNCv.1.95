# Benchmark Results

Here are the detailed accuracy metrics for the requested benchmarks.

## Summary Table

| Level | Name | Note F1 | Precision | Recall | Onset MAE (ms) | Notes (Pred/GT) |
|---|---|---|---|---|---|---|
| **L4** | happy_birthday | 0.909 | N/A* | N/A* | 0.0 | 5 / 6 |
| **L4** | old_macdonald | 0.727 | N/A* | N/A* | 0.0 | 4 / 7 |
| **L5.1** | kal_ho_na_ho | 0.370 | 0.263 | 0.625 | 418.0 | 152 / 64 |
| **L5.2** | tumhare_hi_rahenge | 0.161 | 0.141 | 0.187 | 170.5 | 99 / 75 |

*\*Precision and Recall for L4 were not explicitly captured in the artifact snapshot, but F1 and counts are provided.*

## Details

### L5.1 (Kal Ho Na Ho)
* **Performance:** F1 0.37 (Baseline ~0.56)
* **Issues:** High fragmentation (152 predicted notes vs 64 ground truth) and large onset error (418ms), likely due to `synthetic_model=False` being used on synthesized audio in this environment, or CPU-based separation artifacts.

### L5.2 (Tumhare Hi Rahenge)
* **Performance:** F1 0.16 (Baseline ~0.43)
* **Issues:** Low recall (0.19) and low precision (0.14), indicating difficulty in isolating the melody from the dense polyphonic mix using the default CPU configuration.

### L4 (Real Songs)
* **Performance:** High accuracy on simple real songs (F1 0.91 / 0.73).
* **Notes:** "Old MacDonald" missed 3 notes (4 vs 7), while "Happy Birthday" was nearly perfect.
