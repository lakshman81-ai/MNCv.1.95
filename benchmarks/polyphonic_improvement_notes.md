# Polyphonic note creation improvement ideas

Context: Running the L2 "Poly Dominant" synthetic benchmark with HTDemucs separation enabled kept the F1 at ~0.67. The synthetic sine/saw mixtures differ sharply from the real-instrument audio Demucs was trained on, so the current separation pass does not help tease apart melody vs. bass. The suggestions below aim to improve polyphonic extraction before or without a perfect separator.

## Make synthetic-friendly stems and priors
- **Train or fine-tune a separator on synthetic waveforms** (sine, saw, square, simple FM) to match the benchmark distribution. A small Demucs/MDX variant trained on procedurally generated stems can better isolate melody vs. bass than a real-instrument model.
- **Augment the existing separator with harmonic masks**: the Stage B harmonic masking hook (`stage_b.separation.harmonic_masking`) now uses a fast SwiftF0 prior to emphasize harmonic bins and emit synthetic melody/residual stems before detector fusion.【F:backend/pipeline/config.py†L64-L132】【F:backend/pipeline/stage_b.py†L172-L332】

## Tighten Stage B for polyphony
- **Use polyphonic peeling (ISS) deliberately**: enable `polyphonic_peeling.max_layers` in `StageBConfig`—with `force_on_mix` keeping ISS on synthetic mixes—to route the residual through `iterative_spectral_subtraction` before ensemble merging.【F:backend/pipeline/config.py†L102-L133】【F:backend/pipeline/stage_b.py†L331-L381】 This helps when chords share similar timbres and separation is weak.
- **Lower disagreement tolerance for the ensemble**: `pitch_disagreement_cents` now defaults to 45 cents and `confidence_voicing_threshold` to 0.65 so the merged F0 timeline favors consistent cross-detector peaks instead of averaging melody+bass candidates.【F:backend/pipeline/config.py†L80-L108】 
- **Prioritize top voice during skyline filtering**: keep `polyphony_filter.mode="skyline_top_voice"` so Stage C/D bias toward the dominant melodic layer while ISS peels the rest; alternate skyline modes still select the highest-confidence pitch.【F:backend/pipeline/config.py†L185-L188】【F:backend/pipeline/stage_b.py†L421-L452】 If chord completeness is required, expose a mode that preserves multiple concurrent layers from the ISS output instead of flattening to the skyline.

## Improve segmentation for overlapped notes
- **Length-aware segmentation**: `stage_c.min_note_duration_ms_poly` raises the minimum duration whenever skyline frames report multiple active pitches, suppressing micro-notes driven by bass modulation.【F:backend/pipeline/config.py†L140-L188】【F:backend/pipeline/stage_c.py†L180-L227】 For fast figurations, pair a shorter min duration with stricter RMS gating.
- **Layer-specific confidence gating**: `stage_c.polyphonic_confidence` lets skyline frames with multiple active pitches use higher confidence gates while keeping the melodic layer permissive, reducing cross-talk when merging timelines.【F:backend/pipeline/config.py†L170-L188】【F:backend/pipeline/stage_c.py†L180-L227】

## Benchmarking workflow adjustments
- **Add synthetic polyphonic fixtures with known stems** to measure whether stem-specific ISS improves precision/recall independent of Demucs.
- **Report F1 by role (melody vs. bass)** so we can see whether errors come from missing bass suppression or melody dilution; this helps pick the right combination of peeling depth and skyline filtering.
