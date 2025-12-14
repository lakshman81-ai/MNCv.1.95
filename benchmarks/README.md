# Benchmark Plan

This directory holds reusable fixtures and notes for evaluating the transcription pipeline. Use the structure and checklists below to keep inputs, ground truths, and timing results consistent across scenarios.

The actual benchmark runners are now located in `backend/benchmarks/`.

## Directory layout

```
benchmarks/
  01_scales/
    README.md          # scenario-specific notes and fixture checklist
    results.md         # recorded runs using the template below
    audio/             # raw audio fixtures (wav, mp3, etc.)
    references/        # MIDI or MusicXML ground truth when available
  02_simple_melodies/
  03_melody_plus_chords/
  04_pop_loops/
```
Alternatively use MIDIs from https://github.com/bytedance/GiantMIDI-Piano/tree/master/midis_for_evaluation
Add new scenarios by creating additional numbered folders that follow the same pattern (README + results.md + fixture subfolders).

## Naming conventions
- Use short, descriptive filenames such as `c_major_scale_100bpm.wav` or `folk_tune_in_g.mid`.
- Keep corresponding reference files aligned by basename (e.g., `folk_tune_in_g.mid` and `folk_tune_in_g_reference.musicxml`).
- Store mock-friendly XML/MIDI fixtures under `references/` so the mock pipeline can be exercised without audio dependencies.

## Running benchmarks
1. Use the unified benchmark runner:
   ```bash
   python -m backend.benchmarks.benchmark_runner
   ```
2. Or run specific real song benchmarks:
   ```bash
   python -m backend.benchmarks.run_real_songs --song happy_birthday
   ```

## Results template
Copy this template into each scenario's `results.md` (already seeded in the starter files):

```
# Results

| Date       | Environment (commit, OS, deps) | Fixture                          | Mode  | Iterations | Min (s) | Avg (s) | Max (s) | Notes |
|------------|--------------------------------|----------------------------------|-------|------------|---------|---------|---------|-------|
| 2025-02-15 | abc1234, Ubuntu, mock pipeline | c_major_scale_100bpm.wav         | mock  | 5          | 0.012   | 0.015   | 0.021   | Baseline sanity check |
| 2025-02-15 | abc1234, Ubuntu, full pipeline | c_major_scale_100bpm.wav         | full  | 5          | 0.480   | 0.525   | 0.610   | Initial full run |
```

Include extra context below the table when comparing different model versions, noise levels, or quantization settings.
