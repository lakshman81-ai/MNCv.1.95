
import csv
import math
import os
from typing import List, Dict, Iterable, Tuple, Any, Optional

A4_HZ = 440.0
A4_MIDI = 69

def hz_to_cents(hz: float) -> float:
    if hz <= 0:
        return float("nan")
    return 1200.0 * math.log2(hz / A4_HZ) + (A4_MIDI * 100.0)

def cents_to_midi(cents: float) -> float:
    return cents / 100.0

def write_frame_timeline_csv(path: str, frames: Iterable[Dict[str, Any]]) -> None:
    """
    Export frame-by-frame debug data.
    frames: iterable of dicts with keys:
      "t_sec", "f0_hz", "midi", "cents", "confidence", "detector_name",
      "voiced", "harmonic_rank", "fused_cents", "smoothed_cents"
    """
    cols = [
        "t_sec", "f0_hz", "midi", "cents", "confidence", "detector_name",
        "voiced", "harmonic_rank", "fused_cents", "smoothed_cents", "onset_strength"
    ]

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for fr in frames:
            w.writerow(fr)

def note_center(n: Dict[str, Any]) -> float:  # n has onset_sec, offset_sec
    return 0.5 * (float(n["onset_sec"]) + float(n["offset_sec"]))

def match_notes_nearest(
    gt_notes: List[Dict[str, Any]],
    pred_notes: List[Dict[str, Any]],
    max_center_dist_sec: float = 0.5
) -> Tuple[List[Tuple[Dict, Optional[Dict], bool]], List[Dict]]:
    """
    Simple bipartite-ish greedy match (good enough for debugging).
    Returns (pairs, extras).
    Pairs is list of (gt, pred, missed_bool).
    """
    used = set()
    pairs = []

    # Sort for slightly better greedy behavior (optional)
    gt_sorted = sorted(gt_notes, key=lambda x: x["onset_sec"])
    pred_sorted = sorted(pred_notes, key=lambda x: x["onset_sec"])

    for g in gt_sorted:
        gc = note_center(g)
        best = None
        best_d = 1e9
        best_j = None

        for j, p in enumerate(pred_sorted):
            if j in used:
                continue

            # Filter by coarse window first
            pc = note_center(p)
            if abs(pc - gc) > max_center_dist_sec:
                continue

            d = abs(pc - gc)
            if d < best_d:
                best_d = d
                best = p
                best_j = j

        if best is not None and best_d <= max_center_dist_sec:
            used.add(best_j)
            pairs.append((g, best, False))  # False = not missed
        else:
            pairs.append((g, None, True))   # missed

    extras = [p for j, p in enumerate(pred_sorted) if j not in used]
    return pairs, extras

def write_error_slices_jsonl(path: str, pairs: List[Tuple[Dict, Optional[Dict], bool]], extras: List[Dict]) -> None:
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for gt, pred, missed in pairs:
            record = {
                "type": "missed" if missed else "match",
                "gt": gt,
                "pred": pred,
            }
            if not missed and pred:
                record["err_onset_ms"] = (pred["onset_sec"] - gt["onset_sec"]) * 1000.0
                record["err_offset_ms"] = (pred["offset_sec"] - gt["offset_sec"]) * 1000.0
                record["err_pitch_cents"] = (pred["pitch_midi"] - gt["pitch_midi"]) * 100.0
            f.write(json.dumps(record) + "\n")

        for ex in extras:
            f.write(json.dumps({"type": "extra", "pred": ex}) + "\n")
