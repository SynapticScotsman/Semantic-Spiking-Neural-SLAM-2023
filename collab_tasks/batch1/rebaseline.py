"""Re-point the guard's baselines after a deliberate change to the build path.

Run this ONLY after a change to the trace build (codebook, encoder, cap rule)
that is intended to stand. It does three things and refuses to do any of them
silently:

  1. asserts every handoff/<scene>_cgfront/vsa_labels.npz now reproduces
     class_fields() EXACTLY -- the two-implementation cross-check between
     04_vsa_labels.py and collab_tasks/batch1/common.py. If that fails, the
     baselines were not regenerated with the matching invocation and nothing
     is written.
  2. prints the new BASELINE_MACC dict for common.py, old vs new side by side.
  3. reports the headline shift against the 0.017 seed-noise floor, so a
     change that moves the published number cannot pass unnoticed.

It does NOT edit common.py -- paste the printed dict, so the change is a
reviewable diff rather than a silent rewrite.

PREREQUISITE: quarantine outputs/batch1/cache/ first. A cache holding
old-build fields next to new-build variants is worse than no cache: run_screen
reads baselines from cache and builds variants fresh, so a mixed cache
produces a large, broad, entirely fake effect.

    python collab_tasks/batch1/rebaseline.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    BASELINE_MACC, CACHE_DIR, SCENES, SEEDS, class_fields, load_scene,
    predict, score)

NOISE_2SD = 0.017


def main():
    n_cached = len([f for f in os.listdir(CACHE_DIR)
                    if f.endswith(".npz")]) if os.path.isdir(CACHE_DIR) else 0
    if n_cached:
        raise SystemExit(
            f"REFUSING: {n_cached} files still in {CACHE_DIR}. Quarantine "
            f"them first (mv to cache_stale_*/), or the screen will compare "
            f"old-build baselines against new-build variants.")

    new, old, bad = {}, {}, []
    for s in SCENES:
        d = load_scene(s)
        F, names, cell = class_fields(d, seeds=SEEDS[0])
        pred = predict(F, names, cell)
        stored = np.load(
            f"student_gpu_package/handoff/{s}_cgfront/vsa_labels.npz",
            allow_pickle=True)["pred_class"].astype(str)
        agree = float((pred == stored).mean())
        if agree != 1.0:
            bad.append((s, agree))
        new[s] = round(float(score(d["gt"], pred)["macc"]), 4)
        old[s] = BASELINE_MACC[s]
        print(f"  {s:<9} parity {agree:.6f}   mAcc {old[s]:.3f} -> "
              f"{new[s]:.4f}   {new[s]-old[s]:+.4f}", flush=True)

    if bad:
        print("\nHARD STOP: these scenes do not reproduce class_fields:")
        for s, a in bad:
            print(f"  {s}: agreement {a:.6f}")
        raise SystemExit(
            "Regenerate them with the CANONICAL command (note the "
            "non-default args):\n"
            "  python student_gpu_package/04_vsa_labels.py --scene "
            "<scene>_cgfront \\n"
            "      --labels-from-points --max-per-class 400 "
            "--length-scale 0.45,0.27 --grid 96")

    om, nm = np.mean(list(old.values())), np.mean(list(new.values()))
    print(f"\nheadline mean mAcc {om:.4f} -> {nm:.4f}  ({nm-om:+.4f})")
    print(f"  {'INSIDE' if abs(nm-om) < NOISE_2SD else '*** OUTSIDE'} the "
          f"{NOISE_2SD} seed-noise band"
          + ("" if abs(nm - om) < NOISE_2SD else
             " -- the published number MOVED; say so explicitly"))
    print(f"  largest per-scene shift: "
          f"{max(abs(new[s]-old[s]) for s in SCENES):.4f}")

    print("\nPaste into collab_tasks/batch1/common.py:\n")
    items = [f'"{s}": {new[s]:.4f}' for s in SCENES]
    print("BASELINE_MACC = {" + ",\n                 ".join(items) + "}")


if __name__ == "__main__":
    main()
