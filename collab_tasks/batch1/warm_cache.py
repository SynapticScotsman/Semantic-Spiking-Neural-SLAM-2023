"""Pre-warm the baseline field cache for all (scene, seed-tuple) pairs.

Run once before the Stage-1 screens so h1/h2 (pure F-transforms) cost seconds
and h3-h6 pay only for their variant builds. Also doubles as the full parity
gate: the reference tuple's labels are checked against the stored baseline for
every scene while warming.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    BASELINE_MACC, SCENES, SEEDS, default_fields, load_scene, predict, score)

t0 = time.time()
for ti in range(len(SEEDS)):
    for s in SCENES:
        data = load_scene(s)
        t1 = time.time()
        F, nm, cell = default_fields(data, ti)
        pred = predict(F, nm, cell)
        if ti == 0:
            stored = np.load(
                f"student_gpu_package/handoff/{s}_cgfront/vsa_labels.npz",
                allow_pickle=True)["pred_class"].astype(str)
            if not (pred == stored).all():
                raise SystemExit(f"HARD STOP: label parity fail on {s}")
            m = score(data["gt"], pred)
            if abs(m["macc"] - BASELINE_MACC[s]) > 0.005:
                raise SystemExit(f"HARD STOP: score parity fail on {s}")
        print(f"t{ti} {s}: cached in {time.time()-t1:.0f}s"
              + ("  [parity OK]" if ti == 0 else ""), flush=True)
print(f"cache warm in {(time.time()-t0)/60:.1f} min")
