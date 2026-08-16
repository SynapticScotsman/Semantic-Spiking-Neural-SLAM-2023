"""H2: normalise each class's field by its own spatial spread and mass,
so diffuse heavy classes stop out-shouting compact ones.

Pure F-transform over the cached baseline fields (Amendment A7: cheap), so
variants read default_fields(data, ti) and never rebuild the trace.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    default_fields, make_blob_data, predict, run_screen, score)

PRED = ("Pre-registered: recovers bleed cells; helps most on scenes the "
        "decomposition marks bleed-dominant. Global z-score (the crude "
        "version) was +0.010 mean / 3 of 8 scenes; spread-aware should "
        "match or beat it WITHOUT the office4 dependence.")

PREDICTIONS = [
    {"id": "h2-acc",
     "text": "best p/q variant mAcc up at least 0.005",
     "metric": "macc", "scope": "best", "test": "ge", "value": 0.005},
    {"id": "h2-breadth",
     "text": ("best variant improves mAcc on at least 4 of 8 scenes (unlike "
              "zscore's office4 dependence)"),
     "metric": "macc", "scope": "best", "test": "scenes_ge", "value": 4},
]


def _spread_mass(data, names):
    """Per class: RMS distance from the median centre over data["obs"]
    (floored at 0.05 m), and the observation count."""
    sp, ms = [], []
    for c in names:
        P = np.array([[o["x"], o["y"]] for o in data["obs"]
                      if o["cls"] == c], float)
        if len(P) == 0:
            sp.append(1.0)
            ms.append(1.0)
            continue
        ctr = np.median(P, 0)
        rms = float(np.sqrt(((P - ctr) ** 2).sum(1).mean()))
        sp.append(max(rms, 0.05))
        ms.append(float(len(P)))
    return np.array(sp), np.array(ms)


def variant(p, q):
    def fn(data, ti):
        F, names, cell = default_fields(data, ti)
        sp, ms = _spread_mass(data, names)
        Fp = np.asarray(F, float) / (sp[:, None] ** p * ms[:, None] ** q)
        return predict(Fp, names, cell)
    return fn


def zscore(data, ti):
    F, names, cell = default_fields(data, ti)
    F = np.asarray(F, float)
    Z = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)
    return predict(Z, names, cell)


VARIANTS = {f"p{p}_q{q}": variant(p, q)
            for p in (0.0, 0.5, 1.0) for q in (0.0, 0.5, 1.0)
            if not (p == 0.0 and q == 0.0)}
VARIANTS["zscore_control"] = zscore


def _self_test():
    blob = make_blob_data()
    for lab in ("p0.5_q0.5", "zscore_control"):
        m = score(blob["gt"], VARIANTS[lab](blob, 0))
        assert m["macc"] > 0.9, (lab, m)
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h2_spread_norm", PRED, VARIANTS, predictions=PREDICTIONS)
