"""H3: kernel width per class, proportional to that class's observation
spread per axis. One component per observation, shaped well -- the rule the
multi-scale failure taught us.

Rebuilds the trace per variant per tuple: class_fields(..., scales=...,
seeds=SEEDS[ti]).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SEEDS, class_fields, make_blob_data, predict, run_screen, score)

PRED = ("Pre-registered: mAcc up on scenes with mixed object sizes; risk is "
        "per-scene overfit, so k is a single global multiplier, never tuned "
        "per scene. Derived from observation spread, no per-class hand "
        "tuning.")

PREDICTIONS = [
    {"id": "h3-acc",
     "text": "some global k gives mAcc up at least 0.005",
     "metric": "macc", "scope": "best", "test": "ge", "value": 0.005},
]

CLIP_LO, CLIP_HI = 0.10, 0.90


def _scales(data, k):
    """{cls: (clip(k*rms_x, 0.10, 0.90), clip(k*rms_y, 0.10, 0.90))} from
    the per-axis RMS spread about the class's median centre."""
    out = {}
    by = {}
    for o in data["obs"]:
        by.setdefault(o["cls"], []).append((o["x"], o["y"]))
    for c, pts in by.items():
        P = np.array(pts, float)
        ctr = np.median(P, 0)
        rx = float(np.sqrt(((P[:, 0] - ctr[0]) ** 2).mean()))
        ry = float(np.sqrt(((P[:, 1] - ctr[1]) ** 2).mean()))
        out[c] = (float(np.clip(k * rx, CLIP_LO, CLIP_HI)),
                  float(np.clip(k * ry, CLIP_LO, CLIP_HI)))
    return out


def variant(k):
    def fn(data, ti):
        F, names, cell = class_fields(data, scales=_scales(data, k),
                                      seeds=SEEDS[ti])
        return predict(F, names, cell)
    return fn


VARIANTS = {f"k{k}": variant(k) for k in (0.5, 1.0, 1.5)}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["k1.0"](blob, 0))
    assert m["macc"] > 0.9, m
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h3_perclass_lambda", PRED, VARIANTS,
                   predictions=PREDICTIONS)
