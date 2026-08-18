"""H7: scale-space bundling -- S traces at S length scales, pooled at decode.

WHY THIS IS DIFFERENT IN KIND FROM THE SEVEN THAT FAILED. Every screened
mechanism operates on ONE field: h1/h2 threshold or rescale it, h3/h3b change
its width, h4/h5/h6 change what enters it, and the codebook family changes the
symbols. This changes what the memory STORES: S traces at S scales.

The measured motivation (proximity_ceiling.py, proximity_decomposition.json):
at equal quantisation a PERFECT proximity decoder over our own stream buys
only +0.0053 -- so "our kernel discards proximity" is NOT the gap, and the
field-native story that said so is retracted. But distance-weighted kNN-5,
which differs from our field by having an ADAPTIVE bandwidth, reaches 0.3844
against our 0.3235. A single fixed-bandwidth linear field cannot adapt; a
stack of scales, made commensurable and summed, approximates one that can.

    F_c^s(x) = Re<phi_s(x), T_s (/) sem_c>              one trace per scale s
    Z_c^s(x) = (F_c^s - mean_c F^s) / std_c F^s         z ACROSS CLASSES, per cell
    label(x) = argmax_c sum_s Z_c^s(x)

The z is across CLASSES within a cell (axis 0). h2 z-scored across CELLS per
class (axis 1) -- different axis, different object. For a SINGLE scale this
transform is strictly monotone within a cell and therefore leaves argmax
exactly unchanged, so the mechanism is inert unless S > 1 by construction.
Binding and unbinding are untouched; only the decode pools.

Controls that identify the mechanism (both required):
  sumF6  -- same 6 scales, summed WITHOUT the z. If this matches sumz6, the
            gain is just "more scales" and the commensurability story is dead.
  maxz6  -- same 6 z-scored scales, MAX instead of SUM. If this matches, the
            gain is "pick the best scale", not evidence integration.

    python collab_tasks/batch1/h7_scalespace.py --self-test
    python collab_tasks/batch1/h7_scalespace.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    LX, LY, SEEDS, class_fields, make_blob_data, predict, run_screen, score)

MULT6 = (0.30, 0.45, 0.70, 1.00, 1.60, 2.50)
MULT3 = (0.45, 1.00, 2.50)

PRED = ("Pre-registered 2026-08-18 before any guarded run. Scale-space "
        "bundling: S traces at S length scales, z-scored ACROSS CLASSES per "
        "cell, summed, argmax. Motivated by a measurement, not a story: "
        "proximity alone is worth +0.0053 (below noise) but kNN-5, whose only "
        "structural difference from our field is an ADAPTIVE bandwidth, "
        "reaches 0.3844 vs our 0.3235. Predicts a real, broad gain that "
        "REQUIRES both the z (vs sumF6) and the sum (vs maxz6). Sized against "
        "the measured stream ceiling of 0.3844, not against their 0.402.")

PREDICTIONS = [
    {"id": "h7-acc",
     "text": "sumz6 gains >= +0.017 mAcc, i.e. clears the measured seed-noise "
             "floor rather than merely the 0.005 adoption bar",
     "metric": "macc", "scope": "sumz6", "test": "ge", "value": 0.017},
    {"id": "h7-breadth",
     "text": "the gain is broad, not one scene: positive on >= 7 of 8",
     "metric": "macc", "scope": "sumz6", "test": "scenes_ge", "value": 7},
    {"id": "h7-needs-z",
     "text": "the z-score is load-bearing: sumz6 beats the no-z control "
             "sumF6 by >= 0.010, gap resolved against tuple noise",
     "metric": "macc", "scope": ["sumz6", "sumF6"], "test": "pair_ge",
     "value": 0.010},
    {"id": "h7-needs-sum",
     "text": "integration not selection: sumz6 beats the max-pool control "
             "maxz6 by >= 0.010, gap resolved",
     "metric": "macc", "scope": ["sumz6", "maxz6"], "test": "pair_ge",
     "value": 0.010},
    {"id": "h7-fmiou",
     "text": "F-mIoU does not regress -- a decode that only trades recall for "
             "precision is not what we want here",
     "metric": "fmiou", "scope": "sumz6", "test": "ge", "value": 0.0},
    {"id": "h7-scales",
     "text": "more scales help: sumz6 beats sumz3 by >= 0.005, gap resolved",
     "metric": "macc", "scope": ["sumz6", "sumz3"], "test": "pair_ge",
     "value": 0.005},
]


def _z_across_classes(F):
    F = np.asarray(F, float)
    return (F - F.mean(0)) / (F.std(0) + 1e-12)


def stack(mults, pool):
    def fn(data, ti):
        acc, names, cell = None, None, None
        for m in mults:
            F, names, cell = class_fields(data, lx=LX * m, ly=LY * m,
                                          seeds=SEEDS[ti])
            G = _z_across_classes(F) if pool in ("sumz", "maxz") \
                else np.asarray(F, float)
            if acc is None:
                acc = G.copy()
            elif pool == "maxz":
                np.maximum(acc, G, out=acc)
            else:
                acc += G
        return predict(acc, names, cell)
    return fn


VARIANTS = {
    "sumz6": stack(MULT6, "sumz"),
    "sumz3": stack(MULT3, "sumz"),
    "sumF6": stack(MULT6, "sumF"),      # control: no z
    "maxz6": stack(MULT6, "maxz"),      # control: select, not integrate
}


def _self_test():
    blob = make_blob_data()
    # single scale must be EXACTLY the baseline -- the z is monotone per cell
    one = stack((1.0,), "sumz")(blob, 0)
    base = predict(*class_fields(blob, lx=LX, ly=LY, seeds=SEEDS[0]))
    assert (one == base).all(), "single-scale z changed the argmax"
    # The blob is 3 well-separated gaussian clusters where ONE scale is
    # already near-optimal, so extra scales only add blur: measured baseline
    # 0.9708, sumz6 0.8717, sumF6 0.9625, maxz6 0.8658. Recorded rather than
    # asserted away -- it says the mechanism is data-dependent, which is what
    # the guard's breadth clause exists to adjudicate. Do NOT tune the
    # mechanism to make this number go up; the blob is not the target.
    m = score(blob["gt"], VARIANTS["sumz6"](blob, 0))
    assert m["macc"] > 0.5, m          # smoke only: it still finds the blobs
    print(f"SELF-TEST PASS (single scale inert; blob sumz6 {m['macc']:.4f} "
          f"vs baseline 0.9708 -- expected, see comment)")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h7_scalespace", PRED, VARIANTS, predictions=PREDICTIONS)
