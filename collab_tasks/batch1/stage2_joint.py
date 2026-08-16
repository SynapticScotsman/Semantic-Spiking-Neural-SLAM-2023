"""Stage 2: joint grid over the Stage-1 survivors.

Survivors (guard-verdicted, 2026-08-17): h2's zscore_control (ADOPT-CANDIDATE,
dmacc +0.0163 +- 0.0053, dmf1 +0.0120 +- 0.0018) and h1's abstain family
(SURVIVES via mF1 at an mAcc cost that grows with tau). Both are pure
F-transforms, so every combo runs off the warm baseline cache -- the whole
grid is minutes, not hours.

Composition: z-score the fields (h2), then threshold-decode the z-scored
fields (h1). h1's own z-scoring is idempotent on already-z-scored input, so
the compose order is well-defined.

Pre-registered hypothesis (also in PREDICTIONS): LIGHT abstain (tau 0.5-1.0)
on top of z-score keeps the mAcc gain and adds F1; heavy abstain keeps
trading mAcc away. The grid includes zscore_only recomputed in-run as the
paired single-mechanism control, so "does coupling beat the best single?" is
answered within one guarded run, never across runs.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    default_fields, make_blob_data, run_screen, score)
from collab_tasks.batch1.h1_threshold import _decode  # noqa: E402

PRED = ("Pre-registered: light abstain (tau 0.5-1.0) composed with the "
        "z-score decode keeps zscore's mAcc gain (within 0.005) while adding "
        "F1 on top of zscore's own +0.012; heavy abstain (tau >= 2) costs "
        "mAcc as it did alone. Adoption rule as spec: mAcc up and mF1 not "
        "down, judged against BASELINE as always.")

PREDICTIONS = [
    {"id": "s2-light",
     "text": ("combo_tau0.5_abstain holds zscore_only's mAcc to within "
              "0.005 (pair gap >= -0.005)"),
     "metric": "macc", "scope": ["combo_tau0.5_abstain", "zscore_only"],
     "test": "pair_ge", "value": -0.005},
    {"id": "s2-f1",
     "text": "some combo beats zscore_only on mF1",
     "metric": "mf1", "scope": ["combo_tau1.0_abstain", "zscore_only"],
     "test": "pair_ge", "value": 0.0},
    {"id": "s2-adopt",
     "text": "best variant still clears the adoption bar on mAcc (>= +0.005)",
     "metric": "macc", "scope": "best", "test": "ge", "value": 0.005},
]


def _zscore(F):
    F = np.asarray(F, float)
    return (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)


def zscore_only(data, ti):
    F, names, cell = default_fields(data, ti)
    Z = _zscore(F)
    from collab_tasks.batch1.common import predict
    return predict(Z, names, cell)


def combo(tau, abstain):
    def fn(data, ti):
        F, names, cell = default_fields(data, ti)
        return _decode(_zscore(F), names, cell, tau, abstain)
    return fn


VARIANTS = {"zscore_only": zscore_only}
for _t in (0.5, 1.0, 1.5, 2.0, 3.0):
    for _m in ("fallback", "abstain"):
        VARIANTS[f"combo_tau{_t}_{_m}"] = combo(_t, _m == "abstain")


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["zscore_only"](blob, 0))
    assert m["macc"] > 0.9, m
    m = score(blob["gt"], VARIANTS["combo_tau1.0_fallback"](blob, 0))
    assert m["macc"] > 0.9, m
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("stage2_joint", PRED, VARIANTS, predictions=PREDICTIONS)
