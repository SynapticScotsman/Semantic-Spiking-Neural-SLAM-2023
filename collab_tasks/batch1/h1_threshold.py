"""H1: per-class z-threshold decode, abstain and fallback forms.

The field transform is pure post-processing over the cached baseline fields
(Amendment A7: cheap), so variants read default_fields(data, ti) and never
rebuild the trace.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SEEDS, class_fields, default_fields, make_blob_data, run_screen, score)

PRED = ("Pre-registered: threshold is an ABSTAIN mechanism, i.e. a precision "
        "tool. Expect mF1 up via mprec at some tau; mAcc within +-0.01 of "
        "baseline. Prior +0.014 was 3 scenes at the stale config; this is "
        "the 8-scene, precision-aware re-test.")

PREDICTIONS = [
    {"id": "h1-f1",
     "text": ("threshold is an abstain/precision tool: best variant mF1 up "
              "at least 0.005"),
     "metric": "mf1", "scope": "best", "test": "ge", "value": 0.005},
    {"id": "h1-acc",
     "text": "mAcc stays within 0.010 of baseline at the best variant",
     "metric": "macc", "scope": "best", "test": "within", "value": 0.010},
]


def _decode(F, names, cell, tau, abstain):
    """Per-class z-score of the field; cells where no class clears tau
    either abstain (emit "__none__") or fall back to the plain argmax."""
    F = np.asarray(F, float)
    Z = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)
    above = Z > tau
    masked = np.where(above, F, -np.inf)
    win = masked.argmax(0)
    none = ~above.any(0)
    if abstain:
        lab = np.array([("__none__" if n else names[w])
                        for w, n in zip(win, none)])
    else:
        win = win.copy()
        if none.any():
            win[none] = F[:, none].argmax(0)
        lab = np.array([names[w] for w in win])
    return lab[cell]


def variant(tau, abstain):
    def fn(data, ti):
        F, names, cell = default_fields(data, ti)
        return _decode(F, names, cell, tau, abstain)
    return fn


VARIANTS = {f"tau{t}_{m}": variant(t, m == "abstain")
            for t in (0.5, 1.0, 1.5, 2.0, 3.0)
            for m in ("fallback", "abstain")}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["tau1.0_fallback"](blob, 0))
    assert m["macc"] > 0.9, m
    # abstain must actually abstain: a probe far from every blob. Build the
    # modified blob's fields with class_fields, not default_fields, so no
    # cached field/cell state can leak in.
    blob2 = dict(blob)
    blob2["xyz"] = np.vstack([blob["xyz"], [[9.0, 9.0, 0.0]]])
    blob2["gt"] = np.append(blob["gt"], "chair")
    F, nm, cell = class_fields(blob2, seeds=SEEDS[0])
    p = _decode(F, nm, cell, 2.0, True)
    assert p[-1] == "__none__", "far-away point did not abstain"
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h1_threshold", PRED, VARIANTS, predictions=PREDICTIONS)
