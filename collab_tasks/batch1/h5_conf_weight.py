"""H5: weight each observation by its detector confidence when bundling.
The harness prints the conf distribution first -- if conf is near-constant
this mechanism is dead on arrival and the prediction says so.

Rebuilds the trace per variant per tuple: class_fields(..., weight_fn=...,
seeds=SEEDS[ti]).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, SEEDS, class_fields, load_scene, make_blob_data, predict,
    run_screen, score)

PRED = ("Pre-registered: small, likely +-0.005 -- conf in the cgfront "
        "stream is expected near-constant (their objects re-emit one conf). "
        "Included because it is cheap and would otherwise stay folklore.")

PREDICTIONS = [
    {"id": "h5-null",
     "text": ("conf weighting is within 0.005 mAcc (expected UNDECIDABLE - "
              "conf is near-constant)"),
     "metric": "macc", "scope": "best", "test": "within", "value": 0.005},
]


def variant(gamma):
    def fn(data, ti):
        F, names, cell = class_fields(
            data, weight_fn=lambda o: float(o.get("conf", 1.0)) ** gamma,
            seeds=SEEDS[ti])
        return predict(F, names, cell)
    return fn


VARIANTS = {"gamma1": variant(1.0), "gamma2": variant(2.0)}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["gamma1"](blob, 0))
    assert m["macc"] > 0.9, m
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        confs = [float(o.get("conf", 1.0)) for s in SCENES[:2]
                 for o in load_scene(s)["obs"]]
        print(f"conf stats (2 scenes): mean {np.mean(confs):.3f} "
              f"sd {np.std(confs):.3f} min {min(confs):.3f} "
              f"max {max(confs):.3f}")
        run_screen("h5_conf_weight", PRED, VARIANTS, predictions=PREDICTIONS)
