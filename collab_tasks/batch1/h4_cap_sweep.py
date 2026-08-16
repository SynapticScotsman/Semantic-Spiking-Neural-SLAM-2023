"""H4: insertion cap at the corrected configuration. The old 'saturates by
200' was measured at stale lambda and buggy labels; this is the 8-scene,
seed-battery re-sweep.

Rebuilds the trace per variant per tuple: class_fields(..., cap=cap,
seeds=SEEDS[ti]).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SEEDS, class_fields, make_blob_data, predict, run_screen, score)

PRED = ("Pre-registered: current 400 near-optimal; the no-cap audit showed "
        "-0.044, so 800 should be flat-to-down and 100 should cost recall. "
        "cap400_sanity must be delta 0.000 exactly at every tuple.")

PREDICTIONS = [
    {"id": "h4-flat",
     "text": "no cap beats 400 by more than 0.005 mAcc",
     "metric": "macc", "scope": "best", "test": "le", "value": 0.005},
]


def variant(cap):
    def fn(data, ti):
        F, names, cell = class_fields(data, cap=cap, seeds=SEEDS[ti])
        return predict(F, names, cell)
    return fn


# cap400_sanity: cap_per_class draws are nested prefixes at fixed seed
# (Amendment A5, verified k200 subset-of k400 subset-of k800), so its delta
# is 0.000 true BY CONSTRUCTION at every tuple. It is a plumbing check on
# the harness code path only and carries ZERO evidence about draw
# sensitivity.
VARIANTS = {"cap100": variant(100), "cap200": variant(200),
            "cap400_sanity": variant(400), "cap800": variant(800)}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["cap200"](blob, 0))
    assert m["macc"] > 0.9, m
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h4_cap_sweep", PRED, VARIANTS, predictions=PREDICTIONS)
