"""H6: geometric outlier rejection before bundling -- the cap generalised.
An observation farther than max(r * MAD, 0.5 m) from its class's median
centre is rejected. Attacks bleed at the source.

Amendment A4: filtering before cap_per_class changes the cap pool, so the
RNG redraws a large fraction of the bundled subset for reasons unrelated to
geometry. Every r{X} variant therefore has a matched rand{X} control that
drops the SAME COUNT per class uniformly at random (seeded per tuple), and
the geometric claim is evaluated as delta(r{X}) - delta(rand{X}), never
against baseline alone.

Rebuilds the trace per variant per tuple: class_fields(..., keep_fn=...,
seeds=SEEDS[ti]).
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SEEDS, class_fields, make_blob_data, predict, run_screen, score)

PRED = ("Pre-registered: mAcc up on vent-type scenes (room0/office2-4), "
        "mF1 up broadly via precision, and the geometric filter must beat "
        "its matched random-drop control (Amendment A4) or the effect is "
        "cap-pool churn, not geometry. Floor of 0.5 m stops tight classes "
        "being nuked. Risk: legitimately large objects (sofa) losing their "
        "own extent at small r.")

PREDICTIONS = [
    {"id": "h6-geo",
     "text": ("geometric filter beats its matched random-drop control at a "
              "majority of r values"),
     "metric": "macc",
     "scope": [["r2.0", "rand2.0"], ["r3.0", "rand3.0"], ["r4.0", "rand4.0"]],
     "test": "pairs_majority_ge", "value": 0.0},
    {"id": "h6-f1",
     "text": "best geometric filter variant mF1 up at least 0.005",
     "metric": "mf1", "scope": "best", "test": "ge", "value": 0.005},
]

FLOOR = 0.5   # metres -- never reject inside this radius


def keep_factory(data, r):
    """Keep an observation iff it lies within max(r*MAD, FLOOR) of its
    class's median centre, both computed over data["obs"]."""
    stats = {}
    by = {}
    for o in data["obs"]:
        by.setdefault(o["cls"], []).append((o["x"], o["y"]))
    for c, pts in by.items():
        P = np.array(pts, float)
        ctr = np.median(P, 0)
        d = np.sqrt(((P - ctr) ** 2).sum(1))
        mad = float(np.median(d))
        stats[c] = (ctr, max(r * mad, FLOOR))

    def keep(o):
        ctr, lim = stats[o["cls"]]
        return float(np.hypot(o["x"] - ctr[0], o["y"] - ctr[1])) <= lim
    return keep


def rand_keep_factory(data, r, ti):
    """Matched control: drop the SAME COUNT per class as the geometric
    filter at this r, chosen uniformly at random from the SAME data["obs"]
    the geometric filter saw, with the redraw seeded per tuple."""
    geo = keep_factory(data, r)
    by = {}
    for o in data["obs"]:
        by.setdefault(o["cls"], []).append(o)
    rng = np.random.default_rng(1000 * ti + int(r * 10))
    dropped = set()
    for c in sorted(by):        # sorted: deterministic RNG consumption order
        rows = by[c]
        n_drop = sum(1 for o in rows if not geo(o))
        if n_drop:
            picks = rng.choice(len(rows), size=n_drop, replace=False)
            dropped.update(id(rows[i]) for i in picks)

    def keep(o):
        return id(o) not in dropped
    return keep


def variant(r):
    def fn(data, ti):
        F, names, cell = class_fields(data, keep_fn=keep_factory(data, r),
                                      seeds=SEEDS[ti])
        return predict(F, names, cell)
    return fn


def rand_variant(r):
    def fn(data, ti):
        F, names, cell = class_fields(
            data, keep_fn=rand_keep_factory(data, r, ti), seeds=SEEDS[ti])
        return predict(F, names, cell)
    return fn


VARIANTS = {}
for _r in (2.0, 3.0, 4.0):
    VARIANTS[f"r{_r}"] = variant(_r)
    VARIANTS[f"rand{_r}"] = rand_variant(_r)


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["r3.0"](blob, 0))
    assert m["macc"] > 0.9, m
    # a far outlier must be rejected by the geometric filter
    blob2 = dict(blob, obs=blob["obs"] + [dict(frame=0, cls="chair",
                 conf=1.0, det=9999, x=50.0, y=50.0)])
    geo = keep_factory(blob2, 3.0)
    assert not geo(blob2["obs"][-1]), "outlier at (50,50) not rejected"
    # the matched control must drop exactly the same count per class
    rnd = rand_keep_factory(blob2, 3.0, 0)
    kept_geo = Counter(o["cls"] for o in blob2["obs"] if geo(o))
    kept_rnd = Counter(o["cls"] for o in blob2["obs"] if rnd(o))
    assert kept_geo == kept_rnd, (kept_geo, kept_rnd)
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h6_insertion_filter", PRED, VARIANTS,
                   predictions=PREDICTIONS)
