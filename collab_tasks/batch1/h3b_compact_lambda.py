"""H3b: shrink lambda for COMPACT classes only -- the targeted h3 retest.

Mechanism-level motivation (local_loss_forensics.py, 2026-08-18): in the 38%
of local losses where the GT class's point is nearer, 80% of winners are also
inside one lambda and 52% inside two grid cells -- sub-resolution ties, with
compact classes (cushion, pillar, bin, vent) beating their extended hosts
(sofa, table, comforter) at cm scales the lambda = 0.45 m kernel cannot
resolve. A compact class does not NEED a 0.45 m kernel: shrinking its lambda
makes its field fall off faster away from the object it actually is, so the
host class reclaims the interleaved cells. Batch-1's h3 scaled every class's
lambda from spread and was UNDECIDABLE (-0.0068 +- 0.0101); this targets the
shrink at the only classes the mechanism says should shrink.

Selection is GEOMETRY-ONLY, deterministic, and frozen here before any run.
A first draft used RMS spread about a single class centre; that conflates
"compact object" with "single cluster" -- nine cushions spread around a room
have large global spread even though each is small, and the rule missed
exactly the classes the mechanism names. Caught BEFORE any screen ran.
The rule as frozen: cluster each class's raw observations by single linkage
on a 0.30 m occupancy grid (8-neighbour connectivity), then a class is
COMPACT iff the observation-weighted mean per-cluster RMS spread is
< 0.35 m. Per-instance size, not scene-wide footprint. No error data, no
per-scene tuning, no hand-picked class lists.

Controls, both essential:
  rand0.6 -- shrink a RANDOM subset of classes of the same size (matched
             count, seeded per scene+tuple). If random shrinking matches
             targeted shrinking, the targeting story is dead (h6's lesson).
  inv0.6  -- shrink the EXTENDED classes instead. The mechanism predicts
             this does NOT help; if it does, the mechanism story is wrong.

Honest ceiling: the 38% branch is ~5,400 cells of ~14,100 local losses; the
62% branch (stream's own nearest label wrong) is untouchable by any decode
or kernel change. Expected effect is small; the point is whether it is REAL
and TARGETED, not whether it is large.

    python collab_tasks/batch1/h3b_compact_lambda.py --self-test
    python collab_tasks/batch1/h3b_compact_lambda.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    LX, LY, SEEDS, class_fields, make_blob_data, predict, run_screen, score)

PRED = ("Pre-registered 2026-08-18, frozen in git before any run. Compact "
        "classes (raw-obs RMS spread < 0.50 m) get lambda scaled by gamma; "
        "extended classes keep the default. The mechanism predicts a small "
        "REAL gain that BEATS a matched random-target control, and that "
        "shrinking the extended classes instead does NOT help. The 62% "
        "branch (stream labels wrong) is out of reach by construction; "
        "expected magnitude is +0.005..+0.02, not a gap-closer.")

SPREAD_T = 0.35          # metres; compact iff mean per-CLUSTER RMS below
LINK_CELL = 0.30         # metres; single-linkage occupancy-grid cell
GAMMAS = (0.4, 0.6, 0.8)

PREDICTIONS = [
    {"id": "b2l-acc",
     "text": "the best variant gains >= +0.005 mAcc (the adoption bar), "
             "resolved against tuple noise",
     "metric": "macc", "scope": "best", "test": "ge", "value": 0.005},
    {"id": "b2l-breadth",
     "text": "the gain is broad: positive on >= 6 of 8 scenes (the office4 "
             "lesson, now a pre-registered clause)",
     "metric": "macc", "scope": "best", "test": "scenes_ge", "value": 6},
    {"id": "b2l-target",
     "text": "targeting matters: compact-shrink at gamma 0.6 beats the "
             "matched random-target control by >= 0.002, gap resolved",
     "metric": "macc", "scope": ["c0.6", "rand0.6"], "test": "pair_ge",
     "value": 0.002},
    {"id": "b2l-inv",
     "text": "falsifier: shrinking the EXTENDED classes instead does not "
             "gain more than +0.002 -- if it does, the sub-lambda-tie "
             "mechanism is the wrong explanation",
     "metric": "macc", "scope": "inv0.6", "test": "le", "value": 0.002},
    {"id": "b2l-f1",
     "text": "the compact classes are not sacrificed: mF1 of c0.6 does not "
             "fall more than 0.005",
     "metric": "mf1", "scope": "c0.6", "test": "ge", "value": -0.005},
]


def _clusters(P):
    """Single-linkage components of 2D points on a LINK_CELL occupancy grid
    (8-neighbour connectivity). Deterministic, label-free, pure numpy."""
    cells = {}
    for i, (x, y) in enumerate(P):
        cells.setdefault((int(np.floor(x / LINK_CELL)),
                          int(np.floor(y / LINK_CELL))), []).append(i)
    parent = {k: k for k in cells}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for (cx, cy) in list(cells):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (cx + dx, cy + dy)
                if nb in cells:
                    parent[find((cx, cy))] = find(nb)
    groups = {}
    for k, idxs in cells.items():
        groups.setdefault(find(k), []).extend(idxs)
    return [np.array(v, int) for v in groups.values()]


def _spread(data):
    """{cls: observation-weighted mean per-CLUSTER RMS spread}, raw stream.

    Per-instance size, not scene-wide footprint: nine cushions around a room
    are compact; one sofa is not. Deterministic (no cap draw), so the class
    partition is identical across seed tuples and a variant differs from
    baseline ONLY in the scales dict.
    """
    by = {}
    for o in data["obs"]:
        by.setdefault(o["cls"], []).append((o["x"], o["y"]))
    out = {}
    for c, pts in by.items():
        P = np.array(pts, float)
        num = den = 0.0
        for idx in _clusters(P):
            Q = P[idx]
            rms = float(np.sqrt(((Q - Q.mean(0)) ** 2).mean(0)).mean())
            num += rms * len(idx)
            den += len(idx)
        out[c] = num / max(den, 1.0)
    return out


def compact_classes(data):
    sp = _spread(data)
    return sorted(c for c, v in sp.items() if v < SPREAD_T)


def _run(data, ti, scales):
    F, names, cell = class_fields(data, scales=scales, seeds=SEEDS[ti])
    return predict(F, names, cell)


def variant_compact(gamma):
    def fn(data, ti):
        sc = {c: (gamma * LX, gamma * LY) for c in compact_classes(data)}
        return _run(data, ti, sc)
    return fn


def variant_random(gamma):
    def fn(data, ti):
        names = sorted({o["cls"] for o in data["obs"]})
        k = len(compact_classes(data))
        rng = np.random.RandomState(1000 + ti * 97
                                    + sum(map(ord, data["scene"])) % 1000)
        pick = rng.choice(names, size=min(k, len(names)), replace=False)
        sc = {c: (gamma * LX, gamma * LY) for c in pick}
        return _run(data, ti, sc)
    return fn


def variant_inverse(gamma):
    def fn(data, ti):
        comp = set(compact_classes(data))
        names = sorted({o["cls"] for o in data["obs"]})
        sc = {c: (gamma * LX, gamma * LY) for c in names if c not in comp}
        return _run(data, ti, sc)
    return fn


VARIANTS = {
    "c0.4": variant_compact(0.4),
    "c0.6": variant_compact(0.6),
    "c0.8": variant_compact(0.8),
    "rand0.6": variant_random(0.6),
    "inv0.6": variant_inverse(0.6),
}


def _self_test():
    blob = make_blob_data()
    sp = _spread(blob)
    assert all(v < SPREAD_T for v in sp.values()), \
        f"blob classes (std 0.25) must all be compact: {sp}"
    m = score(blob["gt"], VARIANTS["c0.6"](blob, 0))
    assert m["macc"] > 0.9, m
    # inverse on blob shrinks nothing -> must equal the default build exactly
    base = predict(*class_fields(blob, seeds=SEEDS[0]))
    inv = VARIANTS["inv0.6"](blob, 0)
    assert (base == inv).all(), "inverse variant must be identity here"
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h3b_compact_lambda", PRED, VARIANTS,
                   predictions=PREDICTIONS)
