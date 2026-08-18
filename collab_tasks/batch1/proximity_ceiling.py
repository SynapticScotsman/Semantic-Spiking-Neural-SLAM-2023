"""Does a PERFECT proximity decoder on our own stream close the gap?

The field-native diagnosis said: our kernel is nearly flat at short range, so
it discards proximity, which is exactly the signal ConceptGraphs' nearest-
neighbour readout runs on. That story predicts a proximity decoder over our
own observations should be much better than our field.

This tests it directly and cheaply. Label every eval cell by the class of its
NEAREST capped observation -- a perfect, count-free proximity decoder on our
exact input, with no kernel, no superposition and no crosstalk. Then sweep
distance-weighted kNN, the strongest non-parametric decoder available on
(x, y) -> class, to bound what ANY readout over this stream can reach.

If NN is barely above our field, "we discard proximity" is not the gap.
If even the best kNN falls well short of their 0.402, the remaining distance
is not memory-addressable at all -- it is information their map has and our
observation stream never carried.

    python collab_tasks/batch1/proximity_ceiling.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    GRID, LX, SCENES, SEEDS, cap_per_class, default_fields, load_scene,
    predict, score)

KS = (1, 5, 9, 25, 75)


def main():
    rows = {"field": [], **{f"knn{k}": [] for k in KS}}
    cover = []
    for s in SCENES:
        data = load_scene(s)
        F, names, cell = default_fields(data, 0)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        rows["field"].append(score(gt, predict(F, names, cell)))

        capped = cap_per_class(list(data["obs"]), 400, seed=SEEDS[0][2])
        P = np.array([[o["x"], o["y"]] for o in capped], float)
        L = np.array([o["cls"] for o in capped], object)
        Q = np.c_[xyz[:, a], xyz[:, b]]
        # full distance matrix in chunks (30k x ~6k is fine in float32)
        order_cls = sorted(set(L))
        ci = {c: i for i, c in enumerate(order_cls)}
        Li = np.array([ci[c] for c in L])
        pred = {k: np.empty(len(Q), object) for k in KS}
        near_own = np.zeros(len(Q), bool)
        absent = 0
        gt_set = set(order_cls)
        for st in range(0, len(Q), 4000):
            q = Q[st:st + 4000]
            d = np.sqrt(((q[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            idx = np.argsort(d, axis=1)
            for k in KS:
                sel = idx[:, :k]
                dd = np.take_along_axis(d, sel, 1)
                w = 1.0 / np.maximum(dd, 1e-6)
                votes = np.zeros((len(q), len(order_cls)))
                for j in range(k):
                    np.add.at(votes, (np.arange(len(q)), Li[sel[:, j]]),
                              w[:, j])
                pred[k][st:st + 4000] = np.array(order_cls,
                                                 object)[votes.argmax(1)]
            # coverage: is ANY observation of this point's own GT class
            # within one lambda?
            for r, gq in enumerate(gt[st:st + 4000]):
                if gq not in gt_set:
                    absent += 1
                    continue
                m = Li == ci[gq]
                if m.any() and d[r][m].min() <= LX:
                    near_own[st + r] = True
        for k in KS:
            rows[f"knn{k}"].append(score(gt, pred[k]))
        cover.append(dict(scene=s, near_own=float(near_own.mean()),
                          gt_absent=absent / len(Q)))
        print(f"{s} done", flush=True)

    print("\nWHAT ANY DECODER OVER OUR OWN STREAM CAN REACH")
    print(f"{'decoder':<12}{'mAcc':>9}{'F-mIoU':>10}{'vs field':>11}")
    print("-" * 44)
    fm = np.mean([r["macc"] for r in rows["field"]])
    ff = np.mean([r["fmiou"] for r in rows["field"]])
    print(f"{'our field':<12}{fm:>9.4f}{ff:>10.4f}{'--':>11}")
    best = ("field", fm)
    for k in KS:
        m = np.mean([r["macc"] for r in rows[f"knn{k}"]])
        f = np.mean([r["fmiou"] for r in rows[f"knn{k}"]])
        if m > best[1]:
            best = (f"knn{k}", m)
        print(f"{'kNN k='+str(k):<12}{m:>9.4f}{f:>10.4f}{m-fm:>+11.4f}")
    print("-" * 44)
    print(f"{'ConceptGraphs':<12}{0.402:>9.4f}")
    nn = np.mean([r["macc"] for r in rows["knn1"]])
    print(f"\nPerfect proximity (k=1) buys {nn-fm:+.4f} mAcc over our field.")
    print(f"Best decoder on our stream: {best[0]} at {best[1]:.4f}.")
    print(f"Memory-addressable budget = {best[1]-fm:+.4f} "
          f"(NOT the +0.113 'reachable headroom').")
    print(f"Unreachable from any memory = {0.402-best[1]:+.4f} "
          f"-- information their map has that our stream never carried.")
    c1 = np.mean([c["near_own"] for c in cover])
    c2 = np.mean([c["gt_absent"] for c in cover])
    print(f"\nStream coverage: {100*c1:.1f}% of eval points have ANY "
          f"observation of their own GT class within one lambda;")
    print(f"                 {100*c2:.1f}% have their GT class absent from "
          f"the stream entirely.")

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(field_macc=float(fm), field_fmiou=float(ff),
                   knn={f"k{k}": dict(
                       macc=float(np.mean([r["macc"] for r in rows[f"knn{k}"]])),
                       fmiou=float(np.mean([r["fmiou"] for r in rows[f"knn{k}"]])))
                       for k in KS},
                   best=best[0], best_macc=float(best[1]),
                   addressable=float(best[1] - fm),
                   unreachable=float(0.402 - best[1]),
                   coverage=cover),
              open("outputs/batch1/proximity_ceiling.json", "w"), indent=1)
    print("\nwrote outputs/batch1/proximity_ceiling.json")


if __name__ == "__main__":
    main()
