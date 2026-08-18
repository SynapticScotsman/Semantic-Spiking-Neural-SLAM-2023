"""What question does the Replica benchmark actually ask?

Paul, playing with the artifact: at a high isolevel the surviving points look
perfect, so is the benchmark asking about THOSE points, or about points near
them?

The answer decides whether our memory is being scored on the task it suits.
Two things settle it, both measurable:

  1. WHERE are the eval points? If they are surface points on objects, then
     every one of them belongs to some object and the benchmark is per-point
     SEMANTIC SEGMENTATION: for each surface point, which object class is it.
     Abstaining is then simply refusing to answer part of the question.
     If they were scattered through free space, a peak-only answer would be
     defensible.
  2. HOW FAR is a typical eval point from the nearest observation of ANY
     class? If the map's evidence is dense everywhere the points are, then
     the peaks-only reading is discarding coverage the benchmark demands.

    python collab_tasks/batch1/what_is_being_asked.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    CAP, CG_EXCLUDE_6, LX, SCENES, SEEDS, cap_per_class, load_scene)


def main():
    rows = []
    for s in SCENES:
        d = load_scene(s)
        gt, xyz, a, b = d["gt"], d["xyz"], d["a"], d["b"]
        keep = ~np.isin(gt, CG_EXCLUDE_6)
        Q = np.c_[xyz[keep, a], xyz[keep, b]]

        capped = cap_per_class(list(d["obs"]), CAP, seed=SEEDS[0][2])
        P = np.array([[o["x"], o["y"]] for o in capped], float)

        dmin = np.empty(len(Q))
        for st in range(0, len(Q), 4000):
            q = Q[st:st + 4000]
            dd = np.sqrt(((q[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            dmin[st:st + 4000] = dd.min(1)

        rows.append(dict(
            scene=s, n_scored=int(keep.sum()),
            med=float(np.median(dmin)),
            within_5cm=float((dmin <= 0.05).mean()),
            within_lambda=float((dmin <= LX).mean()),
            beyond_2lambda=float((dmin > 2 * LX).mean())))
        print(f"  {s} done", flush=True)

    print("\nHOW FAR IS A SCORED EVAL POINT FROM THE NEAREST OBSERVATION "
          "(any class)?\n")
    print(f"{'scene':<9}{'scored pts':>12}{'median':>9}{'within 5cm':>12}"
          f"{'within lambda':>15}{'beyond 2 lambda':>17}")
    print("-" * 74)
    for r in rows:
        print(f"{r['scene']:<9}{r['n_scored']:>12,}{r['med']:>8.3f}m"
              f"{100*r['within_5cm']:>11.0f}%{100*r['within_lambda']:>14.0f}%"
              f"{100*r['beyond_2lambda']:>16.0f}%")
    print("-" * 74)
    md = float(np.mean([r["med"] for r in rows]))
    w5 = float(np.mean([r["within_5cm"] for r in rows]))
    wl = float(np.mean([r["within_lambda"] for r in rows]))
    b2 = float(np.mean([r["beyond_2lambda"] for r in rows]))
    print(f"{'MEAN':<9}{'':>12}{md:>8.3f}m{100*w5:>11.0f}%{100*wl:>14.0f}%"
          f"{100*b2:>16.0f}%")

    print(f"\nThe scored points sit a median {md*100:.0f} cm from the nearest "
          f"observation and\n{100*wl:.0f}% are inside one lambda. They are "
          f"SURFACE points on objects, not free space:")
    print("the benchmark asks, for every surface point, which object class it "
          "belongs to.")
    print("That is per-point semantic segmentation. A peaks-only answer "
          "refuses most of\nthe question, which is exactly why raising the "
          "abstain threshold trades recall away.")

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(per_scene=rows, mean_median_dist=md,
                   mean_within_lambda=wl),
              open("outputs/batch1/what_is_being_asked.json", "w"), indent=1)
    print("\nwrote outputs/batch1/what_is_being_asked.json")


if __name__ == "__main__":
    main()
