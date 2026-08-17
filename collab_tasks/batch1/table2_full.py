"""Everything Table II can honestly give us, which we have not been reporting.

Three things, none of which is a new measurement -- all read artifacts already
on disk and use ConceptGraphs' own scorer:

  1. F-mIoU beside mAcc. Their published row is 40.63 / 35.95; we have only
     ever quoted the mAcc half. Reproducing a published row on TWO independent
     metrics is far stronger evidence than one scalar.
  2. Per-scene table with seed variance, not a single averaged number -- our
     own house rule says one number is not a result.
  3. Their n_exclude LADDER (1 / 4 / 6). The paper never states which
     exclusion setting produced Table II; we inferred 6 because it lands us at
     0.402 against their published 0.406. That inference is the single most
     likely attack on the whole reproduction claim, and running their own
     three settings costs 30 seconds.

    python collab_tasks/batch1/table2_full.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, SEEDS, _SC, default_fields, load_scene, predict, score)

PUB = dict(macc=0.4063, fmiou=0.3595)     # ConceptGraphs Table II, zero-shot
LADDER = {1: ("other",),
          4: ("other", "floor", "wall", "ceiling"),
          6: ("other", "floor", "wall", "ceiling", "door", "window")}


def sign_test(deltas):
    """Two-sided exact sign test. n=8 bottoms out at p=0.0078; scipy is not a
    repo dependency so this is done by hand from the binomial."""
    from math import comb
    n = sum(1 for d in deltas if d != 0)
    k = sum(1 for d in deltas if d > 0)
    lo = min(k, n - k)
    p = 2 * sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return k, n, min(p, 1.0)


def main():
    rows, out = {}, {}
    for s in SCENES:
        data = load_scene(s)
        gt, xyz = data["gt"], data["xyz"]
        F, nm, cell = default_fields(data, 0)
        ours = predict(F, nm, cell)
        C = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                    allow_pickle=True)
        lab = _SC.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
        theirs = np.array([v if v is not None else "__none__" for v in lab])
        rows[s] = (gt, ours, theirs)

    # ---- 1 + 2: both metrics, per scene, with across-seed sd for ours ----
    print("TABLE II, REPRODUCED IN FULL (their scorer, n_exclude 6)")
    print(f"{'scene':<9}{'theirs mAcc':>12}{'ours mAcc':>11}{'+-sd':>8}"
          f"{'theirs FmIoU':>14}{'ours FmIoU':>12}")
    print("-" * 66)
    tm, om, tf, of, seedsd = [], [], [], [], []
    for s in SCENES:
        gt, ours, theirs = rows[s]
        a = score(gt, theirs)
        b = score(gt, ours)
        per_seed = []
        for ti in range(len(SEEDS)):
            d = load_scene(s)
            Fi, ni, ci = default_fields(d, ti)
            per_seed.append(score(d["gt"], predict(Fi, ni, ci))["macc"])
        sd = float(np.std(per_seed, ddof=1))
        tm.append(a["macc"]); om.append(b["macc"])
        tf.append(a["fmiou"]); of.append(b["fmiou"]); seedsd.append(sd)
        win = "  <- we lead" if b["macc"] > a["macc"] else ""
        print(f"{s:<9}{a['macc']:>12.3f}{b['macc']:>11.3f}{sd:>8.3f}"
              f"{a['fmiou']:>14.3f}{of[-1]:>12.3f}{win}")
        out[s] = dict(theirs=a, ours=b, ours_seed_sd=sd)
    print("-" * 66)
    print(f"{'MEAN':<9}{np.mean(tm):>12.3f}{np.mean(om):>11.3f}"
          f"{np.mean(seedsd):>8.3f}{np.mean(tf):>14.3f}{np.mean(of):>12.3f}")
    print(f"{'PUBLISHED':<9}{PUB['macc']:>12.3f}{'':>11}{'':>8}"
          f"{PUB['fmiou']:>14.3f}")
    print(f"\nreproduction fidelity: mAcc {100*np.mean(tm)/PUB['macc']:.1f}% "
          f"of published, F-mIoU {100*np.mean(tf)/PUB['fmiou']:.1f}% "
          "-- BOTH metrics, not one")
    print(f"our ratio to their measured score: "
          f"mAcc {np.mean(om)/np.mean(tm):.2f}, "
          f"F-mIoU {np.mean(of)/np.mean(tf):.2f}")
    d = [o - t for o, t in zip(om, tm)]
    k, n, p = sign_test(d)
    print(f"paired sign test over {n} scenes: we lead on {k}, "
          f"two-sided p = {p:.4f}")
    print("  (n=8 cannot show a small difference is significant; it shows the "
          "gap is\n   consistent in DIRECTION, and reverses on the scenes we "
          "win)")

    # ---- 3: their own exclusion ladder ----
    print("\nTHEIR n_exclude LADDER -- the setting their paper never states")
    print(f"{'n_exclude':>10}{'classes dropped':>18}"
          f"{'theirs mAcc':>13}{'ours mAcc':>11}{'gap':>8}"
          f"{'theirs FmIoU':>14}{'ours FmIoU':>12}")
    print("-" * 86)
    lad = {}
    for n_ex, excl in LADDER.items():
        a = [ _SC.macc_fmiou(rows[s][0], rows[s][2], exclude=excl)
              for s in SCENES]
        b = [ _SC.macc_fmiou(rows[s][0], rows[s][1], exclude=excl)
              for s in SCENES]
        TM, OM = np.mean([x[0] for x in a]), np.mean([x[0] for x in b])
        TF, OF = np.mean([x[1] for x in a]), np.mean([x[1] for x in b])
        lad[n_ex] = dict(theirs_macc=TM, ours_macc=OM,
                         theirs_fmiou=TF, ours_fmiou=OF)
        print(f"{n_ex:>10}{len(excl):>18}{TM:>13.3f}{OM:>11.3f}"
              f"{OM-TM:>+8.3f}{TF:>14.3f}{OF:>12.3f}")
    print("-" * 86)
    best = min(LADDER, key=lambda n: abs(lad[n]["theirs_macc"] - PUB["macc"]))
    print(f"closest to their published {PUB['macc']:.4f}: n_exclude={best} "
          f"(theirs {lad[best]['theirs_macc']:.4f}) -- this is the empirical "
          "basis\nfor our protocol inference, now stated explicitly rather "
          "than assumed")
    signs = {n: np.sign(lad[n]["ours_macc"] - lad[n]["theirs_macc"])
             for n in LADDER}
    print(f"gap sign across the ladder: "
          + ", ".join(f"n{n}={'ours' if v > 0 else 'theirs'}"
                      for n, v in signs.items())
          + ("  -- STABLE" if len(set(signs.values())) == 1
             else "  -- FLIPS, report the ladder"))

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(per_scene=out, ladder=lad, published=PUB,
                   sign_test=dict(k=k, n=n, p=p)),
              open("outputs/batch1/table2_full.json", "w"), indent=1)
    print("\nwrote outputs/batch1/table2_full.json")


if __name__ == "__main__":
    main()
