"""Why the FIELD loses, in the field's own quantities. Not shares of cells.

For every local_loss cell where the GT class's own observation is NEARER than
the winner's (the 5,366 cells that indict the decode rather than the stream),
read the two numbers the VSA actually decides on:

    f_gt  = F_gt(cell)      the GT class's field height, after unbinding
    f_win = F_win(cell)     the winner's field height

and compare them against what a proximity decoder would have seen. Three
field-native failure signatures, which are NOT the same thing and were being
lumped together as "sub-lambda resolution":

  INTERFERENCE  f_gt <= 0 -- the class's own field is at or below zero at a
                cell within centimetres of its own observation. Crosstalk from
                the rest of the superposition has cancelled it. No kernel
                width or per-class lambda can fix this; it is the price of
                bundling N classes into one 4096-d vector.
  LOCAL MASS    f_gt > 0 but the winner has >= 2x as many observations inside
                one kernel width. The kernel is nearly flat over the distances
                involved (k(4cm)=0.987 vs k(15cm)=0.830), so a 2x count beats
                a 3x proximity advantage. This is the one per-class lambda
                could attack -- and h3b says it is not enough on its own.
  FLAT KERNEL   f_gt > 0, mass roughly equal: the two fields are genuinely
                close and the kernel cannot resolve which object owns the
                cell.

Also records the margin distribution, because "we lost narrowly" and "we lost
by 20x" are different diagnoses and the earlier writeups never separated them.

    python collab_tasks/batch1/field_why.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    LX, SCENES, SEEDS, cap_per_class, default_fields, load_scene, predict)
from collab_tasks.batch1.error_decomposition import decompose  # noqa: E402

MASS_R = 2.0


def main():
    rows = []
    for s in SCENES:
        data = load_scene(s)
        cats, _, pred = decompose(data)
        F, names, cell = default_fields(data, 0)
        F = np.asarray(F)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        idx = {c: i for i, c in enumerate(names)}
        capped = cap_per_class(list(data["obs"]), 400, seed=SEEDS[0][2])
        obs = {}
        for o in capped:
            obs.setdefault(o["cls"], []).append((o["x"], o["y"]))
        obs = {c: np.array(v) for c, v in obs.items()}
        # per-scene field scale, so heights are comparable across scenes
        fmax = float(np.abs(F).max())
        for i in sorted(cats["local_loss"]):
            g, w = gt[i], pred[i]
            if g not in idx or w not in idx:
                continue
            px, py = float(xyz[i, a]), float(xyz[i, b])
            dgv = np.hypot(*(obs[g] - [px, py]).T)
            dwv = np.hypot(*(obs[w] - [px, py]).T)
            if dgv.min() >= dwv.min():
                continue                     # 62% branch, not our subject
            fg = float(F[idx[g], cell[i]])
            fw = float(F[idx[w], cell[i]])
            ng = int((dgv <= LX).sum())
            nw = int((dwv <= LX).sum())
            rows.append(dict(scene=s, gt=g, win=w,
                             f_gt=fg / fmax, f_win=fw / fmax,
                             d_gt=float(dgv.min()), d_win=float(dwv.min()),
                             n_gt=ng, n_win=nw))
        print(f"{s} done", flush=True)

    n = len(rows)
    inter = [r for r in rows if r["f_gt"] <= 0]
    rest = [r for r in rows if r["f_gt"] > 0]
    mass = [r for r in rest if r["n_win"] >= MASS_R * max(r["n_gt"], 1)]
    flat = [r for r in rest if r not in mass]

    print(f"\n{n} cells where the GT class's own observation is NEARER "
          f"and we still lose")
    print("=" * 72)
    for lab, S in (("INTERFERENCE  f_gt <= 0", inter),
                   ("LOCAL MASS    winner >=2x obs in one lambda", mass),
                   ("FLAT KERNEL   close fields, similar mass", flat)):
        if not S:
            continue
        mg = np.median([r["f_gt"] for r in S])
        mw = np.median([r["f_win"] for r in S])
        dg = np.median([r["d_gt"] for r in S])
        dw = np.median([r["d_win"] for r in S])
        print(f"{lab:<44}{100*len(S)/n:>5.1f}%  ({len(S)} cells)")
        print(f"     median f_gt {mg:+.4f}   f_win {mw:+.4f}   "
              f"d_gt {dg:.3f} m   d_win {dw:.3f} m")

    print("\nHow badly do we lose, when we lose? (winner/GT field ratio)")
    pos = [r for r in rows if r["f_gt"] > 0]
    ratio = np.array([r["f_win"] / r["f_gt"] for r in pos])
    for q in (50, 75, 90):
        print(f"  p{q}: {np.percentile(ratio, q):.2f}x")
    print(f"  cells where the winner's field is >5x ours: "
          f"{100*(ratio > 5).mean():.1f}% of the positive-field cells")
    print(f"  -> these are NOT near-ties; a threshold or abstain rule cannot "
          f"reach them")

    from collections import Counter
    print("\nsignature by dominant pair:")
    tag = {id(r): "INTERF" for r in inter}
    tag.update({id(r): "MASS" for r in mass})
    tag.update({id(r): "FLAT" for r in flat})
    for (g, w), c in Counter((r["gt"], r["win"]) for r in rows).most_common(6):
        S = [r for r in rows if r["gt"] == g and r["win"] == w]
        cnt = Counter(tag[id(r)] for r in S)
        share = "  ".join(f"{k} {100*v/len(S):.0f}%"
                          for k, v in cnt.most_common())
        print(f"  {g+' -> '+w:<30}{c:>5} cells   {share}")

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(
        n=n,
        interference=dict(n=len(inter), pct=100 * len(inter) / n,
                          med_f_gt=float(np.median([r["f_gt"] for r in inter])
                                         ) if inter else None),
        local_mass=dict(n=len(mass), pct=100 * len(mass) / n),
        flat_kernel=dict(n=len(flat), pct=100 * len(flat) / n),
        ratio_p50=float(np.percentile(ratio, 50)),
        ratio_p90=float(np.percentile(ratio, 90)),
        pct_over_5x=float(100 * (ratio > 5).mean()),
        cells=rows),
        open("outputs/batch1/field_why.json", "w"))
    print("\nwrote outputs/batch1/field_why.json")


if __name__ == "__main__":
    main()
