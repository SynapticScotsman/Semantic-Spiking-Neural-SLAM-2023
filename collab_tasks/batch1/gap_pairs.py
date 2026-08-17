"""The gap under the microscope: which (GT -> our answer) pairs carry each
category, and a representative real cell per category for visualisation.

Two outputs:
  1. printed cross-tab of gap cells (theirs right, ours wrong) by category and
     class pair -- answers "is local_loss a few furniture pairs or diffuse?"
  2. outputs/batch1/gap_pairs.json -- top pairs per category plus, for the #1
     pair of each, one representative cell with the actual observation
     positions of both classes (subsampled), so the categories can be DRAWN
     from real data rather than described.

Reference seed tuple; diagnosis, not a screened result.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    CG_EXCLUDE_6, SCENES, _SC, default_fields, load_scene, predict)
from collab_tasks.batch1.gap_anatomy import classify  # noqa: E402

CATS = ("near_tie", "bleed", "local_loss")   # the three that exist in the gap


def main():
    per_cat = {c: collections.Counter() for c in CATS}
    exemplars = {}
    for s in SCENES:
        data = load_scene(s)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        F, names, cell = default_fields(data, 0)
        ours = predict(F, names, cell)
        C = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                    allow_pickle=True)
        lab = _SC.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
        theirs = np.array([v if v is not None else "__none__" for v in lab])
        idx = {c: i for i, c in enumerate(names)}
        obs_xy = {}
        for o in data["obs"]:
            obs_xy.setdefault(o["cls"], []).append((o["x"], o["y"]))
        obs_xy = {c: np.array(v) for c, v in obs_xy.items()}
        scored = ~np.isin(gt, list(CG_EXCLUDE_6))
        gap = np.flatnonzero(scored & (theirs == gt) & (ours != gt))
        for i in gap:
            cat = classify(int(i), gt, ours, F, names, cell, xyz, a, b,
                           obs_xy, idx)
            if cat in per_cat:
                per_cat[cat][(s, gt[i], ours[i])] += 1

    print("top (scene, GT -> we said) pairs per category:")
    top_share = {}
    for cat in CATS:
        tot = sum(per_cat[cat].values())
        print(f"\n{cat}  ({tot} cells across all scenes)")
        top10 = per_cat[cat].most_common(10)
        for (s, g, w), n in top10:
            print(f"  {s:<9} {g:<15} -> {w:<15} {n:>5}  ({100*n/tot:4.1f}%)")
        top_share[cat] = sum(n for _, n in top10) / max(tot, 1)
        print(f"  top-10 pairs carry {100*top_share[cat]:.0f}% of {cat}")

    # one representative real cell per category, from its #1 pair, for drawing
    for cat in CATS:
        (s, g, w), _ = per_cat[cat].most_common(1)[0]
        data = load_scene(s)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        F, names, cell = default_fields(data, 0)
        ours = predict(F, names, cell)
        C = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                    allow_pickle=True)
        lab = _SC.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
        theirs = np.array([v if v is not None else "__none__" for v in lab])
        idx = {c: i for i, c in enumerate(names)}
        obs_xy = {}
        for o in data["obs"]:
            obs_xy.setdefault(o["cls"], []).append((o["x"], o["y"]))
        obs_xy = {c: np.array(v) for c, v in obs_xy.items()}
        scored = ~np.isin(gt, list(CG_EXCLUDE_6))
        cand = [int(i) for i in np.flatnonzero(
            scored & (theirs == gt) & (ours != gt)
            & (gt == g) & (ours == w))
            if classify(int(i), gt, ours, F, names, cell, xyz, a, b,
                        obs_xy, idx) == cat]
        pts = xyz[cand][:, [a, b]]
        ctr = pts.mean(0)
        i = cand[int(np.argmin(((pts - ctr) ** 2).sum(1)))]
        rng = np.random.default_rng(0)

        def sub(arr, k=140):
            if len(arr) <= k:
                return arr
            return arr[rng.choice(len(arr), k, replace=False)]

        exemplars[cat] = dict(
            scene=s, gt=g, said=w,
            cell_xy=[round(float(xyz[i, a]), 3), round(float(xyz[i, b]), 3)],
            f_gt=round(float(F[idx[g], cell[i]]), 4),
            f_said=round(float(F[idx[w], cell[i]]), 4),
            gt_obs=[[round(float(x), 2), round(float(y), 2)]
                    for x, y in sub(obs_xy[g])],
            said_obs=[[round(float(x), 2), round(float(y), 2)]
                      for x, y in sub(obs_xy.get(w, np.empty((0, 2))))],
            room_x=[round(float(xyz[:, a].min()), 2),
                    round(float(xyz[:, a].max()), 2)],
            room_y=[round(float(xyz[:, b].min()), 2),
                    round(float(xyz[:, b].max()), 2)])
        print(f"\nexemplar {cat}: {s} cell {exemplars[cat]['cell_xy']} "
              f"GT {g} (field {exemplars[cat]['f_gt']}) vs said {w} "
              f"(field {exemplars[cat]['f_said']})")

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(
        # Counter keys contain numpy str_ and counts are ints; coerce every
        # element to a builtin explicitly -- np scalar VALUES are what break
        # json.dump, and the crash ate the file write on the first run.
        pairs={cat: [[[str(x) for x in k], int(n)]
                     for k, n in per_cat[cat].most_common(12)]
               for cat in CATS},
        top10_share={c: round(top_share[c], 3) for c in CATS},
        exemplars=exemplars),
        open("outputs/batch1/gap_pairs.json", "w"), indent=1)
    print("\nwrote outputs/batch1/gap_pairs.json")


if __name__ == "__main__":
    main()
