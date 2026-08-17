"""Are our 'local losses' actually GAP, or shared failures of the input?

A category error has been sitting in every writeup and this script settles it.

  error_decomposition.decompose() classifies EVERY cell where OUR prediction
  is wrong (14,134 local_loss cells over 8 scenes). It never asks whether
  ConceptGraphs is right there.

  gap_anatomy.py measured the 8,386 cells where THEY are right and WE are
  wrong -- the actual gap.

Those are different populations, and the wiki has been quoting local_loss
shares ("65% of the gap") as if they were the second. This script measures
the overlap directly, using their own transferred labels and their own
scorer, so every downstream claim can be restated on the right denominator.

Definitions, all on eval points with CG_EXCLUDE_6 applied:
  ours   = predict(default_fields(data, 0))          our trace, reference draw
  theirs = _SC.transfer(cg_labels.npz -> eval points)  their map, as in
                                                       table2_full.py:56-59
  GAP cell        : theirs == gt AND ours != gt   (their win, our loss)
  SHARED cell     : theirs != gt AND ours != gt   (neither system gets it)

Then split by the proximity branch from local_loss_forensics:
  W = winner's nearest observation closer than the GT class's  (the '62%')
  G = GT class's nearest observation closer                    (the '38%')

    python collab_tasks/batch1/shared_ceiling.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, SEEDS, _SC, cap_per_class, default_fields, load_scene, predict,
    score)
from collab_tasks.batch1.error_decomposition import decompose  # noqa: E402


def main():
    tot = dict(local_loss=0, W=0, G=0,
               gap_W=0, shared_W=0, gap_G=0, shared_G=0)
    per_scene, headroom = {}, {}
    for s in SCENES:
        data = load_scene(s)
        cats, _, ours = decompose(data)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]

        C = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                    allow_pickle=True)
        lab = _SC.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
        theirs = np.array([v if v is not None else "__none__" for v in lab])

        capped = cap_per_class(list(data["obs"]), 400, seed=SEEDS[0][2])
        obs = {}
        for o in capped:
            obs.setdefault(o["cls"], []).append((o["x"], o["y"]))
        obs = {c: np.array(v) for c, v in obs.items()}

        c = dict(local_loss=0, W=0, G=0,
                 gap_W=0, shared_W=0, gap_G=0, shared_G=0)
        for i in sorted(cats["local_loss"]):
            g, w = gt[i], ours[i]
            if g not in obs or w not in obs:
                continue
            px, py = float(xyz[i, a]), float(xyz[i, b])
            dg = float(np.hypot(*(obs[g] - [px, py]).T).min())
            dw = float(np.hypot(*(obs[w] - [px, py]).T).min())
            br = "G" if dg < dw else "W"
            c["local_loss"] += 1
            c[br] += 1
            key = ("gap_" if theirs[i] == g else "shared_") + br
            c[key] += 1
        for k in c:
            tot[k] += c[k]
        per_scene[s] = c

        # honest headroom: what mAcc would we reach if we fixed only the
        # cells THEY get right (the reachable ceiling), vs fixing everything
        base = score(gt, ours)["macc"]
        p_gap = ours.copy()
        m = theirs == gt
        p_gap[m] = gt[m]
        p_all = gt.copy()
        headroom[s] = dict(base=base,
                           fix_their_right=score(gt, p_gap)["macc"],
                           fix_everything=score(gt, p_all)["macc"])
        print(f"{s} done", flush=True)

    n = tot["local_loss"]
    print(f"\n{n} local_loss cells (cells where OUR prediction is wrong)")
    print("=" * 74)
    print(f"{'branch':<10}{'cells':>8}{'GAP (they are right)':>24}"
          f"{'SHARED (both wrong)':>22}")
    print("-" * 74)
    for br, lab in (("W", "winner nearer"), ("G", "GT nearer")):
        tb = tot[br]
        gp, sh = tot[f"gap_{br}"], tot[f"shared_{br}"]
        print(f"{lab:<10}{tb:>8}{gp:>14} ({100*gp/tb:>4.1f}%)"
              f"{sh:>13} ({100*sh/tb:>4.1f}%)")
    print("-" * 74)
    gap = tot["gap_W"] + tot["gap_G"]
    shared = tot["shared_W"] + tot["shared_G"]
    print(f"{'TOTAL':<10}{n:>8}{gap:>14} ({100*gap/n:>4.1f}%)"
          f"{shared:>13} ({100*shared/n:>4.1f}%)")

    print(f"\nTHE CORRECTION: only {100*gap/n:.1f}% of our 'local losses' are "
          f"cells ConceptGraphs actually wins.")
    print(f"The other {100*shared/n:.1f}% are cells NEITHER system gets right "
          f"-- a shared input ceiling,\nnot a decode deficit. Within the "
          f"winner-nearer branch specifically, "
          f"{100*tot['shared_W']/tot['W']:.1f}%\nare shared failures.")

    print("\nHONEST HEADROOM (mean over 8 scenes)")
    b = np.mean([h["base"] for h in headroom.values()])
    r = np.mean([h["fix_their_right"] for h in headroom.values()])
    e = np.mean([h["fix_everything"] for h in headroom.values()])
    print(f"  our trace now                        {b:.4f}")
    print(f"  fixing every cell THEY get right     {r:.4f}   "
          f"(+{r-b:.4f} <- the reachable ceiling)")
    print(f"  fixing every cell (perfect decode)   {e:.4f}   (+{e-b:.4f})")
    print(f"  => backend headroom against them is +{r-b:.4f} mAcc, and the "
          f"rest of the\n     distance to 1.0 is frontend/GT-coverage bound "
          f"for BOTH systems.")

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(totals=tot, per_scene=per_scene, headroom=headroom,
                   summary=dict(gap_pct=100 * gap / n,
                                shared_pct=100 * shared / n,
                                shared_pct_in_W=100 * tot["shared_W"] / tot["W"],
                                base=float(b), reachable=float(r),
                                perfect=float(e))),
              open("outputs/batch1/shared_ceiling.json", "w"), indent=1)
    print("\nwrote outputs/batch1/shared_ceiling.json")


if __name__ == "__main__":
    main()
