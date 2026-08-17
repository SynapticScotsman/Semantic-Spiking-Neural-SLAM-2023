"""WHY we lose local competitions: whose observation point is nearer?

The gap anatomy says 65% of the cells where we are wrong are `local_loss`:
both the GT class and our winning class have observations within 0.9 m, and
we lose a local competition. This script asks the question behind that label,
per cell:

  d_gt  = distance from the cell to the NEAREST observation of the GT class
  d_win = distance from the cell to the NEAREST observation of the class our
          field actually answered

If d_gt < d_win, the nearest-point evidence favours the RIGHT class -- a
nearest-neighbour decoder (theirs) answers correctly by construction, and our
field must have been pulled the other way by something other than proximity.
The candidate puller is GLOBAL MASS: every observation of the winner class in
the whole scene contributes to its field height through the kernel. We
measure that too:

  m_win/m_gt = ratio of observation counts within radius r of the cell
               (r = 0.45 = lambda_x, and r = 0.9 = 2*lambda_x)
  M_win/M_gt = ratio of TOTAL scene observation counts (global mass)

Everything from the reference draw's cached fields; the decomposition is
error_decomposition.decompose verbatim.

    python collab_tasks/batch1/local_loss_forensics.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import SCENES, load_scene  # noqa: E402
from collab_tasks.batch1.error_decomposition import decompose  # noqa: E402


def main():
    from vsa_cognitive_mapping.object_grounding import cap_per_class
    rows = []
    for s in SCENES:
        data = load_scene(s)
        cats, _, pred = decompose(data)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        # measure against what the TRACE contains: the capped stream at the
        # reference draw (CAP=400, cap seed 11 = SEEDS[0][2]), not the raw one
        capped = cap_per_class(list(data["obs"]), 400, seed=11)
        obs_xy, tot = {}, {}
        for o in capped:
            obs_xy.setdefault(o["cls"], []).append((o["x"], o["y"]))
        obs_xy = {c: np.array(v) for c, v in obs_xy.items()}
        tot = {c: len(v) for c, v in obs_xy.items()}
        spread = {c: float(v.std(0).mean()) for c, v in obs_xy.items()}
        for i in sorted(cats["local_loss"]):
            g, w = gt[i], pred[i]
            px, py = float(xyz[i, a]), float(xyz[i, b])
            dg = np.hypot(*(obs_xy[g] - [px, py]).T)
            dw = np.hypot(*(obs_xy[w] - [px, py]).T)
            rows.append(dict(
                scene=s, gt=g, win=w,
                d_gt=float(dg.min()), d_win=float(dw.min()),
                n_gt_045=int((dg <= 0.45).sum()),
                n_win_045=int((dw <= 0.45).sum()),
                n_gt_09=int((dg <= 0.9).sum()),
                n_win_09=int((dw <= 0.9).sum()),
                N_gt=tot[g], N_win=tot[w],
                sp_gt=spread[g], sp_win=spread[w]))
        print(f"{s}: {len(cats['local_loss'])} local_loss cells", flush=True)

    R = rows
    n = len(R)
    gt_nearer = sum(1 for r in R if r["d_gt"] < r["d_win"])
    win_nearer = n - gt_nearer
    print(f"\n{n} local_loss cells over 8 scenes")
    print(f"GT class's point is NEARER in {gt_nearer} ({100*gt_nearer/n:.1f}%)"
          f" -- their NN decoder gets these right by construction")
    print(f"winner's point is nearer in  {win_nearer} "
          f"({100*win_nearer/n:.1f}%)")

    # among the cells where proximity says GT, what does mass say?
    A = [r for r in R if r["d_gt"] < r["d_win"]]
    def frac(f): return 100 * sum(1 for r in A if f(r)) / max(len(A), 1)
    print(f"\nAmong the {len(A)} cells where the GT point is nearer "
          f"(the cells that indict the decode):")
    print(f"  winner has MORE obs within 0.45 m: "
          f"{frac(lambda r: r['n_win_045'] > r['n_gt_045']):.1f}%")
    print(f"  winner has MORE obs within 0.90 m: "
          f"{frac(lambda r: r['n_win_09'] > r['n_gt_09']):.1f}%")
    print(f"  winner has MORE obs in the WHOLE SCENE: "
          f"{frac(lambda r: r['N_win'] > r['N_gt']):.1f}%")
    med = lambda k: float(np.median([r[k] for r in A]))
    print(f"  median d_gt {med('d_gt'):.2f} m vs d_win {med('d_win'):.2f} m")
    print(f"  median local obs within 0.9 m: gt {med('n_gt_09'):.0f} vs "
          f"winner {med('n_win_09'):.0f}")
    print(f"  median TOTAL capped obs: gt {med('N_gt'):.0f} vs "
          f"winner {med('N_win'):.0f}")
    print(f"  GT class is more SPREAD OUT than the winner: "
          f"{frac(lambda r: r['sp_gt'] > r['sp_win']):.1f}%  "
          f"(median spread gt {med('sp_gt'):.2f} m vs win "
          f"{med('sp_win'):.2f} m)")

    # the confusion pairs that carry it
    from collections import Counter
    pairs = Counter((r["gt"], r["win"]) for r in A)
    print(f"\ntop pairs among GT-nearer local losses:")
    for (g, w), c in pairs.most_common(8):
        print(f"  {g:>14} lost to {w:<14} {c:>5} cells "
              f"({100*c/len(A):.1f}%)")

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(
        n_local_loss=n, gt_nearer=gt_nearer, win_nearer=win_nearer,
        gt_nearer_frac=gt_nearer / n,
        among_gt_nearer=dict(
            n=len(A),
            win_more_045=frac(lambda r: r["n_win_045"] > r["n_gt_045"]),
            win_more_09=frac(lambda r: r["n_win_09"] > r["n_gt_09"]),
            win_more_total=frac(lambda r: r["N_win"] > r["N_gt"]),
            med_d_gt=med("d_gt"), med_d_win=med("d_win"),
            med_n_gt_09=med("n_gt_09"), med_n_win_09=med("n_win_09"),
            med_N_gt=med("N_gt"), med_N_win=med("N_win"),
            gt_more_spread=frac(lambda r: r["sp_gt"] > r["sp_win"]),
            med_sp_gt=med("sp_gt"), med_sp_win=med("sp_win")),
        top_pairs=[dict(gt=g, win=w, n=c)
                   for (g, w), c in pairs.most_common(12)],
        cells=R),
        open("outputs/batch1/local_loss_forensics.json", "w"), indent=1)
    print("\nwrote outputs/batch1/local_loss_forensics.json")


if __name__ == "__main__":
    main()
