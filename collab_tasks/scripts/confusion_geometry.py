"""Are our confusions vertical stacks, or just wrong labels?

The per-class table (room0_cgfront, 2026-08-14) refuted the first version of the
projection story: classes we FAIL average height -0.64 and classes we get right
-0.29, so the failures are LOWER, not higher. A mean-height comparison was the
wrong test — it averages over the whole room instead of asking the question that
matters.

The question that matters: for a confused pair A→B (GT says A, we said B), do A
and B occupy the SAME floor coordinates at DIFFERENT heights? If they do, a 2D
floor-plane field cannot separate them by construction, and a third axis would
recover exactly those pairs. If they do not, the confusion is a labelling
disagreement and no amount of geometry will fix it.

Two mechanisms are visible in that table and this script separates them:

  stacked      indoor-plant -> vase 90%, plant-stand -> vase 96%
               (a potted plant IS stand + pot + plant at one (x,y))
  vocabulary   pot <-> vase 100% both ways, cushion -> pillow, blanket ->
               comforter, basket -> bin — near-synonyms from their CLIP
               assignment, on which THEIR OWN map also scores 0.000

    python collab_tasks/scripts/confusion_geometry.py --scene room0_cgfront

Reports, per confused pair: footprint overlap in (x,y), height separation, and a
verdict. Pure geometry over eval_points.npz — no GPU, no models, runs locally.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PKG = "student_gpu_package"
EXCLUDED = {"other", "floor", "wall", "ceiling", "door", "window"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room0_cgfront")
    ap.add_argument("--cell", type=float, default=0.15,
                    help="floor cell size (m) for the footprint test — two "
                         "classes 'share footprint' if they occupy the same cell")
    ap.add_argument("--min-share", type=float, default=0.20,
                    help="fraction of A's footprint that must be shared with B "
                         "to call the pair co-located")
    ap.add_argument("--min-frac", type=float, default=0.15,
                    help="ignore confusions accounting for less than this share "
                         "of A's GT points")
    args = ap.parse_args()

    d = f"{PKG}/handoff/{args.scene}"
    E = np.load(f"{d}/eval_points.npz", allow_pickle=True)
    xyz, gt = E["xyz"], E["gt_class"].astype(str)
    pred = np.load(f"{d}/vsa_labels.npz", allow_pickle=True)["pred_class"].astype(str)

    var = xyz.var(0)
    a, b = sorted(np.argsort(var)[-2:])
    up = ({0, 1, 2} - {a, b}).pop()
    X, Y, Z = xyz[:, a], xyz[:, b], xyz[:, up]
    print(f"floor axes {'xyz'[a]},{'xyz'[b]}; height {'xyz'[up]}; "
          f"cell {args.cell} m\n")

    def cells(mask):
        return set(zip(np.floor(X[mask] / args.cell).astype(int),
                       np.floor(Y[mask] / args.cell).astype(int)))

    rows = []
    for A in sorted(set(gt)):
        if A in EXCLUDED:
            continue
        mA = gt == A
        if mA.sum() < 10:
            continue
        wrong = collections.Counter(pred[mA][pred[mA] != A])
        for B, n in wrong.most_common(3):
            frac = n / mA.sum()
            if frac < args.min_frac or B in ("__none__",):
                continue
            mB = gt == B
            cA = cells(mA)
            if not cA:
                continue
            if mB.sum() >= 10:
                share = len(cA & cells(mB)) / len(cA)
                dz = float(Z[mB].mean() - Z[mA].mean())
                bz = f"{Z[mB].mean():+.2f}"
            else:
                # B has no GT footprint of its own — a label that exists in our
                # output but barely in the ground truth
                share, dz, bz = float("nan"), float("nan"), "  n/a"
            rows.append((A, B, frac, int(mA.sum()), share, dz,
                         float(Z[mA].mean()), bz))

    print(f"{'GT class':<15}{'we said':<15}{'share':>7}{'A pts':>7}"
          f"{'foot∩':>7}{'A z':>7}{'B z':>7}{'Δz':>7}   verdict")
    print("-" * 100)
    stacked = vocab = unclear = 0
    for A, B, frac, nA, share, dz, az, bz in sorted(rows, key=lambda r: -r[2]):
        if share != share:                      # NaN — B has no GT presence
            verdict = "B absent from GT — vocabulary"; vocab += 1
        elif share >= args.min_share and abs(dz) >= 0.20:
            verdict = "STACKED — a height axis would separate"; stacked += 1
        elif share >= args.min_share:
            verdict = "co-located, same height — vocabulary"; vocab += 1
        else:
            verdict = "different places — neither"; unclear += 1
        sh = "    n/a" if share != share else f"{share:7.2f}"
        dzs = "    n/a" if dz != dz else f"{dz:+7.2f}"
        print(f"{A:<15}{B:<15}{frac:>7.0%}{nA:>7}{sh}{az:>7.2f}{bz:>7}{dzs}   {verdict}")

    tot = stacked + vocab + unclear
    print(f"\n{tot} confusions above {args.min_frac:.0%} of a class's GT points")
    print(f"  stacked (height would separate) : {stacked}")
    print(f"  vocabulary / label disagreement : {vocab}")
    print(f"  neither                         : {unclear}")
    print("\nA third encoded axis can only recover the STACKED rows. Vocabulary "
          "rows are\ntheir CLIP assignment disagreeing with Replica's granularity "
          "— and their own\nmap scores 0.000 on those classes too, so those are "
          "the benchmark's, not ours.")


if __name__ == "__main__":
    main()
