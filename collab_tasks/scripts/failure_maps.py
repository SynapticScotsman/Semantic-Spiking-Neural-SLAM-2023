"""Show WHERE and HOW our labelling fails, not just that it does.

per_class_breakdown said 18 of 24 scored classes come out at exactly 0.000,
including 'vent' with 2511 observations — more than any other class. Aggregate
numbers cannot distinguish "we never saw it" from "we saw it and something else
won that spot". This answers that, two ways:

  1. a confusion table  — at the GT points of a failing class, what did we say
     instead? If it is consistently the object occupying the same floor area,
     the 2D projection is the culprit rather than the memory.
  2. floor-plan figures — GT points of the class, our observations of it, and
     our prediction over the room, so the overlap is visible rather than
     inferred.

    python collab_tasks/scripts/failure_maps.py --scene room0_cgfront \
        --out outputs/replica_room0_cgfront/failure_maps

Writes one overview PNG (GT map vs ours) plus one PNG per failing class.
"""
from __future__ import annotations

import argparse
import collections
import json
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
    ap.add_argument("--compare-scene", default=None,
                    help="scene whose cg_labels.npz holds ConceptGraphs' labels; "
                         "adds a third panel so their map can be seen beside "
                         "ours on the same points, not just compared as a number")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=6,
                    help="how many failing classes to draw, worst-supported first")
    ap.add_argument("--min-support", type=int, default=10,
                    help="ignore classes with fewer GT points than this")
    args = ap.parse_args()
    out = args.out or f"outputs/replica_{args.scene}/failure_maps"
    os.makedirs(out, exist_ok=True)

    d = f"{PKG}/handoff/{args.scene}"
    E = np.load(f"{d}/eval_points.npz", allow_pickle=True)
    xyz, gt = E["xyz"], E["gt_class"].astype(str)
    pred = np.load(f"{d}/vsa_labels.npz", allow_pickle=True)["pred_class"].astype(str)

    # floor axes: the two largest-variance axes, same rule as everywhere else
    var = xyz.var(0)
    a, b = sorted(np.argsort(var)[-2:])
    up = ({0, 1, 2} - {a, b}).pop()
    X, Y, Z = xyz[:, a], xyz[:, b], xyz[:, up]
    print(f"floor axes {'xyz'[a]},{'xyz'[b]}; height axis {'xyz'[up]}")

    obs = json.load(open(f"outputs/replica_{args.scene}/object_points.json"))["points"]
    n_obs = collections.Counter(p["cls"] for p in obs)

    # ConceptGraphs' own labels, transferred onto the SAME eval points with
    # their scorer's own nearest-neighbour rule, so the two maps are drawn from
    # identical geometry and any visible difference is labelling, not sampling.
    theirs = None
    if args.compare_scene:
        p = f"{PKG}/handoff/{args.compare_scene}/cg_labels.npz"
        if os.path.exists(p):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "score05", os.path.join(PKG, "05_score.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.argv = [os.path.join(PKG, "05_score.py")]
            spec.loader.exec_module(mod)
            C = np.load(p, allow_pickle=True)
            lab = mod.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
            theirs = np.array([v if v is not None else "__none__" for v in lab])
            print(f"loaded their labels from {p}")
        else:
            print(f"(no {p} — their panel will be skipped)")

    # which classes fail despite having observations — the interesting ones
    rows = []
    for c in sorted(set(gt)):
        if c in EXCLUDED:
            continue
        m = gt == c
        if m.sum() < args.min_support:
            continue
        acc = float((pred[m] == c).mean())
        rows.append((c, acc, int(m.sum()), n_obs[c], float(Z[m].mean())))

    print(f"\n{'class':<16}{'acc':>7}{'GT pts':>8}{'our obs':>9}"
          f"{'mean height':>13}   what we said there instead")
    print("-" * 96)
    failing = []
    for c, acc, sup, nob, h in sorted(rows, key=lambda r: (r[1], -r[3])):
        m = gt == c
        wrong = collections.Counter(pred[m][pred[m] != c])
        top = ", ".join(f"{k} {v/max(m.sum(),1):.0%}" for k, v in wrong.most_common(3))
        print(f"{c:<16}{acc:>7.3f}{sup:>8}{nob:>9}{h:>13.2f}   {top}")
        if acc < 0.05 and nob > 0:
            failing.append(c)

    # height check: are the failing classes systematically off the floor?
    if failing:
        hf = np.mean([Z[gt == c].mean() for c in failing])
        ok = [c for c, acc, s, n, h in rows if acc > 0.5]
        ho = np.mean([Z[gt == c].mean() for c in ok]) if ok else float("nan")
        print(f"\nmean height of FAILING classes with observations: {hf:+.2f}")
        print(f"mean height of classes we get right (>0.5)       : {ho:+.2f}")
        print("If those differ, the 2D floor-plane projection is implicated: two "
              "objects at the same (x,y) but different heights are one cell to us.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(matplotlib unavailable: {e} — tables only, no figures)")
        return

    # overview: GT, ours, and (when available) theirs — same points, same colours,
    # so the panels are directly comparable
    panels = [(gt, "ground truth"), (pred, f"ours — 32 KB trace")]
    if theirs is not None:
        panels.append((theirs, "ConceptGraphs — point-cloud map"))
    names = sorted(set(gt) | set(pred) | (set(theirs) if theirs is not None else set()))
    cid = {c: i for i, c in enumerate(names)}
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.5),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (lab, ttl) in zip(axes, panels):
        acc = "" if ttl.startswith("ground") else f"  (mAcc-relevant agreement {np.mean(lab == gt):.1%})"
        ax.scatter(X, Y, c=[cid[v] for v in lab], s=1, cmap="tab20", linewidths=0)
        ax.set_title(ttl + acc, fontsize=10)
        ax.set_aspect("equal"); ax.set_xlabel("xyz"[a])
    axes[0].set_ylabel("xyz"[b])
    fig.tight_layout()
    fig.savefig(f"{out}/overview_gt_ours_theirs.png", dpi=130)
    plt.close(fig)

    # per failing class: where its GT is, where our observations are, what we said
    for c in failing[:args.top]:
        m = gt == c
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.scatter(X, Y, c="0.88", s=1, linewidths=0, label="all eval points")
        ox = [p["x"] for p in obs if p["cls"] == c]
        oy = [p["y"] for p in obs if p["cls"] == c]
        if ox:
            ax.scatter(ox, oy, c="tab:blue", s=6, alpha=.5,
                       label=f"our observations of {c} ({len(ox)})")
        ax.scatter(X[m], Y[m], c="tab:red", s=8,
                   label=f"GT {c} ({int(m.sum())} pts)")
        said = collections.Counter(pred[m]).most_common(1)
        ttl = (f"{c}: ours {float((pred[m]==c).mean()):.3f} — "
               f"we mostly said '{said[0][0]}' there")
        if theirs is not None:
            t_said = collections.Counter(theirs[m]).most_common(1)
            ttl += (f"\nConceptGraphs {float((theirs[m]==c).mean()):.3f} "
                    f"(they said '{t_said[0][0]}')")
        ax.set_title(ttl + f"\nmean height {Z[m].mean():+.2f} vs room mean {Z.mean():+.2f}",
                     fontsize=9)
        ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{out}/fail_{c.replace(' ','_')}.png", dpi=130)
        plt.close(fig)

    print(f"\nwrote {len(os.listdir(out))} figures to {out}")


if __name__ == "__main__":
    main()
