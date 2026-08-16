"""Where exactly does ConceptGraphs beat our trace? Class by class, and on the floor.

The aggregate says theirs 0.402, ours 0.324 over 8 Replica scenes. A single gap
number cannot be worked on. This decomposes it two ways:

  1. PER CLASS -- for every scored class: GT support, our recall, their recall,
     and the class's contribution to the mAcc gap. Because mAcc is an UNWEIGHTED
     per-class mean, every class contributes 1/n_classes regardless of size, so
     a tiny class we miss costs exactly as much as a wall we miss. Sorting by
     contribution says what to actually work on.

  2. ON THE FLOOR -- exports downsampled eval points with GT / our / their labels
     so the three maps can be drawn side by side. Numbers say how much is lost;
     only a map says WHERE, and whether the loss is a few coherent regions
     (fixable by geometry) or salt-and-pepper (a labelling problem).

Also separates the three ways a class can score zero, which need different fixes:

    unreachable   no observation of that class exists at all -- the shared
                  frontend never detected it. Neither system can win it. This is
                  39% of the metric (ceiling 0.609) and is a FRONTEND problem.
    contested     we have observations but lose the cells to another class
                  -- a decode/competition problem.
    misplaced     we have observations and win cells, but in the wrong place
                  -- a geometry problem.

    python collab_tasks/scripts/where_they_win.py --scenes room0 room2 office4

CPU only, numpy only, no matplotlib -- it writes JSON for SVG rendering.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

EXCLUDE = {"other", "floor", "wall", "ceiling", "door", "window"}
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path]
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--map-scene", default="room0",
                    help="scene to export floor-plan points for")
    ap.add_argument("--map-points", type=int, default=4000)
    ap.add_argument("--out", default="outputs/where_they_win.json")
    args = ap.parse_args()
    score05 = load_mod("student_gpu_package/05_score.py", "score05")

    allrows, summary = [], {}
    for s in args.scenes:
        d = f"student_gpu_package/handoff/{s}_cgfront"
        ep, vp, cp = (f"{d}/eval_points.npz", f"{d}/vsa_labels.npz",
                      f"{d}/cg_labels.npz")
        op = f"outputs/replica_{s}_cgfront/object_points.json"
        if not all(os.path.exists(p) for p in (ep, vp, cp, op)):
            print(f"{s}: missing artifacts, skipping")
            continue
        E = np.load(ep, allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        ours = np.load(vp, allow_pickle=True)["pred_class"].astype(str)
        C = np.load(cp, allow_pickle=True)
        lab = score05.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
        theirs = np.array([v if v is not None else "__none__" for v in lab])
        obs = json.load(open(op))["points"]
        have = collections.Counter(p["cls"] for p in obs)

        scored = sorted(set(gt) - EXCLUDE)
        n = len(scored)
        rows = []
        for c in scored:
            m = gt == c
            ro = float((ours[m] == c).mean())
            rt = float((theirs[m] == c).mean())
            said = collections.Counter(ours[m][ours[m] != c]).most_common(1)
            kind = ("unreachable" if have[c] == 0 else
                    "we win" if ro >= rt else
                    "contested")
            rows.append(dict(scene=s, cls=c, support=int(m.sum()),
                             obs=int(have[c]), ours=ro, theirs=rt,
                             gap=ro - rt, contrib=(rt - ro) / n, kind=kind,
                             said=said[0][0] if said else "-"))
        allrows += rows
        unreach = [r for r in rows if r["kind"] == "unreachable"]
        summary[s] = dict(n_scored=n, unreachable=len(unreach),
                          ceiling=(n - len(unreach)) / n,
                          ours=float(np.mean([r["ours"] for r in rows])),
                          theirs=float(np.mean([r["theirs"] for r in rows])))

    print(f"{'scene':<9}{'scored':>7}{'unreach':>8}{'ceiling':>9}"
          f"{'theirs':>8}{'ours':>7}")
    print("-" * 50)
    for s, v in summary.items():
        print(f"{s:<9}{v['n_scored']:>7}{v['unreachable']:>8}{v['ceiling']:>9.3f}"
              f"{v['theirs']:>8.3f}{v['ours']:>7.3f}")

    # where the gap actually lives, biggest contributors first
    losing = sorted([r for r in allrows if r["gap"] < -0.01],
                    key=lambda r: r["gap"])
    print(f"\n{'='*84}\nWHERE THE GAP LIVES — classes they win, worst first"
          f"\n{'='*84}")
    print(f"{'scene':<9}{'class':<15}{'GT pts':>7}{'our obs':>8}"
          f"{'ours':>7}{'theirs':>8}{'gap':>8}   we said instead")
    print("-" * 84)
    for r in losing[:22]:
        print(f"{r['scene']:<9}{r['cls']:<15}{r['support']:>7}{r['obs']:>8}"
              f"{r['ours']:>7.3f}{r['theirs']:>8.3f}{r['gap']:>+8.3f}   {r['said']}")

    tot = -sum(r["contrib"] for r in allrows) / max(len(summary), 1)
    top = -sum(r["contrib"] for r in losing[:10]) / max(len(summary), 1)
    print(f"\ntotal mAcc gap {tot:+.3f}; the 10 worst classes account for "
          f"{top:+.3f} of it ({100*top/tot if tot else 0:.0f}%)")

    won = [r for r in allrows if r["gap"] > 0.01]
    print(f"\nclasses WE win: {len(won)} of {len(allrows)} scored "
          f"({100*len(won)/max(len(allrows),1):.0f}%)")
    for r in sorted(won, key=lambda r: -r["gap"])[:8]:
        print(f"  {r['scene']:<9}{r['cls']:<15}ours {r['ours']:.3f}  "
              f"theirs {r['theirs']:.3f}  {r['gap']:+.3f}")

    # floor-plan export for the map figure
    s = args.map_scene
    d = f"student_gpu_package/handoff/{s}_cgfront"
    E = np.load(f"{d}/eval_points.npz", allow_pickle=True)
    xyz, gt = E["xyz"], E["gt_class"].astype(str)
    ours = np.load(f"{d}/vsa_labels.npz", allow_pickle=True)["pred_class"].astype(str)
    C = np.load(f"{d}/cg_labels.npz", allow_pickle=True)
    lab = score05.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
    theirs = np.array([v if v is not None else "__none__" for v in lab])
    v = xyz.var(0)
    a, b = sorted(np.argsort(v)[-2:])
    keep = ~np.isin(gt, list(EXCLUDE))
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(0)
    if len(idx) > args.map_points:
        idx = rng.choice(idx, args.map_points, replace=False)
    names = sorted(set(gt[idx]) | set(ours[idx]) | set(theirs[idx]))
    cid = {c: i for i, c in enumerate(names)}
    mp = dict(scene=s, axes=["xyz"[a], "xyz"[b]], names=names,
              x=[round(float(q), 3) for q in xyz[idx, a]],
              y=[round(float(q), 3) for q in xyz[idx, b]],
              gt=[cid[c] for c in gt[idx]],
              ours=[cid[c] for c in ours[idx]],
              theirs=[cid[c] for c in theirs[idx]])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(summary=summary, rows=allrows, floor=mp),
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out} (includes {len(idx)} floor points for {s})")


if __name__ == "__main__":
    main()
