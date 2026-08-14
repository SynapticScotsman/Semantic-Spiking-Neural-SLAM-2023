"""Per-class accuracy, to test whether our losses are concentrated in rare classes.

The cap sweep on room0_cgfront (their frontend, our trace) showed F-mIoU rising
0.240 -> 0.351 with more observations while mAcc barely moved (0.178 -> 0.187).
Those two metrics weight classes differently — F-mIoU by frequency, mAcc
unweighted — so that split says the gains landed on common classes and the rare
ones did not improve. This script checks that directly.

Prediction being tested (state it before reading the table): if bundle
interference is the cause, per-class accuracy should correlate with how many
observations that class contributed — rare classes near zero, common classes
fine. If accuracy is flat across support, interference is the wrong story and
the 2D floor-plane decode becomes the leading explanation.

    python collab_tasks/scripts/per_class_breakdown.py --scene room0_cgfront \
        --compare-scene room0

--compare-scene adds ConceptGraphs' own per-class accuracy from that scene's
cg_labels.npz, transferred onto the eval points with THEIR scorer's own
nearest-neighbour rule (imported from 05_score.py, not reimplemented).
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

PKG = "student_gpu_package"


def load_scorer():
    """05_score.py starts with a digit, so it cannot be imported by name.
    Reuse its transfer() and exclusion list rather than writing our own —
    a second implementation of the metric is exactly how numbers drift."""
    path = os.path.join(PKG, "05_score.py")
    spec = importlib.util.spec_from_file_location("score05", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path]                      # its argparse runs only under main
    spec.loader.exec_module(mod)
    return mod


def per_class_acc(gt, pred):
    """Recall per GT class — the quantity mAcc averages."""
    out = {}
    for c in sorted(set(gt)):
        m = gt == c
        out[c] = (float((pred[m] == c).mean()), int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room0_cgfront")
    ap.add_argument("--compare-scene", default=None,
                    help="scene whose cg_labels.npz holds ConceptGraphs' labels")
    args = ap.parse_args()

    d = f"{PKG}/handoff/{args.scene}"
    E = np.load(f"{d}/eval_points.npz", allow_pickle=True)
    exyz, gt = E["xyz"], E["gt_class"].astype(str)
    Z = np.load(f"{d}/vsa_labels.npz", allow_pickle=True)
    ours = Z["pred_class"].astype(str)
    if len(ours) != len(gt):
        raise SystemExit(f"label/GT length mismatch: {len(ours)} vs {len(gt)}")

    obs = json.load(open(f"outputs/replica_{args.scene}/object_points.json"))["points"]
    n_obs = collections.Counter(p["cls"] for p in obs)

    mod = load_scorer()
    excluded = set(getattr(mod, "CG_EXCLUDE_6", ()))

    theirs = None
    if args.compare_scene:
        p = f"{PKG}/handoff/{args.compare_scene}/cg_labels.npz"
        if os.path.exists(p):
            C = np.load(p, allow_pickle=True)
            lab = mod.transfer(C["xyz"], C["pred_class"].astype(object), exyz)
            theirs = np.array([v if v is not None else "__none__" for v in lab])
        else:
            print(f"(no {p} — skipping their column)")

    ours_acc = per_class_acc(gt, ours)
    their_acc = per_class_acc(gt, theirs) if theirs is not None else {}

    print(f"\n{'class':<16}{'GT pts':>8}{'our obs':>9}{'ours':>8}"
          f"{'theirs':>9}   note")
    print("-" * 62)
    rows = sorted(ours_acc.items(), key=lambda kv: -kv[1][1])
    for c, (acc, sup) in rows:
        t = their_acc.get(c, (float('nan'), 0))[0] if their_acc else float('nan')
        note = "excluded" if c in excluded else ("NO OBS" if n_obs[c] == 0 else "")
        ts = f"{t:8.3f}" if t == t else "       -"
        print(f"{c:<16}{sup:>8}{n_obs[c]:>9}{acc:>8.3f}{ts}   {note}")

    scored = [(c, a, s) for c, (a, s) in ours_acc.items() if c not in excluded]
    if scored:
        sup_sorted = sorted(scored, key=lambda r: r[2])
        half = len(sup_sorted) // 2
        lo = [a for _, a, _ in sup_sorted[:half]]
        hi = [a for _, a, _ in sup_sorted[half:]]
        print(f"\nscored classes: {len(scored)} (excluded {len(excluded)})")
        print(f"  rare  half (fewest GT points): mean acc {np.mean(lo):.3f}")
        print(f"  common half                  : mean acc {np.mean(hi):.3f}")
        zero = [c for c, a, _ in scored if a == 0.0]
        print(f"  classes we score EXACTLY zero: {len(zero)}/{len(scored)}"
              + (f" -> {', '.join(zero[:12])}" if zero else ""))
        with_obs = [(a, n_obs[c]) for c, a, _ in scored if n_obs[c] > 0]
        if len(with_obs) > 2:
            a = np.array([x[0] for x in with_obs])
            n = np.array([x[1] for x in with_obs], float)
            r = np.corrcoef(a, np.log10(n + 1))[0, 1]
            print(f"  corr(accuracy, log10 observations) = {r:+.2f}  "
                  f"(positive => rare classes are the ones failing)")


if __name__ == "__main__":
    main()
