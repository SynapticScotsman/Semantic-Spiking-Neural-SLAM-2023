"""Classify every wrong cell BEFORE running any fix, so mechanism choice is
evidence-ordered rather than intuition-ordered.

Decision tree per wrong eval point (priority order guarantees a partition):

  unreachable  no observation of the GT class exists in this scene at all --
               the shared frontend never produced one. Neither backend can
               ever win these; they bound what ANY backend change can recover.
  near_tie     the GT class's field is within FRAC of the winner's at that
               cell (headline 5%, sensitivity at 2%/10%) -- normalisation and
               threshold territory (h1/h2).
  bleed        the winning class's nearest observation is > BLEED_DIST away:
               it won this cell from a distance (h6/h3/h4 territory).
  misplaced    our observations of the GT class exist but their nearest is
               > BLEED_DIST from the cell -- geometry/export, out of batch
               scope.
  local_loss   none of the above: lost a local, legitimate competition
               (h2/h3 territory).

Runs at the REFERENCE seed tuple only -- this is diagnosis of the measured
baseline, not a screened result, and the guard does not apply to it.

    python collab_tasks/batch1/error_decomposition.py
    python collab_tasks/batch1/error_decomposition.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    CG_EXCLUDE_6, SCENES, default_fields, load_scene, make_blob_data,
    predict, score)

BLEED_DIST = 0.9          # metres; 2 x the lambda_x kernel scale
TIE_FRACS = (0.02, 0.05, 0.10)
CATS = ("unreachable", "near_tie", "bleed", "misplaced", "local_loss")


def _nearest_dist(px, py, pts):
    """Min distance from (px, py) to a 2D point set. Pure numpy."""
    if len(pts) == 0:
        return np.inf
    d2 = (pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2
    return float(np.sqrt(d2.min()))


def decompose(data, frac=0.05):
    F, names, cell = default_fields(data)
    gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
    pred = predict(F, names, cell)
    idx = {c: i for i, c in enumerate(names)}
    obs_xy = {}
    for o in data["obs"]:
        obs_xy.setdefault(o["cls"], []).append((o["x"], o["y"]))
    obs_xy = {c: np.array(v) for c, v in obs_xy.items()}

    excl = set(CG_EXCLUDE_6)
    cats = {c: set() for c in CATS}
    for i in np.flatnonzero(pred != gt):
        g = gt[i]
        if g in excl:
            continue
        px, py = float(xyz[i, a]), float(xyz[i, b])
        w = pred[i]
        if g not in idx or g not in obs_xy:
            cats["unreachable"].add(int(i))
        else:
            fg = F[idx[g], cell[i]]
            fw = F[idx[w], cell[i]] if w in idx else np.inf
            if fg >= fw - frac * abs(fw):
                cats["near_tie"].add(int(i))
            elif _nearest_dist(px, py, obs_xy.get(w, np.empty((0, 2)))) \
                    > BLEED_DIST:
                cats["bleed"].add(int(i))
            elif _nearest_dist(px, py, obs_xy[g]) > BLEED_DIST:
                cats["misplaced"].add(int(i))
            else:
                cats["local_loss"].add(int(i))

    # upper bound: mAcc if every cell in that category were flipped correct
    bounds = {}
    for c, members in cats.items():
        p2 = pred.copy()
        m = np.array(sorted(members), int)
        if len(m):
            p2[m] = gt[m]
        bounds[c] = score(gt, p2)["macc"]
    return cats, bounds, pred


def main():
    out = {}
    print(f"{'scene':<9}" + "".join(f"{c:>12}" for c in CATS))
    print("-" * (9 + 12 * len(CATS)))
    for s in SCENES:
        data = load_scene(s)
        cats, bounds, pred = decompose(data, frac=0.05)
        base = score(data["gt"], pred)["macc"]
        out[s] = {"counts": {c: len(v) for c, v in cats.items()},
                  "base_macc": base,
                  "recoverable_macc_if_fixed": bounds}
        print(f"{s:<9}" + "".join(f"{len(cats[c]):>12}" for c in CATS),
              flush=True)

    print("\nmAcc upper bound if one category were FULLY fixed "
          "(baseline in col 2)")
    print(f"{'scene':<9}{'base':>7}" + "".join(f"{c:>12}" for c in CATS))
    print("-" * (16 + 12 * len(CATS)))
    for s in SCENES:
        r = out[s]
        print(f"{s:<9}{r['base_macc']:>7.3f}" + "".join(
            f"{r['recoverable_macc_if_fixed'][c]:>12.3f}" for c in CATS))
    mean_bound = {c: float(np.mean(
        [out[s]["recoverable_macc_if_fixed"][c] for s in SCENES]))
        for c in CATS}
    mean_base = float(np.mean([out[s]["base_macc"] for s in SCENES]))
    print(f"{'MEAN':<9}{mean_base:>7.3f}" + "".join(
        f"{mean_bound[c]:>12.3f}" for c in CATS))
    print("\nheadroom per category (mean bound minus mean baseline; theirs "
          "is 0.402):")
    for c in CATS:
        print(f"  {c:<12} {mean_bound[c] - mean_base:+.3f}")

    print("\ntie-frac sensitivity (pooled counts across scenes):")
    for fr in TIE_FRACS:
        pooled = {c: 0 for c in CATS}
        for s in SCENES:
            cats, _, _ = decompose(load_scene(s), frac=fr)
            for c in CATS:
                pooled[c] += len(cats[c])
        print(f"  frac {fr:.2f}: " +
              "  ".join(f"{c}={pooled[c]}" for c in CATS))

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(out, open("outputs/batch1/decomposition.json", "w"), indent=1)
    print("\nmechanism map: near_tie -> h1/h2; bleed -> h6/h3/h4; "
          "misplaced -> geometry (out of scope); unreachable -> frontend "
          "constant; local_loss -> h2/h3")
    print("wrote outputs/batch1/decomposition.json")


def _self_test():
    blob = make_blob_data()
    cats, bounds, _ = decompose(blob, frac=0.05)
    total_wrong = sum(len(v) for v in cats.values())
    union = set().union(*cats.values()) if total_wrong else set()
    assert total_wrong == len(union), "categories overlap"
    print(f"partition OK ({total_wrong} wrong cells, all disjoint)")
    print("SELF-TEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        main()
