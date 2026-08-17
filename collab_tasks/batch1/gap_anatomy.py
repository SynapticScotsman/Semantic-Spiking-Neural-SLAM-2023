"""Anatomy of the VSA-vs-ConceptGraphs gap, cell by cell.

The aggregate gap (theirs 0.402, ours 0.324 mAcc) says nothing about WHERE
they beat us. This takes the exact cells where THEIR transferred labels are
right and OURS are wrong -- the gap cells, literally -- and classifies each
with the error-decomposition tree. It also scores the batch-adopted decode
(z-score + tau0.5 abstain) on the same cells, so "how much of the gap did the
batch close, and out of which categories" is measured rather than inferred.

Reference seed tuple only: this is diagnosis against the stored baseline.

    python collab_tasks/batch1/gap_anatomy.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    CG_EXCLUDE_6, SCENES, _SC, default_fields, load_scene, predict, score)
from collab_tasks.batch1.error_decomposition import (  # noqa: E402
    BLEED_DIST, CATS, _nearest_dist)
from collab_tasks.batch1.h1_threshold import _decode  # noqa: E402


def _zscore(F):
    F = np.asarray(F, float)
    return (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)


def classify(i, gt, pred, F, names, cell, xyz, a, b, obs_xy, idx, frac=0.05):
    g, w = gt[i], pred[i]
    if g not in idx or g not in obs_xy:
        return "unreachable"
    fg = F[idx[g], cell[i]]
    fw = F[idx[w], cell[i]] if w in idx else np.inf
    if fg >= fw - frac * abs(fw):
        return "near_tie"
    px, py = float(xyz[i, a]), float(xyz[i, b])
    if _nearest_dist(px, py, obs_xy.get(w, np.empty((0, 2)))) > BLEED_DIST:
        return "bleed"
    if _nearest_dist(px, py, obs_xy[g]) > BLEED_DIST:
        return "misplaced"
    return "local_loss"


def main():
    out = {}
    tot = {c: 0 for c in CATS}
    tot_recovered = {c: 0 for c in CATS}
    print(f"{'scene':<9}{'ours':>7}{'adopt':>7}{'theirs':>8}{'gapcell':>8}"
          + "".join(f"{c[:9]:>10}" for c in CATS) + f"{'recovered':>10}")
    print("-" * (39 + 10 * len(CATS) + 10))
    for s in SCENES:
        data = load_scene(s)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        F, names, cell = default_fields(data, 0)
        ours = predict(F, names, cell)
        adopted = _decode(_zscore(F), names, cell, 0.5, True)
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
        gap_cells = np.flatnonzero(scored & (theirs == gt) & (ours != gt))
        counts = {c: 0 for c in CATS}
        recovered = {c: 0 for c in CATS}
        for i in gap_cells:
            cat = classify(int(i), gt, ours, F, names, cell, xyz, a, b,
                           obs_xy, idx)
            counts[cat] += 1
            if adopted[i] == gt[i]:
                recovered[cat] += 1
        for c in CATS:
            tot[c] += counts[c]
            tot_recovered[c] += recovered[c]

        m_ours = score(gt, ours)["macc"]
        m_adopt = score(gt, adopted)["macc"]
        m_theirs = score(gt, theirs)["macc"]
        out[s] = dict(ours=m_ours, adopted=m_adopt, theirs=m_theirs,
                      gap_cells=int(len(gap_cells)), by_cat=counts,
                      recovered_by_cat=recovered,
                      only_ours=int((scored & (ours == gt)
                                     & (theirs != gt)).sum()),
                      both_wrong=int((scored & (ours != gt)
                                      & (theirs != gt)).sum()))
        print(f"{s:<9}{m_ours:>7.3f}{m_adopt:>7.3f}{m_theirs:>8.3f}"
              f"{len(gap_cells):>8}"
              + "".join(f"{counts[c]:>10}" for c in CATS)
              + f"{sum(recovered.values()):>10}", flush=True)

    n = sum(tot.values())
    print("-" * (39 + 10 * len(CATS) + 10))
    print("share of THEIR-advantage cells per category, and how many the "
          "adopted decode recovers:")
    for c in CATS:
        pct = 100 * tot[c] / max(n, 1)
        rec = 100 * tot_recovered[c] / max(tot[c], 1)
        print(f"  {c:<12} {tot[c]:>6} cells ({pct:4.1f}% of the gap)   "
              f"adopted recovers {tot_recovered[c]:>5} ({rec:4.1f}%)")
    mo = float(np.mean([out[s]["ours"] for s in SCENES]))
    ma = float(np.mean([out[s]["adopted"] for s in SCENES]))
    mt = float(np.mean([out[s]["theirs"] for s in SCENES]))
    print(f"\nreference-tuple means: ours {mo:.3f} -> adopted {ma:.3f}, "
          f"theirs {mt:.3f}")
    print(f"gap closed by the batch at the reference draw: "
          f"{100*(ma-mo)/max(mt-mo,1e-9):.0f}% of {mt-mo:.3f}")
    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(out, open("outputs/batch1/gap_anatomy.json", "w"), indent=1)
    print("wrote outputs/batch1/gap_anatomy.json")


if __name__ == "__main__":
    main()
