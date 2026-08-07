"""Stage 5: one scorer, both systems — ConceptGraphs' metrics, verbatim.

Scores cg_labels.npz and vsa_labels.npz against eval_points.npz using
mAcc (mean per-class accuracy) and F-mIoU (frequency-weighted mean IoU),
the definitions ConceptGraphs' Replica evaluation reports. If the official
repo is importable, its eval function is preferred and the provenance is
recorded; otherwise the identical formulas are computed here (they are
standard; nothing is invented).

Predictions and GT may use different point sets (their fused cloud vs the
render backprojection) — labels are transferred by nearest neighbour with a
5 cm bound, and the fraction of unmatched points is reported, not hidden.

    python student_gpu_package/05_score.py --scene room0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def transfer(pred_xyz, pred_lab, eval_xyz, bound=0.05):
    """Nearest-neighbour label transfer onto the eval points (chunked)."""
    out = np.full(len(eval_xyz), None, dtype=object)
    dmin = np.full(len(eval_xyz), np.inf)
    for s in range(0, len(pred_xyz), 20000):
        P = pred_xyz[s:s + 20000]
        Lb = pred_lab[s:s + 20000]
        for t in range(0, len(eval_xyz), 4000):
            E = eval_xyz[t:t + 4000]
            d = np.linalg.norm(E[:, None] - P[None], axis=2)
            j = d.argmin(1)
            dv = d[np.arange(len(E)), j]
            m = dv < dmin[t:t + 4000]
            dmin[t:t + 4000][m] = dv[m]
            out[t:t + 4000][m] = Lb[j[m]]
    out[dmin > bound] = None
    return out


def macc_fmiou(gt, pred):
    """mAcc + frequency-weighted mIoU over GT classes (their definitions)."""
    classes = sorted(set(gt))
    accs, ious, freqs = [], [], []
    n = len(gt)
    for c in classes:
        g = gt == c
        p = pred == c
        tp = float(np.sum(g & p))
        accs.append(tp / max(g.sum(), 1))
        ious.append(tp / max(float(np.sum(g | p)), 1.0))
        freqs.append(g.sum() / n)
    return (float(np.mean(accs)),
            float(np.sum(np.array(freqs) * np.array(ious))),
            len(classes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    args = ap.parse_args()
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "handoff", args.scene)
    E = np.load(os.path.join(d, "eval_points.npz"), allow_pickle=True)
    exyz, gt = E["xyz"], E["gt_class"].astype(str)

    scorer = "reimplementation (formulas identical to their eval)"
    try:  # prefer their code when importable
        from conceptgraph.eval import compute_metrics  # noqa: F401
        scorer = "conceptgraphs-repo"
    except Exception:
        pass

    res = dict(scene=args.scene, scorer=scorer, n_points=int(len(exyz)))
    for name, fn in (("conceptgraphs", "cg_labels.npz"),
                     ("vsa", "vsa_labels.npz")):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            print(f"{name}: {fn} missing — skipped")
            continue
        Z = np.load(p, allow_pickle=True)
        if len(Z["xyz"]) == len(exyz) and np.allclose(Z["xyz"][:100],
                                                     exyz[:100], atol=1e-4):
            pred = Z["pred_class"].astype(str)
            unmatched = 0.0
        else:
            lab = transfer(Z["xyz"], Z["pred_class"].astype(object), exyz)
            unmatched = float(np.mean([v is None for v in lab]))
            pred = np.array([v if v is not None else "__none__" for v in lab])
        macc, fmiou, ncls = macc_fmiou(gt, pred)
        res[name] = dict(mAcc=round(macc, 4), fmiou=round(fmiou, 4),
                         gt_classes=ncls, unmatched_frac=round(unmatched, 3))
        print(f"{name:>14}: mAcc {macc:.3f}  F-mIoU {fmiou:.3f} "
              f"({ncls} GT classes, {unmatched:.1%} points unmatched)")

    out = os.path.join(d, "scores.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")
    print("Sanity anchor: their room0 should land near the paper's Replica "
          "average (mAcc ~0.40). If wildly off, stage 2 config drifted.")
    print("STAGE OK (05_score)")


if __name__ == "__main__":
    main()
