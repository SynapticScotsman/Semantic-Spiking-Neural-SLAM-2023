"""What the isolevel actually buys: the coverage / precision trade, measured.

Paul, playing with the artifact's isolevel: raising it visibly separates sofa
from cushion, but at high settings only the labelled points survive. Is that
bad?

There are THREE different thresholds in play and they are easy to conflate:

  1. a DRAWING clip  - what the artifact paints. Changes nothing about the
     decision, only what you see.
  2. a PEAK-PICKING threshold - how many maxima you extract when you ask
     "where is the sofa". This is what visibly separates two objects.
  3. a DECODE ABSTAIN threshold - refuse to label a point when no class is
     confident. This DOES change predictions, and it is what h1 screened.

Only (3) moves the benchmark number, so this measures (3) honestly: sweep the
abstain threshold and report coverage, precision, recall and F1 together,
because quoting any one of them alone hides the trade.

Abstention is scored the way the benchmark scores it: an unanswered point is
simply wrong for recall. That is the rule ConceptGraphs' metric applies, and
choosing a kinder rule here would be inventing a metric.

    python collab_tasks/batch1/threshold_coverage.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, SEEDS, _SC, CG_EXCLUDE_6, default_fields, load_scene, score)

TAUS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def main():
    rows = []
    pair_rows = []
    for tau in TAUS:
        macc, prec, f1, cov = [], [], [], []
        for s in SCENES:
            d = load_scene(s)
            F, names, cell = default_fields(d, 0)
            F = np.asarray(F, float)
            # z-score across classes per cell: the standard confidence proxy
            mu, sd = F.mean(0), F.std(0) + 1e-12
            Z = (F - mu) / sd
            best = Z.argmax(0)[cell]
            conf = Z.max(0)[cell]
            pred = np.array([names[w] for w in best], object)
            keep = conf >= tau
            pred_abst = np.where(keep, pred, "__abstain__")
            m = _SC.macc_full(d["gt"], pred_abst, exclude=CG_EXCLUDE_6)
            macc.append(m["macc"])
            prec.append(m["mprec"])
            f1.append(m["mf1"])
            # coverage over scored points only
            ok = ~np.isin(d["gt"], CG_EXCLUDE_6)
            cov.append(float(keep[ok].mean()))
        rows.append(dict(tau=tau, macc=float(np.mean(macc)),
                         mprec=float(np.mean(prec)), mf1=float(np.mean(f1)),
                         coverage=float(np.mean(cov))))
        print(f"  tau {tau:.2f} done", flush=True)

    print("\nABSTAIN THRESHOLD SWEEP (8 scenes, reference draw)")
    print("abstentions count as WRONG for recall, as the benchmark does\n")
    print(f"{'tau':>6}{'coverage':>11}{'recall(mAcc)':>14}{'precision':>11}"
          f"{'F1':>9}")
    print("-" * 51)
    for r in rows:
        print(f"{r['tau']:>6.2f}{100*r['coverage']:>10.0f}%{r['macc']:>14.4f}"
              f"{r['mprec']:>11.4f}{r['mf1']:>9.4f}")
    print("-" * 51)
    b_r = max(rows, key=lambda r: r["macc"])
    b_p = max(rows, key=lambda r: r["mprec"])
    b_f = max(rows, key=lambda r: r["mf1"])
    print(f"best recall    at tau {b_r['tau']:.2f}  ({b_r['macc']:.4f}, "
          f"coverage {100*b_r['coverage']:.0f}%)")
    print(f"best precision at tau {b_p['tau']:.2f}  ({b_p['mprec']:.4f}, "
          f"coverage {100*b_p['coverage']:.0f}%)")
    print(f"best F1        at tau {b_f['tau']:.2f}  ({b_f['mf1']:.4f}, "
          f"coverage {100*b_f['coverage']:.0f}%)")
    print(f"\nConceptGraphs, for reference: mAcc 0.4020, precision 0.2900 "
          f"at 100% coverage")
    print("They never abstain, so any threshold we adopt has to beat them on "
          "a metric\nthat charges us for the points we refuse to answer.")

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(taus=rows), open("outputs/batch1/threshold_coverage.json",
                                    "w"), indent=1)
    print("\nwrote outputs/batch1/threshold_coverage.json")


if __name__ == "__main__":
    main()
