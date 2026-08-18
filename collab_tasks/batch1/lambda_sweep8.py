"""Global lambda sweep on all 8 cgfront scenes, corrected pipeline.

Flagged as untried on 2026-08-18: LX=0.45 / LY=0.27 came from a magnitude
0.35 / ratio 1.67 fit on room0 and room2 ONLY, before the two scoring-bug
fixes, and every batch-1 harness inherited it unquestioned.

The h7 post-mortem made it urgent: on room0 at tuple 0, lambda multipliers
0.30/0.45/0.70/1.00/1.60/2.50 scored 0.046/0.016/0.032/0.259/0.060/0.021 --
only the shipped value carried any signal at all. If that razor-sharp optimum
is real and general it explains a great deal (why h3 and h3b, which perturb
lambda per class, could not resolve; why multi-scale bundling collapses), and
it closes the lambda question rather than leaving it open.

Both metrics, all 8 scenes. Single tuple: this is a LANDSCAPE sweep to locate
the optimum, not a mechanism screen -- any winner must then go through
run_screen at 5 tuples before it means anything.

    python collab_tasks/batch1/lambda_sweep8.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    LX, LY, SCENES, SEEDS, class_fields, load_scene, predict, score)

MULTS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)


def main():
    res = {}
    for m in MULTS:
        ma, fm = [], []
        for s in SCENES:
            d = load_scene(s)
            F, n, c = class_fields(d, lx=LX * m, ly=LY * m, seeds=SEEDS[0])
            sc = score(d["gt"], predict(F, n, c))
            ma.append(sc["macc"])
            fm.append(sc["fmiou"])
        res[m] = dict(macc=ma, fmiou=fm)
        print(f"  mult {m:.2f} (lx={LX*m:.3f}) done  mAcc {np.mean(ma):.4f}",
              flush=True)

    print("\nGLOBAL LAMBDA SWEEP, 8 scenes, tuple 0")
    print(f"{'mult':>6}{'lambda_x':>10}{'mAcc':>9}{'F-mIoU':>9}"
          f"{'scenes better':>15}")
    print("-" * 49)
    base = np.mean(res[1.0]["macc"])
    for m in MULTS:
        mm = np.mean(res[m]["macc"])
        nb = sum(1 for a, b in zip(res[m]["macc"], res[1.0]["macc"]) if a > b)
        star = "  <- shipped" if m == 1.0 else ""
        print(f"{m:>6.2f}{LX*m:>10.3f}{mm:>9.4f}"
              f"{np.mean(res[m]['fmiou']):>9.4f}{nb:>13}/8{star}")
    print("-" * 49)
    bm = max(MULTS, key=lambda k: np.mean(res[k]["macc"]))
    bf = max(MULTS, key=lambda k: np.mean(res[k]["fmiou"]))
    print(f"best mAcc at mult {bm:.2f} ({np.mean(res[bm]['macc']):.4f}), "
          f"best F-mIoU at mult {bf:.2f} ({np.mean(res[bf]['fmiou']):.4f})")
    if bm == 1.0:
        print("=> the shipped lambda IS the optimum on mAcc across 8 scenes; "
              "the lambda\n   question is CLOSED, not open.")
    else:
        print(f"=> shipped lambda is NOT optimal: {np.mean(res[bm]['macc'])-base:+.4f} "
              f"available at mult {bm:.2f}. Must go through run_screen at 5 "
              f"tuples before adoption.")
    json.dump({str(k): v for k, v in res.items()},
              open("outputs/batch1/lambda_sweep8.json", "w"), indent=1)
    print("\nwrote outputs/batch1/lambda_sweep8.json")


if __name__ == "__main__":
    main()
