"""Room extent per Replica scene — the independent variable for the lambda rule.

Measured on room0 (2026-08-15): the best per-axis length scale was 0.45, 0.27,
a ratio of 1.67, and room0's floor extent is 7.7 x 4.6 m, a ratio of 1.67. That
coincidence is either a rule — lambda per axis proportional to extent per axis —
or one lucky pair of numbers fitted to the scene we report.

The rule is only testable if the scenes actually DIFFER in aspect ratio. If all
eight Replica rooms are near-square, a rule keyed on aspect ratio predicts almost
nothing and cannot be distinguished from "0.45, 0.27 is just good". So measure
the spread before running any sweep.

Extent is taken from the GT instance centroids (the same floor axes everything
else uses: the two largest-variance translation axes), and reported alongside the
camera path extent, which is what our observations actually cover.

    python collab_tasks/scripts/scene_extents.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]
EXCLUDED = {"other", "floor", "wall", "ceiling", "door", "window", "class_-1"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    args = ap.parse_args()

    print(f"{'scene':<10}{'GT x (m)':>10}{'GT y (m)':>10}{'ratio':>8}"
          f"{'obs x':>9}{'obs y':>9}{'ratio':>8}{'n_inst':>8}"
          f"   predicted lambda")
    print("-" * 96)
    rows = []
    for s in args.scenes:
        gp = f"outputs/replica_{s}/gt_instances.json"
        op = f"outputs/replica_{s}/object_points.json"
        if not (os.path.exists(gp) and os.path.exists(op)):
            print(f"{s:<10} missing artifacts")
            continue
        G = json.load(open(gp))["instances"]
        keep = [i for i in G if i["cls"] not in EXCLUDED]
        gx = np.array([i["x"] for i in keep])
        gy = np.array([i["y"] for i in keep])
        ex, ey = gx.max() - gx.min(), gy.max() - gy.min()

        P = json.load(open(op))["points"]
        ox = np.array([p["x"] for p in P])
        oy = np.array([p["y"] for p in P])
        ox_e, oy_e = ox.max() - ox.min(), oy.max() - oy.min()

        # room0's winner was 0.45, 0.27 at extent 7.7 x 4.6. Carry that scale
        # across: lambda_axis = 0.45 * (extent_axis / 7.7). This is the rule
        # being tested, written down explicitly so it can be wrong.
        lx, ly = 0.45 * ex / 7.7, 0.45 * ey / 7.7
        rows.append((s, ex, ey, ex / ey, lx, ly))
        print(f"{s:<10}{ex:>10.2f}{ey:>10.2f}{ex/ey:>8.2f}"
              f"{ox_e:>9.2f}{oy_e:>9.2f}{ox_e/oy_e:>8.2f}{len(keep):>8}"
              f"   {lx:.2f},{ly:.2f}")

    if len(rows) > 1:
        r = np.array([x[3] for x in rows])
        print(f"\naspect ratio across {len(rows)} scenes: min {r.min():.2f} "
              f"max {r.max():.2f} spread {r.max()-r.min():.2f}")
        if r.max() - r.min() < 0.3:
            print("SPREAD TOO SMALL -- these rooms are near-identical in shape, so "
                  "an\naspect-ratio rule predicts almost nothing here and a sweep "
                  "cannot\ndistinguish it from a single good pair of numbers.")
        else:
            print("Spread is usable: the rule predicts a DIFFERENT lambda per "
                  "scene, so a\nper-scene sweep can confirm or refute it. If the "
                  "best ratio tracks the\nextent ratio across scenes, it is a "
                  "rule; if 0.45,0.27 wins everywhere\nregardless of shape, it is "
                  "not.")


if __name__ == "__main__":
    main()
