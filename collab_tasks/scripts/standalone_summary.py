"""Collate the 8-scene standalone (YOLO boxes + CLIP labels) result.

Reads student_gpu_package/handoff/<scene>/scores.json -- the STANDALONE
pipeline -- and puts it beside the two numbers already published from the
_cgfront pipeline, with the distinction stated rather than implied:

  standalone : our YOLOv8n boxes, relabelled by CLIP over Replica's 101-class
               vocabulary, into our 32 KB trace. Our own frontend end to end.
  cgfront    : ConceptGraphs' SAM+CLIP object stream into the same trace.
               This is the one comparable to their 0.402 -- same input, only
               the memory differs.

The two are NOT interchangeable and must never share a column.

    python collab_tasks/scripts/standalone_summary.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]
# published cgfront numbers (outputs/batch1/table2_full.json)
CGF = json.load(open("outputs/batch1/table2_full.json"))["per_scene"]


def main():
    rows, miss = [], []
    for s in SCENES:
        p = f"student_gpu_package/handoff/{s}/scores.json"
        if not os.path.exists(p):
            miss.append(s)
            continue
        d = json.load(open(p))["vsa"]
        rows.append(dict(scene=s,
                         macc=d["their_protocol"]["mAcc"],
                         fmiou=d["their_protocol"]["fmiou"],
                         ncls=d["their_protocol"]["gt_classes"],
                         unmatched=d.get("unmatched_frac", float("nan"))))
    if miss:
        print(f"MISSING (not yet run): {', '.join(miss)}\n")
    if not rows:
        raise SystemExit("no scenes scored yet")

    print("STANDALONE PIPELINE -- our YOLO boxes + CLIP relabel -> 32 KB trace")
    print("(NOT comparable to the 0.402/0.324 head-to-head, which uses THEIR "
          "SAM+CLIP stream)\n")
    print(f"{'scene':<9}{'mAcc':>8}{'F-mIoU':>9}{'GT cls':>8}{'unmatched':>11}"
          f"{'  |  cgfront mAcc':>18}{'ratio':>8}")
    print("-" * 74)
    for r in rows:
        cg = CGF.get(r["scene"], {}).get("ours", {}).get("macc")
        cgs = f"{cg:.3f}" if cg else "n/a"
        rat = f"{r['macc']/cg:.2f}x" if cg else "n/a"
        print(f"{r['scene']:<9}{r['macc']:>8.3f}{r['fmiou']:>9.3f}"
              f"{r['ncls']:>8}{r['unmatched']:>11.3f}{cgs:>18}{rat:>8}")
    print("-" * 74)
    m = np.mean([r["macc"] for r in rows])
    f = np.mean([r["fmiou"] for r in rows])
    cgm = np.mean([CGF[r["scene"]]["ours"]["macc"] for r in rows
                   if r["scene"] in CGF])
    print(f"{'MEAN':<9}{m:>8.3f}{f:>9.3f}{'':>8}{'':>11}{cgm:>18.3f}"
          f"{m/cgm:>7.2f}x")
    print(f"\n{len(rows)} scenes. Standalone mean mAcc {m:.4f}, "
          f"F-mIoU {f:.4f}.")
    print(f"room0 was previously the ONLY standalone scene measured "
          f"({rows[0]['macc']:.4f}); this is the first 8-scene number.")
    print(f"\nThe frontend gap: same memory, same decode, same scorer. The "
          f"only difference\nbetween the two columns is which detector "
          f"produced the observations.")
    json.dump(dict(per_scene=rows, mean_macc=float(m), mean_fmiou=float(f),
                   cgfront_mean_macc=float(cgm),
                   note="standalone = our YOLO boxes + CLIP relabel; NOT "
                        "comparable to the 0.402 head-to-head"),
              open("outputs/standalone_8scene.json", "w"), indent=1)
    print("\nwrote outputs/standalone_8scene.json")


if __name__ == "__main__":
    main()
