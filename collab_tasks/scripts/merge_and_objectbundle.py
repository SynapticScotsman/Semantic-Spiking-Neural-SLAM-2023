"""Two claims made concrete: (1) merge is exact, (2) bundle objects not detections.

CLAIM 1 -- MERGEABILITY. Our trace is a sum, so the trace of two sessions is
the sum of their traces, EXACTLY, at constant size. ConceptGraphs' map is a
list of objects, so merging two sessions means re-running data association
between them, the map grows, and both full maps must be shipped because the
point clouds and CLIP features ARE the association keys.

This splits one scene's stream by FRAME into two disjoint traverses, builds a
trace for each, adds them, and checks the result against the trace built from
everything at once.

CLAIM 2 -- OBJECT BUNDLING. The fair-degradation result showed their per-object
fusion is an ~8x position denoiser (median 64 detections per object). We
currently bundle all ~13k raw detections and throw that away. Bundling their
FUSED objects instead would use ~76 items rather than ~13,124 -- 170x fewer,
so far less crosstalk -- and inherit the denoising. The risk is losing object
EXTENT: a sofa becomes a point. An earlier probe using OUR OWN clustering
found that hurt, but this is the first test with THEIR object ids.

    python collab_tasks/scripts/merge_and_objectbundle.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from collab_tasks.batch1.common import (  # noqa: E402
    CAP, GRID, HD, LX, LY, SCENES, SEEDS, _Enc, _bundle, cap_per_class,
    class_fields, load_scene, predict, score)
from vsa_cognitive_mapping.object_grounding import class_phasors  # noqa: E402


def merge_test(scene="room0"):
    d = load_scene(scene)
    obs = cap_per_class(list(d["obs"]), CAP, seed=SEEDS[0][2])
    names = sorted({o["cls"] for o in obs})
    sem = class_phasors(names, HD, seed=SEEDS[0][1])
    enc = _Enc(HD, SEEDS[0][0], LX, LY)
    Bx, By = enc.Bx.values, enc.By.values

    fr = sorted({o["frame"] for o in obs})
    mid = fr[len(fr) // 2]
    A = [o for o in obs if o["frame"] < mid]
    B = [o for o in obs if o["frame"] >= mid]
    print(f"CLAIM 1 -- MERGE  ({scene})")
    print(f"  session A: {len(A):>5} obs, frames {fr[0]}-{mid-1}")
    print(f"  session B: {len(B):>5} obs, frames {mid}-{fr[-1]}")

    T_full = _bundle(obs, names, sem, Bx, By, None, None)
    T_A = _bundle(A, names, sem, Bx, By, None, None)
    T_B = _bundle(B, names, sem, Bx, By, None, None)
    rel = np.abs(T_A + T_B - T_full).max() / np.abs(T_full).max()
    print(f"  |T_A + T_B - T_full| / max = {rel:.2e}  -> "
          f"{'EXACT (float round-off only)' if rel < 1e-12 else 'APPROXIMATE'}")
    print(f"  sizes: A {T_A.nbytes/1024:.0f} KB + B {T_B.nbytes/1024:.0f} KB "
          f"= {(T_A+T_B).nbytes/1024:.0f} KB  <- CONSTANT, not additive")

    cgobs = json.load(open(
        f"student_gpu_package/handoff/{scene}_cgfront/cg_observations.json"))
    oA = {c["obj"] for c, o in zip(cgobs, d["obs"]) if o["frame"] < mid}
    oB = {c["obj"] for c, o in zip(cgobs, d["obs"]) if o["frame"] >= mid}
    print(f"  THEIRS on the same split: {len(oA)} objects + {len(oB)} objects, "
          f"{len(oA & oB)} shared")
    print(f"    merge = {len(oA)*len(oB):,} pairwise similarity comparisons, "
          f"then greedy assignment")
    print(f"    and both full maps must travel: their point clouds and CLIP "
          f"features ARE the keys\n")


def object_bundle_test():
    print("CLAIM 2 -- BUNDLE THEIR FUSED OBJECTS INSTEAD OF RAW DETECTIONS")
    print(f"{'scene':<9}{'detections':>11}{'objects':>9}"
          f"{'mAcc detections':>17}{'mAcc objects':>14}{'delta':>9}")
    print("-" * 70)
    dd, oo = [], []
    for s in SCENES:
        d = load_scene(s)
        cgobs = json.load(open(
            f"student_gpu_package/handoff/{s}_cgfront/cg_observations.json"))
        # baseline: bundle raw detections (production)
        F, nm, cell = class_fields(d, seeds=SEEDS[0])
        m_det = score(d["gt"], predict(F, nm, cell))["macc"]

        # variant: one item per THEIR object -- fused centroid, majority label
        by = {}
        for c, o in zip(cgobs, d["obs"]):
            by.setdefault(int(c["obj"]), []).append(o)
        objs = []
        for oid, rows in by.items():
            from collections import Counter
            lab = Counter(r["cls"] for r in rows).most_common(1)[0][0]
            objs.append(dict(cls=lab, frame=0, conf=1.0, det=oid,
                             x=float(np.mean([r["x"] for r in rows])),
                             y=float(np.mean([r["y"] for r in rows]))))
        d2 = dict(d, obs=objs)
        F2, nm2, cell2 = class_fields(d2, cap=10**9, seeds=SEEDS[0])
        m_obj = score(d["gt"], predict(F2, nm2, cell2))["macc"]
        dd.append(m_det)
        oo.append(m_obj)
        print(f"{s:<9}{len(d['obs']):>11,}{len(objs):>9}"
              f"{m_det:>17.4f}{m_obj:>14.4f}{m_obj-m_det:>+9.4f}")
    print("-" * 70)
    print(f"{'MEAN':<9}{'':>11}{'':>9}{np.mean(dd):>17.4f}"
          f"{np.mean(oo):>14.4f}{np.mean(oo)-np.mean(dd):>+9.4f}")
    n = sum(1 for a, b in zip(oo, dd) if a > b)
    print(f"\nobject bundling wins on {n}/8 scenes. Items drop ~170x, so "
          f"crosstalk falls,\nbut each object becomes a POINT -- all extent is "
          f"lost. That trade is the result.")


if __name__ == "__main__":
    merge_test()
    object_bundle_test()
