"""Route A killer: can the trace answer "where is the X" by unbinding a class?

The gate, pre-registered before running (scoping report A.6): the class field
must beat the class-agnostic null by >= 0.20 Recall@1. If it does not, the
`T (/) sem[c]` channel is not localising anything, Route A is a labelling
result only, and the retrieval harness does not get built.

Nothing new is bound. This is the SAME cached field array batch-1 already
scores -- the only change is the reduction: argmax over SPACE for one class,
instead of argmax over CLASSES at each point (common.py:184).

The null is the class-agnostic field F.mean(0): the same trace, the same
peak picker, no class channel. Any gain over it is the class symbol doing
work and nothing else.

    python collab_tasks/scripts/class_query_killer.py            # room0 gate
    python collab_tasks/scripts/class_query_killer.py --all      # 8 scenes
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
    GRID, SCENES, SEEDS, default_fields, load_scene, predict)
from vsa_cognitive_mapping.object_grounding import field_peaks  # noqa: E402

# ConceptGraphs' n_exclude 6 list -- their protocol, not ours (05_score.py:61)
CG_EXCLUDE_6 = ("other", "floor", "wall", "ceiling", "door", "window")
RADII = (0.5, 0.75, 1.0)   # repo-local, from instance_recall.py:79; swept
K = 3


def gt_targets(scene, a, b):
    """GT instance centroids projected onto the trace's own (a, b) plane."""
    raw = json.load(open(f"outputs/replica_{scene}/gt_instances.json"))
    out = {}
    for g in raw["instances"]:
        if g["cls"] in CG_EXCLUDE_6:
            continue
        p = [g["x"], g["y"], g["z"]]
        out.setdefault(g["cls"], []).append((p[a], p[b]))
    return out


def recall_at_k(peaks, targets, radius):
    """[R@1, R@2, R@3]: does any of the first k peaks land within radius of
    ANY GT instance of that class."""
    ok = [any(np.hypot(px - tx, py - ty) <= radius for tx, ty in targets)
          for px, py in peaks]
    return [int(any(ok[:k + 1])) for k in range(K)]


def run_scene(scene, ti=0):
    data = load_scene(scene)
    F, names, cell = default_fields(data, ti)
    xyz, a, b = data["xyz"], data["a"], data["b"]
    gx = np.linspace(xyz[:, a].min(), xyz[:, a].max(), GRID)
    gy = np.linspace(xyz[:, b].min(), xyz[:, b].max(), GRID)

    tgt = gt_targets(scene, a, b)
    null_peaks = field_peaks(np.asarray(F).mean(0), gx, gy, GRID, k=K)

    res = {r: {"classq": [], "null": []} for r in RADII}
    obs_res = {r: {"classq": [], "null": []} for r in RADII}
    seen, unobserved = [], []
    hits = {"C": [], "N": []}
    for c in sorted(tgt):
        if c in names:
            pk = field_peaks(np.asarray(F)[names.index(c)], gx, gy, GRID, k=K)
            seen.append(c)
        else:
            pk = []                      # never observed -> scores 0, correct
            unobserved.append(c)
        for r in RADII:
            cq = recall_at_k(pk, tgt[c], r)
            nl = recall_at_k(null_peaks, tgt[c], r)
            res[r]["classq"].append(cq)
            res[r]["null"].append(nl)
            if c in names:               # DIAGNOSTIC split, not the gate
                obs_res[r]["classq"].append(cq)
                obs_res[r]["null"].append(nl)
        hits["C"].append(recall_at_k(pk, tgt[c], 0.75)[0])
        hits["N"].append(recall_at_k(null_peaks, tgt[c], 0.75)[0])
    C, N = np.array(hits["C"], bool), np.array(hits["N"], bool)
    overlap = dict(classq=int(C.sum()), null=int(N.sum()),
                   both=int((C & N).sum()), classq_only=int((C & ~N).sum()),
                   null_only=int((~C & N).sum()),
                   null_on_unobserved=int(sum(
                       n for c, n in zip(sorted(tgt), N) if c not in names)))
    return (res, obs_res, overlap, len(tgt), seen, unobserved,
            data, F, names, cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tuples", type=int, default=1)
    args = ap.parse_args()
    scenes = SCENES if args.all else ["room0"]
    tuples = range(min(args.tuples, len(SEEDS)))

    print("ROUTE A KILLER -- class-name query vs class-agnostic null")
    print(f"gate: R@1(classq) - R@1(null) >= 0.20   radii {RADII}  k={K}")
    print(f"queries = GT classes after CG n_exclude 6; unobserved classes "
          f"score 0 by construction\n")

    acc = {r: {"classq": [], "null": []} for r in RADII}
    oacc = {r: {"classq": [], "null": []} for r in RADII}
    per_scene, parity, ov = {}, [], []
    for s in scenes:
        for ti in tuples:
            (res, obs_res, overlap, nq, seen, unobs,
             data, F, names, cell) = run_scene(s, ti)
            ov.append(overlap)
            for r in RADII:
                for v in ("classq", "null"):
                    oacc[r][v].append(np.mean(obs_res[r][v], axis=0))
            # cheap sanity gate: the argmax-over-CLASSES reduction of the very
            # same array must still reproduce the stored baseline labels
            vp = f"student_gpu_package/handoff/{s}_cgfront/vsa_labels.npz"
            if ti == 0 and os.path.exists(vp):
                ref = np.load(vp, allow_pickle=True)["pred_class"].astype(str)
                got = predict(F, names, cell)
                parity.append((s, float((ref == got).mean())))
            for r in RADII:
                for v in ("classq", "null"):
                    acc[r][v].append(np.mean(res[r][v], axis=0))
            r0 = np.mean(res[0.75]["classq"], axis=0)
            n0 = np.mean(res[0.75]["null"], axis=0)
            per_scene.setdefault(s, {})[ti] = dict(
                classq=r0.tolist(), null=n0.tolist(), n_queries=nq,
                n_unobserved=len(unobs), unobserved=unobs)
            if ti == 0:
                print(f"{s:<9} q={nq:>3}  unobserved={len(unobs):>3}  "
                      f"R@1/2/3 classq {r0[0]:.3f}/{r0[1]:.3f}/{r0[2]:.3f}"
                      f"   null {n0[0]:.3f}/{n0[1]:.3f}/{n0[2]:.3f}"
                      f"   d(R@1) {r0[0]-n0[0]:+.3f}")

    print("\n" + "-" * 78)
    print(f"{'radius':>8}{'R@1 classq':>13}{'R@1 null':>11}{'delta':>9}"
          f"{'R@2':>9}{'R@3':>9}")
    summary = {}
    for r in RADII:
        cq = np.mean(acc[r]["classq"], axis=0)
        nl = np.mean(acc[r]["null"], axis=0)
        summary[r] = dict(classq=cq.tolist(), null=nl.tolist(),
                          delta_r1=float(cq[0] - nl[0]))
        print(f"{r:>8.2f}{cq[0]:>13.3f}{nl[0]:>11.3f}{cq[0]-nl[0]:>+9.3f}"
              f"{cq[1]:>9.3f}{cq[2]:>9.3f}")
    print("-" * 78)

    # ---- DIAGNOSTIC (not the gate): why the aggregate can tie ----
    print("\nDIAGNOSTIC -- the aggregate hides the behaviour")
    tot = {k: sum(o[k] for o in ov) for k in ov[0]}
    print(f"  hit-set overlap @0.75 m, summed over scenes: "
          f"classq {tot['classq']}, null {tot['null']}, "
          f"BOTH {tot['both']}, classq-only {tot['classq_only']}, "
          f"null-only {tot['null_only']}")
    print(f"  of the null's hits, {tot['null_on_unobserved']} are on classes "
          f"the trace NEVER OBSERVED --\n  a constant answer is credited "
          f"where the memory correctly cannot answer at all")
    print(f"\n{'radius':>8}{'R@1 classq':>13}{'R@1 null':>11}{'delta':>9}"
          f"   (observed classes only -- diagnostic split)")
    odel = {}
    for r in RADII:
        cq = np.mean(oacc[r]["classq"], axis=0)
        nl = np.mean(oacc[r]["null"], axis=0)
        odel[r] = dict(classq=cq.tolist(), null=nl.tolist(),
                       delta_r1=float(cq[0] - nl[0]))
        print(f"{r:>8.2f}{cq[0]:>13.3f}{nl[0]:>11.3f}{cq[0]-nl[0]:>+9.3f}")

    if parity:
        worst = min(p[1] for p in parity)
        print(f"label-parity gate vs stored vsa_labels.npz: "
              f"min {worst:.4f} over {len(parity)} scenes"
              + ("  OK" if worst > 0.999 else "  *** FAILED -- indexing bug"))

    d = summary[0.75]["delta_r1"]
    verdict = "PASS" if d >= 0.20 else "FAIL"
    print(f"\nGATE @0.75 m: delta R@1 = {d:+.3f}  ->  {verdict}")
    if verdict == "FAIL":
        print("  class channel is not localising; do not build the harness")
    else:
        print("  the unbind localises: proceed to the full 147-query run")
    if all(summary[r]["delta_r1"] >= 0.20 for r in RADII):
        print("  gain survives all three radii -- not a threshold artefact")

    os.makedirs("outputs/routeA", exist_ok=True)
    json.dump(dict(summary={str(k): v for k, v in summary.items()},
                   observed_only={str(k): v for k, v in odel.items()},
                   overlap=tot,
                   per_scene=per_scene, parity=parity,
                   gate=dict(metric="r1", radius=0.75, test="ge",
                             value=0.20, measured=d, verdict=verdict)),
              open("outputs/routeA/class_query_killer.json", "w"), indent=1)
    print("\nwrote outputs/routeA/class_query_killer.json")


if __name__ == "__main__":
    main()
