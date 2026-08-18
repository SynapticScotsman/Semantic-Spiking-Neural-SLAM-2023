"""Re-decide their object labels from their own clip_ft, and re-score BOTH.

relabel_headroom.py found the truth sits at rank 2 for 37.9% of shared-failure
cells and inside the top-3 for 48.5%, with only 6.1% beyond rank 10. This asks
what that is actually worth, and it must separate two very different things:

  ORACLE     use GT to pick which objects are wrong, then take the truth.
             An UPPER BOUND, not a method -- reported to size the prize.
  HONEST     a rule with no access to the scene's GT. The failure shape says
             what it should be: 60.6% of shared-failure cells are a SMALL
             class name on a BIG object (switch/camera/indoor-plant on
             chair/table/sofa) against 0.1% the other way, so a size prior
             is the principled correction.

The size prior is LEAVE-ONE-SCENE-OUT: a class's expected extent comes from
GT in the OTHER seven scenes only, never the scene being scored. Object
extent comes from the spread of that object's own observations.

    adjusted = cosine(clip_ft, text[c]) - LAM * |log(extent_obj/extent_c)|

CRITICAL SCOPE: relabelling rewrites the observation stream OUR trace is
built from AND their cg_labels.npz. Both columns move. This cannot close
0.402 vs 0.324 and is not an attempt to.

    python collab_tasks/batch1/rank2_experiment.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, SEEDS, _SC, CG_EXCLUDE_6, class_fields, load_scene, predict, score)

SRC = "student_gpu_package/handoff_clipft"
LAMS = (0.0, 0.05, 0.10, 0.20, 0.40)


def obj_extent(obs):
    """Spatial extent of each object, from its own observation positions."""
    P = defaultdict(list)
    for o in obs:
        P[o["obj"]].append((o["x"], o["y"], o["z"]))
    return {k: float(np.mean(np.std(np.array(v), axis=0)) + 1e-6)
            for k, v in P.items()}


def class_extent_from(scenes):
    """Expected extent per class, from GT eval points of the given scenes."""
    acc = defaultdict(list)
    for s in scenes:
        E = np.load(f"student_gpu_package/handoff/{s}_cgfront/eval_points.npz",
                    allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        for c in set(gt.tolist()):
            m = gt == c
            if m.sum() >= 20:
                acc[c].append(float(np.mean(np.std(xyz[m], axis=0))))
    return {c: float(np.median(v)) for c, v in acc.items()}


def main():
    T = np.load(f"{SRC}/cg_clip_text.npz", allow_pickle=True)
    text = T["text_ft"] / np.linalg.norm(T["text_ft"], axis=1, keepdims=True)
    names = [str(c) for c in T["class_names"]]

    res = {k: [] for k in ["base_theirs", "base_ours", "oracle_theirs",
                           "oracle_ours"]}
    for lam in LAMS:
        res[f"size{lam}_theirs"] = []
        res[f"size{lam}_ours"] = []

    for s in SCENES:
        obs = json.load(open(f"{SRC}/{s}/cg_observations.json"))
        Z = np.load(f"{SRC}/{s}/cg_clip_ft.npz", allow_pickle=True)
        ft = Z["clip_ft"] / np.linalg.norm(Z["clip_ft"], axis=1, keepdims=True)
        row = {int(o): i for i, o in enumerate(Z["obj_id"])}

        data = load_scene(s)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        E = np.load(f"student_gpu_package/handoff/{s}_cgfront/eval_points.npz",
                    allow_pickle=True)
        in_scene = set(E["gt_class"].astype(str).tolist())
        sel = [i for i, c in enumerate(names)
               if c in in_scene and c not in CG_EXCLUDE_6]
        selnames = [names[i] for i in sel]
        sim = ft @ text[sel].T                       # (n_obj, n_sel)

        ext = obj_extent(obs)
        cext = class_extent_from([x for x in SCENES if x != s])   # LOSO

        # GT class per object, majority over the eval points nearest it
        # (used ONLY for the oracle arm)
        cent = {int(o["id"]): np.array([o["x"], o["y"], o["z"]])
                for o in json.load(open(f"{SRC}/{s}/cg_objects.json"))}
        ids = sorted(cent)
        C = np.array([cent[i] for i in ids])
        d = ((xyz[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        owner = d.argmin(1)
        truth = {}
        for j, oid in enumerate(ids):
            m = owner == j
            if m.sum():
                vals, cnt = np.unique(gt[m], return_counts=True)
                truth[oid] = str(vals[cnt.argmax()])

        def rescore(label_of):
            pts = [dict(o, cls=label_of(o["obj"])) for o in obs]
            xy = [dict(frame=p["frame"], cls=p["cls"], conf=1.0, det=i,
                       x=[p["x"], p["y"], p["z"]][a],
                       y=[p["x"], p["y"], p["z"]][b])
                  for i, p in enumerate(pts)]
            d2 = dict(data, obs=xy)
            F, nm, cell = class_fields(d2, seeds=SEEDS[0])
            ours = score(gt, predict(F, nm, cell))["macc"]
            # theirs: nearest-object label per eval point
            th = np.array([label_of(ids[j]) for j in owner])
            theirs = _SC.macc_fmiou(gt, th, exclude=CG_EXCLUDE_6)[0]
            return theirs, ours

        base_lbl = {int(o["id"]): o["cls"]
                    for o in json.load(open(f"{SRC}/{s}/cg_objects.json"))}
        t, o = rescore(lambda oid: base_lbl.get(oid, "__none__"))
        res["base_theirs"].append(t); res["base_ours"].append(o)

        t, o = rescore(lambda oid: truth.get(oid, base_lbl.get(oid, "__none__")))
        res["oracle_theirs"].append(t); res["oracle_ours"].append(o)

        for lam in LAMS:
            adj = sim.copy()
            if lam:
                for j, c in enumerate(selnames):
                    if c not in cext:
                        continue
                    pen = np.array([abs(np.log(ext.get(oid, cext[c]) / cext[c]))
                                    for oid in ids])
                    adj[[row[i] for i in ids], j] -= lam * pen
            newl = {oid: selnames[int(adj[row[oid]].argmax())]
                    for oid in ids if oid in row}
            t, o = rescore(lambda oid: newl.get(oid, base_lbl.get(oid, "__none__")))
            res[f"size{lam}_theirs"].append(t)
            res[f"size{lam}_ours"].append(o)
        print(f"{s} done", flush=True)

    m = {k: float(np.mean(v)) for k, v in res.items()}
    print("\nmAcc, mean over 8 scenes, their protocol")
    print("=" * 62)
    print(f"{'rule':<34}{'THEIRS':>9}{'OURS':>9}{'gap':>9}")
    print("-" * 62)
    rows = [("their labels as shipped", "base"),
            ("ORACLE: GT label per object", "oracle")]
    rows += [(f"size prior, lambda={l} (LOSO)", f"size{l}") for l in LAMS]
    for lab, k in rows:
        t, o = m[f"{k}_theirs"], m[f"{k}_ours"]
        print(f"{lab:<34}{t:>9.3f}{o:>9.3f}{o-t:>+9.3f}")
    print("=" * 62)
    print(f"published reference: theirs 0.402, ours 0.324")
    json.dump(dict(mean=m, per_scene={k: v for k, v in res.items()},
                   lambdas=list(LAMS)),
              open("outputs/batch1/rank2_experiment.json", "w"), indent=1)
    print("\nwrote outputs/batch1/rank2_experiment.json")


if __name__ == "__main__":
    main()
