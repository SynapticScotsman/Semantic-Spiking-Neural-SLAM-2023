"""The robustness comparison, with THEIR fusion given back to them.

WHY THIS EXISTS. degradation_sweep.py corrupts ConceptGraphs' FINISHED map
(cg_labels.npz) and then applies their nearest-neighbour transfer. Their
Object Fusion stage never re-runs. But their paper (arXiv:2309.16650,
"Object Fusion") fuses every detection into the mapped object as a running
mean,

    f_oj = (n_oj * f_oj + f_t,i) / (n_oj + 1)

which is averaging across detections -- the SAME mechanism we claim as our
advantage when we say bundling averages a corrupted label into noise. So the
published 84/95/80 vs 53/53/51 result may compare our averaging against their
NON-averaging, with no property of VSAs involved. That is the first thing a
reviewer will ask and it currently has no answer.

WHAT THIS DOES INSTEAD. Corruption is applied at the DETECTION level, to the
shared observation stream, before either system consolidates:

  ours   : corrupt observations -> 32 KB trace -> per-class field -> argmax
           (bundling does our averaging, exactly as before)
  theirs : corrupt the SAME observations -> group by THEIR object id ->
           per-object majority vote (the argmax limit of their running mean)
           -> nearest-neighbour transfer to eval points

Both sides now average over the same corrupted evidence. Any remaining
difference is representational.

The per-object majority vote is a SIMULATION of their fusion, not their code.
It is the argmax limit of a running mean over one-hot class evidence, which is
what their semantic feature reduces to under the argmax their scorer applies.
Stated as an approximation; the exact version needs their SAM detections and a
re-run of cfslam_pipeline_batch.py (~6 h GPU).

Requires cg_observations.json per scene (their obj ids), verified to join
positionally onto object_points.json via det=i.

    python collab_tasks/scripts/fair_degradation.py
    python collab_tasks/scripts/fair_degradation.py --scenes room0 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from collab_tasks.batch1.common import (  # noqa: E402
    CAP, GRID, LX, LY, SEEDS, _SC, cap_per_class, class_fields, load_scene,
    predict, score)

SWEEPS = [("drop", [1.0, 0.5, 0.25, 0.1, 0.05]),
          ("label", [0.0, 0.1, 0.25, 0.5]),
          ("jitter", [0.0, 0.05, 0.1, 0.25, 0.5])]


def corrupt(obs, mode, level, rng, classes):
    """Apply one degradation to the shared observation stream."""
    if mode == "drop":
        if level >= 1.0:
            return obs
        keep = rng.random(len(obs)) < level
        return [o for o, k in zip(obs, keep) if k]
    if mode == "label":
        if level <= 0:
            return obs
        out, flip = [], rng.random(len(obs)) < level
        for o, f in zip(obs, flip):
            if f:
                alt = [c for c in classes if c != o["cls"]]
                o = dict(o, cls=alt[rng.integers(len(alt))]) if alt else o
            out.append(o)
        return out
    if mode == "jitter":
        if level <= 0:
            return obs
        d = rng.normal(0, level, size=(len(obs), 2))
        return [dict(o, x=o["x"] + dx, y=o["y"] + dy)
                for o, (dx, dy) in zip(obs, d)]
    raise ValueError(mode)


def theirs_from_stream(obs, xyz, a, b):
    """Their pipeline's own defence, simulated: fuse detections into objects by
    per-object majority vote, then nearest-neighbour transfer. This is the
    argmax limit of their running-mean feature update."""
    by = defaultdict(list)
    for o in obs:
        by[o["obj"]].append(o)
    P, L = [], []
    for oid, rows in by.items():
        lab = Counter(r["cls"] for r in rows).most_common(1)[0][0]
        P.append([np.mean([r["x"] for r in rows]),
                  np.mean([r["y"] for r in rows])])
        L.append(lab)
    if not P:
        return np.array(["__none__"] * len(xyz), object)
    P = np.array(P)
    L = np.array(L, object)
    Q = np.c_[xyz[:, a], xyz[:, b]]
    out = np.empty(len(Q), object)
    for st in range(0, len(Q), 4000):
        q = Q[st:st + 4000]
        d = ((q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        out[st:st + 4000] = L[d.argmin(1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*",
                    default=["room0", "room1", "room2", "office0",
                             "office1", "office2", "office3", "office4"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="outputs/fair_degradation.json")
    args = ap.parse_args()

    results = {}
    for s in args.scenes:
        data = load_scene(s)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        cgo = json.load(open(
            f"student_gpu_package/handoff/{s}_cgfront/cg_observations.json"))
        pts = data["obs"]
        assert len(cgo) == len(pts), f"{s}: stream length mismatch"
        # positional join, verified: cg_frontend_to_trace assigned det=i
        obs = [dict(p, obj=int(c["obj"])) for p, c in zip(pts, cgo)]
        classes = sorted({o["cls"] for o in obs})

        results[s] = {}
        for mode, levels in SWEEPS:
            rows = []
            for lvl in levels:
                to, tt = [], []
                for sd in range(args.seeds):
                    rng = np.random.default_rng(1000 + sd)
                    c = corrupt(obs, mode, lvl, rng, classes)
                    if not c:
                        to.append(0.0); tt.append(0.0); continue
                    # ours: the trace, capped exactly as in production
                    d2 = dict(data, obs=c)
                    F, nm, cell = class_fields(d2, seeds=SEEDS[0])
                    to.append(score(gt, predict(F, nm, cell))["macc"])
                    # theirs: their fusion, then their transfer
                    cc = cap_per_class(list(c), CAP, seed=SEEDS[0][2])
                    tt.append(score(gt, theirs_from_stream(cc, xyz, a, b))["macc"])
                rows.append(dict(level=lvl, ours=float(np.mean(to)),
                                 theirs=float(np.mean(tt)),
                                 ours_sd=float(np.std(to)),
                                 theirs_sd=float(np.std(tt))))
            base_o, base_t = rows[0]["ours"], rows[0]["theirs"]
            for r in rows:
                r["ours_rel"] = r["ours"] / max(base_o, 1e-9)
                r["theirs_rel"] = r["theirs"] / max(base_t, 1e-9)
            results[s][mode] = rows
            print(f"  {s} {mode} done", flush=True)

    print("\nFAIR DEGRADATION -- both sides average over the same corrupted "
          "evidence\n")
    for mode, levels in SWEEPS:
        print(f"--- {mode} ---")
        print(f"{'level':>7}{'theirs':>9}{'ours':>8}{'theirs ret':>12}"
              f"{'ours ret':>10}{'we lead':>9}")
        for i, lvl in enumerate(levels):
            t = np.mean([results[s][mode][i]["theirs"] for s in args.scenes])
            o = np.mean([results[s][mode][i]["ours"] for s in args.scenes])
            rt = np.mean([results[s][mode][i]["theirs_rel"] for s in args.scenes])
            ro = np.mean([results[s][mode][i]["ours_rel"] for s in args.scenes])
            n = sum(1 for s in args.scenes
                    if results[s][mode][i]["ours"] > results[s][mode][i]["theirs"])
            print(f"{lvl:>7}{t:>9.3f}{o:>8.3f}{100*rt:>11.0f}%{100*ro:>9.0f}%"
                  f"{n:>7}/{len(args.scenes)}")
        print()
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
