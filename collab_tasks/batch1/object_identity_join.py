"""Is the 87.6% shared-ceiling branch a FEW bad objects, or MANY slightly-wrong ones?

This is the question the whole clip_ft export exists to answer, and it has
been unanswerable because `cg_frontend_to_trace.py` replaced ConceptGraphs'
object id with a running index (fixed 2026-08-17, commit c3aae37).

Background, measured in shared_ceiling.py: of the 14,134 cells our trace gets
wrong, only 38.3% are cells ConceptGraphs wins. The other 61.7% neither system
gets right -- and in the branch where the stream's own nearest observation is
closer, 87.6% are shared failures. That is either

  CONCENTRATED  a handful of their objects carry a wrong label, each covering
                many cells -> a fixable frontend problem, and re-deciding
                those objects' labels from their clip_ft is the fix; or
  DIFFUSE       many objects each contribute a few cells -> a genuine shared
                ceiling, to be REPORTED as one rather than chased.

The join: our object_points.json carries det=i, the positional index into
their cg_observations.json (cg_frontend_to_trace.py:104). Position is only
meaningful if the export matches the one our published trace was built from,
so the row count is a HARD GATE here, not a warning -- a mismatched export
silently maps every point to the wrong object.

    python collab_tasks/batch1/object_identity_join.py --src <dir with per-scene exports>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, SEEDS, _SC, cap_per_class, default_fields, load_scene, predict)
from collab_tasks.batch1.error_decomposition import decompose  # noqa: E402

EXPECTED = dict(room0=13124, room1=8383, room2=8543, office0=10956,
                office1=7156, office2=11061, office3=12882, office4=10047)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="student_gpu_package/handoff",
                    help="dir holding <scene>/cg_observations.json etc")
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()
    scenes = args.scenes or SCENES

    rows, per_scene = [], {}
    for s in scenes:
        f = f"{args.src}/{s}/cg_observations.json"
        if not os.path.exists(f):
            print(f"{s}: no export at {f} -- skipped")
            continue
        obs = json.load(open(f))
        if len(obs) != EXPECTED[s]:
            print(f"{s}: HARD GATE FAILED -- {len(obs)} rows, expected "
                  f"{EXPECTED[s]}. This export does NOT match the published "
                  f"trace; positional join would be wrong. SKIPPED.")
            continue
        objs = {o["id"]: o for o in
                json.load(open(f"{args.src}/{s}/cg_objects.json"))}

        data = load_scene(s)
        cats, _, ours = decompose(data)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        pts = data["obs"]                      # carries det (and now obj)
        capped = cap_per_class(list(pts), 400, seed=SEEDS[0][2])
        by_cls = {}
        for o in capped:
            by_cls.setdefault(o["cls"], []).append(o)

        C = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                    allow_pickle=True)
        lab = _SC.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
        theirs = np.array([v if v is not None else "__none__" for v in lab])

        blame = Counter()
        n_shared = 0
        for i in sorted(cats["local_loss"]):
            g, w = gt[i], ours[i]
            if theirs[i] == g:                 # they win it: real gap, not here
                continue
            cand = by_cls.get(w)
            if not cand:
                continue
            px, py = float(xyz[i, a]), float(xyz[i, b])
            d = [(abs(o["x"] - px) ** 2 + abs(o["y"] - py) ** 2, o)
                 for o in cand]
            _, nearest = min(d, key=lambda t: t[0])
            det = nearest.get("det")
            if det is None or det >= len(obs):
                continue
            oid = obs[det]["obj"]
            blame[(s, oid)] += 1
            n_shared += 1

        per_scene[s] = dict(n_shared=n_shared, n_objects=len(objs),
                            n_blamed=len(blame))
        rows.append((s, blame, n_shared, objs))
        print(f"{s}: {n_shared} shared-failure cells traced to "
              f"{len(blame)} of {len(objs)} objects", flush=True)

    if not rows:
        raise SystemExit("\nNo scene passed the row-count gate -- nothing to do.")

    allb = Counter()
    for s, blame, _, _ in rows:
        allb.update(blame)
    total = sum(allb.values())
    print(f"\n{total} shared-failure cells over {len(rows)} scenes, "
          f"traced to {len(allb)} distinct ConceptGraphs objects")
    print("=" * 66)
    ranked = allb.most_common()
    for frac in (0.5, 0.8, 0.9):
        need, acc = 0, 0
        for _, c in ranked:
            acc += c; need += 1
            if acc >= frac * total:
                break
        print(f"  {frac:.0%} of the cells come from the worst "
              f"{need} objects ({100*need/len(allb):.1f}% of blamed objects)")

    print("\nworst 10 objects:")
    objmap = {s: o for s, _, _, o in rows}
    for (s, oid), c in ranked[:10]:
        ob = objmap[s].get(oid, {})
        print(f"  {s:<9} obj {oid:<4} labelled '{ob.get('cls','?'):<14}' "
              f"{ob.get('n_points',0):>6} pts   {c:>5} cells "
              f"({100*c/total:.1f}%)")

    top1 = 100 * ranked[0][1] / total
    need50 = next(i + 1 for i, _ in enumerate(ranked)
                  if sum(c for _, c in ranked[:i + 1]) >= 0.5 * total)
    verdict = ("CONCENTRATED -- a small set of mislabelled objects carries the "
               "branch; re-deciding their labels from clip_ft is a real lever"
               if need50 <= 0.1 * len(allb) else
               "DIFFUSE -- no small set dominates; this is a shared ceiling "
               "and should be reported as one, not chased")
    print(f"\nsingle worst object = {top1:.1f}% of all shared-failure cells")
    print(f"VERDICT: {verdict}")

    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(total=total, n_objects_blamed=len(allb),
                   per_scene=per_scene, need_for_50pct=need50,
                   worst=[dict(scene=s, obj=o, cells=c) for (s, o), c in ranked[:40]]),
              open("outputs/batch1/object_identity_join.json", "w"), indent=1)
    print("\nwrote outputs/batch1/object_identity_join.json")


if __name__ == "__main__":
    main()
