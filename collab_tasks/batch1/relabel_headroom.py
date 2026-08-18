"""Is re-deciding their labels from clip_ft actually a LEVER, or is CLIP
confidently wrong?

object_identity_join.py showed the shared-ceiling branch is concentrated:
11 of 611 objects carry 50% of the 8,719 cells neither system gets right.
That makes "re-label those objects" sound obvious -- but it is only a lever
if the CORRECT class is recoverable from the features they already store.
Two possibilities, and they point opposite ways:

  RECOVERABLE  the true class sits at rank 2-3 in their own clip_ft
               similarity ranking -> a better decision rule over the SAME
               features fixes it, and the information was there all along.
  CONFIDENTLY  the true class is far down the ranking -> CLIP genuinely
  WRONG        does not see it, no decision rule helps, and this is a real
               shared ceiling to report rather than chase.

For every blamed object: take the GT class of the cells it is blamed for
(majority vote), then find where that class ranks in cosine(clip_ft, text)
restricted to the in-scene classes -- their own protocol.

    python collab_tasks/batch1/relabel_headroom.py
"""
from __future__ import annotations

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

SRC = "clipft_export"
EXCLUDE = ["other", "floor", "wall", "ceiling", "door", "window"]


def main():
    T = np.load(f"{SRC}/cg_clip_text.npz", allow_pickle=True)
    text, names = T["text_ft"], [str(c) for c in T["class_names"]]
    text = text / np.linalg.norm(text, axis=1, keepdims=True)

    ranks, rows = [], []
    for s in SCENES:
        obs = json.load(open(f"{SRC}/{s}/cg_observations.json"))
        Z = np.load(f"{SRC}/{s}/cg_clip_ft.npz", allow_pickle=True)
        ft = Z["clip_ft"] / np.linalg.norm(Z["clip_ft"], axis=1, keepdims=True)
        oid2row = {int(o): i for i, o in enumerate(Z["obj_id"])}

        data = load_scene(s)
        cats, _, ours = decompose(data)
        gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
        E = np.load(f"student_gpu_package/handoff/{s}_cgfront/eval_points.npz",
                    allow_pickle=True)
        in_scene = set(E["gt_class"].astype(str).tolist())
        sel = [i for i, c in enumerate(names)
               if c in in_scene and c not in EXCLUDE]

        C = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                    allow_pickle=True)
        lab = _SC.transfer(C["xyz"], C["pred_class"].astype(object), xyz)
        theirs = np.array([v if v is not None else "__none__" for v in lab])

        capped = cap_per_class(list(data["obs"]), 400, seed=SEEDS[0][2])
        by_cls = {}
        for o in capped:
            by_cls.setdefault(o["cls"], []).append(o)

        blame = {}
        for i in sorted(cats["local_loss"]):
            if theirs[i] == gt[i]:
                continue
            cand = by_cls.get(ours[i])
            if not cand:
                continue
            px, py = float(xyz[i, a]), float(xyz[i, b])
            nearest = min(cand, key=lambda o: (o["x"]-px)**2 + (o["y"]-py)**2)
            det = nearest.get("det")
            if det is None or det >= len(obs):
                continue
            blame.setdefault(obs[det]["obj"], []).append(gt[i])

        for oid, gts in blame.items():
            if oid not in oid2row:
                continue
            truth = Counter(gts).most_common(1)[0][0]
            if truth not in names or truth in EXCLUDE:
                continue
            sim = ft[oid2row[oid]] @ text[sel].T
            order = [names[sel[j]] for j in np.argsort(-sim)]
            r = order.index(truth) if truth in order else 999
            ranks.append((r, len(gts)))
            rows.append(dict(scene=s, obj=int(oid), cells=len(gts),
                             assigned=order[0], truth=truth, rank=r,
                             n_candidates=len(sel)))
        print(f"{s}: {len(blame)} blamed objects scored", flush=True)

    ranks.sort(key=lambda t: -t[1])
    tot = sum(c for _, c in ranks)
    print(f"\n{len(ranks)} blamed objects, {tot} cells")
    print("Where does the TRUE class sit in their own clip_ft ranking?")
    print("=" * 64)
    for lo, hi, lab in ((0, 0, "rank 1  (already top -- our decode lost it)"),
                        (1, 1, "rank 2  <- one better guess away"),
                        (2, 4, "rank 3-5"),
                        (5, 9, "rank 6-10"),
                        (10, 998, "rank 11+  (CLIP does not see it)"),
                        (999, 999, "not selectable at all")):
        n = sum(1 for r, _ in ranks if lo <= r <= hi)
        c = sum(c for r, c in ranks if lo <= r <= hi)
        if n:
            print(f"  {lab:<44}{n:>4} objs{c:>7} cells ({100*c/tot:>4.1f}%)")
    top3 = sum(c for r, c in ranks if r <= 2)
    print("=" * 64)
    print(f"cells whose true class is in their own TOP-3: {top3} "
          f"({100*top3/tot:.1f}%)")
    print("\nworst objects, with where the truth actually ranked:")
    for r in sorted(rows, key=lambda d: -d["cells"])[:10]:
        print(f"  {r['scene']:<9} obj {r['obj']:<4} {r['cells']:>5} cells  "
              f"CG said '{r['assigned']:<14}' truth '{r['truth']:<14}' "
              f"rank {r['rank']+1}/{r['n_candidates']}")
    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(dict(n_objects=len(ranks), n_cells=tot,
                   pct_truth_in_top3=100*top3/tot, objects=rows),
              open("outputs/batch1/relabel_headroom.json", "w"), indent=1)
    print("\nwrote outputs/batch1/relabel_headroom.json")


if __name__ == "__main__":
    main()
