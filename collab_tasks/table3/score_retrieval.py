"""Score our memory on ConceptGraphs' 20 published text queries.

Pipeline, chosen to mirror their CLIP variant rather than their LLM one,
because ours is an embedding-similarity memory and theirs is a GPT reasoning
step over captions. Comparing a dot product to a reasoner would be the
apples-to-oranges version.

  query text -> CLIP text embedding
             -> most similar Replica CLASS NAME (open-vocabulary step)
             -> unbind that class from the 32 KB trace
             -> top-k peaks of the resulting field
  hit if any of the top-k peaks lands within RADIUS of a RELEVANT instance.

Relevance comes from outputs/table3/relevance_proposed.json, which is
MACHINE-PROPOSED and clearly marked as such. Rerun with --labels <file> to
score against human labels once the sheet comes back; the number is not
publishable until that happens.

Honest framing constraints, all of which must travel with any number:
  - their relevance judgements are unpublished, so this is OUR re-annotation
    on OUR ground-truth namespace, not a reproduction of their Table III;
  - the radius is repo-local (0.75 m, from instance_recall.py) because their
    evaluation is object-index based and has no metre threshold at all;
  - a (scene, query) pair with no relevant object is scored as unanswerable
    and excluded from recall, and the count is reported.

    .venv_lejepa/Scripts/python.exe collab_tasks/table3/score_retrieval.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from collab_tasks.batch1.common import (  # noqa: E402
    CG_EXCLUDE_6, GRID, SCENES, SEEDS, default_fields, load_scene)
from vsa_cognitive_mapping.object_grounding import field_peaks  # noqa: E402

RADII = (0.5, 0.75, 1.0)
KMAX = 3


def clip_text(strings):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from vsa_cognitive_mapping.clip_compat import text_features
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        t = p(text=strings, return_tensors="pt", padding=True)
        E = text_features(m, t)
        E = E / E.norm(dim=-1, keepdim=True)
    return E.numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/table3/relevance_proposed.json")
    ap.add_argument("--k", type=int, default=KMAX)
    args = ap.parse_args()
    L = json.load(open(args.labels))
    machine = "MACHINE-PROPOSED" in L.get("note", "")

    # embed every query once, and every class name once
    qtext, qkey = [], []
    for kind in ("room", "office"):
        for qid, s in L["queries"][kind].items():
            qkey.append((kind, qid))
            qtext.append(s)
    allcls = sorted({c for v in L["scenes"].values() for c in v["classes"]})
    QE = clip_text(qtext)
    CE = clip_text([f"an image of a {c}" for c in allcls])
    ci = {c: i for i, c in enumerate(allcls)}
    qi = {k: i for i, k in enumerate(qkey)}

    rows, unans = [], 0
    for s in SCENES:
        sc = L["scenes"][s]
        kind = sc["kind"]
        data = load_scene(s)
        F, names, cell = default_fields(data, 0)
        F = np.asarray(F, float)
        xyz, a, b = data["xyz"], data["a"], data["b"]
        gx = np.linspace(xyz[:, a].min(), xyz[:, a].max(), GRID)
        gy = np.linspace(xyz[:, b].min(), xyz[:, b].max(), GRID)
        inst = json.load(open(f"outputs/replica_{s}/gt_instances.json"))["instances"]
        inst = [g for g in inst if g["cls"] not in CG_EXCLUDE_6]

        for qid in L["queries"][kind]:
            rel_cls = set(sc["proposed"][qid])
            tgt = [( [g["x"], g["y"], g["z"]][a], [g["x"], g["y"], g["z"]][b] )
                   for g in inst if g["cls"] in rel_cls]
            if not tgt:
                unans += 1
                rows.append(dict(scene=s, qid=qid, answerable=False))
                continue
            # open-vocabulary step: query -> nearest class name present here
            sub = [c for c in sc["classes"]]
            sims = CE[[ci[c] for c in sub]] @ QE[qi[(kind, qid)]]
            order = np.argsort(sims)[::-1]
            picked = [sub[o] for o in order[:args.k]]
            # the memory answers: unbind the top class, take its peaks
            top = picked[0]
            pk = (field_peaks(F[names.index(top)], gx, gy, GRID, k=args.k)
                  if top in names else [])
            hit = {}
            for r in RADII:
                ok = [any(np.hypot(px - tx, py - ty) <= r for tx, ty in tgt)
                      for px, py in pk]
                hit[r] = [int(any(ok[:kk + 1])) for kk in range(args.k)]
                while len(hit[r]) < args.k:
                    hit[r].append(0)
            rows.append(dict(scene=s, qid=qid, answerable=True,
                             picked=picked, in_trace=top in names,
                             correct_class=top in rel_cls,
                             hit={str(k_): v for k_, v in hit.items()}))
        print(f"  {s} done", flush=True)

    ans = [r for r in rows if r["answerable"]]
    print(f"\n{'='*64}")
    print("RETRIEVAL ON CONCEPTGRAPHS' 20 PUBLISHED QUERIES")
    if machine:
        print("*** relevance labels are MACHINE-PROPOSED, not human. "
              "NOT PUBLISHABLE. ***")
    print(f"{'='*64}")
    print(f"{len(rows)} (scene, query) pairs, {unans} unanswerable "
          f"(no relevant object in that scene), {len(ans)} scored\n")
    print(f"{'radius':>8}{'R@1':>9}{'R@2':>9}{'R@3':>9}")
    print("-" * 35)
    for r in RADII:
        v = [np.mean([x["hit"][str(r)][k] for x in ans]) for k in range(args.k)]
        print(f"{r:>8.2f}" + "".join(f"{x:>9.3f}" for x in v))
    print("-" * 35)
    cc = np.mean([x["correct_class"] for x in ans])
    it = np.mean([x["in_trace"] for x in ans])
    print(f"\nCLIP picked a RELEVANT class first on {100*cc:.0f}% of queries")
    print(f"that class was present in the trace on {100*it:.0f}%")
    print("\nfor reference, their Replica Table III:")
    print("  CLIP  affordance 0.43/0.57/0.63   negation 0.26/0.60/0.71")
    print("  LLM   affordance 0.57/0.63/0.66   negation 0.80/0.89/0.97")
    print("NOT the same measurement: their relevance judgements are")
    print("unpublished, so this is our re-annotation on our GT namespace.")

    for split, ids in (("affordance", "A"), ("negation", "N")):
        sub = [x for x in ans if x["qid"].startswith(ids)]
        if not sub:
            continue
        v = [np.mean([x["hit"]["0.75"][k] for x in sub]) for k in range(args.k)]
        print(f"  ours {split:<11}" + "/".join(f"{x:.2f}" for x in v)
              + f"   (n={len(sub)})")

    os.makedirs("outputs/table3", exist_ok=True)
    json.dump(dict(machine_labels=machine, rows=rows, n_unanswerable=unans),
              open("outputs/table3/retrieval_scores.json", "w"), indent=1)
    print("\nwrote outputs/table3/retrieval_scores.json")


if __name__ == "__main__":
    main()
