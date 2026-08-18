"""Do ConceptGraphs' human captions join onto the objects we hold?

Their `query` branch ships mturk_annotated_scenegraph_<scene>.json, a list of
{id, caption} where the caption is a human description of scene-graph node
`id`. For the DESCRIPTIVE row of their Table III that mapping IS the ground
truth: the answer to "This is a lamp and it is above the end table" is node id
0. No human relevance labelling is needed for that row.

The catch is whether `id` indexes the same object list we now hold in
cg_objects.json. If it does, 316 human-written queries with ground truth
become scorable for free. If it does not, the captions are unusable and only
the 20 affordance/negation queries remain, and those DO need labelling.

The test is honest and cheap: every caption starts "This is a <noun>", so
check whether that noun matches the class of the object at the same index.
A real alignment should agree far above chance; a mismatch means their
annotated run is a different map from the one we exported.

    python collab_tasks/table3/fetch_and_join_mturk.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

BASE = ("https://raw.githubusercontent.com/concept-graphs/concept-graphs/"
        "query/conceptgraph/clip_v_llm_exps/"
        "replica_mturk_annotated_scenegraph_nodes/"
        "mturk_annotated_scenegraph_{}.json")
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3"]
OUT = "collab_tasks/table3/mturk"


def head_noun(cap):
    m = re.match(r"\s*this is (?:a|an|the)?\s*([a-z\- ]+?)\b", cap.strip().lower())
    return m.group(1).strip() if m else None


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'scene':<9}{'captions':>9}{'valid':>7}{'max id':>8}"
          f"{'our objs':>10}{'in range':>10}{'noun==class':>13}")
    print("-" * 66)
    tot_ok = tot_n = 0
    summary = []
    for s in SCENES:
        p = f"{OUT}/{s}.json"
        if not os.path.exists(p):
            urllib.request.urlretrieve(BASE.format(s), p)
        rows = json.load(open(p))
        valid = [r for r in rows
                 if isinstance(r.get("caption"), str)
                 and "invalid" not in r["caption"].lower()]
        objp = f"student_gpu_package/handoff/{s}_cgfront/cg_objects.json"
        if not os.path.exists(objp):
            print(f"{s:<9}{len(rows):>9}{len(valid):>7}{'':>8}"
                  f"{'MISSING cg_objects.json':>33}")
            continue
        objs = json.load(open(objp))
        by_id = {int(o["id"]): str(o["cls"]) for o in objs}
        mx = max(int(r["id"]) for r in valid) if valid else -1
        inrange = sum(1 for r in valid if int(r["id"]) in by_id)
        agree = 0
        checked = 0
        for r in valid:
            n = head_noun(r["caption"])
            c = by_id.get(int(r["id"]))
            if n is None or c is None:
                continue
            checked += 1
            cl = c.lower()
            if n == cl or n in cl or cl in n:
                agree += 1
        frac = agree / checked if checked else 0.0
        tot_ok += agree
        tot_n += checked
        summary.append(dict(scene=s, captions=len(rows), valid=len(valid),
                            max_id=mx, our_objs=len(objs), in_range=inrange,
                            checked=checked, agree=agree, frac=frac))
        print(f"{s:<9}{len(rows):>9}{len(valid):>7}{mx:>8}{len(objs):>10}"
              f"{inrange:>10}{f'{100*frac:.0f}%':>13}")
    print("-" * 66)
    overall = tot_ok / tot_n if tot_n else 0.0
    print(f"overall head-noun agreement: {100*overall:.1f}% "
          f"({tot_ok}/{tot_n})")
    print()
    if overall > 0.5:
        print("=> The ids JOIN. Their 316 human captions become usable ground")
        print("   truth for descriptive retrieval, with no labelling needed.")
    else:
        print("=> The ids DO NOT join to our export. Their annotated run is a")
        print("   different map, so the captions are unusable as ground truth")
        print("   for us, and only the 20 affordance/negation queries remain")
        print("   -- and those genuinely need relevance labels.")
    json.dump(summary, open("outputs/table3_mturk_join.json", "w"), indent=1)
    print("\nwrote outputs/table3_mturk_join.json")


if __name__ == "__main__":
    main()
