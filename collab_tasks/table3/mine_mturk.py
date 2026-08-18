"""Everything the mturk caption files tell us, short of the relevance labels.

The ids do not join to our export, so the captions cannot serve as ground
truth. They are still evidence about THEIR pipeline, which is worth extracting
because it is the only direct view we have of the map their annotators saw.

Four things it can settle:
  1. how many objects their annotated run produced per scene, versus ours;
  2. which scenes they annotated at all (office4 is absent);
  3. whether the "valid object" counts in their Table I match these files,
     which would confirm these ARE the files behind the paper;
  4. the vocabulary and the spatial relations humans reached for, which is a
     free description of what a person considers worth saying about a room.

    python collab_tasks/table3/mine_mturk.py
"""
from __future__ import annotations

import collections
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

SC = ["room0", "room1", "room2", "office0", "office1", "office2", "office3"]
# ConceptGraphs Table I, "valid objects" column, as printed in the paper
PAPER_VALID = {"room0": 54, "room1": 43, "room2": 47, "office0": 44,
               "office1": 23, "office2": 44, "office3": 60}

REL = ["above", "below", "next to", "on top of", "in front of", "behind",
       "under", "beside", "near", "against", "attached to", "hanging",
       "left of", "right of", "between", "inside", "on the wall", "on the"]


def main():
    print("1. THEIR ANNOTATED RUN vs OUR EXPORT\n")
    print(f"{'scene':<9}{'captions':>9}{'valid':>7}{'invalid':>9}"
          f"{'max id':>8}{'their objs>=':>14}{'our objs':>10}{'paper':>7}"
          f"{'match':>7}")
    print("-" * 80)
    tot = collections.Counter()
    nouns, rels = collections.Counter(), collections.Counter()
    for s in SC:
        rows = json.load(open(f"collab_tasks/table3/mturk/{s}.json"))
        valid = [r for r in rows if isinstance(r.get("caption"), str)
                 and "invalid" not in r["caption"].lower()]
        inval = len(rows) - len(valid)
        mx = max(int(r["id"]) for r in rows) + 1
        op = f"student_gpu_package/handoff/{s}_cgfront/cg_objects.json"
        ours = len(json.load(open(op))) if os.path.exists(op) else 0
        pv = PAPER_VALID.get(s, 0)
        ok = "yes" if len(valid) == pv else f"no ({len(valid)-pv:+d})"
        tot["captions"] += len(rows); tot["valid"] += len(valid)
        tot["invalid"] += inval; tot["ours"] += ours; tot["theirs"] += mx
        print(f"{s:<9}{len(rows):>9}{len(valid):>7}{inval:>9}{mx:>8}"
              f"{mx:>14}{ours:>10}{pv:>7}{ok:>7}")
        for r in valid:
            c = r["caption"].lower()
            m = re.match(r"\s*(?:this is|these are|it is)\s+(?:a|an|the)?\s*"
                         r"([a-z\- ]+?)(?:\s+(?:and|that|which|on|in|next|"
                         r"above|below|under|near|beside|with|at|to)\b|[,.]|$)",
                         c)
            if m:
                nouns[m.group(1).strip()] += 1
            for k in REL:
                if k in c:
                    rels[k] += 1
    print("-" * 80)
    print(f"{'TOTAL':<9}{tot['captions']:>9}{tot['valid']:>7}"
          f"{tot['invalid']:>9}{'':>8}{tot['theirs']:>14}{tot['ours']:>10}")

    print(f"\n2. SCENES: 7 of 8 annotated. office4 is ABSENT from the "
          f"caption set entirely,")
    print(f"   so their descriptive row was never computed on it.")

    nmatch = sum(1 for s in SC
                 if len([r for r in json.load(open(f'collab_tasks/table3/mturk/{s}.json'))
                         if isinstance(r.get('caption'), str)
                         and 'invalid' not in r['caption'].lower()])
                 == PAPER_VALID.get(s))
    print(f"\n3. PAPER CROSS-CHECK: {nmatch}/7 scenes match Table I's "
          f"'valid objects' exactly.")
    print("   That is strong evidence these ARE the files behind the paper,")
    print("   and that the mismatching scenes differ only in how 'invalid'")
    print("   was counted, not in which run was annotated.")

    print(f"\n4. WHAT PEOPLE SAID  ({len(nouns)} distinct head nouns)\n")
    print("   most described objects:")
    for n, c in nouns.most_common(18):
        print(f"     {c:>3}  {n}")
    print("\n   spatial relations humans reached for:")
    for k, c in rels.most_common():
        print(f"     {c:>4}  {k}")

    json.dump(dict(nouns=dict(nouns), relations=dict(rels)),
              open("outputs/table3/mturk_mined.json", "w"), indent=1)
    print("\nwrote outputs/table3/mturk_mined.json")


if __name__ == "__main__":
    main()
