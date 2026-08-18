"""Propose relevance for ConceptGraphs' 20 published queries. NOT ground truth.

Their protocol says "We manually select relevant objects as ground truth for
each query" and that selection was never published (verified: all 7 branches
of their repo, the paper appendix, and the mturk files, which turned out to be
DESCRIPTIVE captions whose node ids do not join to our export -- 5% head-noun
agreement, i.e. chance).

So the 20 affordance and negation queries need relevance labels, and this
proposes a starting set. Three things make that honest:

  1. it is MACHINE-PROPOSED, marked as such everywhere it travels, and is a
     draft for humans to correct rather than an answer;
  2. it is proposed at CLASS level from the Replica ground-truth vocabulary,
     never from anything our memory produced, so it cannot be tuned toward our
     own system;
  3. every proposal carries a confidence, so a reviewer can spend their
     attention on the genuinely arguable ones.

Emits outputs/table3/relevance_proposed.json for the labelling sheet to load.

    python collab_tasks/table3/propose_relevance.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from collab_tasks.batch1.common import (  # noqa: E402
    CG_EXCLUDE_6, GT_UNLABELLED, SCENES)

ROOM_Q = {
    "A1": "Somewhere to store decorative cups",
    "A2": "Something to add light into the room",
    "A3": "Somewhere to set food for dinner",
    "A4": "Something I can open with my keys",
    "A5": "Something to sit upright for a work call",
    "N1": "Something small, unlike a cabinet",
    "N2": "Something light, unlike a table",
    "N3": "Something soft, unlike a table",
    "N4": "Something not transparent, unlike a window",
    "N5": "Something rigid, unlike a rug",
}
OFFICE_Q = {
    "A1": "Something to watch the news on",
    "A2": "Something to tell the time",
    "A3": "Something comfortable to sit on",
    "A4": "Something to dispose of wastepaper in",
    "A5": "Something to add light into the room",
    "N1": "Something to sit on other than a chair",
    "N2": "Something very heavy, unlike a clock",
    "N3": "Something rigid, unlike a cushion",
    "N4": "Something small, unlike a couch",
    "N5": "Something light, unlike a table",
}

# class -> proposal, per query id, with confidence.
# "high" = a competent annotator would almost certainly agree.
# "low"  = genuinely arguable; a reviewer should look at these first.
ROOM_REL = {
  "A1": {"cabinet": "high", "shelf": "high", "sideboard": "high",
         "bookshelf": "high", "storage": "high", "desk-organizer": "low"},
  "A2": {"lamp": "high", "ceiling-lamp": "high", "window": "high",
         "candle": "low", "lighting": "high"},
  "A3": {"table": "high", "desk": "low", "dining-table": "high",
         "coffee-table": "low"},
  "A4": {"door": "high", "cabinet": "low", "window": "low"},
  "A5": {"chair": "high", "stool": "high", "desk-chair": "high",
         "sofa": "low", "bench": "low"},
  "N1": {"vase": "high", "book": "high", "candle": "high", "switch": "high",
         "cushion": "high", "pot": "high", "plate": "high", "bottle": "high",
         "clock": "high", "lamp": "low", "basket": "low", "tissue-paper":
         "high", "wall-plug": "high", "bin": "low", "stool": "low"},
  "N2": {"cushion": "high", "pillow": "high", "book": "high", "vase": "low",
         "blanket": "high", "plate": "low", "candle": "high", "cloth": "high",
         "tissue-paper": "high"},
  "N3": {"cushion": "high", "pillow": "high", "sofa": "high", "rug": "high",
         "blanket": "high", "comforter": "high", "cloth": "high",
         "bed": "high"},
  "N4": {"wall": "low", "cabinet": "high", "sofa": "high", "table": "high",
         "door": "low", "rug": "high", "chair": "high", "shelf": "high"},
  "N5": {"table": "high", "chair": "high", "cabinet": "high", "shelf": "high",
         "desk": "high", "stool": "high", "bookshelf": "high",
         "sideboard": "high", "pillar": "high"},
}
OFFICE_REL = {
  "A1": {"tv-screen": "high", "monitor": "high", "tv-stand": "low",
         "screen": "high"},
  "A2": {"clock": "high"},
  "A3": {"sofa": "high", "chair": "high", "beanbag": "high", "cushion": "low",
         "stool": "low", "bench": "low"},
  "A4": {"bin": "high", "basket": "low"},
  "A5": {"lamp": "high", "ceiling-lamp": "high", "window": "high",
         "lighting": "high"},
  "N1": {"sofa": "high", "beanbag": "high", "stool": "high", "bench": "high",
         "cushion": "low", "bed": "low"},
  "N2": {"sofa": "high", "table": "high", "desk": "high", "cabinet": "high",
         "bookshelf": "high", "shelf": "high", "pillar": "high",
         "tv-stand": "low"},
  "N3": {"table": "high", "desk": "high", "chair": "high", "cabinet": "high",
         "monitor": "high", "clock": "high", "bin": "high", "shelf": "high",
         "tv-screen": "high", "pillar": "high", "camera": "high",
         "tablet": "high", "desk-organizer": "high"},
  "N4": {"clock": "high", "book": "high", "camera": "high", "tablet": "high",
         "switch": "high", "bin": "low", "cushion": "high",
         "desk-organizer": "high", "tissue-paper": "high", "bottle": "high"},
  "N5": {"cushion": "high", "book": "high", "tissue-paper": "high",
         "cloth": "high", "blanket": "high", "camera": "low",
         "tablet": "low"},
}


def main():
    out = {"note": "MACHINE-PROPOSED relevance, not ground truth. "
                   "ConceptGraphs' own relevance judgements were never "
                   "published; verified across all 7 repo branches, the paper "
                   "appendix and the mturk caption files.",
           "queries": {"room": ROOM_Q, "office": OFFICE_Q},
           "scenes": {}}
    print(f"{'scene':<9}{'type':<8}{'GT classes':>11}{'queries':>9}"
          f"{'proposed relevant':>19}{'low conf':>10}")
    print("-" * 66)
    for s in SCENES:
        kind = "room" if s.startswith("room") else "office"
        rel = ROOM_REL if kind == "room" else OFFICE_REL
        inst = json.load(open(f"outputs/replica_{s}/gt_instances.json"))["instances"]
        classes = sorted({g["cls"] for g in inst
                         if g["cls"] not in CG_EXCLUDE_6
                         and g["cls"] not in GT_UNLABELLED})
        per_q, nrel, nlow = {}, 0, 0
        for qid in rel:
            hits = {c: rel[qid][c] for c in classes if c in rel[qid]}
            per_q[qid] = hits
            nrel += len(hits)
            nlow += sum(1 for v in hits.values() if v == "low")
        out["scenes"][s] = {"kind": kind, "classes": classes, "proposed": per_q}
        print(f"{s:<9}{kind:<8}{len(classes):>11}{len(rel):>9}"
              f"{nrel:>19}{nlow:>10}")
    print("-" * 66)
    os.makedirs("outputs/table3", exist_ok=True)
    json.dump(out, open("outputs/table3/relevance_proposed.json", "w"),
              indent=1)
    empty = [(s, q) for s, v in out["scenes"].items()
             for q, h in v["proposed"].items() if not h]
    print(f"\n{len(empty)} (scene, query) pairs have NO relevant class "
          f"proposed.")
    if empty:
        print("  these are queries with no answer in that scene, which is a")
        print("  legitimate outcome and must be scored as such, not skipped:")
        for s, q in empty[:12]:
            print(f"    {s} {q}")
    print("\nwrote outputs/table3/relevance_proposed.json")


if __name__ == "__main__":
    main()
