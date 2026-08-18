"""Re-examine the mturk join properly. The first test was too weak to trust.

The first pass matched the caption's head noun to our class name as STRINGS
and got 5%, then concluded the ids do not join. That test is flawed: humans
write "couch" where Replica's vocabulary says "sofa", "throw pillow" where it
says "cushion", "art" where it says "picture". A vocabulary mismatch would
produce a near-zero score even if the ids joined perfectly.

So this checks the same question four ways, and prints the raw pairs so a
person can judge rather than trusting any single score:

  1. string match, as before (the weak test)
  2. synonym-aware match, using an explicit hand-written map
  3. CLIP semantic match: does the caption's own text rank our object's class
     top among all classes in the scene? This needs no vocabulary agreement.
  4. id OFFSET scan: if their node list is ours shifted by a constant, the
     right offset would show a spike.

Chance level is reported alongside, because "9%" means nothing without it.

    .venv_lejepa/Scripts/python.exe collab_tasks/table3/recheck_mturk_join.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3"]

SYN = {
    "couch": "sofa", "settee": "sofa", "throw pillow": "cushion",
    "pillow": "cushion", "cushion": "cushion", "art": "picture",
    "painting": "picture", "artwork": "picture", "picture frame": "picture",
    "end table": "table", "side table": "table", "coffee table": "table",
    "dining table": "table", "desk": "desk", "tv": "tv-screen",
    "television": "tv-screen", "monitor": "monitor", "screen": "tv-screen",
    "plant": "indoor-plant", "potted plant": "indoor-plant",
    "houseplant": "indoor-plant", "rug": "rug", "carpet": "rug",
    "books": "book", "book": "book", "lamp": "lamp", "light": "lamp",
    "bin": "bin", "trash can": "bin", "wastebasket": "bin",
    "blinds": "blinds", "curtain": "blinds", "shelf": "shelf",
    "bookshelf": "shelf", "cabinet": "cabinet", "stool": "stool",
    "chair": "chair", "vase": "vase", "pot": "pot", "clock": "clock",
    "window": "window", "door": "door", "vent": "vent", "pillar": "pillar",
    "blanket": "blanket", "comforter": "comforter", "bed": "bed",
}


def head_noun(cap):
    c = cap.strip().lower()
    m = re.match(r"\s*(?:this is|these are|it is)\s+(?:a|an|the)?\s*"
                 r"([a-z\- ]+?)(?:\s+(?:and|that|which|on|in|next|above|"
                 r"below|under|near|beside|with|at|to)\b|[,.]|$)", c)
    return m.group(1).strip() if m else None


def norm(n):
    if n is None:
        return None
    n = n.strip()
    if n in SYN:
        return SYN[n]
    for k in sorted(SYN, key=len, reverse=True):
        if k in n:
            return SYN[k]
    return n


def clip_text(strings):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from vsa_cognitive_mapping.clip_compat import text_features
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    E = []
    with torch.no_grad():
        for i in range(0, len(strings), 64):
            t = p(text=strings[i:i + 64], return_tensors="pt", padding=True,
                  truncation=True)
            e = text_features(m, t)
            E.append((e / e.norm(dim=-1, keepdim=True)).numpy())
    return np.vstack(E).astype(np.float64)


def main():
    print("RAW PAIRS, room0 (judge for yourself)\n")
    m0 = json.load(open("collab_tasks/table3/mturk/room0.json"))
    o0 = json.load(open("student_gpu_package/handoff/room0_cgfront/cg_objects.json"))
    b0 = {int(o["id"]): o["cls"] for o in o0}
    shown = 0
    for r in m0:
        if "invalid" in str(r["caption"]).lower():
            continue
        i = int(r["id"])
        hn = head_noun(r["caption"])
        print(f"  id {i:>3} | said: {str(hn):<16} | ours: {b0.get(i,'--'):<16}"
              f" | {r['caption'][:56]}")
        shown += 1
        if shown >= 18:
            break

    print("\n\nAGREEMENT FOUR WAYS\n")
    print(f"{'scene':<9}{'n':>5}{'string':>9}{'synonym':>10}{'CLIP top1':>11}"
          f"{'chance':>9}{'best offset':>13}")
    print("-" * 66)
    tot = {"n": 0, "s": 0, "y": 0, "c": 0, "ch": 0.0}
    for s in SCENES:
        mt = json.load(open(f"collab_tasks/table3/mturk/{s}.json"))
        op = f"student_gpu_package/handoff/{s}_cgfront/cg_objects.json"
        objs = json.load(open(op))
        by = {int(o["id"]): str(o["cls"]) for o in objs}
        classes = sorted({str(o["cls"]) for o in objs})
        rows = [r for r in mt
                if isinstance(r.get("caption"), str)
                and "invalid" not in r["caption"].lower()
                and int(r["id"]) in by]
        if not rows:
            continue
        caps = [r["caption"] for r in rows]
        CE = clip_text([f"an image of a {c}" for c in classes])
        QE = clip_text(caps)
        sims = QE @ CE.T
        top = [classes[i] for i in sims.argmax(1)]

        ns = sum(1 for r in rows
                 if (head_noun(r["caption"]) or "") and
                 (head_noun(r["caption"]) in by[int(r["id"])].lower()
                  or by[int(r["id"])].lower() in (head_noun(r["caption"]) or "")))
        ny = sum(1 for r in rows if norm(head_noun(r["caption"])) == by[int(r["id"])])
        nc = sum(1 for r, t in zip(rows, top) if t == by[int(r["id"])])
        # chance = probability of hitting the right class by drawing from the
        # scene's own class distribution
        from collections import Counter
        cnt = Counter(by[int(r["id"])] for r in rows)
        ch = sum((v / len(rows)) ** 2 for v in cnt.values())

        best_off, best_hit = 0, -1
        for off in range(-6, 7):
            h = 0
            for r, t in zip(rows, top):
                j = int(r["id"]) + off
                if j in by and t == by[j]:
                    h += 1
            if h > best_hit:
                best_hit, best_off = h, off
        tot["n"] += len(rows); tot["s"] += ns; tot["y"] += ny
        tot["c"] += nc; tot["ch"] += ch * len(rows)
        print(f"{s:<9}{len(rows):>5}{100*ns/len(rows):>8.0f}%"
              f"{100*ny/len(rows):>9.0f}%{100*nc/len(rows):>10.0f}%"
              f"{100*ch:>8.0f}%{f'{best_off:+d} ({100*best_hit/len(rows):.0f}%)':>13}")
    n = tot["n"]
    print("-" * 66)
    print(f"{'ALL':<9}{n:>5}{100*tot['s']/n:>8.0f}%{100*tot['y']/n:>9.0f}%"
          f"{100*tot['c']/n:>10.0f}%{100*tot['ch']/n:>8.0f}%")
    print(f"\nCLIP top-1 {100*tot['c']/n:.0f}% against chance "
          f"{100*tot['ch']/n:.0f}%.")
    if tot["c"] / n < tot["ch"] / n * 1.8:
        print("=> Still at or near chance under a vocabulary-free test.")
        print("   The ids do NOT index our objects. Conclusion stands, and it")
        print("   now survives the synonym objection.")
    else:
        print("=> ABOVE chance: the ids may join after all. Re-open this.")


if __name__ == "__main__":
    main()
