"""Show, in pixels, what class_-1 actually is.

class_-1 appears in gt_instances.json with replica_class_id = -1 and enormous
pixel counts (office0: 27.5 million). It carries ZERO eval points, so it never
touches the benchmark score, but it does leak into any class list built from
gt_instances.json, and it did leak into the Route A retrieval killer.

Rather than assert what it is, render it: take vMAP's own semantic_class_*.png
renders, mark every pixel whose class id is -1, and lay that over the matching
RGB frame. Whatever is highlighted IS class_-1.

Writes side-by-side PNGs to outputs/crops_minus1/.

    python collab_tasks/scripts/show_class_minus1.py --scene room0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

VMAP = {"room0": "room_0_vmap", "room1": "room_1_vmap", "room2": "room_2_vmap",
        "office0": "office_0_vmap", "office1": "office_1_vmap",
        "office2": "office_2_vmap", "office3": "office_3_vmap",
        "office4": "office_4_vmap"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", default="outputs/crops_minus1")
    args = ap.parse_args()

    vd = f"data/replica/{VMAP[args.scene]}"
    info = json.load(open(f"{vd}/info_semantic.json"))
    # vMAP's mapping: id_to_label[i] is the class id of instance i, -1 = void
    id2lab = info.get("id_to_label", info.get("classes"))
    print(f"info_semantic.json keys: {list(info.keys())}")
    if isinstance(id2lab, list):
        arr = np.array(id2lab)
        print(f"id_to_label: {len(arr)} entries, "
              f"{(arr == -1).sum()} map to -1 (void)")
    names = {c["id"]: c["name"] for c in info.get("classes", [])} \
        if "classes" in info else {}

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(vd) if f.startswith("semantic_class_"))
    step = max(1, len(files) // args.frames)
    picks = files[::step][:args.frames]

    tot_void = tot_px = 0
    for fn in picks:
        idx = int("".join(ch for ch in fn if ch.isdigit()))
        sem = np.array(Image.open(f"{vd}/{fn}"))
        # the vMAP semantic_class png stores class id directly; void shows as
        # 0 or 65535 depending on export, so treat both plus any id that maps
        # to -1 in id_to_label
        void = (sem == 0)
        if isinstance(id2lab, list):
            bad = {i for i, v in enumerate(np.array(id2lab)) if v == -1}
            if bad:
                void |= np.isin(sem, list(bad))
        frac = float(void.mean())
        tot_void += int(void.sum())
        tot_px += void.size

        # matching RGB frame: vMAP renders every 10th source frame
        rgb_i = idx * 10
        rp = f"data/replica/{args.scene}/frame{rgb_i:06d}.jpg"
        if not os.path.exists(rp):
            rp = f"data/replica/{args.scene}/frame{idx:06d}.jpg"
        if not os.path.exists(rp):
            print(f"  {fn}: void {100*frac:.1f}% (no matching RGB found)")
            continue
        rgb = Image.open(rp).convert("RGB").resize(
            (sem.shape[1], sem.shape[0]))
        a = np.array(rgb).astype(np.float32)
        overlay = a.copy()
        overlay[void] = overlay[void] * 0.25 + np.array([255, 61, 166]) * 0.75
        pair = np.concatenate([a, overlay], axis=1).astype(np.uint8)
        out = f"{args.out}/{args.scene}_{idx:04d}_void.jpg"
        Image.fromarray(pair).save(out, quality=88)
        print(f"  {fn}: void {100*frac:>5.1f}% of pixels  -> "
              f"{os.path.basename(out)}")

    print(f"\nacross the sampled frames, {100*tot_void/max(tot_px,1):.1f}% "
          f"of pixels are unlabelled")
    print("left = RGB, right = the same frame with unlabelled pixels in pink")
    print("\nWhatever is pink has NO semantic class in Replica's annotation.")


if __name__ == "__main__":
    main()
