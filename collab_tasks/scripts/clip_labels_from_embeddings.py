"""Materialise crop_clip_replica.pt for every scene from the cached embeddings.

04_vsa_labels.clip_relabel() encodes crops with CLIP and caches
{det_id -> class_name} to outputs/replica_<scene>/crop_clip_replica.pt. The
embeddings are now precomputed (embed_crops_clip.py), so the same labels are
one matmul instead of a 6-minute encode per scene. Byte-for-byte the same
computation: same model, same 'an image of a {c}' template, same L2 norm,
same argmax over the same Replica class list.

room0 already has a cache built by the encode path, so it is used as a
CORRECTNESS GATE: regenerate it here and require exact agreement before
writing any of the other seven.

    .venv_lejepa/Scripts/python.exe collab_tasks/scripts/clip_labels_from_embeddings.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]


def labels_for(scene):
    import numpy as np
    z = np.load(f"outputs/replica_{scene}/crop_clip_embeddings.npz",
                allow_pickle=True)
    V, T, det = z["clip_ft"], z["text_ft"], z["det_id"]
    classes = [str(c) for c in z["class_names"]]
    best = (V @ T.T).argmax(1)
    return {int(d): classes[int(b)] for d, b in zip(det, best)}


def main():
    import torch

    # ---- gate on room0, whose cache came from the independent encode path --
    ref_p = "outputs/replica_room0/crop_clip_replica.pt"
    new = labels_for("room0")
    if os.path.exists(ref_p):
        ref = torch.load(ref_p, map_location="cpu", weights_only=False)
        shared = set(ref) & set(new)
        agree = sum(1 for k in shared if ref[k] == new[k])
        frac = agree / max(len(shared), 1)
        print(f"GATE room0: {len(shared)} shared det_ids, agreement {frac:.6f}")
        if frac < 1.0:
            raise SystemExit(
                f"HARD STOP: embeddings do not reproduce the encode-path "
                f"labels ({agree}/{len(shared)}). Do not write caches.")
        print("  exact match -- embedding path is equivalent to the encoder\n")

    for s in SCENES:
        p = f"outputs/replica_{s}/crop_clip_replica.pt"
        lab = new if s == "room0" else labels_for(s)
        if os.path.exists(p) and s == "room0":
            print(f"{s}: cache already correct, left untouched "
                  f"({len(lab)} crops)")
            continue
        torch.save(lab, p)
        from collections import Counter
        top = Counter(lab.values()).most_common(3)
        print(f"{s}: wrote {len(lab)} labels, "
              f"{len(set(lab.values()))} distinct classes; top: "
              + ", ".join(f"{c}={n}" for c, n in top))


if __name__ == "__main__":
    main()
