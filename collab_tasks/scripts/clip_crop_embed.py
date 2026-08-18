"""Cache CLIP crop EMBEDDINGS (not just the argmax label) for our own crops.

Why this exists. `student_gpu_package/04_vsa_labels.py:clip_relabel` already
runs exactly this forward pass, but keeps only `argmax` and throws the 512-d
features away (04_vsa_labels.py:96-102), so `crop_clip_replica.pt` is a
{det_id -> class-name} dict. Every appearance-channel experiment needs the
vector, and re-deriving it means re-running CLIP.

One pass, two artifacts, so nothing is ever encoded twice:

  outputs/replica_<scene>/crop_clip_embed.pt      NEW
      {"det_id": int64[N], "embedding": float16[N,512] (L2-normalised),
       "classes": [101 names], "text": float32[101,512] (L2-normalised),
       "argmax": int64[N] index into classes, "meta": {...}}
  outputs/replica_<scene>/crop_clip_replica.pt    same schema as stage 4's
      {det_id -> class name} -- written ONLY if absent, so room0's
      already-measured cache is never touched (pass --force to override).

Identical to clip_relabel by construction: same model
(openai/clip-vit-base-patch32), same prompt ("an image of a {c}"), same class
list (replica_class_list -> the scene's info_semantic.json, 101 names), same
L2-normalise-then-argmax. Verified: the B1 killer re-encoded 500 room0 crops
this way and reproduced the stored labels exactly
(outputs/routeB/clip_phasor_retention.json: "agree_with_stored": 1.0).

Run from the repo root with a torch+transformers interpreter:

    .venv_lejepa/Scripts/python.exe collab_tasks/scripts/clip_crop_embed.py \
        --scene room0 room1 room2 office0 office1 office2 office3 office4
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "student_gpu_package"))
os.chdir(ROOT)

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]


def run_scene(scene, model, proc, image_features, text_features, torch,
              batch=32, limit=0, force=False):
    from vsa_cognitive_mapping.sequences import load_sequence
    v04 = importlib.import_module("04_vsa_labels")

    out_dir = f"outputs/replica_{scene}"
    csv_path = os.path.join(out_dir, "detections_crops.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"FAIL: {csv_path} missing -- this scene has no crops; "
                         "run the embed-crops stage for it first")
    rows = list(csv.DictReader(open(csv_path)))
    if limit:
        rows = rows[:limit]

    # exactly stage 4's class list: sorted names from info_semantic.json
    classes, _ = v04.replica_class_list(scene)
    seq = load_sequence(f"vsa_cognitive_mapping/configs/replica_{scene}.json")
    pos = {int(f): i for i, f in enumerate(seq.frame_ids())}

    with torch.no_grad():
        t = proc(text=[f"an image of a {c}" for c in classes],
                 return_tensors="pt", padding=True)
        T = text_features(model, t)
        T = T / T.norm(dim=-1, keepdim=True)

    print(f"{scene}: {len(rows)} crops, {len(classes)} classes, "
          f"{len(seq)} frames", flush=True)
    V = np.empty((len(rows), T.shape[1]), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            imgs = []
            for r in chunk:
                img = seq.image(pos[int(r["frame_idx"])])
                imgs.append(img.crop((max(0, float(r["x1"])),
                                      max(0, float(r["y1"])),
                                      min(img.size[0], float(r["x2"])),
                                      min(img.size[1], float(r["y2"])))))
            px = proc(images=imgs, return_tensors="pt")
            E = image_features(model, px)
            V[s:s + len(chunk)] = (E / E.norm(dim=-1, keepdim=True)).numpy()
            if (s // batch) % 25 == 0:
                el = time.time() - t0
                rate = (s + len(chunk)) / max(el, 1e-9)
                print(f"  {s + len(chunk)}/{len(rows)}  {rate:.1f} crops/s  "
                      f"eta {(len(rows) - s - len(chunk)) / max(rate, 1e-9) / 60:.1f} min",
                      flush=True)

    Tn = T.numpy().astype(np.float32)
    argmax = (V @ Tn.T).argmax(1)
    det_id = np.array([int(r["det_id"]) for r in rows], np.int64)

    # A --limit run must never leave a PARTIAL artifact behind: stage 4 treats
    # crop_clip_replica.pt as a complete cache and would silently relabel only
    # the crops it happens to cover, dropping the rest at 04_vsa_labels.py:261.
    suffix = f".smoke{len(rows)}" if limit else ""
    emb_path = os.path.join(out_dir, f"crop_clip_embed{suffix}.pt")
    torch.save({"det_id": torch.from_numpy(det_id),
                "embedding": torch.from_numpy(V).half(),
                "classes": classes,
                "text": torch.from_numpy(Tn),
                "argmax": torch.from_numpy(argmax.astype(np.int64)),
                "meta": {"model": "openai/clip-vit-base-patch32",
                         "prompt": "an image of a {c}",
                         "normalised": True, "dim": int(V.shape[1]),
                         "class_source": "info_semantic.json",
                         "n_crops": len(rows)}}, emb_path)
    mb = os.path.getsize(emb_path) / 1e6

    lab_path = os.path.join(out_dir, "crop_clip_replica.pt")
    if limit:
        print(f"  --limit set: NOT writing {lab_path} (a partial label cache "
              "would silently truncate stage 4)")
    elif os.path.exists(lab_path) and not force:
        ref = torch.load(lab_path, weights_only=False)
        shared = [(d, a) for d, a in zip(det_id, argmax) if int(d) in ref]
        agree = (float(np.mean([ref[int(d)] == classes[int(a)]
                                for d, a in shared])) if shared else float("nan"))
        print(f"  {lab_path} exists ({len(ref)} labels) -- NOT overwritten; "
              f"agreement with this pass on {len(shared)} shared det_ids: "
              f"{agree:.4f}")
    else:
        torch.save({int(d): classes[int(a)] for d, a in zip(det_id, argmax)},
                   lab_path)
        print(f"  wrote {lab_path} ({len(det_id)} labels) -- stage 4 will now "
              f"use it as its cache")

    dt = (time.time() - t0) / 60
    print(f"  wrote {emb_path} ({mb:.1f} MB) -- {len(rows)} crops in "
          f"{dt:.1f} min ({len(rows) / max(dt * 60, 1e-9):.1f} crops/s)\n",
          flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", nargs="+", default=["room0"],
                    help=f"scene ids, or 'all' for {' '.join(SCENES)}")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="encode only the first N crops (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="also overwrite an existing crop_clip_replica.pt "
                         "(default: leave already-measured caches alone)")
    args = ap.parse_args()
    scenes = SCENES if args.scene == ["all"] else args.scene

    import torch
    from transformers import CLIPModel, CLIPProcessor

    from vsa_cognitive_mapping.clip_compat import image_features, text_features
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    if torch.cuda.is_available():
        print("NOTE: CUDA is available but this script is CPU-only by design "
              "(the whole 8-scene sweep is under an hour on CPU).")
    print(f"CLIP ViT-B/32 on cpu, {torch.get_num_threads()} threads")

    for s in scenes:
        run_scene(s, model, proc, image_features, text_features, torch,
                  batch=args.batch, limit=args.limit, force=args.force)
    print("STAGE OK (clip_crop_embed)")


if __name__ == "__main__":
    main()
