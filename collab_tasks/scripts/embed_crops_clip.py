"""CLIP-embed every detection crop in all 8 Replica scenes. Resumable.

Produces the artifact that has never existed in this repo: per-crop CLIP
IMAGE EMBEDDINGS. What is on disk today is only crop_clip_replica.pt, which
is CLIP's argmax LABEL per crop (room0 only), and crop_embeddings.pt, which
is YOLOv8n neck features, not CLIP.

SCOPE, stated up front so no downstream table misuses this: these crops come
from OUR YOLO frontend, whose COCO vocabulary covers a fraction of Replica's
scored classes. These embeddings therefore sit beside our own standalone
pipeline's baseline. They are NOT comparable to the 0.324 head-to-head, which
is scored on ConceptGraphs' SAM+CLIP frontend stream.

Measured on this machine (Core Ultra 7 165H, 16c/22t): encode 40.4 crops/s at
22 threads, 27.2 at 4; serial image IO 124 crops/s. Combined ~22 crops/s at
4 threads => ~39 min for all 52,197 crops, ~18% CPU so the laptop stays
usable. Output ~13 MB/scene, ~107 MB total.

    .venv_lejepa/Scripts/python.exe collab_tasks/scripts/embed_crops_clip.py
    ... --threads 22          # full tilt, ~29 min, laptop pegged
    ... --scenes room0 room1  # subset
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]
MODEL = "openai/clip-vit-base-patch32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from vsa_cognitive_mapping.clip_compat import image_features, text_features
    from vsa_cognitive_mapping.sequences import load_sequence

    sys.path.insert(0, os.path.join(ROOT, "student_gpu_package"))
    import importlib
    v04 = importlib.import_module("04_vsa_labels")

    torch.set_num_threads(args.threads)
    print(f"CLIP {MODEL} | torch threads={args.threads} | "
          f"scenes={len(args.scenes)}", flush=True)
    model = CLIPModel.from_pretrained(MODEL).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)

    t_start = time.perf_counter()
    grand = 0
    for scene in args.scenes:
        out = f"outputs/replica_{scene}/crop_clip_embeddings.npz"
        if os.path.exists(out) and not args.force:
            print(f"{scene}: already done -> {out} (use --force to redo)",
                  flush=True)
            continue
        csv_p = f"outputs/replica_{scene}/detections_crops.csv"
        if not os.path.exists(csv_p):
            print(f"{scene}: no detections_crops.csv, skipping", flush=True)
            continue
        rows = list(csv.DictReader(open(csv_p)))
        seq = load_sequence(f"vsa_cognitive_mapping/configs/replica_{scene}.json")
        pos = {int(f): i for i, f in enumerate(seq.frame_ids())}

        # their vocabulary + their prompt template, so the text side is
        # directly comparable to 04_vsa_labels.clip_relabel
        classes, _ = v04.replica_class_list(scene)
        with torch.no_grad():
            tk = proc(text=[f"an image of a {c}" for c in classes],
                      return_tensors="pt", padding=True)
            T = text_features(model, tk)
            T = (T / T.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)

        V = np.zeros((len(rows), T.shape[1]), np.float32)
        det = np.array([int(r["det_id"]) for r in rows], np.int64)
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in range(0, len(rows), args.batch):
                chunk = rows[s:s + args.batch]
                imgs = []
                for r in chunk:
                    im = seq.image(pos[int(r["frame_idx"])])
                    imgs.append(im.crop((
                        max(0, float(r["x1"])), max(0, float(r["y1"])),
                        min(im.size[0], float(r["x2"])),
                        min(im.size[1], float(r["y2"])))))
                px = proc(images=imgs, return_tensors="pt")
                E = image_features(model, px)
                V[s:s + len(chunk)] = (
                    E / E.norm(dim=-1, keepdim=True)).numpy()
                if (s // args.batch) % 25 == 0:
                    el = time.perf_counter() - t0
                    rate = (s + len(chunk)) / max(el, 1e-9)
                    eta = (len(rows) - s - len(chunk)) / max(rate, 1e-9)
                    print(f"  {scene} {s+len(chunk):>6}/{len(rows)}  "
                          f"{rate:5.1f} crops/s  eta {eta/60:4.1f} min",
                          flush=True)
        dt = time.perf_counter() - t0
        tmp = out + ".tmp.npz"
        np.savez_compressed(tmp, det_id=det, clip_ft=V,
                            class_names=np.array(classes), text_ft=T,
                            model=MODEL, prompt="an image of a {c}")
        os.replace(tmp, out)
        grand += len(rows)
        print(f"{scene}: {len(rows)} crops in {dt/60:.1f} min "
              f"({len(rows)/dt:.1f}/s) -> {out} "
              f"({os.path.getsize(out)/1e6:.1f} MB)", flush=True)

    print(f"\nDONE: {grand} crops in "
          f"{(time.perf_counter()-t_start)/60:.1f} min total", flush=True)


if __name__ == "__main__":
    main()
