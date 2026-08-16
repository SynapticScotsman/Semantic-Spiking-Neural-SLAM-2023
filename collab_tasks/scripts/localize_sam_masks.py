"""SAM-from-boxes re-localisation — targets the measured placement defect.

object_grounding.localize() takes the depth MEDIAN OVER THE WHOLE BOX and
shoots the ray through the BOX CENTRE:

    patch = d[v1:v2, u1:u2]          # box, so wall/floor behind the object too
    z     = median(valid)            # thin object vs far wall -> wall wins
    u, v  = box centre

Measured consequence on room0 (open-vocab run, mAcc 0.149): distance-to-truth
is bimodal — book 0.02 m, chair 0.02 m, cabinet 0.19 m, but vase 4.42 m,
wall-plug 4.75 m, switch 5.38 m. That is the signature of "sometimes we
measured the object, sometimes the wall behind it".

This script prompts SAM with each EXISTING box (no re-detection, no change of
vocabulary, no new model of the scene) and then uses only the mask's own
pixels:

    z     = median(depth[mask & valid])
    u, v  = mask centroid

Everything else in the pipeline is untouched, so any score change is
attributable to placement alone. Writes a parallel namespace
outputs/replica_<scene>_openvocab_sam/ — the 0.149 open-vocab artifacts and
the 0.091 baseline are never written to.

    python collab_tasks/scripts/localize_sam_masks.py --scene room0

Then score it exactly like any other frontend variant:

    python student_gpu_package/04_vsa_labels.py \
        --scene room0_openvocab_sam --gt-scene room0
    python student_gpu_package/05_score.py --scene room0_openvocab_sam
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    FX, FY, CX, CY, DEPTH_SCALE, IMG_WH)


def clone_namespace(scene, src_suffix="_openvocab", dst_suffix="_openvocab_sam"):
    """Copy the parts of the source run that are NOT position-dependent:
    the detections themselves and the cached CLIP labels for those crops.
    Same boxes, same names — only where they land in 3D will differ, which is
    the whole point of the experiment."""
    src = f"outputs/replica_{scene}{src_suffix}"
    dst = f"outputs/replica_{scene}{dst_suffix}"
    if not os.path.exists(os.path.join(src, "detections_crops.csv")):
        raise SystemExit(f"missing {src}/detections_crops.csv — run B1 first "
                         f"(collab_tasks/scripts/embed_crops_openvocab.py "
                         f"--scene {scene})")
    os.makedirs(dst, exist_ok=True)
    for f in ("detections_crops.csv", "crop_clip_replica.pt",
              "crop_embeddings_openvocab.pt"):
        p = os.path.join(src, f)
        if os.path.exists(p) and not os.path.exists(os.path.join(dst, f)):
            shutil.copy(p, dst)
            print(f"  reused {f}")
    cfg_src = f"vsa_cognitive_mapping/configs/replica_{scene}{src_suffix}.json"
    cfg_dst = f"vsa_cognitive_mapping/configs/replica_{scene}{dst_suffix}.json"
    cfg = json.load(open(cfg_src))
    cfg["name"] = f"{cfg['name']}_sam"
    cfg["out_dir"] = dst
    json.dump(cfg, open(cfg_dst, "w"), indent=2)
    print(f"  wrote {cfg_dst}")
    return src, dst, cfg


def load_sam(variant):
    """Ultralytics carries both SAM variants and is already a repo dep.
    mobile_sam is ~40 MB and enough for box-prompted masks; sam_b/sam_l are
    heavier and only worth it if mobile_sam's masks look poor."""
    from ultralytics import SAM
    ckpt = {"mobile_sam": "mobile_sam.pt", "sam_b": "sam_b.pt",
            "sam_l": "sam_l.pt"}[variant]
    print(f"loading SAM checkpoint {ckpt} (downloads once)")
    return SAM(ckpt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--sam", default="mobile_sam",
                    choices=["mobile_sam", "sam_b", "sam_l"])
    ap.add_argument("--min-mask-px", type=int, default=40,
                    help="masks smaller than this fall back to the box "
                         "(reported, never silent)")
    args = ap.parse_args()
    scene = args.scene

    print(f"=========== SAM-from-boxes: {scene} ===========")
    src, dst, cfg = clone_namespace(scene)

    img_dir = os.path.dirname(cfg["images"])
    traj = np.loadtxt(os.path.join(img_dir, "traj.txt")).reshape(-1, 4, 4)
    T = traj[:, :3, 3]
    var = T.var(axis=0)
    a, b = sorted(np.argsort(var)[-2:])
    print(f"floor axes {'xyz'[a]},{'xyz'[b]}")

    rows = list(csv.DictReader(open(os.path.join(src, "detections_crops.csv"),
                                    newline="")))
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[int(r["frame_idx"])].append(r)
    print(f"{len(rows)} detections over {len(by_frame)} frames")

    model = load_sam(args.sam)
    from PIL import Image

    pts, n_mask, n_fallback, n_skip = [], 0, 0, 0
    t0 = time.time()
    for k, fi in enumerate(sorted(by_frame)):
        dp = os.path.join(img_dir, f"depth{fi:06d}.png")
        ip = os.path.join(img_dir, f"frame{fi:06d}.jpg")
        if not (os.path.exists(dp) and os.path.exists(ip)):
            n_skip += len(by_frame[fi])
            continue
        d = np.asarray(Image.open(dp), np.float32) / DEPTH_SCALE
        assert d.shape[1] == IMG_WH[0] and d.shape[0] == IMG_WH[1], \
            f"unexpected render size {d.shape} — intrinsics constants are wrong"

        frows = by_frame[fi]
        boxes = [[float(r["x1"]), float(r["y1"]),
                  float(r["x2"]), float(r["y2"])] for r in frows]
        # one image-encoder pass for the whole frame, all boxes prompted at once
        res = model(ip, bboxes=boxes, verbose=False)[0]
        masks = (res.masks.data.cpu().numpy().astype(bool)
                 if res.masks is not None else np.zeros((0,) + d.shape, bool))

        for i, r in enumerate(frows):
            x1, y1 = float(r["x1"]), float(r["y1"])
            x2, y2 = float(r["x2"]), float(r["y2"])
            m = masks[i] if i < len(masks) else None
            use_mask = m is not None and m.shape == d.shape and m.sum() >= args.min_mask_px
            if use_mask:
                sel = m & (d > 0.1) & (d < 10.0)
                if sel.sum() < args.min_mask_px:
                    use_mask = False
            if use_mask:
                z = float(np.median(d[sel]))
                vv, uu = np.nonzero(m)
                u, v = float(uu.mean()), float(vv.mean())
                n_mask += 1
            else:
                # identical to object_grounding.localize(): box median + centre
                u1, u2 = max(0, int(x1)), min(d.shape[1], int(x2))
                v1, v2 = max(0, int(y1)), min(d.shape[0], int(y2))
                if u2 <= u1 or v2 <= v1:
                    n_skip += 1
                    continue
                patch = d[v1:v2, u1:u2]
                valid = patch[(patch > 0.1) & (patch < 10.0)]
                if valid.size < max(10, 0.02 * patch.size):
                    n_skip += 1
                    continue
                z = float(np.median(valid))
                u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                n_fallback += 1

            p_cam = np.array([(u - CX) / FX * z, (v - CY) / FY * z, z])
            M = traj[fi]
            p_world = M[:3, :3] @ p_cam + M[:3, 3]
            pts.append(dict(frame=fi, cls=r["class_name"],
                            conf=float(r["confidence"]), det=int(r["det_id"]),
                            x=float(p_world[a]), y=float(p_world[b])))
        if k % 100 == 0:
            el = (time.time() - t0) / 60
            print(f"  frame {k}/{len(by_frame)}  placed={len(pts)}  {el:.1f} min",
                  flush=True)

    if not pts:
        raise SystemExit("no detections placed — SAM returned no usable masks")

    out = os.path.join(dst, "object_points.json")
    with open(out, "w") as f:
        json.dump(dict(axes=["xyz"[a], "xyz"[b]], points=pts), f)
    tot = n_mask + n_fallback
    print(f"\nplaced {tot}/{len(rows)} detections "
          f"({n_mask} from SAM masks, {n_fallback} fell back to the box, "
          f"{n_skip} unusable)")
    print(f"mask coverage {n_mask/max(tot,1):.1%} -- a low number here means the "
          "experiment did not really test the hypothesis")
    print(f"wrote {out}  ({(time.time()-t0)/60:.1f} min)")
    print("\nSTAGE OK -- now score it:\n")
    print(f"python student_gpu_package/04_vsa_labels.py "
          f"--scene {scene}_openvocab_sam --gt-scene {scene}")
    print(f"python student_gpu_package/05_score.py --scene {scene}_openvocab_sam")
    print("\nCompare against the same-frontend box-median run: mAcc 0.149. "
          "Only the placement rule differs.")


if __name__ == "__main__":
    main()
