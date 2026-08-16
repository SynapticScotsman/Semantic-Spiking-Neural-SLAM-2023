"""Show the actual pixels behind a class confusion, anchored to GROUND TRUTH.

Every "vent" and "cushion" in our confusion tables is a CLIP label on a SAM mask.
Reasoning about labelling failures using the labels produced by the system whose
labelling is in question is circular: if CLIP called a wall panel "vent" 7,890
times, the analysis inherits that error and presents it as a finding. Paul's
objection, 2026-08-16, and it is correct.

This cuts the loop. It does NOT use CLIP, SAM, or any predicted label. It takes
the ground-truth 3D points of a class from eval_points.npz, projects them into
the camera trajectory, finds the frames that actually saw that object, and crops
there. What comes out is the object as the camera saw it, so you can judge for
yourself whether "vent" is a sensible name for it and whether "cushion" and
"sofa" are really distinct things in that room.

    python collab_tasks/scripts/show_me_the_object.py --scene room0 \
        --classes vent sofa cushion --out outputs/crops

Camera: Replica's NICE-SLAM render config, fx = fy = 600, cx = 599.5,
cy = 339.5, 1200 x 680 — confirmed against the frame size on disk. traj.txt rows
are row-major 4x4 camera-to-world.

Needs only PIL and numpy, and the scene frames under data/replica/<scene>/.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

W, H = 1200, 680
FX = FY = 600.0
CX, CY = 599.5, 339.5


def project(pts_w, c2w):
    """World points -> pixel coords + depth, for one camera-to-world pose.

    traj.txt rows are 4x4 camera-to-world (confirmed: the translations lie inside
    the room's own bounds) and are ALREADY in OpenCV convention -- +z forward,
    +y down. No OpenGL flip is needed.

    Both wrong guesses were caught by LOOKING, not by reasoning:
      - assuming -z forward with a y flip put room0's sofa on the CEILING;
      - then adding the NICE-SLAM `c2w[:3, 1:3] *= -1` conversion dropped sofa
        visibility from 97% of points to 20%.
    Checked across all four conventions, this one puts 97.3% of the sofa in frame
    with a mean v of 341 on a 680-pixel image. Verify any change here the same
    way: project a class whose position is obvious and look at the crop.
    """
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (pts_w - t) @ R                      # R^T (p - t) for row vectors
    z = cam[:, 2]
    valid = z > 1e-6
    zz = np.where(valid, z, 1.0)
    u = FX * (cam[:, 0] / zz) + CX
    v = FY * (cam[:, 1] / zz) + CY
    inside = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, zz, inside


def best_frames(pts, poses, k, min_pts):
    """Frames that see the most of this object, nearest first among those."""
    scored = []
    for i, c2w in enumerate(poses):
        u, v, z, ins = project(pts, c2w)
        n = int(ins.sum())
        if n >= min_pts:
            scored.append((n, -float(z[ins].mean()), i))
    if not scored:
        return []
    scored.sort(key=lambda r: (-r[0], -r[1]))
    # spread the picks across the trajectory rather than taking k adjacent frames
    picked, used = [], []
    for n, negd, i in scored:
        if all(abs(i - j) > 40 for j in used):
            picked.append(i)
            used.append(i)
        if len(picked) == k:
            break
    return picked or [scored[0][2]]


def crop(scene, frame, pts, c2w, pad=70, box=384):
    fp = f"data/replica/{scene}/frame{frame:06d}.jpg"
    if not os.path.exists(fp):
        return None, None
    im = Image.open(fp).convert("RGB")
    u, v, z, ins = project(pts, c2w)
    if ins.sum() == 0:
        return None, None
    uu, vv = u[ins], v[ins]
    x0, x1 = max(0, uu.min() - pad), min(W, uu.max() + pad)
    y0, y1 = max(0, vv.min() - pad), min(H, vv.max() + pad)
    # keep it squareish and not absurdly small
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(box / 2, (x1 - x0) / 2, (y1 - y0) / 2)
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    c = im.crop((x0, y0, x1, y1))
    # mark where the GT points land, so the object is unambiguous
    from PIL import ImageDraw
    d = ImageDraw.Draw(c, "RGBA")
    for a, b in zip(uu, vv):
        d.ellipse([a - x0 - 4, b - y0 - 4, a - x0 + 4, b - y0 + 4],
                  fill=(255, 61, 166, 170))
    c.thumbnail((360, 360))
    return c, (int(x0), int(y0), int(x1), int(y1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--classes", nargs="*",
                    default=["vent", "sofa", "cushion", "rug", "indoor-plant"])
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--out", default="outputs/crops")
    ap.add_argument("--json", default=None,
                    help="also write base64 crops as JSON for artifact embedding")
    args = ap.parse_args()

    ep = f"student_gpu_package/handoff/{args.scene}_cgfront/eval_points.npz"
    E = np.load(ep, allow_pickle=True)
    xyz, gt = E["xyz"], E["gt_class"].astype(str)
    poses = np.loadtxt(f"data/replica/{args.scene}/traj.txt").reshape(-1, 4, 4)
    os.makedirs(args.out, exist_ok=True)
    print(f"{args.scene}: {len(poses)} poses, {len(xyz)} eval points\n")

    payload = {}
    for c in args.classes:
        m = gt == c
        if not m.any():
            print(f"{c:<14} not in this scene's GT")
            continue
        pts = xyz[m]
        frames = best_frames(pts, poses, args.per_class, args.min_pts)
        print(f"{c:<14} {int(m.sum()):>5} GT pts   seen in frames {frames}")
        payload[c] = dict(n_gt=int(m.sum()), frames=[], centroid=
                          [round(float(q), 2) for q in pts.mean(0)])
        for f in frames:
            im, bx = crop(args.scene, f, pts, poses[f])
            if im is None:
                continue
            fn = f"{args.out}/{args.scene}_{c.replace(' ', '_')}_{f:06d}.jpg"
            im.save(fn, quality=88)
            b = io.BytesIO()
            im.save(b, format="JPEG", quality=80)
            payload[c]["frames"].append(dict(
                frame=int(f), box=bx,
                b64=base64.b64encode(b.getvalue()).decode()))
            print(f"    frame {f:>4}  crop {bx}  -> {os.path.basename(fn)}")

    if args.json:
        json.dump(payload, open(args.json, "w"))
        kb = os.path.getsize(args.json) / 1024
        print(f"\nwrote {args.json} ({kb:.0f} kB of embedded crops)")
    print(f"\nCrops are GT-anchored: pink dots are the ground-truth eval points "
          f"of that\nclass, projected into the frame. No CLIP, no SAM, no "
          f"predicted label involved.")


if __name__ == "__main__":
    main()
