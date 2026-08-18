"""Show, in pixels, that ConceptGraphs' caption ids do not index our objects.

The numeric argument was: head-noun agreement 2%, synonym-aware 1%, CLIP
top-1 6% against a 21% chance baseline. Below chance. But a percentage is easy
to wave away, so this renders the thing itself.

For each caption id, find OUR object with that same id in cg_objects.json,
project its own observed points into the camera trajectory, and crop the frame
that sees it best. Put the human caption next to it.

If the ids joined, the crop would show what the caption describes. Whatever
comes out is the answer.

    python collab_tasks/table3/show_mturk_mismatch.py --scene room0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

W, H = 1200, 680
FX = FY = 600.0
CX, CY = 599.5, 339.5


def project(pts_w, c2w):
    """Verified convention from show_me_the_object.py: traj rows are 4x4
    camera-to-world already in OpenCV axes."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (pts_w - t) @ R
    z = cam[:, 2]
    valid = z > 1e-6
    zz = np.where(valid, z, 1.0)
    u = FX * (cam[:, 0] / zz) + CX
    v = FY * (cam[:, 1] / zz) + CY
    return u, v, zz, valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="outputs/table3/mturk_mismatch.jpg")
    args = ap.parse_args()

    mt = json.load(open(f"collab_tasks/table3/mturk/{args.scene}.json"))
    objs = json.load(open(
        f"student_gpu_package/handoff/{args.scene}_cgfront/cg_objects.json"))
    obs = json.load(open(
        f"student_gpu_package/handoff/{args.scene}_cgfront/cg_observations.json"))
    poses = np.loadtxt(f"data/replica/{args.scene}/traj.txt").reshape(-1, 4, 4)
    by_cls = {int(o["id"]): str(o["cls"]) for o in objs}

    pts_of = {}
    for o in obs:
        pts_of.setdefault(int(o["obj"]), []).append(
            [float(o["x"]), float(o["y"]), float(o["z"])])

    rows = [r for r in mt
            if isinstance(r.get("caption"), str)
            and "invalid" not in r["caption"].lower()
            and int(r["id"]) in by_cls][:args.n]

    TH, TW = 260, 260
    tiles = []
    for r in rows:
        oid = int(r["id"])
        P = np.array(pts_of.get(oid, []), float)
        tile = Image.new("RGB", (TW, TH), (14, 16, 20))
        if len(P):
            best, bn = None, 0
            for i in range(0, len(poses), 5):
                _, _, _, ins = project(P, poses[i])
                if int(ins.sum()) > bn:
                    bn, best = int(ins.sum()), i
            if best is not None and bn > 0:
                fp = f"data/replica/{args.scene}/frame{best:06d}.jpg"
                if os.path.exists(fp):
                    im = Image.open(fp).convert("RGB")
                    u, v, _, ins = project(P, poses[best])
                    uu, vv = u[ins], v[ins]
                    pad = 70
                    x0 = max(0, uu.min() - pad); x1 = min(W, uu.max() + pad)
                    y0 = max(0, vv.min() - pad); y1 = min(H, vv.max() + pad)
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    half = max(130, (x1 - x0) / 2, (y1 - y0) / 2)
                    box = (int(max(0, cx - half)), int(max(0, cy - half)),
                           int(min(W, cx + half)), int(min(H, cy + half)))
                    c = im.crop(box)
                    d = ImageDraw.Draw(c, "RGBA")
                    for a_, b_ in zip(uu, vv):
                        d.ellipse([a_ - box[0] - 3, b_ - box[1] - 3,
                                   a_ - box[0] + 3, b_ - box[1] + 3],
                                  fill=(255, 61, 166, 150))
                    c = c.resize((TW, TH))
                    tile.paste(c, (0, 0))
        dd = ImageDraw.Draw(tile)
        dd.rectangle([0, TH - 62, TW, TH], fill=(5, 7, 10))
        cap = r["caption"]
        cap = cap[:44] + ("..." if len(cap) > 44 else "")
        dd.text((7, TH - 56), f"id {oid}", fill=(200, 205, 210))
        dd.text((7, TH - 42), f"they: {cap}", fill=(255, 61, 166))
        dd.text((7, TH - 26), f"ours: {by_cls[oid]}", fill=(124, 224, 192))
        dd.text((7, TH - 12), "pink dots = this object's own points",
                fill=(120, 128, 136))
        tiles.append(tile)

    cols = 4
    rowsn = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TW, rowsn * TH), (14, 16, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * TW, (i // cols) * TH))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sheet.save(args.out, quality=90)
    print(f"wrote {args.out}  ({len(tiles)} objects)")
    for r in rows:
        print(f"  id {int(r['id']):>3}  ours={by_cls[int(r['id'])]:<16} "
              f"they={r['caption'][:52]}")


if __name__ == "__main__":
    main()
