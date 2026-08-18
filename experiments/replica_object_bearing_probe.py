"""Gating check for the object-centric line on a real scene.

Before building any object file on Replica, measure the geometry the code would
have to encode. Per the 2026-08-17 handoff (section 5), azimuth-only recovery is
*exact* when the traversed set is near-constant elevation -- that is the same
reason the ring worked on CO3D. So the question is not "is 3D harder", it is
"does this trajectory actually leave the ring".

Measures, for every ground-truth instance in a scene:
  * elevation of the camera->object bearing, per observing frame
  * azimuth coverage of those bearings
  * observation count and range

No VSA here, no decoding. Just the geometry, so the encoder choice is made from
a measurement rather than an assumption.

Usage:
    .venv/Scripts/python.exe experiments/replica_object_bearing_probe.py --scene room0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "replica_object_bearing"


def load_poses(scene):
    """traj.txt: one flattened 4x4 camera-to-world matrix per frame."""
    t = np.loadtxt(ROOT / "data" / "replica" / scene / "traj.txt")
    T = t.reshape(-1, 4, 4)
    return T[:, :3, 3], T                      # camera centres, full poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--min-obs", type=int, default=20)
    args = ap.parse_args()

    cams, T = load_poses(args.scene)
    gt = json.loads((ROOT / "outputs" / f"replica_{args.scene}" /
                     "gt_instances.json").read_text())
    # Replica ships unlabelled geometry as class_id -1 (class_name 'undefined'): real objects with no semantic class. They must never enter a class list.
    inst = [g for g in gt["instances"] if g.get("cls") != "class_-1"]
    print(f"{args.scene}: {len(cams)} frames, {len(inst)} GT instances")

    # Which frames see which instance is not recorded. A distance cut is far too
    # loose -- room0 is ~6 m across, so every frame passes and frames facing the
    # opposite wall get counted as observations. Use a real frustum test instead:
    # transform the object point into the camera frame, require it in front of
    # the camera and projecting inside the image.
    W, H, fx, fy, cx, cy = 1200, 680, 600.0, 600.0, 599.5, 339.5
    Rwc, twc = T[:, :3, :3], T[:, :3, 3]
    rows = []
    for o in inst:
        p = np.array([o["x"], o["y"], o.get("z", 0.0)])
        # world -> camera:  X_c = R^T (p - t)
        Xc = np.einsum("fij,fj->fi", np.transpose(Rwc, (0, 2, 1)), p[None] - twc)
        # Replica/OpenGL convention: camera looks down -z
        zc = -Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * (Xc[:, 0] / zc) + cx
            v = -fy * (Xc[:, 1] / zc) + cy
        d = cams - p
        r = np.linalg.norm(d, axis=1)
        m = (zc > 0.3) & (zc < 8.0) & (u > 0) & (u < W) & (v > 0) & (v < H)
        if m.sum() < args.min_obs:
            continue
        dd = d[m] / r[m][:, None]
        el = np.degrees(np.arcsin(np.clip(dd[:, 2], -1, 1)))
        az = np.degrees(np.arctan2(dd[:, 1], dd[:, 0])) % 360
        cover = len(np.unique((az // 10).astype(int))) / 36
        rows.append({
            "cls": o["cls"], "n": int(m.sum()),
            "el_med": float(np.median(el)),
            "el_p5_p95": float(np.percentile(el, 95) - np.percentile(el, 5)),
            "el_range": float(el.max() - el.min()),
            "az_cover": float(cover),
            "range_m": [float(r[m].min()), float(r[m].max())],
        })

    rows.sort(key=lambda r: -r["el_p5_p95"])
    print(f"\n{len(rows)} instances with >= {args.min_obs} nearby frames")
    print(f"{'class':<16}{'n':>6}{'el med':>9}{'el p5-95':>10}{'el range':>10}{'az cov':>8}")
    for r in rows[:18]:
        print(f"{r['cls']:<16}{r['n']:>6}{r['el_med']:>9.1f}"
              f"{r['el_p5_p95']:>10.1f}{r['el_range']:>10.1f}{r['az_cover']:>8.2f}")

    sp = np.array([r["el_p5_p95"] for r in rows])
    cv = np.array([r["az_cover"] for r in rows])
    print(f"\nacross instances: elevation p5-95 spread  median {np.median(sp):.1f} deg, "
          f"max {sp.max():.1f}")
    print(f"                  azimuth coverage         median {np.median(cv):.2f}, "
          f"min {cv.min():.2f}")
    print("\nVerdict: a ring code is the right encoder iff the elevation spread is "
          "small AND azimuth coverage is broad.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.scene}_bearings.json").write_text(json.dumps(rows, indent=1))
    print("wrote", OUT / f"{args.scene}_bearings.json")


if __name__ == "__main__":
    main()
