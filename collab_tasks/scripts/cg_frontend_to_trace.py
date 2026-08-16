"""B2 rung 2 — ConceptGraphs' frontend feeding OUR memory.

Rung 1 (system vs system) compares two whole pipelines, so a difference could
come from either the frontend or the backend. This is the isolating run: take
THEIR observation stream — the objects their SAM+CLIP frontend produced, before
their 3D fusion is consulted for anything — and bundle it into our 32 KB trace
instead of our YOLO detections. Both backends then consume byte-identical
frontend output and the only remaining difference is the memory.

    ours     : YOLO/YOLO-World detections -> our trace   (mAcc 0.099 / 0.149)
    theirs   : SAM+CLIP objects           -> their map   (mAcc 0.290)
    THIS RUN : SAM+CLIP objects           -> our trace   (the missing cell)

Input is handoff/<scene>/cg_observations.json, written by 03_export_cg.py.
Output is a parallel namespace outputs/replica_<scene>_cgfront/ so the measured
baseline artifacts are never touched.

    python collab_tasks/scripts/cg_frontend_to_trace.py --scene room0

then score it with the same two stages everything else uses:

    python student_gpu_package/04_vsa_labels.py \
        --scene room0_cgfront --gt-scene room0 --labels-from-points
    python student_gpu_package/05_score.py --scene room0_cgfront

HONEST LIMITATION, stated up front: 03_export_cg.py records each observation at
its OBJECT's centroid, not at a per-observation 3D position — their per-frame
detection geometry is not preserved in that export. So this measures "their
objects, their labels, our bundling+decode", not "their raw per-frame
observations through our placement". That is still the backend comparison, but
it is not a frontend-for-frontend swap at the observation level.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PKG = "student_gpu_package"


def floor_axes(scene):
    """Which two world axes are the floor plane.

    Our baseline already recorded its answer in object_points.json's "axes"
    field, so prefer that: it is authoritative, free, and survives a VM reset
    (it lives on Drive). Only fall back to recomputing from traj.txt — the
    same rule object_grounding.localize() uses — when that file is absent,
    because raw scene data is an 11 GB re-download on a fresh VM.

    Getting this wrong would silently rotate their map relative to ours, and
    the damage would look like a scoring difference rather than a bug.
    """
    cfg = json.load(open(f"vsa_cognitive_mapping/configs/replica_{scene}.json"))
    recorded = f"outputs/replica_{scene}/object_points.json"
    if os.path.exists(recorded):
        axes = json.load(open(recorded)).get("axes")
        if axes and len(axes) == 2:
            a, b = ("xyz".index(axes[0]), "xyz".index(axes[1]))
            print(f"floor axes {axes[0]},{axes[1]} (as recorded by our baseline "
                  f"in {recorded})")
            return a, b, cfg
    img_dir = os.path.dirname(cfg["images"])
    traj_path = os.path.join(img_dir, "traj.txt")
    if not os.path.exists(traj_path):
        raise SystemExit(
            f"cannot determine floor axes: neither {recorded} nor {traj_path} "
            "exists. The first is on Drive if our baseline ever ran; the "
            "second needs tools/prepare_replica.py (11 GB).")
    traj = np.loadtxt(traj_path).reshape(-1, 4, 4)
    var = traj[:, :3, 3].var(axis=0)
    a, b = sorted(np.argsort(var)[-2:])
    print(f"floor axes {'xyz'[a]},{'xyz'[b]} (recomputed from traj.txt)")
    return a, b, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--suffix", default="_cgfront")
    args = ap.parse_args()
    scene = args.scene
    dst_scene = f"{scene}{args.suffix}"

    src = f"{PKG}/handoff/{scene}/cg_observations.json"
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} — run 03_export_cg.py --scene {scene} "
                         "first (it needs their pkl.gz in cg_out/)")
    obs = json.load(open(src))
    if not obs:
        raise SystemExit(f"{src} is empty — their export produced no observations")

    a, b, cfg = floor_axes(scene)

    pts = []
    for i, o in enumerate(obs):
        xyz = [o["x"], o["y"], o["z"]]
        pts.append(dict(frame=int(o["frame"]), cls=str(o["cls"]),
                        conf=1.0, det=i,
                        x=float(xyz[a]), y=float(xyz[b])))
    classes = sorted({p["cls"] for p in pts})
    print(f"{len(pts)} observations over {len(classes)} classes from "
          f"{len({o['obj'] for o in obs})} of their objects")

    out_dir = f"outputs/replica_{dst_scene}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "object_points.json"), "w") as f:
        json.dump(dict(axes=["xyz"[a], "xyz"[b]], points=pts), f)

    # clone the sequence config so downstream stages resolve paths the same way
    cfg["name"] = f"{cfg['name']}{args.suffix}"
    cfg["out_dir"] = out_dir
    cfg_path = f"vsa_cognitive_mapping/configs/replica_{dst_scene}.json"
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    print(f"wrote {out_dir}/object_points.json and {cfg_path}")
    print("\nSTAGE OK -- now score it:\n")
    print(f"python {PKG}/04_vsa_labels.py --scene {dst_scene} "
          f"--gt-scene {scene} --labels-from-points")
    print(f"python {PKG}/05_score.py --scene {dst_scene}")
    print("\nCompare: their frontend+their backend was mAcc 0.290; our frontend"
          "+our backend 0.099. This run is their frontend + our backend.")


if __name__ == "__main__":
    main()
