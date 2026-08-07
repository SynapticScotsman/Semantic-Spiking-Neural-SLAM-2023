"""Stage 3: export ConceptGraphs' outputs to the handoff schema.

Reads their saved result (pkl/pkl.gz under cg_out/<scene>/), writes:
  handoff/<scene>/cg_objects.json       [{class, x, y, z, n_detections}]
  handoff/<scene>/cg_observations.json  per-frame observation stream
  handoff/<scene>/eval_points.npz       the eval point cloud (xyz + gt label)
  handoff/<scene>/cg_labels.npz         their per-point predicted labels

Their result format has varied between releases; this script handles the
documented MapObjectList pickle. If loading fails it prints what it found —
send that log to Paul rather than guessing.

    python 03_export_cg.py --scene room0
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import pickle

import numpy as np


def load_their_result(out_dir):
    cands = (glob.glob(os.path.join(out_dir, "**", "*.pkl.gz"), recursive=True)
             + glob.glob(os.path.join(out_dir, "**", "*.pkl"), recursive=True))
    if not cands:
        raise SystemExit(f"FAIL: no pkl under {out_dir}. Contents:\n"
                         + "\n".join(glob.glob(os.path.join(out_dir, "*"))))
    path = max(cands, key=os.path.getsize)
    print(f"loading {path}")
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        return pickle.load(f), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--eval-points", type=int, default=200_000,
                    help="subsample of the fused cloud used as eval points")
    args = ap.parse_args()

    res, src = load_their_result(os.path.join("cg_out", args.scene))
    # Documented layout: dict with 'objects' (MapObjectList) or the list itself
    objects = res.get("objects", res) if isinstance(res, dict) else res
    out_dir = os.path.join("handoff", args.scene)
    os.makedirs(out_dir, exist_ok=True)

    obj_rows, obs_rows = [], []
    pts_all, lab_all = [], []
    for oid, ob in enumerate(objects):
        # per their schema: 'pcd'/'points', 'class_name'/'class_id', 'conf',
        # 'image_idx' (frames the object was seen in)
        if hasattr(ob, "get"):
            pts = np.asarray(ob.get("pcd_np", ob.get("points",
                             getattr(ob.get("pcd", None), "points", []))))
            cname = ob.get("class_name", ob.get("most_common_class", "unknown"))
            frames = ob.get("image_idx", [])
        else:
            raise SystemExit(f"FAIL: object entry type {type(ob)} — send this "
                             f"log + {src} layout to Paul")
        if isinstance(cname, (list, np.ndarray)):
            cname = max(set(map(str, cname)), key=list(map(str, cname)).count)
        if len(pts) == 0:
            continue
        c = pts.mean(0)
        obj_rows.append(dict(id=oid, cls=str(cname), x=float(c[0]),
                             y=float(c[1]), z=float(c[2]),
                             n_points=int(len(pts))))
        for fr in list(frames)[:2000]:
            obs_rows.append(dict(frame=int(fr), obj=oid, cls=str(cname),
                                 x=float(c[0]), y=float(c[1]), z=float(c[2])))
        pts_all.append(pts)
        lab_all.append(np.full(len(pts), oid))

    with open(os.path.join(out_dir, "cg_objects.json"), "w") as f:
        json.dump(obj_rows, f, indent=1)
    with open(os.path.join(out_dir, "cg_observations.json"), "w") as f:
        json.dump(obs_rows, f)

    P = np.concatenate(pts_all)
    L = np.concatenate(lab_all)
    if len(P) > args.eval_points:
        sel = np.random.RandomState(0).choice(len(P), args.eval_points,
                                              replace=False)
        P, L = P[sel], L[sel]
    cls_of = {o["id"]: o["cls"] for o in obj_rows}
    names = np.array([cls_of[int(i)] for i in L])
    np.savez_compressed(os.path.join(out_dir, "cg_labels.npz"),
                        xyz=P.astype(np.float32), pred_class=names)
    print(f"{len(obj_rows)} objects, {len(obs_rows)} observations, "
          f"{len(P)} labelled points")
    print("NOTE: eval_points.npz (GT labels) is built by 04_vsa_labels.py "
          "from the semantic renders so BOTH systems are scored on the "
          "identical point set.")
    print("STAGE OK (03_export_cg)")


if __name__ == "__main__":
    main()
