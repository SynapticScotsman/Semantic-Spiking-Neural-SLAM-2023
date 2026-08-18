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
import gc
import glob
import gzip
import json
import os
import pickle

import numpy as np

# anchor to this package dir like stages 4-5, so cg_out/handoff resolve the
# same regardless of the caller's working directory (notebook runs from the
# repo root, the README runs from student_gpu_package/)
PKG = os.path.dirname(os.path.abspath(__file__))


def _rss(tag):
    """Print resident memory. Colab OOM-kills silently, so the last line
    printed before death is the only diagnostic available."""
    cur = ""
    try:
        with open("/proc/self/statm") as f:
            cur = f" current {int(f.read().split()[1]) * 4096 / 1048576:.0f} MB"
    except Exception:
        pass
    try:
        import resource
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[mem] {tag}: peak {mb:.0f} MB{cur}", flush=True)
    except Exception:
        pass


def load_their_result(out_dir, prefer_post=True):
    """Load their saved map, preferring the POST-PROCESSED one.

    cfslam_pipeline_batch.py saves two maps per scene:

        full_pcd_<suffix>.pkl.gz        the map as accumulated
        full_pcd_<suffix>_post.pkl.gz   after their final denoise/merge pass

    Selecting by max(getsize) picked the WRONG one every time, because the post
    pass REMOVES objects and points and so the _post file is the smaller of the
    two (room1: 23.0 MB vs 22.5 MB). That means every score reported for their
    system up to 2026-08-15 was computed on their map with its last stage
    missing -- an unfair comparison, and unfair in the direction that flatters
    us. Suspected contributor to our measured 0.271 mean against their
    published ~0.40.

    prefer_post=False reproduces the old behaviour for an explicit A/B.
    """
    cands = (glob.glob(os.path.join(out_dir, "**", "*.pkl.gz"), recursive=True)
             + glob.glob(os.path.join(out_dir, "**", "*.pkl"), recursive=True))
    if not cands:
        raise SystemExit(f"FAIL: no pkl under {out_dir}. Contents:\n"
                         + "\n".join(glob.glob(os.path.join(out_dir, "*"))))
    post = [p for p in cands if "_post." in os.path.basename(p)]
    if prefer_post and post:
        path = max(post, key=os.path.getsize)
        other = [p for p in cands if p not in post]
        if other:
            print(f"NOTE: using their POST-PROCESSED map "
                  f"({os.path.getsize(path)/1e6:.1f} MB); the un-post-processed "
                  f"one is {os.path.getsize(max(other, key=os.path.getsize))/1e6:.1f} MB")
    else:
        path = max(cands, key=os.path.getsize)
        if prefer_post:
            print("NOTE: no _post map found — scoring the un-post-processed map")
    print(f"loading {path}")
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        return pickle.load(f), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--eval-points", type=int, default=200_000,
                    help="subsample of the fused cloud used as eval points")
    ap.add_argument("--cg-out", default=None,
                    help="dir holding their pkl.gz (default <pkg>/cg_out/<scene>)")
    ap.add_argument("--obs-geometry", default="sampled",
                    choices=["sampled", "centroid"],
                    help="where each observation in cg_observations.json sits. "
                         "'centroid' (the original behaviour) puts every "
                         "observation of an object at that object's centre, so "
                         "110 objects collapse to 110 distinct positions — a "
                         "downstream memory then has no extent to work with, "
                         "while their own map is scored on full per-object "
                         "point clouds. 'sampled' draws each observation's "
                         "position from the object's own pcd_np, preserving "
                         "extent at the same observation count.")
    ap.add_argument("--no-in-scene", action="store_true",
                    help="assign their labels over the FULL class list instead "
                         "of only classes present in the scene GT. Their eval "
                         "restricts to in-scene classes; this reproduces our "
                         "pre-2026-08-16 behaviour for an explicit A/B.")
    ap.add_argument("--no-post", action="store_true",
                    help="score their UN-post-processed map, reproducing the "
                         "behaviour before 2026-08-15. Only for an explicit A/B "
                         "measuring what their post pass is worth — every "
                         "reported comparison should use the post map, which is "
                         "the one their own evaluation consumes.")
    args = ap.parse_args()

    res, src = load_their_result(args.cg_out
                                 or os.path.join(PKG, "cg_out", args.scene),
                                 prefer_post=not args.no_post)
    # Documented layout: dict with 'objects' (MapObjectList) or the list itself
    objects = res.get("objects", res) if isinstance(res, dict) else res
    out_dir = os.path.join(PKG, "handoff", args.scene)
    os.makedirs(out_dir, exist_ok=True)

    def sniff(ob, keys, default):
        """Try many key spellings; works for dicts and attribute objects."""
        for k in keys:
            if hasattr(ob, "get") and ob.get(k) is not None:
                return ob.get(k)
            if hasattr(ob, k) and getattr(ob, k) is not None:
                return getattr(ob, k)
        return default

    obj_rows, obs_rows, feats = [], [], []
    pts_all, lab_all = [], []
    for oid, ob in enumerate(objects):
        # schema drifts between releases — sniff the known spellings and,
        # on total failure, print what IS there instead of guessing
        raw_pts = sniff(ob, ["pcd_np", "points", "pcd_points"], None)
        if raw_pts is None:
            pcd = sniff(ob, ["pcd"], None)
            raw_pts = getattr(pcd, "points", None)
        pts = np.asarray(raw_pts) if raw_pts is not None else np.empty((0, 3))
        cname = sniff(ob, ["class_name", "most_common_class", "object_tag",
                           "caption", "class_id"], "unknown")
        frames = sniff(ob, ["image_idx", "frame_idx", "obs_frames"], [])
        if oid == 0:
            avail = (list(ob.keys()) if hasattr(ob, "keys")
                     else [a for a in dir(ob) if not a.startswith("_")])
            print(f"first object's available keys/attrs: {avail[:30]}")
        if pts.size == 0 and oid == 0:
            raise SystemExit("FAIL: cannot find point data on their objects — "
                             "send the printed key list above to Paul")
        if isinstance(cname, (list, np.ndarray)):
            cname = max(set(map(str, cname)), key=list(map(str, cname)).count)
        if len(pts) == 0:
            continue
        c = pts.mean(0)
        obj_rows.append(dict(id=oid, cls=str(cname), x=float(c[0]),
                             y=float(c[1]), z=float(c[2]),
                             n_points=int(len(pts))))
        feats.append(sniff(ob, ["clip_ft"], None))
        frs = list(frames)[:2000]
        if args.obs_geometry == "sampled" and len(frs):
            # one position per observation, drawn from this object's own cloud,
            # so the stream carries the object's extent rather than collapsing
            # it to a point. Seeded per object => reproducible across runs.
            sel = np.random.RandomState(oid).choice(
                len(pts), size=len(frs), replace=len(pts) < len(frs))
            for k, fr in enumerate(frs):
                q = pts[sel[k]]
                obs_rows.append(dict(frame=int(fr), obj=oid, cls=str(cname),
                                     x=float(q[0]), y=float(q[1]), z=float(q[2])))
        else:
            for fr in frs:
                obs_rows.append(dict(frame=int(fr), obj=oid, cls=str(cname),
                                     x=float(c[0]), y=float(c[1]), z=float(c[2])))
        pts_all.append(pts.astype(np.float32, copy=False))
        lab_all.append(np.full(len(pts), oid, dtype=np.int32))

    # A class-agnostic map (class_set none / class_agnostic=True) carries no
    # semantics — every object's class_name is literally "item", so reading it
    # exported one class for every point and scored mAcc 0.000. Their own eval
    # (conceptgraph/scripts/eval_replica_semseg.py, lines 283-296 and 134-141)
    # assigns labels by matching each object's clip_ft against CLIP text
    # embeddings of the class list. Reproduced here verbatim — same model
    # (ViT-H-14 / laion2b_s32b_b79k), same prompt, same L2 normalisation, same
    # -1e10 suppression of the excluded classes, same argmax. Nothing invented.
    _rss("after object loop")
    # Everything needed downstream is already extracted into obj_rows /
    # obs_rows / feats / pts_all. Their map object holds per-detection masks
    # and crops; freeing it here is what makes the CLIP relabel survive on a
    # standard Colab runtime.
    del res, objects
    gc.collect()
    _rss("after freeing their map")

    if len({r["cls"] for r in obj_rows}) <= 1 and feats and feats[0] is not None:
        import torch

        # ---- cache FIRST ----------------------------------------------
        # The cache holds BOTH the class list and its text embeddings, and
        # neither depends on the scene. Checking it before any import means a
        # cache hit needs neither `conceptgraph` NOR ViT-H-14 -- which matters
        # because both have failed mid-run: the import raised
        # "No module named 'conceptgraph'" on one scene while succeeding on
        # its neighbours, and loading the model per scene is what OOM-kills a
        # standard Colab runtime.
        _cache = os.path.join(PKG, "handoff", "cg_clip_text.npz")
        class_names, _tf_cached = None, None
        if os.path.exists(_cache):
            try:
                _z = np.load(_cache, allow_pickle=True)
                class_names = [str(c) for c in _z["class_names"]]
                _tf_cached = _z["text_ft"]
                print(f"reusing cached class list + text embeddings "
                      f"({_tf_cached.shape}) -- conceptgraph and ViT-H-14 "
                      f"BOTH skipped", flush=True)
            except Exception as e:
                print(f"text cache unreadable ({e}) -- recomputing")
                class_names, _tf_cached = None, None

        if class_names is None:
            # Prefer THEIR module when importable, so an upstream change to
            # the class list is picked up. Fall back to the vendored copy --
            # derived from that same file and byte-identical in content and
            # ORDER, which is load-bearing because it fixes the argmax index
            # mapping. Needed because a recycled Colab runtime loses their
            # repo while ours survives, which stalled all 8 scenes.
            try:
                from conceptgraph.dataset.replica_constants import (
                    REPLICA_CLASSES, REPLICA_EXISTING_CLASSES)
                class_names = [REPLICA_CLASSES[i]
                               for i in REPLICA_EXISTING_CLASSES]
                print("class list: conceptgraph module")
            except Exception as e:
                _vend = os.path.join(PKG, "replica_class_names.json")
                if not os.path.exists(_vend):
                    raise SystemExit(
                        f"cannot import their class list ({e}) and no vendored "
                        f"copy at {_vend}")
                class_names = json.load(open(_vend))["class_names"]
                print(f"class list: VENDORED copy ({len(class_names)} classes) "
                      f"-- conceptgraph unavailable ({e})")
        print(f"class-agnostic map detected -> assigning labels from clip_ft "
              f"against {len(class_names)} Replica classes, their method")

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if _tf_cached is None:
            print(f"CLIP device: {dev}"
                  + ("" if dev == "cuda" else
                     "  <- NO GPU: ViT-H-14 will take ~2.5 GB of SYSTEM RAM"),
                  flush=True)
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k")
            model = model.to(dev).eval()
            tok = open_clip.get_tokenizer("ViT-H-14")
            _rss("after loading ViT-H-14")
        # their n_exclude=6 list; suppressed BEFORE argmax exactly as they do
        EXCLUDE = ["other", "floor", "wall", "ceiling", "door", "window"]

        # And the piece we were missing until 2026-08-16. Their eval builds
        # ignore_index as excluded classes PLUS every class absent from this
        # scene's GT (eval_replica_semseg.py lines 93-105), then suppresses all
        # of it before the argmax (line 139):
        #
        #     existing_index     = gt_class.unique()
        #     non_existing_index = setdiff1d(all_class_index, existing_index)
        #     ignore_index       = append(ignore_index, non_existing_index)
        #     object_class_sim[:, ignore_index] = -1e10
        #
        # So their objects can only be labelled with classes that actually
        # occur in the room. Ours could pick anything in the 51-class list and
        # then be scored wrong for it. Because mAcc is an UNWEIGHTED per-class
        # mean, that asymmetry hits it far harder than frequency-weighted
        # F-mIoU -- which is exactly the shape of our shortfall against their
        # published numbers (mAcc at 67% of theirs, F-mIoU at 83%).
        in_scene = None
        if not args.no_in_scene:
            for p in (f"{PKG}/handoff/{args.scene}_cgfront/eval_points.npz",
                      f"{PKG}/handoff/{args.scene}/eval_points.npz"):
                if os.path.exists(p):
                    in_scene = set(np.load(p, allow_pickle=True)
                                   ["gt_class"].astype(str).tolist())
                    print(f"in-scene classes from {p}: {len(in_scene)}")
                    break
            if in_scene is None:
                gp = f"outputs/replica_{args.scene}/gt_instances.json"
                if os.path.exists(gp):
                    in_scene = {i["cls"] for i in json.load(open(gp))["instances"]}
                    print(f"in-scene classes from {gp}: {len(in_scene)}")
            if in_scene is None:
                print("WARNING: no GT found for in-scene suppression; their "
                      "labels will be assigned over the FULL class list, which "
                      "is not what their eval does")

        with torch.no_grad():
            if _tf_cached is None:
                tf = model.encode_text(
                    tok([f"an image of {c}" for c in class_names]).to(dev))
                tf = tf / tf.norm(dim=-1, keepdim=True)
                np.savez_compressed(
                    _cache, class_names=np.array(class_names),
                    text_ft=tf.cpu().numpy().astype(np.float32))
                print(f"wrote {_cache} -- later scenes will skip the model")
                del model, tok
                gc.collect()
                if dev == "cuda":
                    torch.cuda.empty_cache()
                _rss("after freeing ViT-H-14")
            else:
                tf = torch.from_numpy(_tf_cached).to(dev).float()
            of = torch.from_numpy(
                np.stack([np.asarray(f) for f in feats])).to(dev).float()
            of = of / of.norm(dim=-1, keepdim=True)
            sim = of @ tf.T
            n_sup = 0
            for c in EXCLUDE:
                if c in class_names:
                    sim[:, class_names.index(c)] = -1e10
            if in_scene:
                for j, c in enumerate(class_names):
                    if c not in in_scene:
                        sim[:, j] = -1e10
                        n_sup += 1
                print(f"suppressed {n_sup} classes absent from this scene "
                      f"(their eval does this); {len(class_names)-n_sup-len(EXCLUDE)}"
                      " remain selectable")
            idx = sim.argmax(dim=-1).cpu().numpy()
        for r, i in zip(obj_rows, idx):
            r["cls"] = class_names[int(i)]
        by_id = {r["id"]: r["cls"] for r in obj_rows}
        for r in obs_rows:
            r["cls"] = by_id.get(r["obj"], r["cls"])
        import collections as _c
        print("assigned:", _c.Counter(r["cls"] for r in obj_rows).most_common(10))

    with open(os.path.join(out_dir, "cg_objects.json"), "w") as f:
        json.dump(obj_rows, f, indent=1)
    with open(os.path.join(out_dir, "cg_observations.json"), "w") as f:
        json.dump(obs_rows, f)

    # Persist their per-object CLIP features (ViT-H-14, 1024-d). These were
    # collected into `feats` at line 149 and then discarded on every export
    # to date -- re-deriving them means re-downloading the 417 MB pkl.gz maps.
    have = [(r["id"], f) for r, f in zip(obj_rows, feats) if f is not None]
    if have:
        np.savez_compressed(
            os.path.join(out_dir, "cg_clip_ft.npz"),
            obj_id=np.array([i for i, _ in have], np.int64),
            clip_ft=np.stack([np.asarray(f, np.float32).reshape(-1)
                              for _, f in have]))
        print(f"wrote cg_clip_ft.npz: {len(have)}/{len(obj_rows)} objects "
              f"carry clip_ft")

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
    n_distinct = len({(round(r["x"], 3), round(r["y"], 3), round(r["z"], 3))
                      for r in obs_rows})
    print(f"{len(obj_rows)} objects, {len(obs_rows)} observations "
          f"({args.obs_geometry} geometry -> {n_distinct} distinct positions), "
          f"{len(P)} labelled points")
    print("NOTE: eval_points.npz (GT labels) is built by 04_vsa_labels.py "
          "from the semantic renders so BOTH systems are scored on the "
          "identical point set.")
    print("STAGE OK (03_export_cg)")


if __name__ == "__main__":
    main()
