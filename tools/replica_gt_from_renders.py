"""True GT instance centroids for Replica room_0 — from the vMAP renders.

The vMAP release (HuggingFace kxic/vMAP, demo_replica_room_0.zip) carries
per-frame SEMANTIC INSTANCE maps + depth + poses rendered from the same
Replica mesh as the NICE-SLAM sequence we evaluate on. Backprojecting each
instance's pixels through depth and pose gives per-instance world centroids
— real ground truth for the grounding battery, no GPU or Habitat needed.

Two-step usage (layout differs between releases, so inventory first):

    python tools/replica_gt_from_renders.py --inspect      # list zip layout
    python tools/replica_gt_from_renders.py                # extract + build

Outputs `outputs/replica_room0/gt_instances.json`:
    {"instances": [{"class": str, "x": float, "y": float, "z": float,
                    "n_px": int}],
     "class_map": {...}, "frame_check": {...}}
in the same floor-plane axes as `data/replica/room0/poses.csv` (the script
re-derives them from OUR traj.txt and prints the alignment check — the two
renders share the mesh's world frame; the check must pass before any truth
number is quoted).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

VMAP_URL = "https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip"


def scene_paths(scene):
    """Our scene tag 'room0' <-> vMAP's 'room_0'."""
    vscene = re.sub(r"(\D)(\d+)$", r"\1_\2", scene)
    local_zip = f"data/replica/{vscene}_vmap.zip" \
        if scene == "room0" else None          # room0 came as its own demo zip
    return vscene, local_zip, (f"data/replica/{vscene}_vmap",
                               f"data/replica/{scene}/traj.txt",
                               f"outputs/replica_{scene}/gt_instances.json")

# Replica renders (iMAP/NICE-SLAM/vMAP family) share this camera
FX = FY = 600.0
CX, CY = 599.5, 339.5
DEPTH_SCALE = 6553.5


def inspect():
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    print(f"{len(names)} members. Directory-level summary:")
    dirs = {}
    for n in names:
        d = os.path.dirname(n)
        dirs.setdefault(d, [0, None])
        dirs[d][0] += 1
        if dirs[d][1] is None and not n.endswith("/"):
            dirs[d][1] = os.path.basename(n)
    for d, (cnt, ex) in sorted(dirs.items()):
        print(f"  {d or '.'}: {cnt} files (e.g. {ex})")
    for pat in ("*info*", "*.json", "*.txt"):
        hits = fnmatch.filter(names, pat)
        if hits:
            print(f"\n{pat}: {hits[:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--stride", type=int, default=10,
                    help="use every Nth GT frame (centroids converge fast)")
    ap.add_argument("--min-px", type=int, default=2000,
                    help="ignore instances with fewer total pixels")
    args = ap.parse_args()

    global ZIP, EXTRACT, OUR_TRAJ, OUT
    vscene, local_zip, (EXTRACT, OUR_TRAJ, OUT) = scene_paths(args.scene)
    if local_zip and os.path.exists(local_zip):
        ZIP = local_zip
        print(f"local zip {ZIP}")
    else:
        # remote range-access into the full vmap.zip: only this scene's GT
        # members are transferred (semantic_instance + depth + traj + info)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from prepare_replica import RangeFile
        ZIP = RangeFile(VMAP_URL)
        print(f"remote vmap.zip: {ZIP.size / 1e9:.1f} GB; range-pulling "
              f"only {vscene} GT members")

    if args.inspect:
        inspect()
        return

    from PIL import Image

    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist()
                 if f"{vscene}/imap/00/" in n]
        base = os.path.basename
        inst = sorted(n for n in names
                      if re.fullmatch(r"semantic_instance_\d+\.png", base(n)))
        depth = sorted(n for n in names
                       if re.fullmatch(r"depth_\d+\.png", base(n)))
        traj = [n for n in names if base(n) == "traj_w_c.txt"]
        info = [n for n in z.namelist()
                if base(n) == "info_semantic.json" and vscene in n]
        sem_cls = sorted(n for n in names
                         if re.fullmatch(r"semantic_class_\d+\.png", base(n)))
        print(f"found: {len(inst)} instance maps, {len(depth)} depth, "
              f"{len(sem_cls)} class maps, traj={traj}, info={info}")
        if not inst or not depth or not traj:
            raise SystemExit("layout not as expected — run --inspect and "
                             "adapt the patterns")
        os.makedirs(EXTRACT, exist_ok=True)

        def pull(member):
            dest = os.path.join(EXTRACT, os.path.basename(member))
            if not os.path.exists(dest):
                with z.open(member) as src, open(dest, "wb") as f:
                    f.write(src.read())
            return dest

        traj_path = pull(traj[0])
        info_path = pull(info[0]) if info else None
        idx = lambda n: int(re.search(r"(\d+)\.png$", n).group(1))
        inst_by = {idx(n): n for n in inst}
        depth_by = {idx(n): n for n in depth}
        cls_by = {idx(n): n for n in sem_cls}
        frames = sorted(set(inst_by) & set(depth_by))[::args.stride]
        print(f"using {len(frames)} GT frames (stride {args.stride})")

        T = np.loadtxt(traj_path).reshape(-1, 4, 4)

        # depth scale differs between render packs (6553.5 vs 1000/mm):
        # pick the one that puts the median room depth in a sane range
        D0 = np.asarray(Image.open(pull(depth_by[frames[0]])), np.float32)
        scale = None
        for s in (DEPTH_SCALE, 1000.0):
            med = np.median(D0[D0 > 0]) / s
            if 0.8 <= med <= 6.0:
                scale = s
                break
        if scale is None:
            raise SystemExit(f"cannot determine depth scale (raw median "
                             f"{np.median(D0[D0 > 0]):.0f})")
        print(f"depth scale {scale} (median room depth "
              f"{np.median(D0[D0 > 0]) / scale:.2f} m)")

        # accumulate world points per instance id
        sums = {}
        counts = {}
        inst_cls_votes = {}
        for k, fi in enumerate(frames):
            I = np.asarray(Image.open(pull(inst_by[fi])))
            D = np.asarray(Image.open(pull(depth_by[fi])),
                           np.float32) / scale
            assert I.shape[:2] == D.shape[:2], (I.shape, D.shape)
            H, W = D.shape
            # camera intrinsics scale if the GT render size differs
            sx, sy = W / 1200.0, H / 680.0
            fx, fy, cx, cy = FX * sx, FY * sy, CX * sx, CY * sy
            M = T[fi]
            vv, uu = np.where((D > 0.1) & (D < 10.0))
            zd = D[vv, uu]
            pc = np.stack([(uu - cx) / fx * zd, (vv - cy) / fy * zd, zd])
            pw = M[:3, :3] @ pc + M[:3, 3:4]
            ids = I[vv, uu]
            C = (np.asarray(Image.open(pull(cls_by[fi])))[vv, uu]
                 if fi in cls_by else None)
            for iid in np.unique(ids):
                m = ids == iid
                sums.setdefault(int(iid), np.zeros(3))
                counts.setdefault(int(iid), 0)
                sums[int(iid)] += pw[:, m].sum(axis=1)
                counts[int(iid)] += int(m.sum())
                if C is not None:
                    v = inst_cls_votes.setdefault(int(iid), {})
                    cls_id, n = np.unique(C[m], return_counts=True)
                    for ci, nn in zip(cls_id, n):
                        v[int(ci)] = v.get(int(ci), 0) + int(nn)
            if k % 20 == 0:
                print(f"  frame {k}/{len(frames)}", flush=True)

    # class-id -> name
    id2name = {}
    if info_path:
        meta = json.load(open(info_path))
        for c in meta.get("classes", []):
            id2name[int(c["id"])] = c["name"]
        # id_to_label maps instance id -> class id directly when present
        i2l = meta.get("id_to_label")
    else:
        i2l = None

    # our floor plane (same rule as prepare_replica)
    Tours = np.loadtxt(OUR_TRAJ).reshape(-1, 4, 4)[:, :3, 3]
    var = Tours.var(axis=0)
    a, b = sorted(np.argsort(var)[-2:])

    instances = []
    for iid, n_px in counts.items():
        if n_px < args.min_px:
            continue
        c3 = sums[iid] / n_px
        cls_id = None
        if i2l is not None and iid < len(i2l):
            cls_id = int(i2l[iid])
        elif iid in inst_cls_votes:
            cls_id = max(inst_cls_votes[iid], key=inst_cls_votes[iid].get)
        name = id2name.get(cls_id, f"class_{cls_id}")
        instances.append(dict(cls=name, replica_class_id=cls_id,
                              x=float(c3[a]), y=float(c3[b]),
                              z=float(c3[3 - a - b] if a + b != 3 else c3[2]),
                              n_px=int(n_px)))
    instances.sort(key=lambda i: -i["n_px"])

    # frame-alignment check: GT points must live inside our room extents
    gx = [i["x"] for i in instances]
    gy = [i["y"] for i in instances]
    ours = dict(x=(float(Tours[:, a].min()), float(Tours[:, a].max())),
                y=(float(Tours[:, b].min()), float(Tours[:, b].max())))
    check = dict(gt_x=(min(gx), max(gx)), gt_y=(min(gy), max(gy)), ours=ours)
    print(f"\nalignment check (floor axes {'xyz'[a]},{'xyz'[b]}):")
    print(f"  GT instance extent  x {min(gx):+.2f}..{max(gx):+.2f}  "
          f"y {min(gy):+.2f}..{max(gy):+.2f}")
    print(f"  our camera extent   x {ours['x'][0]:+.2f}..{ours['x'][1]:+.2f}  "
          f"y {ours['y'][0]:+.2f}..{ours['y'][1]:+.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(dict(instances=instances, frame_check=check,
                       source="vMAP demo_replica_room_0 renders",
                       stride=args.stride, min_px=args.min_px), f, indent=1)
    print(f"\n{len(instances)} instances -> {OUT}")
    for i in instances[:25]:
        print(f"  {i['cls']:>20}  ({i['x']:+.2f}, {i['y']:+.2f})  px={i['n_px']}")


if __name__ == "__main__":
    main()
