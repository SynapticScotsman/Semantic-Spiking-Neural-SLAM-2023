"""Fetch + convert a Microsoft 7-Scenes scene into sequence configs.

Why: every result so far is a blocked split of ONE self-collected trajectory.
7-Scenes gives OFFICIAL separate train/test traverses (chess: 4 train + 2
test sequences of 1000 frames) with KinectFusion ground-truth poses, and
image-retrieval relocalisation baselines publish median errors on the same
split (DenseVLAD 0.21 m on chess) — a literature-comparable protocol.

Disk-frugal: the scene zip nests one zip per sequence; we stream-extract ONLY
`*.color.png` + `*.pose.txt` (depth is skipped, ~half the bytes) and delete
the archives unless --keep-zip.

Conversion: each `frame-XXXXXX.pose.txt` is a 4x4 camera-to-world matrix.
We pick the floor plane as the two axes with the largest translation variance
over the WHOLE scene (all sequences — one consistent frame), take yaw as the
heading of the camera's view direction projected onto that plane, and write
per-sequence `poses.csv` (time,x,y,yaw + tx,ty,tz for later 3D error) plus a
`vsa_cognitive_mapping/configs/7scenes_<scene>_seqXX.json` folder-config per
traverse. No changes to sequences.py are needed: FolderSequence +
read_csv_poses cover this, and `frame-%06d.color.png` filenames give unique
frame ids within a sequence.

    python tools/prepare_7scenes.py --scene chess
    python tools/prepare_7scenes.py --scene chess --skip-download   # zip present
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BASE = ("http://download.microsoft.com/download/2/8/5/"
        "28564B23-0828-408F-8631-23B1EFF1DAC8/{scene}.zip")
# Official splits (TrainSplit.txt/TestSplit.txt inside the zip confirm these;
# parsed when present, this table is the fallback).
KNOWN_SPLITS = {
    "chess": (["01", "02", "04", "06"], ["03", "05"]),
    "fire": (["01", "02"], ["03", "04"]),
    "heads": (["02"], ["01"]),
    "office": (["01", "03", "04", "05", "08", "10"], ["02", "06", "07", "09"]),
}

KEEP = re.compile(r"frame-\d{6}\.(color\.png|pose\.txt)$")


def download(url, dest):
    def hook(n, bs, total):
        done = n * bs
        if n % 400 == 0 or done >= total:
            mb = done / 1e6
            pct = f" ({100.0 * done / total:.0f}%)" if total > 0 else ""
            print(f"  {mb:,.0f} MB{pct}", flush=True)
    print(f"downloading {url}\n -> {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=hook)


def extract_scene(zip_path, scene_dir, keep_zip):
    """Outer zip nests seq-XX.zip archives; pull only color+pose members."""
    n_kept = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base.lower().endswith(".zip"):
                seq = os.path.splitext(base)[0]          # seq-01
                out = os.path.join(scene_dir, seq)
                os.makedirs(out, exist_ok=True)
                print(f"  {base} ...", flush=True)
                with z.open(name) as fh:
                    # nested zips need random access: buffer into memory
                    # (~700 MB worst case) rather than extracting to disk
                    inner = zipfile.ZipFile(io.BytesIO(fh.read()))
                for m in inner.namelist():
                    mb = os.path.basename(m)
                    if KEEP.search(mb):
                        with inner.open(m) as src, \
                                open(os.path.join(out, mb), "wb") as dst:
                            dst.write(src.read())
                        n_kept += 1
                inner.close()
            elif base.lower().endswith(".txt"):
                with z.open(name) as src:
                    with open(os.path.join(scene_dir, base), "wb") as dst:
                        dst.write(src.read())
            elif KEEP.search(base):                       # flat layout fallback
                seq_m = re.search(r"(seq-\d+)", name)
                out = os.path.join(scene_dir, seq_m.group(1) if seq_m else ".")
                os.makedirs(out, exist_ok=True)
                with z.open(name) as src, \
                        open(os.path.join(out, base), "wb") as dst:
                    dst.write(src.read())
                n_kept += 1
    if not keep_zip:
        os.remove(zip_path)
        print(f"  removed {zip_path}")
    print(f"  kept {n_kept} color/pose files")


def read_split_file(path):
    if not os.path.exists(path):
        return None
    out = []
    for line in open(path):
        m = re.search(r"(\d+)", line)
        if m:
            out.append(f"{int(m.group(1)):02d}")
    return out or None


def load_pose(path):
    M = np.loadtxt(path).reshape(4, 4)
    return M[:3, 3], M[:3, :3]


def convert(scene, scene_dir, cfg_dir):
    seqs = sorted(d for d in os.listdir(scene_dir)
                  if d.startswith("seq-")
                  and os.path.isdir(os.path.join(scene_dir, d)))
    if not seqs:
        raise SystemExit(f"no seq-* directories under {scene_dir}")

    # one consistent floor plane for the whole scene
    all_t = []
    per_seq = {}
    for s in seqs:
        d = os.path.join(scene_dir, s)
        poses = sorted(f for f in os.listdir(d) if f.endswith(".pose.txt"))
        rows = []
        for p in poses:
            t3, R = load_pose(os.path.join(d, p))
            rows.append((int(re.search(r"\d{6}", p).group(0)), t3, R))
        per_seq[s] = rows
        all_t.append(np.stack([r[1] for r in rows]))
    T = np.concatenate(all_t)
    var = T.var(axis=0)
    a, b = sorted(np.argsort(var)[-2:])          # two largest-variance axes
    ax_names = "xyz"
    print(f"{scene}: translation std per axis "
          f"{np.sqrt(var).round(3).tolist()} -> floor plane = "
          f"({ax_names[a]},{ax_names[b]})")

    # converter self-check: recompose one pose file and compare
    s0 = seqs[0]
    fid0, t0, R0 = per_seq[s0][0]
    M = np.loadtxt(os.path.join(scene_dir, s0,
                                f"frame-{fid0:06d}.pose.txt")).reshape(4, 4)
    assert np.allclose(M[:3, 3], t0) and np.allclose(M[:3, :3], R0)

    train = read_split_file(os.path.join(scene_dir, "TrainSplit.txt"))
    test = read_split_file(os.path.join(scene_dir, "TestSplit.txt"))
    if train is None or test is None:
        train, test = KNOWN_SPLITS.get(scene, (None, None))
        if train is None:
            raise SystemExit("no split files and no fallback known "
                             f"for scene {scene!r}")
        print("  (split files missing; using the known official split)")

    configs = {}
    for s in seqs:
        d = os.path.join(scene_dir, s)
        csv_path = os.path.join(d, "poses.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time", "x", "y", "yaw", "tx", "ty", "tz"])
            for fid, t3, R in per_seq[s]:
                view = R @ np.array([0.0, 0.0, 1.0])     # camera looks +z
                yaw = float(np.arctan2(view[b], view[a]))
                w.writerow([fid, f"{fid / 30.0:.4f}",
                            f"{t3[a]:.6f}", f"{t3[b]:.6f}", f"{yaw:.6f}",
                            f"{t3[0]:.6f}", f"{t3[1]:.6f}", f"{t3[2]:.6f}"])
        tag = f"7scenes_{scene}_{s.replace('-', '')}"
        cfg = {
            "_comment": f"7-Scenes {scene} {s} ({'train' if s[4:] in train else 'test'} "
                        "split). Generated by tools/prepare_7scenes.py; floor "
                        f"plane = world axes ({ax_names[a]},{ax_names[b]}), yaw = "
                        "camera view direction projected onto it. tx/ty/tz "
                        "columns keep the full 3D translation for 3D error.",
            "name": tag,
            "kind": "folder",
            "images": os.path.join(d, "frame-*.color.png").replace("\\", "/"),
            "fps": 30,
            "out_dir": f"outputs/{tag}",
            "poses": {"kind": "csv",
                      "path": csv_path.replace("\\", "/"),
                      "time": "time", "x": "x", "y": "y", "yaw": "yaw"},
        }
        cfg_path = os.path.join(cfg_dir, f"{tag}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        configs[s] = cfg_path
        print(f"  {s}: {len(per_seq[s])} frames -> {cfg_path}")

    meta = dict(scene=scene, floor_axes=[ax_names[a], ax_names[b]],
                translation_std=np.sqrt(var).tolist(),
                train=[f"seq-{x}" for x in train if f"seq-{x}" in seqs],
                test=[f"seq-{x}" for x in test if f"seq-{x}" in seqs],
                configs=configs)
    meta_path = os.path.join(scene_dir, "convert_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}\n  train: {meta['train']}\n  test:  {meta['test']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="chess")
    ap.add_argument("--url", default=None, help="override the download URL")
    ap.add_argument("--data-root", default="data/7scenes")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    scene_dir = os.path.join(args.data_root, args.scene)
    os.makedirs(scene_dir, exist_ok=True)
    zip_path = os.path.join(args.data_root, f"{args.scene}.zip")

    if not args.skip_extract:
        if not args.skip_download and not os.path.exists(zip_path):
            download(args.url or BASE.format(scene=args.scene), zip_path)
        if os.path.exists(zip_path):
            extract_scene(zip_path, scene_dir, args.keep_zip)
        elif not any(d.startswith("seq-") for d in os.listdir(scene_dir)):
            raise SystemExit(f"neither {zip_path} nor extracted sequences exist")

    convert(args.scene, scene_dir,
            os.path.join("vsa_cognitive_mapping", "configs"))


if __name__ == "__main__":
    main()
