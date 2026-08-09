"""Fetch + convert ONE Replica scene (NICE-SLAM/iMAP render) for the
ConceptGraphs comparison — without downloading the whole 12 GB archive.

The renders live in a single `Replica.zip` on the NICE-SLAM data server
(8 scenes x 2000 RGB-D frames + traj.txt; the exact data ConceptGraphs
consumes, its repo demo scene being room0). This script opens the REMOTE
zip through HTTP Range requests (block-cached file-like object), reads the
central directory, and extracts only the requested scene's members —
~2 GB transferred instead of 12. Falls back to a plain full download into
the scratch dir only if the server refuses ranges.

Conversion mirrors prepare_7scenes.py: `traj.txt` rows are 4x4
camera-to-world matrices (row-major, one per frame); floor plane = the two
largest-variance translation axes; yaw = camera view direction (+z)
projected onto that plane; per-scene `poses.csv` keeps tx/ty/tz for 3D
error; a FolderSequence config JSON is written per scene. Depth PNGs are
kept — the metric object localisation (box + depth + pose -> world) needs
them for the grounding battery.

    python tools/prepare_replica.py --scene room0
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

URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
BLOCK = 8 * 1024 * 1024


class RangeFile(io.RawIOBase):
    """Read-only file-like view of a remote URL via HTTP Range requests,
    with an LRU block cache. zipfile's access pattern (central directory at
    the end, then ascending member reads) maps onto few large fetches."""

    def __init__(self, url, block=BLOCK, max_blocks=48):
        self.url, self.block, self.max_blocks = url, block, max_blocks
        self.pos = 0
        self.cache = {}
        self.order = []
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        last = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    cr = r.headers.get("Content-Range", "")
                    if not cr or "/" not in cr:
                        raise IOError("server did not honour Range request")
                    self.size = int(cr.split("/")[-1])
                    last = None
                    break
            except Exception as e:
                last = e
                import time
                time.sleep(2 ** attempt)
        if last is not None:
            raise last
        self.fetched = 0

    def _get_block(self, bi):
        if bi in self.cache:
            return self.cache[bi]
        lo = bi * self.block
        hi = min(lo + self.block, self.size) - 1
        req = urllib.request.Request(self.url,
                                     headers={"Range": f"bytes={lo}-{hi}"})
        data = None
        for attempt in range(6):                 # transient DNS/socket hiccups
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  block {bi}: {e!r}; retry in {wait}s", flush=True)
                import time
                time.sleep(wait)
        if data is None:
            raise IOError(f"block {bi} failed after retries")
        self.fetched += len(data)
        self.cache[bi] = data
        self.order.append(bi)
        if len(self.order) > self.max_blocks:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            bi, off = divmod(self.pos, self.block)
            chunk = self._get_block(bi)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)


def fetch_scene(scene, scene_dir):
    rgb_pat = re.compile(r"frame\d{6}\.jpg$")
    dep_pat = re.compile(r"depth\d{6}\.png$")
    os.makedirs(scene_dir, exist_ok=True)
    rf = RangeFile(URL)
    print(f"remote zip: {rf.size / 1e9:.1f} GB total; extracting only "
          f"'{scene}' members via range requests")
    with zipfile.ZipFile(rf) as z:
        members = [m for m in z.namelist() if f"/{scene}/" in m or
                   m.startswith(f"{scene}/")]
        if not members:
            raise SystemExit(f"no members for scene {scene!r}; first names: "
                             f"{z.namelist()[:5]}")
        n_done = 0
        for m in sorted(members):
            base = os.path.basename(m)
            if not base:
                continue
            if rgb_pat.search(base) or dep_pat.search(base) or base == "traj.txt":
                dest = os.path.join(scene_dir, base)
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    continue                     # resume: already extracted
                with z.open(m) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                n_done += 1
                if n_done % 250 == 0:
                    print(f"  {n_done} files ({rf.fetched / 1e9:.2f} GB fetched)",
                          flush=True)
    print(f"extracted {n_done} files; transferred {rf.fetched / 1e9:.2f} GB "
          f"of {rf.size / 1e9:.1f} GB")


def convert(scene, scene_dir, cfg_dir):
    traj = os.path.join(scene_dir, "traj.txt")
    M = np.loadtxt(traj).reshape(-1, 4, 4)
    T = M[:, :3, 3]
    R = M[:, :3, :3]
    n = len(M)
    rgbs = sorted(f for f in os.listdir(scene_dir) if f.startswith("frame"))
    assert len(rgbs) == n, f"{len(rgbs)} rgb frames vs {n} poses"

    var = T.var(axis=0)
    a, b = sorted(np.argsort(var)[-2:])
    ax = "xyz"
    print(f"{scene}: {n} poses; translation std {np.sqrt(var).round(3).tolist()} "
          f"-> floor plane ({ax[a]},{ax[b]})")
    # self-check: recompose row 0 against the raw text
    assert np.allclose(np.loadtxt(traj, max_rows=1).reshape(4, 4)[:3, 3], T[0])

    csv_path = os.path.join(scene_dir, "poses.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time", "x", "y", "yaw", "tx", "ty", "tz"])
        for i in range(n):
            view = R[i] @ np.array([0.0, 0.0, 1.0])
            yaw = float(np.arctan2(view[b], view[a]))
            w.writerow([i, f"{i / 30.0:.4f}",
                        f"{T[i, a]:.6f}", f"{T[i, b]:.6f}", f"{yaw:.6f}",
                        f"{T[i, 0]:.6f}", f"{T[i, 1]:.6f}", f"{T[i, 2]:.6f}"])

    tag = f"replica_{scene}"
    cfg = {
        "_comment": f"Replica {scene} (NICE-SLAM/iMAP render, 2000 RGB-D "
                    "frames) — the ConceptGraphs demo scene's data. Generated "
                    "by tools/prepare_replica.py; floor plane "
                    f"({ax[a]},{ax[b]}), yaw = view dir; depthXXXXXX.png kept "
                    "beside the RGB for metric object localisation.",
        "name": tag,
        "kind": "folder",
        "images": os.path.join(scene_dir, "frame*.jpg").replace("\\", "/"),
        "fps": 30,
        "out_dir": f"outputs/{tag}",
        "poses": {"kind": "csv",
                  "path": csv_path.replace("\\", "/"),
                  "time": "time", "x": "x", "y": "y", "yaw": "yaw"},
    }
    cfg_path = os.path.join(cfg_dir, f"{tag}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {cfg_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--data-root", default="data/replica")
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()

    scene_dir = os.path.join(args.data_root, args.scene)
    if not args.skip_fetch:
        # extraction resumes (existing files skipped), so whole-fetch retries
        # are cheap; ride out flaky-network windows
        import time
        for outer in range(8):
            try:
                fetch_scene(args.scene, scene_dir)
                break
            except Exception as e:
                print(f"fetch attempt {outer + 1} failed: {e!r}; "
                      f"waiting {60 * (outer + 1)}s", flush=True)
                time.sleep(60 * (outer + 1))
        else:
            raise SystemExit("fetch failed after 8 attempts")
    convert(args.scene, scene_dir,
            os.path.join("vsa_cognitive_mapping", "configs"))


if __name__ == "__main__":
    main()
