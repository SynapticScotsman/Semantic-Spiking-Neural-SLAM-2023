"""Decode-rule sweep that runs on your laptop — no Colab, no GPU, no VM.

Everything after the frontend is NumPy on CPU, and the inputs are small:
eval_points.npz + vsa_labels' source observations. So the decode experiments do
not need a GPU session at all. Download one handoff folder from Drive and
iterate locally.

Sweeps three readout rules on the same trace:
  argmax      — current behaviour: per-cell winner across classes
  normalised  — per-class z-score before the argmax (tested 2026-08-14: no gain)
  threshold   — SSPictR's rule (Penzkofer et al., AIC 2025, Eq. 3): decode each
                class INDEPENDENTLY against a threshold so classes never compete
                for a cell. The scorer needs one label per point, so among the
                classes clearing their threshold we take the highest z-scored
                response; cells where nothing clears fall back to plain argmax
                (reported, so the fallback rate is never hidden).

Why threshold is the interesting one: per_class_breakdown showed 18 of 24
scored classes at EXACTLY 0.000 — including vent, which contributed more
observations than any other class. That is cross-class winner-take-all, and it
is our own choice: the closest published system decodes per class.

    python collab_tasks/scripts/local_decode_sweep.py \
        --handoff downloaded/room0_cgfront \
        --points  downloaded/room0_cgfront/object_points.json

Needs numpy and this repo importable; no torch, no CLIP, no Colab.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Both modules import torch at the top for their CLIP paths, but the four
# functions used here are pure NumPy. On a laptop without torch, stand in a
# placeholder rather than pulling a multi-hundred-MB wheel for symbols that are
# never called. If any of them DOES touch torch, the placeholder raises on
# attribute access, so this fails loudly instead of silently doing something
# else. Verified with --self-test.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import types

    class _Absent(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(
                f"torch.{name} was actually used — the no-torch shim in "
                "local_decode_sweep.py assumed the decode path never touches "
                "torch. Install torch (CPU build is enough) and re-run.")

    sys.modules["torch"] = _Absent("torch")

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, cap_per_class)

HD, LS = 4096, 0.6


def load_metric():
    """Reuse 05_score's own macc_fmiou and exclusion list. It starts with a
    digit so it cannot be imported by name; a second implementation of the
    metric is how numbers quietly drift apart."""
    path = os.path.join("student_gpu_package", "05_score.py")
    spec = importlib.util.spec_from_file_location("score05", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path]
    spec.loader.exec_module(mod)
    return mod


def fields(pts, xyz, a, b, grid, max_per_class):
    """Class response maps over the floor grid — identical to 04_vsa_labels."""
    enc = ClassroomEncoders(HD, 0, LS, 20.0)
    sem = class_phasors(sorted({p["cls"] for p in pts}), HD)
    trace = build_trace(cap_per_class(pts, max_per_class), enc, sem, HD)
    trace /= max(np.abs(trace).max(), 1e-12)
    xs, ys = xyz[:, a], xyz[:, b]
    gx = np.linspace(xs.min(), xs.max(), grid)
    gy = np.linspace(ys.min(), ys.max(), grid)
    G = np.empty((grid ** 2, HD), np.complex64)
    k = 0
    for yy in gy:
        for xx in gx:
            G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
            k += 1
    names = sorted(sem)
    F = np.stack([((trace / sem[c])[None, :] @ np.conj(G).T).real[0] for c in names])
    ix = np.clip(np.searchsorted(gx, xs), 0, grid - 1)
    iy = np.clip(np.searchsorted(gy, ys), 0, grid - 1)
    cell = iy * grid + ix
    return F, names, cell


def decode(F, names, cell, mode, tau):
    Z = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)
    if mode == "argmax":
        win = F.argmax(0); miss = 0
    elif mode == "normalised":
        win = Z.argmax(0); miss = 0
    else:                                   # per-class threshold, SSPictR-style
        above = Z > tau                     # each class judged on its own scale
        masked = np.where(above, Z, -np.inf)
        win = masked.argmax(0)
        none_clear = ~above.any(0)
        win[none_clear] = F[:, none_clear].argmax(0)   # fall back, and report it
        miss = float(none_clear.mean())
    return np.array([names[w] for w in win[cell]]), miss


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", help="folder with eval_points.npz")
    ap.add_argument("--points", help="object_points.json to bundle")
    ap.add_argument("--self-test", action="store_true",
                    help="run the whole pipeline on synthetic data — proves the "
                         "encode/decode/score path works on this machine "
                         "(including the no-torch shim) before you download "
                         "anything from Drive")
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--taus", default="0.5,1.0,1.5,2.0,3.0")
    args = ap.parse_args()

    if args.self_test:
        # three well-separated blobs, one class each: a correct pipeline should
        # score near 1.0, so anything low here is a local setup problem rather
        # than a finding about the memory.
        rng = np.random.RandomState(0)
        centres = {"chair": (1.0, 1.0), "table": (4.0, 1.0), "lamp": (1.0, 4.0)}
        xyz, gt, pts = [], [], []
        for c, (cx, cy) in centres.items():
            p = rng.normal([cx, cy], 0.25, size=(400, 2))
            xyz.append(np.c_[p, np.zeros(len(p))]); gt += [c] * len(p)
            for k in range(120):
                q = rng.normal([cx, cy], 0.25, size=2)
                pts.append(dict(frame=k, cls=c, conf=1.0, det=len(pts),
                                x=float(q[0]), y=float(q[1])))
        xyz = np.concatenate(xyz); gt = np.array(gt)
        print(f"SELF-TEST: {len(pts)} synthetic observations, 3 classes, "
              f"{len(gt)} eval points (expect mAcc near 1.0)")
    else:
        if not (args.handoff and args.points):
            raise SystemExit("need --handoff and --points (or --self-test)")
        E = np.load(os.path.join(args.handoff, "eval_points.npz"), allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        pts = json.load(open(args.points))["points"]
    print(f"{len(pts)} observations, {len({p['cls'] for p in pts})} classes, "
          f"{len(gt)} eval points")

    var = xyz.var(0)
    a, b = sorted(np.argsort(var)[-2:])
    F, names, cell = fields(pts, xyz, a, b, args.grid, args.max_per_class)
    mod = load_metric()

    print(f"\n{'rule':<22}{'mAcc':>8}{'F-mIoU':>9}{'nonzero':>9}{'fallback':>10}")
    print("-" * 58)
    runs = [("argmax", None), ("normalised", None)]
    runs += [("threshold", float(t)) for t in args.taus.split(",")]
    for mode, tau in runs:
        pred, miss = decode(F, names, cell, mode, tau)
        acc, fmiou, ncls = mod.macc_fmiou(gt, pred, exclude=mod.CG_EXCLUDE_6)
        nz = sum(1 for c in set(gt) - set(mod.CG_EXCLUDE_6)
                 if (pred[gt == c] == c).mean() > 0)
        label = mode if tau is None else f"threshold tau={tau}"
        print(f"{label:<22}{acc:>8.3f}{fmiou:>9.3f}{nz:>9}{miss:>9.1%}")

    print("\nReference (room0_cgfront, cap 400): argmax scored 0.187 / 0.351 with"
          "\n18 of 24 classes at exactly zero. Their backend on the same frontend"
          "\nscored 0.290 / 0.410. A rule that raises 'nonzero' is doing the thing"
          "\nthe per-class breakdown said was missing.")


if __name__ == "__main__":
    main()
