"""Test the third FPE axis PROPERLY, after the first attempt was under-powered.

height_axis_test.py ran one scene, one guessed lz (0.35), the 2D-optimal lx/ly
unchanged, one trace size, and reported only the mean. It said -0.059 and I
called elevation dead. That is the same mistake made five times this week: a
single-scene, single-parameter result treated as a verdict, when the 2D length
scale itself needed a 13-run sweep before it looked good.

What is actually being asked here, separated:

  1. Is there ANY lz at which 3D beats 2D?   -> sweep lz
  2. Does 3D need the floor scales retuned?  -> coarse joint grid over lx,ly,lz
  3. Is 3D capacity-starved rather than bad? -> repeat at 2x dimension. The
     dimension sweep showed mAcc flat to 128 KB, but that was measured IN 2D;
     a volume needs more than a plane to cover at the same resolution.
  4. Even if the MEAN falls, does it fix the classes it should? -> per-class
     deltas for high-elevation classes, which is the whole motivation.

Question 4 is the one the mean cannot answer, and the reason to run this at all:
room0's `vent` is a ceiling grille at z=+1.26 with the sofa at -0.93, so if
elevation carries information those classes must improve even if untuned floor
scales drag the average down.

    python collab_tasks/scripts/height_axis_sweep.py --scenes room0 room2

CPU only. Bundles from their labelled points (cg_labels.npz), which carry full
xyz, so no re-export is needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    class _Absent(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(f"torch.{name} used; decode path assumed NumPy")

    sys.modules["torch"] = _Absent("torch")

from vsa_cognitive_mapping.vsa import Phasor  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path]
    spec.loader.exec_module(mod)
    return mod


def bundle_decode(P, cls, Q, names, bases, scales, cap, hd, rng):
    """Bundle observations, then decode at every query point. Vectorised per
    class: FPE on a whole (n, hd) block at once rather than per observation."""
    sem = {c: Phasor(dim=hd, seed=9000 + i).values for i, c in enumerate(names)}
    tr = np.zeros(hd, np.complex128)
    for c in names:
        idx = np.flatnonzero(cls == c)
        if len(idx) == 0:
            continue
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        acc = np.ones((len(idx), hd), np.complex128)
        for a, (B, l) in enumerate(zip(bases, scales)):
            acc *= B[None, :] ** (P[idx, a, None] / l)
        tr += sem[c] * (acc.sum(0) / len(idx))
    tr /= max(np.abs(tr).max(), 1e-12)

    V = np.stack([tr / sem[c] for c in names])
    F = np.empty((len(names), len(Q)))
    CH = 2048
    for i0 in range(0, len(Q), CH):
        i1 = min(i0 + CH, len(Q))
        g = np.ones((i1 - i0, hd), np.complex128)
        for a, (B, l) in enumerate(zip(bases, scales)):
            g *= B[None, :] ** (Q[i0:i1, a, None] / l)
        F[:, i0:i1] = (V @ np.conj(g).T).real
    return np.array([names[w] for w in F.argmax(0)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=["room0", "room2"])
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--lzs", default="0.15,0.3,0.6,1.2,2.5,5.0",
                    help="elevation scales; large lz -> the axis carries almost "
                         "nothing, which is the graceful limit back to 2D")
    ap.add_argument("--floor", default="0.45,0.27;0.65,0.39;0.9,0.54",
                    help="lx,ly pairs to retry, since adding an axis moves the "
                         "floor optimum too")
    ap.add_argument("--hd", default="4096,8192")
    ap.add_argument("--out", default="outputs/height_axis_sweep.json")
    args = ap.parse_args()
    score05 = load_mod("student_gpu_package/05_score.py", "score05")
    EX = score05.CG_EXCLUDE_6
    lzs = [float(v) for v in args.lzs.split(",")]
    floors = [tuple(float(x) for x in p.split(",")) for p in args.floor.split(";")]
    hds = [int(v) for v in args.hd.split(",")]
    out = {}

    for s in args.scenes:
        d = f"student_gpu_package/handoff/{s}_cgfront"
        E = np.load(f"{d}/eval_points.npz", allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        C = np.load(f"{d}/cg_labels.npz", allow_pickle=True)
        P, cls = C["xyz"].astype(float), C["pred_class"].astype(str)
        v = xyz.var(0)
        a, b = sorted(np.argsort(v)[-2:])
        up = ({0, 1, 2} - {a, b}).pop()
        names = sorted(set(cls))
        # classes that sit high in the room -- the ones elevation must fix
        high = sorted({c for c in set(gt) - set(EX)
                       if (gt == c).sum() >= 5
                       and xyz[gt == c, up].mean() > xyz[:, up].mean() + 0.5})
        print(f"\n{'='*74}\n{s}: {len(P)} labelled pts, {len(names)} classes")
        print(f"high classes (elevation should help these): {high or 'none'}")
        print(f"{'='*74}")

        out[s] = {"high": high, "rows": []}
        for hd in hds:
            rng = np.random.default_rng(0)
            Bx = Phasor(dim=hd, seed=11).values
            By = Phasor(dim=hd, seed=12).values
            Bz = Phasor(dim=hd, seed=13).values
            print(f"\n  dim {hd} ({hd*8//1024} KB)")
            print(f"    {'lx,ly':>12}{'lz':>7}{'mAcc':>8}{'F-mIoU':>9}"
                  f"{'high-cls acc':>14}")
            for (lx, ly) in floors:
                # 2D reference at this floor scale
                pr = bundle_decode(P[:, [a, b]], cls, xyz[:, [a, b]], names,
                                   [Bx, By], [lx, ly], args.cap, hd, rng)
                m, f, _ = score05.macc_fmiou(gt, pr, exclude=EX)
                hi = (st.mean(float((pr[gt == c] == c).mean()) for c in high)
                      if high else float("nan"))
                print(f"    {f'{lx},{ly}':>12}{'2D':>7}{m:>8.3f}{f:>9.3f}"
                      f"{hi:>14.3f}", flush=True)
                out[s]["rows"].append(dict(hd=hd, lx=lx, ly=ly, lz=None,
                                           macc=m, fmiou=f, high=hi))
                for lz in lzs:
                    pr = bundle_decode(P[:, [a, b, up]], cls,
                                       xyz[:, [a, b, up]], names,
                                       [Bx, By, Bz], [lx, ly, lz], args.cap,
                                       hd, rng)
                    m, f, _ = score05.macc_fmiou(gt, pr, exclude=EX)
                    hi = (st.mean(float((pr[gt == c] == c).mean()) for c in high)
                          if high else float("nan"))
                    print(f"    {f'{lx},{ly}':>12}{lz:>7.2f}{m:>8.3f}{f:>9.3f}"
                          f"{hi:>14.3f}", flush=True)
                    out[s]["rows"].append(dict(hd=hd, lx=lx, ly=ly, lz=lz,
                                               macc=m, fmiou=f, high=hi))

    print(f"\n{'='*74}\nDOES THE THIRD AXIS EVER WIN?\n{'='*74}")
    for s, v in out.items():
        r2 = [r for r in v["rows"] if r["lz"] is None]
        r3 = [r for r in v["rows"] if r["lz"] is not None]
        b2 = max(r2, key=lambda r: r["macc"])
        b3 = max(r3, key=lambda r: r["macc"])
        print(f"{s:<9} best 2D {b2['macc']:.3f} (lx {b2['lx']}, dim {b2['hd']})"
              f"   best 3D {b3['macc']:.3f} (lz {b3['lz']}, dim {b3['hd']})"
              f"   {b3['macc']-b2['macc']:+.3f}")
        if v["high"]:
            h2 = max(r2, key=lambda r: r["high"])["high"]
            h3 = max(r3, key=lambda r: r["high"])["high"]
            print(f"          high-elevation classes: 2D {h2:.3f} -> 3D {h3:.3f}"
                  f"  {h3-h2:+.3f}   {v['high']}")
    print("\nIf the MEAN never wins but the HIGH-ELEVATION classes do, elevation "
          "carries\nreal information and the loss is elsewhere -- untuned floor "
          "scales, or the\nsame budget spread over a volume. If neither wins at "
          "any lz or any dim, the\naxis is genuinely not the answer here.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
