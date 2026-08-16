"""Does encoding elevation as a third FPE axis fix the ceiling-onto-floor collapse?

Measured 2026-08-16, by looking at the actual pixels: room0's `vent` is a CEILING
GRILLE at z = +1.26, the same height as the ceiling (+1.24). The sofa beneath it
is at -0.93. Our export drops z entirely -- object_points.json records only
['cls','conf','det','frame','x','y'] -- so a vent 2.2 m above the sofa is
projected onto the sofa's own floor cell.

That is the 7.73 m smear: `vent` has 14 GT points spanning 0.16 m, and 2,806 of
our observations spanning the entire room. It is not a labelling error and not a
decode error. It is information discarded at export.

ConceptGraphs never has this problem because their nearest-neighbour runs in 3D:
a ceiling point is 2.2 m from a floor point and never wins.

This tests the fix WITHOUT re-exporting anything, by bundling from THEIR labelled
points (cg_labels.npz), which carry full xyz:

    2D   phi = Bx**(x/lx) * By**(y/ly)              (current)
    3D   phi = Bx**(x/lx) * By**(y/ly) * Bz**(z/lz)  (proposed)
    cut  2D, but discard observations above a height threshold (crude control --
         if simply dropping ceiling observations recovers most of the gain, the
         third axis is not carrying its weight)

Same trace size for all three, so this is a fair test of what the axis buys, not
of extra capacity.

    python collab_tasks/scripts/height_axis_test.py --scenes room0 room2 office4

CPU only.
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

HD = 4096
SCENES = ["room0", "room2", "office4"]


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path]
    spec.loader.exec_module(mod)
    return mod


def bundle(P, cls, names, bases, scales, cap, rng):
    """Trace = sum over observations of sem[c] * prod_axis B_a**(v_a / l_a)."""
    tr = np.zeros(HD, np.complex128)
    sem = {c: Phasor(dim=HD, seed=9000 + i).values for i, c in enumerate(names)}
    for c in names:
        idx = np.flatnonzero(cls == c)
        if len(idx) == 0:
            continue
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        acc = np.zeros(HD, np.complex128)
        for i in idx:
            v = np.ones(HD, np.complex128)
            for a, (B, l) in enumerate(zip(bases, scales)):
                v = v * (B ** (P[i, a] / l))          # B is an ndarray of
                                                      # unit phasors, so **
                                                      # is elementwise FPE
            acc += v
        tr += sem[c] * (acc / max(len(idx), 1))
    return tr / max(np.abs(tr).max(), 1e-12), sem


def decode(tr, sem, names, Q, bases, scales):
    """Field per class at each query point; argmax wins. Chunked."""
    V = np.stack([tr / sem[c] for c in names])
    F = np.empty((len(names), len(Q)))
    CH = 2048
    for i0 in range(0, len(Q), CH):
        i1 = min(i0 + CH, len(Q))
        g = np.ones((i1 - i0, HD), np.complex128)
        for a, (B, l) in enumerate(zip(bases, scales)):
            g = g * (B[None, :] ** (Q[i0:i1, a, None] / l))
        F[:, i0:i1] = (V @ np.conj(g).T).real
    return np.array([names[w] for w in F.argmax(0)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--lx", type=float, default=0.45)
    ap.add_argument("--ly", type=float, default=0.27)
    ap.add_argument("--lz", type=float, default=0.35,
                    help="elevation length scale; rooms are ~2.8 m tall so this "
                         "is deliberately finer than the floor scales")
    ap.add_argument("--cut", type=float, default=0.9,
                    help="height above which observations are dropped, for the "
                         "crude control")
    ap.add_argument("--out", default="outputs/height_axis_test.json")
    args = ap.parse_args()
    score05 = load_mod("student_gpu_package/05_score.py", "score05")
    EX = score05.CG_EXCLUDE_6
    res = {}

    print(f"{'scene':<9}{'2D (now)':>10}{'3D +z':>9}{'delta':>8}"
          f"{'2D cut':>9}{'delta':>8}   n_obs")
    print("-" * 66)
    for s in args.scenes:
        d = f"student_gpu_package/handoff/{s}_cgfront"
        E = np.load(f"{d}/eval_points.npz", allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        C = np.load(f"{d}/cg_labels.npz", allow_pickle=True)
        P, cls = C["xyz"].astype(float), C["pred_class"].astype(str)
        # floor axes = the two largest-variance axes, same rule as everywhere
        v = xyz.var(0)
        a, b = sorted(np.argsort(v)[-2:])
        up = ({0, 1, 2} - {a, b}).pop()
        names = sorted(set(cls))
        rng = np.random.default_rng(0)
        Bx = Phasor(dim=HD, seed=11).values
        By = Phasor(dim=HD, seed=12).values
        Bz = Phasor(dim=HD, seed=13).values

        P2 = P[:, [a, b]]
        Q2 = xyz[:, [a, b]]
        P3 = P[:, [a, b, up]]
        Q3 = xyz[:, [a, b, up]]

        tr, sem = bundle(P2, cls, names, [Bx, By], [args.lx, args.ly],
                         args.cap, rng)
        m2 = score05.macc_fmiou(gt, decode(tr, sem, names, Q2, [Bx, By],
                                           [args.lx, args.ly]), exclude=EX)

        tr, sem = bundle(P3, cls, names, [Bx, By, Bz],
                         [args.lx, args.ly, args.lz], args.cap, rng)
        m3 = score05.macc_fmiou(gt, decode(tr, sem, names, Q3, [Bx, By, Bz],
                                           [args.lx, args.ly, args.lz]),
                                exclude=EX)

        keep = P[:, up] < args.cut
        trc, semc = bundle(P2[keep], cls[keep], names, [Bx, By],
                           [args.lx, args.ly], args.cap, rng)
        mc = score05.macc_fmiou(gt, decode(trc, semc, names, Q2, [Bx, By],
                                           [args.lx, args.ly]), exclude=EX)

        res[s] = dict(d2=m2[:2], d3=m3[:2], cut=mc[:2],
                      n=len(P), n_cut=int(keep.sum()))
        print(f"{s:<9}{m2[0]:>10.3f}{m3[0]:>9.3f}{m3[0]-m2[0]:>+8.3f}"
              f"{mc[0]:>9.3f}{mc[0]-m2[0]:>+8.3f}   {len(P)} -> {int(keep.sum())}",
              flush=True)

    print("-" * 66)
    for k, lab in (("d2", "2D, current"), ("d3", "3D, + elevation"),
                   ("cut", "2D, ceiling dropped")):
        print(f"  {lab:<22} mAcc {st.mean(res[s][k][0] for s in res):.3f}"
              f"   F-mIoU {st.mean(res[s][k][1] for s in res):.3f}")
    g3 = st.mean(res[s]["d3"][0] - res[s]["d2"][0] for s in res)
    gc = st.mean(res[s]["cut"][0] - res[s]["d2"][0] for s in res)
    print(f"\nthird axis {g3:+.3f} mAcc;  crude height cut {gc:+.3f}")
    if g3 > gc + 0.01:
        print("The AXIS beats the CUT: elevation is carrying real structure, not "
              "just\nremoving ceiling clutter. Worth encoding properly.")
    elif gc >= g3 - 0.005 and gc > 0.01:
        print("The CUT matches the AXIS: nearly all the gain is just dropping "
              "ceiling\nobservations. A filter is far cheaper than a third "
              "encoded dimension.")
    else:
        print("Neither helps much -- the ceiling-collapse story does not survive.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({k: {kk: list(vv) if isinstance(vv, tuple) else vv
                   for kk, vv in v.items()} for k, v in res.items()},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
