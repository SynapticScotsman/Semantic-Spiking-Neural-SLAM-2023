"""Is 32 KB actually the binding constraint, or just the number we inherited?

Everything this project claims rests on one figure: a fixed-size trace of 4096
complex64 = 32 KB, against ConceptGraphs' unbounded point-cloud map. Every
experiment so far has tuned what happens INSIDE that budget -- length scale,
decode grid, insertion cap, decode rule -- and not one has asked whether the
budget itself is where the loss comes from. It has been 4096 since the code was
written.

That is the first-order question and it is unmeasured, which makes every
second-order result provisional. Two outcomes, both useful:

  FLAT above ~4096   we are not capacity-limited. The trace holds everything the
                     frontend gives it, the deficit is decode or frontend, and
                     "32 KB" becomes a strong claim rather than an arbitrary one
                     -- we can even show the curve and say where it saturates.

  STEEP              we ARE capacity-limited, the honest framing changes, and
                     every length-scale gain measured so far is a small effect on
                     top of a large constraint we had not looked at.

The x-axis is what a reviewer will ask for anyway: accuracy against bytes, with
their point-cloud map's size marked for scale.

    python collab_tasks/scripts/dimension_sweep.py --scene room0_cgfront --gt-scene room0
    python collab_tasks/scripts/dimension_sweep.py --scenes room0 room1 room2

CPU only, no GPU, no Colab -- runs against handoff artifacts already on disk.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import types

    class _Absent(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(
                f"torch.{name} was used — the decode path was assumed pure NumPy.")

    sys.modules["torch"] = _Absent("torch")

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, cap_per_class)

# 4096 complex64 = 4096 * 8 bytes = 32 KB. Sweep well below and well above so
# the shape of the curve is visible, not just two points.
DIMS = [512, 1024, 2048, 4096, 8192, 16384]


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path]
    spec.loader.exec_module(mod)
    return mod


class Enc(ClassroomEncoders):
    def __init__(self, hd, seed, lx, ly, tl):
        super().__init__(hd, seed, 1.0, tl)
        self.lx, self.ly = lx, ly

    def ctx_pos(self, x, y):
        return (self.Bx ** float(x / self.lx)) * (self.By ** float(y / self.ly))


def score_at_dim(pts, xyz, gt, a, b, hd, grid, cap, lx, ly, seed, score05):
    """One (dimension, seed) evaluation. Separable grid build: phi(x,y) factorises
    into an x-only and a y-only term, so this costs 2*grid power ops, not grid**2
    (verified against the naive build in lambda_transfer_test.py --verify)."""
    enc = Enc(hd, seed, lx, ly, 20.0)
    sem = class_phasors(sorted({p["cls"] for p in pts}), hd)
    trace = build_trace(cap_per_class(pts, cap), enc, sem, hd)
    trace /= max(np.abs(trace).max(), 1e-12)

    xs, ys = xyz[:, a], xyz[:, b]
    gx = np.linspace(xs.min(), xs.max(), grid)
    gy = np.linspace(ys.min(), ys.max(), grid)
    names = sorted(sem)
    PX = np.conj(enc.Bx.values[None, :] ** (gx[:, None] / lx))
    PY = np.conj(enc.By.values[None, :] ** (gy[:, None] / ly))
    F = np.empty((len(names), grid * grid))
    for n, c in enumerate(names):
        v = trace / sem[c]
        F[n] = ((PX * v[None, :]) @ PY.T).T.reshape(-1).real
    ix = np.clip(np.searchsorted(gx, xs), 0, grid - 1)
    iy = np.clip(np.searchsorted(gy, ys), 0, grid - 1)
    pred = np.array([names[w] for w in F.argmax(0)[iy * grid + ix]])
    macc, fmiou, _ = score05.macc_fmiou(gt, pred, exclude=score05.CG_EXCLUDE_6)
    return macc, fmiou


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=["room0", "room1", "room2"],
                    help="base scene names; the cgfront variant is used")
    ap.add_argument("--dims", default=",".join(str(d) for d in DIMS))
    ap.add_argument("--seeds", type=int, default=3,
                    help="encoder seeds per dimension — a single draw at low "
                         "dimension is noisy, and the whole point is the SHAPE "
                         "of the curve, not one point on it")
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--length-scale", default="0.45,0.27")
    ap.add_argument("--out", default="outputs/dimension_sweep.json")
    args = ap.parse_args()

    lx, ly = [float(v) for v in args.length_scale.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    score05 = load_mod("student_gpu_package/05_score.py", "score05")
    results = {}

    for s in args.scenes:
        cg = f"{s}_cgfront"
        ep = f"student_gpu_package/handoff/{cg}/eval_points.npz"
        op = f"outputs/replica_{cg}/object_points.json"
        if not (os.path.exists(ep) and os.path.exists(op)):
            print(f"{s}: missing {ep if not os.path.exists(ep) else op} — skipping")
            continue
        E = np.load(ep, allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        var = xyz.var(0)
        a, b = sorted(np.argsort(var)[-2:])
        pts = json.load(open(op))["points"]

        print(f"\n{s}: {len(pts)} observations, {len(set(gt))} GT classes")
        print(f"  {'dim':>7}{'KB':>8}{'mAcc':>19}{'F-mIoU':>17}")
        rows = []
        for hd in dims:
            accs, fms = [], []
            t0 = time.time()
            for sd in range(args.seeds):
                m, f = score_at_dim(pts, xyz, gt, a, b, hd, args.grid,
                                    args.max_per_class, lx, ly, sd, score05)
                accs.append(m); fms.append(f)
            kb = hd * 8 / 1024
            star = "  <- current" if hd == 4096 else ""
            print(f"  {hd:>7}{kb:>8.0f}   {np.mean(accs):>8.3f} +-{np.std(accs):<5.3f}"
                  f"   {np.mean(fms):>7.3f} +-{np.std(fms):<5.3f}"
                  f"  [{time.time()-t0:.0f}s]{star}", flush=True)
            rows.append(dict(dim=hd, kb=kb, macc=float(np.mean(accs)),
                             macc_sd=float(np.std(accs)), fmiou=float(np.mean(fms)),
                             fmiou_sd=float(np.std(fms))))
        results[s] = rows

        # Where does it saturate? Report the smallest dimension within one
        # standard deviation of the best, which is the honest "you need at least
        # this much" number rather than the argmax of a noisy curve.
        best = max(r["macc"] for r in rows)
        tol = max(r["macc_sd"] for r in rows)
        enough = [r for r in rows if r["macc"] >= best - tol]
        if enough:
            e = min(enough, key=lambda r: r["dim"])
            print(f"  -> within 1 sd of best from dim {e['dim']} ({e['kb']:.0f} KB) "
                  f"onward; best {best:.3f}")

    if results:
        print(f"\n{'='*76}\nIS 32 KB THE BINDING CONSTRAINT?\n{'='*76}")
        for s, rows in results.items():
            at4096 = next(r["macc"] for r in rows if r["dim"] == 4096)
            top = max(r["macc"] for r in rows)
            head = max(r["dim"] for r in rows)
            attop = next(r["macc"] for r in rows if r["dim"] == head)
            print(f"{s:<16} 4096 -> {at4096:.3f}   best {top:.3f}   "
                  f"{head} -> {attop:.3f}   headroom {attop-at4096:+.3f}")
        gains = []
        for rows in results.values():
            a4 = next(r["macc"] for r in rows if r["dim"] == 4096)
            hi = max(r["dim"] for r in rows)
            gains.append(next(r["macc"] for r in rows if r["dim"] == hi) - a4)
        g = float(np.mean(gains))
        print(f"\nmean mAcc gained by going 4096 -> {max(dims)} "
              f"({4096*8//1024} KB -> {max(dims)*8//1024} KB): {g:+.3f}")
        if g < 0.01:
            print("FLAT: we are NOT capacity-limited at 32 KB. Quadrupling the "
                  "budget buys\nnothing, so the deficit is the frontend or the "
                  "decode, and the fixed-size\nclaim is strong -- 32 KB is "
                  "sufficient, not merely what we happened to pick.")
        elif g > 0.03:
            print("STEEP: we ARE capacity-limited. Every decode result measured "
                  "so far is a\nsecond-order effect on top of a first-order "
                  "constraint, and the 32 KB\nfigure needs justifying rather "
                  "than asserting.")
        else:
            print("MILD: some headroom, not decisive. Report the curve rather "
                  "than a verdict.")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
