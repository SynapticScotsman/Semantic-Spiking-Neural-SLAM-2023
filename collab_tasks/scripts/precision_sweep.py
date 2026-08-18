"""The OTHER axis of "why 32 KB": bytes per coefficient, not number of them.

`dimension_sweep.py` asks how many dimensions the trace needs. It does not ask
how many bits each one needs -- and on that question the codebase is currently
inconsistent with its own headline number:

  Phasor builds complex128 (vsa.py:51), build_trace accumulates complex128, and
  dimension_sweep.py scores the complex128 array directly. At 4096 dims that is
  64 KB. The "32 KB" figure assumes a complex64 downcast that other modules
  state as the convention (cross_recall.py:192) but that this eval path never
  applies.

So either 32 KB is right and the downcast is free -- in which case it should be
applied and shown to be free -- or it costs something, and the headline number
is describing a trace we have never actually scored. Both outcomes are load
bearing, and neither is currently measured.

The ladder below goes further down than complex64, because "we could be smaller
but don't" deserves a number rather than an assertion. A bundled FHRR trace is
not unit modulus -- magnitudes carry the evidence count -- so phase-only is a
real lossy step, not a free one, and the test says by how much.

  complex128   4096 x 16 B   64 KB   what is actually scored today
  complex64    4096 x  8 B   32 KB   the claimed budget
  8+8 bit      4096 x  2 B    8 KB   phase and log-magnitude, 8 bits each
  phase 8 bit  4096 x  1 B    4 KB   phase only, magnitudes discarded

Run:
  python collab_tasks/scripts/precision_sweep.py --scenes room0 room1 room2

CPU only, runs against handoff artifacts already on disk.
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
                f"torch.{name} was used -- the decode path was assumed pure NumPy.")

    sys.modules["torch"] = _Absent("torch")

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, cap_per_class)


# ------------------------------------------------------------------ quantisers
# Each takes the raw complex128 trace and returns (quantised trace, bytes per
# coefficient). The trace is normalised by its max modulus before quantisation,
# exactly as the decode path already does, so the scale factor is one float that
# rides along and is not counted -- it would be 4 bytes on a 4096-dim trace.

def q_c128(v):
    return v.astype(np.complex128), 16.0


def q_c64(v):
    return v.astype(np.complex64).astype(np.complex128), 8.0


def _quant(a, lo, hi, bits):
    """Uniform mid-rise quantisation of `a` onto `bits` levels spanning [lo, hi]."""
    n = (1 << bits) - 1
    step = (hi - lo) / n if hi > lo else 1.0
    return lo + np.round((np.clip(a, lo, hi) - lo) / step) * step


def q_phase_mag_8_8(v):
    """8 bits of phase + 8 bits of log-magnitude. Log rather than linear because
    bundled magnitudes are heavy tailed -- a few classes dominate the modulus and
    linear levels would spend nearly all of them on the head."""
    ph = _quant(np.angle(v), -np.pi, np.pi, 8)
    m = np.abs(v)
    nz = m[m > 0]
    if nz.size == 0:
        return v * 0.0, 2.0
    lg = np.log10(np.maximum(m, nz.min() * 1e-3))
    lg = _quant(lg, lg.min(), lg.max(), 8)
    return (10.0 ** lg) * np.exp(1j * ph), 2.0


def q_phase_8(v):
    """Phase only, 8 bits. Magnitudes discarded -- every coefficient unit modulus.
    This is the aggressive end: it throws away the evidence counts that bundling
    accumulated, and is the step most likely to actually cost accuracy."""
    return np.exp(1j * _quant(np.angle(v), -np.pi, np.pi, 8)), 1.0


SCHEMES = [
    ("complex128", q_c128),
    ("complex64", q_c64),
    ("phase8+mag8", q_phase_mag_8_8),
    ("phase8", q_phase_8),
]


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


def score_trace(trace, enc, sem, xyz, gt, a, b, hd, grid, lx, ly, score05):
    """Decode and score one already-built (and possibly quantised) trace.

    Note the asymmetry, which is deliberate: only the TRACE is quantised. The
    decode grid and class phasors stay at full precision because they are
    regenerated from a seed at query time and are not part of the stored map --
    quantising them would measure a different claim than the one on the table.
    """
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
    return score05.macc_fmiou(gt, pred, exclude=score05.CG_EXCLUDE_6)[:2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*",
                    default=["room0", "room1", "room2", "office0",
                             "office1", "office2", "office3", "office4"])
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--length-scale", default="0.45,0.27")
    ap.add_argument("--out", default="outputs/precision_sweep.json")
    args = ap.parse_args()

    lx, ly = [float(v) for v in args.length_scale.split(",")]
    hd = args.dim
    score05 = load_mod("student_gpu_package/05_score.py", "score05")
    results = {}

    for s in args.scenes:
        cg = f"{s}_cgfront"
        ep = f"student_gpu_package/handoff/{cg}/eval_points.npz"
        op = f"outputs/replica_{cg}/object_points.json"
        if not (os.path.exists(ep) and os.path.exists(op)):
            print(f"{s}: missing artifacts -- skipping")
            continue
        E = np.load(ep, allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        a, b = sorted(np.argsort(xyz.var(0))[-2:])
        pts = json.load(open(op))["points"]

        print(f"\n{s}: {len(pts)} observations, {len(set(gt))} GT classes, dim {hd}")
        print(f"  {'scheme':>14}{'KB':>7}{'mAcc':>19}{'F-mIoU':>17}")
        rows = []
        for name, q in SCHEMES:
            accs, fms = [], []
            t0 = time.time()
            for sd in range(args.seeds):
                enc = Enc(hd, sd, lx, ly, 20.0)
                sem = class_phasors(sorted({p["cls"] for p in pts}), hd)
                trace = build_trace(cap_per_class(pts, args.max_per_class),
                                    enc, sem, hd)
                trace /= max(np.abs(trace).max(), 1e-12)
                tq, bpc = q(trace)
                m, f = score_trace(tq, enc, sem, xyz, gt, a, b, hd,
                                   args.grid, lx, ly, score05)
                accs.append(m); fms.append(f)
            kb = hd * bpc / 1024
            star = "  <- claimed budget" if name == "complex64" else (
                   "  <- actually scored today" if name == "complex128" else "")
            print(f"  {name:>14}{kb:>7.0f}   {np.mean(accs):>8.3f} +-{np.std(accs):<5.3f}"
                  f"   {np.mean(fms):>7.3f} +-{np.std(fms):<5.3f}"
                  f"  [{time.time()-t0:.0f}s]{star}", flush=True)
            rows.append(dict(scheme=name, bytes_per_coef=bpc, kb=kb,
                             macc=float(np.mean(accs)), macc_sd=float(np.std(accs)),
                             fmiou=float(np.mean(fms)), fmiou_sd=float(np.std(fms))))
        results[s] = rows

    if results:
        print(f"\n{'='*76}\nIS THE complex64 DOWNCAST FREE?\n{'='*76}")
        for name, _ in SCHEMES:
            dm, df = [], []
            for rows in results.values():
                base = next(r for r in rows if r["scheme"] == "complex128")
                r = next(r for r in rows if r["scheme"] == name)
                dm.append(r["macc"] - base["macc"])
                df.append(r["fmiou"] - base["fmiou"])
            print(f"{name:>14}  vs complex128:  mAcc {np.mean(dm):+.3f}   "
                  f"F-mIoU {np.mean(df):+.3f}   (mean over {len(dm)} scenes)")
        print("\nA downcast is 'free' only if BOTH deltas sit inside the seed "
              "spread reported above. Judge on both metrics -- mAcc saturates "
              "where F-mIoU is still moving.")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
