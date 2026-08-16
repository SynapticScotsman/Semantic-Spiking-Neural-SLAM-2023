"""Audit OUR OWN handicaps, the way we audited theirs.

Two bugs in how we scored ConceptGraphs were found on 2026-08-16 and both
flattered us; fixing them took their measured score from 0.271 to 0.402 against
a published 0.406. That audit was entirely one-directional: nobody has checked
what WE are doing to ourselves. Two candidates, both inherited defaults that
nobody chose:

  1. GRID QUANTISATION. We evaluate the class fields on a 96x96 grid and give
     each eval point its cell's label -- a 0.08 m rounding on every prediction.
     Their labels are per-point and carry no such quantisation. The earlier grid
     control only showed that the RANKING of length scales was stable at 96 vs
     192; it never asked whether the quantisation costs absolute accuracy. A
     gridless decode evaluates the field at each eval point's true (x, y) and
     removes the artefact entirely. It is also CHEAPER: ~30k field evaluations
     instead of 9216 grid cells plus a nearest-cell lookup.

  2. THE PER-CLASS CAP. cap_per_class(pts, 400) keeps at most 400 observations
     per class, which on room0 discards roughly half of 13,124 observations
     before the trace is built. The "saturates by 200" finding that justified it
     was measured at the inherited lambda 0.6 against the labels produced by the
     two scoring bugs -- exactly the kind of conclusion that does not survive a
     configuration change, and we have already been caught by that three times.
     The dimension sweep also showed we have capacity headroom, so discarding
     data has no obvious justification.

Note the two are not equivalent as claims. Gridless is FREE: it removes an
artefact without changing the representation, so any gain was always ours.
Raising the cap is a REAL TRADE -- more observations bundled into the same
32 KB -- so a gain there means the old saturation finding was wrong, not that
something new was discovered.

    python collab_tasks/scripts/self_audit.py

CPU only, reads handoff artifacts, no GPU and no Colab.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import time
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

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, cap_per_class)

HD = 4096
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]


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


def run(pts, xyz, gt, a, b, cap, grid, score05, lx=0.45, ly=0.27):
    enc = Enc(HD, 0, lx, ly, 20.0)
    sem = class_phasors(sorted({p["cls"] for p in pts}), HD)
    tr = build_trace(cap_per_class(pts, cap), enc, sem, HD)
    tr /= max(np.abs(tr).max(), 1e-12)
    names = sorted(sem)
    xs, ys = xyz[:, a], xyz[:, b]

    if grid:                       # current behaviour: quantise to grid cells
        gx = np.linspace(xs.min(), xs.max(), grid)
        gy = np.linspace(ys.min(), ys.max(), grid)
        PX = np.conj(enc.Bx.values[None, :] ** (gx[:, None] / lx))
        PY = np.conj(enc.By.values[None, :] ** (gy[:, None] / ly))
        F = np.empty((len(names), grid * grid))
        for n, c in enumerate(names):
            v = tr / sem[c]
            F[n] = ((PX * v[None, :]) @ PY.T).T.reshape(-1).real
        ix = np.clip(np.searchsorted(gx, xs), 0, grid - 1)
        iy = np.clip(np.searchsorted(gy, ys), 0, grid - 1)
        pred = np.array([names[w] for w in F.argmax(0)[iy * grid + ix]])
    else:
        # Gridless: evaluate the field at each eval point's true (x, y).
        # Materialising the whole (n_eval, HD) complex array is ~2 GB for 30k
        # points at HD 4096, which thrashes. Chunk it: 2048 points at a time is
        # ~130 MB and the result is identical, since each point is independent.
        V = np.stack([tr / sem[c] for c in names])      # (C, HD)
        F = np.empty((len(names), len(xs)))
        CH = 2048
        for i0 in range(0, len(xs), CH):
            i1 = min(i0 + CH, len(xs))
            gx = np.conj(enc.Bx.values[None, :] ** (xs[i0:i1, None] / lx))
            gy = np.conj(enc.By.values[None, :] ** (ys[i0:i1, None] / ly))
            F[:, i0:i1] = (V @ (gx * gy).T).real        # (C, chunk)
        pred = np.array([names[w] for w in F.argmax(0)])
    m, f, _ = score05.macc_fmiou(gt, pred, exclude=score05.CG_EXCLUDE_6)
    return float(m), float(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--out", default="outputs/self_audit.json")
    args = ap.parse_args()
    score05 = load_mod("student_gpu_package/05_score.py", "score05")

    print(f"{'scene':<9}{'grid96/400':>12}{'gridless':>10}{'delta':>8}"
          f"{'gl/800':>9}{'gl/1600':>9}{'gl/ALL':>8}{'obs':>8}")
    print("-" * 72)
    res = {}
    for s in args.scenes:
        ep = f"student_gpu_package/handoff/{s}_cgfront/eval_points.npz"
        op = f"outputs/replica_{s}_cgfront/object_points.json"
        if not (os.path.exists(ep) and os.path.exists(op)):
            print(f"{s:<9} missing artifacts")
            continue
        E = np.load(ep, allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        v = xyz.var(0)
        a, b = sorted(np.argsort(v)[-2:])
        pts = json.load(open(op))["points"]
        t0 = time.time()
        base = run(pts, xyz, gt, a, b, 400, 96, score05)
        gl = run(pts, xyz, gt, a, b, 400, None, score05)
        c8 = run(pts, xyz, gt, a, b, 800, None, score05)
        c16 = run(pts, xyz, gt, a, b, 1600, None, score05)
        ca = run(pts, xyz, gt, a, b, 10 ** 9, None, score05)
        res[s] = dict(base=base, gridless=gl, cap800=c8, cap1600=c16, capall=ca,
                      n_obs=len(pts))
        print(f"{s:<9}{base[0]:>12.3f}{gl[0]:>10.3f}{gl[0]-base[0]:>+8.3f}"
              f"{c8[0]:>9.3f}{c16[0]:>9.3f}{ca[0]:>8.3f}{len(pts):>8}"
              f"   [{time.time()-t0:.0f}s]", flush=True)

    if not res:
        raise SystemExit("no scenes ran")
    print("-" * 72)
    for k, lab in [("base", "grid 96, cap 400  (current)"),
                   ("gridless", "gridless, cap 400"),
                   ("cap800", "gridless, cap 800"),
                   ("cap1600", "gridless, cap 1600"),
                   ("capall", "gridless, no cap")]:
        print(f"  {lab:<30} mAcc {st.mean(res[s][k][0] for s in res):.3f}"
              f"   F-mIoU {st.mean(res[s][k][1] for s in res):.3f}")

    g = st.mean(res[s]["gridless"][0] - res[s]["base"][0] for s in res)
    c = st.mean(res[s]["capall"][0] - res[s]["gridless"][0] for s in res)
    print(f"\ngridless is worth {g:+.3f} mAcc — a FREE gain if positive, since it "
          f"only removes\na 0.08 m rounding artefact we imposed on ourselves and "
          "their labels never had.")
    print(f"lifting the cap is worth {c:+.3f} mAcc on top — NOT free: it bundles "
          f"more\nobservations into the same 32 KB, so a gain means the old "
          '"saturates by 200"\nfinding was measured in a configuration that no '
          "longer holds.")
    print("\nreference: ConceptGraphs 0.402 on the same 8 scenes; the shared "
          "frontend\ncaps mAcc at 0.609, so quote progress against that ceiling "
          "too.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
