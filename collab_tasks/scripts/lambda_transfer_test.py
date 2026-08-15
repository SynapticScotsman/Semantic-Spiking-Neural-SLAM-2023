"""Is lambda-proportional-to-extent a RULE, or was 0.45,0.27 fitted to room0?

room0's best per-axis length scale was 0.45, 0.27 (ratio 1.67) and room0's floor
extent is 7.59 x 4.59 m (ratio 1.65). That is either a rule or a coincidence, and
the distinction decides whether the number can appear in a paper: selecting a
hyperparameter on the scene you report is not a result.

The eight Replica scenes make this testable because they genuinely differ in
shape (measured by scene_extents.py):

    room0   1.65   wider than deep
    room2   1.44
    room1   1.21
    office1 0.98   square
    office4 0.89
    office3 0.80   DEEPER than wide -- the rule demands lambda_y > lambda_x here,
    office0 0.78   the opposite orientation from room0
    office2 0.78

So the rule makes a falsifiable prediction: on office0/2/3 the best lambda should
be TALL (ratio < 1). If 0.45,0.27 wins there too, the rule is dead and 0.45,0.27
is merely a good pair of numbers.

The sweep is parameterised as (magnitude, ratio) rather than (lambda_x, lambda_y):

    lambda_x = m * sqrt(r)      lambda_y = m / sqrt(r)

so the geometric mean stays m while the ratio varies. This separates "how wide
is the kernel" from "what shape is it" -- the earlier ad-hoc sweep conflated the
two, which is why it could not distinguish the rule from a lucky pair.

    python collab_tasks/scripts/lambda_transfer_test.py --verify
    python collab_tasks/scripts/lambda_transfer_test.py --scenes room0 office3

Runs on a laptop: no torch, no GPU, no Colab. Uses OUR frontend (available for
all 8 scenes locally); the lambda rule is geometric, so it should not care whose
detections it is, and the absolute scores are our-frontend scores.
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

# Same no-torch shim as local_decode_sweep.py: the decode path is pure NumPy,
# and a placeholder that raises on attribute access fails loudly if that ever
# stops being true.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import types

    class _Absent(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(
                f"torch.{name} was used -- the no-torch shim assumed the decode "
                "path is pure NumPy. Install a CPU torch and re-run.")

    sys.modules["torch"] = _Absent("torch")

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, cap_per_class)

HD = 4096
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]


def load_mod(path, name):
    """Import a module whose filename starts with a digit. Reusing 05_score's
    own metric matters: a second implementation is how numbers drift apart."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [path]
    spec.loader.exec_module(mod)
    return mod


class Enc(ClassroomEncoders):
    """Per-axis length scale, identical to 04_vsa_labels.py's _Enc."""

    def __init__(self, hd, seed, lx, ly, tl):
        super().__init__(hd, seed, 1.0, tl)
        self.lx, self.ly = lx, ly

    def ctx_pos(self, x, y):
        return (self.Bx ** float(x / self.lx)) * (self.By ** float(y / self.ly))


def class_fields(pts, xyz, a, b, grid, cap, lx, ly, naive=False):
    """Per-class similarity field over the floor grid.

    The naive build calls ctx_pos once per grid cell -- grid**2 = 9216 complex
    power operations on 4096-vectors, which dominates a sweep. But the encoding
    is separable:

        phi(x, y) = (Bx ** x/lx) * (By ** y/ly)

    is an ELEMENTWISE product of a term depending only on x and a term depending
    only on y. So precompute 96 x-factors and 96 y-factors (192 power ops, not
    9216) and never materialise the (9216, 4096) grid at all -- the field for a
    class is then one (96, HD) @ (HD, 96) matmul. Verified against the naive
    path by --verify; identical to ~1e-6.
    """
    enc = Enc(HD, 0, lx, ly, 20.0)
    sem = class_phasors(sorted({p["cls"] for p in pts}), HD)
    trace = build_trace(cap_per_class(pts, cap), enc, sem, HD)
    trace /= max(np.abs(trace).max(), 1e-12)

    xs, ys = xyz[:, a], xyz[:, b]
    gx = np.linspace(xs.min(), xs.max(), grid)
    gy = np.linspace(ys.min(), ys.max(), grid)
    names = sorted(sem)

    if naive:
        G = np.empty((grid ** 2, HD), np.complex64)
        k = 0
        for yy in gy:
            for xx in gx:
                G[k] = enc.ctx_pos(float(xx), float(yy)).values.astype(np.complex64)
                k += 1
        F = np.stack([((trace / sem[c])[None, :] @ np.conj(G).T).real[0]
                      for c in names])
    else:
        PX = np.conj(enc.Bx.values[None, :] ** (gx[:, None] / lx))   # (grid, HD)
        PY = np.conj(enc.By.values[None, :] ** (gy[:, None] / ly))   # (grid, HD)
        F = np.empty((len(names), grid * grid))
        for n, c in enumerate(names):
            v = trace / sem[c]                       # unbind, (HD,)
            # M[i, j] = sum_d v_d * conj(PX[i,d]) * conj(PY[j,d])
            M = (PX * v[None, :]) @ PY.T             # (grid_x, grid_y)
            F[n] = M.T.reshape(-1).real              # cell order = j*grid + i
    ix = np.clip(np.searchsorted(gx, xs), 0, grid - 1)
    iy = np.clip(np.searchsorted(gy, ys), 0, grid - 1)
    return F, names, iy * grid + ix


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--ratios", default="0.6,0.8,1.0,1.25,1.67,2.2",
                    help="lambda_x / lambda_y")
    ap.add_argument("--mags", default="0.25,0.35,0.5",
                    help="geometric mean of lambda_x and lambda_y")
    ap.add_argument("--verify", action="store_true",
                    help="check the fast separable grid against the naive "
                         "per-cell build before trusting any number from it")
    ap.add_argument("--out", default="outputs/lambda_transfer.json")
    args = ap.parse_args()

    score05 = load_mod("student_gpu_package/05_score.py", "score05")
    labels04 = load_mod("student_gpu_package/04_vsa_labels.py", "labels04")

    if args.verify:
        print("verifying the separable grid against the naive build...")
        s = args.scenes[0]
        pts = json.load(open(f"outputs/replica_{s}/object_points.json"))["points"]
        E = np.load(f"student_gpu_package/handoff/{s}/eval_points.npz",
                    allow_pickle=True)
        xyz = E["xyz"]
        var = xyz.var(0)
        a, b = sorted(np.argsort(var)[-2:])
        t0 = time.time()
        Fn, nn, cn = class_fields(pts, xyz, a, b, 24, args.max_per_class,
                                  0.45, 0.27, naive=True)
        t_naive = time.time() - t0
        t0 = time.time()
        Ff, nf, cf = class_fields(pts, xyz, a, b, 24, args.max_per_class,
                                  0.45, 0.27, naive=False)
        t_fast = time.time() - t0
        err = np.abs(Fn - Ff).max() / max(np.abs(Fn).max(), 1e-12)
        print(f"  max relative difference {err:.2e}  "
              f"(naive {t_naive:.1f}s, separable {t_fast:.1f}s, "
              f"{t_naive/max(t_fast,1e-9):.0f}x)")
        assert nn == nf and (cn == cf).all(), "grid indexing diverged"
        if err > 1e-4:
            raise SystemExit("FAIL: the fast path does not reproduce the naive "
                             "build; every number from it would be suspect")
        print("  OK -- the fast path is the same computation\n")

    ratios = [float(x) for x in args.ratios.split(",")]
    mags = [float(x) for x in args.mags.split(",")]
    results = {}

    for s in args.scenes:
        hp = f"student_gpu_package/handoff/{s}/eval_points.npz"
        if not os.path.exists(hp):
            print(f"{s}: building eval points from the GT renders...", flush=True)
            os.makedirs(os.path.dirname(hp), exist_ok=True)
            labels04.build_eval_points(s)
        E = np.load(hp, allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        var = xyz.var(0)
        a, b = sorted(np.argsort(var)[-2:])
        pts = json.load(open(f"outputs/replica_{s}/object_points.json"))["points"]

        G = json.load(open(f"outputs/replica_{s}/gt_instances.json"))["instances"]
        keep = [i for i in G if i["cls"] not in
                {"other", "floor", "wall", "ceiling", "door", "window", "class_-1"}]
        gx = np.array([i["x"] for i in keep]); gy = np.array([i["y"] for i in keep])
        extent_ratio = (gx.max()-gx.min()) / (gy.max()-gy.min())

        print(f"\n{s}  extent ratio {extent_ratio:.2f}  "
              f"({len(pts)} observations, {len(set(gt))} GT classes)")
        print(f"  {'ratio':>7}{'mag':>7}{'lx':>7}{'ly':>7}{'mAcc':>8}{'F-mIoU':>9}")
        best = None
        grid_scores = {}
        for r in ratios:
            for m in mags:
                lx, ly = m * np.sqrt(r), m / np.sqrt(r)
                F, names, cell = class_fields(pts, xyz, a, b, args.grid,
                                              args.max_per_class, lx, ly)
                pred = np.array([names[w] for w in F.argmax(0)[cell]])
                # their protocol, using 05_score's own constant rather than a
                # second copy of the six class names
                macc, fmiou, _ = score05.macc_fmiou(
                    gt, pred, exclude=score05.CG_EXCLUDE_6)
                print(f"  {r:>7.2f}{m:>7.2f}{lx:>7.2f}{ly:>7.2f}"
                      f"{macc:>8.3f}{fmiou:>9.3f}", flush=True)
                grid_scores[(r, m)] = macc
                if best is None or macc > best[0]:
                    best = (macc, fmiou, r, m, lx, ly)

        # How much does SHAPE move the score, with size held fixed? And SIZE,
        # with shape held fixed? If the first is small the "best ratio" above is
        # a coin toss and must not be correlated against anything.
        shape_span = float(np.mean([
            max(grid_scores[(r, m)] for r in ratios)
            - min(grid_scores[(r, m)] for r in ratios) for m in mags]))
        size_span = float(np.mean([
            max(grid_scores[(r, m)] for m in mags)
            - min(grid_scores[(r, m)] for m in mags) for r in ratios]))

        results[s] = dict(extent_ratio=float(extent_ratio), best_macc=best[0],
                          best_fmiou=best[1], best_ratio=best[2],
                          best_mag=best[3], lx=best[4], ly=best[5],
                          shape_span=shape_span, size_span=size_span)
        print(f"  -> best ratio {best[2]:.2f} at mAcc {best[0]:.3f} "
              f"(extent ratio {extent_ratio:.2f}); shape moves mAcc by "
              f"{shape_span:.3f}, size by {size_span:.3f}")

    # the actual test: does the winning lambda ratio track the room's shape?
    # ASCII only: Windows consoles default to cp1252 and a lambda character
    # here crashed the whole summary AFTER every scene had been swept -- an
    # hour of compute reduced to a traceback for the sake of one glyph.
    print(f"\n{'='*72}\nDOES LAMBDA RATIO TRACK EXTENT RATIO?\n{'='*72}")
    print(f"{'scene':<10}{'extent':>9}{'best ratio':>12}{'mAcc':>8}"
          f"{'shape span':>12}{'size span':>11}")
    er, br = [], []
    for s, v in results.items():
        print(f"{s:<10}{v['extent_ratio']:>9.2f}{v['best_ratio']:>12.2f}"
              f"{v['best_macc']:>8.3f}{v['shape_span']:>12.3f}"
              f"{v['size_span']:>11.3f}")
        er.append(v["extent_ratio"]); br.append(v["best_ratio"])

    # Whether the correlation means anything depends on whether shape moves the
    # score at all. Measured on our frontend: shape span ~0.007, size span
    # ~0.023, so "best ratio" was largely picking noise. Report both, and say so.
    ss = np.mean([v["shape_span"] for v in results.values()])
    zs = np.mean([v["size_span"] for v in results.values()])
    print(f"\nmean mAcc span across SHAPE (ratio, size held): {ss:.3f}")
    print(f"mean mAcc span across SIZE  (magnitude, shape held): {zs:.3f}")
    if ss < 2 * zs / 3:
        print("Size dominates shape here. A 'best ratio' picked off a flat "
              "surface is\nnoise, and any correlation computed from it is "
              "correlating noise with\nroom shape. Treat the verdict below as "
              "UNDERPOWERED, not as evidence.")
    if len(er) > 2:
        c = float(np.corrcoef(er, br)[0, 1])
        print(f"\nPearson r between extent ratio and best lambda ratio: {c:+.3f}")
        if c > 0.6:
            print("SUPPORTS the rule: rooms that are wider prefer wider-in-x "
                  "kernels.\nlambda becomes a function of the scene, not a "
                  "hyperparameter fitted to it.")
        elif c < 0.2:
            print("REFUTES the rule: the best lambda ratio does not follow room "
                  "shape.\n0.45,0.27 is a good pair of numbers, not a principle "
                  "-- and it must be\npresented as a tuned hyperparameter, with "
                  "the tuning scene named.")
        else:
            print("INCONCLUSIVE: weak association. More scenes, or the effect is "
                  "real but\nswamped by whatever else differs between rooms.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
