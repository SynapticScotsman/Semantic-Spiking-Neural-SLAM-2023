"""Which representation degrades more gracefully when the frontend gets worse?

We now know 32 KB is not capacity-limited (dimension_sweep.py: mAcc moves ~0.06
across a 32x memory range) and that we trail ConceptGraphs by 0.078 mAcc with a
faithful reproduction of their system (0.324 vs 0.402, 8 scenes). "We are 0.078
behind" is a weak claim. The interesting question for a compressed associative
memory is not peak accuracy but how it FAILS: given a worse frontend, does a
fixed-size trace lose accuracy faster or slower than an explicit list of
labelled points?

That is on-thesis, fair to them, and answerable without touching their pipeline.

WHAT IS AND IS NOT BEING COMPARED. We cannot perturb inside their mapping stage
without the SAM detections (lost with a reclaimed Colab VM, ~6 h to regenerate).
So this degrades the SHARED INPUT and re-runs each backend's own prediction rule
over it:

  theirs : their labelled points (cg_labels.npz) -> nearest-neighbour transfer
           to the eval points. This IS their scoring rule, so a degraded version
           of it is a faithful analogue of a degraded map.
  ours   : their observation stream (object_points.json) -> 32 KB trace ->
           per-class field -> argmax.

The two underlying sets differ in size, so the same degradation RATE is applied
to each rather than the same absolute count. This measures the representations,
not the implementations, and that distinction belongs in any writeup.

Three degradations, each simulating a real frontend failure:

  drop     keep a fraction of points/observations   (missed detections, sparse
                                                     coverage, shorter traverse)
  label    flip a fraction of labels to a wrong class (CLIP misassignment)
  jitter   add Gaussian noise to positions in metres  (pose drift, depth error)

    python collab_tasks/scripts/degradation_sweep.py
    python collab_tasks/scripts/degradation_sweep.py --scenes room0 --seeds 3

CPU only. Reads the handoff artifacts; no GPU, no Colab, no models.
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
            raise RuntimeError(f"torch.{name} used; decode path assumed pure NumPy")

    sys.modules["torch"] = _Absent("torch")

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, cap_per_class)

HD = 4096


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


def ours(pts, xyz, a, b, grid, cap, lx, ly, seed, names_all):
    """32 KB trace -> per-class field -> argmax. Separable grid build."""
    if not pts:
        return np.array(["__none__"] * len(xyz))
    enc = Enc(HD, 0, lx, ly, 20.0)
    sem = class_phasors(sorted({p["cls"] for p in pts}), HD)
    trace = build_trace(cap_per_class(pts, cap), enc, sem, HD)
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
    return np.array([names[w] for w in F.argmax(0)[iy * grid + ix]])


def theirs(cg_xyz, cg_lab, xyz, score05):
    """Their own rule: nearest labelled point wins."""
    if len(cg_xyz) == 0:
        return np.array(["__none__"] * len(xyz))
    lab = score05.transfer(cg_xyz, cg_lab.astype(object), xyz)
    return np.array([v if v is not None else "__none__" for v in lab])


def degrade_pts(pts, kind, level, rng, classes):
    """Apply one degradation to a list of observation dicts."""
    if kind == "drop":
        if level >= 1.0:
            return pts
        keep = rng.random(len(pts)) < level
        return [p for p, k in zip(pts, keep) if k]
    if kind == "label":
        if level <= 0:
            return pts
        out = []
        for p in pts:
            if rng.random() < level and len(classes) > 1:
                alt = [c for c in classes if c != p["cls"]]
                out.append(dict(p, cls=alt[rng.integers(len(alt))]))
            else:
                out.append(p)
        return out
    if kind == "jitter":
        if level <= 0:
            return pts
        return [dict(p, x=p["x"] + rng.normal(0, level),
                     y=p["y"] + rng.normal(0, level)) for p in pts]
    raise ValueError(kind)


def degrade_map(cg_xyz, cg_lab, kind, level, rng, classes):
    if kind == "drop":
        if level >= 1.0:
            return cg_xyz, cg_lab
        keep = rng.random(len(cg_xyz)) < level
        return cg_xyz[keep], cg_lab[keep]
    if kind == "label":
        if level <= 0:
            return cg_xyz, cg_lab
        lab = cg_lab.copy()
        hit = rng.random(len(lab)) < level
        for i in np.flatnonzero(hit):
            alt = [c for c in classes if c != lab[i]]
            if alt:
                lab[i] = alt[rng.integers(len(alt))]
        return cg_xyz, lab
    if kind == "jitter":
        if level <= 0:
            return cg_xyz, cg_lab
        n = cg_xyz.copy()
        n[:, :2] += rng.normal(0, level, size=(len(n), 2))
        return n, cg_lab
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=["room0", "room2", "office4"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--length-scale", default="0.45,0.27")
    ap.add_argument("--out", default="outputs/degradation_sweep.json")
    args = ap.parse_args()
    lx, ly = [float(v) for v in args.length_scale.split(",")]
    score05 = load_mod("student_gpu_package/05_score.py", "score05")

    SWEEPS = [("drop",   [1.0, 0.5, 0.25, 0.1, 0.05]),
              ("label",  [0.0, 0.1, 0.25, 0.5]),
              ("jitter", [0.0, 0.05, 0.1, 0.25, 0.5])]
    results = {}

    for s in args.scenes:
        cg = f"{s}_cgfront"
        ep = f"student_gpu_package/handoff/{cg}/eval_points.npz"
        op = f"outputs/replica_{cg}/object_points.json"
        cp = f"student_gpu_package/handoff/{cg}/cg_labels.npz"
        if not all(os.path.exists(p) for p in (ep, op, cp)):
            print(f"{s}: missing artifacts, skipping")
            continue
        E = np.load(ep, allow_pickle=True)
        xyz, gt = E["xyz"], E["gt_class"].astype(str)
        var = xyz.var(0)
        a, b = sorted(np.argsort(var)[-2:])
        pts0 = json.load(open(op))["points"]
        C = np.load(cp, allow_pickle=True)
        cg_xyz0, cg_lab0 = C["xyz"], C["pred_class"].astype(str)
        classes = sorted(set(cg_lab0) | {p["cls"] for p in pts0})

        print(f"\n{'='*78}\n{s}: {len(pts0)} observations, {len(cg_xyz0)} of their "
              f"labelled points, {len(set(gt))} GT classes\n{'='*78}")
        results[s] = {}
        for kind, levels in SWEEPS:
            print(f"\n  {kind}:")
            unit = {"drop": "kept", "label": "corrupted", "jitter": "sigma (m)"}[kind]
            print(f"    {unit:>12}{'theirs':>10}{'ours':>9}{'gap':>8}"
                  f"{'theirs rel':>12}{'ours rel':>10}")
            rows = []
            base_t = base_o = None
            for lv in levels:
                tm, om = [], []
                for sd in range(args.seeds):
                    rng = np.random.default_rng(1000 * sd + 7)
                    dp = degrade_pts(pts0, kind, lv, rng, classes)
                    rng = np.random.default_rng(1000 * sd + 7)
                    dx, dl = degrade_map(cg_xyz0, cg_lab0, kind, lv, rng, classes)
                    pt = theirs(dx, dl, xyz, score05)
                    po = ours(dp, xyz, a, b, args.grid, args.max_per_class,
                              lx, ly, sd, classes)
                    tm.append(score05.macc_fmiou(gt, pt,
                                                 exclude=score05.CG_EXCLUDE_6)[0])
                    om.append(score05.macc_fmiou(gt, po,
                                                 exclude=score05.CG_EXCLUDE_6)[0])
                t, o = float(np.mean(tm)), float(np.mean(om))
                if base_t is None:
                    base_t, base_o = t, o
                rt = t / base_t if base_t else float("nan")
                ro = o / base_o if base_o else float("nan")
                print(f"    {lv:>12.2f}{t:>10.3f}{o:>9.3f}{o-t:>+8.3f}"
                      f"{rt:>12.1%}{ro:>10.1%}", flush=True)
                rows.append(dict(level=lv, theirs=t, ours=o,
                                 theirs_rel=rt, ours_rel=ro))
            results[s][kind] = rows
            worst = rows[-1]
            verdict = ("OURS degrades more gracefully"
                       if worst["ours_rel"] > worst["theirs_rel"] + 0.02 else
                       "THEIRS degrades more gracefully"
                       if worst["theirs_rel"] > worst["ours_rel"] + 0.02 else
                       "comparable")
            print(f"    at the worst level: theirs keeps {worst['theirs_rel']:.0%}, "
                  f"ours keeps {worst['ours_rel']:.0%}  -> {verdict}")

    if results:
        print(f"\n{'='*78}\nWHICH REPRESENTATION FAILS MORE GRACEFULLY?\n{'='*78}")
        for kind, _ in SWEEPS:
            tr = [results[s][kind][-1]["theirs_rel"] for s in results
                  if kind in results[s]]
            orl = [results[s][kind][-1]["ours_rel"] for s in results
                   if kind in results[s]]
            if tr:
                print(f"  {kind:<8} at worst level, mean retained:  "
                      f"theirs {np.mean(tr):.0%}   ours {np.mean(orl):.0%}   "
                      f"{'OURS' if np.mean(orl) > np.mean(tr) else 'THEIRS'} better")
        print("\nRetained fraction is relative to each system's OWN undegraded "
              "score, so this\nmeasures graceful failure, not peak accuracy. "
              "Both are worth reporting: we\ntrail on peak (0.324 vs 0.402) and "
              "this asks whether that ordering holds up\nwhen the frontend is "
              "poor -- the regime a real robot operates in.")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
