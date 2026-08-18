"""Shared machinery for the batch-1 gap-closing screen.

Every harness goes through this module so that (a) the baseline is one
computation, not six copies, and (b) mechanisms are hooks, not forks.

Amended per Amendment A (docs/superpowers/plans/2026-08-16-gap-closing-batch.md)
after the 2026-08-17 adversarial audit:

  A1  every screen runs over a five-tuple SEED BATTERY. The original plan
      pinned all three RNG draws (positional bases, semantic codebook, cap
      subset) and its 0.005 decision threshold sat INSIDE the measured seed
      noise -- h4's verdict flipped across codebook seeds. Deltas are paired
      within a tuple and averaged across tuples.
  A2  _bundle is a RAW SUM, mirroring build_trace exactly. The first draft
      divided by the class weight sum (a mean); label parity only holds with
      the sum, and the self-test enforces allclose against build_trace itself.
  A3  the exact-label-parity hard stop applies ONLY at SEEDS[0], the draw the
      stored vsa_labels.npz was produced at (verified 100.0% reproducible on
      all 8 scenes, 2026-08-16). Other tuples recompute their own baseline.
"""
from __future__ import annotations

import importlib.util
import json
import os
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
            raise RuntimeError(f"torch.{name} used; batch1 is pure NumPy")

    sys.modules["torch"] = _Absent("torch")

from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402
from vsa_cognitive_mapping.object_grounding import (  # noqa: E402
    class_phasors, build_trace, cap_per_class)

HD, GRID, CAP, LX, LY = 4096, 96, 400, 0.45, 0.27
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2",
          "office3", "office4"]
# (base_seed, codebook_seed, cap_seed). Tuple 0 is the REFERENCE draw --
# the one every stored baseline artifact was produced at.
SEEDS = [(0, 7, 11), (1, 8, 12), (2, 9, 13), (3, 10, 14), (4, 15, 16)]
CACHE_DIR = "outputs/batch1/cache"
# RE-BASELINED 2026-08-18 after class_phasors moved to name-hashed keys
# (see wiki 2026-08-17-gap-closing-batch-results.md). Regenerate with
# collab_tasks/batch1/rebaseline.py; headline moved 0.3237 -> 0.3197,
# inside the 0.017 noise band. Reference-tuple
# sanity values -- the real gate at the reference tuple is LABEL identity
BASELINE_MACC = {"room0": 0.2586,
                 "room1": 0.2347,
                 "room2": 0.3590,
                 "office0": 0.3092,
                 "office1": 0.2490,
                 "office2": 0.3739,
                 "office3": 0.2707,
                 "office4": 0.5029}


def _load_score_mod():
    spec = importlib.util.spec_from_file_location(
        "score05", "student_gpu_package/05_score.py")
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [spec.origin]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv          # do not clobber the harness's own argparse
    return mod


_SC = _load_score_mod()
CG_EXCLUDE_6 = _SC.CG_EXCLUDE_6


def score(gt, pred):
    d = _SC.macc_full(gt, pred, exclude=CG_EXCLUDE_6)
    return {k: float(d[k]) for k in ("macc", "fmiou", "mprec", "mf1")}


class _Enc(ClassroomEncoders):
    def __init__(self, hd, seed, lx, ly):
        super().__init__(hd, seed, 1.0, 20.0)
        self.lx, self.ly = lx, ly

    def ctx_pos(self, x, y):
        return (self.Bx ** float(x / self.lx)) * (self.By ** float(y / self.ly))


def load_scene(scene):
    d = f"student_gpu_package/handoff/{scene}_cgfront"
    E = np.load(f"{d}/eval_points.npz", allow_pickle=True)
    xyz, gt = E["xyz"], E["gt_class"].astype(str)
    v = xyz.var(0)
    a, b = sorted(np.argsort(v)[-2:])
    obs = json.load(open(
        f"outputs/replica_{scene}_cgfront/object_points.json"))["points"]
    return dict(scene=scene, xyz=xyz, gt=gt, a=a, b=b, obs=obs)


def _bundle(obs, names, sem, Bx, By, weight_fn, scales):
    """Mirror of build_trace extended with weights and per-class scales.

    RAW SUM (Amendment A2): build_trace does not normalise per class, and
    exact label parity against the stored baseline only holds with the sum.
    The self-test asserts allclose against build_trace itself; if they ever
    disagree, change THIS function, never the test.
    """
    tr = np.zeros(HD, np.complex128)
    for c in names:
        rows = [o for o in obs if o["cls"] == c]
        if not rows:
            continue
        lx, ly = (scales or {}).get(c, (LX, LY))
        P = np.array([[o["x"], o["y"]] for o in rows], float)
        ph = (Bx[None, :] ** (P[:, 0, None] / lx)) \
            * (By[None, :] ** (P[:, 1, None] / ly))
        if weight_fn is not None:
            w = np.array([weight_fn(o) for o in rows], float)
            tr += sem[c] * (w[:, None] * ph).sum(0)
        else:
            tr += sem[c] * ph.sum(0)
    return tr


def class_fields(data, grid=GRID, cap=CAP, lx=LX, ly=LY,
                 weight_fn=None, scales=None, keep_fn=None,
                 seeds=SEEDS[0]):
    """F (n_classes, grid*grid), sorted names, per-eval-point cell index.

    seeds = (base_seed, codebook_seed, cap_seed), Amendment A1. The default
    is the reference draw, at which the output is label-identical to the
    stored baseline (verified 100.0% on all 8 scenes).
    """
    base_seed, cb_seed, cap_seed = seeds
    xyz, a, b = data["xyz"], data["a"], data["b"]
    obs = data["obs"]
    if keep_fn is not None:
        obs = [o for o in obs if keep_fn(o)]
    obs = cap_per_class(list(obs), cap, seed=cap_seed)
    names = sorted({o["cls"] for o in obs})
    sem = class_phasors(names, HD, seed=cb_seed)
    enc = _Enc(HD, base_seed, lx, ly)
    Bx, By = enc.Bx.values, enc.By.values
    tr = _bundle(obs, names, sem, Bx, By, weight_fn, scales)
    tr /= max(np.abs(tr).max(), 1e-12)
    xs, ys = xyz[:, a], xyz[:, b]
    gx = np.linspace(xs.min(), xs.max(), grid)
    gy = np.linspace(ys.min(), ys.max(), grid)
    F = np.empty((len(names), grid * grid))
    for n, c in enumerate(names):
        clx, cly = (scales or {}).get(c, (lx, ly))
        PX = np.conj(Bx[None, :] ** (gx[:, None] / clx))
        PY = np.conj(By[None, :] ** (gy[:, None] / cly))
        v = tr / sem[c]
        F[n] = ((PX * v[None, :]) @ PY.T).T.reshape(-1).real
    ix = np.clip(np.searchsorted(gx, xs), 0, grid - 1)
    iy = np.clip(np.searchsorted(gy, ys), 0, grid - 1)
    return F, names, iy * grid + ix


def default_fields(data, ti=0):
    """Baseline fields for seed tuple ti, disk-cached (Amendment A7). The
    cache is keyed by (scene, tuple index) and holds ONLY default-parameter
    builds -- variant builds are never cached, so no variant can leak into
    another's baseline."""
    s = data["scene"]
    if s.startswith("_"):        # synthetic scenes are never cached
        return class_fields(data, seeds=SEEDS[ti])
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = f"{CACHE_DIR}/{s}_t{ti}.npz"
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["F"], [str(x) for x in z["names"]], z["cell"]
    F, names, cell = class_fields(data, seeds=SEEDS[ti])
    np.savez_compressed(p, F=F.astype(np.float32),
                        names=np.array(names), cell=cell)
    return F, names, cell


def predict(F, names, cell):
    return np.array([names[w] for w in np.asarray(F).argmax(0)[cell]])


def make_blob_data():
    rng = np.random.RandomState(0)
    centres = {"chair": (1.0, 1.0), "table": (4.0, 1.0), "lamp": (1.0, 4.0)}
    xyz, gt, obs = [], [], []
    for c, (cx, cy) in centres.items():
        p = rng.normal([cx, cy], 0.25, size=(400, 2))
        xyz.append(np.c_[p, np.zeros(len(p))])
        gt += [c] * len(p)
        for k in range(120):
            q = rng.normal([cx, cy], 0.25, size=2)
            obs.append(dict(frame=k, cls=c, conf=1.0, det=len(obs),
                            x=float(q[0]), y=float(q[1])))
    return dict(scene="_blobs", xyz=np.concatenate(xyz), gt=np.array(gt),
                a=0, b=1, obs=obs)


def run_screen(mechanism, prediction, variants, predictions=None,
               out_path=None):
    """Run a mechanism screen over all scenes and the full seed battery.

    variants: {label: fn(data, ti) -> pred array}. Each fn must build with
    seeds=SEEDS[ti] (or read default_fields(data, ti)) so its delta pairs
    with that tuple's own baseline.
    predictions: Amendment A6 machine-checkable list, scored by the guard.
    """
    from collab_tasks.batch1.report import report
    print(f"== {mechanism} ==")
    print(f"PRE-REGISTERED: {prediction}")
    per = {"baseline": {}}
    for ti in range(len(SEEDS)):
        for s in SCENES:
            data = load_scene(s)
            F, nm, cell = default_fields(data, ti)
            pred = predict(F, nm, cell)
            if ti == 0:
                # Amendment A3: exact label identity, reference tuple only.
                stored = np.load(
                    f"student_gpu_package/handoff/{s}_cgfront/vsa_labels.npz",
                    allow_pickle=True)["pred_class"].astype(str)
                if not (pred == stored).all():
                    n = int((pred != stored).sum())
                    raise SystemExit(
                        f"HARD STOP: baseline label parity fail on {s}: "
                        f"{n}/{len(pred)} labels differ from vsa_labels.npz. "
                        "Do NOT loosen this gate - find the drift.")
            base = score(data["gt"], pred)
            if ti == 0 and abs(base["macc"] - BASELINE_MACC[s]) > 0.005:
                raise SystemExit(
                    f"HARD STOP: baseline score parity fail on {s}: "
                    f"{base['macc']:.3f} vs {BASELINE_MACC[s]:.3f}")
            per["baseline"].setdefault(s, {})[f"t{ti}"] = base
            for lab, fn in variants.items():
                per.setdefault(lab, {}).setdefault(s, {})[f"t{ti}"] = \
                    score(data["gt"], fn(data, ti))
            print(f"  t{ti} {s} done", flush=True)
    return report(mechanism, prediction, per,
                  out_path or f"outputs/batch1/{mechanism}.json",
                  predictions=predictions)


def _self_test():
    print("1/4 trace parity vs build_trace (raw sum, Amendment A2)...")
    data = load_scene("room0")
    obs = cap_per_class(list(data["obs"]), CAP, seed=SEEDS[0][2])
    names = sorted({o["cls"] for o in obs})
    sem = class_phasors(names, HD, seed=SEEDS[0][1])
    enc = _Enc(HD, SEEDS[0][0], LX, LY)
    want = build_trace(obs, enc, sem, HD)
    got = _bundle(obs, names, sem, enc.Bx.values, enc.By.values, None, None)
    assert np.allclose(got, want, atol=1e-9), "manual bundle != build_trace"
    print("   OK")
    print("2/4 baseline LABEL parity on room0 at the reference tuple...")
    F, nm, cell = class_fields(data)
    pred = predict(F, nm, cell)
    stored = np.load(
        "student_gpu_package/handoff/room0_cgfront/vsa_labels.npz",
        allow_pickle=True)["pred_class"].astype(str)
    agree = float((pred == stored).mean())
    assert agree == 1.0, f"label agreement {agree:.4f}, must be exactly 1.0"
    m = score(data["gt"], pred)
    assert abs(m["macc"] - BASELINE_MACC["room0"]) <= 0.005, m["macc"]
    print(f"   OK (100% labels, mAcc {m['macc']:.3f})")
    print("3/4 seed battery actually changes the draw (guards A1 against a "
          "silent no-op)...")
    F2, nm2, cell2 = class_fields(data, seeds=SEEDS[1])
    pred2 = predict(F2, nm2, cell2)
    moved = float((pred2 != pred).mean())
    assert moved > 0.0, "tuple 1 reproduced tuple 0 exactly; seeds not threaded"
    print(f"   OK ({moved:.1%} of labels move under tuple 1)")
    print("4/4 synthetic blobs decode near 1.0...")
    blob = make_blob_data()
    Fb, nb, cb = class_fields(blob)
    mb = score(blob["gt"], predict(Fb, nb, cb))
    assert mb["macc"] > 0.9, mb["macc"]
    print(f"   OK ({mb['macc']:.3f})")
    print("SELF-TEST PASS")


if __name__ == "__main__":
    _self_test()
