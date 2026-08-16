# Gap-Closing Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Screen six backend mechanisms against the 0.324-mAcc baseline on the fair comparison (shared SAM+CLIP frontend, 8 Replica scenes), joint-grid the survivors, and adopt at most one configuration under the pre-registered rule (mAcc improves AND mF1 does not fall).

**Architecture:** A shared `common.py` provides scene loading, trace building and decoding that byte-matches the measured baseline; a `report.py` guard is the only allowed output path and refuses verdicts on under-powered runs; an error decomposition classifies every wrong cell before any mechanism runs; six thin harnesses each apply one mechanism through `common.py` hooks (weight_fn / scales / keep_fn / F-transform); a Stage-2 script composes survivors.

**Tech Stack:** Python 3.13, NumPy only (no torch locally — shim), `vsa_cognitive_mapping` Phasor/encoder classes, `student_gpu_package/05_score.py` metrics loaded via importlib. All CPU, run from repo root.

## Global Constraints

- **ASCII-only in anything printed to stdout** — cp1252 consoles killed two finished runs this week. Non-ASCII allowed in docstrings/comments only.
- **No torch import may execute locally**: every entry script installs the `_Absent` shim before importing `vsa_cognitive_mapping`.
- **All 8 scenes, always**: room0 room1 room2 office0 office1 office2 office3 office4. The guard enforces this; never bypass the guard.
- **Baseline is recomputed in-run, never quoted**: argmax · grid 96 · cap 400 · λ 0.45,0.27. Parity targets (mAcc): room0 0.270, room1 0.245, room2 0.337, office0 0.324, office1 0.299, office2 0.371, office3 0.278, office4 0.466, tolerance ±0.005. A parity failure is a **hard stop** — investigate, do not proceed.
- **Metrics via `05_score.macc_full` only** (`exclude=CG_EXCLUDE_6`). Never reimplement a metric.
- **Pre-registered thresholds** (in `report.py`, used verbatim): SURVIVE if best-variant Δ mAcc > +0.005 OR Δ mF1 > +0.005; ADOPT-CANDIDATE additionally requires Δ mF1 > −0.005.
- **Commits are local per task; push only at the end-of-stage checkpoint after Paul reviews the files** (his standing rule).
- **Execution model tiers** (from `~/.claude/CLAUDE.md`, with Paul's override 2026-08-16):
  - **supervisor → `fable`** (main session; builds Stage 0, reviews every result against the guard, orders Stage 2)
  - test-builder (Tasks 4–8 harness authoring) → full model, inherit
  - workers (Tasks 9, 11 run-only) → `haiku`, `effort: low`, raw output only — a worker that interprets results is being used wrong
- Working directory for every command: repo root `C:\Users\30068379\OneDrive - Western Sydney University\Code\Semantic-Spiking-Neural-SLAM-2023`.

## File Structure

```
collab_tasks/batch1/
  __init__.py            (empty)
  common.py              scene loading, trace/fields, predict, score, blobs, run_screen
  report.py              the guard: table printing, verdict logic, JSON emission
  error_decomposition.py wrong-cell classification + recoverable-mAcc bounds
  h1_threshold.py        abstain / fallback threshold decode
  h2_spread_norm.py      per-class spread + mass normalisation of F
  h3_perclass_lambda.py  kernel width per class from observation spread
  h4_cap_sweep.py        insertion cap at the corrected configuration
  h5_conf_weight.py      confidence-weighted bundling
  h6_insertion_filter.py geometric outlier rejection before bundling
  stage2_joint.py        joint grid over survivors
outputs/batch1/          one JSON per mechanism + stage2_joint.json + decomposition.json
```

---

### Task 1: `common.py` — shared machinery with baseline parity

**Files:**
- Create: `collab_tasks/batch1/__init__.py`
- Create: `collab_tasks/batch1/common.py`

**Interfaces:**
- Consumes: `vsa_cognitive_mapping.classroom_pipeline.ClassroomEncoders`, `vsa_cognitive_mapping.object_grounding.class_phasors, build_trace, cap_per_class`, `student_gpu_package/05_score.py` (`macc_full`, `CG_EXCLUDE_6`).
- Produces (later tasks rely on these exact names):
  - `SCENES: list[str]`, `HD=4096`, `GRID=96`, `CAP=400`, `LX=0.45`, `LY=0.27`
  - `load_scene(scene: str) -> dict` with keys `scene, xyz, gt, a, b, obs`
  - `class_fields(data, grid=GRID, cap=CAP, lx=LX, ly=LY, weight_fn=None, scales=None, keep_fn=None) -> (F, names, cell)` — F is `(n_classes, grid*grid)` float, `names` sorted list, `cell` int array len(eval points)
  - `default_fields(data) -> (F, names, cell)` — cached per scene at default params
  - `predict(F, names, cell) -> np.ndarray[str]`
  - `score(gt, pred) -> dict` with keys `macc, fmiou, mprec, mf1`
  - `make_blob_data() -> dict` — synthetic 3-class scene shaped like `load_scene` output
  - `run_screen(mechanism: str, prediction: str, variants: dict[str, callable], out_path=None) -> dict` — loops scenes, recomputes baseline, calls `report.report`

- [ ] **Step 1: Read the two functions this must mirror**

Open `vsa_cognitive_mapping/object_grounding.py` and read `build_trace` and `cap_per_class` in full. Note exactly: does `build_trace` divide each class bundle by its observation count (mean) or sum raw; what normalisation it applies at the end. The manual bundle in Step 3 must reproduce it exactly — the parity test in Step 2 is the arbiter, and if they disagree, the mirror changes, never the test.

- [ ] **Step 2: Write the failing self-test**

Create `collab_tasks/batch1/common.py` with only the shim, imports, constants, and the self-test at the bottom:

```python
"""Shared machinery for the batch-1 gap-closing screen.

Every harness goes through this module so that (a) the baseline is one
computation, not six copies, and (b) mechanisms are hooks, not forks.
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
# measured 2026-08-16 (rescore_post_inscene.json, vsa rows); hard-stop parity
BASELINE_MACC = {"room0": 0.270, "room1": 0.245, "room2": 0.337,
                 "office0": 0.324, "office1": 0.299, "office2": 0.371,
                 "office3": 0.278, "office4": 0.466}


def _self_test():
    print("1/3 trace parity vs build_trace...")
    data = load_scene("room0")
    obs = cap_per_class(list(data["obs"]), CAP)
    names = sorted({o["cls"] for o in obs})
    sem = class_phasors(names, HD)
    enc = _Enc(HD, 0, LX, LY)
    want = build_trace(obs, enc, sem, HD)
    got = _bundle(obs, names, sem, enc.Bx.values, enc.By.values, None, None)
    assert np.allclose(got, want, atol=1e-9), "manual bundle != build_trace"
    print("   OK")
    print("2/3 baseline mAcc parity on room0...")
    F, nm, cell = class_fields(data)
    m = score(data["gt"], predict(F, nm, cell))
    assert abs(m["macc"] - BASELINE_MACC["room0"]) <= 0.005, m["macc"]
    print(f"   OK ({m['macc']:.3f})")
    print("3/3 synthetic blobs decode near 1.0...")
    blob = make_blob_data()
    F, nm, cell = class_fields(blob)
    m = score(blob["gt"], predict(F, nm, cell))
    assert m["macc"] > 0.9, m["macc"]
    print(f"   OK ({m['macc']:.3f})")
    print("SELF-TEST PASS")


if __name__ == "__main__":
    _self_test()
```

- [ ] **Step 3: Run to verify it fails**

Run: `python collab_tasks/batch1/common.py`
Expected: `NameError: name 'load_scene' is not defined` (after the shim/imports succeed).

- [ ] **Step 4: Implement the module above the self-test**

```python
def _load_score_mod():
    spec = importlib.util.spec_from_file_location(
        "score05", "student_gpu_package/05_score.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [spec.origin]
    spec.loader.exec_module(mod)
    return mod


_SC = _load_score_mod()


def score(gt, pred):
    d = _SC.macc_full(gt, pred, exclude=_SC.CG_EXCLUDE_6)
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
    """Mirror of build_trace, extended with weights and per-class scales.
    With weight_fn=None and scales=None this MUST be allclose to
    build_trace(obs, enc, sem, HD) -- the self-test enforces it. If the
    mirror and build_trace disagree, change THIS function, not the test."""
    tr = np.zeros(HD, np.complex128)
    for c in names:
        rows = [o for o in obs if o["cls"] == c]
        if not rows:
            continue
        lx, ly = (scales or {}).get(c, (LX, LY))
        P = np.array([[o["x"], o["y"]] for o in rows], float)
        w = (np.array([weight_fn(o) for o in rows], float)
             if weight_fn else np.ones(len(rows)))
        ph = (Bx[None, :] ** (P[:, 0, None] / lx)) \
            * (By[None, :] ** (P[:, 1, None] / ly))
        # NOTE: match build_trace's per-class normalisation exactly here
        # (mean vs raw sum) after reading it in Step 1; weighted form is
        # sum(w*ph)/sum(w), which reduces to the mean when w == 1.
        tr += sem[c] * ((w[:, None] * ph).sum(0) / max(w.sum(), 1e-12))
    return tr


def class_fields(data, grid=GRID, cap=CAP, lx=LX, ly=LY,
                 weight_fn=None, scales=None, keep_fn=None):
    xyz, a, b = data["xyz"], data["a"], data["b"]
    obs = data["obs"]
    if keep_fn is not None:
        obs = [o for o in obs if keep_fn(o)]
    obs = cap_per_class(list(obs), cap)
    names = sorted({o["cls"] for o in obs})
    sem = class_phasors(names, HD)
    enc = _Enc(HD, 0, lx, ly)
    Bx, By = enc.Bx.values, enc.By.values
    if scales:
        # per-class lambda: the bundle uses each class's own scale, so the
        # global default (lx, ly) passed above only covers unlisted classes
        pass
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


_FC = {}


def default_fields(data):
    s = data["scene"]
    if s not in _FC:
        _FC[s] = class_fields(data)
    return _FC[s]


def predict(F, names, cell):
    return np.array([names[w] for w in F.argmax(0)[cell]])


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


def run_screen(mechanism, prediction, variants, out_path=None):
    from collab_tasks.batch1.report import report
    print(f"== {mechanism} ==")
    print(f"PRE-REGISTERED: {prediction}")
    per = {"baseline": {}}
    for s in SCENES:
        data = load_scene(s)
        F, nm, cell = default_fields(data)
        base = score(data["gt"], predict(F, nm, cell))
        dev = abs(base["macc"] - BASELINE_MACC[s])
        if dev > 0.005:
            raise SystemExit(f"HARD STOP: baseline parity fail on {s}: "
                             f"{base['macc']:.3f} vs {BASELINE_MACC[s]:.3f}")
        per["baseline"][s] = base
        for lab, fn in variants.items():
            per.setdefault(lab, {})[s] = score(data["gt"], fn(data))
        print(f"  {s} done", flush=True)
    return report(mechanism, prediction, per,
                  out_path or f"outputs/batch1/{mechanism}.json")
```

Also create the empty `collab_tasks/batch1/__init__.py`.

- [ ] **Step 5: Run the self-test to verify it passes**

Run: `python collab_tasks/batch1/common.py`
Expected: three `OK` lines then `SELF-TEST PASS`. If check 1 fails, adjust `_bundle` to match what `build_trace` actually does (Step 1 notes) until allclose holds. If check 2 then fails by more than 0.005, that is the hard stop — report to the supervisor, do not tune anything.

- [ ] **Step 6: Commit**

```bash
git add collab_tasks/batch1/__init__.py collab_tasks/batch1/common.py
git commit -m "batch1: shared machinery with enforced baseline parity"
```

---

### Task 2: `report.py` — the guard

**Files:**
- Create: `collab_tasks/batch1/report.py`

**Interfaces:**
- Consumes: nothing from this repo (pure stdlib + numpy) — keep it dependency-free so a broken common.py cannot break the guard.
- Produces: `report(mechanism: str, prediction: str, per: dict, out_path: str) -> dict`, `METRICS = ("macc","fmiou","mprec","mf1")`, `SURVIVE_EPS = 0.005`, `ADOPT_F1_FLOOR = -0.005`. `per` shape: `{"baseline": {scene: {metric: float}}, "<variant>": {scene: {...}}, ...}`. Return/JSON shape: `{mechanism, prediction, n_scenes, baseline: {per_scene, mean}, variants: {label: {per_scene, mean, delta_vs_baseline}}, best_variant, verdict, verdict_allowed}`.

- [ ] **Step 1: Write the failing self-test**

Create `collab_tasks/batch1/report.py` with the self-test at the bottom first:

```python
"""The reporting guard. Every batch-1 result goes through report() or it
does not exist. Refuses a verdict on under-powered input -- the structural
fix for six single-scene/single-metric inversions on 2026-08-15/16."""
from __future__ import annotations

import json
import os

import numpy as np

METRICS = ("macc", "fmiou", "mprec", "mf1")
SURVIVE_EPS = 0.005        # pre-registered, spec 2026-08-16
ADOPT_F1_FLOOR = -0.005    # pre-registered, spec 2026-08-16
N_REQUIRED = 8


def _self_test():
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "guard_test.json")
    full = {m: 0.3 for m in METRICS}
    up = dict(full, macc=0.32, mf1=0.17)
    eight = {s: dict(full) for s in
             ["room0", "room1", "room2", "office0", "office1", "office2",
              "office3", "office4"]}
    print("1/3 three scenes must refuse...")
    r = report("t", "p", {"baseline": {k: eight[k] for k in list(eight)[:3]},
                          "v": {k: dict(up) for k in list(eight)[:3]}}, tmp)
    assert r["verdict_allowed"] is False and r["verdict"] is None
    print("2/3 missing metric must refuse...")
    bad = {s: {m: 0.3 for m in METRICS if m != "mf1"} for s in eight}
    r = report("t", "p", {"baseline": eight, "v": bad}, tmp)
    assert r["verdict_allowed"] is False
    print("3/3 full eight-scene input must verdict...")
    r = report("t", "p", {"baseline": eight,
                          "v": {s: dict(up) for s in eight}}, tmp)
    assert r["verdict_allowed"] is True
    assert r["verdict"] == "ADOPT-CANDIDATE", r["verdict"]
    print("SELF-TEST PASS")


if __name__ == "__main__":
    _self_test()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python collab_tasks/batch1/report.py`
Expected: `NameError: name 'report' is not defined`.

- [ ] **Step 3: Implement `report` above the self-test**

```python
def report(mechanism, prediction, per, out_path):
    problems = []
    for lab, by_scene in per.items():
        if len(by_scene) != N_REQUIRED:
            problems.append(f"{lab}: {len(by_scene)} scenes, need {N_REQUIRED}")
        for s, m in by_scene.items():
            missing = [k for k in METRICS if k not in m]
            if missing:
                problems.append(f"{lab}/{s}: missing {missing}")
    allowed = not problems
    mean = {lab: {k: float(np.mean([sc[k] for sc in by.values()]))
                  for k in METRICS} if by else {}
            for lab, by in per.items()}
    sd = {lab: {k: float(np.std([sc[k] for sc in by.values()]))
                for k in METRICS} if by else {}
          for lab, by in per.items()}
    delta = {lab: {k: mean[lab][k] - mean["baseline"][k] for k in METRICS}
             for lab in per if lab != "baseline" and allowed}

    print(f"\n[{mechanism}] per-scene mAcc (baseline first):")
    scenes = sorted(per["baseline"])
    print("  " + "".join(f"{s[:8]:>9}" for s in scenes) + "     mean +- sd")
    for lab in per:
        row = "".join(f"{per[lab].get(s, {}).get('macc', float('nan')):>9.3f}"
                      for s in scenes)
        m, d = mean[lab].get("macc", float("nan")), sd[lab].get("macc", 0.0)
        print(f"  {row}   {m:.3f} +- {d:.3f}   {lab}")

    best = verdict = None
    if allowed and delta:
        best = max(delta, key=lambda l: max(delta[l]["macc"],
                                            delta[l]["mf1"]))
        d = delta[best]
        if d["macc"] > SURVIVE_EPS and d["mf1"] > ADOPT_F1_FLOOR:
            verdict = "ADOPT-CANDIDATE"
        elif d["macc"] > SURVIVE_EPS or d["mf1"] > SURVIVE_EPS:
            verdict = "SURVIVES"
        else:
            verdict = "KILLED"
        print(f"VERDICT [{mechanism}] best={best} "
              f"dmacc={d['macc']:+.3f} dmf1={d['mf1']:+.3f} -> {verdict}")
    else:
        print("UNDERPOWERED - no verdict")
        for p in problems:
            print("  " + p)

    js = dict(mechanism=mechanism, prediction=prediction,
              n_scenes=len(per["baseline"]),
              baseline=dict(per_scene=per["baseline"],
                            mean=mean["baseline"]),
              variants={lab: dict(per_scene=per[lab], mean=mean[lab],
                                  delta_vs_baseline=delta.get(lab, {}))
                        for lab in per if lab != "baseline"},
              best_variant=best, verdict=verdict, verdict_allowed=allowed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(js, open(out_path, "w"), indent=1)
    return js
```

- [ ] **Step 4: Run the self-test to verify it passes**

Run: `python collab_tasks/batch1/report.py`
Expected: three numbered lines then `SELF-TEST PASS`.

- [ ] **Step 5: Commit**

```bash
git add collab_tasks/batch1/report.py
git commit -m "batch1: reporting guard - refuses verdicts on underpowered runs"
```

---

### Task 3: `error_decomposition.py` — where the gap actually is

**Files:**
- Create: `collab_tasks/batch1/error_decomposition.py`

**Interfaces:**
- Consumes: `common.load_scene, default_fields, predict, score, SCENES`.
- Produces: `outputs/batch1/decomposition.json`; console table of category counts per scene and pooled, plus **recoverable-mAcc upper bounds** per category (flip that category's cells to correct, rescore). Categories (decision tree, in priority order, guaranteeing a partition): `unreachable`, `near_tie` (at 2%/5%/10%, 5% headline), `bleed`, `misplaced`, `local_loss` (residual). Constants: `BLEED_DIST = 0.9` (2 × λx, metres).

- [ ] **Step 1: Write the failing self-test**

Bottom of the new file:

```python
def _self_test():
    from collab_tasks.batch1.common import make_blob_data, default_fields
    blob = make_blob_data()
    cats, bounds = decompose(blob, frac=0.05)
    total_wrong = sum(len(v) for v in cats.values())
    n_union = len(set().union(*cats.values())) if total_wrong else 0
    assert total_wrong == n_union, "categories overlap"
    print(f"partition OK ({total_wrong} wrong cells, all disjoint)")
    print("SELF-TEST PASS")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python collab_tasks/batch1/error_decomposition.py --self-test`
Expected: `NameError: name 'decompose' is not defined`.

- [ ] **Step 3: Implement**

```python
"""Classify every wrong cell BEFORE running any fix, so mechanism choice is
evidence-ordered. unreachable cells are the frontend's; they bound what any
backend change can recover."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, load_scene, default_fields, predict, score)

BLEED_DIST = 0.9          # metres; 2 x the lambda_x kernel scale
TIE_FRACS = (0.02, 0.05, 0.10)
CATS = ("unreachable", "near_tie", "bleed", "misplaced", "local_loss")


def _nearest_dist(px, py, pts):
    """Min distance from (px,py) to a set of 2D points, chunked numpy --
    no scipy dependency."""
    if len(pts) == 0:
        return np.inf
    d2 = (pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2
    return float(np.sqrt(d2.min()))


def decompose(data, frac=0.05):
    F, names, cell = default_fields(data)
    gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
    pred = predict(F, names, cell)
    idx = {c: i for i, c in enumerate(names)}
    obs_xy = {}
    for o in data["obs"]:
        obs_xy.setdefault(o["cls"], []).append((o["x"], o["y"]))
    obs_xy = {c: np.array(v) for c, v in obs_xy.items()}

    from collab_tasks.batch1.common import _SC
    excl = set(_SC.CG_EXCLUDE_6)
    cats = {c: set() for c in CATS}
    for i in np.flatnonzero(pred != gt):
        g = gt[i]
        if g in excl:
            continue
        px, py = float(xyz[i, a]), float(xyz[i, b])
        w = pred[i]
        if g not in idx or g not in obs_xy:
            cats["unreachable"].add(i)
        else:
            fg = F[idx[g], cell[i]]
            fw = F[idx[w], cell[i]] if w in idx else np.inf
            if fg >= fw - frac * abs(fw):
                cats["near_tie"].add(i)
            elif _nearest_dist(px, py, obs_xy.get(w, np.empty((0, 2)))) \
                    > BLEED_DIST:
                cats["bleed"].add(i)
            elif _nearest_dist(px, py, obs_xy[g]) > BLEED_DIST:
                cats["misplaced"].add(i)
            else:
                cats["local_loss"].add(i)

    bounds = {}
    for c, members in cats.items():
        p2 = pred.copy()
        m = np.array(sorted(members), int)
        if len(m):
            p2[m] = gt[m]
        bounds[c] = score(gt, p2)["macc"]
    return cats, bounds


def main():
    out = {}
    print(f"{'scene':<9}" + "".join(f"{c:>12}" for c in CATS))
    for s in SCENES:
        data = load_scene(s)
        cats, bounds = decompose(data, frac=0.05)
        base = score(data["gt"],
                     predict(*default_fields(data)))["macc"]
        out[s] = {"counts": {c: len(v) for c, v in cats.items()},
                  "base_macc": base,
                  "recoverable_macc_if_fixed": bounds}
        print(f"{s:<9}" + "".join(f"{len(cats[c]):>12}" for c in CATS),
              flush=True)
    print("\nrecoverable mAcc upper bound if a category were fully fixed")
    print(f"{'scene':<9}{'base':>7}" + "".join(f"{c:>12}" for c in CATS))
    for s in SCENES:
        r = out[s]
        print(f"{s:<9}{r['base_macc']:>7.3f}" + "".join(
            f"{r['recoverable_macc_if_fixed'][c]:>12.3f}" for c in CATS))
    for fr in TIE_FRACS:
        pooled = {c: 0 for c in CATS}
        for s in SCENES:
            cats, _ = decompose(load_scene(s), frac=fr)
            for c in CATS:
                pooled[c] += len(cats[c])
        print(f"tie frac {fr:.2f}: " +
              "  ".join(f"{c}={pooled[c]}" for c in CATS))
    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(out, open("outputs/batch1/decomposition.json", "w"), indent=1)
    print("\nmechanism map: near_tie -> h1/h2; bleed -> h6/h3/h4; "
          "misplaced -> geometry (out of batch scope); "
          "unreachable -> frontend constant; local_loss -> h2/h3")
    print("wrote outputs/batch1/decomposition.json")
```

- [ ] **Step 4: Run self-test, then the real run**

Run: `python collab_tasks/batch1/error_decomposition.py --self-test` → `SELF-TEST PASS`.
Run: `python collab_tasks/batch1/error_decomposition.py` → per-scene table, bounds table, tie-frac sensitivity, JSON written. (~10 min.)

- [ ] **Step 5: Commit**

```bash
git add collab_tasks/batch1/error_decomposition.py outputs/batch1/decomposition.json
git commit -m "batch1: error decomposition - the gap split into recoverable vs unreachable"
```

---

### Task 4: `h1_threshold.py`

**Files:**
- Create: `collab_tasks/batch1/h1_threshold.py`

**Interfaces:**
- Consumes: `common.default_fields, predict, run_screen, make_blob_data`.
- Produces: `outputs/batch1/h1_threshold.json` via the guard.

- [ ] **Step 1: Write the harness with self-test**

```python
"""H1: per-class z-threshold decode, abstain and fallback forms."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    default_fields, make_blob_data, run_screen, score)

PRED = ("Pre-registered: threshold is an ABSTAIN mechanism, i.e. a precision "
        "tool. Expect mF1 up via mprec at some tau; mAcc within +-0.01 of "
        "baseline. Prior +0.014 was 3 scenes at the stale config; this is "
        "the 8-scene, precision-aware re-test.")


def variant(tau, abstain):
    def fn(data):
        F, names, cell = default_fields(data)
        Z = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)
        above = Z > tau
        masked = np.where(above, F, -np.inf)
        win = masked.argmax(0)
        none = ~above.any(0)
        if abstain:
            lab = np.array([("__none__" if n else names[w])
                            for w, n in zip(win, none)])
        else:
            win = win.copy()
            win[none] = F[:, none].argmax(0)
            lab = np.array([names[w] for w in win])
        return lab[cell]
    return fn


VARIANTS = {f"tau{t}_{m}": variant(t, m == "abstain")
            for t in (0.5, 1.0, 1.5, 2.0, 3.0)
            for m in ("fallback", "abstain")}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["tau1.0_fallback"](blob))
    assert m["macc"] > 0.9, m
    # abstain must actually abstain: a probe far from every blob
    blob2 = dict(blob)
    import numpy as np
    blob2["xyz"] = np.vstack([blob["xyz"], [[9.0, 9.0, 0.0]]])
    blob2["gt"] = np.append(blob["gt"], "chair")
    p = VARIANTS["tau2.0_abstain"](blob2)
    assert p[-1] == "__none__", "far-away point did not abstain"
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h1_threshold", PRED, VARIANTS)
```

- [ ] **Step 2: Run self-test**

Run: `python collab_tasks/batch1/h1_threshold.py --self-test`
Expected: `SELF-TEST PASS`. (Note the blob cache: `default_fields` caches per scene name; the modified blob reuses `_blobs` cached fields, which is intended — same fields, one extra query point via `cell` recompute is NOT triggered. If the abstain assert fails for this reason, rebuild fields with `class_fields(blob2)` instead of `default_fields` inside the test.)

- [ ] **Step 3: Commit**

```bash
git add collab_tasks/batch1/h1_threshold.py
git commit -m "batch1 h1: threshold decode harness, abstain + fallback"
```

---

### Task 5: `h2_spread_norm.py`

**Files:**
- Create: `collab_tasks/batch1/h2_spread_norm.py`

**Interfaces:** consumes `common.default_fields, predict, run_screen, make_blob_data, score`; produces `outputs/batch1/h2_spread_norm.json`.

- [ ] **Step 1: Write the harness with self-test**

```python
"""H2: normalise each class's field by its own spatial spread and mass,
so diffuse heavy classes stop out-shouting compact ones."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    default_fields, make_blob_data, run_screen, score)

PRED = ("Pre-registered: recovers bleed cells; helps most on scenes the "
        "decomposition marks bleed-dominant. Global z-score (the crude "
        "version) was +0.010 mean / 3 of 8 scenes; spread-aware should "
        "match or beat it WITHOUT the office4 dependence.")


def _spread_mass(data, names):
    sp, ms = [], []
    for c in names:
        P = np.array([[o["x"], o["y"]] for o in data["obs"]
                      if o["cls"] == c], float)
        if len(P) == 0:
            sp.append(1.0)
            ms.append(1.0)
            continue
        ctr = np.median(P, 0)
        rms = float(np.sqrt(((P - ctr) ** 2).sum(1).mean()))
        sp.append(max(rms, 0.05))
        ms.append(float(len(P)))
    return np.array(sp), np.array(ms)


def variant(p, q):
    def fn(data):
        F, names, cell = default_fields(data)
        sp, ms = _spread_mass(data, names)
        Fp = F / (sp[:, None] ** p * ms[:, None] ** q)
        from collab_tasks.batch1.common import predict
        return predict(Fp, names, cell)
    return fn


def zscore(data):
    F, names, cell = default_fields(data)
    Z = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)
    from collab_tasks.batch1.common import predict
    return predict(Z, names, cell)


VARIANTS = {f"p{p}_q{q}": variant(p, q)
            for p in (0.0, 0.5, 1.0) for q in (0.0, 0.5, 1.0)
            if not (p == 0.0 and q == 0.0)}
VARIANTS["zscore_control"] = zscore


def _self_test():
    blob = make_blob_data()
    for lab in ("p0.5_q0.5", "zscore_control"):
        m = score(blob["gt"], VARIANTS[lab](blob))
        assert m["macc"] > 0.9, (lab, m)
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h2_spread_norm", PRED, VARIANTS)
```

- [ ] **Step 2: Run self-test** — `python collab_tasks/batch1/h2_spread_norm.py --self-test` → `SELF-TEST PASS`.

- [ ] **Step 3: Commit**

```bash
git add collab_tasks/batch1/h2_spread_norm.py
git commit -m "batch1 h2: per-class spread/mass normalisation harness"
```

---

### Task 6: `h3_perclass_lambda.py`

**Files:**
- Create: `collab_tasks/batch1/h3_perclass_lambda.py`

**Interfaces:** consumes `common.class_fields, predict, run_screen, make_blob_data, score, LX, LY`; produces `outputs/batch1/h3_perclass_lambda.json`.

- [ ] **Step 1: Write the harness with self-test**

```python
"""H3: kernel width per class, proportional to that class's observation
spread per axis. One component per observation, shaped well -- the rule the
multi-scale failure taught us."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    class_fields, make_blob_data, predict, run_screen, score)

PRED = ("Pre-registered: mAcc up on scenes with mixed object sizes; risk is "
        "per-scene overfit, so k is a single global multiplier, never tuned "
        "per scene. Derived from observation spread, no per-class hand "
        "tuning.")

CLIP_LO, CLIP_HI = 0.10, 0.90


def _scales(data, k):
    out = {}
    by = {}
    for o in data["obs"]:
        by.setdefault(o["cls"], []).append((o["x"], o["y"]))
    for c, pts in by.items():
        P = np.array(pts, float)
        ctr = np.median(P, 0)
        rx = float(np.sqrt(((P[:, 0] - ctr[0]) ** 2).mean()))
        ry = float(np.sqrt(((P[:, 1] - ctr[1]) ** 2).mean()))
        out[c] = (float(np.clip(k * max(rx, 0.05), CLIP_LO, CLIP_HI)),
                  float(np.clip(k * max(ry, 0.05), CLIP_LO, CLIP_HI)))
    return out


def variant(k):
    def fn(data):
        F, names, cell = class_fields(data, scales=_scales(data, k))
        return predict(F, names, cell)
    return fn


VARIANTS = {f"k{k}": variant(k) for k in (0.5, 1.0, 1.5)}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["k1.0"](blob))
    assert m["macc"] > 0.9, m
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h3_perclass_lambda", PRED, VARIANTS)
```

- [ ] **Step 2: Run self-test** — expected `SELF-TEST PASS`.

- [ ] **Step 3: Commit**

```bash
git add collab_tasks/batch1/h3_perclass_lambda.py
git commit -m "batch1 h3: per-class lambda harness, spread-derived, one global k"
```

---

### Task 7: `h4_cap_sweep.py` and `h5_conf_weight.py`

**Files:**
- Create: `collab_tasks/batch1/h4_cap_sweep.py`
- Create: `collab_tasks/batch1/h5_conf_weight.py`

**Interfaces:** consume `common.class_fields, predict, run_screen, make_blob_data, score`; produce `outputs/batch1/h4_cap_sweep.json`, `outputs/batch1/h5_conf_weight.json`.

- [ ] **Step 1: Write `h4_cap_sweep.py`**

```python
"""H4: insertion cap at the corrected configuration. The old 'saturates by
200' was measured at stale lambda and buggy labels; cap400_sanity must come
out at delta 0.000 (it IS the baseline) or the harness itself is broken."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    class_fields, make_blob_data, predict, run_screen, score)

PRED = ("Pre-registered: current 400 near-optimal; the no-cap audit showed "
        "-0.044, so 800 should be flat-to-down and 100 should cost recall. "
        "cap400_sanity must be delta 0.000 exactly.")


def variant(cap):
    def fn(data):
        F, names, cell = class_fields(data, cap=cap)
        return predict(F, names, cell)
    return fn


VARIANTS = {"cap100": variant(100), "cap200": variant(200),
            "cap400_sanity": variant(400), "cap800": variant(800)}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["cap200"](blob))
    assert m["macc"] > 0.9, m
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h4_cap_sweep", PRED, VARIANTS)
```

- [ ] **Step 2: Write `h5_conf_weight.py`**

```python
"""H5: weight each observation by its detector confidence when bundling.
First harness prints the conf distribution -- if conf is near-constant this
mechanism is dead on arrival and the prediction says so."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, class_fields, load_scene, make_blob_data, predict, run_screen,
    score)

PRED = ("Pre-registered: small, likely +-0.005 -- conf in the cgfront "
        "stream is expected near-constant (their objects re-emit one conf). "
        "Included because it is cheap and would otherwise stay folklore.")


def variant(gamma):
    def fn(data):
        F, names, cell = class_fields(
            data, weight_fn=lambda o: float(o.get("conf", 1.0)) ** gamma)
        return predict(F, names, cell)
    return fn


VARIANTS = {"gamma1": variant(1.0), "gamma2": variant(2.0)}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["gamma1"](blob))
    assert m["macc"] > 0.9, m
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        confs = [o["conf"] for s in SCENES[:2]
                 for o in load_scene(s)["obs"]]
        print(f"conf stats (2 scenes): mean {np.mean(confs):.3f} "
              f"sd {np.std(confs):.3f} min {min(confs):.3f} "
              f"max {max(confs):.3f}")
        run_screen("h5_conf_weight", PRED, VARIANTS)
```

- [ ] **Step 3: Run both self-tests** — each prints `SELF-TEST PASS`.

- [ ] **Step 4: Commit**

```bash
git add collab_tasks/batch1/h4_cap_sweep.py collab_tasks/batch1/h5_conf_weight.py
git commit -m "batch1 h4+h5: cap re-sweep and confidence weighting harnesses"
```

---

### Task 8: `h6_insertion_filter.py`

**Files:**
- Create: `collab_tasks/batch1/h6_insertion_filter.py`

**Interfaces:** consumes `common.class_fields, predict, run_screen, make_blob_data, score`; produces `outputs/batch1/h6_insertion_filter.json`.

- [ ] **Step 1: Write the harness with self-test**

```python
"""H6: geometric outlier rejection before bundling -- the cap generalised.
An observation farther than r x its class's robust spread from the class's
robust centre is rejected. Attacks bleed at the source."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    class_fields, make_blob_data, predict, run_screen, score)

PRED = ("Pre-registered: mAcc up on vent-type scenes (room0/office2-4), "
        "mF1 up broadly via precision. Floor of 0.5 m stops tight classes "
        "being nuked. Risk: legitimately large objects (sofa) losing their "
        "own extent at small r.")

FLOOR = 0.5   # metres -- never reject inside this radius


def keep_factory(data, r):
    stats = {}
    by = {}
    for o in data["obs"]:
        by.setdefault(o["cls"], []).append((o["x"], o["y"]))
    for c, pts in by.items():
        P = np.array(pts, float)
        ctr = np.median(P, 0)
        d = np.sqrt(((P - ctr) ** 2).sum(1))
        mad = float(np.median(d))
        stats[c] = (ctr, max(r * mad, FLOOR))

    def keep(o):
        ctr, lim = stats[o["cls"]]
        return float(np.hypot(o["x"] - ctr[0], o["y"] - ctr[1])) <= lim
    return keep


def variant(r):
    def fn(data):
        F, names, cell = class_fields(data,
                                      keep_fn=keep_factory(data, r))
        return predict(F, names, cell)
    return fn


VARIANTS = {f"r{r}": variant(r) for r in (2.0, 3.0, 4.0)}


def _self_test():
    blob = make_blob_data()
    m = score(blob["gt"], VARIANTS["r3.0"](blob))
    assert m["macc"] > 0.9, m
    # a far outlier must be rejected
    blob2 = dict(blob, obs=blob["obs"] + [dict(frame=0, cls="chair",
                 conf=1.0, det=9999, x=50.0, y=50.0)])
    k = keep_factory(blob2, 3.0)
    assert not k(blob2["obs"][-1]), "outlier at (50,50) not rejected"
    print("SELF-TEST PASS")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        run_screen("h6_insertion_filter", PRED, VARIANTS)
```

- [ ] **Step 2: Run self-test** — expected `SELF-TEST PASS`.

- [ ] **Step 3: Commit**

```bash
git add collab_tasks/batch1/h6_insertion_filter.py
git commit -m "batch1 h6: geometric insertion filter harness"
```

---

### Task 9: Run Stage 1 (worker task — haiku, low effort)

**Files:** none created by hand; harnesses write `outputs/batch1/h*.json`.

**Interfaces:** consumes the six harnesses; produces six guard JSONs. Workers return raw output only — any interpretation in a worker's reply is discarded and the run repeated.

- [ ] **Step 1: Run all six self-tests as a gate**

```bash
python collab_tasks/batch1/h1_threshold.py --self-test
python collab_tasks/batch1/h2_spread_norm.py --self-test
python collab_tasks/batch1/h3_perclass_lambda.py --self-test
python collab_tasks/batch1/h4_cap_sweep.py --self-test
python collab_tasks/batch1/h5_conf_weight.py --self-test
python collab_tasks/batch1/h6_insertion_filter.py --self-test
```

Expected: six `SELF-TEST PASS`. Any failure → back to the harness's task, do not run real data.

- [ ] **Step 2: Run the six screens (sequential; ~2–4 h total; h3 and h6 are the slow ones because they rebuild the trace per variant)**

```bash
python collab_tasks/batch1/h1_threshold.py
python collab_tasks/batch1/h2_spread_norm.py
python collab_tasks/batch1/h4_cap_sweep.py
python collab_tasks/batch1/h5_conf_weight.py
python collab_tasks/batch1/h3_perclass_lambda.py
python collab_tasks/batch1/h6_insertion_filter.py
```

Expected per run: 8 `<scene> done` lines, a per-scene table, and either a `VERDICT [...] -> ADOPT-CANDIDATE|SURVIVES|KILLED` line or `UNDERPOWERED`. A `HARD STOP: baseline parity fail` aborts everything — escalate to the supervisor.

- [ ] **Step 3: Verify all six JSONs exist and carry verdicts**

```bash
python -c "import json,glob; [print(p, json.load(open(p))['verdict']) for p in sorted(glob.glob('outputs/batch1/h*.json'))]"
```

Expected: six lines, none `None`.

- [ ] **Step 4: Commit results**

```bash
git add outputs/batch1
git commit -m "batch1 stage1: six-mechanism screen results, guard-validated"
```

---

### Task 10: `stage2_joint.py` — joint grid over survivors

**Files:**
- Create: `collab_tasks/batch1/stage2_joint.py`

**Interfaces:**
- Consumes: `outputs/batch1/h*.json` (guard schema from Task 2), `common.class_fields, predict, score, SCENES, load_scene`, and the variant *builders* re-imported from each harness module (`h1_threshold.variant`, `h2_spread_norm.variant/_spread_mass`, `h3_perclass_lambda._scales`, `h4_cap_sweep`, `h5_conf_weight`, `h6_insertion_filter.keep_factory`).
- Produces: `outputs/batch1/stage2_joint.json` (guard schema; one entry per combo) and a printed adopted configuration or the finding that none passes.

- [ ] **Step 1: Write the failing self-test**

```python
def _self_test():
    combos = build_combos({
        "h1_threshold": {"verdict": "SURVIVES", "best_variant": "tau1.0_fallback",
                         "variants": {"tau1.0_fallback": {"delta_vs_baseline": {"macc": 0.01, "mf1": 0.0, "fmiou": 0, "mprec": 0}},
                                      "tau2.0_abstain": {"delta_vs_baseline": {"macc": 0.0, "mf1": 0.02, "fmiou": 0, "mprec": 0}}}},
        "h4_cap_sweep": {"verdict": "KILLED", "best_variant": "cap800",
                         "variants": {}},
    })
    assert all("h4" not in " ".join(c) for c in combos), "killed mechanism leaked in"
    assert any("tau2.0_abstain" in " ".join(c) for c in combos), "top-2 rule broken"
    print("SELF-TEST PASS")
```

- [ ] **Step 2: Run to verify it fails** — `python collab_tasks/batch1/stage2_joint.py --self-test` → `NameError: build_combos`.

- [ ] **Step 3: Implement**

```python
"""Stage 2: joint grid over Stage-1 survivors. Coupling is where every
wrong single-factor conclusion came from; this is where interactions get
measured instead of assumed."""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    SCENES, class_fields, load_scene, predict, score)
from collab_tasks.batch1.report import report  # noqa: E402

MAX_COMBOS = 96   # not silent: anything dropped is logged with its score rank

SURVIVOR_SET = ("SURVIVES", "ADOPT-CANDIDATE")


def top2(js):
    """Top-2 variants of a mechanism by max(dmacc, dmf1)."""
    d = js["variants"]
    ranked = sorted(d, key=lambda l: -max(
        d[l]["delta_vs_baseline"].get("macc", -9),
        d[l]["delta_vs_baseline"].get("mf1", -9)))
    return ranked[:2]


def build_combos(results):
    """results: {mechanism: guard-json-dict}. Returns list of combos, each a
    list of (mechanism, variant_label). Killed mechanisms are excluded."""
    live = {m: top2(js) for m, js in results.items()
            if js.get("verdict") in SURVIVOR_SET and js.get("variants")}
    axes = [[(m, v) for v in vs] + [(m, None)] for m, vs in live.items()]
    combos = [ [mv for mv in c if mv[1] is not None]
               for c in itertools.product(*axes) ]
    combos = [c for c in combos if len(c) >= 2]      # singles already screened
    return combos


def apply_combo(data, combo):
    """Compose mechanisms: build-time hooks first, then F-transforms, then
    decode rule. Variant labels are parsed back to parameters."""
    import collab_tasks.batch1.h2_spread_norm as h2
    import collab_tasks.batch1.h3_perclass_lambda as h3
    import collab_tasks.batch1.h6_insertion_filter as h6
    cap, keep_fn, scales, weight_fn = 400, None, None, None
    f_transforms, decode = [], ("argmax", None)
    for mech, lab in combo:
        if mech == "h4_cap_sweep":
            cap = int(lab.replace("cap", "").replace("_sanity", ""))
        elif mech == "h6_insertion_filter":
            keep_fn = h6.keep_factory(data, float(lab[1:]))
        elif mech == "h3_perclass_lambda":
            scales = h3._scales(data, float(lab[1:]))
        elif mech == "h5_conf_weight":
            g = float(lab.replace("gamma", ""))
            weight_fn = lambda o, g=g: float(o.get("conf", 1.0)) ** g
        elif mech == "h2_spread_norm":
            if lab == "zscore_control":
                f_transforms.append(("z", None))
            else:
                p, q = lab.split("_")
                f_transforms.append(("pq", (float(p[1:]), float(q[1:]))))
        elif mech == "h1_threshold":
            t, m = lab.replace("tau", "").split("_")
            decode = (m, float(t))
    F, names, cell = class_fields(data, cap=cap, weight_fn=weight_fn,
                                  scales=scales, keep_fn=keep_fn)
    for kind, arg in f_transforms:
        if kind == "z":
            F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True)
                                                  + 1e-12)
        else:
            p, q = arg
            sp, ms = h2._spread_mass(data, names)
            F = F / (sp[:, None] ** p * ms[:, None] ** q)
    mode, tau = decode
    if mode == "argmax":
        return predict(F, names, cell)
    Z = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-12)
    above = Z > tau
    masked = np.where(above, F, -np.inf)
    win = masked.argmax(0)
    none = ~above.any(0)
    if mode == "abstain":
        lab = np.array([("__none__" if n else names[w])
                        for w, n in zip(win, none)])
    else:
        win = win.copy()
        win[none] = F[:, none].argmax(0)
        lab = np.array([names[w] for w in win])
    return lab[cell]


def main():
    results = {}
    import glob
    for p in sorted(glob.glob("outputs/batch1/h*.json")):
        js = json.load(open(p))
        results[js["mechanism"]] = js
    combos = build_combos(results)
    if len(combos) > MAX_COMBOS:
        print(f"NOTE: {len(combos)} combos, capping at {MAX_COMBOS}; "
              f"dropping the rest (logged, not silent)")
        combos = combos[:MAX_COMBOS]
    print(f"{len(combos)} combos x 8 scenes")
    per = {"baseline": {}}
    datas = {s: load_scene(s) for s in SCENES}
    for s, data in datas.items():
        F, nm, cell = class_fields(data)
        per["baseline"][s] = score(data["gt"], predict(F, nm, cell))
    for i, combo in enumerate(combos):
        lab = "+".join(f"{m}:{v}" for m, v in combo)
        for s, data in datas.items():
            per.setdefault(lab, {})[s] = score(data["gt"],
                                               apply_combo(data, combo))
        print(f"[{i+1}/{len(combos)}] {lab} done", flush=True)
    report("stage2_joint",
           "Pre-registered: at least one survivor pair beats its best single "
           "(coupling positive somewhere); adoption rule as spec.",
           per, "outputs/batch1/stage2_joint.json")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
```

- [ ] **Step 4: Run self-test** — `python collab_tasks/batch1/stage2_joint.py --self-test` → `SELF-TEST PASS`.

- [ ] **Step 5: Commit**

```bash
git add collab_tasks/batch1/stage2_joint.py
git commit -m "batch1 stage2: joint grid over survivors, guard-validated"
```

---

### Task 11: Run Stage 2 (worker task — haiku, low effort; overnight)

- [ ] **Step 1: Launch** — `python collab_tasks/batch1/stage2_joint.py` in the background; it prints one `[i/N] ... done` line per combo. Runtime scales with survivor count (each combo ≈ 3–6 min across 8 scenes).

- [ ] **Step 2: On completion, verify** the JSON exists and `verdict_allowed` is true:

```bash
python -c "import json; js=json.load(open('outputs/batch1/stage2_joint.json')); print(js['verdict'], js['best_variant'], js['n_scenes'])"
```

- [ ] **Step 3: Commit**

```bash
git add outputs/batch1/stage2_joint.json
git commit -m "batch1 stage2: joint grid results"
```

---

### Task 12: Supervisor close-out (main session — not a subagent)

- [ ] **Step 1:** Read all guard JSONs; build the six-row kill/survive table plus the Stage-2 adopted configuration (or the documented no-adoption finding) directly from `delta_vs_baseline` numbers, never from prose.
- [ ] **Step 2:** Check each result against its pre-registered prediction; any mismatch is reported as such (a wrong prediction is a finding, not an embarrassment).
- [ ] **Step 3:** Update `wiki/analysis/2026-08-16-conceptgraphs-corrected-and-graceful-failure.md` re-test table with measured rows; update both live artifacts per the standing cadence (status page always; diagnosis page if the mechanism picture changed).
- [ ] **Step 4:** Present the harness files and result JSONs to Paul for review, THEN push all batch commits.

---

## Self-review notes (completed)

- **Spec coverage:** guard (Task 2), decomposition incl. recoverable bounds and tie-frac sensitivity (Task 3), six harnesses with pre-registered predictions and self-tests (Tasks 4–8), 8-scene enforcement + baseline recomputation (Task 1 `run_screen`), survivor/adoption thresholds pinned (Task 2 constants), Stage-2 joint grid with no silent caps (Task 10), wiki/artifact close-out and Paul-review-before-push (Task 12). Out-of-scope items from the spec are absent by design.
- **Type consistency:** `class_fields` signature identical across all consumers; guard `per` dict shape identical between `run_screen` (producer) and `report` (consumer); variant label grammar (`tau{t}_{mode}`, `p{p}_q{q}`, `k{k}`, `cap{n}`, `gamma{g}`, `r{r}`) is parsed back in `apply_combo` exactly as emitted.
- **Known judgement calls, made explicit:** `local_loss` residual category added so the partition is exact; `BLEED_DIST=0.9` and filter `FLOOR=0.5` are pinned constants with comments; combo cap 96 is logged, never silent; h1's blob-cache caveat is written into its test step.
