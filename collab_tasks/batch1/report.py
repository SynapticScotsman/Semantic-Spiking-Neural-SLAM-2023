"""The reporting guard. Every batch-1 result goes through report() or it does
not exist.

Refuses a verdict on under-powered input -- the structural fix for six
single-scene / single-metric inversions on 2026-08-15/16. Amended (A1, A6,
2026-08-17) for the seed battery and machine-checkable predictions:

  - input carries all 8 scenes x all 5 seed tuples x all 4 metrics, or no
    verdict is printed at all;
  - deltas are PAIRED per tuple (variant minus that tuple's own baseline),
    averaged across tuples, with the across-tuple sd printed beside them;
  - a delta is RESOLVED only if |mean| >= 2*sd. Unresolved effects verdict as
    UNDECIDABLE, never KILLED -- an effect inside the noise band is
    unmeasured, not absent;
  - PREDICTIONS (Amendment A6 schema) are evaluated HIT/MISS mechanically and
    written into the JSON, so pre-registration cannot be reinterpreted after
    the fact.

Deliberately dependency-free (stdlib + numpy only): a broken common.py must
not be able to break the guard.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

METRICS = ("macc", "fmiou", "mprec", "mf1")
SURVIVE_EPS = 0.005        # pre-registered, spec 2026-08-16
ADOPT_F1_FLOOR = -0.005    # pre-registered, spec 2026-08-16
N_SCENES = 8
N_TUPLES = 5


def _mean_sd(vals):
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _validate(per):
    problems = []
    if "baseline" not in per:
        return ["no baseline block"]
    for lab, by_scene in per.items():
        if len(by_scene) != N_SCENES:
            problems.append(f"{lab}: {len(by_scene)} scenes, need {N_SCENES}")
        for s, by_t in by_scene.items():
            if len(by_t) != N_TUPLES:
                problems.append(f"{lab}/{s}: {len(by_t)} tuples, "
                                f"need {N_TUPLES}")
            for t, m in by_t.items():
                missing = [k for k in METRICS if k not in m]
                if missing:
                    problems.append(f"{lab}/{s}/{t}: missing {missing}")
                bad = [k for k in METRICS
                       if k in m and not math.isfinite(float(m[k]))]
                if bad:
                    problems.append(f"{lab}/{s}/{t}: non-finite {bad}")
    return problems


def _deltas(per, lab):
    """Paired per-tuple deltas (8-scene mean, variant minus baseline), then
    mean and sd across tuples, per metric. Also per-scene mean deltas."""
    tuples = sorted(next(iter(per["baseline"].values())))
    out = {}
    for k in METRICS:
        per_t = []
        for t in tuples:
            d = [per[lab][s][t][k] - per["baseline"][s][t][k]
                 for s in per["baseline"]]
            per_t.append(float(np.mean(d)))
        mean, sd = _mean_sd(per_t)
        out[k] = dict(mean=mean, sd=sd, per_tuple=per_t,
                      resolved=abs(mean) >= 2 * sd)
    out["per_scene_macc"] = {
        s: float(np.mean([per[lab][s][t]["macc"] - per["baseline"][s][t]["macc"]
                          for t in tuples]))
        for s in per["baseline"]}

    # BREADTH (added 2026-08-17). The seed battery guards variance across
    # DRAWS; it says nothing about concentration across SCENES. Measured
    # failure that motivated this: the batch-1 adopted decode reads +0.0155
    # over 8 scenes but +0.0018 with office4 removed -- 6 of 8 scenes improve,
    # so the direction is real, but essentially all the magnitude is one
    # scene. A mean carried by a single scene is not a general mechanism.
    ps = out["per_scene_macc"]
    scenes = sorted(ps)
    loo = {s: float(np.mean([ps[o] for o in scenes if o != s]))
           for s in scenes}
    worst = min(loo, key=lambda s: loo[s])   # scene whose removal hurts most
    out["breadth"] = dict(
        n_improved=int(sum(v > 0 for v in ps.values())),
        leave_one_out_min=loo[worst],
        carried_by=worst,
        robust=bool(loo[worst] > SURVIVE_EPS))
    return out


def _status(d):
    """Verdict for one variant from its delta block (Amendment A1 rules,
    plus the 2026-08-17 breadth requirement).

    ADOPT-CANDIDATE now additionally requires the effect to survive removal of
    its best scene: a mean carried by one scene is SCENE-DEPENDENT, not
    adoptable. Such variants fall back to SURVIVES so the effect is still
    reported and can be investigated -- it is a narrowing, not a kill.
    """
    ma, f1 = d["macc"], d["mf1"]
    if ma["resolved"] and ma["mean"] > SURVIVE_EPS \
            and f1["mean"] > ADOPT_F1_FLOOR:
        if not d.get("breadth", {}).get("robust", True):
            return "SCENE-DEPENDENT"
        return "ADOPT-CANDIDATE"
    if (ma["resolved"] and ma["mean"] > SURVIVE_EPS) or \
            (f1["resolved"] and f1["mean"] > SURVIVE_EPS):
        return "SURVIVES"
    if ma["resolved"] and f1["resolved"]:
        return "KILLED"
    return "UNDECIDABLE"


_PRECEDENCE = ["ADOPT-CANDIDATE", "SCENE-DEPENDENT", "SURVIVES",
               "UNDECIDABLE", "KILLED"]


def _eval_prediction(p, deltas, best):
    """Amendment A6. p = {id, text, metric, scope, test, value}."""
    metric, test, value = p["metric"], p["test"], float(p["value"])

    def mean_of(lab):
        return deltas[lab][metric]["mean"]

    def paired_gap(a, b):
        """Per-tuple gap between two variants, with its own sd.

        The first version compared only the means, so a gap sitting inside the
        noise band read as a clean HIT -- and a variant that internally applies
        an already-adopted mechanism could inherit that mechanism's gain
        unchallenged. The gap now has to clear twice its own sd, exactly like a
        delta against baseline.
        """
        pa = deltas[a][metric]["per_tuple"]
        pb = deltas[b][metric]["per_tuple"]
        g = np.asarray(pa, float) - np.asarray(pb, float)
        m = float(g.mean())
        sd = float(g.std(ddof=1)) if len(g) > 1 else 0.0
        return m, sd, abs(m) >= 2 * sd

    if test == "pairs_majority_ge":
        res = [paired_gap(a, b) for a, b in p["scope"]]
        ok = [(m >= value and r) for m, _, r in res]
        return dict(p, hit=bool(sum(ok) > len(res) / 2),
                    measured=[[round(m, 4), round(sd, 4), r]
                              for m, sd, r in res],
                    note="each gap must be resolved")
    if test == "pair_ge":
        a, b = p["scope"]
        m, sd, r = paired_gap(a, b)
        return dict(p, hit=bool(m >= value and r),
                    measured=[round(m, 4), round(sd, 4), r],
                    note="gap must be resolved")

    lab = best if p["scope"] == "best" else p["scope"]
    if lab not in deltas:
        return dict(p, hit=False, measured=None,
                    note=f"scope {lab} not present")
    if test == "scenes_ge":
        n = sum(v > 0 for v in deltas[lab]["per_scene_macc"].values())
        return dict(p, hit=bool(n >= value), measured=n)
    m = mean_of(lab)
    hit = {"ge": m >= value, "le": m <= value,
           "within": abs(m) <= value}[test]
    return dict(p, hit=bool(hit), measured=round(m, 4))


def report(mechanism, prediction, per, out_path, predictions=None):
    problems = _validate(per)
    allowed = not problems
    tuples = (sorted(next(iter(per["baseline"].values())))
              if per.get("baseline") else [])

    deltas, statuses = {}, {}
    if allowed:
        for lab in per:
            if lab == "baseline":
                continue
            deltas[lab] = _deltas(per, lab)
            statuses[lab] = _status(deltas[lab])

    # per-scene mAcc table at the reference tuple, mean +- across-tuple sd
    print(f"\n[{mechanism}] mAcc, mean across {len(tuples)} seed tuples "
          "(sd across tuples):")
    scenes = sorted(per.get("baseline", {}))
    for lab in per:
        cells = []
        for s in scenes:
            # robust to the malformed input this guard exists to refuse --
            # a missing scene/tuple prints as '-', never crashes the report
            vals = [per[lab][s][t]["macc"] for t in tuples
                    if s in per[lab] and t in per[lab][s]
                    and "macc" in per[lab][s][t]]
            cells.append(f"{np.mean(vals):.3f}" if vals else "-")
        row = "".join(f"{c:>8}" for c in cells)
        if lab == "baseline":
            print(f"  {row}   baseline")
        else:
            d = deltas.get(lab, {}).get("macc", {})
            print(f"  {row}   {lab}  d={d.get('mean', float('nan')):+.4f} "
                  f"+-{d.get('sd', float('nan')):.4f} "
                  f"[{statuses.get(lab, '?')}]")

    verdict = best = None
    if allowed and deltas:
        n_var = len(deltas)
        for want in _PRECEDENCE:
            cands = [l for l, st in statuses.items() if st == want]
            if cands:
                best = max(cands, key=lambda l: max(deltas[l]["macc"]["mean"],
                                                    deltas[l]["mf1"]["mean"]))
                verdict = want
                break
        d = deltas[best]
        br = d.get("breadth", {})
        print(f"VERDICT [{mechanism}] best={best} "
              f"dmacc={d['macc']['mean']:+.4f}+-{d['macc']['sd']:.4f} "
              f"dmf1={d['mf1']['mean']:+.4f}+-{d['mf1']['sd']:.4f} "
              f"-> {verdict}")
        if br:
            print(f"  breadth: {br['n_improved']}/8 scenes improved; "
                  f"drop {br['carried_by']} and the mean becomes "
                  f"{br['leave_one_out_min']:+.4f} "
                  f"({'robust' if br['robust'] else 'CARRIED BY ONE SCENE'})")
        print(f"  (selected over {n_var} variants; the 2*sd resolution rule "
              "is the multiple-comparisons brake -- treat a marginal best "
              "among many variants with suspicion)")
    else:
        print("UNDERPOWERED - no verdict")
        for p in problems:
            print("  " + p)

    pred_results = []
    if predictions and allowed and deltas:
        for p in predictions:
            r = _eval_prediction(p, deltas, best)
            pred_results.append(r)
            print(f"  PREDICTION {r['id']}: "
                  f"{'HIT' if r['hit'] else 'MISS'}  "
                  f"(measured {r['measured']})  {r['text']}")

    js = dict(mechanism=mechanism, prediction=prediction,
              n_scenes=len(per.get("baseline", {})), n_tuples=len(tuples),
              baseline_per_scene_per_tuple=per.get("baseline", {}),
              variants={lab: dict(per_scene_per_tuple=per[lab],
                                  deltas=deltas.get(lab, {}),
                                  status=statuses.get(lab))
                        for lab in per if lab != "baseline"},
              best_variant=best, verdict=verdict, verdict_allowed=allowed,
              problems=problems, predictions=pred_results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(js, open(out_path, "w"), indent=1)
    return js


def _self_test():
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "guard_test.json")
    scenes = ["room0", "room1", "room2", "office0", "office1", "office2",
              "office3", "office4"]
    ts = [f"t{i}" for i in range(5)]

    def block(macc, mf1, jitter=0.0):
        rng = np.random.RandomState(0)
        return {s: {t: dict(macc=macc + jitter * rng.randn(),
                            fmiou=0.3, mprec=0.15,
                            mf1=mf1 + jitter * rng.randn())
                    for t in ts} for s in scenes}

    print("1/5 three scenes must refuse...")
    small = {s: block(0.3, 0.16)[s] for s in scenes[:3]}
    r = report("t", "p", {"baseline": small, "v": small}, tmp)
    assert r["verdict_allowed"] is False and r["verdict"] is None

    print("2/5 missing tuple must refuse...")
    partial = block(0.3, 0.16)
    partial["room0"] = {t: partial["room0"][t] for t in ts[:3]}
    r = report("t", "p", {"baseline": block(0.3, 0.16), "v": partial}, tmp)
    assert r["verdict_allowed"] is False

    print("3/5 clear resolved gain must ADOPT-CANDIDATE...")
    r = report("t", "p", {"baseline": block(0.30, 0.16),
                          "v": block(0.32, 0.17, jitter=0.0005)}, tmp)
    assert r["verdict"] == "ADOPT-CANDIDATE", r["verdict"]

    print("4/5 in-noise effect must be UNDECIDABLE, never KILLED...")
    r = report("t", "p", {"baseline": block(0.30, 0.16),
                          "v": block(0.302, 0.161, jitter=0.01)}, tmp)
    assert r["verdict"] == "UNDECIDABLE", r["verdict"]

    print("5/5 predictions score mechanically...")
    preds = [dict(id="x-ge", text="macc up 0.005", metric="macc",
                  scope="best", test="ge", value=0.005),
             dict(id="x-within", text="mf1 within 0.02", metric="mf1",
                  scope="best", test="within", value=0.02)]
    r = report("t", "p", {"baseline": block(0.30, 0.16),
                          "v": block(0.32, 0.17, jitter=0.0005)}, tmp,
               predictions=preds)
    got = {p["id"]: p["hit"] for p in r["predictions"]}
    assert got == {"x-ge": True, "x-within": True}, got
    print("SELF-TEST PASS")


if __name__ == "__main__":
    _self_test()
