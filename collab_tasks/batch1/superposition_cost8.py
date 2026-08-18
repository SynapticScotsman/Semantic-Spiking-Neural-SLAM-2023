"""The cost of superposition, on all 8 scenes. Resolves a disputed number.

A workflow subagent reported the zero-crosstalk ceiling as +0.0098 and used it
to retire the whole interference/codebook family. A single-scene re-measurement
gave +0.0439 on room0. Those disagree by 4.5x, so neither is quotable until
this runs on all 8.

Method: decode each class from a trace built ONLY of that class -- no other
class present, so zero crosstalk -- and argmax across those per-class fields.
That is the limit of not superposing at all, and therefore a hard upper bound
on anything a better codebook, more dimensions, or fewer items could buy.

It costs n_classes x the memory, so it is not a proposal; it is a ceiling.

    python collab_tasks/batch1/superposition_cost8.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    CAP, GRID, HD, LX, LY, SCENES, SEEDS, _Enc, _bundle, cap_per_class,
    class_fields, load_scene, predict, score)
from vsa_cognitive_mapping.object_grounding import class_phasors  # noqa: E402


def main():
    print(f"{'scene':<9}{'classes':>8}{'superposed':>12}{'per-class':>11}"
          f"{'delta':>9}{'memory cost':>13}")
    print("-" * 62)
    sup, per, rows = [], [], []
    for s in SCENES:
        d = load_scene(s)
        obs = cap_per_class(list(d["obs"]), CAP, seed=SEEDS[0][2])
        names = sorted({o["cls"] for o in obs})
        sem = class_phasors(names, HD, seed=SEEDS[0][1])
        enc = _Enc(HD, SEEDS[0][0], LX, LY)
        Bx, By = enc.Bx.values, enc.By.values
        xyz, a, b = d["xyz"], d["a"], d["b"]
        gx = np.linspace(xyz[:, a].min(), xyz[:, a].max(), GRID)
        gy = np.linspace(xyz[:, b].min(), xyz[:, b].max(), GRID)
        PX = np.conj(Bx[None, :] ** (gx[:, None] / LX))
        PY = np.conj(By[None, :] ** (gy[:, None] / LY))

        F, nm, cell = class_fields(d, seeds=SEEDS[0])
        m_sup = score(d["gt"], predict(F, nm, cell))["macc"]

        solo = []
        for c in names:
            keep = [o for o in obs if o["cls"] == c]
            tr = _bundle(keep, [c], sem, Bx, By, None, None)
            tr /= max(np.abs(tr).max(), 1e-12)
            v = tr / sem[c]
            solo.append(((PX * v[None, :]) @ PY.T).T.reshape(-1).real)
        m_per = score(d["gt"], predict(np.stack(solo), names, cell))["macc"]

        sup.append(m_sup)
        per.append(m_per)
        rows.append(dict(scene=s, n_classes=len(names),
                         superposed=float(m_sup), per_class=float(m_per)))
        print(f"{s:<9}{len(names):>8}{m_sup:>12.4f}{m_per:>11.4f}"
              f"{m_per-m_sup:>+9.4f}{str(len(names))+'x':>13}")
    print("-" * 62)
    dm = np.mean(per) - np.mean(sup)
    print(f"{'MEAN':<9}{'':>8}{np.mean(sup):>12.4f}{np.mean(per):>11.4f}"
          f"{dm:>+9.4f}")
    n = sum(1 for a_, b_ in zip(per, sup) if a_ > b_)
    print(f"\nremoving superposition entirely helps on {n}/8 scenes, "
          f"mean {dm:+.4f} mAcc")
    print(f"that is {100*dm/0.0822:.0f}% of the 0.0822 memory gap, "
          f"bought with {np.mean([r['n_classes'] for r in rows]):.0f}x "
          f"the memory")
    print(f"seed-noise floor is 0.017, so this ceiling is "
          f"{'ABOVE' if dm > 0.017 else 'BELOW'} it")
    json.dump(dict(per_scene=rows, mean_superposed=float(np.mean(sup)),
                   mean_per_class=float(np.mean(per)),
                   cost_of_superposition=float(dm)),
              open("outputs/batch1/superposition_cost8.json", "w"), indent=1)
    print("\nwrote outputs/batch1/superposition_cost8.json")


if __name__ == "__main__":
    main()
