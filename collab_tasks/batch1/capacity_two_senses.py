"""'Capacity' means two different things. Measure both, separately.

SENSE 1 -- RETRIEVAL FIDELITY. Can we store many things and still read one out
without the answer being swamped by the others? This is the VSA sense:
crosstalk between superposed items. Measured here by rebuilding each class's
field ALONE (no other class in the trace) and comparing it to that class's
field read out of the full superposed trace. The difference IS the crosstalk.

SENSE 2 -- TOTAL BYTES. How much does the representation cost to store? That
is the dimension sweep (already run: 512..16384 dims = 4..128 KB, mAcc moves
~0.06 across the whole 32x range and only +0.006 from 32->128 KB).

They are linked -- more bytes buys more retrieval fidelity -- but the CLAIM
"capacity is not the constraint" is about sense 1, established by varying
sense 2 and watching accuracy saturate.

Why 13,124 observations fit in 4,096 dimensions at all: the trace regroups as
T = sum_c sem[c] * (sum_{o in c} phi(x_o)), so observations of the SAME class
add COHERENTLY into one class term. The superposition is over ~16 class terms,
not 13k independent items. This script also sweeps the class count to find
where that stops being true -- the operating-envelope question.

    python collab_tasks/batch1/capacity_two_senses.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    CAP, GRID, HD, LX, LY, SEEDS, _Enc, _bundle, cap_per_class, class_fields,
    load_scene, predict, score)
from vsa_cognitive_mapping.object_grounding import class_phasors  # noqa: E402


def fields_for(obs, names, sem, Bx, By, gx, gy, subset):
    """Field of each class in `subset`, from a trace built ONLY of `subset`."""
    keep = [o for o in obs if o["cls"] in subset]
    tr = _bundle(keep, subset, sem, Bx, By, None, None)
    tr /= max(np.abs(tr).max(), 1e-12)
    out = {}
    PX = np.conj(Bx[None, :] ** (gx[:, None] / LX))
    PY = np.conj(By[None, :] ** (gy[:, None] / LY))
    for c in subset:
        v = tr / sem[c]
        out[c] = ((PX * v[None, :]) @ PY.T).T.reshape(-1).real
    return out


def main():
    scene = "room0"
    d = load_scene(scene)
    obs = cap_per_class(list(d["obs"]), CAP, seed=SEEDS[0][2])
    names = sorted({o["cls"] for o in obs})
    sem = class_phasors(names, HD, seed=SEEDS[0][1])
    enc = _Enc(HD, SEEDS[0][0], LX, LY)
    Bx, By = enc.Bx.values, enc.By.values
    xyz, a, b = d["xyz"], d["a"], d["b"]
    gx = np.linspace(xyz[:, a].min(), xyz[:, a].max(), GRID)
    gy = np.linspace(xyz[:, b].min(), xyz[:, b].max(), GRID)

    print(f"SENSE 1 -- RETRIEVAL FIDELITY ({scene}, {len(obs)} observations, "
          f"{len(names)} classes, HD={HD})\n")
    full = fields_for(obs, names, sem, Bx, By, gx, gy, names)
    print(f"{'classes in trace':>17}{'corr with solo field':>23}"
          f"{'signal/crosstalk':>19}")
    print("-" * 60)
    rows = []
    for k in (2, 4, 8, 12, len(names)):
        subset = names[:k]
        multi = fields_for(obs, names, sem, Bx, By, gx, gy, subset)
        cs, snrs = [], []
        for c in subset:
            solo = fields_for(obs, names, sem, Bx, By, gx, gy, [c])[c]
            m = multi[c]
            cs.append(float(np.corrcoef(solo, m)[0, 1]))
            # crosstalk = what the superposition added to the solo answer
            noise = m - solo * (np.std(m) / max(np.std(solo), 1e-12))
            snrs.append(float(np.std(m) / max(np.std(noise), 1e-12)))
        rows.append((k, float(np.mean(cs)), float(np.mean(snrs))))
        print(f"{k:>17}{np.mean(cs):>23.4f}{np.mean(snrs):>19.2f}")
    print("-" * 60)
    print("corr 1.0 = the superposed readout is identical to reading that")
    print("class alone. Anything below 1.0 is crosstalk from the others.\n")

    # what does removing superposition entirely buy on the METRIC?
    F, nm, cell = class_fields(d, seeds=SEEDS[0])
    m_super = score(d["gt"], predict(F, nm, cell))["macc"]
    solo_stack = np.stack([fields_for(obs, names, sem, Bx, By, gx, gy, [c])[c]
                           for c in names])
    m_solo = score(d["gt"], predict(solo_stack, names, cell))["macc"]
    print(f"mAcc, one shared 32 KB trace  : {m_super:.4f}")
    print(f"mAcc, one trace PER CLASS     : {m_solo:.4f}   "
          f"({m_solo-m_super:+.4f})")
    print(f"  -> that delta is the ENTIRE cost of superposition. Everything")
    print(f"     any codebook or extra-dimension fix could ever buy is bounded")
    print(f"     by it, because it is the limit of not superposing at all.")
    print(f"  -> and it costs {len(names)}x the memory to get.")

    json.dump(dict(scene=scene, hd=HD, n_obs=len(obs), n_classes=len(names),
                   fidelity=[dict(k=k, corr=c, snr=s) for k, c, s in rows],
                   macc_superposed=float(m_super), macc_per_class=float(m_solo),
                   cost_of_superposition=float(m_super - m_solo)),
              open("outputs/batch1/capacity_two_senses.json", "w"), indent=1)
    print("\nwrote outputs/batch1/capacity_two_senses.json")


if __name__ == "__main__":
    main()
