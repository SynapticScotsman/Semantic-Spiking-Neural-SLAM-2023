"""The STRONGEST form of the codebook question, at a fixed bit budget.

lens2_codebook.py measures that making the class codebook exactly orthogonal
removes only the COHERENT crosstalk term, which is 1-2% of the crosstalk. The
remaining 98% is the residual Re<phi(x), A_c*(w-wbar)>, and no choice of
unit-modulus codebook can remove it, because it does not depend on the
codebook's inner products at all.

There IS one codebook that removes cross-class crosstalk exactly: a BLOCK
codebook. Partition the classes into K groups, give group k its own trace of
dimension HD/K, and let the "class vector" be the one-hot group indicator.
Total memory is unchanged (K traces x HD/K dims = HD dims = 32 KB). K=1 is
today's system. K=n_classes is one field per class -- no superposition left,
which is structurally ConceptGraphs' explicit map at our bit budget.

The model in lens2_codebook.cmd_theory predicts the field SNR ratio

    z_K / z_1 = sqrt( (n-1) / (n-K) )         (equal-mass classes)

so K=2 buys ~3%, K=4 ~12%, and the large gains only arrive as K -> n, i.e. as
superposition is abandoned. This measures the actual mAcc ladder rather than
trusting that.

Fields are RAW (unnormalised). In the single-trace case common.class_fields
divides by max|T|, a positive constant that cannot change an argmax; with K
traces a per-trace normaliser WOULD change the argmax, so it is dropped from
all arms including K=1 -- verified label-identical to the baseline at K=1.

    python collab_tasks/batch1/lens2_partition.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    CAP, GRID, HD, LX, LY, SEEDS, _Enc, cap_per_class, class_phasors,
    default_fields, load_scene, predict, score)

SCENES = ["room0", "room1", "room2", "office0"]


def partitioned_pred(data, K, seeds, cap=CAP, lx=LX, ly=LY, grid=GRID):
    base_seed, cb_seed, cap_seed = seeds
    obs = cap_per_class(list(data["obs"]), cap, seed=cap_seed)
    names = sorted({o["cls"] for o in obs})
    n = len(names)
    K = min(K, n)
    d = HD // K
    rng = np.random.RandomState(cb_seed + 1000)
    grp = np.array([i % K for i in rng.permutation(n)])   # balanced groups
    xyz, a, b = data["xyz"], data["a"], data["b"]
    xs, ys = xyz[:, a], xyz[:, b]
    gx = np.linspace(xs.min(), xs.max(), grid)
    gy = np.linspace(ys.min(), ys.max(), grid)
    F = np.empty((n, grid * grid))
    for k in range(K):
        members = [i for i in range(n) if grp[i] == k]
        if not members:
            continue
        enc = _Enc(d, base_seed + 97 * k, lx, ly)
        Bx, By = enc.Bx.values, enc.By.values
        sem = class_phasors([names[i] for i in members], d, seed=cb_seed + k)
        tr = np.zeros(d, np.complex128)
        for i in members:
            rows = [o for o in obs if o["cls"] == names[i]]
            P = np.array([[o["x"], o["y"]] for o in rows], float)
            tr += sem[names[i]] * ((Bx[None, :] ** (P[:, 0, None] / lx))
                                   * (By[None, :] ** (P[:, 1, None] / ly))
                                   ).sum(0)
        PX = np.conj(Bx[None, :] ** (gx[:, None] / lx))
        PY = np.conj(By[None, :] ** (gy[:, None] / ly))
        for i in members:
            v = tr / sem[names[i]]
            F[i] = ((PX * v[None, :]) @ PY.T).T.reshape(-1).real
    ix = np.clip(np.searchsorted(gx, xs), 0, grid - 1)
    iy = np.clip(np.searchsorted(gy, ys), 0, grid - 1)
    cell = iy * grid + ix
    return np.array([names[w] for w in F.argmax(0)[cell]])


def main():
    Ks = [1, 2, 4, 8, 16]
    out = {}
    print("mAcc vs number of trace SHARDS at a FIXED 32 KB total budget")
    print("(K=1 is today's single superposed trace; K=n is one field per "
          "class,\n no cross-class superposition left)")
    print(f"\n{'scene':<9}" + "".join(f"{'K=' + str(k):>9}" for k in Ks)
          + f"{'baseline':>10}")
    print("-" * (9 + 9 * len(Ks) + 10))
    for s in SCENES:
        data = load_scene(s)
        F, nm, cell = default_fields(data, 0)
        base = score(data["gt"], predict(F, nm, cell))["macc"]
        row = []
        for k in Ks:
            accs = [score(data["gt"],
                          partitioned_pred(data, k, SEEDS[t]))["macc"]
                    for t in range(3)]
            row.append(float(np.mean(accs)))
        out[s] = dict(zip([str(k) for k in Ks], row), baseline=base)
        print(f"{s:<9}" + "".join(f"{v:>9.3f}" for v in row)
              + f"{base:>10.3f}", flush=True)
    print("-" * (9 + 9 * len(Ks) + 10))
    print(f"{'mean':<9}" + "".join(
        f"{np.mean([out[s][str(k)] for s in SCENES]):>9.3f}" for k in Ks)
        + f"{np.mean([out[s]['baseline'] for s in SCENES]):>10.3f}")
    print("\n3 seed tuples per cell; the batch-1 believability threshold on "
          "the 8-scene\nmean is 0.017, so read only large moves here.")
    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(out, open("outputs/batch1/lens2_partition.json", "w"), indent=1)
    print("wrote outputs/batch1/lens2_partition.json")


if __name__ == "__main__":
    main()
