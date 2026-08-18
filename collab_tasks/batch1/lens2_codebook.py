"""LENS 2 -- the codebook / interference angle.

Question: the class codebook (vsa_cognitive_mapping/object_grounding.py:133)
draws INDEPENDENT random phasors per class. Only ~16 classes live in 4096
complex dims, so an exactly-orthogonal unit-modulus codebook is available for
free. Does the non-orthogonality explain the 7.7% INTERFERENCE signature --
our own class's field driven BELOW ZERO within 4 cm of its own evidence --
and would orthogonalising remove it?

Everything here is MEASURED on the real traces via collab_tasks/batch1/common.
Nothing is asserted from the algebra alone; the algebra only says which two
numbers to print.

THE DECOMPOSITION (this is the whole idea)
------------------------------------------
The trace is  T = sum_c sem[c] * A_c   with   A_c = sum_{o in c} phi(x_o).
The class-g field at a query point x is

    F_g(x) = Re<phi(x), T * conj(sem[g])>
           = Re<phi(x), A_g>                       SELF
           + sum_{c != g} Re<phi(x), A_c * w_cg>   CROSS,  w_cg = sem_c*conj(sem_g)

and each cross term splits exactly, with wbar = mean_d(w_cg) = <sem_c,sem_g>/HD:

    Re<phi(x), A_c * w_cg> = Re[ wbar * M_c(x) ]   COHERENT   (dies iff wbar == 0)
                           + Re<phi(x), A_c * (w_cg - wbar)>  RESIDUAL

where M_c(x) = <phi(x), A_c> is class c's own proximity mass at x -- the very
quantity a nearest-neighbour readout uses. So:

  * the COHERENT part is codebook overlap x competitor mass. An exactly
    orthogonal codebook makes it identically zero, at every x, for free.
  * the RESIDUAL part survives any codebook. It is the price of bundling.

Measuring the split at the losing cells therefore measures, with no new
experiment, the CEILING on what orthogonalisation can buy.

    python collab_tasks/batch1/lens2_codebook.py decompose
    python collab_tasks/batch1/lens2_codebook.py codebooks
    python collab_tasks/batch1/lens2_codebook.py swap --scenes room0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    CAP, GRID, HD, LX, LY, SCENES, SEEDS, _Enc, cap_per_class, class_fields,
    class_phasors, default_fields, load_scene, predict, score)

OUT = "outputs/batch1"


# ------------------------------------------------------------ codebooks ----

def cb_random(names, hd, seed):
    """The shipped codebook: object_grounding.class_phasors."""
    return class_phasors(names, hd, seed=seed)


def cb_dft(names, hd, seed):
    """Plain DFT rows: exp(2i.pi.k.d/HD), k = 0..n-1. Unit modulus, exactly
    orthogonal, and DELIBERATELY the structured worst case -- these are the
    rows the brief warns may correlate with position."""
    d = np.arange(hd)
    return {c: np.exp(2j * np.pi * k * d / hd) for k, c in enumerate(names)}


def cb_rdft(names, hd, seed):
    """Randomised DFT: exp(i(2.pi.k_c.perm(d)/HD + theta_d)).

    The per-column phase theta_d and the column permutation cancel in every
    pairwise inner product, so orthogonality is still EXACT and the modulus is
    still exactly 1 -- but no class vector is a monotone frequency ramp any
    more, which is the structure that could in principle alias onto position.
    """
    rng = np.random.RandomState(seed)
    theta = rng.uniform(-np.pi, np.pi, hd)
    perm = rng.permutation(hd)
    ks = rng.choice(np.arange(1, hd), size=len(names), replace=False)
    return {c: np.exp(1j * (2 * np.pi * ks[i] * perm / hd + theta))
            for i, c in enumerate(names)}


def _hadamard_row(r, hd):
    d = np.arange(hd)
    bits = np.bitwise_and(r, d)
    par = np.zeros(hd, np.int64)
    x = bits.copy()
    while np.any(x):
        par ^= (x & 1)
        x >>= 1
    return np.where(par == 1, -1.0, 1.0)


def cb_walsh(names, hd, seed):
    """Randomised Sylvester-Hadamard rows: +-1 (unit modulus) times a random
    per-column phase. Exactly orthogonal; HD=4096 is a power of two so the
    matrix exists. Row 0 (all ones) is skipped."""
    assert hd & (hd - 1) == 0, "Walsh needs a power-of-two HD"
    rng = np.random.RandomState(seed)
    theta = np.exp(1j * rng.uniform(-np.pi, np.pi, hd))
    rows = rng.choice(np.arange(1, hd), size=len(names), replace=False)
    return {c: _hadamard_row(int(rows[i]), hd) * theta
            for i, c in enumerate(names)}


def cb_gs(names, hd, seed):
    """Gram-Schmidt on random phasors, then renormalise to unit modulus.

    Included precisely BECAUSE it is the obvious thing to try and it is
    unsafe: orthogonalisation destroys unit modulus, and unbinding is
    np.divide (vsa.py Phasor.unbind), so renormalising is compulsory -- which
    in turn destroys exact orthogonality. Both defects are printed.
    """
    rng = np.random.RandomState(seed)
    V = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(names), hd)))
    Q, _ = np.linalg.qr(V.conj().T)
    Q = Q.conj().T                                  # (n, hd) orthonormal rows
    Q = Q / np.abs(Q)                               # back to unit modulus
    return {c: Q[i] for i, c in enumerate(names)}


def cb_ap(names, hd, seed, iters=200):
    """Alternating projection between {unit modulus} and {orthogonal rows}.

    The honest attempt to get both properties from a RANDOM start. Converges
    to machine-level orthogonality with 16 rows in 4096 dims (measured).
    """
    rng = np.random.RandomState(seed)
    V = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(names), hd)))
    for _ in range(iters):
        Q, _ = np.linalg.qr(V.conj().T)
        V = Q.conj().T * np.sqrt(hd)
        V = V / np.abs(V)
    return {c: V[i] for i, c in enumerate(names)}


CODEBOOKS = {"random": cb_random, "dft": cb_dft, "rdft": cb_rdft,
             "walsh": cb_walsh, "gs": cb_gs, "ap": cb_ap}


# ------------------------------------------------------------- machinery ---

def build(data, seeds=SEEDS[0], cb="random", cap=CAP, lx=LX, ly=LY,
          grid=GRID):
    """Rebuild the trace with an explicit codebook, keeping every other detail
    byte-identical to common.class_fields. Returns the PARTS, not just F."""
    base_seed, cb_seed, cap_seed = seeds
    obs = cap_per_class(list(data["obs"]), cap, seed=cap_seed)
    names = sorted({o["cls"] for o in obs})
    sem = CODEBOOKS[cb](names, HD, cb_seed)
    enc = _Enc(HD, base_seed, lx, ly)
    Bx, By = enc.Bx.values, enc.By.values

    A = np.zeros((len(names), HD), np.complex128)
    for n, c in enumerate(names):
        rows = [o for o in obs if o["cls"] == c]
        P = np.array([[o["x"], o["y"]] for o in rows], float)
        A[n] = ((Bx[None, :] ** (P[:, 0, None] / lx))
                * (By[None, :] ** (P[:, 1, None] / ly))).sum(0)
    S = np.array([sem[c] for c in names])
    tr = (S * A).sum(0)
    scale = 1.0 / max(np.abs(tr).max(), 1e-12)

    xyz, a, b = data["xyz"], data["a"], data["b"]
    xs, ys = xyz[:, a], xyz[:, b]
    gx = np.linspace(xs.min(), xs.max(), grid)
    gy = np.linspace(ys.min(), ys.max(), grid)
    PX = np.conj(Bx[None, :] ** (gx[:, None] / lx))
    PY = np.conj(By[None, :] ** (gy[:, None] / ly))
    F = np.empty((len(names), grid * grid))
    trn = tr * scale
    for n, c in enumerate(names):
        v = trn / sem[c]
        F[n] = ((PX * v[None, :]) @ PY.T).T.reshape(-1).real
    ix = np.clip(np.searchsorted(gx, xs), 0, grid - 1)
    iy = np.clip(np.searchsorted(gy, ys), 0, grid - 1)
    return dict(names=names, sem=sem, S=S, A=A, tr=tr, scale=scale, F=F,
                cell=iy * grid + ix, gx=gx, gy=gy, PX=PX, PY=PY, obs=obs,
                Bx=Bx, By=By, lx=lx, ly=ly, grid=grid)


def phi_at(B, gx, gy, cellidx, grid):
    """phi(x) for a flat cell index, from the same grid the fields use."""
    Bx, By = B
    iy, ix = divmod(int(cellidx), grid)
    return (Bx ** (gx[ix] / LX)) * (By ** (gy[iy] / LY))


def parity_check(data):
    """build(cb='random') must reproduce common.class_fields exactly."""
    b = build(data)
    F0, nm0, cell0 = class_fields(data)
    assert nm0 == b["names"], "class name order drifted"
    assert (cell0 == b["cell"]).all(), "cell index drifted"
    err = np.abs(np.asarray(F0) - b["F"]).max()
    assert err < 1e-8, f"field parity fail, max abs err {err:.3e}"
    return err


# --------------------------------------------------------- (a) decompose ---

def losing_cells(data, b):
    """The field_why.py INTERFERENCE population and its parent set: wrong
    cells inside local_loss where the GT class's OWN observation is nearer."""
    from collab_tasks.batch1.error_decomposition import decompose
    cats, _, pred = decompose(data)
    gt, xyz, a, bb = data["gt"], data["xyz"], data["a"], data["b"]
    idx = {c: i for i, c in enumerate(b["names"])}
    obs = {}
    for o in b["obs"]:
        obs.setdefault(o["cls"], []).append((o["x"], o["y"]))
    obs = {c: np.array(v) for c, v in obs.items()}
    rows = []
    for i in sorted(cats["local_loss"]):
        g, w = gt[i], pred[i]
        if g not in idx or w not in idx:
            continue
        px, py = float(xyz[i, a]), float(xyz[i, bb])
        dg = np.hypot(*(obs[g] - [px, py]).T).min()
        dw = np.hypot(*(obs[w] - [px, py]).T).min()
        if dg >= dw:
            continue
        rows.append(dict(i=int(i), g=g, w=w, gi=idx[g], wi=idx[w],
                         cell=int(b["cell"][i]), d_gt=float(dg),
                         d_win=float(dw)))
    return rows


def decompose_cells(b, rows, keys):
    """Exact SELF / COHERENT / RESIDUAL split of F_g at each listed cell."""
    S, A, names = b["S"], b["A"], b["names"]
    cells = np.array(sorted({r["cell"] for r in rows}), int)
    grid = b["grid"]
    PHI = np.empty((len(cells), HD), np.complex128)
    for j, cl in enumerate(cells):
        PHI[j] = phi_at((b["Bx"], b["By"]), b["gx"], b["gy"], cl, grid)
    pos = {int(c): j for j, c in enumerate(cells)}
    M = (np.conj(PHI) @ A.T)                       # (cells, classes) complex
    WB = (S @ np.conj(S).T) / HD                   # (c, g) codebook overlap
    out = []
    for r in rows:
        j, g, w = pos[r["cell"]], r["gi"], r["wi"]
        ph = PHI[j]
        self_t = float(np.real(np.vdot(ph, A[g]))) * b["scale"]
        per = np.empty(len(names))
        coh = np.empty(len(names))
        for c in range(len(names)):
            if c == g:
                per[c] = coh[c] = 0.0
                continue
            per[c] = float(np.real(np.vdot(ph, A[c] * S[c] * np.conj(S[g]))))
            coh[c] = float(np.real(WB[c, g] * M[j, c]))
        per *= b["scale"]
        coh *= b["scale"]
        d = dict(r)
        d.update(f_gt=float(b["F"][g, r["cell"]]),
                 f_win=float(b["F"][w, r["cell"]]),
                 self_term=self_t, cross_total=float(per.sum()),
                 coherent=float(coh.sum()), residual=float(per.sum()-coh.sum()),
                 worst_class=names[int(np.argmin(per))],
                 worst_cross=float(per.min()),
                 mass_gt=float(np.real(M[j, g])) * b["scale"])
        if keys:
            d["per_class"] = {names[c]: float(per[c]) for c in range(len(names))}
            d["per_class_coh"] = {names[c]: float(coh[c])
                                  for c in range(len(names))}
        out.append(d)
    return out


def cmd_decompose(args):
    data = load_scene(args.scene)
    err = parity_check(data)
    print(f"parity vs common.class_fields: max |dF| = {err:.2e}  OK")
    b = build(data)
    S, names = b["S"], b["names"]
    n = len(names)
    G = np.abs(S @ np.conj(S).T) / HD
    off = G[~np.eye(n, dtype=bool)]
    print(f"\n{args.scene}: {n} classes, HD={HD}, {len(b['obs'])} capped obs")
    print(f"codebook |<sem_c,sem_g>|/HD  off-diagonal: mean {off.mean():.4f}  "
          f"max {off.max():.4f}   (1/sqrt(HD) = {1/np.sqrt(HD):.4f})")
    print(f"trace normaliser max|T| = {1/b['scale']:.1f}")

    rows = losing_cells(data, b)
    print(f"\n{len(rows)} cells: local_loss AND the GT class's own observation "
          f"is nearer\n" + "=" * 76)
    dec = decompose_cells(b, rows, keys=True)
    inter = [d for d in dec if d["f_gt"] <= 0]
    print(f"INTERFERENCE (f_gt <= 0): {len(inter)}/{len(dec)} = "
          f"{100*len(inter)/max(len(dec),1):.1f}%")

    def block(tag, S_):
        if not S_:
            return
        q = lambda k: np.array([d[k] for d in S_])  # noqa: E731
        print(f"\n{tag}  ({len(S_)} cells)   medians")
        print(f"   F_gt        {np.median(q('f_gt')):+9.3f}")
        print(f"   SELF        {np.median(q('self_term')):+9.3f}   "
              f"(same-class evidence only -- always >= 0 in practice)")
        print(f"   CROSS       {np.median(q('cross_total')):+9.3f}   "
              f"= COHERENT {np.median(q('coherent')):+8.3f}  + RESIDUAL "
              f"{np.median(q('residual')):+8.3f}")
        print(f"   |CROSS|/SELF{np.median(np.abs(q('cross_total'))/np.maximum(q('self_term'),1e-9)):9.2f}")
        print(f"   |COH| share of |CROSS|: "
              f"{np.median(np.abs(q('coherent'))/np.maximum(np.abs(q('cross_total')),1e-9)):.2f}")
        cf = q("coherent")
        print(f"   cells the COHERENT part alone pushes negative-ward: "
              f"{100*(cf < 0).mean():.0f}%")

    block("ALL GT-nearer losses", dec)
    block("INTERFERENCE subset", inter)

    # counterfactual: what F_gt would have been with wbar == 0 (exact
    # orthogonal codebook), holding every observation and every base fixed.
    for tag, S_ in (("ALL", dec), ("INTERFERENCE", inter)):
        if not S_:
            continue
        f = np.array([d["f_gt"] for d in S_])
        c = np.array([d["coherent"] for d in S_])
        fw = np.array([d["f_win"] for d in S_])
        print(f"\n{tag}: counterfactual F_gt with the COHERENT term deleted")
        print(f"   median F_gt {np.median(f):+.3f} -> {np.median(f-c):+.3f}")
        print(f"   sign flips to positive: {100*((f<=0)&((f-c)>0)).mean():.1f}% "
              f"of the subset")
        print(f"   would now BEAT the winner (F_gt-coh > F_win, winner "
              f"unchanged): {100*((f-c) > fw).mean():.1f}%")

    # the cell field_transect.py published
    tgt = min(dec, key=lambda d: abs(d["d_gt"] - 0.0438) + abs(d["d_win"] - 0.1237))
    print("\nthe published transect cell (room0, GT sofa, winner cushion)")
    print(f"   d_gt {tgt['d_gt']:.4f} m   d_win {tgt['d_win']:.4f} m")
    print(f"   F_gt {tgt['f_gt']:+.3f}   F_win {tgt['f_win']:+.3f}")
    print(f"   SELF {tgt['self_term']:+.3f}   CROSS {tgt['cross_total']:+.3f} "
          f"(COH {tgt['coherent']:+.3f} + RES {tgt['residual']:+.3f})")
    top = sorted(tgt["per_class"].items(), key=lambda kv: kv[1])[:5]
    print("   worst cross terms:")
    for c, v in top:
        print(f"      {c:<14}{v:+9.3f}   of which coherent "
              f"{tgt['per_class_coh'][c]:+8.3f}")

    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/lens2_decompose_{args.scene}.json"
    json.dump(dict(scene=args.scene, n_classes=n, hd=HD,
                   codebook_offdiag_mean=float(off.mean()),
                   codebook_offdiag_max=float(off.max()),
                   n_cells=len(dec), n_interference=len(inter),
                   cells=dec), open(p, "w"))
    print(f"\nwrote {p}")


# --------------------------------------------------------- (b) codebooks ---

def cmd_codebooks(args):
    data = load_scene(args.scene)
    b0 = build(data)
    names, grid = b0["names"], b0["grid"]
    Bx, By, gx, gy = b0["Bx"], b0["By"], b0["gx"], b0["gy"]
    # a coarse position grid, in the SCENE's own coordinates
    sx = gx[::args.step]
    sy = gy[::args.step]
    PHI = ((Bx[None, :] ** (sx[:, None] / LX))[:, None, :]
           * (By[None, :] ** (sy[:, None] / LY))[None, :, :]
           ).reshape(-1, HD)
    print(f"{args.scene}: {len(names)} classes, HD={HD}, position probe grid "
          f"{len(sx)}x{len(sy)} = {len(PHI)} points\n")
    hdr = (f"{'codebook':<9}{'|sem| min':>11}{'|sem| max':>11}"
           f"{'max offdiag':>13}{'max |<w,phi(x)>|':>18}{'max |<sem,phi>|':>17}")
    print(hdr)
    print("-" * len(hdr))
    res = {}
    for tag, fn in CODEBOOKS.items():
        sem = fn(names, HD, SEEDS[0][1])
        S = np.array([sem[c] for c in names])
        mod = np.abs(S)
        G = np.abs(S @ np.conj(S).T) / HD
        offd = float(G[~np.eye(len(names), dtype=bool)].max())
        # (i) do the class vectors themselves look like a position?
        c_sem = float((np.abs(np.conj(PHI) @ S.T) / HD).max())
        # (ii) THE ONE THAT MATTERS: does the crosstalk carrier w = sem_c
        #      conj(sem_g) look like a position, i.e. can class c's map appear
        #      as a GHOST inside class g's field, translated?
        W = []
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    W.append(S[i] * np.conj(S[j]))
        W = np.array(W)
        c_w = float((np.abs(np.conj(PHI) @ W.T) / HD).max())
        print(f"{tag:<9}{mod.min():>11.4f}{mod.max():>11.4f}"
              f"{offd:>13.2e}{c_w:>18.4f}{c_sem:>17.4f}")
        res[tag] = dict(mod_min=float(mod.min()), mod_max=float(mod.max()),
                        max_offdiag=offd, max_w_pos=c_w, max_sem_pos=c_sem)
    print(f"\nreference: 1/sqrt(HD) = {1/np.sqrt(HD):.4f}; a random codebook's "
          f"columns are the null model for the last two columns.")
    print("division safety: unbinding is np.divide (vsa.py Phasor.unbind), so "
          "|sem| min\nmust be 1 -- anything below it amplifies that dimension "
          "by 1/|sem| on decode.")
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/lens2_codebooks_{args.scene}.json"
    json.dump(res, open(p, "w"), indent=1)
    print(f"wrote {p}")


# --------------------------------------------------------------- (b/c) ----

def cmd_swap(args):
    """End-to-end mAcc with the codebook swapped. Guard-compliant: the same
    seed battery, paired deltas, verdict only when |mean| >= 2*sd."""
    from collab_tasks.batch1.report import report
    which = args.codebooks or ["rdft", "walsh", "dft", "ap"]
    scenes = args.scenes or SCENES
    per = {"baseline": {}}
    for ti in range(len(SEEDS)):
        for s in scenes:
            data = load_scene(s)
            F, nm, cell = default_fields(data, ti)
            per["baseline"].setdefault(s, {})[f"t{ti}"] = \
                score(data["gt"], predict(F, nm, cell))
            for cb in which:
                b = build(data, seeds=SEEDS[ti], cb=cb)
                pr = predict(b["F"], b["names"], b["cell"])
                per.setdefault(cb, {}).setdefault(s, {})[f"t{ti}"] = \
                    score(data["gt"], pr)
            print(f"  t{ti} {s} done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    return report("lens2_codebook_swap",
                  "orthogonal class codebook removes the coherent crosstalk "
                  "term; predicted mAcc effect small (<0.017) because the "
                  "coherent share of cross is small, but sign should be +",
                  per, f"{OUT}/lens2_codebook_swap.json")


# ------------------------------------------------- (a) closed form check ---

def cmd_theory(args):
    """Closed form for the crosstalk, checked against the measured fields.

    Over the codebook draw (w_cg = sem_c conj(sem_g), unit phases iid across
    dimensions and independent of everything spatial),

        E[X_c(x)] = 0 ,   Var[X_c(x)] = ||A_c||^2 / 2        (any x)

    so the total crosstalk standard deviation is

        sd(CROSS) = sqrt( sum_{c != g} ||A_c||^2 / 2 )

    which does NOT depend on the query point. The crosstalk floor is FLAT
    over the room while the signal decays with the kernel. And

        ||A_c||^2 = HD * ( N_c + sum_{o != o'} k(x_o - x_o') )
                  = HD * N_c * ( 1 + (N_c - 1) * kbar_c )

    with kbar_c the mean within-class pairwise kernel. For a clumped class
    kbar_c -> 1 and ||A_c|| -> sqrt(HD) * N_c: crosstalk grows LINEARLY in the
    observation count, exactly as fast as the signal it competes with. That is
    why capacity (h4) could not help.
    """
    data = load_scene(args.scene)
    b = build(data)
    names, A, S = b["names"], b["A"], b["S"]
    sc = b["scale"]
    nA2 = (np.abs(A) ** 2).sum(1)                  # ||A_c||^2
    N = np.array([sum(1 for o in b["obs"] if o["cls"] == c) for c in names])
    kbar = (nA2 / HD - N) / np.maximum(N * (N - 1), 1)
    print(f"{args.scene}: per-class bundled mass (scaled by the trace "
          f"normaliser 1/{1/sc:.1f})")
    print(f"{'class':<14}{'N_c':>6}{'||A_c||':>12}{'kbar_c':>9}"
          f"{'sd contrib':>12}")
    print("-" * 53)
    order = np.argsort(-nA2)
    for n in order:
        print(f"{names[n]:<14}{N[n]:>6}{np.sqrt(nA2[n]):>12.0f}"
              f"{kbar[n]:>9.3f}{np.sqrt(nA2[n]/2)*sc:>12.3f}")
    tot = float(np.sqrt(nA2.sum() / 2) * sc)
    print(f"\npredicted sd(CROSS) over all classes = {tot:.2f} "
          f"(position-independent)")

    rows = losing_cells(data, b)
    dec = decompose_cells(b, rows, keys=False)
    ct = np.array([d["cross_total"] for d in dec])
    st = np.array([d["self_term"] for d in dec])
    # per-cell prediction drops the GT class's own ||A_g||
    pred_sd = np.array([float(np.sqrt((nA2.sum() - nA2[d["gi"]]) / 2) * sc)
                        for d in dec])
    print(f"measured   sd(CROSS) at the {len(dec)} GT-nearer losing cells "
          f"= {ct.std(ddof=1):.2f}")
    print(f"           mean predicted sd at those cells = {pred_sd.mean():.2f}")
    print(f"           |CROSS| median {np.median(np.abs(ct)):.2f}, "
          f"p90 {np.percentile(np.abs(ct), 90):.2f}")
    print(f"           SELF   median {np.median(st):.2f}")
    print(f"\nSIGNAL-TO-CROSSTALK at these cells: median SELF/sd(CROSS) = "
          f"{np.median(st/pred_sd):.2f}")
    print("A negative F_gt needs CROSS < -SELF, i.e. a "
          f"{np.median(st/pred_sd):.2f}-sigma downward draw of a zero-mean "
          "crosstalk term.")
    z = np.median(st / pred_sd)
    from math import erfc, sqrt
    print(f"Gaussian tail at that z: {0.5*erfc(z/sqrt(2)):.3f}  "
          f"(observed interference rate "
          f"{np.mean([d['f_gt'] <= 0 for d in dec]):.3f})")
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/lens2_theory_{args.scene}.json"
    json.dump(dict(scene=args.scene, classes=names, N=N.tolist(),
                   normA=np.sqrt(nA2).tolist(), kbar=kbar.tolist(),
                   pred_sd_cross=tot,
                   meas_sd_cross=float(ct.std(ddof=1)),
                   median_self=float(np.median(st)),
                   median_z=float(z)), open(p, "w"), indent=1)
    print(f"wrote {p}")


# ----------------------------------------------------- (c) robustness ------

def cmd_robust(args):
    """Does an orthogonal codebook cost the graceful-degradation headline?

    Mirrors collab_tasks/scripts/degradation_sweep.py exactly (same three
    degradations, same levels, same retention definition = score relative to
    that codebook's OWN undegraded score) but sweeps the codebook instead of
    the backend. Only OUR side moves, so the comparison is codebook vs
    codebook with everything else pinned.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "degsweep", "collab_tasks/scripts/degradation_sweep.py")
    dg = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [spec.origin]
    try:
        spec.loader.exec_module(dg)
    finally:
        sys.argv = argv

    which = args.codebooks or ["random", "rdft", "walsh"]
    scenes = args.scenes or ["room0", "room2", "office4"]
    SW = [("drop", [1.0, 0.5, 0.25, 0.1, 0.05]),
          ("label", [0.0, 0.1, 0.25, 0.5]),
          ("jitter", [0.0, 0.05, 0.1, 0.25, 0.5])]
    res = {}
    for s in scenes:
        data = load_scene(s)
        classes = sorted({o["cls"] for o in data["obs"]})
        res[s] = {}
        for kind, levels in SW:
            res[s][kind] = {cb: [] for cb in which}
            for lv in levels:
                acc = {cb: [] for cb in which}
                for sd in range(args.seeds):
                    rng = np.random.default_rng(1000 * sd + 7)
                    dp = dg.degrade_pts(data["obs"], kind, lv, rng, classes)
                    d2 = dict(data, obs=dp)
                    for cb in which:
                        if not dp:
                            acc[cb].append(0.0)
                            continue
                        bb = build(d2, seeds=SEEDS[sd % len(SEEDS)], cb=cb)
                        acc[cb].append(score(
                            data["gt"],
                            predict(bb["F"], bb["names"], bb["cell"]))["macc"])
                for cb in which:
                    res[s][kind][cb].append(float(np.mean(acc[cb])))
                print(f"  {s} {kind} {lv} "
                      + "  ".join(f"{cb} {res[s][kind][cb][-1]:.3f}"
                                  for cb in which), flush=True)
    print("\nRETENTION at the worst level (score / own undegraded score)")
    print(f"{'scene':<9}{'degradation':<12}"
          + "".join(f"{cb:>12}" for cb in which))
    for s in res:
        for kind in res[s]:
            r = res[s][kind]
            print(f"{s:<9}{kind:<12}"
                  + "".join(f"{r[cb][-1]/max(r[cb][0],1e-9):>11.0%} "
                            for cb in which))
    print("\nmean retention across scenes")
    for kind, _ in SW:
        m = {cb: np.mean([res[s][kind][cb][-1] / max(res[s][kind][cb][0], 1e-9)
                          for s in res]) for cb in which}
        print(f"  {kind:<8}" + "  ".join(f"{cb} {m[cb]:.0%}" for cb in which))
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/lens2_robust.json"
    json.dump(res, open(p, "w"), indent=1)
    print(f"wrote {p}")


def cmd_oracle(args):
    """The ceiling on EVERY interference-removal mechanism, in one number.

    F_g(x) = SELF_g(x) + CROSS_g(x), and CROSS is exactly the part that an
    orthogonal codebook, a bigger HD, a partitioned trace, or a resonator-style
    interference-cancellation pass are all trying to reduce. SELF_g(x) =
    Re<phi(x), A_g> is computable exactly from the same build, so the argmax of
    the SELF fields IS the decode we would get with crosstalk driven to zero --
    the HD -> infinity limit of this memory, with the observation stream, the
    kernel, the cap and the grid all held fixed.

    Whatever that number is, no codebook can beat it. If it is not far above
    the baseline, the codebook lens is closed.
    """
    print("zero-crosstalk ORACLE: argmax of the SELF fields "
          "(Re<phi(x), A_c>), i.e.\nthe HD -> infinity limit. Upper bound for "
          "any interference mechanism.\n")
    print(f"{'scene':<10}{'base mAcc':>10}{'oracle':>9}{'delta':>9}"
          f"{'their NN':>10}{'base F-mIoU':>13}{'oracle':>9}{'delta':>9}")
    print("-" * 80)
    rows = {}
    for s in SCENES:
        data = load_scene(s)
        accs, orac = [], []
        for t in range(args.tuples):
            b = build(data, seeds=SEEDS[t])
            accs.append(score(data["gt"], predict(b["F"], b["names"],
                                                  b["cell"])))
            A, PX, PY = b["A"], b["PX"], b["PY"]
            FS = np.empty_like(b["F"])
            for n in range(len(b["names"])):
                FS[n] = ((PX * A[n][None, :]) @ PY.T).T.reshape(-1).real
            orac.append(score(data["gt"], predict(FS, b["names"], b["cell"])))
        fb = float(np.mean([a["fmiou"] for a in accs]))
        fo = float(np.mean([a["fmiou"] for a in orac]))
        accs = [a["macc"] for a in accs]
        orac = [a["macc"] for a in orac]
        cg = np.load(f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz",
                     allow_pickle=True) \
            if os.path.exists(
                f"student_gpu_package/handoff/{s}_cgfront/cg_labels.npz") \
            else None
        their = float("nan")
        if cg is not None:
            their = _SC_transfer(data, cg)
        rows[s] = dict(baseline=float(np.mean(accs)),
                       oracle=float(np.mean(orac)), theirs=their,
                       fmiou_base=fb, fmiou_oracle=fo)
        print(f"{s:<10}{rows[s]['baseline']:>10.3f}{rows[s]['oracle']:>9.3f}"
              f"{rows[s]['oracle']-rows[s]['baseline']:>+9.3f}{their:>10.3f}"
              f"{fb:>13.3f}{fo:>9.3f}{fo-fb:>+9.3f}", flush=True)
    print("-" * 80)
    mb = np.mean([r["baseline"] for r in rows.values()])
    mo = np.mean([r["oracle"] for r in rows.values()])
    mt = np.nanmean([r["theirs"] for r in rows.values()])
    fb = np.mean([r["fmiou_base"] for r in rows.values()])
    fo = np.mean([r["fmiou_oracle"] for r in rows.values()])
    print(f"{'mean':<10}{mb:>10.3f}{mo:>9.3f}{mo-mb:>+9.3f}{mt:>10.3f}"
          f"{fb:>13.3f}{fo:>9.3f}{fo-fb:>+9.3f}")
    print(f"\nInterference accounts for {mo-mb:+.3f} of the {mt-mb:+.3f} "
          f"memory deficit -> {100*(mo-mb)/max(mt-mb,1e-9):.0f}% of the gap.")
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/lens2_oracle.json"
    json.dump(rows, open(p, "w"), indent=1)
    print(f"wrote {p}")


def _SC_transfer(data, cg):
    """Their own nearest-labelled-point rule, for the same eval points."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "score05b", "student_gpu_package/05_score.py")
    m = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = [spec.origin]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    lab = m.transfer(cg["xyz"], cg["pred_class"].astype(str).astype(object),
                     data["xyz"])
    pred = np.array([v if v is not None else "__none__" for v in lab])
    return float(score(data["gt"], pred)["macc"])


def cmd_churn(args):
    """Which of the three seeds actually moves the labels, and does an
    orthogonal codebook damp the codebook-seed component?

    The batch-1 finding is that changing the (base, codebook, cap) TUPLE moves
    41% of labels. That was never attributed to a single draw. If the codebook
    draw is a large share of it, an exactly-orthogonal codebook has a variance
    argument in its favour even with no mean effect; if it is small, it does
    not.
    """
    data = load_scene(args.scene)
    base = dict(base=SEEDS[0][0], cb=SEEDS[0][1], cap=SEEDS[0][2])
    plans = [("codebook seed only", "random",
              [(base["base"], s, base["cap"]) for s in (7, 8, 9, 10, 15)]),
             ("codebook seed only", "rdft",
              [(base["base"], s, base["cap"]) for s in (7, 8, 9, 10, 15)]),
             ("codebook seed only", "walsh",
              [(base["base"], s, base["cap"]) for s in (7, 8, 9, 10, 15)]),
             ("base (FPE) seed only", "random",
              [(s, base["cb"], base["cap"]) for s in (0, 1, 2, 3, 4)]),
             ("cap seed only", "random",
              [(base["base"], base["cb"], s) for s in (11, 12, 13, 14, 16)]),
             ("full tuple", "random", SEEDS)]
    print(f"{args.scene}: label churn attributable to each draw")
    print(f"{'varied':<22}{'codebook':<9}{'mean pairwise churn':>21}"
          f"{'mAcc mean':>11}{'mAcc sd':>9}")
    print("-" * 72)
    out = {}
    for tag, cb, seedlist in plans:
        preds, accs = [], []
        for sd in seedlist:
            b = build(data, seeds=tuple(sd), cb=cb)
            p = predict(b["F"], b["names"], b["cell"])
            preds.append(p)
            accs.append(score(data["gt"], p)["macc"])
        ch = [float((preds[i] != preds[j]).mean())
              for i in range(len(preds)) for j in range(i + 1, len(preds))]
        print(f"{tag:<22}{cb:<9}{np.mean(ch):>20.1%}"
              f"{np.mean(accs):>11.3f}{np.std(accs, ddof=1):>9.4f}")
        out[f"{tag}|{cb}"] = dict(churn=float(np.mean(ch)),
                                  macc_mean=float(np.mean(accs)),
                                  macc_sd=float(np.std(accs, ddof=1)))
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/lens2_churn_{args.scene}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ch = sub.add_parser("churn"); ch.add_argument("--scene", default="room0")
    o = sub.add_parser("oracle"); o.add_argument("--tuples", type=int, default=3)
    d = sub.add_parser("decompose"); d.add_argument("--scene", default="room0")
    c = sub.add_parser("codebooks")
    c.add_argument("--scene", default="room0")
    c.add_argument("--step", type=int, default=6)
    w = sub.add_parser("swap")
    w.add_argument("--scenes", nargs="*")
    w.add_argument("--codebooks", nargs="*")
    t = sub.add_parser("theory"); t.add_argument("--scene", default="room0")
    r = sub.add_parser("robust")
    r.add_argument("--scenes", nargs="*")
    r.add_argument("--codebooks", nargs="*")
    r.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    {"decompose": cmd_decompose, "codebooks": cmd_codebooks,
     "swap": cmd_swap, "theory": cmd_theory, "churn": cmd_churn,
     "oracle": cmd_oracle, "robust": cmd_robust}[args.cmd](args)


if __name__ == "__main__":
    main()
