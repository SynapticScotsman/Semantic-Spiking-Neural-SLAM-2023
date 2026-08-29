"""E4: how many sides does an object file need, and can you predict it?

FINDINGS.md sec.0 E4 established the qualitative fact -- the object file
interpolates between stored views and does not extrapolate past them.  The
published claim (Poggio & Edelman; Logothetis, Pauls & Poggio, sec.15 leg 3) is
stronger: objects are stored as a set of views, novel views are handled by
interpolating across them, and performance against the *number* of stored views
has a knee.

Two things this measures that sec.0 E4 could not:

1. **Where the knee is, uncontaminated by capacity.**  Sec.16 E0's K sweep ran
   at ``max_harmonic=8``, where sec.16 E2 then showed the object file is
   capacity-bound -- its error rose with K because the bundle was saturating,
   not because coverage was improving.  At ``max_harmonic=4`` that confound is
   gone and the curve measures coverage, which is what E4 is about.

2. **Whether the knee is predictable per object.**  Sec.13 gets a descriptor
   half-width from the crops alone, with no map, no split and no decode.  If
   that number predicts how densely a given object must be orbited, then "how
   many views do I need of this thing?" is answerable before storing anything
   -- which is the practical question an exploring robot actually has.

The list of stored views is carried throughout as the control: it shares the
store and the coverage but none of the code, so anywhere the two curves agree
the limit is coverage rather than the representation.

    python experiments/run_k_curve.py
    python experiments/run_k_curve.py --symmetric-set --seeds 5
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import (CircularSSPSpace, bind, bundle,  # noqa: E402
                               wrap_angle)
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import encode_hog, fit_basis, condition  # noqa: E402
from run_blocked_split import blocked_masks  # noqa: E402
from run_nn_baseline import (pick_store, decode_vsa, decode_nearest,  # noqa: E402
                             hier_bootstrap, DROP)
from run_frontend_diagnostics import diagnose  # noqa: E402

KS = (2, 3, 4, 6, 8, 12, 18, 24, 36)
ALIAS_THRESH = 0.75


def run_K(Z, obj, az, kept, held, names, K, d, kmax, seeds):
    """Per-crop errors for both decoders, keyed by object, plus the store gap."""
    mu, V, lam = fit_basis(Z[kept])
    n_obj = len(names)
    vsa = {o: [] for o in range(n_obj)}
    lst = {o: [] for o in range(n_obj)}
    gaps = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
        keys = condition(Z, mu, V, lam, W, drop=DROP)
        vs = CircularSSPSpace(1, ssp_dim=d, max_harmonic=kmax,
                              rng=np.random.default_rng(seed + 1000))
        for o in range(n_obj):
            sel = np.where((obj == o) & kept)[0]
            pick = sel[pick_store(az[sel], K)]
            book = bundle(np.stack([bind(k, vs.encode([a]))
                                    for k, a in zip(keys[pick], az[pick])]))
            m = held & (obj == o)
            q, t = keys[m], az[m]
            a, _ = decode_vsa(book, q, vs)
            vsa[o].append(np.abs(np.rad2deg(wrap_angle(a - t))))
            a, _ = decode_nearest(keys[pick], az[pick], q)
            lst[o].append(np.abs(np.rad2deg(wrap_angle(a - t))))
            if seed == 0:
                gaps.append(np.median([np.rad2deg(
                    np.abs(wrap_angle(az[pick] - x)).min()) for x in t]))
    return ({o: np.concatenate(v) for o, v in vsa.items()},
            {o: np.concatenate(v) for o, v in lst.items()},
            float(np.median(gaps)))


def knee(ks, errs, tol=1.10):
    """Smallest K whose error is within ``tol`` of the best achieved.

    Deliberately not a fitted breakpoint: with nine K values and a noisy
    curve, a fit would invent precision the data cannot support.
    """
    best = min(errs)
    for k, e in zip(ks, errs):
        if e <= tol * best:
            return k
    return ks[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-views", type=int, default=72)
    ap.add_argument("--gap-deg", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ssp-dim", type=int, default=2401)
    ap.add_argument("--kmax", type=int, default=4,
                    help="4 is sec.16 E2's measured optimum; 8 is the repo "
                         "default and is capacity-bound")
    ap.add_argument("--symmetric-set", action="store_true")
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(
        n_views=args.n_views,
        extra="symmetric" if args.symmetric_set else None)
    Z = encode_hog(imgs)
    nv, n_obj = args.n_views, len(names)
    vi = np.concatenate([np.arange(nv)] * n_obj)
    kept, held = blocked_masks(vi, nv, args.gap_deg)
    period = 2 * max(int(round(args.gap_deg / (360.0 / nv))), 1)
    obj_h, arc_h = obj[held], (vi // period)[held]

    print(f"{len(Z)} crops, {Z.shape[1]}-D HOG, {n_obj} objects x {nv} azimuths")
    print(f"blocked split, {args.gap_deg:.0f} deg arcs; d={args.ssp_dim}, "
          f"max_harmonic={args.kmax}; {args.seeds} seeds; chance = 90 deg\n")

    # sec.13's diagnostics, from the crops alone, for the prediction test
    mu, V, lam = fit_basis(Z[kept])
    rng = np.random.default_rng(0)
    W = rng.standard_normal((V.shape[0], args.ssp_dim)) / np.sqrt(V.shape[0])
    keys0 = condition(Z, mu, V, lam, W, drop=DROP)
    vs0 = CircularSSPSpace(1, ssp_dim=args.ssp_dim, max_harmonic=args.kmax,
                           rng=np.random.default_rng(1000))
    diag = diagnose(keys0, obj, az, names, nv, np.rad2deg(vs0.lobe_width()))

    results = {}
    print("[A] the K curve, pooled over objects")
    hdr = (f"  {'K':>4s} {'spacing':>8s} {'median gap':>11s} "
           f"{'object file':>12s} {'<15deg':>7s} {'list':>8s} {'<15deg':>7s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for K in KS:
        v, l, gap = run_K(Z, obj, az, kept, held, names, K, args.ssp_dim,
                          args.kmax, args.seeds)
        results[K] = (v, l, gap)
        ev = np.concatenate([v[o] for o in range(n_obj)])
        el = np.concatenate([l[o] for o in range(n_obj)])
        print(f"  {K:4d} {360.0 / K:7.0f}d {gap:10.0f}d "
              f"{np.median(ev):11.1f}d {(ev < 15).mean():7.2f} "
              f"{np.median(el):7.1f}d {(el < 15).mean():7.2f}")
    print("  'spacing' is 360/K; 'median gap' is how far a held-out view "
          "actually sits\n  from the nearest stored one, which the blocked "
          "arcs make larger.")

    ks = list(KS)
    pooled_v = [np.median(np.concatenate([results[K][0][o]
                                          for o in range(n_obj)])) for K in ks]
    pooled_l = [np.median(np.concatenate([results[K][1][o]
                                          for o in range(n_obj)])) for K in ks]
    print(f"\n  knee (smallest K within 10% of best): object file "
          f"K={knee(ks, pooled_v)}, list K={knee(ks, pooled_l)}")
    print(f"  view kernel lobe {np.rad2deg(vs0.lobe_width()):.0f} deg; "
          f"median descriptor half-width "
          f"{np.median([r['half_width'] for r in diag]):.0f} deg")

    print("\n[B] per object: the knee, against what sec.13 predicts from the "
          "crops alone")
    print(f"  {'object':>9s} {'half-width':>11s} {'alias peak':>11s} "
          f"{'knee K':>7s} {'spacing':>8s} {'best err':>9s} {'list knee':>10s}")
    print("  " + "-" * 70)
    rows = []
    for o, nm in enumerate(names):
        ev = [np.median(results[K][0][o]) for K in ks]
        el = [np.median(results[K][1][o]) for K in ks]
        kv, kl = knee(ks, ev), knee(ks, el)
        rows.append((nm, diag[o]["half_width"], diag[o]["alias_peak"], kv,
                     min(ev)))
        print(f"  {nm:>9s} {diag[o]['half_width']:10.0f}d "
              f"{diag[o]['alias_peak']:11.3f} {kv:7d} {360.0 / kv:7.0f}d "
              f"{min(ev):8.1f}d {kl:10d}")

    hw = np.array([r[1] for r in rows])
    ap_ = np.array([r[2] for r in rows])
    sp = np.array([360.0 / r[3] for r in rows])
    ok = ap_ < ALIAS_THRESH
    print(f"\n  Does the sec.13 half-width predict the required spacing?")
    for label, mask in (("all objects", np.ones(len(rows), bool)),
                        (f"alias peak < {ALIAS_THRESH}", ok)):
        if mask.sum() < 3:
            print(f"    {label}: {mask.sum()} objects, too few to correlate")
            continue
        if np.std(sp[mask]) < 1e-9:
            print(f"    {label} (n={mask.sum()}): every knee is identical "
                  f"(K={rows[int(np.argmax(mask))][3]}), so there is nothing "
                  f"to correlate")
            continue
        r = float(np.corrcoef(hw[mask], sp[mask])[0, 1])
        print(f"    {label} (n={mask.sum()}): Pearson r = {r:+.2f} between "
              f"half-width and 360/knee")
    print("  an aliased object never reaches a useful error at any K, so its "
          "knee is\n  meaningless -- which is why the second row excludes "
          "them.")
    print("  if every knee lands on the same K, look at the 'median gap' "
          "column in [A]:\n  past that point extra stored views no longer "
          "sit any closer to a held-out\n  one, because the blocked arcs "
          "fix how far apart they can be. K is then the\n  wrong axis, and "
          "[D] sweeps the right one.")

    print(f"\n[C] does the object file need more views than the list? "
          f"({args.n_boot} draws)")
    print(f"  {'K':>4s} {'vsa - list':>11s} {'95% CI':>19s}")
    print("  " + "-" * 38)
    for K in ks:
        v, l, _ = results[K]
        # rebuild per-seed arrays in held-out order for the bootstrap
        order = np.concatenate([np.where(obj_h == o)[0] for o in range(n_obj)])
        inv = np.argsort(order)
        n_per = len(v[0]) // args.seeds
        va = [np.concatenate([v[o][i * n_per:(i + 1) * n_per]
                              for o in range(n_obj)])[inv]
              for i in range(args.seeds)]
        la = [np.concatenate([l[o][i * n_per:(i + 1) * n_per]
                              for o in range(n_obj)])[inv]
              for i in range(args.seeds)]
        lo, hi = hier_bootstrap(va, obj_h, arc_h, n_obj, n_boot=args.n_boot,
                                per_seed_b=la)
        m = (np.median(np.concatenate(va)) - np.median(np.concatenate(la)))
        tag = "" if lo <= 0 <= hi else ("  <-- worse" if lo > 0 else
                                        "  <-- better")
        print(f"  {K:4d} {m:+10.1f}d [{lo:+6.1f},{hi:+6.1f}]{tag}")
    print("  where the interval spans zero, coverage is the limit and the "
          "code is not.")

    gap_sweep(Z, obj, az, vi, names, diag, args)


# ---------------------------------------------------------------------------
# [D] The question K cannot answer.
#
# Adding stored views stops helping once every held-out view already has a
# neighbour, and with fixed held-out arcs that happens at K=12 for every
# object.  What actually differs between objects is how far apart their sides
# can be before the file stops interpolating -- so sweep the ARC WIDTH, which
# is the quantity a robot controls when it decides how much of an orbit to
# walk, and ask whether sec.13's half-width predicts it.
# ---------------------------------------------------------------------------

def gap_sweep(Z, obj, az, vi, names, diag, args):
    nv, n_obj = args.n_views, len(names)
    gaps = (10, 20, 30, 45, 60, 90)
    print(f"\n[D] per object: how wide a hole can the file fill in? "
          f"(median error, degrees)")
    print(f"  {'object':>9s} {'half-width':>11s} "
          + " ".join(f"{str(g) + 'd':>7s}" for g in gaps) + f" {'tol':>6s}")
    print("  " + "-" * (22 + 8 * len(gaps) + 7))
    table = {}
    for g in gaps:
        kept, held = blocked_masks(vi, nv, float(g))
        K = max(int(kept.sum() // n_obj) // 3, 2)
        v, _, _ = run_K(Z, obj, az, kept, held, names, K, args.ssp_dim,
                        args.kmax, args.seeds)
        table[g] = {o: float(np.median(v[o])) for o in range(n_obj)}

    hw, tol = [], []
    for o, nm in enumerate(names):
        row = [table[g][o] for g in gaps]
        good = [g for g, e in zip(gaps, row) if e < 15.0]
        t = max(good) if good else 0
        print(f"  {nm:>9s} {diag[o]['half_width']:10.0f}d "
              + " ".join(f"{e:6.1f}d" for e in row)
              + (f" {t:5d}d" if t else f" {'--':>6s}"))
        if diag[o]["alias_peak"] < ALIAS_THRESH and t:
            hw.append(diag[o]["half_width"])
            tol.append(t)
    print("  'tol' is the widest arc the file still fills to under 15 deg. "
          "'--' means\n  it never does, which is what an aliased object "
          "looks like here.")
    if len(hw) >= 3 and np.std(tol) > 1e-9:
        r = float(np.corrcoef(hw, tol)[0, 1])
        print(f"\n  sec.13 half-width vs tolerable arc, over the {len(hw)} "
              f"non-aliased objects: r = {r:+.2f}")
        print("  positive means an object whose appearance changes slowly can "
              "be orbited\n  more coarsely -- which is the claim, and it is "
              "answerable from the crops\n  alone, before any map exists.")
    else:
        print(f"\n  {len(hw)} usable objects with "
              f"{len(set(tol))} distinct tolerances -- not enough spread to "
              f"correlate.")


if __name__ == "__main__":
    main()
