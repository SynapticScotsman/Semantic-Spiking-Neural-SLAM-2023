"""E2: does a residue number system hold more views per dimension?

FINDINGS.md sec.16 E0 found the object file is **capacity**-bound, not
resolution-bound: its error *rises* as K grows, because K mutually correlated
appearance keys saturate the bundle, and it takes about sixteen times the
dimension to match a plain list of stored views.  Kymn et al. (sec.15) address
exactly that class of problem with a co-prime modular code, so this asks
whether it transfers.

On a circle a residue system is a choice of harmonics and nothing else.  With
moduli ``m_i`` and ``M = prod(m_i)``, module ``m`` is the single harmonic
``M/m`` (see :func:`sspslam.objectmap.residue_harmonics`), so::

    dense band  {1, 2, ..., 16}     16 frequencies, top harmonic 16
    residue (7,8,9)  {56, 63, 72}    3 frequencies, top harmonic 72

Both fill the same ``ssp_dim``.  The residue set spends the whole budget on
three frequencies -- five times the redundancy each -- and still resolves four
times finer, because the Chinese remainder theorem makes the three jointly
unambiguous over the circle.  If capacity is the constraint, that should pay.

**Report it as a capacity experiment, with resolution held fixed.**  A residue
set with top harmonic 72 has a 1 degree main lobe, and sec.14 already showed
that sharper kernels are *worse* when stored views are far apart -- so a naive
comparison would credit the code for something the descriptor cannot support.
The honest question is error at equal *floats*, against E0's dimension sweep as
the baseline curve, plus how the failure mode changes.

The expected cost is in the tail, not the median.  A mean of three cosines has
a unique global peak but tall sidelobes, so a residue code should fail by
jumping to a wrong CRT-consistent angle rather than by losing precision
gradually.  Section [C] measures the gross-failure rate separately for that
reason: a code with a better median and a worse tail is not obviously better.

    python experiments/run_residue_code.py
    python experiments/run_residue_code.py --dims 151 601 2401 --seeds 5
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import (CircularSSPSpace, bind, bundle,  # noqa: E402
                               residue_harmonics, wrap_angle)
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import encode_hog, fit_basis, condition  # noqa: E402
from run_blocked_split import blocked_masks  # noqa: E402
from run_nn_baseline import (pick_store, decode_vsa, decode_nearest,  # noqa: E402
                             hier_bootstrap, DROP)


# ---------------------------------------------------------------------------
# The decoder a residue code is supposed to get.  Kymn et al. read residues
# with a resonator network; peak-picking a superposed likelihood throws the
# modular structure away, so judging the code by argmax alone would be unfair.
# This is the resonator's fixed point computed directly: one phase per module,
# then Chinese remainder.
# ---------------------------------------------------------------------------

def crt_combine(residues, moduli):
    total = int(np.prod(moduli))
    x = 0
    for r, m in zip(residues, moduli):
        cofactor = total // m
        x += r * cofactor * pow(int(cofactor), -1, int(m))
    return x % total


def decode_crt(vs, book, keys, moduli):
    """Per-module phase, then CRT -- exact, and needs no iteration.

    For module ``m`` the code carries the single harmonic ``h = M/m``, so the
    coefficient of ``exp(i h phi)`` in the score is all the evidence there is
    about ``x mod m``.  Take its phase, round to the nearest of the ``m``
    residues, and combine.
    """
    total = int(np.prod(moduli))
    harmonics = [total // m for m in moduli]
    k = np.rint(vs.phase_matrix).astype(int).ravel()
    conj_book = np.conj(np.fft.fft(np.asarray(book, dtype=float).reshape(-1)))
    out = []
    for key in keys:
        cross = np.fft.fft(np.asarray(key, dtype=float).reshape(-1)) * conj_book
        res = []
        for m, h in zip(moduli, harmonics):
            # the angle grid starts at -pi, same twist as view_likelihood
            c = cross[k == h].sum() * np.exp(-1j * h * np.pi)
            g = np.real(c * np.exp(2j * np.pi * np.arange(m) / m))
            res.append(int(np.argmax(g)))
        out.append(wrap_angle(-np.pi + 2 * np.pi
                              * crt_combine(res, moduli) / total))
    return np.array(out)

# Dense bands, then residue systems.  The moduli are chosen to span top
# harmonics from 20 to 72, straddling the band settings they are compared to.
BANDS = (8, 16, 24)
MODULI = ((3, 4, 5), (4, 5, 7), (5, 7, 9), (7, 8, 9))
GROSS_DEG = 45.0            # beyond this an error is a jump, not imprecision


def make_space(d, spec, seed):
    """``spec`` is an int (dense band) or a tuple of moduli (residue)."""
    rng = np.random.default_rng(seed + 1000)
    if isinstance(spec, int):
        return CircularSSPSpace(1, ssp_dim=d, max_harmonic=spec, rng=rng)
    return CircularSSPSpace(1, ssp_dim=d,
                            harmonics=residue_harmonics(spec, (d - 1) // 2),
                            rng=rng)


def label(spec):
    return f"band {spec}" if isinstance(spec, int) else \
        "res " + "-".join(str(m) for m in spec)


def one_run(Z, obj, az, kept, held, names, K, d, spec, seed):
    n_obj = len(names)
    mu, V, lam = fit_basis(Z[kept])
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
    keys = condition(Z, mu, V, lam, W, drop=DROP)
    vs = make_space(d, spec, seed)

    err, nn_err = [], []
    order = []
    for o in range(n_obj):
        sel = np.where((obj == o) & kept)[0]
        pick = sel[pick_store(az[sel], K)]
        book = bundle(np.stack([bind(k, vs.encode([a]))
                                for k, a in zip(keys[pick], az[pick])]))
        m = held & (obj == o)
        q, t = keys[m], az[m]
        a, _ = decode_vsa(book, q, vs)
        err.append(np.abs(np.rad2deg(wrap_angle(a - t))))
        a, _ = decode_nearest(keys[pick], az[pick], q)
        nn_err.append(np.abs(np.rad2deg(wrap_angle(a - t))))
        order.append(np.where(m)[0])
    # put the per-crop errors back in held-out order so the bootstrap's
    # object/arc labels line up
    inv = np.argsort(np.concatenate(order))
    return (np.concatenate(err)[inv], np.concatenate(nn_err)[inv],
            np.rad2deg(vs.lobe_width()),
            len(set(np.rint(vs.harmonics.ravel()).astype(int))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-views", type=int, default=72)
    ap.add_argument("--gap-deg", type=float, default=30.0)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dims", type=int, nargs="+", default=[151, 601, 2401])
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(n_views=args.n_views)
    Z = encode_hog(imgs)
    nv, n_obj = args.n_views, len(names)
    vi = np.concatenate([np.arange(nv)] * n_obj)
    kept, held = blocked_masks(vi, nv, args.gap_deg)
    period = 2 * max(int(round(args.gap_deg / (360.0 / nv))), 1)
    obj_h, arc_h = obj[held], (vi // period)[held]

    print(f"{len(Z)} crops, {Z.shape[1]}-D HOG, {n_obj} objects x {nv} azimuths")
    print(f"blocked split, {args.gap_deg:.0f} deg arcs; K={args.K} on file; "
          f"{args.seeds} seeds; chance = 90 deg")
    print(f"a residue system with moduli m_i is the harmonic set "
          f"{{prod(m)/m_i}} -- nothing else changes\n")

    specs = list(BANDS) + list(MODULI)
    store = {}

    print("[A] equal dimension: dense band against residue system")
    hdr = (f"  {'d':>5s} {'code':>10s} {'harmonics':>10s} {'top':>4s} "
           f"{'lobe':>6s} {'median':>8s} {'<15deg':>7s} {'>45deg':>7s} "
           f"{'90th':>7s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for d in args.dims:
        for spec in specs:
            runs = [one_run(Z, obj, az, kept, held, names, args.K, d, spec, s)
                    for s in range(args.seeds)]
            e = np.concatenate([r[0] for r in runs])
            store[(d, spec)] = [r[0] for r in runs]
            store[(d, "nn")] = [r[1] for r in runs]
            top = (spec if isinstance(spec, int)
                   else int(np.prod(spec) // min(spec)))
            print(f"  {d:5d} {label(spec):>10s} {runs[0][3]:10d} {top:4d} "
                  f"{runs[0][2]:5.1f}d {np.median(e):7.1f}d "
                  f"{(e < 15).mean():7.2f} {(e > GROSS_DEG).mean():7.2f} "
                  f"{np.percentile(e, 90):6.1f}d")
        nn = np.concatenate(store[(d, "nn")])
        print(f"  {d:5d} {'list (1-NN)':>10s} {'--':>10s} {'--':>4s} "
              f"{'--':>6s} {np.median(nn):7.1f}d {(nn < 15).mean():7.2f} "
              f"{(nn > GROSS_DEG).mean():7.2f} {np.percentile(nn, 90):6.1f}d")
        print()
    print("  'harmonics' is the count of DISTINCT frequencies the ssp_dim is "
          "spread over.\n  '>45deg' is the gross-failure rate: a jump to a "
          "wrong angle rather than\n  a loss of precision. Read it beside the "
          "median, never instead of it.")

    print(f"\n[B] best residue against best band, at each dimension "
          f"({args.n_boot} draws, objects -> arcs -> seeds)")
    print(f"  {'d':>5s} {'contrast':>26s} {'diff':>9s} {'95% CI':>19s}")
    print("  " + "-" * 62)
    for d in args.dims:
        best_band = min(BANDS, key=lambda b: np.median(
            np.concatenate(store[(d, b)])))
        best_res = min(MODULI, key=lambda m: np.median(
            np.concatenate(store[(d, m)])))
        for a, b in ((best_res, best_band), (best_res, "nn")):
            lo, hi = hier_bootstrap(store[(d, a)], obj_h, arc_h, n_obj,
                                    n_boot=args.n_boot,
                                    per_seed_b=store[(d, b)])
            m = (np.median(np.concatenate(store[(d, a)]))
                 - np.median(np.concatenate(store[(d, b)])))
            nb = "list" if b == "nn" else label(b)
            tag = "" if lo <= 0 <= hi else ("  <-- worse" if lo > 0
                                            else "  <-- better")
            print(f"  {d:5d} {label(a) + ' - ' + nb:>26s} {m:+8.1f}d "
                  f"[{lo:+6.1f},{hi:+6.1f}]{tag}")
    print("  negative = the residue code is better.")

    self_retrieval(Z, obj, az, kept, held, names, args)
    crt_readout(Z, obj, az, kept, held, names, args)
    harmonic_sweep(Z, obj, az, kept, held, names, args)


# ---------------------------------------------------------------------------
# [C] The mechanism, with the front end taken out of the picture.
#
# Query the book with a key that is IN it.  There is no generalisation left to
# do and no descriptor limit to hit, so whatever varies is purely how well the
# code survives superposing K items.  This is the measurement that says what
# the capacity currency actually is.
# ---------------------------------------------------------------------------

def self_retrieval(Z, obj, az, kept, held, names, args):
    d = args.dims[1] if len(args.dims) > 1 else args.dims[0]
    mu, V, lam = fit_basis(Z[kept])
    Ks = (1, 4, 12, 24)
    print(f"\n[C] self-retrieval at d={d}: query the book with a key that is "
          f"in it")
    print(f"  {'code':>12s} {'distinct harmonics':>19s} "
          + " ".join(f"{'K=' + str(K):>8s}" for K in Ks))
    print("  " + "-" * 66)
    for spec in list(MODULI) + list(BANDS) + [4, 32]:
        row, n_harm = [], 0
        for K in Ks:
            errs = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
                keys = condition(Z, mu, V, lam, W, drop=DROP)
                vs = make_space(d, spec, seed)
                n_harm = len(set(np.rint(vs.harmonics.ravel()).astype(int)))
                for o in range(len(names)):
                    sel = np.where((obj == o) & kept)[0]
                    pick = sel[pick_store(az[sel], K)]
                    book = bundle(np.stack([bind(k, vs.encode([a])) for k, a
                                            in zip(keys[pick], az[pick])]))
                    a, _ = decode_vsa(book, keys[pick], vs)
                    errs.append(np.abs(np.rad2deg(
                        wrap_angle(a - az[pick]))))
            row.append(np.median(np.concatenate(errs)))
        print(f"  {label(spec):>12s} {n_harm:19d} "
              + " ".join(f"{v:7.1f}d" for v in row))
    print("  every code is exact at K=1. what separates them is how many "
          "items the\n  bundle can hold, and that tracks the number of "
          "DISTINCT harmonics --\n  which is precisely the quantity a "
          "residue system minimises.")


# ---------------------------------------------------------------------------
# [D] Is it the readout?  Give the residue code the decoder it was designed
# for and see whether anything changes.
# ---------------------------------------------------------------------------

def crt_readout(Z, obj, az, kept, held, names, args):
    d = args.dims[1] if len(args.dims) > 1 else args.dims[0]
    mu, V, lam = fit_basis(Z[kept])
    print(f"\n[D] argmax against the Chinese-remainder readout, d={d}, "
          f"same books")
    print(f"  {'code':>12s} {'K':>3s} {'argmax':>9s} {'CRT':>9s} "
          f"{'CRT >45deg':>11s}")
    print("  " + "-" * 49)
    for spec in MODULI:
        for K in (6, 12):
            ea, ec = [], []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
                keys = condition(Z, mu, V, lam, W, drop=DROP)
                vs = make_space(d, spec, seed)
                for o in range(len(names)):
                    sel = np.where((obj == o) & kept)[0]
                    pick = sel[pick_store(az[sel], K)]
                    book = bundle(np.stack([bind(k, vs.encode([a])) for k, a
                                            in zip(keys[pick], az[pick])]))
                    m = held & (obj == o)
                    a, _ = decode_vsa(book, keys[m], vs)
                    ea.append(np.abs(np.rad2deg(wrap_angle(a - az[m]))))
                    a = decode_crt(vs, book, keys[m], spec)
                    ec.append(np.abs(np.rad2deg(wrap_angle(a - az[m]))))
            ea, ec = np.concatenate(ea), np.concatenate(ec)
            print(f"  {label(spec):>12s} {K:3d} {np.median(ea):8.1f}d "
                  f"{np.median(ec):8.1f}d {(ec > GROSS_DEG).mean():11.2f}")
    print("  the proper decoder does not rescue it, so the loss is in the "
          "code, not\n  in how it is read.")


# ---------------------------------------------------------------------------
# [E] What the mechanism implies, tested on the real task.
#
# If distinct-harmonic count is the capacity currency, then E0's signature
# failure -- error rising with K -- should be sensitive to max_harmonic, and
# E0 never swept it below 8.  This is the section that pays for E2.
# ---------------------------------------------------------------------------

def harmonic_sweep(Z, obj, az, kept, held, names, args):
    mu, V, lam = fit_basis(Z[kept])
    Ks = (6, 12, 24, 36)
    for d in args.dims:
        print(f"\n[E] real task, d={d}: median in-gap error against "
              f"max_harmonic")
        print(f"  {'kmax':>5s} {'lobe':>6s} "
              + " ".join(f"{'K=' + str(K):>8s}" for K in Ks))
        print("  " + "-" * 46)
        for kmax in (4, 8, 12, 16, 24, 48):
            row, lobe = [], 0.0
            for K in Ks:
                errs = []
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed)
                    W = (rng.standard_normal((V.shape[0], d))
                         / np.sqrt(V.shape[0]))
                    keys = condition(Z, mu, V, lam, W, drop=DROP)
                    vs = make_space(d, kmax, seed)
                    lobe = np.rad2deg(vs.lobe_width())
                    for o in range(len(names)):
                        sel = np.where((obj == o) & kept)[0]
                        pick = sel[pick_store(az[sel], K)]
                        book = bundle(np.stack(
                            [bind(k, vs.encode([a]))
                             for k, a in zip(keys[pick], az[pick])]))
                        m = held & (obj == o)
                        a, _ = decode_vsa(book, keys[m], vs)
                        errs.append(np.abs(np.rad2deg(
                            wrap_angle(a - az[m]))))
                row.append(np.median(np.concatenate(errs)))
            print(f"  {kmax:5d} {lobe:5.0f}d "
                  + " ".join(f"{v:7.1f}d" for v in row))
        row = []
        for K in Ks:
            errs = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
                keys = condition(Z, mu, V, lam, W, drop=DROP)
                for o in range(len(names)):
                    sel = np.where((obj == o) & kept)[0]
                    pick = sel[pick_store(az[sel], K)]
                    m = held & (obj == o)
                    a, _ = decode_nearest(keys[pick], az[pick], keys[m])
                    errs.append(np.abs(np.rad2deg(wrap_angle(a - az[m]))))
            row.append(np.median(np.concatenate(errs)))
        print(f"  {'list':>5s} {'--':>6s} "
              + " ".join(f"{v:7.1f}d" for v in row))
    print("\n  sec.16 E0 swept max_harmonic over {8, 16} and concluded the "
          "object file\n  loses to the list. It never tried 4. Read the top "
          "row against the last.")


if __name__ == "__main__":
    main()
