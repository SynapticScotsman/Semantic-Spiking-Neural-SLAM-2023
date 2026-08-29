"""E0: does unbinding actually beat a list of stored views?

The honest control this project was missing.  Neubert & Schubert do
viewpoint-*invariant* recognition by bundling descriptors, with no unbinding
anywhere; nothing here has yet shown that the object file beats keeping the K
conditioned descriptors in a list and taking the nearest one.  Until it does,
FINDINGS.md sec.15's gap claim is unearned.

Three decoders, identical inputs -- same crops, same conditioning, same blocked
arcs, same K views on file:

  nearest        cosine to each stored key, report that key's angle.  Cannot
                 do better than half the store spacing (360/K/2 degrees); it
                 quantises, it does not interpolate
  kernel         circular mean of stored angles, softmax-weighted by cosine.
                 Interpolates.  Temperature fitted leave-one-out on the STORE,
                 never on held-out data
  vsa            sec.4's one-FFT likelihood over the object file
  vsa-scene      the same, but all six object files superposed into ONE
                 d-vector with bind(ID_o, book_o), unbound at query time.
                 No list-based method has an analogue at that budget

Section [D] then sweeps the SSP dimension, because [A] compares at the repo's
default d=151 and that turns out to be the binding constraint rather than
anything about unbinding: bundling K correlated appearance keys saturates a
151-D vector, so the object file's error *grows* with K while the list's
falls.  The sweep is what says whether that is a capacity limit or a dead end.

Metrics are in-gap pose error, identification, and floats of map.  The
comparison that matters is at equal K: if `nearest` ties `vsa` there, the
object file is a compression scheme rather than a better estimator, and the
write-up has to say so.

Sampling follows CLAUDE.md: blocked arcs, several seeds, and a hierarchical
bootstrap over objects -> arcs -> seeds.  Frames are not samples; no statistic
here is computed as if they were.

    python experiments/run_nn_baseline.py
    python experiments/run_nn_baseline.py --gap-deg 45 --seeds 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import (AtomVocab, CircularSSPSpace, bind,  # noqa: E402
                               bundle, unbind, wrap_angle)
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import encode_hog, fit_basis, condition  # noqa: E402
from run_blocked_split import blocked_masks  # noqa: E402

SSP_DIM, MAX_HARMONIC, DROP = 151, 8, 2
BETAS = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])


# ---------------------------------------------------------------------------
# The store: K views per object, as evenly spaced round the circle as the
# kept arcs allow.  Every decoder sees exactly this and nothing else.
# ---------------------------------------------------------------------------

def pick_store(az_kept, K):
    """Indices into ``az_kept`` closest to K evenly spaced target angles."""
    if K >= len(az_kept):
        return np.arange(len(az_kept))
    targets = np.linspace(-np.pi, np.pi, K, endpoint=False)
    chosen = []
    for t in targets:
        order = np.argsort(np.abs(wrap_angle(az_kept - t)))
        for j in order:
            if j not in chosen:
                chosen.append(int(j))
                break
    return np.array(sorted(chosen))


def circular_mean_weighted(angles, w):
    z = (w * np.exp(1j * angles)).sum(-1)
    return np.angle(z), np.abs(z) / np.maximum(w.sum(-1), 1e-12)


def softmax_rows(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# Decoders.  Each returns (estimated angle per query, per-object score) so the
# identity decision is made the same way for all of them: best score wins.
# ---------------------------------------------------------------------------

def decode_nearest(store_keys, store_az, q):
    sim = q @ store_keys.T
    return store_az[sim.argmax(1)], sim.max(1)


def decode_kernel(store_keys, store_az, q, beta):
    sim = q @ store_keys.T
    w = softmax_rows(beta * sim)
    ang, _ = circular_mean_weighted(store_az[None, :], w)
    return ang, sim.max(1)


def fit_beta_loo(store_keys, store_az):
    """Temperature by leave-one-out on the store.  Uses no held-out data."""
    best, best_err = BETAS[0], np.inf
    for beta in BETAS:
        errs = []
        for ks, azs in zip(store_keys, store_az):
            sim = ks @ ks.T
            np.fill_diagonal(sim, -np.inf)
            w = softmax_rows(beta * sim)
            ang, _ = circular_mean_weighted(azs[None, :], w)
            errs.append(np.abs(np.rad2deg(wrap_angle(ang - azs))))
        e = float(np.median(np.concatenate(errs)))
        if e < best_err:
            best, best_err = beta, e
    return float(best), best_err


def decode_vsa(book, q, vs, n_grid=720):
    grid, field = None, []
    for k in q:
        g, f = vs.view_likelihood(book, k, n_per_dim=n_grid)
        grid = g
        field.append(f)
    field = np.stack(field)
    return grid[field.argmax(1)], field.max(1)


# ---------------------------------------------------------------------------
# One (seed, K) run: build every map, decode every held-out crop
# ---------------------------------------------------------------------------

def one_run(Z, obj, az, kept, held, names, K, seed, d=SSP_DIM,
            kmax=MAX_HARMONIC):
    n_obj = len(names)
    mu, V, lam = fit_basis(Z[kept])                  # kept views only
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
    keys = condition(Z, mu, V, lam, W, drop=DROP)
    vs = CircularSSPSpace(1, ssp_dim=d, max_harmonic=kmax,
                          rng=np.random.default_rng(seed + 1000))
    vocab = AtomVocab(d, seed=seed + 2000)

    store_keys, store_az, books = [], [], []
    for o in range(n_obj):
        sel = np.where((obj == o) & kept)[0]
        pick = sel[pick_store(az[sel], K)]
        store_keys.append(keys[pick])
        store_az.append(az[pick])
        books.append(bundle(np.stack([bind(k, vs.encode([a]))
                                      for k, a in zip(keys[pick], az[pick])])))
    books = np.stack(books)
    scene = bundle(np.stack([bind(vocab.mint(names[o]), books[o])
                             for o in range(n_obj)]))

    beta, _ = fit_beta_loo(store_keys, store_az)

    q = keys[held]
    true = az[held]
    truth_obj = obj[held]
    out = {}
    for name in ("nearest", "kernel", "vsa", "vsa-scene"):
        ang = np.zeros((len(q), n_obj))
        score = np.zeros((len(q), n_obj))
        for o in range(n_obj):
            if name == "nearest":
                a, s = decode_nearest(store_keys[o], store_az[o], q)
            elif name == "kernel":
                a, s = decode_kernel(store_keys[o], store_az[o], q, beta)
            elif name == "vsa":
                a, s = decode_vsa(books[o], q, vs)
            else:
                a, s = decode_vsa(unbind(scene, vocab[names[o]]), q, vs)
            ang[:, o], score[:, o] = a, s
        err = np.abs(np.rad2deg(wrap_angle(ang[np.arange(len(q)), truth_obj]
                                           - true)))
        out[name] = (err, score.argmax(1) == truth_obj)
    return out, beta, keys.shape[1]


# ---------------------------------------------------------------------------
# Hierarchical bootstrap: objects -> arcs -> seeds.  Never frames.
# ---------------------------------------------------------------------------

def hier_bootstrap(per_seed, obj_h, arc_h, n_obj, n_boot=2000, seed=0,
                   stat=np.median, per_seed_b=None, obj_ids=None):
    """95% CI on a summary of a per-crop quantity, resampling only the levels
    that are actually exchangeable: objects, then contiguous arcs, then seeds.
    Frames are never resampled -- see CLAUDE.md and sec.0 E8.

    ``stat`` is ``np.median`` for errors in degrees, ``np.mean`` for a per-crop
    hit indicator.  ``obj_ids`` restricts the object level to a subset (used
    when stratifying), and must match the labels in ``obj_h``.

    Pass ``per_seed_b`` for a paired contrast.  The statistic is then
    ``stat(a) - stat(b)`` on the *same* resampled crops, NOT ``stat(a - b)``.
    The distinction matters: two pipelines that share a stage agree exactly on
    most crops, so the median of their per-crop difference is pinned at zero
    however far apart their medians are.
    """
    rng = np.random.default_rng(seed)
    n_seeds = len(per_seed)
    ids = np.arange(n_obj) if obj_ids is None else np.asarray(obj_ids)
    idx = {o: {a: np.where((obj_h == o) & (arc_h == a))[0]
               for a in np.unique(arc_h[obj_h == o])} for o in ids}
    arcs = {o: np.array(sorted(idx[o])) for o in ids}
    stats = []
    for _ in range(n_boot):
        si = rng.integers(n_seeds)
        take = []
        for o in ids[rng.integers(0, len(ids), len(ids))]:
            aa = arcs[o]
            for a in aa[rng.integers(0, len(aa), len(aa))]:
                take.append(idx[o][a])
        take = np.concatenate(take)
        v = stat(per_seed[si][take])
        if per_seed_b is not None:
            v = v - stat(per_seed_b[si][take])
        stats.append(v)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-views", type=int, default=72)
    ap.add_argument("--gap-deg", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[151, 301, 601, 1201, 2401, 4801])
    ap.add_argument("--sweep-k", type=int, default=12)
    ap.add_argument("--ssp-dim", type=int, default=SSP_DIM)
    ap.add_argument("--kmax", type=int, default=MAX_HARMONIC)
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(n_views=args.n_views)
    Z = encode_hog(imgs)
    nv, n_obj = args.n_views, len(names)
    vi = np.concatenate([np.arange(nv)] * n_obj)
    kept, held = blocked_masks(vi, nv, args.gap_deg)
    step = 360.0 / nv
    period = 2 * max(int(round(args.gap_deg / step)), 1)
    arc = vi // period                                # contiguous held-out arc id
    obj_h, arc_h = obj[held], arc[held]

    print(f"{len(Z)} crops, {Z.shape[1]}-D HOG, {n_obj} objects x {nv} azimuths")
    print(f"blocked split, {args.gap_deg:.0f} deg held-out arcs: "
          f"{int(kept.sum()/n_obj)} kept / {int(held.sum()/n_obj)} held out per "
          f"object, {len(np.unique(arc_h))} arcs")
    print(f"{args.seeds} seeds (random projection, SSP phases, ID atoms all "
          f"redrawn); chance = 90 deg")
    print(f"ssp_dim={args.ssp_dim}, max_harmonic={args.kmax}\n")

    Ks = (6, 12, 24, int(kept.sum() // n_obj))
    Ks = sorted(set(k for k in Ks if k <= kept.sum() // n_obj))

    hdr = (f"{'K':>4s} {'decoder':>10s} {'median err':>11s} {'<15deg':>7s} "
           f"{'ID hit':>7s} {'map floats':>11s} {'quant floor':>12s}")
    print("[A] equal store: every decoder sees the same K views on file")
    print(hdr); print("-" * len(hdr))

    keep_for_boot = {}
    for K in Ks:
        runs = [one_run(Z, obj, az, kept, held, names, K, s,
                        d=args.ssp_dim, kmax=args.kmax)
                for s in range(args.seeds)]
        beta = runs[0][1]
        d = runs[0][2]
        floor = 360.0 / K / 4.0                       # mean |err| of quantising
        for name in ("nearest", "kernel", "vsa", "vsa-scene"):
            errs = [r[0][name][0] for r in runs]
            hits = [r[0][name][1].mean() for r in runs]
            e = np.concatenate(errs)
            mem = (n_obj * K * d if name in ("nearest", "kernel")
                   else n_obj * d if name == "vsa" else d)
            fl = f"{floor:11.1f}d" if name == "nearest" else " " * 12
            print(f"{K:4d} {name:>10s} {np.median(e):10.1f}d "
                  f"{(e < 15).mean():7.2f} {np.mean(hits):7.2f} {mem:11d} "
                  f"{fl}")
            keep_for_boot[(K, name)] = errs
        print(f"     (kernel temperature beta={beta:g}, fitted leave-one-out "
              f"on the store)")
    print("  map floats counts only what grows with the store; the random "
          "projection\n  W is shared by every decoder, ID atoms are "
          "regenerable from a seed")

    print("\n[B] paired difference, hierarchical bootstrap "
          f"(objects -> arcs -> seeds, {args.n_boot} draws)")
    print(f"  positive = the object file is WORSE than the list\n")
    print(f"  {'K':>4s} {'contrast':>22s} {'median diff':>12s} {'95% CI':>20s}")
    print("  " + "-" * 60)
    for K in Ks:
        for a, b in (("vsa", "nearest"), ("vsa", "kernel"),
                     ("vsa-scene", "vsa")):
            ea, eb = keep_for_boot[(K, a)], keep_for_boot[(K, b)]
            lo, hi = hier_bootstrap(ea, obj_h, arc_h, n_obj,
                                    n_boot=args.n_boot, per_seed_b=eb)
            m = np.median(np.concatenate(ea)) - np.median(np.concatenate(eb))
            verdict = ("" if lo <= 0 <= hi else
                       "  <-- " + ("worse" if lo > 0 else "better"))
            print(f"  {K:4d} {a + ' - ' + b:>22s} {m:11.1f}d "
                  f"[{lo:+6.1f}, {hi:+6.1f}]{verdict}")

    print("\n[C] per object at K=12 (median in-gap error, degrees)")
    runs = [one_run(Z, obj, az, kept, held, names, 12, s,
                    d=args.ssp_dim, kmax=args.kmax)
            for s in range(args.seeds)]
    print(f"  {'object':>9s} " + " ".join(f"{n:>10s}" for n in
                                          ("nearest", "kernel", "vsa",
                                           "vsa-scene")))
    print("  " + "-" * 53)
    for i, nm in enumerate(names):
        row = []
        for name in ("nearest", "kernel", "vsa", "vsa-scene"):
            e = np.concatenate([r[0][name][0][obj_h == i] for r in runs])
            row.append(f"{np.median(e):9.1f}d")
        print(f"  {nm:>9s} " + " ".join(row))
    print("\n  a 90-deg-symmetric object should alias for EVERY decoder; if "
          "one of\n  them looks good on the cube it is reading something "
          "other than pose")

    dimension_sweep(Z, obj, az, kept, held, names, args)


# ---------------------------------------------------------------------------
# [D] How much dimension does the object file need to match the list?
#
# [A] holds d at the repo default 151, which is where sec.4-13 were measured.
# That is also where a bundle of K correlated keys runs out of room.  The
# question this answers is not "which decoder is better" but "what does the
# object file cost, in floats, to reach the list's accuracy" -- because the
# only thing it can win on is memory and superposition.
# ---------------------------------------------------------------------------

def dimension_sweep(Z, obj, az, kept, held, names, args):
    n_obj, K = len(names), args.sweep_k
    mu, V, lam = fit_basis(Z[kept])
    print(f"\n[D] dimension sweep at K={K}: what the object file costs to "
          f"match the list")
    hdr = (f"  {'d':>5s} {'kmax':>5s} {'lobe':>6s} {'per-object':>11s} "
           f"{'scene vec':>10s} {'scene<15':>9s} {'list floats':>12s} "
           f"{'scene floats':>13s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for d in args.dims:
        for kmax in (8, 16):
            ev, es, en = [], [], []
            lobe = 0.0
            for s in range(args.seeds):
                rng = np.random.default_rng(s)
                W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
                keys = condition(Z, mu, V, lam, W, drop=DROP)
                vs = CircularSSPSpace(1, ssp_dim=d, max_harmonic=kmax,
                                      rng=np.random.default_rng(s + 1000))
                lobe = np.rad2deg(vs.lobe_width())
                vocab = AtomVocab(d, seed=s + 2000)
                books = []
                for o in range(n_obj):
                    sel = np.where((obj == o) & kept)[0]
                    pick = sel[pick_store(az[sel], K)]
                    books.append(bundle(np.stack(
                        [bind(k, vs.encode([a])) for k, a in
                         zip(keys[pick], az[pick])])))
                    en.append(np.abs(np.rad2deg(wrap_angle(
                        decode_nearest(keys[pick], az[pick],
                                       keys[held & (obj == o)])[0]
                        - az[held & (obj == o)]))))
                scene = bundle(np.stack([bind(vocab.mint(names[o]), books[o])
                                         for o in range(n_obj)]))
                for o in range(n_obj):
                    q, t = keys[held & (obj == o)], az[held & (obj == o)]
                    a, _ = decode_vsa(books[o], q, vs)
                    ev.append(np.abs(np.rad2deg(wrap_angle(a - t))))
                    a, _ = decode_vsa(unbind(scene, vocab[names[o]]), q, vs)
                    es.append(np.abs(np.rad2deg(wrap_angle(a - t))))
            ev, es, en = (np.concatenate(x) for x in (ev, es, en))
            print(f"  {d:5d} {kmax:5d} {lobe:5.0f}d {np.median(ev):10.1f}d "
                  f"{np.median(es):9.1f}d {(es < 15).mean():9.2f} "
                  f"{n_obj * K * d:12d} {d:13d}")
    print(f"  list floats is what `nearest` must keep: {n_obj} objects x "
          f"{K} views x d.\n  scene floats is the whole map -- every object, "
          f"every view -- in ONE vector.\n  The comparison to make is "
          f"error at equal FLOATS, not error at equal d.")



if __name__ == "__main__":
    main()
