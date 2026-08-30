"""E7: does self-occlusion hurt, and is the held-out arc really the hard part?

Two questions that sound like one, both about *where round the orbit* the
errors live.

**Occlusion.**  Sec.13 asserted that the appearance manifold is "smooth over
small rotations, discontinuous where a part appears or disappears", and used
that to justify the renderer having genuine self-occlusion.  It was never
measured.  Dense optical flow between adjacent views gives a direct handle: a
rigid turn produces flow that warps one frame almost exactly onto the next,
and the residual after warping is large precisely where surface has appeared
or vanished, because newly visible pixels cannot be predicted from a frame
that never saw them.  So the warp residual is an **occlusion score per view**,
computed from the images alone with no map and no descriptor.

Then the question has a sign, and both answers are plausible:

  * occlusion **hurts** -- the descriptor changes discontinuously there, so
    interpolating between stored views across an occlusion event is
    extrapolating in disguise;
  * occlusion **helps** -- a handle swinging into view is the single most
    distinctive thing that happens on the whole orbit, and distinctive is what
    localisation needs.

**The blocked arc.**  Sec.0 E4 replaced the interleaved split with contiguous
held-out arcs and the error roughly tripled, which is why every figure since
uses them.  But the arc has an inside: a query at the edge of a 30 degree hole
sits 5 degrees from a stored view, one in the middle sits 15.  Nothing has
looked at whether the damage is spread evenly or concentrated in the middle,
which is what "the blocked part degrades performance" actually asserts.

    python experiments/run_optical_flow.py
    python experiments/run_optical_flow.py --symmetric-set --save-dir data
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
from run_nn_baseline import pick_store, decode_vsa, DROP  # noqa: E402


# ---------------------------------------------------------------------------
# Optical flow between adjacent views, and what the flow cannot explain
# ---------------------------------------------------------------------------

def warp(img, v, u):
    """Move ``img`` by the flow field, nearest-neighbour, clipped at the edge."""
    h, w = img.shape
    r, c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rr = np.clip(np.rint(r + v).astype(int), 0, h - 1)
    cc = np.clip(np.rint(c + u).astype(int), 0, w - 1)
    return img[rr, cc]


def occlusion_scores(gray, n_views):
    """Per-view residual after warping the previous view onto it.

    Large where surface appeared or vanished between the two frames: flow can
    move pixels about, it cannot invent them.
    """
    from skimage.registration import optical_flow_ilk
    res, mag = np.zeros(n_views), np.zeros(n_views)
    for i in range(n_views):
        a, b = gray[(i - 1) % n_views], gray[i]
        v, u = optical_flow_ilk(a, b, radius=5)
        res[i] = float(np.abs(warp(a, v, u) - b).mean())
        mag[i] = float(np.hypot(v, u).mean())
    return res, mag


def localisation_errors(Z, obj, az, kept, held, n_obj, K, d, kmax, seeds):
    """Per-crop view error, kept alongside its view index."""
    mu, V, lam = fit_basis(Z[kept])
    out = {o: {} for o in range(n_obj)}
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
            m = np.where(held & (obj == o))[0]
            a, _ = decode_vsa(book, keys[m], vs)
            e = np.abs(np.rad2deg(wrap_angle(a - az[m])))
            for idx, err in zip(m, e):
                out[o].setdefault(idx, []).append(float(err))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-views", type=int, default=72)
    ap.add_argument("--gap-deg", type=float, default=30.0)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--ssp-dim", type=int, default=2401)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--symmetric-set", action="store_true")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(
        n_views=args.n_views,
        extra="symmetric" if args.symmetric_set else None)
    Z = encode_hog(imgs)
    nv, n_obj = args.n_views, len(names)
    vi = np.concatenate([np.arange(nv)] * n_obj)
    kept, held = blocked_masks(vi, nv, args.gap_deg)
    gray = imgs.mean(-1)

    print(f"{n_obj} objects x {nv} views, {360/nv:.0f} deg apart; "
          f"d={args.ssp_dim}, max_harmonic={args.kmax}, K={args.K}, "
          f"{args.seeds} seeds")
    print("optical flow: iterative Lucas-Kanade between adjacent views; the "
          "score is\nthe residual after warping one onto the next -- what the "
          "flow cannot explain\n")

    occ = {}
    for o, nm in enumerate(names):
        occ[o] = occlusion_scores(gray[obj == o], nv)

    errs = localisation_errors(Z, obj, az, kept, held, n_obj, args.K,
                               args.ssp_dim, args.kmax, args.seeds)

    print("[A] occlusion, per object")
    print(f"  {'object':>9s} {'mean resid':>11s} {'peak resid':>11s} "
          f"{'peak at':>8s} {'flow mag':>9s} {'events >2x':>11s}")
    print("  " + "-" * 65)
    for o, nm in enumerate(names):
        r, m = occ[o]
        peak = int(np.argmax(r))
        events = int((r > 2 * np.median(r)).sum())
        print(f"  {nm:>9s} {r.mean():11.4f} {r.max():11.4f} "
              f"{np.rad2deg(wrap_angle(np.linspace(-np.pi, np.pi, nv, endpoint=False)[peak])):7.0f}d "
              f"{m.mean():9.2f} {events:11d}")
    print("  'events' counts views whose residual is more than twice the "
          "object's own\n  median -- a part appearing or vanishing, not a "
          "smooth turn.")

    print("\n[B] does localisation error concentrate at occlusion events?")
    print(f"  {'object':>9s} {'r(resid, err)':>14s} {'err at events':>14s} "
          f"{'err elsewhere':>14s}")
    print("  " + "-" * 56)
    rs = []
    for o, nm in enumerate(names):
        r, _ = occ[o]
        idx = sorted(errs[o])
        e = np.array([np.median(errs[o][i]) for i in idx])
        rv = r[np.array(idx) % nv]
        if np.std(rv) < 1e-12 or np.std(e) < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(rv, e)[0, 1])
        hi = rv > 2 * np.median(r)
        rs.append(corr)
        a = f"{np.median(e[hi]):13.1f}d" if hi.any() else f"{'--':>14s}"
        b = f"{np.median(e[~hi]):13.1f}d" if (~hi).any() else f"{'--':>14s}"
        print(f"  {nm:>9s} {corr:+14.2f} {a} {b}")
    good = [x for x in rs if np.isfinite(x)]
    print(f"  mean correlation across objects: {np.mean(good):+.2f}")
    print("  positive = occlusion HURTS (error rises where surface appears "
          "or vanishes).\n  negative = occlusion HELPS (the discontinuity is "
          "a landmark).")

    print("\n[C] pooled: bin every held-out view by how occluded it is")
    print("  residual is z-scored within each object, so a hard-edged object "
          "and a\n  smooth one contribute on the same scale. Aliased objects "
          "are dropped:\n  they sit at chance whatever the flow does.")
    from run_frontend_diagnostics import diagnose
    mu_, V_, lam_ = fit_basis(Z[kept])
    W_ = (np.random.default_rng(0).standard_normal((V_.shape[0], args.ssp_dim))
          / np.sqrt(V_.shape[0]))
    k0 = condition(Z, mu_, V_, lam_, W_, drop=DROP)
    vs0 = CircularSSPSpace(1, ssp_dim=args.ssp_dim, max_harmonic=args.kmax,
                           rng=np.random.default_rng(1000))
    dg = diagnose(k0, obj, az, names, nv, np.rad2deg(vs0.lobe_width()))
    zs, es, dose = [], [], []
    for o in range(n_obj):
        if dg[o]["alias_peak"] >= 0.75:
            continue
        r, _ = occ[o]
        idx = sorted(errs[o])
        e = np.array([np.median(errs[o][i]) for i in idx])
        rv = r[np.array(idx) % nv]
        z = (rv - r.mean()) / max(r.std(), 1e-12)
        zs.append(z); es.append(e)
        hi = rv > 2 * np.median(r)
        if hi.any() and (~hi).any():
            dose.append((r.mean(), float(np.median(e[hi])
                                         - np.median(e[~hi]))))
    zs, es = np.concatenate(zs), np.concatenate(es)
    print(f"\n  {'occlusion quartile':>19s} {'n':>6s} {'median err':>11s} "
          f"{'<15deg':>7s}")
    print("  " + "-" * 46)
    qs = np.percentile(zs, [0, 25, 50, 75, 100])
    for i in range(4):
        m = (zs >= qs[i]) & (zs <= qs[i + 1])
        print(f"  {['smoothest', 'Q2', 'Q3', 'most occluded'][i]:>19s} "
              f"{m.sum():6d} {np.median(es[m]):10.1f}d "
              f"{(es[m] < 15).mean():7.2f}")
    r_all = float(np.corrcoef(zs, es)[0, 1])
    print(f"  pooled correlation r = {r_all:+.2f} over {len(zs)} held-out "
          f"views from {len(dose)} objects")
    if len(dose) >= 3:
        a = np.array([d[0] for d in dose]); b = np.array([d[1] for d in dose])
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            print(f"  dose-response across objects: r = "
                  f"{float(np.corrcoef(a, b)[0, 1]):+.2f} between how much an "
                  f"object occludes\n  and how much its occluded views cost "
                  f"-- {len(dose)} objects, so this is a hint, not a result")

    print("\n[D] inside the held-out arc: is the middle worse than the edge?")
    step = 360.0 / nv
    gap = max(int(round(args.gap_deg / step)), 1)
    depth_err = {}
    for o in range(n_obj):
        for idx, v in errs[o].items():
            pos = (idx % nv) % (2 * gap)          # 0..gap-1 inside the hole
            d_edge = min(pos, gap - 1 - pos) + 1  # 1 = touching a stored arc
            depth_err.setdefault(d_edge, []).extend(v)
    print(f"  {'steps from the edge':>21s} {'deg from stored':>16s} "
          f"{'n':>6s} {'median err':>11s} {'<15deg':>7s}")
    print("  " + "-" * 66)
    for dpt in sorted(depth_err):
        v = np.array(depth_err[dpt])
        print(f"  {dpt:21d} {dpt * step:15.0f}d {len(v):6d} "
              f"{np.median(v):10.1f}d {(v < 15).mean():7.2f}")
    ks = sorted(depth_err)
    slope = np.polyfit([k * step for k in ks],
                       [np.median(depth_err[k]) for k in ks], 1)[0]
    print(f"  slope: {slope:+.2f} deg of error per degree of distance from "
          f"the nearest\n  stored view. This is what 'the blocked part "
          f"degrades performance' means,\n  measured rather than assumed.")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out = os.path.join(args.save_dir, "optical_flow.npz")
        np.savez(out,
                 names=np.array(names),
                 residual=np.stack([occ[o][0] for o in range(n_obj)]),
                 flow_mag=np.stack([occ[o][1] for o in range(n_obj)]),
                 err_index=np.array([sorted(errs[o]) for o in range(n_obj)]),
                 err=np.stack([[np.median(errs[o][i]) for i in sorted(errs[o])]
                               for o in range(n_obj)]))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
