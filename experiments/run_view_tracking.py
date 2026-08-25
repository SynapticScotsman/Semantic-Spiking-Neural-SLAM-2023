"""Continuity: a Bayes filter on the view circle, so the estimate cannot teleport.

Read frame by frame, the view-direction estimate jumps -- 50 degrees between
consecutive crops, sometimes 180.  A camera orbiting an object cannot do that.
Every frame is being decoded independently, throwing away the strongest prior
available: you were somewhere near here a moment ago.

The fix is a recursive Bayes filter over the circle, and both of its steps are
native to this representation:

    predict   b(phi) <- b(phi - delta) smoothed by a wrapped Gaussian
              In the harmonic domain this is  B_k <- B_k * exp(-i k delta)
                                                        * exp(-sigma^2 k^2 / 2)
              The first factor is exactly binding by S_view(delta) -- the same
              operation as `orbit()`.  The second is per-harmonic damping.

    update    b <- normalise( b * softmax_beta(likelihood) )
              The likelihood is what `localise_view` already returns.

This is the view-circle twin of the position filter in astm's RESULTS_SO_FAR.md
finding 6 (predict is one bind, update is one bundle), and it is scored the same
way: not on whether it beats the per-frame estimate on median error, but on
whether it removes physically impossible motion.

    python experiments/run_view_tracking.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import CircularSSPSpace, wrap_angle  # noqa: E402
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import (encode_hog, fit_basis, condition,  # noqa: E402
                                   build_books)
from run_blocked_split import blocked_masks  # noqa: E402

SSP_DIM, MAXH, NB = 151, 8, 360


def predict(B, delta, sigma, k):
    """One filter prediction, entirely in the harmonic domain."""
    return B * np.exp(-1j * k * delta) * np.exp(-0.5 * (sigma * k) ** 2)


def track(fields, step, sigma, beta, k, b0=None):
    """Run the filter over a trajectory of likelihood fields."""
    n = fields.shape[1]
    b = np.full(n, 1.0 / n) if b0 is None else b0.copy()
    out = np.empty((len(fields), n))
    for t, L in enumerate(fields):
        if t > 0:
            b = np.real(np.fft.ifft(predict(np.fft.fft(b), step, sigma, k)))
            b = np.maximum(b, 0)
            b /= max(b.sum(), 1e-12)
        w = np.exp(beta * (L - L.max()))
        b = b * w
        s = b.sum()
        b = b / s if s > 1e-300 else np.full(n, 1.0 / n)
        out[t] = b
    return out


def circ_argmax(b, grid):
    return grid[int(np.argmax(b))]


def readout(b, grid, prev, hysteresis=True, margin=1.25):
    """Which mode to report.

    A belief with two near-equal modes has a MAP that flips between them on
    noise alone, which shows up as a 180 degree jump that the belief itself
    never made.  Preferring the mode nearest the previous estimate, unless
    another beats it by `margin`, keeps the read-out as continuous as the
    thing it is reading.
    """
    if not hysteresis or prev is None:
        return circ_argmax(b, grid)
    loc = np.where((b >= np.roll(b, 1)) & (b >= np.roll(b, -1))
                   & (b > 0.2 * b.max()))[0]
    if len(loc) == 0:
        return circ_argmax(b, grid)
    near = loc[np.argmin(np.abs(wrap_angle(grid[loc] - prev)))]
    best = loc[np.argmax(b[loc])]
    return grid[best] if b[best] > margin * b[near] else grid[near]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap-deg", type=float, default=30.0)
    ap.add_argument("--sigma-deg", type=float, default=2.0,
                    help="process noise: how far the belief is allowed to "
                         "spread per frame")
    ap.add_argument("--beta", type=float, default=3.0,
                    help="how sharply appearance evidence is trusted")
    ap.add_argument("--odo-noise-deg", type=float, default=1.5)
    ap.add_argument("--drop", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-deg", type=float, default=0.0,
                    help="width of a prior on the FIRST viewing direction. 0 "
                         "means start from a uniform belief. One known "
                         "starting direction is all it takes to collapse a "
                         "symmetric object's branch ambiguity.")
    ap.add_argument("--no-hysteresis", action="store_true",
                    help="read out the raw MAP instead of preferring the mode "
                         "nearest the previous estimate")
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(n_views=72)
    Z = encode_hog(imgs)
    nv = 72
    vi = np.concatenate([np.arange(nv)] * len(names))
    vs = CircularSSPSpace(1, ssp_dim=SSP_DIM, max_harmonic=MAXH,
                          rng=np.random.default_rng(1))
    kept, _ = blocked_masks(vi, nv, args.gap_deg)
    mu, V, lam = fit_basis(Z[kept])
    rng = np.random.default_rng(args.seed)
    W = rng.standard_normal((V.shape[0], SSP_DIM)) / np.sqrt(V.shape[0])
    keys = condition(Z, mu, V, lam, W, drop=args.drop)
    books = build_books(keys, obj, az, kept, len(names), vs)

    grid = np.linspace(-np.pi, np.pi, NB, endpoint=False)
    kfreq = np.fft.fftfreq(NB, d=1.0 / NB)
    true_step = 2 * np.pi / nv
    sigma = np.deg2rad(args.sigma_deg)

    print(f"Trajectory: one full orbit, {nv} frames, {np.rad2deg(true_step):.0f}° "
          f"per frame. Object files hold kept views only ({args.gap_deg:.0f}° "
          f"held-out arcs).")
    print(f"Filter: sigma {args.sigma_deg:.0f}°/frame, beta {args.beta:.0f}, "
          f"odometry noise {args.odo_noise_deg:.1f}°\n")

    hdr = (f"{'object':9s} {'per-frame err':>14s} {'filtered err':>13s} "
           f"{'max jump':>9s} {'filt max jump':>14s} {'impossible':>11s} "
           f"{'filt imposs':>12s}")
    print(hdr); print("-" * len(hdr))
    agg = {"raw": [], "flt": [], "rj": [], "fj": [], "ri": 0, "fi": 0, "n": 0}
    per_obj = {}
    for o, nm in enumerate(names):
        sel = np.where(obj == o)[0]
        order = sel[np.argsort(az[sel])]
        a_true = az[order]
        fields = np.stack([vs.view_likelihood(books[o], keys[i], n_per_dim=NB)[1]
                           for i in order])
        # per-frame decode, no memory
        raw = np.array([circ_argmax(f, grid) for f in fields])
        # filtered, with noisy odometry
        odo = true_step + rng.normal(0, np.deg2rad(args.odo_noise_deg), len(order))
        if args.anchor_deg > 0:
            b = np.exp(np.cos(grid - a_true[0]) / np.deg2rad(args.anchor_deg) ** 2)
            b /= b.sum()
        else:
            b = np.full(NB, 1.0 / NB)
        flt = np.empty(len(order))
        beliefs = np.empty((len(order), NB))
        prev = None
        for t, L in enumerate(fields):
            if t > 0:
                b = np.real(np.fft.ifft(predict(np.fft.fft(b), odo[t], sigma, kfreq)))
                b = np.maximum(b, 0); b /= max(b.sum(), 1e-12)
            w = np.exp(args.beta * (L - L.max()))
            b = b * w
            s = b.sum(); b = b / s if s > 1e-300 else np.full(NB, 1.0 / NB)
            beliefs[t] = b
            flt[t] = readout(b, grid, prev, not args.no_hysteresis)
            prev = flt[t]

        re_ = np.abs(np.rad2deg(wrap_angle(raw - a_true)))
        fe = np.abs(np.rad2deg(wrap_angle(flt - a_true)))
        rj = np.abs(np.rad2deg(wrap_angle(np.diff(raw))))
        fj = np.abs(np.rad2deg(wrap_angle(np.diff(flt))))
        thr = 3 * np.rad2deg(true_step)          # >3x the real step is impossible
        agg["raw"] += list(re_); agg["flt"] += list(fe)
        agg["rj"] += list(rj); agg["fj"] += list(fj)
        agg["ri"] += int((rj > thr).sum()); agg["fi"] += int((fj > thr).sum())
        agg["n"] += len(rj)
        per_obj[nm] = dict(raw=re_.tolist(), flt=fe.tolist(),
                           rj=rj.tolist(), fj=fj.tolist(),
                           beliefs=beliefs, a_true=a_true)
        print(f"{nm:9s} {np.median(re_):13.1f}° {np.median(fe):12.1f}° "
              f"{rj.max():8.0f}° {fj.max():13.0f}° {int((rj>thr).sum()):11d} "
              f"{int((fj>thr).sum()):12d}")

    print("-" * len(hdr))
    print(f"{'all':9s} {np.median(agg['raw']):13.1f}° {np.median(agg['flt']):12.1f}° "
          f"{max(agg['rj']):8.0f}° {max(agg['fj']):13.0f}° "
          f"{agg['ri']:11d} {agg['fi']:12d}")
    print(f"\nimpossible = frame-to-frame change over {3*np.rad2deg(true_step):.0f}° "
          f"(3x the true {np.rad2deg(true_step):.0f}° step), out of {agg['n']} transitions")
    return per_obj


if __name__ == "__main__":
    main()
