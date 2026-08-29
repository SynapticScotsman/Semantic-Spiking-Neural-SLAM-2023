"""E6: the whole thing, end to end, on a walk.

Every part of this system has been measured alone and none of them have been
run together.  ``recognise()`` names an object with an unbound prototype
(sec.16 E1) and localises with the view book (sec.16 E2's ``max_harmonic``);
sec.12's circular Bayes filter smooths a viewpoint estimate over time; sec.16
E3 carries object vectors through motion with one bind and sec.16 E5 turns with
a 2x2 matrix.  This runs all of it as one loop over a trajectory, which is the
only thing that says whether the pieces compose or merely coexist.

The loop, per frame:

    key   <- crop
    who   <- identify(key)          prototype cosine, never the view book
    L     <- localise_view(who)     the one-FFT likelihood over the circle
    belief<- predict then update    sec.12, shift by the odometry increment
    where <- readout(belief)        with hysteresis, so a two-moded belief
                                    does not flip 180 degrees on noise
    scene <- bind(scene, S(-d))     every object vector carried at once

Three things it can show that no component test can:

1. whether a **wrong name** poisons the viewpoint filter, which is the obvious
   way a two-stage read-out fails and which nothing so far has tested;
2. whether the filter's gain over per-frame decoding survives when the object
   identity is also being estimated rather than given;
3. what the assembled system's error actually is, which is the number a robot
   would live with.

    python experiments/run_pipeline.py
    python experiments/run_pipeline.py --oracle-id     # ablate stage one
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import (CircularSSPSpace, bind, bundle,  # noqa: E402
                               cosine, normalize, wrap_angle)
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import encode_hog, fit_basis, condition  # noqa: E402
from run_blocked_split import blocked_masks  # noqa: E402
from run_nn_baseline import pick_store, DROP  # noqa: E402
from run_view_tracking import track, readout  # noqa: E402


def build_object(keys, angles, vs):
    """The two vectors per object that sec.16 E1 argued for."""
    book = bundle(np.stack([bind(k, vs.encode([a]))
                            for k, a in zip(keys, angles)]))
    return book, bundle(keys)


def walk(rng, n_views, n_steps, step_views):
    """A contiguous orbit: the robot walks round, it does not teleport."""
    start = rng.integers(n_views)
    return (start + step_views * np.arange(n_steps)) % n_views


def run_trajectory(keys, obj, az, vi, books, protos, vs, order, true_obj,
                   step_rad, sigma, beta, oracle_id, n_grid=360):
    """One walk.  Returns per-frame naming and viewpoint outcomes."""
    k = np.fft.fftfreq(n_grid, 1.0 / n_grid)
    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    names = np.zeros(len(order), dtype=int)
    fields = np.empty((len(order), n_grid))
    per_frame = np.empty(len(order))
    for t, idx in enumerate(order):
        key = keys[idx]
        who = true_obj if oracle_id else int(np.argmax(
            [float(cosine(p, key.reshape(1, -1))[0]) for p in protos]))
        names[t] = who
        _, L = vs.view_likelihood(books[who], key, n_per_dim=n_grid)
        fields[t] = L
        per_frame[t] = abs(float(np.rad2deg(
            wrap_angle(grid[int(np.argmax(L))] - az[idx]))))
    belief = track(fields, step_rad, sigma, beta, k)
    filtered, prev = np.empty(len(order)), None
    for t in range(len(order)):
        prev = readout(belief[t], grid, prev)
        filtered[t] = abs(float(np.rad2deg(wrap_angle(prev - az[order[t]]))))
    return names, per_frame, filtered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-views", type=int, default=72)
    ap.add_argument("--gap-deg", type=float, default=30.0)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--ssp-dim", type=int, default=2401)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--step-views", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--beta", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--walks", type=int, default=8)
    ap.add_argument("--oracle-id", action="store_true")
    ap.add_argument("--symmetric-set", action="store_true")
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(
        n_views=args.n_views,
        extra="symmetric" if args.symmetric_set else None)
    Z = encode_hog(imgs)
    nv, n_obj = args.n_views, len(names)
    vi = np.concatenate([np.arange(nv)] * n_obj)
    kept, held = blocked_masks(vi, nv, args.gap_deg)
    mu, V, lam = fit_basis(Z[kept])
    step_rad = 2 * np.pi * args.step_views / nv

    print(f"{len(Z)} crops, {n_obj} objects x {nv} azimuths; "
          f"d={args.ssp_dim}, max_harmonic={args.kmax}, K={args.K}")
    print(f"{args.walks} walks x {args.steps} frames x {args.seeds} seeds, "
          f"{np.rad2deg(step_rad):.0f} deg per step")
    print(f"stage one: {'ORACLE (ablated)' if args.oracle_id else 'prototype'}"
          f"; chance = 90 deg\n")

    got = {"name": [], "frame": [], "filt": [], "frame_named": [],
           "filt_named": [], "frame_mis": [], "filt_mis": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((V.shape[0], args.ssp_dim)) / np.sqrt(V.shape[0])
        keys = condition(Z, mu, V, lam, W, drop=DROP)
        vs = CircularSSPSpace(1, ssp_dim=args.ssp_dim, max_harmonic=args.kmax,
                              rng=np.random.default_rng(seed + 1000))
        books, protos = [], []
        for o in range(n_obj):
            sel = np.where((obj == o) & kept)[0]
            pick = sel[pick_store(az[sel], args.K)]
            b, p = build_object(keys[pick], az[pick], vs)
            books.append(b); protos.append(p)
        books = np.stack(books)

        for w in range(args.walks):
            o = rng.integers(n_obj)
            base = np.where(obj == o)[0]
            order = base[walk(rng, nv, args.steps, args.step_views)]
            nm, pf, fl = run_trajectory(keys, obj, az, vi, books, protos, vs,
                                        order, o, step_rad, args.sigma,
                                        args.beta, args.oracle_id)
            hit = nm == o
            got["name"].append(hit)
            got["frame"].append(pf); got["filt"].append(fl)
            got["frame_named"].append(pf[hit]); got["filt_named"].append(fl[hit])
            got["frame_mis"].append(pf[~hit]); got["filt_mis"].append(fl[~hit])

    cat = {k: np.concatenate(v) for k, v in got.items()}
    print("[A] the assembled system")
    print(f"  naming accuracy, per frame            "
          f"{cat['name'].mean():.2f}")
    print(f"  viewpoint, per frame (no filter)      "
          f"{np.median(cat['frame']):.1f} deg   "
          f"<15 deg {np.mean(cat['frame'] < 15):.2f}")
    print(f"  viewpoint, filtered                   "
          f"{np.median(cat['filt']):.1f} deg   "
          f"<15 deg {np.mean(cat['filt'] < 15):.2f}")
    gain = np.median(cat['frame']) - np.median(cat['filt'])
    print(f"  the filter is worth                   {gain:+.1f} deg")

    print("\n[B] does a wrong name poison the viewpoint?")
    print(f"  {'frames':>22s} {'n':>7s} {'per frame':>11s} {'filtered':>10s}")
    print("  " + "-" * 54)
    for label, a, b in (("named correctly", "frame_named", "filt_named"),
                        ("named wrongly", "frame_mis", "filt_mis")):
        if cat[a].size == 0:
            print(f"  {label:>22s} {0:7d} {'--':>11s} {'--':>10s}")
            continue
        print(f"  {label:>22s} {cat[a].size:7d} {np.median(cat[a]):10.1f}d "
              f"{np.median(cat[b]):9.1f}d")
    print("  a wrong name means the likelihood came from the wrong object's "
          "book, so\n  its viewpoint is meaningless. What matters is whether "
          "the filter carries\n  that damage into the frames either side of "
          "it.")

    print("\n[C] where the filter's gain comes from")
    per_walk_f = np.array([np.median(x) for x in got["frame"]])
    per_walk_l = np.array([np.median(x) for x in got["filt"]])
    better = (per_walk_l < per_walk_f).mean()
    print(f"  walks where filtering helped          {better:.2f} of "
          f"{len(per_walk_f)}")
    print(f"  median per-walk error, unfiltered     "
          f"{np.median(per_walk_f):.1f} deg")
    print(f"  median per-walk error, filtered       "
          f"{np.median(per_walk_l):.1f} deg")
    q = np.percentile(per_walk_l, [10, 50, 90])
    print(f"  filtered per-walk spread (10/50/90)   "
          f"{q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} deg")
    print("  a walk is the unit here, not a frame: frames within one orbit are "
          "not\n  independent (sec.0 E8), and the filter's whole job is to "
          "exploit that.")


if __name__ == "__main__":
    main()
