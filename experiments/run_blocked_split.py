"""Blocked-split re-run of the view-direction figures (FINDINGS.md §0 E4).

The original evaluation held out *alternate* azimuths, so every held-out view
sat within 15 degrees of a stored side.  `astm/docs/RESULTS_SO_FAR.md` corrects
exactly that pattern: a split that leaves neighbours of the query in memory
measures memorisation, not generalisation, and scattering individual frames
does not fix it.

This re-runs it properly, and asks the more useful question while it is at it.
A single blocked number would only say pass or fail; a **gap sweep** says how
far the object file can extrapolate, which is the same thing as how densely you
have to orbit an object to be able to localise on it later.

Protocol per gap width `g`:

  * carve the circle into contiguous arcs of width `g`, alternating held-out
    and kept, so held-out queries are never adjacent to a stored side
  * additionally evict any stored side within `evict` degrees of a held-out
    query (belt and braces -- `g` alone already guarantees separation)
  * fit the latent statistics on kept views ONLY
  * build each object file from kept views only
  * report error inside the gaps, and on kept views as a control

Chance is 90 degrees (uniform error on a circle).  A constant predictor is also
reported, since `RESULTS_SO_FAR.md` warns that a chance baseline alone is not
enough.

    python experiments/run_blocked_split.py
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
                                   build_books, localise)

SSP_DIM, MAX_HARMONIC = 151, 8


def blocked_masks(view_idx, n_views, gap_deg, evict_deg=0.0):
    """Alternating contiguous arcs.  Returns (kept, held_out) boolean masks."""
    step = 360.0 / n_views
    gap = max(int(round(gap_deg / step)), 1)
    period = 2 * gap                      # one held-out arc, one kept arc
    held = (view_idx % period) < gap
    if evict_deg > 0:
        r = int(round(evict_deg / step))
        grown = np.zeros_like(held)
        for s in range(-r, r + 1):
            grown |= held[(np.arange(len(held)) // n_views) * n_views
                          + (view_idx + s) % n_views]
        kept = ~grown
    else:
        kept = ~held
    return kept, held


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-views", type=int, default=72)
    ap.add_argument("--evict-deg", type=float, default=0.0)
    ap.add_argument("--drop", type=int, default=2)
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(n_views=args.n_views)
    Z = encode_hog(imgs)
    nv = args.n_views
    vi = np.concatenate([np.arange(nv)] * len(names))
    vs = CircularSSPSpace(1, ssp_dim=SSP_DIM, max_harmonic=MAX_HARMONIC,
                          rng=np.random.default_rng(1))
    rng = np.random.default_rng(0)
    print(f"{len(Z)} crops, {Z.shape[1]}-D HOG, {len(names)} objects x {nv} "
          f"azimuths ({360/nv:.0f} deg apart)")
    print(f"view kernel half-width {np.rad2deg(vs.lobe_width()):.0f} deg; "
          f"chance = 90 deg\n")

    print("Gap sweep: contiguous held-out arcs, statistics and object files "
          "fitted on kept views only")
    hdr = (f"{'gap':>6s} {'stored/obj':>11s} {'nearest kept':>13s} "
           f"{'IN-GAP median':>14s} {'<15deg':>7s} {'control':>9s} {'ID hit':>7s}")
    print(hdr); print("-" * len(hdr))

    rows = {}
    for gap_deg in (0, 10, 20, 30, 45, 60, 90):
        if gap_deg == 0:
            kept = vi % 2 == 0                       # the ORIGINAL interleaved split
            held = ~kept
            label = "interl."
        else:
            kept, held = blocked_masks(vi, nv, gap_deg, args.evict_deg)
            label = f"{gap_deg}d"
        if held.sum() == 0 or kept.sum() < 20:
            continue

        mu, V, lam = fit_basis(Z[kept])              # stats from kept views only
        W = rng.standard_normal((V.shape[0], SSP_DIM)) / np.sqrt(V.shape[0])
        keys = condition(Z, mu, V, lam, W, drop=args.drop)
        books = build_books(keys, obj, az, kept, len(names), vs)

        err, idh, _, _ = localise(books, keys, obj, az, held, vs)
        ctrl, _, _, _ = localise(books, keys, obj, az, kept, vs)

        # how far is a held-out query from the nearest view still in the file?
        sep = []
        for o in range(len(names)):
            ka = az[(obj == o) & kept]
            for a in az[(obj == o) & held]:
                sep.append(np.rad2deg(np.abs(wrap_angle(ka - a)).min()))
        rows[label] = (err, np.median(sep))
        print(f"{label:>6s} {int(kept.sum()/len(names)):11d} "
              f"{np.median(sep):12.1f}d {np.median(err):13.1f}d "
              f"{(err < 15).mean():7.2f} {np.median(ctrl):8.1f}d {idh:7.2f}")

    # baselines
    const = np.abs(np.rad2deg(wrap_angle(az - np.median(az))))
    print(f"\nbaselines: chance 90.0 deg; constant predictor "
          f"{np.median(const):.1f} deg")

    print("\nper-object, 30 deg gap (the first split with no adjacent side):")
    if "30d" in rows:
        kept, held = blocked_masks(vi, nv, 30, args.evict_deg)
        err = rows["30d"][0]
        oh = obj[held]
        for i, nm in enumerate(names):
            e = err[oh == i]
            print(f"    {nm:9s} median {np.median(e):6.1f} deg   "
                  f"<15 deg {(e < 15).mean():.2f}")


if __name__ == "__main__":
    main()
