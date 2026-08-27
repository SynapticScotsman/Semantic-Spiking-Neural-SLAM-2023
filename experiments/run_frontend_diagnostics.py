"""What the appearance front end alone says, before any VSA is built.

Two kinds of anisotropy get conflated under one word, and they are different
things with different causes and different fixes.

**Between-object.**  The leading principal directions say *which object this
is*.  They are removable, and `run_view_localisation.py` removes them.

**Across-view.**  Views a few degrees apart are nearly collinear, so a short
stretch of orbit occupies very few effective dimensions.  This is **not** an
encoder pathology.  It is what walking a continuous trajectory produces, for
any encoder, and it is the same phenomenon as scene redundancy in a video: two
frames close in time are close in pose and therefore close in appearance.

The two are linked in a way that is easy to get backwards.  An object's
identity does not change as you walk around it, so the identity-carrying
directions are *constant along the orbit* and act as a DC pedestal on the
appearance autocorrelation, inflating every lag equally.  Removing them does
not create angular information; it unmasks what was already there.  The local
effective rank is untouched.  Conditioning buys contrast, not information.

Three numbers come out, all computable without building an object file, and
two of them predict the failure modes:

  half-width        angular lag at which appearance similarity falls to half.
                    The finest angular distinction the descriptor supports.
  alias peak        highest similarity at a *circular* lag beyond the view
                    kernel.  This is the aliasing predictor.  It MUST be
                    circular: on a 72-view ring a lag of 355 degrees is the
                    adjacent frame, and scanning linear lag reports every
                    object's own neighbour as an alias.
  N_eff             effective independent views in one orbit, from the
                    integrated autocorrelation time.  Frames are not samples.

    python experiments/run_frontend_diagnostics.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import CircularSSPSpace  # noqa: E402
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import encode_hog, fit_basis, condition  # noqa: E402
from run_blocked_split import blocked_masks  # noqa: E402


def effective_rank(X):
    """exp of the entropy of the eigenspectrum -- how many dimensions in use."""
    X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-300)).sum()))


def autocorrelation(keys):
    """Mean cosine between views at each circular lag."""
    n = len(keys)
    return np.array([float(np.mean(np.sum(keys * np.roll(keys, -d, axis=0), axis=1)))
                     for d in range(n)])


def n_effective(rho):
    """Independent samples in a correlated ring, via integrated autocorr time."""
    half = rho[1:len(rho) // 2]
    tau = 1 + 2 * np.sum(half[half > 0])
    return len(rho) / max(tau, 1.0)


def diagnose(keys, obj, az, names, n_views, lobe_deg):
    deg = np.arange(n_views) * (360.0 / n_views)
    circ = np.minimum(deg, 360 - deg)          # 355 deg IS 5 deg away
    out = []
    for o, nm in enumerate(names):
        order = np.where(obj == o)[0][np.argsort(az[obj == o])]
        K = keys[order]
        C = autocorrelation(K)
        rho = C / C[0]
        below = np.nonzero(C < 0.5 * C[0])[0]
        hw = float(deg[below[0]]) if below.size else 180.0
        far = circ > 2 * lobe_deg
        ap = float(C[far].max()) if far.any() else float("nan")
        al = float(circ[far][np.argmax(C[far])]) if far.any() else float("nan")
        win = [K[[(i + j) % n_views for j in range(-3, 4)]] for i in range(n_views)]
        local = float(np.mean([effective_rank(w) for w in win]))
        out.append(dict(name=nm, half_width=hw, alias_peak=ap, alias_lag=al,
                        local_rank=local, global_rank=effective_rank(K),
                        n_eff=n_effective(rho)))
    return out


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--n-views", type=int, default=72)
    ap_.add_argument("--gap-deg", type=float, default=30.0)
    ap_.add_argument("--drop", type=int, default=2)
    args = ap_.parse_args()

    imgs, obj, az, names = load_turntable(n_views=args.n_views)
    Z = encode_hog(imgs)
    nv = args.n_views
    vi = np.concatenate([np.arange(nv)] * len(names))
    kept, _ = blocked_masks(vi, nv, args.gap_deg)
    mu, V, lam = fit_basis(Z[kept])
    W = (np.random.default_rng(0).standard_normal((V.shape[0], 151))
         / np.sqrt(V.shape[0]))
    lobe = float(np.rad2deg(CircularSSPSpace(
        1, ssp_dim=151, max_harmonic=8, rng=np.random.default_rng(1)).lobe_width()))

    def unit(X):
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    variants = [("raw HOG, centred only", unit(Z - Z.mean(0))),
                (f"conditioned, top {args.drop} PCs dropped",
                 condition(Z, mu, V, lam, W, drop=args.drop))]

    print(f"{len(names)} objects x {nv} views ({360/nv:.0f}° apart); "
          f"view kernel half-width {lobe:.0f}°\n")
    for label, K in variants:
        rows = diagnose(K, obj, az, names, nv, lobe)
        print(f"=== {label} ===")
        hdr = (f"  {'object':9s} {'half-width':>11s} {'alias peak':>11s} "
               f"{'at lag':>7s} {'local rank':>11s} {'orbit rank':>11s} {'N_eff':>7s}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for r in rows:
            print(f"  {r['name']:9s} {r['half_width']:10.0f}° {r['alias_peak']:11.3f} "
                  f"{r['alias_lag']:6.0f}° {r['local_rank']:11.1f} "
                  f"{r['global_rank']:11.1f} {r['n_eff']:7.1f}")
        hw = np.median([r["half_width"] for r in rows])
        lr = np.median([r["local_rank"] for r in rows])
        ne = np.median([r["n_eff"] for r in rows])
        print(f"  median half-width {hw:.0f}° (kernel {lobe:.0f}°) · "
              f"local rank {lr:.1f} · N_eff {ne:.1f} of {nv}\n")

    print("Read-off:")
    print("  * local rank is unchanged by conditioning -- the across-view")
    print("    anisotropy is the trajectory, not the encoder")
    print("  * the half-width shrinks -- conditioning removes a view-constant")
    print("    pedestal, unmasking angular contrast it did not create")
    print(f"  * an orbit of {nv} frames is worth ~{ne:.0f} independent views, so")
    print("    frame counts must not be used as sample sizes")


if __name__ == "__main__":
    main()
