"""Measure the FPE similarity kernel itself — no data, no GPU, no trace.

The multi-scale sweep (room0_cgfront, 2026-08-15) found that superposing several
length scales loses to a single anisotropic one, monotonically in the number of
scales. The proposed mechanism was signal dilution, and the first writeup put a
1/sqrt(k) figure on it. That was reasoned, not measured, and it is wrong for this
implementation: Phasor.bundle divides by k, and BOTH sides of the query are
k-component means, so the coherent term carries a 1/k**2 * k = 1/k factor rather
than 1/sqrt(k).

Rather than argue the exponent, measure it. The kernel is
    kappa(dx, dy) = < phi(0,0), phi(dx,dy) > / dim
and needs nothing but the encoder — no observations, no labels, no scoring. This
reports, per configuration:

  peak        kappa(0,0), the self-similarity a matching observation contributes
  clutter     RMS of kappa far from the origin, where a correct decode wants zero
  peak/clutter  the quantity the argmax actually competes on
  width       the half-maximum radius along each axis, in metres

    python collab_tasks/scripts/kernel_shape.py
    python collab_tasks/scripts/kernel_shape.py --dim 8192 --repeats 5

Runs on a laptop in seconds. --repeats averages over encoder seeds so the
numbers are not one draw's luck.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# vsa.py imports torch at module level for its projection helpers; Phasor itself
# is pure NumPy. Same shim as local_decode_sweep.py — a placeholder that raises
# on attribute access, so if the kernel path ever does touch torch this fails
# loudly rather than quietly computing something else.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import types

    class _Absent(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(
                f"torch.{name} was actually used — the no-torch shim in "
                "kernel_shape.py assumed Phasor is pure NumPy. Install torch "
                "(CPU build is enough) and re-run.")

    sys.modules["torch"] = _Absent("torch")

from vsa_cognitive_mapping.vsa import Phasor  # noqa: E402

# The configurations the sweep actually ran, plus each bundle's members ALONE.
# The members are the control: if a bundle's width matches its narrowest member
# rather than sitting between its members, then adding a coarse scale buys
# nothing and the "sharp core AND broad tails" story is wrong.
CONFIGS = [
    ("0.60",                            "isotropic, inherited default"),
    ("0.45,0.27",                       "per-axis, best measured"),
    ("0.20,0.12",                       "  member alone — the fine scale"),
    ("0.15,0.09",                       "  member alone — the finer scale"),
    ("0.45,0.27+0.20,0.12",             "bundle x2, 2.2x apart"),
    ("0.45,0.27+0.15,0.09",             "bundle x2, 3x apart"),
    ("0.60,0.36+0.30,0.18+0.15,0.09",   "bundle x3"),
]


def parse_scales(spec: str) -> list[tuple[float, float]]:
    """Same grammar as 04_vsa_labels.py --length-scale."""
    out = []
    for grp in spec.split("+"):
        v = [float(t) for t in grp.split(",")]
        if len(v) == 1:
            v = [v[0], v[0]]
        out.append((v[0], v[1]))
    return out


def encode(Bx, By, scales, x, y):
    """Exactly _Enc.ctx_pos in 04_vsa_labels.py: mean over scales."""
    ps = [(Bx ** float(x / lx)) * (By ** float(y / ly)) for lx, ly in scales]
    return ps[0] if len(ps) == 1 else ps[0] + ps[1:]


def kernel(Bx, By, scales, dx, dy, ox=0.0, oy=0.0):
    """kappa measured from reference point (ox, oy) out along (dx, dy).

    The reference point matters, and that is the whole point. At the origin every
    scale gives Bx**0 = 1, so a k-scale bundle collapses to a SINGLE identical
    vector and self-similarity is trivially 1.0 — the one location where
    multi-scale looks free. Anywhere else the k components differ. Measuring only
    from the origin is what made the first version of this script report peak
    1.000 for every configuration.
    """
    a = encode(Bx, By, scales, ox, oy)
    n = len(dx)
    out = np.empty(n)
    for i in range(n):
        b = encode(Bx, By, scales, ox + float(dx[i]), oy + float(dy[i]))
        out[i] = a % b          # Phasor.__mod__ is similarity, already /dim
    return out


def half_width(offs, k, peak):
    """First offset at which the kernel drops below half its peak (metres)."""
    below = np.flatnonzero(k < peak / 2.0)
    if not len(below):
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    # linear interpolation between the straddling samples
    k0, k1 = k[i - 1], k[i]
    t = (k0 - peak / 2.0) / max(k0 - k1, 1e-12)
    return float(offs[i - 1] + t * (offs[i] - offs[i - 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=4096,
                    help="phasor dimension; 4096 complex64 is the 32 KB trace")
    ap.add_argument("--repeats", type=int, default=5,
                    help="encoder seeds to average over")
    ap.add_argument("--far", type=float, default=2.0,
                    help="offset (m) beyond which the kernel should be ~0; the "
                         "RMS out there is what we call clutter")
    ap.add_argument("--span", type=float, default=3.0,
                    help="how far out to profile, in metres")
    ap.add_argument("--steps", type=int, default=121)
    ap.add_argument("--refs", type=int, default=12,
                    help="reference points to measure the kernel FROM; the "
                         "spread across them tests shift-invariance")
    ap.add_argument("--npz", default=None,
                    help="optional path to save the profiles for plotting")
    args = ap.parse_args()

    offs = np.linspace(0.0, args.span, args.steps)
    zero = np.zeros_like(offs)
    far = offs >= args.far

    # reference points spread over a room-sized area. A shift-invariant kernel
    # gives the same profile from every one of them; the spread across them IS
    # the measurement.
    rng = np.random.RandomState(0)
    refs = np.column_stack([rng.uniform(0.0, 7.7, args.refs),
                            rng.uniform(0.0, 4.6, args.refs)])

    print(f"dim {args.dim}, {args.repeats} seeds, {args.refs} reference points "
          f"over a 7.7 x 4.6 m room, clutter beyond {args.far} m\n")
    print(f"{'length scale':<32}{'peak':>14}{'clutter':>9}{'peak/clut':>11}"
          f"{'wx (m)':>9}{'wy (m)':>9}   note")
    print("-" * 108)

    saved = {}
    for spec, note in CONFIGS:
        scales = parse_scales(spec)
        px, cl, wx, wy = [], [], [], []
        kx_acc = np.zeros_like(offs)
        ky_acc = np.zeros_like(offs)
        n_acc = 0
        for s in range(args.repeats):
            Bx = Phasor(dim=args.dim, seed=1000 + s)
            By = Phasor(dim=args.dim, seed=2000 + s)
            for ox, oy in refs:
                kx = kernel(Bx, By, scales, offs, zero, ox, oy)
                ky = kernel(Bx, By, scales, zero, offs, ox, oy)
                kx_acc += kx
                ky_acc += ky
                n_acc += 1
                px.append(kx[0])
                cl.append(np.sqrt(np.mean(
                    np.concatenate([kx[far], ky[far]]) ** 2)))
                wx.append(half_width(offs, kx, kx[0]))
                wy.append(half_width(offs, ky, ky[0]))
        peak, peak_sd = float(np.mean(px)), float(np.std(px))
        clut = float(np.mean(cl))
        print(f"{spec:<32}{peak:>8.3f} ±{peak_sd:<5.3f}{clut:>9.4f}"
              f"{peak/clut:>11.1f}{np.nanmean(wx):>9.2f}{np.nanmean(wy):>9.2f}"
              f"   {note}")
        saved[spec] = np.stack([offs, kx_acc / n_acc, ky_acc / n_acc])

    print("\npeak is kappa(x0, x0) — how strongly an observation matches its own "
          "location.\nA single scale is 1.000 everywhere by construction. For k "
          "bundled scales it is\n1/k on average, because only k of the k^2 cross "
          "terms are in phase.")
    print("\nThe +- on peak is the part that matters: it is the spread ACROSS "
          "reference\npoints. A single scale has none — the kernel is "
          "shift-invariant, so the same\ndisplacement means the same similarity "
          "anywhere in the room. A bundle's cross\nterms depend on x/l_i - x'/l_j, "
          "which is not a function of (x - x'), so its\nkernel changes shape with "
          "absolute position. For a spatial memory that is worse\nthan a merely "
          "weaker kernel: the metric itself stops being a metric.")

    if args.npz:
        os.makedirs(os.path.dirname(args.npz) or ".", exist_ok=True)
        np.savez_compressed(args.npz, **{k.replace("+", "__"): v
                                         for k, v in saved.items()})
        print(f"\nwrote profiles to {args.npz}")


if __name__ == "__main__":
    main()
