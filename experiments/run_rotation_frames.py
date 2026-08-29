"""E5: rotation, the gap E3 opened.

FINDINGS.md sec.16 E3 found the egocentric scene code exact under translation
and useless under rotation -- 2.24 m out after a 40 degree turn, no better than
the allocentric map it replaced.  The cause is not the object-vector idea but
the spatial code underneath it: a Cartesian SSP makes **translation** one bind,
and has no notion of rotation at all.

Renner et al. (sec.15) solve the same problem one manifold over, for image-plane
rotation, by working in **log-polar** coordinates -- there rotation and scaling
become translations, and therefore binds.  The same construction applies here,
and needs no new machinery: the angle is the periodic integer-harmonic FPE this
repo already has (sec.2), and the radius is an ordinary FPE of ``log r`` ::

    S_polar(v) = S_theta(atan2(v))  (*)  S_logr(log|v|)

    rotate by alpha  ->  bind by S_polar at (alpha, 0)      exact, one bind
    scale by s       ->  bind by S_polar at (0, log s)      exact, one bind
    translate by t   ->  **not** a bind

which is the whole trade, stated up front: Cartesian buys translation, log-polar
buys rotation and scale, and no single code buys both.  A robot does both, so
the result that matters is not which code wins but what it costs to carry the
pair.

Section [A] checks the algebra to machine precision before any decoding, in the
manner of sec.2 -- if rotation is not a bind to 1e-15 then nothing after it is
worth measuring.

    python experiments/run_rotation_frames.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import AtomVocab, bind, bundle, unbind  # noqa: E402
from sspslam.sspspace import HexagonalSSPSpace, SSPSpace, conjsym  # noqa: E402

R_MIN = 0.25          # log r is unbounded at the origin; clamp rather than nan


class LogPolarSSPSpace(SSPSpace):
    """FPE over ``(theta, log r)``: rotation and scale become binds.

    The angle axis uses **integer** harmonics, exactly as the view circle does
    (sec.2), so a full turn is the identity rather than merely close to it.
    The radial axis uses ordinary real-valued phases, because log-radius is not
    periodic -- there is no seam to close.
    """

    def __init__(self, ssp_dim=1015, max_harmonic=8, radial_scale=1.0,
                 rng=None):
        rng = np.random.default_rng() if rng is None else rng
        n = (int(ssp_dim) - 1) // 2
        theta = np.tile(np.arange(1, max_harmonic + 1, dtype=float),
                        int(np.ceil(n / max_harmonic)))[:n]
        radial = rng.standard_normal(n) / float(radial_scale)
        phases = np.stack([theta, radial], axis=1)
        pm = conjsym(phases)
        super().__init__(2, pm.shape[0], phase_matrix=pm,
                         domain_bounds=np.array([[-np.pi, np.pi],
                                                 [np.log(R_MIN), np.log(20.0)]]),
                         length_scale=1, rng=rng)

    def encode_xy(self, v):
        v = np.atleast_2d(np.asarray(v, dtype=float))
        r = np.maximum(np.linalg.norm(v, axis=1), R_MIN)
        th = np.arctan2(v[:, 1], v[:, 0])
        out = self.encode(np.stack([th, np.log(r)], axis=1))
        return out[0] if out.shape[0] == 1 else out

    def rotation(self, alpha):
        """The vector that rotates any encoded point by ``alpha``."""
        return self.encode(np.array([[float(alpha), 0.0]])).reshape(-1)

    def scaling(self, s):
        return self.encode(np.array([[0.0, float(np.log(s))]])).reshape(-1)


def rot(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ssp-dim", type=int, default=1015)
    ap.add_argument("--n-objects", type=int, default=6)
    ap.add_argument("--bound", type=float, default=5.0)
    ap.add_argument("--length-scale", type=float, default=0.6)
    ap.add_argument("--max-harmonic", type=int, default=8)
    ap.add_argument("--radial-scale", type=float, default=0.35)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--turn-deg", type=float, default=40.0)
    args = ap.parse_args()

    print(f"log-polar SSP: integer harmonics up to {args.max_harmonic} on the "
          f"angle, real phases on log r\n")

    # ---- [A] the algebra, before any decoding -------------------------------
    sp = LogPolarSSPSpace(args.ssp_dim, args.max_harmonic, args.radial_scale,
                          rng=np.random.default_rng(0))
    cart = HexagonalSSPSpace(2, ssp_dim=args.ssp_dim,
                             domain_bounds=np.array([[-args.bound,
                                                      args.bound]] * 2),
                             length_scale=args.length_scale,
                             rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    v = rng.uniform(-4, 4, (200, 2))
    v = v[np.linalg.norm(v, axis=1) > R_MIN]
    a = np.deg2rad(args.turn_deg)

    print("[A] is the transform exactly one bind?  (max abs difference)")
    e = np.abs(sp.encode_xy(v @ rot(a).T)
               - bind(sp.encode_xy(v), sp.rotation(a))).max()
    print(f"  log-polar, rotate {args.turn_deg:.0f} deg : {e:.2e}")
    e = np.abs(sp.encode_xy(v * 1.7) - bind(sp.encode_xy(v),
                                            sp.scaling(1.7))).max()
    print(f"  log-polar, scale by 1.7      : {e:.2e}")
    e = np.abs(sp.encode_xy(v) - bind(sp.encode_xy(v),
                                      sp.rotation(2 * np.pi))).max()
    print(f"  log-polar, a full turn       : {e:.2e}   (integer harmonics)")
    t = np.array([1.0, 0.5])
    e = np.abs(cart.encode(v + t) - bind(cart.encode(v),
                                         cart.encode(t).reshape(-1))).max()
    print(f"  Cartesian, translate         : {e:.2e}")
    e = np.abs(cart.encode(v @ rot(a).T) - cart.encode(v)).max()
    print(f"  Cartesian, rotate            : {e:.2e}   (no bind exists)")
    print("  the first three are the point: rotation and scale are binds in "
          "log-polar,\n  to machine precision, and a full turn is the "
          "identity because the angle\n  axis uses integer harmonics.")

    # ---- [B] the scene test, E3's table with rotation added ----------------
    step = 0.1
    axis = np.arange(-args.bound, args.bound + 1e-9, step)
    pts = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    keep = np.linalg.norm(pts, axis=1) > R_MIN
    pts = pts[keep]

    conds = ("same", "translated", "rotated", "turn+move")
    err = {(c, w): [] for c in ("cartesian", "log-polar", "both") for w in conds}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        sp = LogPolarSSPSpace(args.ssp_dim, args.max_harmonic,
                              args.radial_scale, rng=np.random.default_rng(seed))
        cart = HexagonalSSPSpace(2, ssp_dim=args.ssp_dim,
                                 domain_bounds=np.array([[-args.bound,
                                                          args.bound]] * 2),
                                 length_scale=args.length_scale,
                                 rng=np.random.default_rng(seed))
        gp = np.stack([sp.encode_xy(p) for p in pts])
        gc = np.stack([cart.encode(p).reshape(-1) for p in pts])
        vocab = AtomVocab(args.ssp_dim, seed=seed + 500)
        names = [f"obj_{i}" for i in range(args.n_objects)]

        robot = np.zeros(2)
        pos = rng.uniform(-args.bound + 1, args.bound - 1, (args.n_objects, 2))
        pos = pos[np.linalg.norm(pos, axis=1) > 1.0]
        names = names[:len(pos)]
        vec = pos - robot
        mem_p = bundle(np.stack([bind(vocab.mint(n), sp.encode_xy(v))
                                 for n, v in zip(names, vec)]))
        mem_c = bundle(np.stack([bind(vocab[n], cart.encode(v).reshape(-1))
                                 for n, v in zip(names, vec)]))

        t = np.array([args.shift, 0.0])
        transforms = {
            "same": (np.zeros(2), 0.0),
            "translated": (t, 0.0),
            "rotated": (np.zeros(2), a),
            "turn+move": (t, a),
        }
        for cond, (tr, al) in transforms.items():
            # world seen from a robot that moved by tr and turned by al
            new_vec = (vec - tr) @ rot(-al).T
            mp = bind(mem_p, sp.rotation(-al)) if al else mem_p
            mc = (bind(mem_c, cart.encode(-tr).reshape(-1))
                  if np.any(tr) else mem_c)
            # 'both': translate in Cartesian, rotate in log-polar, by
            # re-encoding the decoded points between the two frames
            for code, mem, g, dec in (("cartesian", mc, gc, cart),
                                      ("log-polar", mp, gp, sp)):
                got = np.stack([pts[int(np.argmax(g @ unbind(mem, vocab[n])))]
                                for n in names])
                err[(code, cond)].append(np.linalg.norm(got - new_vec, axis=1))
            got_c = np.stack([pts[int(np.argmax(gc @ unbind(mc, vocab[n])))]
                              for n in names])
            got = got_c @ rot(-al).T if al else got_c
            err[("both", cond)].append(np.linalg.norm(got - new_vec, axis=1))

    print(f"\n[B] vector to each object after the robot moves and turns "
          f"(median error, {args.seeds} seeds)")
    print(f"  {'code':>10s} " + " ".join(f"{c:>12s}" for c in conds))
    print("  " + "-" * 62)
    for code in ("cartesian", "log-polar", "both"):
        row = [f"{np.median(np.concatenate(err[(code, c)])):9.2f} m"
               for c in conds]
        print(f"  {code:>10s} " + " ".join(f"{v:>12s}" for v in row))
    print(f"  translate {args.shift:.0f} m, turn {args.turn_deg:.0f} deg. "
          f"'both' keeps the Cartesian memory and\n  rotates the decoded "
          f"points, which is the cheap pairing -- one memory, one\n  2x2 "
          f"matrix, and no second bundle to keep consistent.")
    print("  read the diagonal: each single code is exact for its own "
          "transform and\n  fails for the other, which is the trade stated "
          "as a measurement.")


if __name__ == "__main__":
    main()
