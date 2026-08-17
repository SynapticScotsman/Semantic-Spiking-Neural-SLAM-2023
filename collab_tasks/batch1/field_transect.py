"""SHOW the loss in the cogmap's own terms: the field, not a percentage.

Everything we have said about the gap so far -- "bleed", "sub-lambda
resolution" -- has been reported as a share of cells. That is not the VSA
answer. The VSA answer is the FIELD: at the cell we get wrong, what do
F_gt(u,v) and F_winner(u,v) actually look like, how wide is the kernel that
produced them, and where do the two curves cross?

This extracts, for a real losing cell:

  1. the FPE similarity kernel itself, measured (not assumed) from the same
     Bx/By the trace uses -- k(d) = Re<phi(0), phi(d)>/HD. This is the
     resolution limit, drawn.
  2. a transect: both class fields sampled along the line joining the GT
     class's nearest observation to the winner's nearest observation, through
     the losing cell, with the observation positions and the crossing point
     marked.
  3. the two 2D fields around that cell, as a small grid, so the shapes are
     visible rather than described.

Writes outputs/batch1/field_transect.json (numbers + SVG-ready series).

    python collab_tasks/batch1/field_transect.py --pair sofa cushion
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from collab_tasks.batch1.common import (  # noqa: E402
    GRID, HD, LX, LY, SEEDS, _Enc, cap_per_class, default_fields, load_scene,
    predict)


def kernel_curve(base_seed=SEEDS[0][0], lx=LX, dmax=1.2, n=240):
    """Measured FPE kernel along x: k(d) = Re<phi(0,0), phi(d,0)>/HD.

    This is the quantity that decides whether two observations at 4 cm and
    15 cm from a cell can be told apart. Measured from the same encoder the
    trace is built with, not from the sinc idealisation.
    """
    enc = _Enc(HD, base_seed, lx, LY)
    Bx = enc.Bx.values
    ds = np.linspace(0, dmax, n)
    k = np.array([float(np.real(np.vdot(Bx ** 0.0, Bx ** (d / lx))) / HD)
                  for d in ds])
    # half-width at half maximum -- the honest "resolution" number
    hi = np.flatnonzero(k < 0.5)
    hwhm = float(ds[hi[0]]) if len(hi) else float("nan")
    return ds, k, hwhm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--pair", nargs=2, default=["sofa", "cushion"],
                    help="GT class we should win, class that actually wins")
    ap.add_argument("--tuple", type=int, default=0)
    args = ap.parse_args()
    gtc, winc = args.pair

    data = load_scene(args.scene)
    F, names, cell = default_fields(data, args.tuple)
    F = np.asarray(F)
    gt, xyz, a, b = data["gt"], data["xyz"], data["a"], data["b"]
    pred = predict(F, names, cell)
    gx = np.linspace(xyz[:, a].min(), xyz[:, a].max(), GRID)
    gy = np.linspace(xyz[:, b].min(), xyz[:, b].max(), GRID)

    # the capped stream is what the trace actually contains
    capped = cap_per_class(list(data["obs"]), 400, seed=SEEDS[args.tuple][2])
    obs = {}
    for o in capped:
        obs.setdefault(o["cls"], []).append((o["x"], o["y"]))
    obs = {c: np.array(v) for c, v in obs.items()}

    # pick the losing cell: gt==gtc, pred==winc, and GT's own point NEAREST
    # (i.e. from the 38% branch -- the cells that indict the decode)
    cand = np.flatnonzero((gt == gtc) & (pred == winc))
    if not len(cand):
        raise SystemExit(f"no cell where {gtc} lost to {winc} in {args.scene}")
    rows = []
    for i in cand:
        px, py = float(xyz[i, a]), float(xyz[i, b])
        dg = float(np.hypot(*(obs[gtc] - [px, py]).T).min())
        dw = float(np.hypot(*(obs[winc] - [px, py]).T).min())
        if dg < dw:
            rows.append((i, px, py, dg, dw))
    if not rows:
        raise SystemExit(f"{gtc}->{winc} has no GT-nearer cells; "
                         "that pair is in the 62% branch")
    # the most typical of them: median d_gt
    rows.sort(key=lambda r: r[3])
    i, px, py, dg, dw = rows[len(rows) // 2]
    print(f"{len(cand)} cells lost {gtc}->{winc}; {len(rows)} with GT nearer")
    print(f"chosen cell: ({px:.2f}, {py:.2f})  d({gtc})={dg:.3f} m  "
          f"d({winc})={dw:.3f} m")

    ig, iw = names.index(gtc), names.index(winc)
    Fg = F[ig].reshape(GRID, GRID)      # [iy, ix]
    Fw = F[iw].reshape(GRID, GRID)

    # ---- transect: through the cell, along GT-nearest -> winner-nearest ----
    pg = obs[gtc][np.hypot(*(obs[gtc] - [px, py]).T).argmin()]
    pw = obs[winc][np.hypot(*(obs[winc] - [px, py]).T).argmin()]
    d = pw - pg
    L = float(np.hypot(*d))
    u = d / max(L, 1e-9)
    # extend a little past both observation points
    t = np.linspace(-0.35, L + 0.35, 200)
    P = pg[None, :] + t[:, None] * u[None, :]

    def sample(Fc):
        ix = np.clip(np.searchsorted(gx, P[:, 0]), 0, GRID - 1)
        iy = np.clip(np.searchsorted(gy, P[:, 1]), 0, GRID - 1)
        return Fc[iy, ix]

    sg, sw = sample(Fg), sample(Fw)
    # where does the winner overtake, measured along the transect
    over = np.flatnonzero(sw > sg)
    cross = float(t[over[0]]) if len(over) else float("nan")
    t_cell = float(np.dot([px, py] - pg, u))

    ds, kk, hwhm = kernel_curve()
    print(f"kernel HWHM = {hwhm:.3f} m  (lambda_x = {LX})")
    print(f"transect: |{gtc} obs -> {winc} obs| = {L:.3f} m, "
          f"cell at t={t_cell:.3f}, winner overtakes at t={cross:.3f}")
    print(f"field at the cell: {gtc} {Fg.reshape(-1)[cell[i]]:.4f}  "
          f"{winc} {Fw.reshape(-1)[cell[i]]:.4f}  "
          f"margin {Fw.reshape(-1)[cell[i]] - Fg.reshape(-1)[cell[i]]:+.4f}")

    # how many observations of each within one kernel width of the cell
    n_g = int((np.hypot(*(obs[gtc] - [px, py]).T) <= LX).sum())
    n_w = int((np.hypot(*(obs[winc] - [px, py]).T) <= LX).sum())
    print(f"obs within one lambda ({LX} m): {gtc} {n_g}, {winc} {n_w}")

    # ---- local 2D patch of both fields, for the heatmap pair ----
    ix0 = int(np.clip(np.searchsorted(gx, px) - 12, 0, GRID - 25))
    iy0 = int(np.clip(np.searchsorted(gy, py) - 12, 0, GRID - 25))
    patch = dict(
        x0=float(gx[ix0]), x1=float(gx[ix0 + 24]),
        y0=float(gy[iy0]), y1=float(gy[iy0 + 24]),
        gt=Fg[iy0:iy0 + 25, ix0:ix0 + 25].tolist(),
        win=Fw[iy0:iy0 + 25, ix0:ix0 + 25].tolist())

    out = dict(
        scene=args.scene, gt_class=gtc, win_class=winc, tuple=args.tuple,
        cell=dict(x=px, y=py, d_gt=dg, d_win=dw,
                  f_gt=float(Fg.reshape(-1)[cell[i]]),
                  f_win=float(Fw.reshape(-1)[cell[i]]),
                  n_obs_gt_1lambda=n_g, n_obs_win_1lambda=n_w),
        n_cells_lost=int(len(cand)), n_cells_gt_nearer=len(rows),
        kernel=dict(lambda_x=LX, hwhm=hwhm,
                    d=ds.round(4).tolist(), k=kk.round(5).tolist(),
                    k_at_4cm=float(np.interp(0.04, ds, kk)),
                    k_at_15cm=float(np.interp(0.15, ds, kk)),
                    k_at_26cm=float(np.interp(0.26, ds, kk))),
        transect=dict(t=t.round(4).tolist(),
                      f_gt=sg.round(5).tolist(), f_win=sw.round(5).tolist(),
                      L=L, t_cell=t_cell, t_cross=cross,
                      p_gt=pg.round(3).tolist(), p_win=pw.round(3).tolist()),
        patch=patch)
    os.makedirs("outputs/batch1", exist_ok=True)
    json.dump(out, open("outputs/batch1/field_transect.json", "w"))
    print("\nwrote outputs/batch1/field_transect.json")
    print(f"kernel value at 4 cm {out['kernel']['k_at_4cm']:.4f}, "
          f"15 cm {out['kernel']['k_at_15cm']:.4f}, "
          f"26 cm {out['kernel']['k_at_26cm']:.4f}"
          "  <- how little the kernel changes over the distances that decide "
          "these cells")


if __name__ == "__main__":
    main()
