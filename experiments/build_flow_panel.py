"""Regenerate the flow, gap and scoreboard data in docs/view_circle.html.

Companion to ``build_view_circle_page.py``.  That one fills the
``<script id="viewdata">`` block, which is the original diagnostic at the
settings sec.4-13 were measured with; this one fills ``<script id="flowdata">``,
which carries what sec.16 added -- the optical-flow view of the orbit, the
error gradient inside a held-out arc, and the ``max_harmonic`` sweep that
corrects E0.

Everything here is computed at the **corrected** settings (``ssp_dim=2401``,
``max_harmonic=4``, K=12, sec.16 E2) rather than the repo defaults, and the
page says so where the two disagree.

Display resolution stays separate from analysis resolution, for the reason in
``build_view_circle_page``'s docstring: the tiles come from a 224px render, the
descriptors and the flow always from the 96px render the experiments use.

    python experiments/build_flow_panel.py
"""
import base64
import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from PIL import Image  # noqa: E402

from sspslam.objectmap import (CircularSSPSpace, bind, bundle,  # noqa: E402
                               cosine, wrap_angle)
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import encode_hog, fit_basis, condition  # noqa: E402
from run_blocked_split import blocked_masks  # noqa: E402
from run_nn_baseline import pick_store, decode_vsa, decode_nearest, DROP  # noqa: E402
from run_optical_flow import warp  # noqa: E402
from run_view_tracking import predict, readout  # noqa: E402

DIM, KMAX, K_STORED, GAP, NV, SEEDS = 2401, 4, 12, 30.0, 72, 3
TILE, ARROWS, NB = 112, 8, 180
SIGMA, BETA = 0.12, 60.0
PAGE = os.path.join(os.path.dirname(HERE), "docs", "view_circle.html")


def heat(x):
    """A dark-to-hot ramp for the residual maps: black, ember, gold, white."""
    stops = np.array([[0.06, 0.09, 0.09], [0.44, 0.14, 0.06],
                      [0.85, 0.42, 0.10], [0.98, 0.86, 0.55],
                      [1.0, 1.0, 0.96]])
    t = np.clip(x, 0.0, 1.0) * (len(stops) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(stops) - 2)
    f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def crop_box(frames, pad=1.14):
    m = frames.sum(-1) > 0.04
    rows, cols = np.where(m.any(0).any(1))[0], np.where(m.any(0).any(0))[0]
    cy, cx = (rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2
    half = max(rows[-1] - rows[0], cols[-1] - cols[0]) / 2 * pad + 3
    R = frames.shape[1]
    return (int(max(cy - half, 0)), int(min(cy + half, R)),
            int(max(cx - half, 0)), int(min(cx + half, R)))


def sheet_b64(tiles, quality=74):
    sheet = Image.new("RGB", (TILE * len(tiles), TILE))
    for j, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t).resize((TILE, TILE), Image.LANCZOS),
                    (TILE * j, 0))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    from skimage.registration import optical_flow_ilk

    imgs, obj, az, names = load_turntable(n_views=NV, extra="symmetric")
    big, _, _, _ = load_turntable(n_views=NV, res=224, extra="symmetric")
    Z = encode_hog(imgs)
    gray = imgs.mean(-1)
    vi = np.concatenate([np.arange(NV)] * len(names))
    kept, held = blocked_masks(vi, NV, GAP)
    mu, V, lam = fit_basis(Z[kept])
    def keys_for(seed):
        W = (np.random.default_rng(seed).standard_normal((V.shape[0], DIM))
             / np.sqrt(V.shape[0]))
        return condition(Z, mu, V, lam, W, drop=DROP)

    def space_for(seed, kmax=KMAX):
        return CircularSSPSpace(1, ssp_dim=DIM, max_harmonic=kmax,
                                rng=np.random.default_rng(seed + 1000))

    def book_for(ks, sp, o, K=K_STORED):
        sel = np.where((obj == o) & kept)[0]
        pick = sel[pick_store(az[sel], K)]
        return pick, bundle(np.stack([bind(k, sp.encode([ang]))
                                      for k, ang in zip(ks[pick], az[pick])]))

    # the animation shows ONE run; the aggregates below pool SEEDS of them
    keys = keys_for(0)
    vs = space_for(0)
    grid = np.linspace(-np.pi, np.pi, NB, endpoint=False)
    kfreq = np.fft.fftfreq(NB, d=1.0 / NB)
    step = 2 * np.pi / NV

    out = {"nv": NV, "tile": TILE, "arrows": ARROWS, "nb": NB, "gap": GAP,
           "dim": DIM, "kmax": KMAX, "K": K_STORED,
           "lobe": round(float(np.rad2deg(vs.lobe_width())), 1), "objects": []}

    for o, nm in enumerate(names):
        order = np.where(obj == o)[0][np.argsort(az[obj == o])]
        a = az[order]
        y0, y1, x0, x1 = crop_box(big[order])
        gy0, gy1, gx0, gx1 = crop_box(imgs[order])

        # --- optical flow around the orbit -----------------------------------
        res_tiles, arrows, resid, magn = [], [], [], []
        for j in range(NV):
            p, c = gray[order[(j - 1) % NV]], gray[order[j]]
            v, u = optical_flow_ilk(p, c, radius=5)
            d = np.abs(warp(p, v, u) - c)
            resid.append(float(d.mean()))
            magn.append(float(np.hypot(v, u).mean()))
            res_tiles.append(d[gy0:gy1, gx0:gx1])
            s = max((gy1 - gy0) // ARROWS, 1)
            vv = v[gy0:gy1, gx0:gx1][::s, ::s][:ARROWS, :ARROWS]
            uu = u[gy0:gy1, gx0:gx1][::s, ::s][:ARROWS, :ARROWS]
            pad = ((0, ARROWS - vv.shape[0]), (0, ARROWS - vv.shape[1]))
            arrows.append([[round(float(x), 2) for x in
                            np.pad(uu, pad).ravel()],
                           [round(float(x), 2) for x in
                            np.pad(vv, pad).ravel()]])
        hi = max(np.percentile(np.concatenate([t.ravel() for t in res_tiles]),
                               99.5), 1e-6)
        res_sheet = sheet_b64([(heat(t / hi) * 255).astype(np.uint8)
                               for t in res_tiles])
        obj_sheet = sheet_b64([(big[i][y0:y1, x0:x1] * 255).astype(np.uint8)
                               for i in order], quality=78)

        # --- localisation at the corrected settings --------------------------
        pick, book = book_for(keys, vs, o)
        proto = bundle(keys[pick])
        fields = np.stack([vs.view_likelihood(book, k, n_per_dim=NB)[1]
                           for k in keys[order]])
        raw = grid[fields.argmax(1)]
        err = np.abs(np.rad2deg(wrap_angle(raw - a)))

        # --- the same orbit, filtered (sec.12) -------------------------------
        b = np.full(NB, 1.0 / NB)
        filt, prev, beliefs = np.empty(NV), None, []
        for t, L in enumerate(fields):
            if t > 0:
                b = np.maximum(np.real(np.fft.ifft(
                    predict(np.fft.fft(b), step, SIGMA, kfreq))), 0)
                b /= max(b.sum(), 1e-12)
            b = b * np.exp(BETA * (L - L.max()))
            s = b.sum()
            b = b / s if s > 1e-300 else np.full(NB, 1.0 / NB)
            beliefs.append([round(float(x), 5) for x in b / b.max()])
            prev = readout(b, grid, prev)
            filt[t] = prev
        ferr = np.abs(np.rad2deg(wrap_angle(filt - a)))

        out["objects"].append(dict(
            name=nm, sprite=obj_sheet, residual=res_sheet, arrows=arrows,
            angles=[round(float(x), 1) for x in np.rad2deg(a)],
            resid=[round(x, 5) for x in resid],
            mag=[round(x, 3) for x in magn],
            err=[round(float(x), 1) for x in err],
            ferr=[round(float(x), 1) for x in ferr],
            est=[round(float(x), 1) for x in np.rad2deg(raw)],
            filt=[round(float(x), 1) for x in np.rad2deg(filt)],
            belief=beliefs,
            stored=[bool(x) for x in kept[order]],
            proto_sim=[round(float(cosine(proto, keys[i].reshape(1, -1))[0]), 3)
                       for i in order]))

    # --- error against distance into the held-out arc (sec.16 E7) ------------
    gsteps = max(int(round(GAP / (360.0 / NV))), 1)
    depth = {}
    for seed in range(SEEDS):
      ks, sp = keys_for(seed), space_for(seed)
      for o in range(len(names)):
        m = np.where(held & (obj == o))[0]
        pick, book = book_for(ks, sp, o)
        e, _ = decode_vsa(book, ks[m], sp)
        e = np.abs(np.rad2deg(wrap_angle(e - az[m])))
        n, _ = decode_nearest(ks[pick], az[pick], ks[m])
        n = np.abs(np.rad2deg(wrap_angle(n - az[m])))
        for idx, ev, nv_ in zip(m, e, n):
            pos = (idx % NV) % (2 * gsteps)
            d = min(pos, gsteps - 1 - pos) + 1
            depth.setdefault(d, {"vsa": [], "list": []})
            depth[d]["vsa"].append(float(ev))
            depth[d]["list"].append(float(nv_))
    out["depth"] = [dict(steps=int(d), deg=round(d * 360.0 / NV, 1),
                         n=int(len(depth[d]["vsa"])),
                         vsa=round(float(np.median(depth[d]["vsa"])), 1),
                         lst=round(float(np.median(depth[d]["list"])), 1),
                         within=round(float(np.mean(
                             np.array(depth[d]["vsa"]) < 15)), 2))
                    for d in sorted(depth)]

    # --- occlusion quartiles, on this page's own objects (sec.16 E7) --------
    # Computed here rather than quoted from FINDINGS so that the headline and
    # the chart beside it come from the same six objects.  The document's own
    # figures are over ten, and the page says so.
    from run_frontend_diagnostics import diagnose
    dg = diagnose(keys, obj, az, names, NV, np.rad2deg(vs.lobe_width()))
    zs, es = [], []
    for seed in range(SEEDS):
        ks, sp = keys_for(seed), space_for(seed)
        for o in range(len(names)):
            if dg[o]["alias_peak"] >= 0.75:
                continue
            order = np.where(obj == o)[0][np.argsort(az[obj == o])]
            m = ~kept[order]
            _, book = book_for(ks, sp, o)
            d, _ = decode_vsa(book, ks[order[m]], sp)
            r = np.array(out["objects"][o]["resid"])
            zs.append((r[m] - r.mean()) / max(r.std(), 1e-12))
            es.append(np.abs(np.rad2deg(wrap_angle(d - az[order[m]]))))
    zs, es = np.concatenate(zs), np.concatenate(es)
    qcut = np.percentile(zs, [0, 25, 50, 75, 100])
    quart = []
    for i in range(4):
        sel_q = (zs >= qcut[i]) & (zs <= qcut[i + 1])
        quart.append(dict(label=["smoothest quarter", "second", "third",
                                 "most occluded quarter"][i],
                          n=int(sel_q.sum()),
                          err=round(float(np.median(es[sel_q])), 1)))
    out["occ"] = dict(quartiles=quart,
                      r=round(float(np.corrcoef(zs, es)[0, 1]), 2),
                      n_obj=int(len(names) - sum(d["alias_peak"] >= 0.75
                                                 for d in dg)),
                      spread=round(quart[-1]["err"] - quart[0]["err"], 1))

    # --- the max_harmonic sweep that corrects E0 (sec.16 E2) ----------------
    # On the ORIGINAL six objects, which is the set sec.16 E2 measured.  The
    # four by-construction symmetric extras sit at chance for every setting,
    # so including them would compress the very contrast this table is for.
    base = [i for i, nm in enumerate(names)
            if nm in ("chair", "mug", "cube", "L_block", "pot", "console")]
    sweep = []
    for kmax in (4, 8, 12, 16, 24, 48):
        row = {"kmax": kmax}
        row["lobe"] = round(float(np.rad2deg(
            space_for(0, kmax).lobe_width())), 1)
        for K in (6, 12, 24, 36):
            e = []
            for seed in range(SEEDS):
                ks, sp = keys_for(seed), space_for(seed, kmax)
                for o in base:
                    _, bk = book_for(ks, sp, o, K)
                    m = held & (obj == o)
                    d, _ = decode_vsa(bk, ks[m], sp)
                    e.append(np.abs(np.rad2deg(wrap_angle(d - az[m]))))
            row[f"K{K}"] = round(float(np.median(np.concatenate(e))), 1)
        sweep.append(row)
    lst = {}
    for K in (6, 12, 24, 36):
        e = []
        for seed in range(SEEDS):
            ks = keys_for(seed)
            for o in base:
                sel = np.where((obj == o) & kept)[0]
                pick = sel[pick_store(az[sel], K)]
                m = held & (obj == o)
                d, _ = decode_nearest(ks[pick], az[pick], ks[m])
                e.append(np.abs(np.rad2deg(wrap_angle(d - az[m]))))
        lst[f"K{K}"] = round(float(np.median(np.concatenate(e))), 1)
    out["seeds"] = SEEDS
    out["sweep_objects"] = [names[i] for i in base]
    out["sweep"] = sweep
    out["sweep_list"] = lst

    blob = json.dumps(out, separators=(",", ":"), default=float)
    page = open(PAGE).read()
    if 'id="flowdata"' not in page:
        raise SystemExit("docs/view_circle.html has no flowdata block yet")
    page = re.sub(r'(<script id="flowdata" type="application/json">).*?(</script>)',
                  lambda m: m.group(1) + blob + m.group(2), page, flags=re.S)
    open(PAGE, "w").write(page)
    print(f"flowdata written — {len(blob)/1024:.0f} KB, page now "
          f"{len(page)/1024:.0f} KB")
    print("  depth gradient:",
          " ".join(f"{d['deg']:.0f}d={d['vsa']}" for d in out["depth"]))
    print("  occlusion quartiles:",
          " ".join(str(q["err"]) for q in out["occ"]["quartiles"]),
          f"(spread {out['occ']['spread']}, r={out['occ']['r']})")
    print("  best kmax at K=12:",
          min(sweep, key=lambda r: r["K12"])["kmax"],
          "| list", lst["K12"])


if __name__ == "__main__":
    main()
