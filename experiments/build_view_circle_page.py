"""Regenerate the data embedded in docs/view_circle.html.

The page is committed built, so it can be opened straight from the repo, but it
is not a mystery blob: everything inside its `<script id="viewdata">` block is
produced here from the same experiment code that produces the tables in
FINDINGS.md.  Run this after changing the encoder, the split or the filter and
the page follows.

    python experiments/build_view_circle_page.py

Display resolution is deliberately separate from analysis resolution: the tiles
are cropped from a 224px render, the descriptors always come from the 96px
render the committed experiments use.  Letting the display resolution reach the
descriptor silently moved one object's error by 47 degrees once already.
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

from sspslam.objectmap import CircularSSPSpace, wrap_angle  # noqa: E402
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import (encode_hog, fit_basis, condition,  # noqa: E402
                                   build_books)
from run_blocked_split import blocked_masks  # noqa: E402
from run_view_tracking import predict, readout  # noqa: E402

SSP_DIM, MAXH, GAP, DROP = 151, 8, 30.0, 2
NB_FIELD, NB_TRACK, TILE = 120, 360, 112
SIGMA, BETA, ODO, ANCHOR = 2.0, 3.0, 1.5, 10.0
PAGE = os.path.join(os.path.dirname(HERE), "docs", "view_circle.html")


def main():
    imgs, obj, az, names = load_turntable(n_views=72)          # analysis
    big, _, _, _ = load_turntable(n_views=72, res=224)         # display only
    Z = encode_hog(imgs)
    nv = 72
    vi = np.concatenate([np.arange(nv)] * len(names))
    vs = CircularSSPSpace(1, ssp_dim=SSP_DIM, max_harmonic=MAXH,
                          rng=np.random.default_rng(1))
    kept, _ = blocked_masks(vi, nv, GAP)
    mu, V, lam = fit_basis(Z[kept])
    W = (np.random.default_rng(0).standard_normal((V.shape[0], SSP_DIM))
         / np.sqrt(V.shape[0]))
    keys = condition(Z, mu, V, lam, W, drop=DROP)
    books = build_books(keys, obj, az, kept, len(names), vs)

    grid_f = np.linspace(-180, 180, NB_FIELD, endpoint=False)
    grid_t = np.linspace(-np.pi, np.pi, NB_TRACK, endpoint=False)
    kfreq = np.fft.fftfreq(NB_TRACK, d=1.0 / NB_TRACK)
    step = 2 * np.pi / nv
    thr = 3 * np.rad2deg(step)
    data = {"lobe": round(float(np.rad2deg(vs.lobe_width())), 1), "gap": GAP,
            "nb": NB_FIELD, "tile": TILE, "objects": [],
            "track_cfg": {"sigma": SIGMA, "beta": BETA, "odo": ODO,
                          "anchor": ANCHOR, "thr": round(thr)}}
    summary = {"perframe": {}, "free": {}, "anchor": {}}

    for o, nm in enumerate(names):
        order = np.where(obj == o)[0][np.argsort(az[obj == o])]
        a = az[order]

        m = big[order].sum(-1) > 0.04
        rows, cols = np.where(m.any(0).any(1))[0], np.where(m.any(0).any(0))[0]
        cy, cx = (rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2
        half = max(rows[-1] - rows[0], cols[-1] - cols[0]) / 2 * 1.14 + 3
        R = big.shape[1]
        y0, y1 = int(max(cy - half, 0)), int(min(cy + half, R))
        x0, x1 = int(max(cx - half, 0)), int(min(cx + half, R))
        sheet = Image.new("RGB", (TILE * nv, TILE))
        for j, ix in enumerate(order):
            crop = Image.fromarray((big[ix][y0:y1, x0:x1] * 255).astype(np.uint8))
            sheet.paste(crop.resize((TILE, TILE), Image.LANCZOS), (TILE * j, 0))
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=78, optimize=True)

        fields, errs, peaks, ests = [], [], [], []
        for ang, k in zip(a, keys[order]):
            _, f = vs.view_likelihood(books[o], k, n_per_dim=NB_FIELD)
            est = float(grid_f[int(np.argmax(f))])
            ests.append(round(est, 1))
            errs.append(round(float(abs(np.rad2deg(
                wrap_angle(np.deg2rad(est - np.rad2deg(ang)))))), 1))
            thr8 = 0.8 * f.max()
            cand = np.where((f >= thr8) & (f >= np.roll(f, 1))
                            & (f >= np.roll(f, -1)))[0]
            ch = []
            for c in cand[np.argsort(f[cand])[::-1]]:
                if all(abs((c - p + NB_FIELD // 2) % NB_FIELD - NB_FIELD // 2)
                       * 360 / NB_FIELD > 2 * data["lobe"] for p in ch):
                    ch.append(c)
            peaks.append(len(ch))
            fields.append([round(float(v), 3) for v in f])

        tfields = np.stack([vs.view_likelihood(books[o], keys[i],
                                               n_per_dim=NB_TRACK)[1]
                            for i in order])
        raw = np.rad2deg(grid_t[np.argmax(tfields, 1)])
        track = {"true": [round(float(x), 1) for x in np.rad2deg(a)],
                 "raw": [round(float(x), 1) for x in raw]}
        for label, anchored in (("free", False), ("anchor", True)):
            r = np.random.default_rng(0)
            odo = step + r.normal(0, np.deg2rad(ODO), len(a))
            b = (np.exp(np.cos(grid_t - a[0]) / np.deg2rad(ANCHOR) ** 2)
                 if anchored else np.full(NB_TRACK, 1.0))
            b = b / b.sum()
            est = np.empty(len(a))
            prev = None
            for t, L in enumerate(tfields):
                if t > 0:
                    b = np.maximum(np.real(np.fft.ifft(predict(
                        np.fft.fft(b), odo[t], np.deg2rad(SIGMA), kfreq))), 0)
                    b /= max(b.sum(), 1e-12)
                b = b * np.exp(BETA * (L - L.max()))
                s = b.sum()
                b = b / s if s > 1e-300 else np.full(NB_TRACK, 1.0 / NB_TRACK)
                est[t] = readout(b, grid_t, prev, True)
                prev = est[t]
            track[label] = [round(float(x), 1) for x in np.rad2deg(est)]
        for key, series in (("perframe", raw), ("free", np.array(track["free"])),
                            ("anchor", np.array(track["anchor"]))):
            e = np.abs(np.rad2deg(wrap_angle(np.deg2rad(series - np.rad2deg(a)))))
            j = np.abs(np.rad2deg(wrap_angle(np.deg2rad(np.diff(series)))))
            summary[key][nm] = {"med": round(float(np.median(e)), 1),
                                "maxjump": round(float(j.max())),
                                "imp": int((j > thr).sum()), "n": int(len(j))}
        # identity-pedestal decomposition (FINDINGS.md section 14)
        Zo = Z[order] - Z.mean(0)
        a_o = Zo.mean(0)
        Vv = Zo - a_o
        lam = float((a_o @ a_o) / ((a_o @ a_o) + np.mean(np.sum(Vv ** 2, axis=1))))

        def _acf(X):
            U = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            return [round(float(np.mean(np.sum(U * np.roll(U, -d, axis=0), axis=1))), 4)
                    for d in range(len(X))]
        pedestal = dict(lam=round(lam, 4), rho=_acf(Zo), r=_acf(Vv))

        data["objects"].append(dict(
            name=nm, sprite=base64.b64encode(buf.getvalue()).decode(),
            angles=[round(float(x), 1) for x in np.rad2deg(a)],
            est=ests, err=errs, peaks=peaks,
            stored=[bool(x) for x in kept[order]],
            fields=fields, track=track, pedestal=pedestal))
    data["track_summary"] = summary

    blob = json.dumps(data, separators=(",", ":"))
    page = open(PAGE).read()
    page = re.sub(r'(<script id="viewdata" type="application/json">).*?(</script>)',
                  lambda m: m.group(1) + blob + m.group(2), page, flags=re.S)
    open(PAGE, "w").write(page)
    print(f"docs/view_circle.html rebuilt — {len(page)/1024:.0f} KB, "
          f"{len(names)} objects")
    for key in ("perframe", "free", "anchor"):
        med = np.median([summary[key][n]["med"] for n in names])
        imp = sum(summary[key][n]["imp"] for n in names)
        print(f"  {key:9s} median-of-medians {med:5.1f}°  impossible {imp}/426")


if __name__ == "__main__":
    main()
