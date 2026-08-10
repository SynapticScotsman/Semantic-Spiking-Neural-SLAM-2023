"""Build the paper's page-1 hero figure from measured artifacts.

Nothing here is drawn by hand or re-typed: every photo, trajectory,
belief field and error number is read out of the inspector payloads that
the export_* tools already produced, so the figure cannot drift from the
measured record.

Three panels, left to right:

  (a) what the memory is asked and what it returns -- real query frames
      from the 7-Scenes chess TEST walks beside the frames the 32 KB
      trace recalled, chosen at the MEDIAN error (not best case);
  (b) where those answers land -- the four stored training walks, the
      test queries, and a segment per query from truth to answer;
  (c) that the answer is a distribution, not a lookup -- the belief field
      for one object-class query on Replica room0, with ground-truth
      instances marked.

Usage:
    python tools/make_hero_figure.py
    python tools/make_hero_figure.py --class-name couch --n-pairs 2
"""
import argparse
import base64
import io
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHESS_HTML = "docs/sites/cross_inspector_7scenes_chess.html"
ROOM0_HTML = "docs/sites/grounding_inspector_room0.html"
OUT_STEM = "paper/figures/hero"

# print-friendly, colour-blind-safe accents
C_MEM = "#9aa7b4"     # stored walks
C_TRUE = "#111820"    # ground truth
C_VSA = "#d1495b"     # what the trace answered
C_GT = "#00a878"      # ground-truth object instances
C_PEAK = "#ffffff"    # field peak


def payload(path):
    """Pull the embedded JSON payload out of an inspector page."""
    with io.open(path, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'<script[^>]*id="(?:data|payload)"[^>]*>(.*?)</script>', src, re.S)
    if not m:
        raise SystemExit("no payload found in %s" % path)
    return json.loads(m.group(1))


def decode(data_uri):
    """data:image/jpeg;base64,... -> PIL image."""
    from PIL import Image
    b64 = data_uri.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def spread_queries(queries, imgs, percentiles):
    """Pick one query per requested error percentile.

    Deliberately not the best cases. Spreading across percentiles shows
    the typical answer AND a worse one, so the panel cannot be read as
    cherry-picked; the caption states which percentiles are shown.
    """
    usable = [q for q in queries if q.get("q") in imgs and q.get("a") in imgs]
    if not usable:
        raise SystemExit("no queries carry both a query and an answer image")
    errs = np.array([q["err"] for q in usable])
    picked, used = [], set()
    for pc in percentiles:
        target = float(np.percentile(errs, pc))
        for i in np.argsort(np.abs(errs - target)):
            if i not in used:
                used.add(i)
                picked.append((usable[i], pc))
                break
    return picked, float(np.median(errs))


def pick_class(panels, wanted=None):
    """A class with ground-truth instances and a real field to show."""
    def ok(p):
        return p.get("field") and p.get("targets")
    if wanted:
        for p in panels:
            if p["cls"] == wanted and ok(p):
                return p
        raise SystemExit("class %r has no field/targets in the payload" % wanted)
    # prefer a correct answer on a multi-instance class: the field then
    # shows several lobes, which is the point of panel (c)
    scored = [p for p in panels if ok(p)]
    if not scored:
        raise SystemExit("no panel carries a field")
    scored.sort(key=lambda p: (not bool(p.get("hit_t") or p.get("hit")),
                               -int(p.get("n_gt") or 0),
                               float(p.get("d1t") or p.get("d1") or 9e9)))
    return scored[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chess", default=CHESS_HTML)
    ap.add_argument("--room0", default=ROOM0_HTML)
    ap.add_argument("--out", default=OUT_STEM)
    ap.add_argument("--class-name", default=None,
                    help="object class for panel (c); default picks automatically")
    ap.add_argument("--percentiles", default="50,75",
                    help="error percentiles to illustrate in panel (a)")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ch = payload(args.chess)
    rm = payload(args.room0)

    pcs = [float(x) for x in args.percentiles.split(",")]
    pairs, med = spread_queries(ch["queries"], ch["imgs"], pcs)
    panel = pick_class(rm["panels"], args.class_name)

    plt.rcParams.update({
        "font.size": 7,
        "font.family": "sans-serif",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    fig = plt.figure(figsize=(7.16, 2.02))
    gs = GridSpec(1, 3, width_ratios=[2.05, 2.15, 2.60], wspace=0.24,
                  left=0.005, right=0.995, top=0.84, bottom=0.02)

    # ---------------------------------------------------------------- (a)
    # invisible container so panel (a)'s title sits on the same baseline
    # as the titles of (b) and (c)
    axa = fig.add_subplot(gs[0])
    axa.set_title("(a) a photo goes in, a place comes back",
                  fontsize=7.2, pad=3.5, weight="bold")
    axa.set_xticks([]); axa.set_yticks([])
    for sp in axa.spines.values():
        sp.set_visible(False)
    sub = gs[0].subgridspec(len(pairs), 2, hspace=0.06, wspace=0.05)
    for r, (q, pc) in enumerate(pairs):
        for c, key in enumerate(("q", "a")):
            ax = fig.add_subplot(sub[r, c])
            ax.imshow(decode(ch["imgs"][q[key]]))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(C_TRUE if c == 0 else C_VSA)
                sp.set_linewidth(1.0)
            if r == 0:
                ax.text(0.5, 1.04, "asked" if c == 0 else "recalled",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.5, color=C_TRUE if c == 0 else C_VSA)
            if c == 1:
                ax.text(0.97, 0.06, "%.2f m" % q["err"], transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6, color="white",
                        bbox=dict(fc=C_VSA, ec="none", pad=1.2, alpha=0.92))
            if c == 0:
                ax.text(0.03, 0.06, "p%d" % round(pc), transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=5.6, color="white",
                        bbox=dict(fc=C_TRUE, ec="none", pad=1.0, alpha=0.85))
    # ---------------------------------------------------------------- (b)
    axb = fig.add_subplot(gs[1])
    for path in ch["paths"]:
        p = np.asarray(path, dtype=float)
        if p.ndim == 2 and len(p):
            axb.plot(p[:, 0], p[:, 1], "-", color=C_MEM, lw=0.7, zorder=1)
    qs = ch["queries"]
    for q in qs:
        axb.plot([q["x"], q["px"]], [q["y"], q["py"]], "-",
                 color=C_VSA, lw=0.45, alpha=0.75, zorder=2)
    axb.scatter([q["x"] for q in qs], [q["y"] for q in qs], s=2.2,
                color=C_TRUE, zorder=3, linewidths=0)
    for q, _pc in pairs:  # tie panel (a) to the map
        axb.scatter([q["x"]], [q["y"]], s=26, facecolor="none",
                    edgecolor=C_TRUE, linewidth=0.9, zorder=4)
    ext = ch["ext"]
    axb.set_xlim(ext[0], ext[1]); axb.set_ylim(ext[2], ext[3])
    axb.set_aspect("equal"); axb.set_xticks([]); axb.set_yticks([])
    axb.set_title("(b) 4 stored walks, 2 unseen walks queried",
                  fontsize=7.2, pad=3.5, weight="bold")
    tbl = {r["system"]: r for r in ch.get("table", [])}
    vsa = tbl.get("VSA argmax", {}); knn = tbl.get("kNN exact", {})
    # the visible red segments include the tail, so give the reader the
    # fraction as well as the median rather than only the headline
    axb.text(0.035, 0.03,
             "32 KB trace   %.2f m med   %.0f%% <0.5 m\n"
             "32.8 MB kNN   %.2f m med   %.0f%% <0.5 m"
             % (vsa.get("med3d") or med, 100 * (vsa.get("le05") or 0),
                knn.get("med3d") or float("nan"), 100 * (knn.get("le05") or 0)),
             transform=axb.transAxes, fontsize=5.4, va="bottom", ha="left",
             linespacing=1.35,
             bbox=dict(fc="white", ec=C_MEM, lw=0.5, pad=1.8, alpha=0.93))

    # ---------------------------------------------------------------- (c)
    axc = fig.add_subplot(gs[2])
    n = int(rm.get("grid", 64))
    field = np.asarray(panel["field"], dtype=float).reshape(n, n)
    ex = rm["ext"]
    axc.imshow(field, origin="lower", extent=[ex[0], ex[1], ex[2], ex[3]],
               cmap="magma", aspect="equal", interpolation="bilinear")
    pth = np.asarray(rm.get("path") or [], dtype=float)
    if pth.ndim == 2 and len(pth):
        axc.plot(pth[:, 0], pth[:, 1], "-", color="white", lw=0.5, alpha=0.35)
    tg = np.asarray(panel["targets"], dtype=float)
    if tg.ndim == 2 and len(tg):
        axc.scatter(tg[:, 0], tg[:, 1], marker="s", s=30, facecolor="none",
                    edgecolor=C_GT, linewidth=1.1, zorder=4)
    pk = np.asarray(panel.get("peaks") or [], dtype=float)
    if pk.ndim == 2 and len(pk):
        axc.scatter(pk[:1, 0], pk[:1, 1], marker="+", s=42, color=C_PEAK,
                    linewidth=1.1, zorder=5)
    axc.set_xlim(ex[0], ex[1]); axc.set_ylim(ex[2], ex[3])
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_title('(c) "where is the %s?" answers with a field'
                  % panel["cls"], fontsize=7.2, pad=3.5, weight="bold")
    axc.text(0.03, 0.035,
             "green = true instances\nwhite + = peak",
             transform=axc.transAxes, fontsize=6.0, va="bottom", ha="left",
             color="white", alpha=0.95)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext_ in ("pdf", "png"):
        path = "%s.%s" % (args.out, ext_)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
        print("wrote %s (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))
    print("panel (a): %d queries at median error %.2f m" % (len(pairs), med))
    print("panel (c): class=%s  n_gt=%s  hit=%s"
          % (panel["cls"], panel.get("n_gt"), panel.get("hit_t") or panel.get("hit")))


if __name__ == "__main__":
    main()
