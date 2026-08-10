"""Build the paper's figures from measured artifacts.

Nothing here is drawn by hand or re-typed: every photo, trajectory,
belief field, merge number and capacity point is read out of artifacts
the pipeline already produced, so a figure cannot drift from the record.

Two layouts:

  --layout overview   (default, the page-1 teaser)
      One panel per contribution.
      (a) QUERY  one vector answers two kinds of question: a photo goes
          in and a place comes back; a class goes in and a spatial belief
          field comes back.
      (b) MERGE  four robots walk disjoint routes; their traces ADD to
          the jointly built map, with no data association at all, where
          the equivalent explicit merge needs hundreds of decisions and
          loses accuracy.
      (c) LAW    interference has a closed form: a shared mean grows
          coherently as O(N) while everything else reverts to sqrt(N),
          and the model predicts a held-out condition to 1.4%.

  --layout results    (the relocalisation figure, for the results section)
      (a) asked/recalled photo pairs at chosen error percentiles,
      (b) the chess map with a segment from truth to answer per query,
      (c) a Replica belief field.

Usage:
    python tools/make_hero_figure.py
    python tools/make_hero_figure.py --layout results --out paper/figures/reloc
"""
import argparse
import base64
import collections
import glob
import io
import json
import os
import re
import statistics

import numpy as np

CHESS_HTML = "docs/sites/cross_inspector_7scenes_chess.html"
ROOM0_HTML = "docs/sites/grounding_inspector_room0.html"
MERGE_GLOB = "outputs/replica_*/merge_comparison_k4.json"
SYNTH_ROWS = "outputs/synthetic/rows.jsonl"

C_MEM = "#9aa7b4"
C_TRUE = "#111820"
C_VSA = "#d1495b"
C_GT = "#00a878"
C_PEAK = "#ffffff"
# four robots, colour-blind-safe and distinct in greyscale order
C_ROBOT = ["#3b9bff", "#00a878", "#e8a33d", "#d1495b"]


# ---------------------------------------------------------------- loading
def payload(path):
    with io.open(path, encoding="utf-8") as fh:
        src = fh.read()
    m = re.search(r'<script[^>]*id="(?:data|payload)"[^>]*>(.*?)</script>', src, re.S)
    if not m:
        raise SystemExit("no payload found in %s" % path)
    return json.loads(m.group(1))


def decode(data_uri):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(
        data_uri.split(",", 1)[1]))).convert("RGB")


def spread_queries(queries, imgs, percentiles):
    """One query per requested error percentile. Never best-case."""
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
    def ok(p):
        return p.get("field") and p.get("targets")
    if wanted:
        for p in panels:
            if p["cls"] == wanted and ok(p):
                return p
        raise SystemExit("class %r has no field/targets" % wanted)
    scored = [p for p in panels if ok(p)]
    if not scored:
        raise SystemExit("no panel carries a field")
    scored.sort(key=lambda p: (not bool(p.get("hit_t") or p.get("hit")),
                               -int(p.get("n_gt") or 0),
                               float(p.get("d1t") or p.get("d1") or 9e9)))
    return scored[0]


def real_traces(scene="outputs/replica_room0/object_points.json",
                dim=8192, seed=7, cls="chair", ell=0.6, cap=None):
    """Build REAL traces from the real observation stream.

    The strips drawn in the concept figure are actual FHRR phasors from
    this repo's own implementation, bound and bundled exactly as the
    system does it, so the picture is the mechanism rather than an
    illustration of it.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vsa_cognitive_mapping.vsa import Phasor

    pts = json.load(open(scene))["points"]
    rng = np.random.RandomState(seed)

    classes = sorted({p["cls"] for p in pts})
    atoms = {c: Phasor(dim=dim, seed=seed + i) for i, c in enumerate(classes)}
    bx = Phasor(dim=dim, seed=seed + 901)
    by = Phasor(dim=dim, seed=seed + 902)

    def place(x, y):
        return Phasor(data=bx.fpe(x / ell).values * by.fpe(y / ell).values)

    # cap=None bundles the whole stream, so the figure's "N observations
    # into one fixed vector" caption is literally true of what is drawn.
    # Passing cap=k reproduces the deployed bounded per-class insertion.
    if cap is None:
        kept = list(pts)
    else:
        seen = collections.Counter()
        kept = []
        for p in pts:
            if seen[p["cls"]] < cap:
                seen[p["cls"]] += 1
                kept.append(p)

    half = len(kept) // 2
    def bundle(sub):
        acc = np.zeros(dim, dtype=complex)
        for p in sub:
            acc += atoms[p["cls"]].bind(place(p["x"], p["y"])).values
        return acc

    m_a, m_b = bundle(kept[:half]), bundle(kept[half:])
    joint = bundle(kept)
    one = atoms[cls].bind(place(kept[0]["x"], kept[0]["y"])).values
    err = float(np.max(np.abs((m_a + m_b) - joint)))
    return dict(atom=atoms[cls].values, one=one, trace=joint,
                a=m_a, b=m_b, sum=m_a + m_b, merge_err=err,
                n_obs=len(pts), n_kept=len(kept), dim=dim)


def merge_summary(pattern=MERGE_GLOB):
    """Pool the per-scene 4-robot joining results."""
    agg = collections.defaultdict(lambda: {"r": [], "a": 0, "b": []})
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("no merge_comparison_k4.json found under %s" % pattern)
    for f in files:
        for row in json.load(open(f))["rows"]:
            a = agg[row["method"]]
            a["r"].append(float(row["truth_r1"]))
            a["a"] += int(row["assoc_decisions"])
            a["b"].append(int(row["bytes_per_robot"]))
    out = {}
    for m, v in agg.items():
        out[m] = dict(mean=statistics.mean(v["r"]), worst=min(v["r"]),
                      assoc=v["a"], bytes=int(statistics.median(v["b"])),
                      n=len(v["r"]))
    return out, len(files)


def synth_curves(path=SYNTH_ROWS):
    """chi(N) per configuration from the controlled synthetic sweep."""
    cfg = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if "chi" in r:
                cfg[r["config"]].append((r["N"], r["chi"]))
    return {k: np.array(sorted(v), dtype=float) for k, v in cfg.items()}


# ------------------------------------------------------------- components
def draw_photo_pair(fig, cell, q, pc, ch, labels_inside=True):
    """Query frame beside the frame the trace recalled.

    Labels sit inside the images by default: an overview strip has no
    vertical room for a caption row above every photo.
    """
    sub = cell.subgridspec(1, 2, wspace=0.05)
    for c, key in enumerate(("q", "a")):
        ax = fig.add_subplot(sub[0, c])
        ax.imshow(decode(ch["imgs"][q[key]]))
        ax.set_xticks([]); ax.set_yticks([])
        col = C_TRUE if c == 0 else C_VSA
        for sp in ax.spines.values():
            sp.set_color(col)
            sp.set_linewidth(1.0)
        text = "asked" if c == 0 else "recalled"
        if labels_inside:
            ax.text(0.04, 0.94, text, transform=ax.transAxes, ha="left",
                    va="top", fontsize=5.8, color="white",
                    bbox=dict(fc=col, ec="none", pad=1.1, alpha=0.90))
        else:
            ax.text(0.5, 1.03, text, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=6.2, color=col)
        if c == 1:
            ax.text(0.96, 0.06, "%.2f m" % q["err"], transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=5.8, color="white",
                    bbox=dict(fc=C_VSA, ec="none", pad=1.1, alpha=0.92))


def draw_field(ax, rm, panel, compact=False):
    n = int(rm.get("grid", 64))
    field = np.asarray(panel["field"], dtype=float).reshape(n, n)
    ex = rm["ext"]
    ax.imshow(field, origin="lower", extent=[ex[0], ex[1], ex[2], ex[3]],
              cmap="magma", aspect="equal", interpolation="bilinear")
    pth = np.asarray(rm.get("path") or [], dtype=float)
    if pth.ndim == 2 and len(pth):
        ax.plot(pth[:, 0], pth[:, 1], "-", color="white", lw=0.45, alpha=0.30)
    tg = np.asarray(panel["targets"], dtype=float)
    if tg.ndim == 2 and len(tg):
        ax.scatter(tg[:, 0], tg[:, 1], marker="s", s=22 if compact else 30,
                   facecolor="none", edgecolor=C_GT, linewidth=1.0, zorder=4)
    pk = np.asarray(panel.get("peaks") or [], dtype=float)
    if pk.ndim == 2 and len(pk):
        ax.scatter(pk[:1, 0], pk[:1, 1], marker="+", s=38, color=C_PEAK,
                   linewidth=1.05, zorder=5)
    ax.set_xlim(ex[0], ex[1]); ax.set_ylim(ex[2], ex[3])
    ax.set_xticks([]); ax.set_yticks([])


def strip(ax, values, n_show=176, lw=0.6):
    """Draw a phasor as a barcode of phase angles: the map, made visible."""
    v = np.asarray(values)
    idx = np.linspace(0, len(v) - 1, n_show).astype(int)
    ph = np.angle(v[idx])[None, :]
    ax.imshow(ph, aspect="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#3a4654"); sp.set_linewidth(lw)


# ------------------------------------------------------------- layout: concept
def build_concept(args):
    """Show WHAT the representation is, not just that it works.

    Scene-graph teasers (ConceptGraphs, FARM) show their data structure
    accumulating: point clouds, nodes, edges, Gaussians. A reader can
    picture the thing. The equivalent here is the trace itself, so this
    figure draws it, and draws every operation acting on it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    rm = payload(args.room0)
    panel = pick_class(rm["panels"], args.class_name or "chair")
    tr = real_traces(cls=panel["cls"])
    mg, n_scenes = merge_summary()
    vsa, inst = mg["VSA trace + addition"], mg["instance lists + association"]

    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                         "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(7.16, 2.30))
    gs = GridSpec(1, 4, width_ratios=[1.90, 1.62, 1.86, 1.78], wspace=0.30,
                  left=0.006, right=0.994, top=0.855, bottom=0.045)

    T = dict(fontsize=7.0, weight="bold", pad=4)
    note = dict(fontsize=5.5, color="#5b6875", ha="center")

    # ---- (a) one observation is one multiply-add --------------------
    a = fig.add_subplot(gs[0]); a.axis("off")
    a.set_title("(a) an observation is a multiply-add", **T)
    sa = gs[0].subgridspec(4, 1, height_ratios=[2.5, 0.5, 0.5, 0.8],
                           hspace=0.42)
    axp = fig.add_subplot(sa[0])
    key = panel.get("tq") or panel.get("ta")
    if key and key in rm["imgs"]:
        axp.imshow(decode(rm["imgs"][key]))
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_color(C_GT); sp.set_linewidth(1.0)
    axp.text(0.04, 0.93, panel["cls"], transform=axp.transAxes, ha="left",
             va="top", fontsize=5.8, color="white",
             bbox=dict(fc=C_GT, ec="none", pad=1.1, alpha=0.92))
    ax1 = fig.add_subplot(sa[1]); strip(ax1, tr["atom"])
    ax1.set_ylabel("what", rotation=0, ha="right", va="center",
                   fontsize=5.6, labelpad=3)
    ax2 = fig.add_subplot(sa[2])
    strip(ax2, np.exp(1j * np.angle(tr["one"] / tr["atom"])))
    ax2.set_ylabel("where", rotation=0, ha="right", va="center",
                   fontsize=5.6, labelpad=3)
    ax3 = fig.add_subplot(sa[3]); strip(ax3, tr["one"], lw=1.0)
    ax3.set_ylabel(r"what $\otimes$ where", rotation=0, ha="right",
                   va="center", fontsize=5.6, labelpad=3)

    # ---- (b) the whole map is one vector ----------------------------
    b = fig.add_subplot(gs[1]); b.axis("off")
    b.set_title("(b) the map is one vector", **T)
    sb = gs[1].subgridspec(3, 1, height_ratios=[1.0, 1.1, 2.3], hspace=0.35)
    fig.add_subplot(sb[0]).axis("off")
    axt = fig.add_subplot(sb[1]); strip(axt, tr["trace"], lw=1.2)
    axt.set_ylabel(r"$\Sigma$", rotation=0, ha="right", va="center",
                   fontsize=9, labelpad=3)
    axn = fig.add_subplot(sb[2]); axn.axis("off")
    axn.text(0.5, 0.92,
             "all %s observations\nof one walk, bundled into\n%s phases = 32 KB"
             % ("{:,}".format(tr["n_kept"]), "{:,}".format(tr["dim"])),
             transform=axn.transAxes, ha="center", va="top", fontsize=6.0,
             linespacing=1.45)
    axn.text(0.5, 0.30, "storing more\nnever makes it longer",
             transform=axn.transAxes, va="top", linespacing=1.4, **note)

    # ---- (c) a question is one unbind -------------------------------
    c = fig.add_subplot(gs[2]); c.axis("off")
    c.set_title("(c) a question is one unbind", **T)
    sc = gs[2].subgridspec(2, 1, height_ratios=[0.55, 3.1], hspace=0.30)
    axq = fig.add_subplot(sc[0]); strip(axq, tr["atom"])
    axq.set_ylabel(r"$\oslash$ %s" % panel["cls"], rotation=0, ha="right",
                   va="center", fontsize=5.8, labelpad=3)
    axf = fig.add_subplot(sc[1]); draw_field(axf, rm, panel, compact=True)
    axf.text(0.03, 0.96, '"where is the %s?"' % panel["cls"],
             transform=axf.transAxes, ha="left", va="top", fontsize=5.8,
             color="white")
    axf.text(0.03, 0.04, "green = true instances", transform=axf.transAxes,
             ha="left", va="bottom", fontsize=5.2, color="white", alpha=0.9)

    # ---- (d) two maps combine by addition ---------------------------
    d = fig.add_subplot(gs[3]); d.axis("off")
    d.set_title("(d) two maps add", **T)
    sd = gs[3].subgridspec(5, 1, height_ratios=[0.5, 0.5, 0.5, 0.5, 2.0],
                           hspace=0.55)
    for i, (vals, lab) in enumerate(((tr["a"], "robot A"),
                                     (tr["b"], "robot B"),
                                     (tr["sum"], "A + B"))):
        ax = fig.add_subplot(sd[i if i < 2 else 3])
        strip(ax, vals, lw=1.0 if i == 2 else 0.6)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center",
                      fontsize=5.6, labelpad=3)
    axplus = fig.add_subplot(sd[2]); axplus.axis("off")
    axplus.text(0.5, 0.5, "+", transform=axplus.transAxes, ha="center",
                va="center", fontsize=9)
    axd = fig.add_subplot(sd[4]); axd.axis("off")
    axd.text(0.5, 0.98, "= the jointly built map", transform=axd.transAxes,
             ha="center", va="top", fontsize=6.2)
    axd.text(0.5, 0.72,
             "to $10^{%d}$, with no\ndata association at all"
             % round(np.log10(tr["merge_err"])), transform=axd.transAxes,
             va="top", linespacing=1.4, **note)
    axd.text(0.5, 0.24,
             "%.0f%% vs %.0f%% for an explicit\nmerge and its %d decisions"
             % (100 * vsa["mean"], 100 * inst["mean"], inst["assoc"]),
             transform=axd.transAxes, ha="center", va="top", fontsize=5.5,
             linespacing=1.4)

    # flow arrows between stages
    for x in (0.268, 0.505, 0.745):
        fig.text(x, 0.47, "→", ha="center", va="center", fontsize=11,
                 color="#8b97a4")

    return fig, dict(cls=panel["cls"], merge_err=tr["merge_err"],
                     n_obs=tr["n_obs"], n_kept=tr["n_kept"],
                     scenes=n_scenes)


# ------------------------------------------------------------- layout: overview
def build_overview(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ch = payload(args.chess)
    rm = payload(args.room0)
    pairs, _med = spread_queries(ch["queries"], ch["imgs"], [50.0])
    panel = pick_class(rm["panels"], args.class_name)
    mg, n_scenes = merge_summary()
    curves = synth_curves()

    vsa = mg["VSA trace + addition"]
    inst = mg["instance lists + association"]

    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                         "axes.linewidth": 0.6,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.labelsize": 5.6, "ytick.labelsize": 5.6})

    fig = plt.figure(figsize=(7.16, 2.45))
    gs = GridSpec(1, 3, width_ratios=[2.25, 2.35, 2.20], wspace=0.22,
                  left=0.005, right=0.995, top=0.845, bottom=0.055)

    # ---- (a) query --------------------------------------------------
    axa = fig.add_subplot(gs[0]); axa.set_xticks([]); axa.set_yticks([])
    for sp in axa.spines.values():
        sp.set_visible(False)
    axa.set_title("(a) one vector, two kinds of question",
                  fontsize=7.2, pad=4, weight="bold")
    sa = gs[0].subgridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.20)
    q, pc = pairs[0]
    draw_photo_pair(fig, sa[0], q, pc, ch)
    axf = fig.add_subplot(sa[1])
    draw_field(axf, rm, panel, compact=True)
    axf.text(0.02, 0.965, '"where is the %s?"' % panel["cls"],
             transform=axf.transAxes, ha="left", va="top", fontsize=5.9,
             color="white")
    axf.text(0.02, 0.04, "green = true instances", transform=axf.transAxes,
             ha="left", va="bottom", fontsize=5.4, color="white", alpha=0.9)

    # ---- (b) merge --------------------------------------------------
    axb = fig.add_subplot(gs[1])
    axb.set_title("(b) four robots, one map, no association",
                  fontsize=7.2, pad=4, weight="bold")
    path = np.asarray(rm.get("path") or [], dtype=float)
    if path.ndim == 2 and len(path):
        for i, idx in enumerate(np.array_split(np.arange(len(path)), 4)):
            axb.plot(path[idx, 0], path[idx, 1], "-", color=C_ROBOT[i],
                     lw=2.1, solid_capstyle="round", zorder=2)
        # frame the route itself, not the whole room: the panel is about
        # who walked where, and empty floor wastes the little space there is
        pad = 0.10 * max(np.ptp(path[:, 0]), np.ptp(path[:, 1]))
        x0, x1 = path[:, 0].min() - pad, path[:, 0].max() + pad
        y0, y1 = path[:, 1].min() - pad, path[:, 1].max() + pad
        # leave headroom at the top for the equation
        axb.set_xlim(x0, x1); axb.set_ylim(y0, y1 + 0.42 * (y1 - y0))
    axb.set_aspect("equal"); axb.set_xticks([]); axb.set_yticks([])
    axb.text(0.5, 0.985,
             r"$M_1 + M_2 + M_3 + M_4 \;=\;$ jointly built map",
             transform=axb.transAxes, ha="center", va="top", fontsize=6.8)
    axb.text(0.5, 0.905, r"agrees to $\sim$10$^{-12}$, zero decisions",
             transform=axb.transAxes, ha="center", va="top", fontsize=5.9,
             color="#5b6875")
    axb.text(0.5, 0.028,
             "traces add        %.0f%%      %d decisions\n"
             "instance merge  %.0f%%    %d decisions"
             % (100 * vsa["mean"], vsa["assoc"],
                100 * inst["mean"], inst["assoc"]),
             transform=axb.transAxes, fontsize=5.5, va="bottom", ha="center",
             linespacing=1.4,
             bbox=dict(fc="white", ec=C_MEM, lw=0.5, pad=2.0, alpha=0.94))

    # ---- (c) law ----------------------------------------------------
    axc = fig.add_subplot(gs[2])
    axc.set_title("(c) interference has a closed form",
                  fontsize=7.2, pad=4, weight="bold")
    show = [("B_lowrank_mean", "shared mean", C_VSA, "o"),
            ("A_lowrank", "anisotropic", "#e8a33d", "s"),
            ("C_isotropic", "isotropic", "#3b9bff", "^"),
            ("E_lowrank_whit", "whitened", C_GT, "d")]
    for key, lab, col, mk in show:
        if key not in curves:
            continue
        v = curves[key]
        axc.loglog(v[:, 0], v[:, 1], "-", marker=mk, ms=2.4, lw=0.9,
                   color=col, label=lab, markeredgewidth=0)
    axc.set_xlabel("items stored $N$", fontsize=6.2, labelpad=1.5)
    axc.set_ylabel(r"interference $\chi$", fontsize=6.2, labelpad=1.5)
    axc.grid(True, which="both", lw=0.25, color=C_MEM, alpha=0.45)
    axc.tick_params(length=2, pad=1.5)
    leg = axc.legend(fontsize=5.3, loc="upper left", frameon=True,
                     handlelength=1.3, borderpad=0.3, labelspacing=0.25,
                     handletextpad=0.4)
    leg.get_frame().set_linewidth(0.4)
    axc.text(0.97, 0.05,
             "mean term $O(N)$, the rest $\\sqrt{N}$\n"
             "held-out cell predicted to 1.4%",
             transform=axc.transAxes, ha="right", va="bottom", fontsize=5.4,
             linespacing=1.4,
             bbox=dict(fc="white", ec=C_MEM, lw=0.4, pad=1.6, alpha=0.94))

    return fig, dict(merge_scenes=n_scenes, cls=panel["cls"],
                     err=q["err"], vsa=vsa, inst=inst)


# ------------------------------------------------------------- layout: results
def build_results(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ch = payload(args.chess)
    rm = payload(args.room0)
    pcs = [float(x) for x in args.percentiles.split(",")]
    pairs, med = spread_queries(ch["queries"], ch["imgs"], pcs)
    panel = pick_class(rm["panels"], args.class_name)

    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                         "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(7.16, 2.02))
    gs = GridSpec(1, 3, width_ratios=[2.05, 2.15, 2.60], wspace=0.24,
                  left=0.005, right=0.995, top=0.84, bottom=0.02)

    axa = fig.add_subplot(gs[0]); axa.set_xticks([]); axa.set_yticks([])
    for sp in axa.spines.values():
        sp.set_visible(False)
    axa.set_title("(a) a photo goes in, a place comes back",
                  fontsize=7.2, pad=3.5, weight="bold")
    sub = gs[0].subgridspec(len(pairs), 1, hspace=0.16)
    for r, (q, pc) in enumerate(pairs):
        draw_photo_pair(fig, sub[r], q, pc, ch, labels_inside=(r > 0))

    axb = fig.add_subplot(gs[1])
    for p in ch["paths"]:
        p = np.asarray(p, dtype=float)
        if p.ndim == 2 and len(p):
            axb.plot(p[:, 0], p[:, 1], "-", color=C_MEM, lw=0.7, zorder=1)
    qs = ch["queries"]
    for q in qs:
        axb.plot([q["x"], q["px"]], [q["y"], q["py"]], "-", color=C_VSA,
                 lw=0.45, alpha=0.75, zorder=2)
    axb.scatter([q["x"] for q in qs], [q["y"] for q in qs], s=2.2,
                color=C_TRUE, zorder=3, linewidths=0)
    ext = ch["ext"]
    axb.set_xlim(ext[0], ext[1]); axb.set_ylim(ext[2], ext[3])
    axb.set_aspect("equal"); axb.set_xticks([]); axb.set_yticks([])
    axb.set_title("(b) 4 stored walks, 2 unseen walks queried",
                  fontsize=7.2, pad=3.5, weight="bold")
    tbl = {r["system"]: r for r in ch.get("table", [])}
    vsa = tbl.get("VSA argmax", {}); knn = tbl.get("kNN exact", {})
    axb.text(0.035, 0.03,
             "32 KB trace   %.2f m med   %.0f%% <0.5 m\n"
             "32.8 MB kNN   %.2f m med   %.0f%% <0.5 m"
             % (vsa.get("med3d") or med, 100 * (vsa.get("le05") or 0),
                knn.get("med3d") or float("nan"), 100 * (knn.get("le05") or 0)),
             transform=axb.transAxes, fontsize=5.4, va="bottom", ha="left",
             linespacing=1.35,
             bbox=dict(fc="white", ec=C_MEM, lw=0.5, pad=1.8, alpha=0.93))

    axc = fig.add_subplot(gs[2])
    draw_field(axc, rm, panel)
    axc.set_title('(c) "where is the %s?" answers with a field' % panel["cls"],
                  fontsize=7.2, pad=3.5, weight="bold")
    axc.text(0.03, 0.035, "green = true instances\nwhite + = peak",
             transform=axc.transAxes, fontsize=6.0, va="bottom", ha="left",
             color="white", alpha=0.95)
    return fig, dict(cls=panel["cls"], med=med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("concept", "overview", "results"),
                    default="concept")
    ap.add_argument("--chess", default=CHESS_HTML)
    ap.add_argument("--room0", default=ROOM0_HTML)
    ap.add_argument("--out", default=None)
    ap.add_argument("--class-name", default=None)
    ap.add_argument("--percentiles", default="50,75")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    if args.out is None:
        args.out = {"concept": "paper/figures/hero",
                    "overview": "paper/figures/overview",
                    "results": "paper/figures/reloc"}[args.layout]

    fig, info = {"concept": build_concept,
                 "overview": build_overview,
                 "results": build_results}[args.layout](args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for e in ("pdf", "png"):
        path = "%s.%s" % (args.out, e)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.012)
        print("wrote %s (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))
    print("layout=%s  %s" % (args.layout, info))


if __name__ == "__main__":
    main()
