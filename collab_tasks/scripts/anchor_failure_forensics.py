"""Why does a coarse-grid anchor land metres from the full-grid peak?

Claim under test: the 5% tail of anchor misses is genuine MULTI-PEAK structure
in the field (several instances of a class -> several real peaks), not a fault.

Paul's objection: that does not sound right, it sounds like a fault in the trace.

An aggregate cannot answer this, so this script does not produce another
summary number. It classifies every query against the alternatives and then
RENDERS the fields of the worst misses, with the class's own events drawn on
top, so the field can be read directly.

Competing explanations, all measured here:
  H1 multi-peak    the field really does have >1 comparable peak
  H2 degenerate    the query has no answer (few events / probe time far from
                   any observation); the peak is noise and BOTH argmaxes are
                   arbitrary. Detectable via the trace's own calibrated z.
  H3 undersampled  one narrow peak, and the coarse lattice steps over it.
                   Detectable: peak width < coarse spacing.
  H4 harness bug   the coarse index -> (row, col) mapping is wrong.
                   Detectable: brute-force recompute of the same quantity.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from vsa_cognitive_mapping.astm_traces import TraceSet, QueryRouter, load_events

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACE = "outputs/classroom/astm_traces.pt"
OUT_DIR = "outputs/classroom"
STRIDE = 4
FAIL_CELLS = 8          # > 1 m: a miss the window cannot recover from
BG, DIM, GRID = "#05070A", "#8A97A0", "#1C242C"
MAGENTA, MINT, AMBER = "#FF5FD0", "#7CE0C0", "#FFB000"


def label_blobs(mask):
    """Count 4-connected components without scipy."""
    seen = np.zeros_like(mask, bool)
    n = 0
    H, W = mask.shape
    for sj in range(H):
        for si in range(W):
            if not mask[sj, si] or seen[sj, si]:
                continue
            n += 1
            stack = [(sj, si)]
            seen[sj, si] = True
            while stack:
                j, i = stack.pop()
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = j + dj, i + di
                    if 0 <= a < H and 0 <= b < W and mask[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        stack.append((a, b))
    return n


def main():
    tr = TraceSet.load(TRACE)
    router = QueryRouter(tr)
    ny, nx = len(tr.gy), len(tr.gx)
    G = tr.G_flat
    step = float(tr.gx[1] - tr.gx[0])
    ev = load_events(OUT_DIR)
    cls_arr = np.asarray(ev["class"])
    ts = np.asarray(ev["t"])
    xs, ys = np.asarray(ev["x"]), np.asarray(ev["y"])
    counts = {c: int((cls_arr == c).sum()) for c in sorted(set(cls_arr))}

    cj = np.arange(0, ny, STRIDE)
    ci = np.arange(0, nx, STRIDE)
    Gc = np.ascontiguousarray(G[np.array([j * nx + i for j in cj for i in ci])])

    # replay the SAME 147 queries as the benchmark (same seed, same order)
    captured = []
    orig = QueryRouter._decode_grid

    def spy(self, residual, probe_time=None):
        captured.append((np.array(residual),
                         None if probe_time is None else np.array(probe_time)))
        return orig(self, residual, probe_time)

    QueryRouter._decode_grid = spy
    rng = np.random.RandomState(0)
    meta = []
    for c in sorted(set(cls_arr)):
        m = cls_arr == c
        if m.sum() < 3:
            continue
        for t in rng.choice(ts[m], size=min(6, int(m.sum())), replace=False):
            try:
                _, info = router.query("where", what=c, when=float(t))
                meta.append(dict(cls=c, t=float(t), n_events=counts[c],
                                 z=float(info.get("z", float("nan"))),
                                 confident=bool(info.get("confident", False)),
                                 sim=float(info.get("sim", float("nan")))))
            except Exception:
                captured.pop()
    QueryRouter._decode_grid = orig
    assert len(meta) == len(captured), (len(meta), len(captured))

    rows = []
    fields = []
    for q, (r, p) in enumerate(captured):
        v = (np.conj(r) * (p if p is not None else 1.0)).astype(np.complex64)
        s = (np.real(G @ v) / tr.hd).reshape(ny, nx)
        fields.append(s)
        rj, ri = np.unravel_index(int(np.argmax(s)), s.shape)
        peak = float(s[rj, ri])

        k = int(np.argmax(np.real(Gc @ v) / tr.hd))
        aj, ai = int(cj[k // len(ci)]), int(ci[k % len(ci)])
        # H4 control: recompute the coarse pick by brute force over the same
        # cells, independent of the index arithmetic under test.
        best, baj, bai = -1e30, -1, -1
        for j in cj:
            for i in ci:
                sc = float(s[j, i])
                if sc > best:
                    best, baj, bai = sc, int(j), int(i)
        assert (aj, ai) == (baj, bai), "H4: coarse index mapping disagrees"

        err = max(abs(aj - rj), abs(ai - ri))
        # H1: how many separate blobs stand above half the peak height
        thr = peak * 0.5
        blobs = label_blobs(s >= thr) if peak > 0 else 0
        # H3: width of the peak blob, in fine cells, along each axis
        col = s[:, ri]
        row = s[rj, :]
        wj = int((col >= thr).sum())
        wi = int((row >= thr).sum())
        m = meta[q]
        rows.append(dict(q=q, **m, peak=peak,
                         coarse_score=float(s[aj, ai]),
                         score_ratio=float(s[aj, ai] / peak) if peak > 0 else float("nan"),
                         err_cells=int(err), err_m=float(err * step),
                         blobs_above_half=int(blobs),
                         peak_w_x_cells=wi, peak_w_y_cells=wj,
                         peak_w_x_m=float(wi * step), peak_w_y_m=float(wj * step),
                         fail=bool(err > FAIL_CELLS)))

    fails = [r for r in rows if r["fail"]]
    oks = [r for r in rows if not r["fail"]]
    print("=" * 78)
    print("coarse spacing = %d cells = %.3f m   |   pos length scale = %.2f m"
          % (STRIDE, STRIDE * step, tr.enc.pos_l))
    print("queries: %d   misses (>%d cells = %.1f m): %d (%.1f%%)"
          % (len(rows), FAIL_CELLS, FAIL_CELLS * step, len(fails),
             100 * len(fails) / len(rows)))
    print("=" * 78)

    def col(rs, key):
        a = np.array([r[key] for r in rs], float)
        return np.nan if not len(a) else float(np.median(a))

    print("\n-- H2: does the trace's OWN confidence separate misses from hits? --")
    print("  group          n   median z   'confident'   median class events")
    for name, rs in (("hits", oks), ("MISSES", fails)):
        nconf = sum(1 for r in rs if r["confident"])
        print("  %-8s %5d %10.2f %8d/%-5d %12.0f"
              % (name, len(rs), col(rs, "z"), nconf, len(rs), col(rs, "n_events")))
    conf_rows = [r for r in rows if r["confident"]]
    conf_fail = [r for r in conf_rows if r["fail"]]
    print("\n  >> among queries the system says it CAN answer (confident=True):")
    print("     %d queries, %d misses = %.1f%%"
          % (len(conf_rows), len(conf_fail),
             100 * len(conf_fail) / max(1, len(conf_rows))))

    print("\n-- H1 vs H3: is the field multi-peak, or is one peak being stepped over? --")
    print("  group     median blobs>half   median peak width x / y (m)   median coarse/peak score")
    for name, rs in (("hits", oks), ("MISSES", fails)):
        print("  %-8s %14.1f %14.2f / %-8.2f %18.3f"
              % (name, col(rs, "blobs_above_half"), col(rs, "peak_w_x_m"),
                 col(rs, "peak_w_y_m"), col(rs, "score_ratio")))
    multi = sum(1 for r in fails if r["blobs_above_half"] >= 2)
    narrow = sum(1 for r in fails if r["peak_w_x_m"] < STRIDE * step
                 or r["peak_w_y_m"] < STRIDE * step)
    print("\n  of %d misses: %d (%.0f%%) have >=2 blobs above half height  [H1]"
          % (len(fails), multi, 100 * multi / max(1, len(fails))))
    print("                 %d (%.0f%%) have a peak NARROWER than the coarse "
          "spacing  [H3]" % (narrow, 100 * narrow / max(1, len(fails))))

    print("\n-- the 8 worst misses, individually --")
    print("  %-14s %7s %6s %6s %7s %6s %8s %8s"
          % ("class", "events", "z", "conf", "err m", "blobs", "w_x m", "coarse/peak"))
    worst = sorted(fails, key=lambda r: -r["err_m"])[:8]
    for r in worst:
        print("  %-14s %7d %6.1f %6s %7.2f %6d %8.2f %8.3f"
              % (r["cls"], r["n_events"], r["z"], r["confident"], r["err_m"],
                 r["blobs_above_half"], r["peak_w_x_m"], r["score_ratio"]))

    # ---- render ------------------------------------------------------------
    # Two fixes over the first version of this figure, both of which were
    # making the colour unreadable AND dishonest:
    #  1. imshow was auto-scaling EVERY panel to its own min/max, so a field
    #     that is pure noise was stretched to full saturation and read as
    #     dramatic structure. One SHARED symmetric scale instead, set by the
    #     answerable panels, so "no peak" renders as flat neutral.
    #  2. magma (a multi-hue sequential ramp) fought three marker hues. The
    #     field is SIGNED, so it wants a diverging map: two hues either side
    #     of a neutral grey midpoint at zero. Both picks now share one ink and
    #     are told apart by SHAPE, not colour.
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "icns_signed", ["#3B9BFF", "#1B2A38", "#2A2F35", "#4A3320", "#FF8C42"])

    def coarse_pick(q):
        r0, p0 = captured[q]
        v = (np.conj(r0) * (p0 if p0 is not None else 1.0)).astype(np.complex64)
        k = int(np.argmax(np.real(Gc @ v) / tr.hd))
        return int(cj[k // len(ci)]), int(ci[k % len(ci)])

    # Selection rule, stated on the figure: strongest 4 answerable queries
    # (highest z) against the 4 largest misses. Not chosen by eye.
    good = sorted([r for r in rows if r["confident"]],
                  key=lambda r: -r["z"])[:4]
    bad = sorted(fails, key=lambda r: -r["err_m"])[:4]
    lim = max(abs(fields[r["q"]]).max() for r in good)

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 9.4))
    fig.patch.set_facecolor(BG)
    # explicit spacing, not tight_layout: the two-line panel titles collide
    # with the row above's tick labels otherwise.
    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.175,
                        wspace=0.22, hspace=0.34)
    ext = [tr.gx[0], tr.gx[-1], tr.gy[0], tr.gy[-1]]
    for row_i, (group, tag) in enumerate(((good, "HAS AN ANSWER"),
                                          (bad, "SYSTEM WOULD DECLINE"))):
        for col_i, r in enumerate(group):
            ax = axes[row_i, col_i]
            s = fields[r["q"]]
            ax.set_facecolor(BG)
            im = ax.imshow(s, origin="lower", extent=ext, cmap=cmap,
                           vmin=-lim, vmax=lim,
                           interpolation="nearest", aspect="equal")
            m = cls_arr == r["cls"]
            ax.scatter(xs[m], ys[m], s=9, c=MINT, linewidths=0, alpha=.85,
                       label="detections of this class", zorder=2)
            rj, ri = np.unravel_index(int(np.argmax(s)), s.shape)
            aj, ai = coarse_pick(r["q"])
            ax.plot(tr.gx[ri], tr.gy[rj], marker="o", ms=15, mfc="none", ls="none",
                    mec="#FFFFFF", mew=1.8, zorder=3,
                    label="full search says here")
            ax.plot(tr.gx[ai], tr.gy[aj], marker="P", ms=10, mfc="#FFFFFF", ls="none",
                    mec=BG, mew=1.2, zorder=4, label="rough guess says here")
            ax.set_title("%s · %d detections\nconfidence z = %+.1f · off by %.2f m"
                         % (r["cls"], r["n_events"], r["z"], r["err_m"]),
                         color="#FFFFFF", fontsize=9.5)
            ax.tick_params(colors=DIM, labelsize=7.5)
            for sp in ax.spines.values():
                sp.set_color(GRID)
        axes[row_i, 0].set_ylabel(tag, color="#FFFFFF", fontsize=10.5,
                                  labelpad=10)

    cax = fig.add_axes([0.32, 0.095, 0.36, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor(GRID)
    cb.set_label("field strength — SAME scale on all eight panels · grey = zero",
                 color=DIM, fontsize=9, labelpad=6)
    cb.ax.tick_params(colors=DIM, labelsize=7.5)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               labelcolor=DIM, fontsize=9.5, bbox_to_anchor=(0.5, 0.008))
    fig.suptitle("A real answer is a single bright spot. A declined query is flat "
                 "grey — there is nothing there to find.\n"
                 "top: 4 highest-confidence queries   ·   bottom: the 4 largest "
                 "misses   ·   selection is by rank, not by eye",
                 color="#FFFFFF", fontsize=12.5, y=0.965)
    p = os.path.join("outputs", "anchor_failure_fields.png")
    fig.savefig(p, facecolor=BG, dpi=130)
    print("\nwrote %s  (shared scale +/- %.4f)" % (p, lim))

    with open("outputs/anchor_failure_forensics.json", "w") as f:
        json.dump(dict(stride=STRIDE, coarse_spacing_m=STRIDE * step,
                       fail_cells=FAIL_CELLS, pos_l=float(tr.enc.pos_l),
                       n=len(rows), n_fail=len(fails),
                       n_confident=len(conf_rows), n_confident_fail=len(conf_fail),
                       rows=rows), f, indent=1)
    print("wrote outputs/anchor_failure_forensics.json")


if __name__ == "__main__":
    main()
