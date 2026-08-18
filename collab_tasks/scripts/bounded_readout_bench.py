"""Measure deployed bytes + decode latency, and the bounded windowed readout.

Turns the projected "420 MB -> ~11 MB, 6.1 ms -> 0.15 ms" from the
episodic-memory pivot note into measured numbers on the real classroom trace.

The identity being exploited is the FPE homomorphism already in ctx_pos:

    ctx_pos(x + dx, y + dy) = ctx_pos(x, y) * ctx_pos(dx, dy)

so one canonical offset window W_off, built once, is re-anchored anywhere by a
single elementwise bind. Scoring at anchor A:

    Re( (W_off * A) @ conj(res) ) == Re( W_off @ (A * conj(res)) )

i.e. one hd-length multiply plus a (n_window, hd) matvec, instead of the full
(5040, hd) matvec against G_flat.

Anchors here come from a COARSE pass over a strided subgrid -- self-contained,
no oracle, no dependence on the (unbuilt) habitual/last-visit trace.
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from vsa_cognitive_mapping.astm_traces import TraceSet, QueryRouter, load_events

TRACE = "outputs/classroom/astm_traces.pt"
N_TIMING = 200
MB = 1024 ** 2


def build_offset_window(tr, r):
    """Canonical (2r+1)^2 offset window in FINE grid steps, as ctx_pos deltas."""
    dx = tr.gx[1] - tr.gx[0]
    dy = tr.gy[1] - tr.gy[0]
    offs, idx = [], []
    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            offs.append(tr.enc.ctx_pos(i * dx, j * dy).values)
            idx.append((j, i))
    return np.asarray(offs, np.complex64), idx


def main():
    tr = TraceSet.load(TRACE)
    router = QueryRouter(tr)
    ny, nx = len(tr.gy), len(tr.gx)
    mem = tr.memory_bytes()

    print("=" * 74)
    print("trace: %s  hd=%d  events=%d  grid=%dx%d=%d"
          % (TRACE, tr.hd, tr.n_events, ny, nx, ny * nx))
    print("bounds=%s  pos_l=%s  fine step=%.4f x %.4f m"
          % ([round(b, 2) for b in tr.bounds], tr.enc.pos_l,
             tr.gx[1] - tr.gx[0], tr.gy[1] - tr.gy[0]))
    print("=" * 74)
    print("\n-- MEASURED memory accounting (bytes) --")
    for k, v in mem.items():
        print("  %-16s %13s  %9.2f MB" % (k, format(v, ","), v / MB))
    print("  headline (traces only)      %.1f KB" % (mem["traces"] / 1024))
    print("  deployed / headline ratio   %.0fx" % (mem["total"] / mem["traces"]))

    # ---- capture REAL residuals from real queries -------------------------
    captured = []
    orig = QueryRouter._decode_grid

    def spy(self, residual, probe_time=None):
        captured.append((np.array(residual),
                         None if probe_time is None else np.array(probe_time)))
        return orig(self, residual, probe_time)

    QueryRouter._decode_grid = spy

    ev = load_events("outputs/classroom")
    ts = np.asarray(ev["t"])
    cls_arr = np.asarray(ev["class"])
    rng = np.random.RandomState(0)
    classes = sorted(set(ev["class"]))
    conf = []
    for c in classes:
        m = cls_arr == c
        if m.sum() < 3:
            continue
        for t in rng.choice(ts[m], size=min(24, int(m.sum())), replace=False):
            try:
                _, info = router.query("where", what=c, when=float(t))
                conf.append(bool(info.get("confident", False)))
            except Exception:
                captured.pop()
    QueryRouter._decode_grid = orig
    assert len(conf) == len(captured), (len(conf), len(captured))
    conf = np.asarray(conf)
    # A query the trace would ABSTAIN on has no answer to reproduce: the field
    # is noise and the full-grid argmax is itself arbitrary. Scoring a decode
    # shortcut on those measures nothing. Keep both populations visible.
    print("\n-- captured %d residuals from %d real 'where' queries over %d classes --"
          % (len(captured), len(conf), len(classes)))
    print("   %d confident (the trace would answer), %d it would abstain on"
          % (int(conf.sum()), int((~conf).sum())))
    if not captured:
        raise SystemExit("no residuals captured -- query routing changed?")

    res = [(np.conj(r) * (p if p is not None else 1.0)).astype(np.complex64)
           for r, p in captured]

    # ---- FULL grid: latency + reference argmax ----------------------------
    G = tr.G_flat
    t0 = time.perf_counter()
    for i in range(N_TIMING):
        s = np.real(G @ res[i % len(res)]) / tr.hd
        _ = int(np.argmax(s))
    full_ms = (time.perf_counter() - t0) / N_TIMING * 1e3
    ref = [int(np.argmax(np.real(G @ v) / tr.hd)) for v in res]
    print("\n-- FULL grid readout: %.3f ms/query over %d reps, %d rows, %.1f MB --"
          % (full_ms, N_TIMING, G.shape[0], G.nbytes / MB))

    # ---- A) window alone, ORACLE anchor -----------------------------------
    # Isolates the cost of the window from the cost of FINDING the anchor.
    # Anchor = true argmax snapped to a stride-8 lattice, so the window must
    # still travel; it just starts in the right basin.
    print("\n-- A) WINDOW ALONE, oracle anchor (isolates window from anchor) --")
    print("%3s %7s %7s %8s %8s %7s %13s"
          % ("r", "rows", "MB", "ms", "speedup", "agree", "exact max|d|"))
    window_only = []
    for r in (2, 3, 4, 5):
        W, widx = build_offset_window(tr, r)
        anchors = []
        for q in range(len(res)):
            rj, ri = divmod(ref[q], nx)
            anchors.append((int(round(rj / 8.0) * 8), int(round(ri / 8.0) * 8)))
        # The anchor vector is a ROW OF THE GRID, not something to rebuild with
        # a complex power each query -- rebuilding it costs ~2 ms on 8192
        # elements and swamps the matvec we are trying to shrink.
        agree, exact_max, oob = 0, 0.0, 0
        for q, v in enumerate(res):
            aj, ai = anchors[q]
            aj, ai = min(aj, ny - 1), min(ai, nx - 1)
            A = G[aj * nx + ai]
            sw = np.real(W @ (A * v)) / tr.hd
            kw = int(np.argmax(sw))
            dj, di = widx[kw]
            bj, bi = aj + dj, ai + di
            if not (0 <= bj < ny and 0 <= bi < nx):
                oob += 1
                continue
            exact_max = max(exact_max,
                            abs(sw[kw] - float(np.real(G[bj * nx + bi] @ v) / tr.hd)))
            if (bj, bi) == divmod(ref[q], nx):
                agree += 1
        n_in = max(1, len(res) - oob)
        t0 = time.perf_counter()
        for i in range(N_TIMING):
            q = i % len(res)
            v = res[q]
            aj, ai = anchors[q]
            aj, ai = min(aj, ny - 1), min(ai, nx - 1)
            _ = int(np.argmax(np.real(W @ (G[aj * nx + ai] * v)) / tr.hd))
        ms = (time.perf_counter() - t0) / N_TIMING * 1e3
        print("%3d %7d %7.2f %8.3f %7.1fx %7.3f %13.2e  (%d edge-clipped)"
              % (r, W.shape[0], W.nbytes / MB, ms, full_ms / ms,
                 agree / n_in, exact_max, oob))
        window_only.append(dict(r=r, rows=int(W.shape[0]), bytes=int(W.nbytes),
                                ms=ms, speedup=full_ms / ms,
                                agree=agree / n_in, edge_clipped=oob,
                                exact_max_absdiff=float(exact_max)))

    # ---- B) how far is a coarse anchor from the true peak? ----------------
    print("\n-- B) anchor quality: coarse-pass anchor vs true argmax (in fine cells) --")
    print("   stratified: a query the trace would ABSTAIN on has no true peak to")
    print("   find, so its 'error' is the distance between two arbitrary picks.")
    print("%7s %7s %11s %7s %7s %5s %11s %7s"
          % ("stride", "rows", "med|conf", "p95", "p99", "max", "med|abstain", "p95"))
    anchor_q = []
    for stride in (4, 6, 8, 10):
        cj = np.arange(0, ny, stride)
        ci = np.arange(0, nx, stride)
        Gc = np.ascontiguousarray(G[np.array([j * nx + i for j in cj for i in ci])])
        d = []
        for q, v in enumerate(res):
            k = int(np.argmax(np.real(Gc @ v) / tr.hd))
            aj, ai = int(cj[k // len(ci)]), int(ci[k % len(ci)])
            rj, ri = divmod(ref[q], nx)
            d.append(max(abs(aj - rj), abs(ai - ri)))  # Chebyshev -> window r
        d = np.asarray(d)
        dc, da = d[conf], d[~conf]
        print("%7d %7d %11.1f %7.1f %7.1f %5d %11.1f %7.1f"
              % (stride, Gc.shape[0], np.median(dc), np.percentile(dc, 95),
                 np.percentile(dc, 99), dc.max(),
                 np.median(da), np.percentile(da, 95)))
        anchor_q.append(dict(stride=stride, coarse_rows=int(Gc.shape[0]),
                             n_confident=int(conf.sum()),
                             med_cells_conf=float(np.median(dc)),
                             p95_cells_conf=float(np.percentile(dc, 95)),
                             p99_cells_conf=float(np.percentile(dc, 99)),
                             max_cells_conf=int(dc.max()),
                             med_cells_abstain=float(np.median(da)),
                             p95_cells_abstain=float(np.percentile(da, 95)),
                             p95_cells=float(np.percentile(dc, 95))))

    # ---- C) full two-stage: coarse anchor + canonical offset window -------
    print("\n-- C) TWO-STAGE end to end (coarse anchor -> offset window) --")
    print("%7s %3s %7s %7s %8s %8s %9s %9s %10s"
          % ("stride", "r", "rows", "MB", "ms", "speedup", "agree|conf",
             "agree|all", "p95 err m"))
    rows_out = []
    for stride in (4, 6, 8, 10):
        cj = np.arange(0, ny, stride)
        ci = np.arange(0, nx, stride)
        coarse_idx = np.array([j * nx + i for j in cj for i in ci])
        Gc = np.ascontiguousarray(G[coarse_idx])
        r = max(1, stride // 2)
        W, widx = build_offset_window(tr, r)
        n_rows = Gc.shape[0] + W.shape[0]
        by = Gc.nbytes + W.nbytes

        agree, errs, exact_max, oob = 0, [], 0.0, 0
        agree_c, n_c = 0, 0
        for q, v in enumerate(res):
            sc = np.real(Gc @ v) / tr.hd
            k = int(np.argmax(sc))
            aj, ai = int(cj[k // len(ci)]), int(ci[k % len(ci)])
            A = Gc[k]  # the coarse row IS the anchor vector
            sw = np.real(W @ (A * v)) / tr.hd
            kw = int(np.argmax(sw))
            dj, di = widx[kw]
            bj, bi = aj + dj, ai + di
            if not (0 <= bj < ny and 0 <= bi < nx):
                oob += 1
                continue
            exact_max = max(exact_max,
                            abs(sw[kw] - float(np.real(G[bj * nx + bi] @ v) / tr.hd)))
            rj, ri = divmod(ref[q], nx)
            hit = (bj, bi) == (rj, ri)
            agree += int(hit)
            if conf[q]:
                n_c += 1
                agree_c += int(hit)
            errs.append(float(np.hypot(tr.gx[bi] - tr.gx[ri],
                                       tr.gy[bj] - tr.gy[rj])))

        t0 = time.perf_counter()
        for i in range(N_TIMING):
            v = res[i % len(res)]
            sc = np.real(Gc @ v) / tr.hd
            k = int(np.argmax(sc))
            _ = int(np.argmax(np.real(W @ (Gc[k] * v)) / tr.hd))
        ms = (time.perf_counter() - t0) / N_TIMING * 1e3

        n = max(1, len(errs))
        ac = agree_c / max(1, n_c)
        print("%7d %3d %7d %7.2f %8.3f %7.1fx %6.3f/%-3d %9.3f %10.4f"
              % (stride, r, n_rows, by / MB, ms, full_ms / ms, ac, n_c,
                 agree / n, np.percentile(errs, 95)))
        rows_out.append(dict(stride=stride, r=r, rows=n_rows, bytes=int(by),
                             ms=ms, speedup=full_ms / ms,
                             agree_confident=ac, n_confident=n_c,
                             agree_all=agree / n,
                             med_err_m=float(np.median(errs)),
                             p95_err_m=float(np.percentile(errs, 95)),
                             out_of_bounds=oob,
                             exact_max_absdiff=float(exact_max)))

    # Selection gate, pre-stated: a swap is only adoptable if it reproduces the
    # full-grid argmax essentially always. Anything less converts a cheap
    # correct answer into a cheap wrong one.
    GATE = 0.99
    passing = [d for d in rows_out if d["agree_confident"] >= GATE]
    print("\n-- adoption gate: agreement with full grid >= %.2f, on the queries" % GATE)
    print("   the trace would actually answer (n = %d) --" % int(conf.sum()))
    if not passing:
        best = max(rows_out, key=lambda d: d["agree_confident"])
        print("  NO CONFIGURATION PASSES. best agreement %.3f (stride %d)."
              % (best["agree_confident"], best["stride"]))
        print("  The windowed readout is EXACT once anchored (max|d| ~1e-9),")
        print("  so the failure is the ANCHOR, not the window.")
        # A window only wins if its radius covers the anchor error. Price that.
        for a in anchor_q:
            need = int(np.ceil(a["p95_cells"]))
            rows_need = (2 * need + 1) ** 2
            print("    stride %2d: p95 anchor error %.0f cells -> window r=%d "
                  "= %d rows vs %d full  (%s)"
                  % (a["stride"], a["p95_cells"], need, rows_need, ny * nx,
                     "LARGER THAN THE FULL GRID" if rows_need > ny * nx
                     else "smaller"))
    else:
        best = min(passing, key=lambda d: d["bytes"])
        # 0 misses out of n is not proof; quote the rule-of-three upper bound.
        ub = 3.0 / max(1, best["n_confident"])
        print("  PASSES: stride %d, %d rows, agreement %.3f on n=%d confident"
              % (best["stride"], best["rows"], best["agree_confident"],
                 best["n_confident"]))
        print("  95%% upper bound on the miss rate at this n: %.1f%% (rule of three)"
              % (100 * ub))
    new_total = (mem["traces"] + mem["class_codebook"] + mem["bases"]
                 + best["bytes"] + tr.C_mat.nbytes)
    print("\n-- deployed state if G_flat and T_axis are dropped for this config --")
    print("  before  %8.2f MB" % (mem["total"] / MB))
    print("  after   %8.2f MB   (stride %d, %d rows, agreement %.3f on n=%d)"
          % (new_total / MB, best["stride"], best["rows"],
             best["agree_confident"], best["n_confident"]))
    print("  ratio   %8.1fx smaller, %.1fx faster%s"
          % (mem["total"] / new_total, best["speedup"],
             "" if passing else "  <-- NOT ADOPTABLE, gate failed"))

    out = dict(trace=TRACE, hd=int(tr.hd), n_events=int(tr.n_events),
               grid=[ny, nx], n_residuals=len(res), n_timing=N_TIMING,
               memory_bytes_before={k: int(v) for k, v in mem.items()},
               full_grid_ms=full_ms, window_only_oracle_anchor=window_only,
               anchor_quality=anchor_q, two_stage=rows_out,
               gate=GATE, gate_passed=bool(passing), chosen=best,
               deployed_after_bytes=int(new_total))
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/bounded_readout_bench.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote outputs/bounded_readout_bench.json")


if __name__ == "__main__":
    main()
