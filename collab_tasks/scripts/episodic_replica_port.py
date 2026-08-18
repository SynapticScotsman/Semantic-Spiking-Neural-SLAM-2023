"""Port the episodic (what/where/WHEN) trace onto Replica.

Everything measured about episodic memory so far has run on ONE classroom
recording, whose frontend is YOLOv8n-COCO — which is why the query population
contained a cat, an airplane and a toilet. All the accuracy numbers we publish
are on Replica. This script puts the episodic trace on Replica so the two lines
share a dataset.

Replica observations carry a `frame` index (0..399), so a time axis exists and
the episodic trace can be built without inventing anything.

It also settles the accounting question the paper narrative turns on: what is
the TOTAL system memory, not the trace size. Three decode strategies are priced:

  materialised   the episodic default: a (grid*grid, hd) position codebook
  windowed       coarse lattice + re-anchored offset window (today's work)
  separable      exploit ctx_pos(x,y) = Bx^x * By^y and never build the grid
                 -- what collab_tasks/batch1/common.py already does on Replica

KNOWN SIMPLIFICATION, stated because it matters: TraceSet uses ONE positional
length scale for both axes, while the Replica line uses anisotropic LX=0.45 /
LY=0.27. The port therefore cannot reproduce the anisotropy that line found to
matter. Flagged, not hidden.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from vsa_cognitive_mapping.astm_traces import TraceSet, QueryRouter
from collab_tasks.batch1.common import load_scene, LX, LY, GRID, CAP

MB = 1024 ** 2
HD = 4096          # match the Replica line, not the classroom's 8192
SCENES = ["room0", "room1", "room2", "office0", "office1",
          "office2", "office3", "office4"]


def build(scene, hd=HD, grid=GRID, pos_l=None, time_l=None, cap=CAP):
    d = load_scene(scene)
    obs = d["obs"]
    names = sorted({o["cls"] for o in obs})
    xs = np.array([o["x"] for o in obs])
    ys = np.array([o["y"] for o in obs])
    fr = np.array([float(o["frame"]) for o in obs])
    bounds = (xs.min(), xs.max(), ys.min(), ys.max())
    t_max = float(fr.max())
    # geometric mean of the Replica line's anisotropic scales (see docstring)
    pos_l = pos_l if pos_l is not None else float(np.sqrt(LX * LY))
    # keep the classroom's ratio of time scale to sequence length
    time_l = time_l if time_l is not None else max(1.0, t_max * (20.0 / 1238.0))

    tr = TraceSet(hd, 0, pos_l, time_l, names, bounds, t_max, grid=grid)
    for o in obs:
        tr.add_event(o["cls"], o["x"], o["y"], float(o["frame"]), o["conf"])
    tr.finalize()
    ev = dict(**{"class": [o["cls"] for o in obs]}, x=xs, y=ys, t=fr,
              conf=np.array([o["conf"] for o in obs]))
    return tr, d, ev, dict(pos_l=pos_l, time_l=time_l, bounds=bounds,
                           t_max=t_max, n_obs=len(obs), n_classes=len(names))


def price(tr):
    """Three decode strategies, in bytes actually held."""
    ny, nx, hd = len(tr.gy), len(tr.gx), tr.hd
    itemsize = tr.G_flat.dtype.itemsize
    materialised = ny * nx * hd * itemsize
    # windowed: coarse lattice at stride 4 + (2*2+1)^2 offset window
    cn = len(np.arange(0, ny, 4)) * len(np.arange(0, nx, 4))
    windowed = (cn + 25) * hd * itemsize
    separable = (ny + nx) * hd * itemsize
    return dict(materialised=materialised, windowed=windowed,
                separable=separable)


def separable_field(tr, v, PX, PY):
    return np.real((PY * v[None, :]) @ PX.T) / tr.hd


TIME_SCALES = [20.0, 50.0, 120.0]


def main():
    only = sys.argv[1:] or SCENES
    rows = []
    for scene in only:
        for time_l in TIME_SCALES:
            tr, d, ev, meta = build(scene, time_l=time_l)
            ny, nx, hd = len(tr.gy), len(tr.gx), tr.hd
            l = tr.enc.pos_l
            PX = (tr.enc.Bx.values[None, :]
                  ** (tr.gx[:, None] / l)).astype(np.complex64)
            PY = (tr.enc.By.values[None, :]
                  ** (tr.gy[:, None] / l)).astype(np.complex64)

            # separable must reproduce the materialised field exactly
            rng = np.random.RandomState(0)
            v = (rng.randn(hd) + 1j * rng.randn(hd)).astype(np.complex64)
            full = (np.real(tr.G_flat @ v) / hd).reshape(ny, nx)
            dmax = float(np.abs(separable_field(tr, v, PX, PY) - full).max())

            tr.calibrate(ev)
            router = QueryRouter(tr)
            cls_arr = np.asarray(ev["class"])
            ts, xs, ys = (np.asarray(ev["t"]), np.asarray(ev["x"]),
                          np.asarray(ev["y"]))
            rs = np.random.RandomState(0)
            n_conf, n_q, err = 0, 0, []
            for c in sorted(set(cls_arr)):
                m = cls_arr == c
                if m.sum() < 3:
                    continue
                for t in rs.choice(ts[m], size=min(8, int(m.sum())),
                                   replace=False):
                    ans, info = router.query("where", what=c, when=float(t))
                    n_q += 1
                    if not info.get("confident", False):
                        continue
                    n_conf += 1
                    sel = m & (ts == t)
                    err.append(float(np.min(np.hypot(xs[sel] - ans[0],
                                                     ys[sel] - ans[1]))))

            p_ = price(tr)
            rows.append(dict(
                scene=scene, **meta, hd=hd, grid=[ny, nx],
                sep_vs_full_maxabs=dmax, n_queries=n_q, n_confident=n_conf,
                coverage=n_conf / max(1, n_q),
                err_med_m=float(np.median(err)) if err else None,
                err_p90_m=float(np.percentile(err, 90)) if err else None,
                trace_kb=sum(x.nbytes for x in tr.M.values()) / 1024,
                codebook_mb=sum(x.nbytes for x in tr.C.values()) / MB, **p_))
            print("%-9s tl=%-6.0f %5d obs %3d cls  cover %3d/%-3d = %4.0f%%  "
                  "err med %6s  p90 %6s  sep-vs-full %.0e"
                  % (scene, time_l, meta["n_obs"], meta["n_classes"], n_conf,
                     n_q, 100 * n_conf / max(1, n_q),
                     ("%.3f" % np.median(err)) if err else "n/a",
                     ("%.3f" % np.percentile(err, 90)) if err else "n/a", dmax))

    print(chr(10) + "=" * 78)
    print("COVERAGE vs ACCURACY -- 'where is <class> at frame t', 8 scenes")
    print("distance from the answer to the nearest true observation of that")
    print("class at that frame, metres. Coverage = fraction the trace answers.")
    print("=" * 78)
    print("%9s %10s %14s %12s %12s" % ("time scale", "coverage", "n answered",
                                       "err med m", "err p90 m"))
    for tl in TIME_SCALES:
        g = [r for r in rows if r["time_l"] == tl and r["err_med_m"] is not None]
        if not g:
            continue
        nq = sum(r["n_queries"] for r in g)
        nc = sum(r["n_confident"] for r in g)
        print("%9.0f %9.0f%% %14d %12.3f %12.3f"
              % (tl, 100 * nc / max(1, nq), nc,
                 float(np.median([r["err_med_m"] for r in g])),
                 float(np.median([r["err_p90_m"] for r in g]))))

    print(chr(10) + "=" * 78)
    print("DECODE STRATEGY and TOTAL SYSTEM MEMORY (per scene, hd=%d)" % HD)
    print("=" * 78)
    print("%-9s %13s %11s %11s %9s %11s %10s"
          % ("scene", "materialised", "windowed", "separable", "trace KB",
             "codebook MB", "TOTAL MB"))
    seen = set()
    for r in rows:
        if r["scene"] in seen:
            continue
        seen.add(r["scene"])
        tot = r["trace_kb"] / 1024 + r["codebook_mb"] + r["separable"] / MB
        r["total_mb"] = tot
        print("%-9s %10.1f MB %8.1f MB %8.2f MB %9.1f %11.2f %10.2f"
              % (r["scene"], r["materialised"] / MB, r["windowed"] / MB,
                 r["separable"] / MB, r["trace_kb"], r["codebook_mb"], tot))

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/episodic_replica_port.json", "w") as f:
        json.dump(dict(hd=HD, grid=GRID, cap=CAP, lx=LX, ly=LY,
                       time_scales=TIME_SCALES, rows=rows), f, indent=1)
    print(chr(10) + "wrote outputs/episodic_replica_port.json")


if __name__ == "__main__":
    main()
