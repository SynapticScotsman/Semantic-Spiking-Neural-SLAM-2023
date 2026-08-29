"""E1: is mirror-symmetric view tuning a useful stage, or just lost information?

FINDINGS.md sec.15 leg 4.  Freiwald & Tsao's face-patch hierarchy runs
ML/MF -> AL -> AM: view-specific, then **mirror-symmetric**, then
view-invariant.  AL neurons genuinely cannot tell a left profile from a right
one.  That is aliasing, in a brain, as a designed intermediate stage rather
than a defect -- and sec.13's cube is the same phenomenon.

If that stage earns its place, we should be able to build it in one line of
algebra and see it pay:

    book_ML = (1/K) sum_k  c(z_k) (*) S_view(phi_k)                 view-specific
    book_AL = (1/K) sum_k  c(z_k) (*) [S_view(phi_k) + S_view(2u - phi_k)]
    book_AM = (1/K) sum_k  c(z_k)                                   view-invariant

``u`` is the mirror axis; reflecting about it sends phi -> 2u - phi.  AM is the
honest end of the sequence: bundle the appearance and bind no angle at all.

The prediction the biology makes, stated so it can fail:

  * identification improves along ML -> AL -> AM
  * pose degrades along the same sequence, and AL degrades in a *specific* way
    -- it should keep the distance from the mirror axis and lose only the sign
  * a two-stage read-out (AL picks the object, ML picks the side) beats ML
    alone end to end

**Falsified if** identification does not improve at AL, in which case mirror
symmetry is only lost information here and Farzmahdi et al.'s CNN result does
not transfer to a bundled code.

sec.16 E0 sharpens the stakes: identification at d=151 is 0.55-0.64 against a
plain list's 0.97, so identification is the axis on which the object file is
losing worst, and AL is aimed straight at it.  E0 also showed the object file
is capacity-bound, and mirroring halves the number of distinct angles the
bundle has to hold -- so if AL helps, part of the help may be dimension rather
than biology.  The dimension sweep separates those.

Protocol is E0's, unchanged: blocked 30 deg arcs, statistics fitted on kept
views only, several seeds, hierarchical bootstrap over objects -> arcs ->
seeds.

    python experiments/run_mirror_stage.py
    python experiments/run_mirror_stage.py --dims 151 601 2401 --seeds 5
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import (CircularSSPSpace, bind, bundle,  # noqa: E402
                               cosine, wrap_angle)
from turntable_dataset import load_turntable  # noqa: E402
from run_view_localisation import encode_hog, fit_basis, condition  # noqa: E402
from run_blocked_split import blocked_masks  # noqa: E402
from run_nn_baseline import (pick_store, decode_vsa, hier_bootstrap,  # noqa: E402
                             DROP)
from run_frontend_diagnostics import diagnose  # noqa: E402

STAGES = ("ML", "AL", "AM")


# ---------------------------------------------------------------------------
# The three books.  One function, one line each -- that is the whole point.
# ---------------------------------------------------------------------------

def build_book(keys, angles, vs, stage, axis=0.0):
    if stage == "AM":
        return bundle(keys)
    if stage == "ML":
        codes = [vs.encode([a]) for a in angles]
    else:                                              # AL: phi and its mirror
        codes = [vs.encode([a]) + vs.encode([2 * axis - a]) for a in angles]
    return bundle(np.stack([bind(k, c) for k, c in zip(keys, codes)]))


def read_book(book, q, vs, stage):
    """Returns (angle estimate or None, per-query score)."""
    if stage == "AM":
        return None, cosine_rows(q, book)
    return decode_vsa(book, q, vs)


def cosine_rows(q, book):
    """Cosine of every query row against one book."""
    return np.asarray(cosine(book, q), dtype=float)


def folded(angle, axis):
    """Distance from the mirror axis -- the quantity AL should still carry."""
    return np.abs(wrap_angle(angle - axis))


# ---------------------------------------------------------------------------
# One run: every stage, every object, one seed
# ---------------------------------------------------------------------------

def one_run(Z, obj, az, kept, held, names, K, d, kmax, seed, axis):
    n_obj = len(names)
    mu, V, lam = fit_basis(Z[kept])
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
    keys = condition(Z, mu, V, lam, W, drop=DROP)
    vs = CircularSSPSpace(1, ssp_dim=d, max_harmonic=kmax,
                          rng=np.random.default_rng(seed + 1000))

    books = {s: [] for s in STAGES}
    for o in range(n_obj):
        sel = np.where((obj == o) & kept)[0]
        pick = sel[pick_store(az[sel], K)]
        for s in STAGES:
            books[s].append(build_book(keys[pick], az[pick], vs, s, axis))

    q, true, truth_obj = keys[held], az[held], obj[held]
    ang = {s: np.zeros((len(q), n_obj)) for s in STAGES}
    score = {s: np.zeros((len(q), n_obj)) for s in STAGES}
    for s in STAGES:
        for o in range(n_obj):
            a, sc = read_book(books[s][o], q, vs, s)
            score[s][:, o] = sc
            if a is not None:
                ang[s][:, o] = a

    out = {}
    rows = np.arange(len(q))
    for s in STAGES:
        hit = (score[s].argmax(1) == truth_obj)
        if s == "AM":
            out[s] = dict(hit=hit, signed=None, unsigned=None, side=None)
            continue
        est = ang[s][rows, truth_obj]
        out[s] = dict(
            hit=hit,
            signed=np.abs(np.rad2deg(wrap_angle(est - true))),
            unsigned=np.abs(np.rad2deg(folded(est, axis) - folded(true, axis))),
            side=(np.sign(wrap_angle(est - axis))
                  == np.sign(wrap_angle(true - axis))),
        )

    # two-stage: AL names the object, ML reads its side.  The end-to-end
    # number, which is what a robot would actually run.
    for picker in ("ML", "AL", "AM"):
        chosen = score[picker].argmax(1)
        est = ang["ML"][rows, chosen]
        e = np.abs(np.rad2deg(wrap_angle(est - true)))
        e[chosen != truth_obj] = 90.0        # wrong object: no credit
        out[f"{picker}->ML"] = dict(hit=(chosen == truth_obj), signed=e,
                                    unsigned=None, side=None)
    return out


def summarise(runs, key, field):
    vals = [r[key][field] for r in runs if r[key][field] is not None]
    return np.concatenate(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-views", type=int, default=72)
    ap.add_argument("--gap-deg", type=float, default=30.0)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dims", type=int, nargs="+", default=[151, 601, 2401])
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--axis-deg", type=float, default=0.0)
    ap.add_argument("--symmetric-set", action="store_true",
                    help="append the four by-construction symmetric objects, "
                         "so [F] has a group to stratify on rather than one "
                         "cube")
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(
        n_views=args.n_views, extra="symmetric" if args.symmetric_set else None)
    Z = encode_hog(imgs)
    nv, n_obj = args.n_views, len(names)
    vi = np.concatenate([np.arange(nv)] * n_obj)
    kept, held = blocked_masks(vi, nv, args.gap_deg)
    period = 2 * max(int(round(args.gap_deg / (360.0 / nv))), 1)
    obj_h, arc_h = obj[held], (vi // period)[held]
    axis = np.deg2rad(args.axis_deg)

    print(f"{len(Z)} crops, {Z.shape[1]}-D HOG, {n_obj} objects x {nv} azimuths")
    print(f"blocked split, {args.gap_deg:.0f} deg arcs; K={args.K} on file; "
          f"{args.seeds} seeds; mirror axis {args.axis_deg:.0f} deg")
    print("ML view-specific / AL mirror-symmetric / AM view-invariant\n")

    print("[A] the three stages, at each dimension")
    hdr = (f"  {'d':>5s} {'stage':>6s} {'ID hit':>7s} {'pose err':>9s} "
           f"{'<15deg':>7s} {'unsigned err':>13s} {'sign acc':>9s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    store = {}
    for d in args.dims:
        runs = [one_run(Z, obj, az, kept, held, names, args.K, d, args.kmax,
                        s, axis) for s in range(args.seeds)]
        store[d] = runs
        for s in STAGES:
            hit = summarise(runs, s, "hit")
            sg = summarise(runs, s, "signed")
            un = summarise(runs, s, "unsigned")
            if sg is None:
                print(f"  {d:5d} {s:>6s} {hit.mean():7.2f} {'--':>9s} "
                      f"{'--':>7s} {'--':>13s} {'--':>9s}")
                continue
            sign_acc = float(summarise(runs, s, "side").mean())
            print(f"  {d:5d} {s:>6s} {hit.mean():7.2f} {np.median(sg):8.1f}d "
                  f"{(sg < 15).mean():7.2f} {np.median(un):12.1f}d "
                  f"{sign_acc:9.2f}")
    print("  unsigned err ignores which side of the mirror axis: it is the "
          "error in\n  |phi - axis|, the quantity AL is supposed to keep. "
          "sign acc is how often\n  it lands on the correct side of the "
          "axis -- AL should be at 0.50, by\n  construction, and ML well "
          "above it.")

    print("\n[B] end to end: who names the object, ML always reads the side")
    hdr = f"  {'d':>5s} {'pipeline':>10s} {'ID hit':>7s} {'pose err':>9s} {'<15deg':>7s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for d in args.dims:
        for p in ("ML->ML", "AL->ML", "AM->ML"):
            runs = store[d]
            hit = summarise(runs, p, "hit")
            sg = summarise(runs, p, "signed")
            print(f"  {d:5d} {p:>10s} {hit.mean():7.2f} {np.median(sg):8.1f}d "
                  f"{(sg < 15).mean():7.2f}")
    print("  a wrong object is scored at 90 deg (chance), so this is the "
          "number a\n  robot would actually live with, not a pose figure "
          "conditioned on\n  already knowing what it is looking at.")

    print(f"\n[C] does AL earn its place? hierarchical bootstrap "
          f"(objects -> arcs -> seeds, {args.n_boot} draws)")
    print(f"  {'d':>5s} {'contrast':>24s} {'diff':>9s} {'95% CI':>18s}")
    print("  " + "-" * 60)
    for d in args.dims:
        runs = store[d]
        pairs = [("AL", "ML", "hit", np.mean, "ID hit, AL - ML"),
                 ("AM", "AL", "hit", np.mean, "ID hit, AM - AL"),
                 ("AL->ML", "ML->ML", "signed", np.median,
                  "end-to-end deg, AL - ML")]
        for a, b, field, stat, label in pairs:
            da = [r[a][field].astype(float) for r in runs]
            db = [r[b][field].astype(float) for r in runs]
            lo, hi = hier_bootstrap(da, obj_h, arc_h, n_obj,
                                    n_boot=args.n_boot, stat=stat,
                                    per_seed_b=db)
            m = stat(np.concatenate(da)) - stat(np.concatenate(db))
            tag = "" if lo <= 0 <= hi else "  <-- real"
            print(f"  {d:5d} {label:>24s} {m:+9.3f} "
                  f"[{lo:+7.3f},{hi:+7.3f}]{tag}")
    print("  for ID hit, positive = the later stage identifies BETTER.")
    print("  for end-to-end degrees, negative = AL naming the object HELPS.")

    print(f"\n[D] per object at d={args.dims[-1]} (K={args.K})")
    runs = store[args.dims[-1]]
    print(f"  {'object':>9s} {'ML ID':>6s} {'AL ID':>6s} {'AM ID':>6s} "
          f"{'ML pose':>8s} {'AL pose':>8s} {'AL unsigned':>12s}")
    print("  " + "-" * 64)
    for i, nm in enumerate(names):
        m = obj_h == i
        row = [np.concatenate([r[s]["hit"][m] for r in runs]).mean()
               for s in STAGES]
        mlp = np.median(np.concatenate([r["ML"]["signed"][m] for r in runs]))
        alp = np.median(np.concatenate([r["AL"]["signed"][m] for r in runs]))
        alu = np.median(np.concatenate([r["AL"]["unsigned"][m] for r in runs]))
        print(f"  {nm:>9s} {row[0]:6.2f} {row[1]:6.2f} {row[2]:6.2f} "
              f"{mlp:7.1f}d {alp:7.1f}d {alu:11.1f}d")
    print("  the cube is already 4-fold aliased, so mirroring should cost it "
          "nothing.\n  a chiral object (L_block) should lose the most pose "
          "and gain the most ID.")

    axis_sweep(Z, obj, az, kept, held, names, args)
    symmetry_split(Z, obj, az, kept, held, names, args, store, obj_h, arc_h)


# ---------------------------------------------------------------------------
# [E] The mirror axis is a choice.  If the effect only appears at axis=0 it is
# an artefact of where the turntable happens to start, not a stage.
# ---------------------------------------------------------------------------

def axis_sweep(Z, obj, az, kept, held, names, args):
    d = args.dims[-1]
    print(f"\n[E] mirror axis sweep at d={d}: is the effect about mirroring, "
          f"or about 0 deg?")
    print(f"  {'axis':>6s} {'AL ID hit':>10s} {'ML ID hit':>10s} "
          f"{'AL pose':>9s} {'AL unsigned':>12s}")
    print("  " + "-" * 52)
    for ax_deg in (0.0, 30.0, 60.0, 90.0):
        runs = [one_run(Z, obj, az, kept, held, names, args.K, d, args.kmax,
                        s, np.deg2rad(ax_deg)) for s in range(args.seeds)]
        al = summarise(runs, "AL", "hit").mean()
        ml = summarise(runs, "ML", "hit").mean()
        sg = np.median(summarise(runs, "AL", "signed"))
        un = np.median(summarise(runs, "AL", "unsigned"))
        print(f"  {ax_deg:5.0f}d {al:10.2f} {ml:10.2f} {sg:8.1f}d "
              f"{un:11.1f}d")
    print("  ML does not depend on the axis, so its column is the control: it "
          "should\n  be flat to seed noise.")


# ---------------------------------------------------------------------------
# [F] The regime question.  Farzmahdi et al. do not claim mirror tuning helps
# for everything -- they show it emerges from training on **symmetric**
# categories.  Faces are near-symmetric, so left/right is nuisance variation
# there.  Most objects on this turntable are chiral, where left/right is the
# signal.  Splitting on the sec.13 alias peak, which is *measured* from the
# descriptors and not asserted here, asks whether AL helps in the regime it
# was proposed for.
# ---------------------------------------------------------------------------

def symmetry_split(Z, obj, az, kept, held, names, args, store, obj_h, arc_h):
    d = args.dims[-1]
    runs = store[d]
    n_obj = len(names)
    mu, V, lam = fit_basis(Z[kept])
    rng = np.random.default_rng(0)
    W = rng.standard_normal((V.shape[0], d)) / np.sqrt(V.shape[0])
    keys = condition(Z, mu, V, lam, W, drop=DROP)
    vs = CircularSSPSpace(1, ssp_dim=d, max_harmonic=args.kmax,
                          rng=np.random.default_rng(1000))
    diag = diagnose(keys, obj, az, names, args.n_views,
                    np.rad2deg(vs.lobe_width()))

    peaks = np.array([r["alias_peak"] for r in diag])
    thresh = 0.75
    sym = peaks >= thresh

    print(f"\n[F] does AL help where it is supposed to? split on the sec.13 "
          f"alias peak\n    (measured from the descriptors, threshold "
          f"{thresh:.2f})")
    print(f"  {'object':>9s} {'alias peak':>11s} {'at lag':>7s} {'group':>10s} "
          f"{'ML ID':>6s} {'AL ID':>6s} {'AL - ML':>8s}")
    print("  " + "-" * 64)
    for i, nm in enumerate(names):
        m = obj_h == i
        ml = np.concatenate([r["ML"]["hit"][m] for r in runs]).mean()
        al = np.concatenate([r["AL"]["hit"][m] for r in runs]).mean()
        print(f"  {nm:>9s} {diag[i]['alias_peak']:11.3f} "
              f"{diag[i]['alias_lag']:6.0f}d {'symmetric' if sym[i] else 'chiral':>10s} "
              f"{ml:6.2f} {al:6.2f} {al - ml:+8.2f}")

    for label, group in (("symmetric", np.where(sym)[0]),
                         ("chiral", np.where(~sym)[0])):
        if len(group) == 0:
            print(f"  {label}: no objects at this threshold")
            continue
        m = np.isin(obj_h, group)
        da = [r["AL"]["hit"][m].astype(float) for r in runs]
        db = [r["ML"]["hit"][m].astype(float) for r in runs]
        pt = np.concatenate(da).mean() - np.concatenate(db).mean()
        if len(group) >= 3:
            lo, hi = hier_bootstrap(
                da, obj_h[m], arc_h[m], len(group), n_boot=args.n_boot,
                stat=np.mean, per_seed_b=db,
                obj_ids=group)
            ci = f"[{lo:+6.3f},{hi:+6.3f}]"
            tag = "" if lo <= 0 <= hi else "  <-- real"
        else:
            ci, tag = "[too few objects to resample]", ""
        print(f"  {label:>10s} ({len(group)} objects): AL - ML ID hit "
              f"{pt:+.3f} {ci}{tag}")
    if not args.symmetric_set:
        print("  only one symmetric object in the default set -- rerun with "
              "--symmetric-set\n  for a group large enough to resample over.")



if __name__ == "__main__":
    main()
