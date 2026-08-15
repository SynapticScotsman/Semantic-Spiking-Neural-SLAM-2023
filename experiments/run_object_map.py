"""Object-centric VSA cognitive map -- end-to-end demo and measurements.

Builds a synthetic room of objects, walks a robot around them, fills an
:class:`~sspslam.objectmap.ObjectCentricMap` from the detections, and then
runs every read-out, reporting numbers rather than assertions:

1. *Where is this instance?*      unbind ID, correlate over space
2. *What is at this place?*       unbind S_allo, clean up the ID
3. *What does this side look like?*  unbind S_view(phi), clean up the key
4. *What should I see from here?* derive phi from the two poses, then (3)
5. *Orbit by delta with no image* bind S_view(delta) onto a view code
6. *The chair moved*              rewrite S_allo, keep the object file

It also measures the three design choices the map makes, so they are numbers
and not opinions:

* two memories vs one fully-bound blob (``ID (*) S_allo (*) V``)
* appearance keys as plain projections vs fractional-power-encoded
* keeping the object file vs re-walking after an object moves

Runs on numpy alone::

    python experiments/run_object_map.py
    python experiments/run_object_map.py --plot --save-dir data/objectmap
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sspslam.objectmap import (ObjectCentricMap, bind, cosine,  # noqa: E402
                               normalize, unbind, wrap_angle)

FEAT_DIM = 64


# ---------------------------------------------------------------------------
# A synthetic room whose objects genuinely look different from different sides
# ---------------------------------------------------------------------------

class SyntheticObject:
    """An object with a view-dependent appearance embedding.

    The embedding is a smooth function of viewing angle built from a handful
    of random "facet" directions, plus per-frame noise -- a stand-in for what
    a DINOv2 or CLIP crop embedding does as you walk around a real chair.
    """

    def __init__(self, name, class_name, position, yaw, rng, n_facets=6,
                 noise=0.15):
        self.name = name
        self.class_name = class_name
        self.position = np.asarray(position, dtype=float)
        self.yaw = float(yaw)
        self.noise = noise
        self.facet_angles = np.linspace(-np.pi, np.pi, n_facets, endpoint=False)
        self.facet_codes = rng.standard_normal((n_facets, FEAT_DIM))
        self.class_bias = rng.standard_normal(FEAT_DIM)
        self.rng = rng

    def embed(self, phi, noisy=True):
        """Crop embedding seen from object-centred azimuth ``phi``."""
        gap = wrap_angle(np.atleast_1d(phi)[0] - self.facet_angles)
        w = np.exp(-0.5 * (gap / 0.6) ** 2)
        z = w @ self.facet_codes + 0.5 * self.class_bias
        if noisy:
            z = z + self.noise * self.rng.standard_normal(FEAT_DIM)
        return z


def build_room(rng):
    specs = [
        ("chair_a", "chair", (-3.0, 2.0), 0.0),
        ("chair_b", "chair", (2.5, 3.0), 1.1),
        ("chair_c", "chair", (4.0, -2.5), -2.0),
        ("table_a", "table", (0.0, 0.0), 0.4),
        ("table_b", "table", (-4.0, -3.0), 2.3),
        ("bin_a", "bin", (3.5, 1.0), 0.0),
        ("bin_b", "bin", (-1.5, -3.5), 0.0),
        ("door_a", "door", (-5.0, 0.5), 1.57),
    ]
    return [SyntheticObject(n, c, p, y, rng) for n, c, p, y in specs]


def walk(objects, rng, n_steps=900, radius=5.5, sensor_range=4.0):
    """A loopy exploration path; yields detections as a robot would see them."""
    t = np.linspace(0, 6 * np.pi, n_steps)
    r = radius * (0.55 + 0.45 * np.sin(0.37 * t))
    path = np.stack([r * np.cos(t), r * np.sin(0.9 * t)], axis=1)
    headings = np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0]))

    detections = []
    for pos, yaw in zip(path, headings):
        for obj in objects:
            d = np.linalg.norm(obj.position - pos)
            if d > sensor_range:
                continue
            phi = wrap_angle(np.arctan2(*(pos - obj.position)[::-1]) - obj.yaw)
            detections.append(dict(
                robot_pos=pos, robot_yaw=float(yaw), truth=obj,
                obj_pos=obj.position + 0.05 * rng.standard_normal(2),
                phi=float(phi), embedding=obj.embed(phi),
            ))
    return path, detections


# ---------------------------------------------------------------------------
# Fill and query
# ---------------------------------------------------------------------------

def fill_map(objects, detections, seed=0, use_class=True, holdout=0.15):
    m = ObjectCentricMap(feat_dim=FEAT_DIM, ssp_dim=151, domain_dim=2,
                         bounds=[[-7, 7], [-7, 7]], length_scale=0.6,
                         grid_step=0.1, assoc_radius=1.2,
                         assoc_threshold=0.55, same_place_radius=0.4,
                         merge_tol=np.deg2rad(25.0), seed=seed)
    # Warm the appearance statistics on a sample of crops before minting keys,
    # so the centring is meaningful from the first detection.
    m.appearance.fit(np.stack([d["embedding"] for d in detections[:200]]))

    rng = np.random.default_rng(seed + 1)
    mask = rng.random(len(detections)) < holdout
    train = [d for d, h in zip(detections, mask) if not h]
    heldout = [d for d, h in zip(detections, mask) if h]

    truth_to_id = {}
    for det in train:
        oid, _ = m.observe(
            robot_pos=det["robot_pos"], obj_pos=det["obj_pos"],
            embedding=det["embedding"], robot_yaw=det["robot_yaw"],
            class_name=det["truth"].class_name if use_class else None,
            obj_yaw=det["truth"].yaw,
        )
        truth_to_id.setdefault(det["truth"].name, []).append(oid)
    return m, truth_to_id, heldout


def report_association(objects, truth_to_id):
    print("\n[1] Data association -- one file per physical instance")
    print(f"    {'truth':10s} {'files minted':>12s}  dominant id")
    n_split = 0
    mapping = {}
    for obj in objects:
        ids = truth_to_id.get(obj.name, [])
        uniq, counts = np.unique(ids, return_counts=True)
        dominant = uniq[np.argmax(counts)] if len(uniq) else "-"
        mapping[obj.name] = dominant
        n_split += max(len(uniq) - 1, 0)
        print(f"    {obj.name:10s} {len(uniq):12d}  {dominant}")
    print(f"    extra files beyond one per object: {n_split}")
    return mapping


def report_places(m, objects, mapping):
    print("\n[2] Where is this instance?  (unbind ID, correlate over space)")
    errs = []
    for obj in objects:
        q = m.where_is(mapping[obj.name])
        err = float(np.linalg.norm(q.position - obj.position))
        errs.append(err)
        print(f"    {obj.name:10s} true {np.round(obj.position, 2)}  "
              f"decoded {np.round(q.position, 2)}  err {err:.3f} m  "
              f"peak {q.score:.3f}")
    print(f"    mean error {np.mean(errs):.3f} m, max {np.max(errs):.3f} m")

    print("\n[3] What is at this place?  (unbind S_allo, clean up the ID)")
    hits = 0
    for obj in objects:
        top = m.what_is_at(obj.position, top_k=2)
        ok = top and top[0][0] == mapping[obj.name]
        hits += bool(ok)
        runner = f"{top[1][0]} {top[1][1]:+.2f}" if len(top) > 1 else "-"
        print(f"    {obj.name:10s} -> {top[0][0]:10s} {top[0][1]:+.3f}   "
              f"{'ok ' if ok else 'MISS'}  runner-up {runner}")
    print(f"    correct: {hits}/{len(objects)}")

    print("\n[4] Where are all the chairs?  (unbind CLASS from the class map)")
    for cls in ("chair", "table", "bin"):
        q = m.where_are(cls, return_heatmap=True)
        truth = np.stack([o.position for o in objects if o.class_name == cls])
        d = np.min(np.linalg.norm(truth - q.position, axis=1))
        print(f"    {cls:6s} peak at {np.round(q.position, 2)}  "
              f"{d:.3f} m from the nearest true {cls} "
              f"(of {len(truth)} in the room)")
    return float(np.mean(errs))


def report_views(m, objects, mapping, heldout, rng):
    print("\n[5] What does this side look like?  "
          "(unbind S_view(phi), clean up the key)")
    print(f"    {'object':10s} {'K':>3s}  {'cov':>5s}  own-view hit   "
          "cos(pred, fresh crop)   cos vs wrong side")
    same, cross, hits, total = [], [], 0, 0
    for obj in objects:
        oid = mapping[obj.name]
        stored = m.objects[oid]
        for entry in stored.views:
            phi = float(entry.phi[0])
            q = m.view_of(oid, phi, top_k=1)
            hits += bool(q.best and q.best[0][0] == oid)
            total += 1
            fresh = m.appearance.encode(obj.embed(phi))
            same.append(float(cosine(fresh, q.residue.reshape(1, -1))[0]))
            away = m.appearance.encode(obj.embed(wrap_angle(phi + np.pi)))
            cross.append(float(cosine(away, q.residue.reshape(1, -1))[0]))
        print(f"    {obj.name:10s} {stored.n_views:3d}  "
              f"{stored.coverage(m.view_space):5.2f}  "
              f"{'':13s}{np.mean(same[-stored.n_views:]):+.3f}"
              f"{'':17s}{np.mean(cross[-stored.n_views:]):+.3f}")
    print(f"    residue identifies the right object's view: {hits}/{total}")
    print(f"    mean cos to the crop actually seen there:  {np.mean(same):+.3f}")
    print(f"    mean cos to the crop from the far side:    {np.mean(cross):+.3f}")

    print("\n[6] What should I see from here?  (phi from the two poses)")
    print("    Held-out detections: poses the map was never written from.")
    print(f"    {'object':10s} {'n':>4s} {'cos(pred, crop seen there)':>27s}"
          f" {'cos vs another object':>22s} {'hit rate':>9s}")
    right, wrong, hits, n_tot = [], [], 0, 0
    other_ids = [mapping[o.name] for o in objects]
    for obj in objects:
        oid = mapping[obj.name]
        held = [d for d in heldout if d["truth"] is obj]
        if not held:
            continue
        r, w, h = [], [], 0
        for det in held:
            q = m.expected_view_from(oid, det["robot_pos"], det["robot_yaw"])
            seen = m.appearance.encode(det["embedding"])
            r.append(float(cosine(seen, q.residue.reshape(1, -1))[0]))
            distractor = next(i for i in other_ids if i != oid)
            qd = m.expected_view_from(distractor, det["robot_pos"],
                                      det["robot_yaw"])
            w.append(float(cosine(seen, qd.residue.reshape(1, -1))[0]))
            h += int(r[-1] > w[-1])
        right += r
        wrong += w
        hits += h
        n_tot += len(held)
        print(f"    {obj.name:10s} {len(held):4d} {np.mean(r):+27.3f}"
              f" {np.mean(w):+22.3f} {h / len(held):9.2f}")
    print(f"    all        {n_tot:4d} {np.mean(right):+27.3f}"
          f" {np.mean(wrong):+22.3f} {hits / n_tot:9.2f}")

    obj = objects[0]
    oid = mapping[obj.name]
    det = next(d for d in heldout if d["truth"] is obj)
    q = m.expected_view_from(oid, det["robot_pos"], det["robot_yaw"])
    print(f"    e.g. {obj.name} from {np.round(det['robot_pos'], 2)}: "
          f"view azimuth {np.rad2deg(q.phi[0]):+.1f} deg, "
          f"camera bearing {np.rad2deg(q.bearing):+.1f} deg")
    print("    The two are different quantities -- the first says which side of")
    print("    the object faces you, the second says where to point the camera.")
    print("    Neither is stored: both are computed from the two poses.")

    print("\n[7] Orbit by delta with no new image  (bind S_view(delta))")
    obj = objects[np.argmax([m.objects[mapping[o.name]].n_views
                             for o in objects])]
    oid = mapping[obj.name]
    book = m.objects[oid].view_book(m.view_space)
    stored = m.objects[oid].angles[:, 0]
    start = float(stored[0])
    print(f"    {obj.name}: reading the book at a fixed angle "
          f"{np.rad2deg(start):+.1f} deg and orbiting the object underneath it")
    print(f"    {'delta':>8s} {'lands on side':>14s} {'cos to that stored key':>24s}"
          f" {'best other key':>16s} {'rank':>6s}")
    base = unbind(book, m.view_space.encode([start]))
    ranks = []
    all_keys = m.objects[oid].keys
    for j, target in enumerate(stored):
        delta = wrap_angle(start - target)
        residue = unbind(m.orbit(book, [delta]), m.view_space.encode([start]))
        sims = cosine(residue, all_keys)
        rank = int(np.sum(sims > sims[j])) + 1
        ranks.append(rank)
        others = np.delete(sims, j)
        print(f"    {np.rad2deg(delta):+8.1f} {np.rad2deg(target):+14.1f}"
              f" {sims[j]:+24.3f} {others.max():+16.3f} {rank:5d}/{len(stored)}")
    print(f"    correct side ranked first for {int(np.sum(np.array(ranks) == 1))}"
          f"/{len(stored)} orbits")
    full = unbind(m.orbit(book, [2 * np.pi]), m.view_space.encode([start]))
    print(f"    a full turn returns the code exactly: "
          f"cos {float(cosine(base, full.reshape(1, -1))[0]):.6f}, "
          f"max abs diff {np.abs(base - full).max():.2e}")
    return float(np.mean(same)), float(np.mean(cross))


def report_design_choices(m, objects, mapping, detections, rng):
    print("\n[8] Two memories vs one fully-bound blob")
    print("    M_obj = ID (*) S_allo (*) V   vs   ID (*) unit(S_allo + V)")
    print(f"    {'object':10s} {'bundle: pos err':>16s} {'bind: pos err':>15s}"
          f" {'bundle: view cos':>18s} {'bind: view cos':>16s}")
    rows = []
    for obj in objects:
        oid = mapping[obj.name]
        entry = m.objects[oid].views[0]
        phi = entry.phi
        row = []
        for mode in ("bundle", "bind"):
            blob = m.object_vector(oid, mode=mode)
            pos_err = float(np.linalg.norm(
                m.where_is(oid, memory=blob).position - obj.position))
            key_code = bind(m.ids[oid], m.view_space.encode(phi))
            residue = unbind(blob, key_code)
            row += [pos_err, float(cosine(entry.key, residue.reshape(1, -1))[0])]
        rows.append(row)
        print(f"    {obj.name:10s} {row[0]:16.3f} {row[2]:15.3f}"
              f" {row[1]:+18.3f} {row[3]:+16.3f}")
    rows = np.array(rows)
    print(f"    mean       {rows[:, 0].mean():16.3f} {rows[:, 2].mean():15.3f}"
          f" {rows[:, 1].mean():+18.3f} {rows[:, 3].mean():+16.3f}")
    print("    Binding the view book into the spatial term replaces S_allo with")
    print("    S_allo (*) V, and V is random-looking, so the spatial peak goes.")
    print("    table_a is the exception that proves it: it sits at the origin,")
    print("    where S_allo is the binding identity, so binding it in is a")
    print("    no-op and the view read-out survives for that one object.")

    print("\n[9] Appearance keys: plain projection vs fractional-power encoding")
    print("    Does the key separate two sides of ONE object, or two objects")
    print("    that look alike?  FPE builds a similarity manifold, which is")
    print("    the wrong job here.")
    z_side_a = np.stack([o.embed(0.0, noisy=False) for o in objects])
    z_side_b = np.stack([o.embed(np.pi, noisy=False) for o in objects])
    plain_a = m.appearance.encode(z_side_a)
    plain_b = m.appearance.encode(z_side_b)
    W = m.rng.standard_normal((FEAT_DIM, m.dim)) / np.sqrt(FEAT_DIM)

    def fpe(z):
        z = np.atleast_2d(z) - m.appearance.mean
        return normalize(np.fft.ifft(np.exp(1j * (z @ W)), axis=1).real, axis=1)

    fpe_a, fpe_b = fpe(z_side_a), fpe(z_side_b)
    for name, A, B in (("plain projection", plain_a, plain_b),
                       ("FPE of embedding", fpe_a, fpe_b)):
        within = np.mean([float(cosine(A[i], B[i:i + 1])[0])
                          for i in range(len(objects))])
        off = np.array([[float(cosine(A[i], A[j:j + 1])[0])
                         for j in range(len(objects))] for i in range(len(objects))])
        between = float(off[~np.eye(len(objects), dtype=bool)].mean())
        print(f"    {name:18s} cos(front, back) of same object {within:+.3f}"
              f"   cos(front, front) across objects {between:+.3f}")
    print("    The key must keep the first number LOW (two sides are different")
    print("    entries) and the second LOW too (different objects are different")
    print("    keys).  FPE raises both: it merges by looks, which is exactly")
    print("    the collapse to avoid.")

    print("\n[10] The chair moved")
    obj = objects[0]
    oid = mapping[obj.name]
    new_pos = obj.position + np.array([2.5, -1.5])
    before = m.objects[oid].n_views
    m.move_object(oid, new_pos, timestamp=1.0)
    q = m.where_is(oid)
    phi = float(m.objects[oid].views[0].phi[0])
    v = m.view_of(oid, phi, top_k=1)
    fresh = m.appearance.encode(obj.embed(phi))
    print(f"    moved {obj.name} to {np.round(new_pos, 2)}: "
          f"decoded {np.round(q.position, 2)}, err "
          f"{np.linalg.norm(q.position - new_pos):.3f} m")
    print(f"    object file survived: {before} views before, "
          f"{m.objects[oid].n_views} after, "
          f"cos(prediction, crop) {float(cosine(fresh, v.residue.reshape(1, -1))[0]):+.3f}")
    print("    Only S_allo was rewritten.  A snapshot map keyed on where the")
    print("    robot stood would have to be re-walked.")
    m.move_object(oid, obj.position, timestamp=2.0)


def plot(m, objects, mapping, path, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    q = m.where_is(mapping[objects[0].name], return_heatmap=True)
    n = int(np.sqrt(q.grid.shape[0]))
    ax.pcolormesh(q.grid[:, 0].reshape(n, n), q.grid[:, 1].reshape(n, n),
                  q.heatmap.reshape(n, n), shading="auto", cmap="viridis",
                  vmin=0.0)
    ax.plot(path[:, 0], path[:, 1], color="w", lw=0.4, alpha=0.35)
    for o in objects:
        ax.plot(*o.position, "w+", ms=7)
    ax.plot(*objects[0].position, "r*", ms=14)
    ax.plot(*q.position, "wo", mfc="none", ms=13, mew=1.6)
    ax.set_title(f"unbind ID({objects[0].name}) from M_scene\n"
                 f"peak {q.score:.2f}, decoded circle on true star", fontsize=10)
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 1])
    q = m.where_are("chair", return_heatmap=True)
    ax.pcolormesh(q.grid[:, 0].reshape(n, n), q.grid[:, 1].reshape(n, n),
                  q.heatmap.reshape(n, n), shading="auto", cmap="magma")
    for o in objects:
        ax.plot(*o.position, "w*" if o.class_name == "chair" else "w+", ms=9)
    ax.set_title("unbind CLASS(chair): every chair lights up")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 2])
    deltas = np.linspace(-np.pi, np.pi, 361)
    for M in (4, 8, 16):
        from sspslam.objectmap import CircularSSPSpace
        sp = CircularSSPSpace(1, ssp_dim=m.dim, max_harmonic=M)
        ax.plot(np.rad2deg(deltas),
                sp.similarity_kernel(deltas.reshape(-1, 1)),
                label=f"max_harmonic={M}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("angle between viewpoints (deg)")
    ax.set_ylabel("S_view similarity")
    ax.set_title("the view circle wraps at +/-180")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0], projection="polar")
    for i, o in enumerate(objects[:4]):
        angles = m.objects[mapping[o.name]].angles[:, 0]
        ax.plot(angles, np.full_like(angles, i + 1), "o", ms=6, label=o.name)
    ax.set_title("stored sides per object (K stays small)", pad=22, fontsize=10)
    ax.set_yticklabels([])
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    obj = objects[0]
    oid = mapping[obj.name]
    probe = np.linspace(-np.pi, np.pi, 181)
    truth = np.stack([m.appearance.encode(obj.embed(p, noisy=False))
                      for p in probe])
    pred = np.stack([m.view_of(oid, p).residue for p in probe])
    ax.plot(np.rad2deg(probe),
            np.sum(normalize(truth, 1) * normalize(pred, 1), axis=1),
            label="cos(predicted, true crop key)")
    for a in m.objects[oid].angles[:, 0]:
        ax.axvline(np.rad2deg(a), color="grey", lw=0.6, ls=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("object-centred azimuth (deg)")
    ax.set_title(f"{obj.name}: prediction vs truth around the orbit")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    # Read each object's file at one of its own stored angles, then score the
    # residue against every object's key at that angle: the diagonal is the
    # signal, everything else is bundling cross-talk.
    ids = [mapping[o.name] for o in objects]
    probe_angles = [float(m.objects[i].angles[0, 0]) for i in ids]
    residues = [m.view_of(i, a).residue for i, a in zip(ids, probe_angles)]
    keys = [m.objects[i].views[0].key for i in ids]
    sims = np.array([[float(cosine(r, np.asarray(k).reshape(1, -1))[0])
                      for k in keys] for r in residues])
    im = ax.imshow(sims, cmap="coolwarm", vmin=-np.abs(sims).max(),
                   vmax=np.abs(sims).max())
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([o.name for o in objects], rotation=90, fontsize=7)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([o.name for o in objects], fontsize=7)
    ax.set_title("read object i's file -> whose key?\n(diagonal = signal, rest = cross-talk)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "object_map.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure written to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=900)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save-dir", default="data/objectmap")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    objects = build_room(rng)
    path, detections = walk(objects, rng, n_steps=args.n_steps)
    print(f"{len(objects)} objects, {len(detections)} detections from a "
          f"{args.n_steps}-step walk")

    m, truth_to_id, heldout = fill_map(objects, detections, seed=args.seed)
    print(m)
    print(f"view circle: max_harmonic={m.view_space.max_harmonic}, "
          f"half-width {np.rad2deg(m.view_space.lobe_width()):.1f} deg")

    mapping = report_association(objects, truth_to_id)
    report_places(m, objects, mapping)
    report_views(m, objects, mapping, heldout, rng)
    report_design_choices(m, objects, mapping, detections, rng)

    if args.plot:
        plot(m, objects, mapping, path, args.save_dir)


if __name__ == "__main__":
    main()
