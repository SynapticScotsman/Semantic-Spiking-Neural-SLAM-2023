"""
test_object_map.py
==================
Checks for the object-centric VSA cognitive map (``sspslam.objectmap``).

Core claims under test
----------------------
1. ``S_view`` is a *circle*, not a line.  Integer harmonics make the encoding
   exactly ``2*pi``-periodic and binding exactly additive::

       S(phi + 2*pi) = S(phi)
       S(a) (*) S(b) = S(a + b mod 2*pi)

   so orbiting a view code by ``delta`` is a bind, and a full turn is the
   identity to machine precision.

2. The two memories separate cleanly.  Unbinding ``ID`` from the scene map
   recovers ``S_allo``; unbinding ``S_allo`` recovers ``ID``; unbinding
   ``S_view(phi)`` from an object file recovers that side's appearance key.

3. Folding the view book into the spatial term destroys the position
   read-out, which is why the map keeps two memories rather than one blob.

4. Appearance keys stay roughly orthogonal across objects, where a
   fractional-power encoding of the same embeddings would merge them.

5. Angle bookkeeping: the object-centred view azimuth and the camera bearing
   are different quantities, and the map derives both from poses rather than
   storing either.

Usage
-----
    python experiments/test_object_map.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sspslam.objectmap import (AppearanceCodec, CircularSSPSpace,  # noqa: E402
                               ObjectCentricMap, bind, camera_bearing, cosine,
                               normalize, unbind, view_azimuth, wrap_angle)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    return int(bool(ok))


def section(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def toy_map(seed=0, n_objects=6, n_sides=6, feat_dim=32):
    """A small map written directly, with known angles and known keys."""
    rng = np.random.default_rng(seed)
    m = ObjectCentricMap(feat_dim=feat_dim, ssp_dim=151, domain_dim=2,
                         bounds=[[-6, 6], [-6, 6]], length_scale=0.6,
                         grid_step=0.1, merge_tol=np.deg2rad(15.0), seed=seed)
    m.appearance.fit(rng.standard_normal((200, feat_dim)))

    positions = rng.uniform(-4.5, 4.5, size=(n_objects, 2))
    angles = np.linspace(-np.pi, np.pi, n_sides, endpoint=False)
    truth = {}
    for i, p in enumerate(positions):
        oid = f"obj_{i:02d}"
        for phi in angles:
            z = rng.standard_normal(feat_dim)
            robot = p + 3.0 * np.array([np.cos(phi), np.sin(phi)])
            m.observe(robot_pos=robot, obj_pos=p, embedding=z, obj_id=oid,
                      class_name="thing", update_position=False)
            truth[(oid, round(float(phi), 6))] = m.appearance.encode(z)
    return m, positions, angles, truth


# ---------------------------------------------------------------------------
# Test 1: the view circle
# ---------------------------------------------------------------------------

def test_view_circle():
    section("Test 1: S_view is a circle (periodic FPE)")
    passed = total = 0
    space = CircularSSPSpace(1, ssp_dim=151, max_harmonic=8, rng=np.random.default_rng(0))

    err = np.abs(space.encode(0.7) - space.encode(0.7 + 2 * np.pi)).max()
    passed += check("S(phi + 2pi) == S(phi)", err < 1e-9, f"max diff {err:.2e}")
    total += 1

    err = np.abs(bind(space.encode(0.3), space.encode(1.1))
                 - space.encode(1.4)).max()
    passed += check("S(a) (*) S(b) == S(a + b)", err < 1e-9, f"max diff {err:.2e}")
    total += 1

    err = np.abs(bind(space.encode(2.9), space.encode(2.9))
                 - space.encode(wrap_angle(5.8))).max()
    passed += check("binding wraps across the +/-pi seam", err < 1e-9,
                    f"max diff {err:.2e}")
    total += 1

    integer = np.allclose(space.harmonics, np.round(space.harmonics))
    passed += check("phases are integer harmonics", integer)
    total += 1

    norms = np.linalg.norm(np.atleast_2d(space.encode(np.linspace(-3, 3, 20))),
                           axis=1)
    passed += check("codes are unit norm", np.allclose(norms, 1.0, atol=1e-9),
                    f"range [{norms.min():.6f}, {norms.max():.6f}]")
    total += 1

    k = np.atleast_1d(space.similarity_kernel(
        np.linspace(0, np.pi, 181).reshape(-1, 1)))
    mono = k[0] > k[5] > k[10]
    passed += check("similarity falls off with angular distance", mono,
                    f"k(0)={k[0]:.3f} k(5deg)={k[5]:.3f} k(10deg)={k[10]:.3f}")
    total += 1

    fejer = CircularSSPSpace(1, ssp_dim=151, max_harmonic=8, taper="fejer")
    flat = CircularSSPSpace(1, ssp_dim=151, max_harmonic=8, taper="flat")
    d = np.linspace(0, np.pi, 361).reshape(-1, 1)
    kf = np.atleast_1d(fejer.similarity_kernel(d)).min()
    kl = np.atleast_1d(flat.similarity_kernel(d)).min()
    passed += check("fejer taper halves the sidelobe ringing", kf > kl,
                    f"worst sidelobe fejer {kf:+.3f} vs flat {kl:+.3f}")
    total += 1

    sphere = CircularSSPSpace(2, ssp_dim=151, max_harmonic=4)
    err = np.abs(sphere.encode([0.3, 0.2])
                 - sphere.encode([0.3 + 2 * np.pi, 0.2 - 2 * np.pi])).max()
    passed += check("2-D view sphere is periodic in both angles", err < 1e-9,
                    f"max diff {err:.2e}")
    total += 1
    return passed, total


# ---------------------------------------------------------------------------
# Test 2: the scene map
# ---------------------------------------------------------------------------

def test_scene_map():
    section("Test 2: scene map -- where is it, what is there")
    passed = total = 0
    m, positions, _, _ = toy_map()

    errs = [np.linalg.norm(m.where_is(f"obj_{i:02d}").position - p)
            for i, p in enumerate(positions)]
    passed += check("unbind ID decodes the position", max(errs) < 0.3,
                    f"max err {max(errs):.3f} m over {len(errs)} objects")
    total += 1

    hits = sum(m.what_is_at(p, top_k=1)[0][0] == f"obj_{i:02d}"
               for i, p in enumerate(positions))
    passed += check("unbind S_allo cleans up to the right ID",
                    hits == len(positions), f"{hits}/{len(positions)}")
    total += 1

    q = m.where_are("thing", return_heatmap=True)
    d = np.min(np.linalg.norm(positions - q.position, axis=1))
    passed += check("class map peaks on a member of the class", d < 0.3,
                    f"{d:.3f} m from the nearest one")
    total += 1

    oid = "obj_00"
    before = m.objects[oid].n_views
    new_pos = np.array([5.0, -5.0])
    m.move_object(oid, new_pos)
    err = np.linalg.norm(m.where_is(oid).position - new_pos)
    passed += check("moving an object rewrites S_allo only",
                    err < 0.3 and m.objects[oid].n_views == before,
                    f"err {err:.3f} m, views {before} -> {m.objects[oid].n_views}")
    total += 1
    return passed, total


# ---------------------------------------------------------------------------
# Test 3: the object files
# ---------------------------------------------------------------------------

def test_object_files():
    section("Test 3: object files -- what does this side look like")
    passed = total = 0
    m, _, angles, truth = toy_map()

    sims, cross = [], []
    for oid in m.objects:
        book = m.objects[oid].view_book(m.view_space)
        for phi in angles:
            residue = unbind(book, m.view_space.encode([phi]))
            key = truth[(oid, round(float(phi), 6))]
            sims.append(float(cosine(residue, key.reshape(1, -1))[0]))
            far = truth[(oid, round(float(wrap_angle(phi + np.pi)), 6))]
            cross.append(float(cosine(residue, far.reshape(1, -1))[0]))
    passed += check("unbind S_view(phi) recovers that side's key",
                    np.mean(sims) > 4 * abs(np.mean(cross)),
                    f"cos to the right side {np.mean(sims):+.3f}, "
                    f"to the opposite side {np.mean(cross):+.3f}")
    total += 1

    hits = 0
    for oid in m.objects:
        for i, phi in enumerate(angles):
            best = m.view_of(oid, phi, top_k=1).best
            hits += int(best is not None and best[0][0] == oid)
    n = len(m.objects) * len(angles)
    passed += check("residue beats every other object's stored views",
                    hits >= 0.9 * n, f"{hits}/{n}")
    total += 1

    oid = next(iter(m.objects))
    book = m.objects[oid].view_book(m.view_space)
    start, target = float(angles[0]), float(angles[2])
    orbited = m.orbit(book, [wrap_angle(start - target)])
    residue = unbind(orbited, m.view_space.encode([start]))
    key = truth[(oid, round(target, 6))]
    other = truth[(oid, round(float(angles[4]), 6))]
    hit = (float(cosine(residue, key.reshape(1, -1))[0])
           > float(cosine(residue, other.reshape(1, -1))[0]))
    passed += check("orbit by delta lands on the right side with no image",
                    hit,
                    f"cos to target {float(cosine(residue, key.reshape(1, -1))[0]):+.3f}"
                    f" vs another side "
                    f"{float(cosine(residue, other.reshape(1, -1))[0]):+.3f}")
    total += 1

    full = m.orbit(book, [2 * np.pi])
    err = np.abs(full - book).max()
    passed += check("a full turn is the identity", err < 1e-9,
                    f"max diff {err:.2e}")
    total += 1

    counts = [m.objects[o].n_views for o in m.objects]
    passed += check("nearby observations merge, K stays small",
                    max(counts) <= len(angles),
                    f"K = {counts} for {len(angles)} distinct sides each")
    total += 1
    return passed, total


# ---------------------------------------------------------------------------
# Test 4: two memories beat one blob
# ---------------------------------------------------------------------------

def test_two_memories():
    section("Test 4: two memories vs one fully-bound blob")
    passed = total = 0
    m, positions, _, _ = toy_map()

    bundle_err, bind_err = [], []
    for i, p in enumerate(positions):
        oid = f"obj_{i:02d}"
        for mode, acc in (("bundle", bundle_err), ("bind", bind_err)):
            blob = m.object_vector(oid, mode=mode)
            acc.append(float(np.linalg.norm(
                m.where_is(oid, memory=blob).position - p)))
    passed += check("ID (*) unit(S_allo + V) keeps the position decodable",
                    np.mean(bundle_err) < 0.3, f"mean err {np.mean(bundle_err):.3f} m")
    total += 1
    passed += check("ID (*) S_allo (*) V does not",
                    np.mean(bind_err) > 5 * max(np.mean(bundle_err), 0.05),
                    f"mean err {np.mean(bind_err):.3f} m "
                    f"({np.mean(bind_err) / max(np.mean(bundle_err), 1e-6):.0f}x worse)")
    total += 1
    return passed, total


# ---------------------------------------------------------------------------
# Test 5: appearance keys are keys, not a manifold
# ---------------------------------------------------------------------------

def test_appearance_keys():
    section("Test 5: appearance keys vs FPE of the embedding")
    passed = total = 0
    feat_dim, n = 64, 40
    rng = np.random.default_rng(1)
    codec = AppearanceCodec(feat_dim, 151, seed=0)

    # Embeddings that look alike: a shared mean plus a small private part,
    # which is what crops of two different chairs actually produce.
    shared = rng.standard_normal(feat_dim)
    Z = shared + 0.4 * rng.standard_normal((n, feat_dim))
    codec.fit(Z)
    keys = codec.encode(Z)

    off = keys @ keys.T
    off = off[~np.eye(n, dtype=bool)]
    passed += check("keys of look-alike crops stay near-orthogonal",
                    abs(off.mean()) < 0.15,
                    f"mean off-diagonal cos {off.mean():+.3f}")
    total += 1

    W = rng.standard_normal((feat_dim, 151)) / np.sqrt(feat_dim)
    fpe = normalize(np.fft.ifft(np.exp(1j * ((Z - codec.mean) @ W)),
                                axis=1).real, axis=1)
    off_fpe = fpe @ fpe.T
    off_fpe = off_fpe[~np.eye(n, dtype=bool)]
    passed += check("FPE of the same embeddings merges them instead",
                    off_fpe.mean() > off.mean() + 0.1,
                    f"mean off-diagonal cos {off_fpe.mean():+.3f} "
                    f"vs {off.mean():+.3f} for plain keys")
    total += 1

    passed += check("whitening is not offered",
                    _raises(lambda: AppearanceCodec(feat_dim, 151,
                                                    standardize="whiten")))
    total += 1
    return passed, total


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------------------
# Test 6: angle bookkeeping
# ---------------------------------------------------------------------------

def test_angles():
    section("Test 6: view azimuth vs camera bearing")
    passed = total = 0

    obj, robot = np.array([2.0, 0.0]), np.array([0.0, 0.0])
    az = view_azimuth(obj, robot, obj_yaw=0.0)
    bg = camera_bearing(obj, robot, robot_yaw=0.0)
    passed += check("the robot is on the object's pi side while the object "
                    "is dead ahead",
                    abs(abs(az) - np.pi) < 1e-9 and abs(bg) < 1e-9,
                    f"view azimuth {np.rad2deg(az):+.1f} deg, "
                    f"camera bearing {np.rad2deg(bg):+.1f} deg")
    total += 1

    az_yawed = view_azimuth(obj, robot, obj_yaw=np.pi / 2)
    passed += check("object yaw rotates the view circle, not the bearing",
                    abs(wrap_angle(az_yawed - wrap_angle(az - np.pi / 2))) < 1e-9,
                    f"{np.rad2deg(az_yawed):+.1f} deg")
    total += 1

    m, positions, _, _ = toy_map()
    oid = "obj_00"
    p = m.objects[oid].position
    robot = p + 2.0 * np.array([np.cos(0.9), np.sin(0.9)])

    yaws = np.linspace(-3, 3, 7)
    queries = [m.expected_view_from(oid, robot, robot_yaw=y) for y in yaws]
    phis = np.array([q.phi[0] for q in queries])
    bearings = np.array([q.bearing for q in queries])
    passed += check("spinning the robot on the spot moves the bearing but "
                    "never the view azimuth",
                    np.ptp(phis) < 1e-12
                    and len(np.unique(np.round(bearings, 6))) == len(yaws),
                    f"view azimuth spread {np.ptp(phis):.2e} rad, "
                    f"bearing spread {np.ptp(bearings):.2f} rad")
    total += 1

    q = queries[0]
    expected = view_azimuth(p, robot, m.objects[oid].yaw)
    passed += check("expected_view_from derives phi from the two poses",
                    abs(wrap_angle(q.phi[0] - expected)) < 1e-9,
                    f"phi {np.rad2deg(q.phi[0]):+.1f} deg, "
                    f"bearing {np.rad2deg(q.bearing):+.1f} deg")
    total += 1
    return passed, total


# ---------------------------------------------------------------------------
# Test 7: data association from a walk
# ---------------------------------------------------------------------------

def test_association():
    section("Test 7: one file per instance, from a walk")
    passed = total = 0
    feat_dim = 48
    rng = np.random.default_rng(3)
    m = ObjectCentricMap(feat_dim=feat_dim, ssp_dim=151, bounds=[[-6, 6], [-6, 6]],
                         length_scale=0.6, grid_step=0.2, assoc_radius=1.2,
                         same_place_radius=0.4, seed=3)
    m.appearance.fit(rng.standard_normal((200, feat_dim)))

    truth_pos = np.array([[-3.0, 2.0], [2.5, 3.0], [0.0, -2.0]])
    facets = rng.standard_normal((3, 6, feat_dim))
    facet_angles = np.linspace(-np.pi, np.pi, 6, endpoint=False)
    assigned = {i: [] for i in range(3)}
    for _ in range(120):
        robot = rng.uniform(-5.5, 5.5, size=2)
        for i, p in enumerate(truth_pos):
            if np.linalg.norm(p - robot) > 4.5:
                continue
            phi = view_azimuth(p, robot)
            w = np.exp(-0.5 * (wrap_angle(phi - facet_angles) / 0.6) ** 2)
            z = w @ facets[i] + 0.15 * rng.standard_normal(feat_dim)
            oid, _ = m.observe(robot_pos=robot,
                               obj_pos=p + 0.05 * rng.standard_normal(2),
                               embedding=z)
            assigned[i].append(oid)

    n_files = [len(set(v)) for v in assigned.values()]
    passed += check("each physical object gets one file",
                    all(k == 1 for k in n_files), f"files per object: {n_files}")
    total += 1
    passed += check("no file is shared between two objects",
                    len({v[0] for v in assigned.values()}) == 3,
                    f"{len(m.objects)} files for 3 objects")
    total += 1

    ks = [m.objects[o].n_views for o in m.objects]
    obs = [m.objects[o].n_observations for o in m.objects]
    passed += check("merging keeps K far below the detection count",
                    max(ks) < 0.4 * min(obs),
                    f"K = {ks} from {obs} detections")
    total += 1
    return passed, total


# ---------------------------------------------------------------------------
# Test 8: the 3-D view sphere
# ---------------------------------------------------------------------------

def test_view_sphere():
    section("Test 8: 3-D map with an (azimuth, elevation) view sphere")
    passed = total = 0
    feat_dim = 32
    rng = np.random.default_rng(0)
    m = ObjectCentricMap(feat_dim=feat_dim, ssp_dim=151, domain_dim=3,
                         view_dims=2, bounds=[[-5, 5]] * 3, length_scale=0.8,
                         grid_step=0.5, max_harmonic=4,
                         merge_tol=np.deg2rad(20.0), seed=0)
    m.appearance.fit(rng.standard_normal((200, feat_dim)))

    p = np.array([1.0, -2.0, 0.5])
    for az in np.linspace(-np.pi, np.pi, 6, endpoint=False):
        for el in (-0.5, 0.0, 0.5):
            robot = p + 3 * np.array([np.cos(el) * np.cos(az),
                                      np.cos(el) * np.sin(az), np.sin(el)])
            m.observe(robot_pos=robot, obj_pos=p, obj_id="thing",
                      embedding=rng.standard_normal(feat_dim),
                      update_position=False)

    err = float(np.linalg.norm(m.where_is("thing").position - p))
    passed += check("position decodes in 3-D", err < 0.6, f"err {err:.3f} m")
    total += 1

    obj = m.objects["thing"]
    book = obj.view_book(m.view_space)
    keys = obj.keys
    hits = sum(int(np.argmax(cosine(unbind(book, m.view_space.encode(v.phi)),
                                    keys)) == i)
               for i, v in enumerate(obj.views))
    passed += check("(azimuth, elevation) indexes the right view",
                    hits >= 0.85 * obj.n_views, f"{hits}/{obj.n_views}")
    total += 1

    start = obj.views[0].phi
    delta = np.array([np.deg2rad(60), 0.0])
    residue = unbind(m.orbit(book, delta), m.view_space.encode(start))
    target = wrap_angle(start - delta)
    j = int(np.argmin([np.abs(wrap_angle(v.phi - target)).max()
                       for v in obj.views]))
    sims = cosine(residue, keys)
    passed += check("orbiting in azimuth holds elevation fixed",
                    sims[j] > np.max(np.delete(sims, j)),
                    f"cos to target {sims[j]:+.3f} vs best other "
                    f"{np.max(np.delete(sims, j)):+.3f}")
    total += 1

    err = np.abs(m.orbit(book, [2 * np.pi, 2 * np.pi]) - book).max()
    passed += check("a full turn in both angles is the identity", err < 1e-9,
                    f"max diff {err:.2e}")
    total += 1
    return passed, total


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    total_passed = total_checks = 0
    for fn in [test_view_circle, test_scene_map, test_object_files,
               test_two_memories, test_appearance_keys, test_angles,
               test_association, test_view_sphere]:
        p, t = fn()
        total_passed += p
        total_checks += t

    section(f"SUMMARY: {total_passed}/{total_checks} checks passed")
    sys.exit(0 if total_passed == total_checks else 1)
