"""A COIL-style turntable: rendered images of 3-D objects at 72 azimuths.

A stand-in for a real viewpoint dataset (COIL-20/100, CRIB, or MuJoCo
renders) that needs no download.  Deliberately a *renderer*, not a feature
generator: the images have genuine self-occlusion, so the appearance manifold
around each object has the same qualitative structure as a turntable capture
-- smooth over small rotations, discontinuous where a part appears or
disappears, and aliased for objects that are symmetric.

The object set is chosen to include the failure mode on purpose: ``cube`` is
90-degree symmetric and *should* be unlocalisable in viewpoint, ``L_block``
is chiral and should be easy.  A view-direction estimator that does well on
the cube is measuring something other than viewpoint.

Swapping in real data
---------------------
Anything downstream only needs ``(images, object_index, azimuth, names)``,
so replace :func:`load_turntable` with a loader for whatever you have::

    def load_coil20(root):
        # obj<i>__<j>.png, azimuth = j * 5 degrees
        ...
        return imgs, obj_idx, np.deg2rad(5 * view_idx), names

For MuJoCo or CRIB renders, the azimuth is whatever you commanded the camera
to -- that is the ground truth the view circle is fitted against.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Mesh primitives
# ---------------------------------------------------------------------------

def box(cx, cy, cz, sx, sy, sz, colour):
    c = np.array([cx, cy, cz], float)
    s = np.array([sx, sy, sz], float) / 2
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
                     dtype=float)
    v = c + signs * s
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris, cols = [], []
    for f, shade in zip(faces, (1.0, 0.55, 0.85, 0.7, 0.95, 0.62)):
        tris += [(v[f[0]], v[f[1]], v[f[2]]), (v[f[0]], v[f[2]], v[f[3]])]
        cols += [np.asarray(colour, float) * shade] * 2
    return tris, cols


def cylinder(cx, cy, cz, r, h, colour, n=16, axis="z"):
    tris, cols = [], []
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i in range(n):
        a, b = th[i], th[(i + 1) % n]
        for lo, hi in ((-h / 2, h / 2),):
            if axis == "z":
                p = [np.array([cx + r * np.cos(t), cy + r * np.sin(t), cz + z])
                     for t, z in ((a, lo), (b, lo), (b, hi), (a, hi))]
            else:
                p = [np.array([cx + z, cy + r * np.cos(t), cz + r * np.sin(t)])
                     for t, z in ((a, lo), (b, lo), (b, hi), (a, hi))]
            tris += [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
            shade = 0.55 + 0.45 * (0.5 + 0.5 * np.cos(a))
            cols += [np.asarray(colour, float) * shade] * 2
    return tris, cols


def torus_arc(cx, cy, cz, R, r, colour, t0=-1.2, t1=1.2, n=12):
    """A handle: a partial ring standing out from the body."""
    tris, cols = [], []
    ts = np.linspace(t0, t1, n)
    prev = None
    for t in ts:
        centre = np.array([cx + R * np.cos(t), cy, cz + R * np.sin(t)])
        ring = [centre + np.array([r * np.cos(u) * np.cos(t), r * np.sin(u),
                                   r * np.cos(u) * np.sin(t)])
                for u in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        if prev is not None:
            for i in range(6):
                j = (i + 1) % 6
                tris += [(prev[i], prev[j], ring[j]), (prev[i], ring[j], ring[i])]
                cols += [np.asarray(colour, float) * (0.6 + 0.4 * i / 6)] * 2
        prev = ring
    return tris, cols


# ---------------------------------------------------------------------------
# Objects -- chosen so some are near-symmetric and some are not
# ---------------------------------------------------------------------------

def make_objects():
    o = {}

    t, c = [], []
    for part in (box(0, 0, -0.35, 0.7, 0.7, 0.10, (0.85, 0.30, 0.25)),
                 box(0, -0.30, 0.05, 0.7, 0.10, 0.70, (0.80, 0.25, 0.20)),
                 box(-0.3, -0.3, -0.62, 0.08, 0.08, 0.45, (0.35, 0.25, 0.22)),
                 box(0.3, -0.3, -0.62, 0.08, 0.08, 0.45, (0.35, 0.25, 0.22)),
                 box(-0.3, 0.3, -0.62, 0.08, 0.08, 0.45, (0.35, 0.25, 0.22)),
                 box(0.3, 0.3, -0.62, 0.08, 0.08, 0.45, (0.35, 0.25, 0.22))):
        t += part[0]; c += part[1]
    o["chair"] = (t, c)          # strongly view-dependent: back panel on one side

    t, c = cylinder(0, 0, 0, 0.32, 0.75, (0.30, 0.55, 0.85))
    h = torus_arc(0.0, 0.0, 0.0, 0.42, 0.07, (0.25, 0.45, 0.75))
    t += h[0]; c += h[1]
    o["mug"] = (t, c)            # handle appears/disappears: one clear cue

    t, c = [], []
    for part in (box(0, 0, 0, 0.55, 0.55, 0.55, (0.90, 0.75, 0.20)),):
        t += part[0]; c += part[1]
    o["cube"] = (t, c)           # 90-degree symmetric: an aliasing stress test

    t, c = [], []
    for part in (box(-0.15, 0, -0.2, 0.7, 0.25, 0.25, (0.25, 0.70, 0.35)),
                 box(0.25, 0, 0.15, 0.25, 0.25, 0.55, (0.20, 0.60, 0.30))):
        t += part[0]; c += part[1]
    o["L_block"] = (t, c)        # chiral: looks different from left and right

    t, c = cylinder(0, 0, 0, 0.30, 0.30, (0.70, 0.35, 0.75))
    for part in (box(0, 0, 0.32, 0.16, 0.16, 0.35, (0.55, 0.28, 0.60)),
                 box(0.28, 0, 0.10, 0.22, 0.06, 0.06, (0.60, 0.30, 0.65))):
        t += part[0]; c += part[1]
    o["pot"] = (t, c)

    t, c = [], []
    for part in (box(0, 0, -0.30, 0.75, 0.45, 0.12, (0.55, 0.55, 0.60)),
                 box(0, 0.16, 0.05, 0.60, 0.10, 0.55, (0.45, 0.45, 0.50)),
                 box(-0.34, 0, 0.05, 0.08, 0.40, 0.45, (0.35, 0.35, 0.40))):
        t += part[0]; c += part[1]
    o["console"] = (t, c)
    return o


def uniform(part, colour):
    """Repaint a primitive a single flat colour.

    The primitives shade their faces by index, which quietly destroys
    rotational symmetry: opposite faces of a ``box`` get 1.0 and 0.55, so even
    a perfectly square bar looks different from the front and the back.  The
    rasteriser shades by ``abs(n . light)``, which *is* symmetric under a
    half-turn, so a flat-painted mesh inherits the symmetry of its geometry.
    Use this for anything whose symmetry is meant to be the point.
    """
    tris, _ = part
    return tris, [np.asarray(colour, float)] * len(tris)


def make_symmetric_objects():
    """An opt-in set whose members are symmetric *by construction*.

    sec.16 E1 needs several near-symmetric instances, not one: Farzmahdi et
    al. show mirror-symmetric tuning emerges from training on **symmetric**
    categories, so testing whether a mirror stage helps needs a group of them
    to stratify on.  ``make_objects`` has exactly one (``cube``), which is not
    enough to resample over.

    Kept separate and off by default so every figure already published against
    the six-object set stays reproducible byte for byte.  Symmetry is still
    *measured* downstream from the alias peak rather than taken on trust here
    -- lighting and perspective mean the orders below are nominal.
    """
    o = {}
    grey = (0.62, 0.62, 0.66)

    o["bar"] = uniform(box(0, 0, 0, 0.90, 0.22, 0.22, grey), grey)   # 2-fold

    t, c = [], []
    for part in (uniform(box(0, 0, 0, 0.90, 0.20, 0.20, grey), grey),
                 uniform(box(0, 0, 0, 0.20, 0.90, 0.20, grey), grey)):
        t += part[0]; c += part[1]
    o["cross"] = (t, c)                                              # 4-fold

    o["drum"] = uniform(cylinder(0, 0, 0, 0.36, 0.55, grey, n=16), grey)

    t, c = [], []
    for k in range(3):                                               # 3-fold
        a = 2 * np.pi * k / 3
        t2, c2 = uniform(box(0.30 * np.cos(a), 0.30 * np.sin(a), 0,
                             0.18, 0.18, 0.70, grey), grey)
        t += t2; c += c2
    o["tripod"] = (t, c)
    return o


# ---------------------------------------------------------------------------
# Rasteriser: perspective camera on a circle, z-buffered triangles
# ---------------------------------------------------------------------------

def render(tris, cols, azimuth, elevation=0.35, dist=2.3, res=96, f=1.6):
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    ce, se = np.cos(elevation), np.sin(elevation)
    eye = np.array([dist * ce * ca, dist * ce * sa, dist * se])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd])

    img = np.zeros((res, res, 3), np.float32)
    zbuf = np.full((res, res), np.inf, np.float32)
    light = np.array([0.4, 0.6, 0.7]); light /= np.linalg.norm(light)

    for tri, col in zip(tris, cols):
        cam = np.stack([R @ (np.asarray(p) - eye) for p in tri])
        if np.any(-cam[:, 2] <= 0.15):
            continue
        z = -cam[:, 2]
        px = (cam[:, 0] * f / z * 0.5 + 0.5) * res
        py = (0.5 - cam[:, 1] * f / z * 0.5) * res
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        shade = 0.35 + 0.65 * abs(float(n / nn @ light))

        x0, x1 = int(max(px.min(), 0)), int(min(px.max() + 1, res))
        y0, y1 = int(max(py.min(), 0)), int(min(py.max() + 1, res))
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = ((py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2]))
        if abs(d) < 1e-9:
            continue
        w0 = ((py[1] - py[2]) * (xs - px[2]) + (px[2] - px[1]) * (ys - py[2])) / d
        w1 = ((py[2] - py[0]) * (xs - px[2]) + (px[0] - px[2]) * (ys - py[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zi = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zbuf[y0:y1, x0:x1]
        hit = inside & (zi < sub)
        sub[hit] = zi[hit]
        img[y0:y1, x0:x1][hit] = (np.asarray(col) * shade)[None, :]
    return np.clip(img, 0, 1)


def load_turntable(n_views=72, res=96, elevation=0.35, seed=0, extra=None):
    """Returns images (N*V, res, res, 3), object index, azimuth in radians.

    ``extra="symmetric"`` appends :func:`make_symmetric_objects`.  Off by
    default: everything already reported was measured on the six-object set
    and must stay reproducible.
    """
    objs = make_objects()
    if extra == "symmetric":
        objs.update(make_symmetric_objects())
    elif extra is not None:
        raise ValueError(f"unknown extra set {extra!r}")
    names = list(objs)
    imgs, oidx, az = [], [], []
    angles = np.linspace(-np.pi, np.pi, n_views, endpoint=False)
    for i, name in enumerate(names):
        tris, cols = objs[name]
        for a in angles:
            imgs.append(render(tris, cols, a, elevation=elevation, res=res))
            oidx.append(i)
            az.append(a)
    return (np.stack(imgs), np.array(oidx), np.array(az), names)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imgs, oidx, az, names = load_turntable(n_views=72)
    print("images", imgs.shape, "objects", names)
    show = [0, 9, 18, 27, 36, 45, 54, 63]
    fig, axes = plt.subplots(len(names), len(show),
                             figsize=(len(show) * 1.25, len(names) * 1.3))
    for r, name in enumerate(names):
        for c, k in enumerate(show):
            ax = axes[r, c]
            ax.imshow(imgs[r * 72 + k]); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{np.rad2deg(az[k]):.0f}°", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.suptitle("rendered turntable: 6 objects x 72 azimuths (COIL stand-in)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig("turntable_contact_sheet.png", dpi=120, bbox_inches="tight")
    print("wrote turntable_contact_sheet.png")
