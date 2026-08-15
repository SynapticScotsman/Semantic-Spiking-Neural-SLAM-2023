"""View-direction localisation on a turntable, and how to treat the latents.

Two questions, measured rather than argued:

1. **Can an object file tell you which way round you are seeing something?**
   This is map localisation with the manifold swapped.  In a room you unbind
   ``ID`` from the scene map and correlate the residue against a grid of
   ``S_allo(x)``; the peak is where you are.  On an object you correlate an
   observed crop key against the object file read at every viewpoint; the
   peak is the direction you are looking from.  No pose input -- appearance
   alone fixes the angle.

2. **How should the crop embeddings be conditioned first?**  Two knobs:
   partial whitening (``alpha``, 0 = centre only, 1 = full PCA whitening) and
   outright removal of the leading principal directions.  They do very
   different things and only one of them helps.

Runs on numpy + scikit-image, no download, no GPU::

    python experiments/run_view_localisation.py
    python experiments/run_view_localisation.py --plot --encoder dinov2

``--encoder dinov2`` needs torch + transformers + weights; ``hog`` is the
default and needs neither.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sspslam.objectmap import (CircularSSPSpace, bind, bundle,  # noqa: E402
                               wrap_angle)
from turntable_dataset import load_turntable  # noqa: E402

SSP_DIM, MAX_HARMONIC, K_STORED = 151, 8, 12


# ---------------------------------------------------------------------------
# Front ends
# ---------------------------------------------------------------------------

def encode_hog(imgs):
    """Shape-only descriptor.  Grayscale on purpose: colour would give the
    object identity away for free and hide what the geometry is doing."""
    from skimage.feature import hog
    return np.stack([hog(g, orientations=9, pixels_per_cell=(12, 12),
                         cells_per_block=(2, 2), feature_vector=True)
                     for g in imgs.mean(-1)]).astype(np.float64)


def encode_dinov2(imgs, model_name="facebook/dinov2-small", batch=32,
                  pooling="cls"):
    """DINOv2 crop embeddings.  Needs torch + transformers + the weights.

    ``pooling='cls'`` gives one vector per crop -- the drop-in for the HOG
    path.  ``pooling='mean'`` averages patch tokens after dropping the
    high-norm outlier tokens that ViTs park in uninformative patches; those
    tokens dominate an unweighted mean and are a form of anisotropy that has
    nothing to do with viewpoint.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            px = proc(images=list((imgs[i:i + batch] * 255).astype(np.uint8)),
                      return_tensors="pt")
            h = model(**px).last_hidden_state          # (B, 1 + P, D)
            if pooling == "cls":
                out.append(h[:, 0].numpy())
            else:
                patches = h[:, 1:]
                norms = patches.norm(dim=-1)
                keep = norms < (norms.median(dim=1, keepdim=True).values * 2.0)
                w = keep.float().unsqueeze(-1)
                out.append(((patches * w).sum(1) / w.sum(1).clamp(min=1)).numpy())
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------------------
# Latent conditioning
# ---------------------------------------------------------------------------

def fit_basis(Ztr):
    mu = Ztr.mean(0)
    _, s, Vt = np.linalg.svd(Ztr - mu, full_matrices=False)
    lam = s ** 2 / max(len(Ztr) - 1, 1)
    keep = lam > lam.max() * 1e-10
    return mu, Vt[keep], lam[keep]


def condition(Z, mu, V, lam, W, alpha=0.0, drop=0):
    """centre -> PCA basis -> lambda^(-alpha/2) -> drop leading PCs -> project."""
    P = ((Z - mu) @ V.T) * lam ** (-alpha / 2.0)
    P = P[:, drop:]
    K = P @ W[drop:P.shape[1] + drop]
    return K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), 1e-12)


def between_object_share(P, obj, n_pc=30):
    """Fraction of each PC's variance that is between objects, not within.

    High share = the direction says *which object*.  Low share = it says
    *which way it is facing*.  This is the criterion that should decide what
    to remove; eigenvalue rank is only a proxy for it.
    """
    grand = P.mean(0)
    out = []
    for i in range(min(n_pc, P.shape[1])):
        p = P[:, i]
        between = sum(((p[obj == o].mean() - grand[i]) ** 2) * (obj == o).sum()
                      for o in np.unique(obj))
        out.append(between / max(((p - grand[i]) ** 2).sum(), 1e-12))
    return np.array(out)


# ---------------------------------------------------------------------------
# The object files, and localising against them
# ---------------------------------------------------------------------------

def build_books(keys, obj, az, stored, n_obj, vs):
    return np.stack([
        bundle(np.stack([bind(k, vs.encode([a])) for k, a in
                         zip(keys[(obj == o) & stored], az[(obj == o) & stored])]))
        for o in range(n_obj)])


def localise(books, keys, obj, az, mask, vs, n_grid=720):
    """Estimate viewpoint (and identity) for every held-out crop."""
    fields = []
    for o in range(len(books)):
        f = np.stack([vs.view_likelihood(books[o], k, n_per_dim=n_grid)[1]
                      for k in keys[mask]])
        fields.append(f)
    fields = np.stack(fields, axis=1)                  # (n, n_obj, n_grid)
    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    own = fields[np.arange(fields.shape[0]), obj[mask]]
    err = np.abs(np.rad2deg(wrap_angle(grid[own.argmax(1)] - az[mask])))
    id_hit = (fields.max(2).argmax(1) == obj[mask]).mean()
    return err, float(id_hit), grid, fields


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="hog", choices=("hog", "dinov2"))
    ap.add_argument("--n-views", type=int, default=72)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save-dir", default="data/objectmap")
    args = ap.parse_args()

    imgs, obj, az, names = load_turntable(n_views=args.n_views)
    Z = (encode_hog(imgs) if args.encoder == "hog" else encode_dinov2(imgs))
    print(f"{len(Z)} crops, {Z.shape[1]}-D {args.encoder}, "
          f"{len(names)} objects x {args.n_views} azimuths")

    vi = np.concatenate([np.arange(args.n_views)] * len(names))
    train = vi % 2 == 0
    stored = vi % max(args.n_views // K_STORED, 1) == 0
    mu, V, lam = fit_basis(Z[train])
    rng = np.random.default_rng(0)
    W = rng.standard_normal((V.shape[0], SSP_DIM)) / np.sqrt(V.shape[0])
    vs = CircularSSPSpace(1, ssp_dim=SSP_DIM, max_harmonic=MAX_HARMONIC,
                          rng=np.random.default_rng(1))
    print(f"{V.shape[0]} PCs, eigenvalues span {lam.max()/lam.min():.1e}; "
          f"K={K_STORED} sides on file; view kernel half-width "
          f"{np.rad2deg(vs.lobe_width()):.0f} deg\n")

    share = between_object_share((Z - mu) @ V.T, obj)
    print("Which principal directions say WHICH OBJECT rather than WHICH WAY:")
    print(f"  {'PC':>3s} {'% var':>7s} {'between-object':>15s}")
    for i in range(8):
        print(f"  {i:3d} {100*lam[i]/lam.sum():7.2f} {share[i]*100:14.1f}%")

    def run(alpha=0.0, drop=0):
        keys = condition(Z, mu, V, lam, W, alpha, drop)
        books = build_books(keys, obj, az, stored, len(names), vs)
        err, idh, _, _ = localise(books, keys, obj, az, ~train, vs)
        te = ~train
        stab = np.mean([np.mean(np.sum(
            keys[(obj == o) & te][np.argsort(az[(obj == o) & te])] *
            np.roll(keys[(obj == o) & te][np.argsort(az[(obj == o) & te])],
                    -1, axis=0), axis=1)) for o in range(len(names))])
        sep = np.mean([float((keys[(obj == o) & te] @ keys[(obj != o) & te].T).mean())
                       for o in range(len(names))])
        return err, idh, stab, sep

    print("\n[A] partial whitening: alpha = 0 centre only, 1 = full whitening")
    hdr = (f"{'alpha':>6s} {'separab':>8s} {'stability':>10s} {'pose err':>9s} "
           f"{'<15deg':>7s} {'ID hit':>7s}")
    print(hdr); print("-" * len(hdr))
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        e, idh, stab, sep = run(alpha=a)
        print(f"{a:6.2f} {sep:+8.3f} {stab:+10.3f} {np.median(e):8.1f}d "
              f"{(e < 15).mean():7.2f} {idh:7.2f}")

    print("\n[B] targeted removal: drop the leading PCs, leave the rest alone")
    print(hdr.replace(" alpha", "drop k")); print("-" * len(hdr))
    best, best_err = 0, None
    for k in (0, 1, 2, 3, 5, 8, 12, 20):
        e, idh, stab, sep = run(drop=k)
        print(f"{k:6d} {sep:+8.3f} {stab:+10.3f} {np.median(e):8.1f}d "
              f"{(e < 15).mean():7.2f} {idh:7.2f}")
        if best_err is None or np.median(e) < np.median(best_err):
            best, best_err = k, e

    print(f"\nper-object view-direction error at drop k={best}:")
    for i, nm in enumerate(names):
        e = best_err[obj[~train] == i]
        print(f"    {nm:9s} median {np.median(e):6.1f} deg   "
              f"90th pct {np.percentile(e, 90):6.1f} deg")
    print("  (a 90-degree-symmetric object SHOULD alias -- the likelihood has"
          "\n   four equal peaks and no estimator can pick between them)")

    if args.plot:
        plot(Z, mu, V, lam, W, obj, az, train, stored, names, vs, share,
             best, args.save_dir)


def plot(Z, mu, V, lam, W, obj, az, train, stored, names, vs, share, best,
         save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = condition(Z, mu, V, lam, W, drop=best)
    books = build_books(keys, obj, az, stored, len(names), vs)
    err, _, grid, fields = localise(books, keys, obj, az, ~train, vs)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].bar(range(len(share)), share * 100,
              color=["crimson" if s > .4 else "steelblue" for s in share])
    ax[0].set_xlabel("principal component")
    ax[0].set_ylabel("between-object share of variance (%)")
    ax[0].set_title("which PCs say WHICH object,\nnot which way it faces")

    o = names.index("L_block") if "L_block" in names else 0
    sel = obj[~train] == o
    f = fields[sel][:, o]
    im = ax[1].imshow(f, aspect="auto", origin="lower", cmap="magma",
                      extent=[-180, 180, 0, sel.sum()])
    ax[1].plot(np.rad2deg(az[~train][sel]), np.arange(sel.sum()) + 0.5,
               "c.", ms=3, label="true azimuth")
    ax[1].set_xlabel("hypothesised viewing direction (deg)")
    ax[1].set_ylabel("held-out crop")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"{names[o]}: likelihood over the view circle")
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    for i, nm in enumerate(names):
        e = np.sort(err[obj[~train] == i])
        ax[2].plot(e, np.linspace(0, 1, len(e)), label=nm)
    ax[2].axvline(90, ls=":", c="grey")
    ax[2].set_xlim(0, 180); ax[2].set_xlabel("view-direction error (deg)")
    ax[2].set_ylabel("fraction of held-out crops")
    ax[2].legend(fontsize=7)
    ax[2].set_title(f"view-direction localisation (drop k={best})")

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "view_localisation.png")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()
