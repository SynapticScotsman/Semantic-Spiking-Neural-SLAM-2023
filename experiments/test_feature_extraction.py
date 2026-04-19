"""
test_feature_extraction.py
==========================
Standalone test script for the image feature extraction → SP encoding pipeline.

Tests three things:
  1. Synthetic images  — generated on the fly, no files needed
  2. Real image files  — pass paths on the command line
  3. SP similarity sanity checks — verifies that similar images produce similar
     SPs and different images produce different SPs

Usage
-----
    # Run all tests with synthetic images only
    python experiments/test_feature_extraction.py

    # Also test on your own images
    python experiments/test_feature_extraction.py \
        --images path/to/img1.jpg path/to/img2.png path/to/img3.jpg

    # Skip a method if the library is not installed
    python experiments/test_feature_extraction.py --methods hog

Requirements
------------
    Core (always needed):   numpy
    SIFT / ORB tests:       opencv-python   (pip install opencv-python)
    HOG tests:              scikit-image    (pip install scikit-image)
    Plots:                  matplotlib      (pip install matplotlib)

Install everything at once:
    pip install numpy opencv-python scikit-image matplotlib
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sspslam.perception import ImageFeatureEncoder
from sspslam.perception.image_feature_encoder import extract_sift, extract_hog, extract_orb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(label, value, lo, hi):
    status = "PASS" if lo <= value <= hi else "FAIL"
    print(f"  [{status}]  {label}: {value:.4f}  (expected {lo:.2f} – {hi:.2f})")
    return status == "PASS"


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def make_solid(h, w, color_bgr):
    """Solid-colour image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


def make_gradient(h, w, horizontal=True):
    """Smooth gradient from black to white."""
    img = np.zeros((h, w), dtype=np.uint8)
    if horizontal:
        img[:] = np.linspace(0, 255, w, dtype=np.uint8)
    else:
        img[:] = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1)
    return img


def make_checkerboard(h, w, tile=32):
    """Checkerboard pattern."""
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            if ((r // tile) + (c // tile)) % 2 == 0:
                img[r:r + tile, c:c + tile] = 255
    return img


def make_circle(h, w, filled=True):
    """White circle on black background."""
    img = np.zeros((h, w), dtype=np.uint8)
    cy, cx, r = h // 2, w // 2, min(h, w) // 3
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = 255
    return img


def make_noisy(base_img, std=25, seed=0):
    """Add Gaussian noise to an image."""
    rng = np.random.RandomState(seed)
    noisy = base_img.astype(np.int16) + rng.randint(-std, std, base_img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


SYNTHETIC_IMAGES = {
    "red_square":      make_solid(128, 128, (0, 0, 200)),
    "blue_square":     make_solid(128, 128, (200, 0, 0)),
    "green_square":    make_solid(128, 128, (0, 200, 0)),
    "horiz_gradient":  make_gradient(128, 128, horizontal=True),
    "vert_gradient":   make_gradient(128, 128, horizontal=False),
    "checkerboard":    make_checkerboard(128, 128, tile=16),
    "circle":          make_circle(128, 128),
    "noisy_circle":    make_noisy(make_circle(128, 128), std=30),
}

# ---------------------------------------------------------------------------
# Feature extraction wrappers (handle missing libraries gracefully)
# ---------------------------------------------------------------------------

def try_extract(method, image, label=""):
    try:
        if method == "sift":
            feat = extract_sift(image)
        elif method == "hog":
            feat = extract_hog(image)
        elif method == "orb":
            feat = extract_orb(image)
        else:
            raise ValueError(f"Unknown method: {method}")
        return feat
    except ImportError as e:
        print(f"  [SKIP] {label} — missing library: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] {label} — {e}")
        return None


# ---------------------------------------------------------------------------
# Test 1: basic encoder properties
# ---------------------------------------------------------------------------

def test_encoder_properties(ssp_dim=97):
    print_section("Test 1: ImageFeatureEncoder basic properties")
    passed = 0
    total = 0

    for feat_dim in [32, 128, 512, 2048]:
        enc = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=0)
        f = np.random.randn(feat_dim)
        sp = enc.encode(f)

        total += 3
        passed += check(f"feat_dim={feat_dim} output shape", sp.shape[0], ssp_dim, ssp_dim)
        passed += check(f"feat_dim={feat_dim} all real", float(np.isreal(sp).all()), 1.0, 1.0)
        # Unitary SPs are not unit-norm in general, but Fourier magnitudes are 1
        fft_mags = np.abs(np.fft.fft(sp))
        passed += check(f"feat_dim={feat_dim} mean FFT magnitude", fft_mags.mean(), 0.9, 1.1)

    # Batch consistency
    enc = ImageFeatureEncoder(feat_dim=128, ssp_dim=ssp_dim, seed=0)
    F = np.random.randn(8, 128)
    batch_sp = enc.encode(F)
    single_sps = np.stack([enc.encode(F[i]) for i in range(8)])
    total += 1
    passed += check("batch == stacked singles max-diff",
                    float(np.max(np.abs(batch_sp - single_sps))), 0.0, 1e-10)

    # encode_sequence
    seq = np.random.randn(10, 5, 128)
    out = enc.encode_sequence(seq)
    total += 1
    passed += check("encode_sequence output shape[0]", out.shape[0], 10, 10)
    total += 1
    passed += check("encode_sequence output shape[1]", out.shape[1], 5, 5)
    total += 1
    passed += check("encode_sequence output shape[2]", out.shape[2], ssp_dim, ssp_dim)

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


# ---------------------------------------------------------------------------
# Test 2: similarity preservation
# ---------------------------------------------------------------------------

def test_similarity_preservation(ssp_dim=97):
    print_section("Test 2: Cosine-similarity preservation (J-L property)")
    enc = ImageFeatureEncoder(feat_dim=128, ssp_dim=ssp_dim, seed=42)

    rng = np.random.RandomState(0)
    base = rng.randn(128)
    base /= np.linalg.norm(base)

    # Near-identical vector (tiny noise)
    near = base + 0.01 * rng.randn(128)
    # Moderately similar (projection of base onto a nearby direction)
    mid = base + 0.5 * rng.randn(128)
    # Orthogonal
    orth = rng.randn(128)
    orth -= orth.dot(base) * base
    # Opposite direction
    opposite = -base

    sp_base     = enc.encode(base)
    sp_near     = enc.encode(near)
    sp_mid      = enc.encode(mid)
    sp_orth     = enc.encode(orth)
    sp_opposite = enc.encode(opposite)

    passed = 0
    total = 0

    sim_near     = cosine_sim(sp_base, sp_near)
    sim_mid      = cosine_sim(sp_base, sp_mid)
    sim_orth     = cosine_sim(sp_base, sp_orth)
    sim_opposite = cosine_sim(sp_base, sp_opposite)

    print(f"  base <-> near-identical:  sim = {sim_near:.4f}")
    print(f"  base <-> moderately-sim:  sim = {sim_mid:.4f}")
    print(f"  base <-> orthogonal:      sim = {sim_orth:.4f}")
    print(f"  base <-> opposite:        sim = {sim_opposite:.4f}")

    total += 1; passed += check("near-identical sim > 0.9", sim_near, 0.9, 1.0)
    total += 1; passed += check("near > mid",
                                 float(sim_near > sim_mid), 1.0, 1.0)
    total += 1; passed += check("mid > |orth|",
                                 float(abs(sim_mid) > abs(sim_orth)), 1.0, 1.0)

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


# ---------------------------------------------------------------------------
# Test 3: classical feature extraction on synthetic images
# ---------------------------------------------------------------------------

def test_classical_features(methods, ssp_dim=97):
    print_section("Test 3: Classical feature extraction on synthetic images")
    all_passed = 0
    all_total = 0

    for method in methods:
        print(f"\n  --- Method: {method.upper()} ---")

        descriptors = {}
        for name, img in SYNTHETIC_IMAGES.items():
            feat = try_extract(method, img, label=f"{name}/{method}")
            if feat is not None:
                descriptors[name] = feat
                print(f"  Extracted {name:20s}  shape={feat.shape}  "
                      f"norm={np.linalg.norm(feat):.2f}")

        if len(descriptors) < 2:
            print("  [SKIP] Not enough descriptors to compare.")
            continue

        feat_dim = next(iter(descriptors.values())).shape[0]
        enc = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=0)

        sps = {name: enc.encode(feat) for name, feat in descriptors.items()}

        # Sanity: circle vs noisy_circle should be more similar than circle vs checkerboard
        if "circle" in sps and "noisy_circle" in sps and "checkerboard" in sps:
            sim_same  = cosine_sim(sps["circle"], sps["noisy_circle"])
            sim_diff  = cosine_sim(sps["circle"], sps["checkerboard"])
            print(f"\n  circle <-> noisy_circle : {sim_same:.4f}")
            print(f"  circle <-> checkerboard : {sim_diff:.4f}")
            all_total += 1
            all_passed += check(
                f"[{method}] noisy-same > diff",
                float(sim_same > sim_diff), 1.0, 1.0,
            )

        # All SP FFT magnitudes ≈ 1
        for name, sp in sps.items():
            mags = np.abs(np.fft.fft(sp)).mean()
            all_total += 1
            all_passed += check(f"[{method}] {name} mean FFT mag", mags, 0.9, 1.1)

    print(f"\n  {all_passed}/{all_total} checks passed")
    return all_passed, all_total


# ---------------------------------------------------------------------------
# Test 4: user-supplied real images
# ---------------------------------------------------------------------------

def test_real_images(image_paths, methods, ssp_dim=97):
    print_section("Test 4: Real image files")

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        print("  [WARN] opencv-python not installed — cannot load images via cv2.")
        print("         Install with: pip install opencv-python")

    if not has_cv2:
        return 0, 0

    images = {}
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [WARN] Could not load: {p}")
        else:
            name = os.path.splitext(os.path.basename(p))[0]
            images[name] = img
            print(f"  Loaded {name:30s}  shape={img.shape}")

    if not images:
        print("  No valid images loaded.")
        return 0, 0

    all_passed = 0
    all_total = 0

    for method in methods:
        print(f"\n  --- Method: {method.upper()} ---")
        descriptors = {}
        for name, img in images.items():
            feat = try_extract(method, img, label=f"{name}/{method}")
            if feat is not None:
                descriptors[name] = feat
                print(f"  {name:30s}  shape={feat.shape}  norm={np.linalg.norm(feat):.2f}")

        if not descriptors:
            continue

        feat_dim = next(iter(descriptors.values())).shape[0]
        enc = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=0)
        sps = {name: enc.encode(feat) for name, feat in descriptors.items()}

        names = list(sps.keys())
        print("\n  SP cosine-similarity matrix:")
        header = f"  {'':20s}" + "".join(f"{n:>14s}" for n in names)
        print(header)
        for a in names:
            row = f"  {a:20s}"
            for b in names:
                row += f"{cosine_sim(sps[a], sps[b]):14.4f}"
            print(row)

        for name, sp in sps.items():
            mags = np.abs(np.fft.fft(sp)).mean()
            all_total += 1
            all_passed += check(f"[{method}] {name} mean FFT mag", mags, 0.9, 1.1)

    print(f"\n  {all_passed}/{all_total} checks passed")
    return all_passed, all_total


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def plot_similarity_matrix(sps_dict, title="SP Cosine Similarity"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed — skipping plot.")
        return

    names = list(sps_dict.keys())
    n = len(names)
    mat = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = cosine_sim(sps_dict[a], sps_dict[b])

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the image feature extraction → SP encoding pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", nargs="*", default=[],
                        help="Paths to real image files to test.")
    parser.add_argument("--methods", nargs="+",
                        default=["sift", "hog", "orb"],
                        choices=["sift", "hog", "orb"],
                        help="Feature extraction methods to test.")
    parser.add_argument("--ssp-dim", default=97, type=int,
                        help="SP dimensionality.")
    parser.add_argument("--plot", action="store_true",
                        help="Show SP similarity matrix for synthetic images.")
    args = parser.parse_args()

    total_passed = 0
    total_checks = 0

    p, t = test_encoder_properties(args.ssp_dim)
    total_passed += p; total_checks += t

    p, t = test_similarity_preservation(args.ssp_dim)
    total_passed += p; total_checks += t

    p, t = test_classical_features(args.methods, args.ssp_dim)
    total_passed += p; total_checks += t

    if args.images:
        p, t = test_real_images(args.images, args.methods, args.ssp_dim)
        total_passed += p; total_checks += t

    if args.plot:
        # Build SPs for all synthetic images using HOG (most robust for varied images)
        print("\nBuilding similarity matrix plot (HOG)…")
        descriptors = {}
        for name, img in SYNTHETIC_IMAGES.items():
            feat = try_extract("hog", img)
            if feat is not None:
                descriptors[name] = feat
        if descriptors:
            feat_dim = next(iter(descriptors.values())).shape[0]
            enc = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=args.ssp_dim, seed=0)
            sps = {name: enc.encode(feat) for name, feat in descriptors.items()}
            plot_similarity_matrix(sps, title="HOG → SP cosine similarity (synthetic images)")

    print_section(f"SUMMARY: {total_passed}/{total_checks} checks passed")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()
