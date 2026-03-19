"""
test_snn_encoder.py
===================
Standalone test script for the SNN latent representation → SP encoding
pipeline.

Covers both integration paths:

  Path A — External latent vectors
      Simulates receiving a latent vector from any SNN framework (snnTorch,
      Norse, BindsNET, Lava, etc.) as a plain numpy array, then encoding it
      to a unitary SP with SNNFeatureEncoder.

  Path B — Built-in Nengo SNN
      Uses NengoSNN (two-layer LIF network) to extract latent vectors from
      synthetic images, then encodes those to unitary SPs.

Usage
-----
    # Run all tests (Nengo must be installed for path B)
    python experiments/test_snn_encoder.py

    # Skip Nengo tests (path A only, no Nengo needed)
    python experiments/test_snn_encoder.py --skip-nengo

    # Adjust dimensions
    python experiments/test_snn_encoder.py --latent-dim 64 --ssp-dim 151

    # Show similarity matrix plot (requires matplotlib)
    python experiments/test_snn_encoder.py --plot

Requirements
------------
    Core (path A):  numpy          pip install numpy
    Path B (SNN):   nengo          pip install nengo
    Plot:           matplotlib     pip install matplotlib
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sspslam.perception import SNNFeatureEncoder, NengoSNN

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def print_section(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def check(label, value, lo, hi):
    ok = lo <= value <= hi
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  {label}: {value:.6g}  (expected {lo:.4g} – {hi:.4g})")
    return ok


# ---------------------------------------------------------------------------
# Synthetic images (no external libraries required)
# ---------------------------------------------------------------------------

def make_circle(h=64, w=64):
    img = np.zeros((h, w), dtype=np.uint8)
    cy, cx, r = h // 2, w // 2, min(h, w) // 3
    Y, X = np.ogrid[:h, :w]
    img[(X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2] = 255
    return img

def make_checkerboard(h=64, w=64, tile=8):
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            if ((r // tile) + (c // tile)) % 2 == 0:
                img[r:r + tile, c:c + tile] = 255
    return img

def make_gradient(h=64, w=64):
    return np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))

def make_noisy(base, std=20, seed=7):
    rng = np.random.RandomState(seed)
    n = base.astype(np.int16) + rng.randint(-std, std, base.shape)
    return np.clip(n, 0, 255).astype(np.uint8)

SYNTHETIC = {
    "circle":          make_circle(),
    "noisy_circle":    make_noisy(make_circle()),
    "checkerboard":    make_checkerboard(),
    "gradient":        make_gradient(),
}

IMAGE_H, IMAGE_W = 64, 64
INPUT_DIM = IMAGE_H * IMAGE_W   # 4096

# ---------------------------------------------------------------------------
# Path A — External latent vectors
# ---------------------------------------------------------------------------

def test_encoder_properties(latent_dim, ssp_dim):
    print_section("Path A · Test 1: SNNFeatureEncoder basic properties")
    passed = 0; total = 0

    for ld in [16, latent_dim, 256]:
        enc = SNNFeatureEncoder(latent_dim=ld, ssp_dim=ssp_dim, seed=0)
        v = np.random.randn(ld)
        sp = enc.encode(v)

        total += 1; passed += check(f"ld={ld} output dim", sp.shape[0], ssp_dim, ssp_dim)
        total += 1; passed += check(f"ld={ld} all real", float(np.isreal(sp).all()), 1.0, 1.0)
        fft_mags = np.abs(np.fft.fft(sp)).mean()
        total += 1; passed += check(f"ld={ld} mean FFT mag ≈ 1", fft_mags, 0.9, 1.1)

    # Batch consistency
    enc = SNNFeatureEncoder(latent_dim=latent_dim, ssp_dim=ssp_dim, seed=0)
    V = np.random.randn(8, latent_dim)
    batch = enc.encode(V)
    stacked = np.stack([enc.encode(V[i]) for i in range(8)])
    total += 1
    passed += check("batch == stacked singles max-diff",
                    float(np.max(np.abs(batch - stacked))), 0.0, 1e-10)

    # encode_sequence
    seq = np.random.randn(5, 3, latent_dim)
    out = enc.encode_sequence(seq)
    total += 1; passed += check("encode_sequence shape T",  out.shape[0], 5, 5)
    total += 1; passed += check("encode_sequence shape N",  out.shape[1], 3, 3)
    total += 1; passed += check("encode_sequence shape dim", out.shape[2], ssp_dim, ssp_dim)

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


def test_similarity_preservation(latent_dim, ssp_dim):
    print_section("Path A · Test 2: Cosine-similarity preservation (J-L)")
    enc = SNNFeatureEncoder(latent_dim=latent_dim, ssp_dim=ssp_dim, seed=42)

    rng = np.random.RandomState(1)
    base = rng.randn(latent_dim); base /= np.linalg.norm(base)
    near = base + 0.01 * rng.randn(latent_dim)
    mid  = base + 0.5  * rng.randn(latent_dim)
    orth = rng.randn(latent_dim)
    orth -= orth.dot(base) * base

    sp_base = enc.encode(base)
    sp_near = enc.encode(near)
    sp_mid  = enc.encode(mid)
    sp_orth = enc.encode(orth)

    sim_near = cosine_sim(sp_base, sp_near)
    sim_mid  = cosine_sim(sp_base, sp_mid)
    sim_orth = cosine_sim(sp_base, sp_orth)

    print(f"  base ↔ near-identical:  {sim_near:.4f}")
    print(f"  base ↔ moderately-sim:  {sim_mid:.4f}")
    print(f"  base ↔ orthogonal:      {sim_orth:.4f}")

    passed = 0; total = 0
    total += 1; passed += check("near > 0.9",  sim_near,          0.9, 1.0)
    total += 1; passed += check("near > mid",  float(sim_near > sim_mid),  1.0, 1.0)
    total += 1; passed += check("mid > |orth|", float(abs(sim_mid) > abs(sim_orth)), 1.0, 1.0)

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


def test_slam_compatibility(latent_dim, ssp_dim):
    """Pass SNNFeatureEncoder to get_slam_input_functions_from_features()."""
    print_section("Path A · Test 3: SLAM pipeline compatibility")
    passed = 0; total = 0

    try:
        from sspslam.networks.slam import get_slam_input_functions_from_features
        from sspslam.sspspace import SSPSpace
    except ImportError as e:
        print(f"  [SKIP] sspslam not fully importable: {e}")
        return 0, 0

    ssp_space = SSPSpace(domain_dim=2, ssp_dim=ssp_dim, seed=0)
    enc = SNNFeatureEncoder(latent_dim=latent_dim, ssp_dim=ssp_space.ssp_dim, seed=0)

    T, N = 50, 3
    rng = np.random.RandomState(0)
    velocity_data     = rng.randn(T, 2) * 0.1
    vec_to_landmarks  = rng.randn(T, N, 2)
    latent_vectors    = rng.randn(T, N, latent_dim)
    view_rad          = 10.0

    result = get_slam_input_functions_from_features(
        ssp_space, enc, velocity_data, vec_to_landmarks, latent_vectors, view_rad
    )

    velocity_func, vel_scaling_factor, is_landmark_in_view, \
        landmark_id_func, landmark_sp_func, landmark_vec_func, \
        landmark_vecssp_func = result

    dt = 0.001
    t_test = dt
    sp_out = landmark_sp_func(t_test)
    total += 1; passed += check("landmark_sp_func output dim", sp_out.shape[0], ssp_space.ssp_dim, ssp_space.ssp_dim)

    vel_out = velocity_func(t_test)
    total += 1; passed += check("velocity_func output dim", vel_out.shape[0], 2, 2)

    total += 1; passed += check("vel_scaling_factor > 0", float(vel_scaling_factor > 0), 1.0, 1.0)

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


# ---------------------------------------------------------------------------
# Path B — Built-in Nengo SNN
# ---------------------------------------------------------------------------

def test_nengo_snn_extraction(latent_dim, ssp_dim):
    print_section("Path B · Test 4: NengoSNN latent extraction")
    passed = 0; total = 0

    try:
        import nengo
    except ImportError:
        print("  [SKIP] nengo not installed — pip install nengo")
        return 0, 0

    snn = NengoSNN(
        input_dim=INPUT_DIM,
        latent_dim=latent_dim,
        n_neurons_layer1=100,   # smaller for fast tests
        n_neurons_layer2=50,
        presentation_time=0.05,
        seed=0,
    )

    circle = SYNTHETIC["circle"]
    checker = SYNTHETIC["checkerboard"]
    gradient = SYNTHETIC["gradient"]

    latent_circle  = snn.extract(circle)
    latent_checker = snn.extract(checker)
    latent_grad    = snn.extract(gradient)

    total += 1; passed += check("circle latent dim",    latent_circle.shape[0],  latent_dim, latent_dim)
    total += 1; passed += check("checker latent dim",   latent_checker.shape[0], latent_dim, latent_dim)
    total += 1; passed += check("gradient latent dim",  latent_grad.shape[0],    latent_dim, latent_dim)

    # Determinism: running twice with the same seed should give the same result
    latent_circle2 = snn.extract(circle)
    diff = float(np.max(np.abs(latent_circle - latent_circle2)))
    total += 1; passed += check("deterministic output (diff=0)", diff, 0.0, 1e-10)

    # Different images should produce different latents
    diff_cc = float(np.linalg.norm(latent_circle - latent_checker))
    diff_cg = float(np.linalg.norm(latent_circle - latent_grad))
    total += 1; passed += check("circle vs checker L2 > 0.01", diff_cc, 0.01, 1e9)
    total += 1; passed += check("circle vs gradient L2 > 0.01", diff_cg, 0.01, 1e9)

    # extract_batch
    batch_out = snn.extract_batch([circle, checker, gradient])
    total += 1; passed += check("extract_batch shape[0]", batch_out.shape[0], 3, 3)
    total += 1; passed += check("extract_batch shape[1]", batch_out.shape[1], latent_dim, latent_dim)

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


def test_nengo_end_to_end(latent_dim, ssp_dim):
    print_section("Path B · Test 5: End-to-end  image → NengoSNN → unitary SP")
    passed = 0; total = 0

    try:
        import nengo
    except ImportError:
        print("  [SKIP] nengo not installed — pip install nengo")
        return 0, 0

    snn = NengoSNN(
        input_dim=INPUT_DIM,
        latent_dim=latent_dim,
        n_neurons_layer1=100,
        n_neurons_layer2=50,
        presentation_time=0.05,
        seed=1,
    )
    enc = SNNFeatureEncoder(latent_dim=latent_dim, ssp_dim=ssp_dim, seed=1, nengo_snn=snn)

    sps = {}
    for name, img in SYNTHETIC.items():
        sp = enc.extract_and_encode(img)
        sps[name] = sp
        fft_mag = np.abs(np.fft.fft(sp)).mean()
        total += 1; passed += check(f"{name} mean FFT mag ≈ 1", fft_mag, 0.9, 1.1)
        total += 1; passed += check(f"{name} output dim", sp.shape[0], ssp_dim, ssp_dim)

    # circle vs noisy_circle should be more similar than circle vs checkerboard
    sim_same = cosine_sim(sps["circle"], sps["noisy_circle"])
    sim_diff = cosine_sim(sps["circle"], sps["checkerboard"])
    print(f"\n  circle ↔ noisy_circle : {sim_same:.4f}")
    print(f"  circle ↔ checkerboard : {sim_diff:.4f}")
    total += 1; passed += check("noisy-same > different", float(sim_same > sim_diff), 1.0, 1.0)

    # extract_and_encode raises without nengo_snn
    enc_no_snn = SNNFeatureEncoder(latent_dim=latent_dim, ssp_dim=ssp_dim, seed=0)
    try:
        enc_no_snn.extract_and_encode(SYNTHETIC["circle"])
        total += 1; passed += check("should have raised RuntimeError", 0.0, 1.0, 1.0)
    except RuntimeError:
        total += 1; passed += check("RuntimeError raised (correct)", 1.0, 1.0, 1.0)

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def plot_similarity_matrix(sps, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed — pip install matplotlib")
        return

    names = list(sps.keys())
    n = len(names)
    mat = np.array([[cosine_sim(sps[a], sps[b]) for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
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
        description="Test SNN latent representation → SP encoding pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--latent-dim", default=32, type=int,
                        help="Latent vector dimensionality.")
    parser.add_argument("--ssp-dim", default=97, type=int,
                        help="SP dimensionality.")
    parser.add_argument("--skip-nengo", action="store_true",
                        help="Skip tests that require Nengo (path B).")
    parser.add_argument("--plot", action="store_true",
                        help="Show SP similarity matrix for synthetic images.")
    args = parser.parse_args()

    total_p = 0; total_t = 0

    # ---- Path A ----
    p, t = test_encoder_properties(args.latent_dim, args.ssp_dim)
    total_p += p; total_t += t

    p, t = test_similarity_preservation(args.latent_dim, args.ssp_dim)
    total_p += p; total_t += t

    p, t = test_slam_compatibility(args.latent_dim, args.ssp_dim)
    total_p += p; total_t += t

    # ---- Path B ----
    if not args.skip_nengo:
        p, t = test_nengo_snn_extraction(args.latent_dim, args.ssp_dim)
        total_p += p; total_t += t

        p, t = test_nengo_end_to_end(args.latent_dim, args.ssp_dim)
        total_p += p; total_t += t

    # ---- Optional plot ----
    if args.plot and not args.skip_nengo:
        try:
            import nengo
            snn = NengoSNN(
                input_dim=INPUT_DIM, latent_dim=args.latent_dim,
                n_neurons_layer1=100, n_neurons_layer2=50,
                presentation_time=0.05, seed=0,
            )
            enc = SNNFeatureEncoder(latent_dim=args.latent_dim,
                                    ssp_dim=args.ssp_dim, seed=0,
                                    nengo_snn=snn)
            sps = {name: enc.extract_and_encode(img)
                   for name, img in SYNTHETIC.items()}
            plot_similarity_matrix(sps, "NengoSNN → SP cosine similarity")
        except ImportError:
            print("\n  [SKIP] plot skipped — nengo not installed")

    print_section(f"SUMMARY: {total_p}/{total_t} checks passed")
    sys.exit(0 if total_p == total_t else 1)


if __name__ == "__main__":
    main()
