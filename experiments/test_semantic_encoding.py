"""
test_semantic_encoding.py
=========================
Unit tests for ``SemanticLandmarkEncoder``.

Core claim under test
---------------------
If a landmark SP is encoded as ``bind(appearance_sp, p1*LABEL1 + p2*LABEL2)``,
then the raw dot product with the class query ``bind(appearance_sp, LABEL1_sp)``
equals approximately ``p1``:

    dot(encode(feat, {LABEL1: p1, LABEL2: p2}),
        bind(appearance_sp, LABEL1_sp))  ≈  p1

This follows from the unitary property of appearance SPs:

    dot(bind(A, X), bind(A, Y)) = dot(X, Y)   (when A is unitary)

so that

    dot(bind(A, p1*L1 + p2*L2), bind(A, L1))
        = dot(p1*L1 + p2*L2, L1)
        = p1 * dot(L1, L1) + p2 * dot(L2, L1)
        ≈ p1 * 1 + p2 * ~0          (large-dim random-unit-vector orthogonality)
        ≈ p1

Usage
-----
    python experiments/test_semantic_encoding.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sspslam import SPSpace
from sspslam.perception import ImageFeatureEncoder, SemanticLandmarkEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(label, value, expected, tol=0.12):
    ok = abs(value - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {value:.4f}  (expected ≈ {expected:.2f} ± {tol})")
    return int(ok)


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Test 1: output shape
# ---------------------------------------------------------------------------

def test_output_shape():
    section("Test 1: output shape")
    ssp_dim, feat_dim = 97, 64
    sp_space = SPSpace(domain_size=3, dim=ssp_dim, seed=0)
    img_enc  = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=0)
    enc = SemanticLandmarkEncoder(img_enc, sp_space, ["A", "B", "C"])

    feat = np.random.RandomState(0).randn(feat_dim)
    sp = enc.encode(feat, {"A": 0.7, "B": 0.3})

    ok = sp.shape == (ssp_dim,)
    print(f"  [{'PASS' if ok else 'FAIL'}] output shape == ({ssp_dim},): got {sp.shape}")
    return int(ok), 1


# ---------------------------------------------------------------------------
# Test 2: unknown label raises KeyError
# ---------------------------------------------------------------------------

def test_unknown_label():
    section("Test 2: unknown label raises KeyError")
    sp_space = SPSpace(domain_size=2, dim=97, seed=0)
    img_enc  = ImageFeatureEncoder(feat_dim=32, ssp_dim=97, seed=0)
    enc = SemanticLandmarkEncoder(img_enc, sp_space, ["A", "B"])

    try:
        enc.encode(np.random.randn(32), {"UNKNOWN": 1.0})
        print("  [FAIL] Expected KeyError — none raised")
        return 0, 1
    except KeyError:
        print("  [PASS] KeyError raised for unknown label")
        return 1, 1


# ---------------------------------------------------------------------------
# Test 3: too many labels raises ValueError
# ---------------------------------------------------------------------------

def test_too_many_labels():
    section("Test 3: too many labels raises ValueError")
    sp_space = SPSpace(domain_size=2, dim=97, seed=0)
    img_enc  = ImageFeatureEncoder(feat_dim=32, ssp_dim=97, seed=0)

    try:
        SemanticLandmarkEncoder(img_enc, sp_space, ["A", "B", "C"])
        print("  [FAIL] Expected ValueError — none raised")
        return 0, 1
    except ValueError:
        print("  [PASS] ValueError raised when len(labels) > domain_size")
        return 1, 1


# ---------------------------------------------------------------------------
# Test 4: semantic probability scaling (the central claim)
# ---------------------------------------------------------------------------

def test_semantic_scaling(ssp_dim=151, feat_dim=128, seed=0):
    """dot(encode(feat, {L1:p, L2:1-p}), bind(app, L1)) ≈ p"""
    section("Test 4: semantic probability scaling  (central claim)")

    # Higher ssp_dim → less approximation error from cross-terms
    sp_space = SPSpace(domain_size=4, dim=ssp_dim, seed=seed)
    img_enc  = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=seed)
    enc = SemanticLandmarkEncoder(
        image_encoder=img_enc,
        sp_space=sp_space,
        label_names=["CHAIR", "TABLE", "DOOR", "WINDOW"],
    )

    rng = np.random.RandomState(42)
    feat = rng.randn(feat_dim)

    app_sp    = enc.get_appearance_sp(feat)
    chair_sp  = enc.label_sp("CHAIR")
    table_sp  = enc.label_sp("TABLE")

    # class query keys  — bind(app, LABEL)
    q_chair = sp_space.bind(np.atleast_2d(app_sp), np.atleast_2d(chair_sp)).squeeze()
    q_table = sp_space.bind(np.atleast_2d(app_sp), np.atleast_2d(table_sp)).squeeze()

    passed, total = 0, 0

    print(f"\n  {'p_chair':>8}  {'p_table':>8}  {'dot(lm,q_chair)':>17}  {'dot(lm,q_table)':>17}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*17}  {'-'*17}")

    for p_chair in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        p_table = 1.0 - p_chair
        lm_sp = enc.encode(feat, {"CHAIR": p_chair, "TABLE": p_table})

        # Raw dot product (not cosine) — see module docstring for derivation
        sim_chair = float(np.dot(lm_sp, q_chair))
        sim_table = float(np.dot(lm_sp, q_table))

        print(f"  {p_chair:8.1f}  {p_table:8.1f}  {sim_chair:17.4f}  {sim_table:17.4f}")

        passed += check(f"p_chair={p_chair:.1f} → dot_chair", sim_chair, p_chair, tol=0.12)
        total  += 1
        passed += check(f"p_table={p_table:.1f} → dot_table", sim_table, p_table, tol=0.12)
        total  += 1

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


# ---------------------------------------------------------------------------
# Test 5: class_query_sp helper
# ---------------------------------------------------------------------------

def test_class_query_sp(ssp_dim=151, feat_dim=64, n_lm=6, seed=7):
    """class_query_sp lights up in proportion to each landmark's class probability."""
    section("Test 5: class_query_sp() — proportional response")

    sp_space = SPSpace(domain_size=3, dim=ssp_dim, seed=seed)
    img_enc  = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=seed)
    enc = SemanticLandmarkEncoder(img_enc, sp_space, ["CHAIR", "TABLE", "DOOR"])

    rng = np.random.RandomState(seed)
    feats = rng.randn(n_lm, feat_dim)
    p_chairs = np.array([0.9, 0.1, 0.5, 0.0, 0.7, 0.3])

    label_probs = [
        {"CHAIR": p, "TABLE": 0.5 * (1 - p), "DOOR": 0.5 * (1 - p)}
        for p in p_chairs
    ]
    landmark_sps = np.vstack([enc.encode(feats[i], label_probs[i]) for i in range(n_lm)])
    appearance_sps = np.vstack([enc.get_appearance_sp(feats[i]) for i in range(n_lm)])

    q_chair = enc.class_query_sp("CHAIR", appearance_sps)

    # Dot of query with each landmark SP should scale with p_chair
    responses = landmark_sps @ q_chair

    passed, total = 0, 0
    print(f"\n  {'lm':>4}  {'p_chair':>8}  {'response':>10}  {'rank-ok':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*8}")

    order = np.argsort(-p_chairs)       # descending by true probability
    ranked = np.argsort(-responses)     # descending by response

    for i in range(n_lm):
        ok_rank = ranked[i] == order[i]
        print(f"  {i:4d}  {p_chairs[i]:8.1f}  {responses[i]:10.4f}  {'yes' if ok_rank else 'no':>8}")

    # Require at least 4 out of 6 ranks to be correct (lenient for small n)
    n_correct = sum(ranked[i] == order[i] for i in range(n_lm))
    passed += check("rank-correct count >= 4", n_correct, n_lm, tol=n_lm - 4)
    total  += 1

    # Highest-probability landmark should have the highest response
    passed += check("argmax(response) == argmax(p_chair)",
                    float(np.argmax(responses)), float(np.argmax(p_chairs)), tol=0.5)
    total  += 1

    print(f"\n  {passed}/{total} checks passed")
    return passed, total


# ---------------------------------------------------------------------------
# Test 6: appearance_query_sp helper
# ---------------------------------------------------------------------------

def test_appearance_query_sp(ssp_dim=151, feat_dim=128, n_lm=4, seed=3):
    """appearance_query_sp with noisy features still peaks at the correct landmark."""
    section("Test 6: appearance_query_sp() — noisy observation retrieval")

    sp_space = SPSpace(domain_size=3, dim=ssp_dim, seed=seed)
    img_enc  = ImageFeatureEncoder(feat_dim=feat_dim, ssp_dim=ssp_dim, seed=seed)
    enc = SemanticLandmarkEncoder(img_enc, sp_space, ["CHAIR", "TABLE", "DOOR"])

    rng = np.random.RandomState(seed)
    true_feats = rng.randn(n_lm, feat_dim)
    true_feats /= np.linalg.norm(true_feats, axis=1, keepdims=True)

    label_probs = [
        {"CHAIR": 0.8, "TABLE": 0.2},
        {"TABLE": 0.9, "DOOR":  0.1},
        {"DOOR":  0.7, "CHAIR": 0.3},
        {"CHAIR": 0.5, "TABLE": 0.5},
    ]
    landmark_sps = np.vstack([enc.encode(true_feats[i], label_probs[i]) for i in range(n_lm)])

    target = 0  # Query for landmark 0
    noisy_feat = true_feats[target] + 0.1 * rng.randn(feat_dim)
    sem_expected = enc.get_semantic_sp(label_probs[target])

    q_app = enc.appearance_query_sp(noisy_feat, sem_expected)
    responses = landmark_sps @ q_app

    best = int(np.argmax(responses))
    ok = best == target
    print(f"  {'[PASS]' if ok else '[FAIL]'} appearance query peak at lm {best} "
          f"(expected {target})  responses={np.round(responses, 3)}")

    return int(ok), 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    total_passed = 0
    total_checks = 0

    for test_fn in [
        test_output_shape,
        test_unknown_label,
        test_too_many_labels,
        test_semantic_scaling,
        test_class_query_sp,
        test_appearance_query_sp,
    ]:
        p, t = test_fn()
        total_passed += p
        total_checks += t

    section(f"SUMMARY: {total_passed}/{total_checks} checks passed")
    sys.exit(0 if total_passed == total_checks else 1)
