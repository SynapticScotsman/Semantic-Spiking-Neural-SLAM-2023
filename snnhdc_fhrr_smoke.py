import torch
import torch.nn as nn
import torchhd

from snnhdc_fhrr_common import (
    NUM_CLASSES,
    RandomFHRRProjection,
    fhrr_similarity,
    make_fhrr_codebook,
    set_seed,
)


def assert_fhrr_unit_phasors():
    vectors = torchhd.random(8, 128, "FHRR", dtype=torch.complex64)
    magnitudes = vectors.abs()
    assert torch.is_complex(vectors)
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)


def assert_bind_unbind_recovery():
    role, symbols, targets = make_fhrr_codebook(NUM_CLASSES, 128, torch.device("cpu"))
    decoded = torchhd.bind(targets, torchhd.inverse(role))
    scores = fhrr_similarity(decoded, symbols)
    expected = torch.arange(NUM_CLASSES)
    assert torch.equal(scores.argmax(dim=1), expected)


def assert_projection_shape_and_unit_magnitude():
    encoder = RandomFHRRProjection(32, 128, sigma=0.31)
    features = torch.randn(5, 32)
    encoded = encoder(features)
    assert encoded.shape == (5, 128)
    assert torch.is_complex(encoded)
    assert torch.allclose(encoded.abs(), torch.ones_like(encoded.abs()), atol=1e-5)


def assert_similarity_is_real_inner_product_not_magnitude():
    vector = torch.ones(1, 16, dtype=torch.complex64)
    opposite = -torch.ones(1, 16, dtype=torch.complex64)
    score = fhrr_similarity(vector, opposite).item()
    assert score < -0.99


def assert_gradients_flow_to_frontend_only():
    frontend = nn.Linear(16, 32)
    encoder = RandomFHRRProjection(32, 128, sigma=0.31)
    _, _, prototypes = make_fhrr_codebook(4, 128, torch.device("cpu"))

    inputs = torch.randn(6, 16)
    targets = torch.tensor([0, 1, 2, 3, 0, 1])
    features = frontend(inputs)
    encoded = encoder(features)
    logits = fhrr_similarity(encoded, prototypes[:4]) * 10.0
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    assert frontend.weight.grad is not None
    assert frontend.weight.grad.abs().sum().item() > 0
    assert encoder.projection.grad is None
    assert encoder.phase_bias.grad is None


@torch.no_grad()
def assert_prototype_sanity_beats_chance():
    classes = 4
    in_features = 16
    dimensions = 512
    train_per_class = 16
    test_per_class = 8

    generator = torch.Generator()
    generator.manual_seed(123)
    centers = torch.randn(classes, in_features, generator=generator)
    encoder = RandomFHRRProjection(in_features, dimensions, sigma=2.0, seed=7)

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for class_index in range(classes):
        train_features.append(
            centers[class_index]
            + 0.03 * torch.randn(train_per_class, in_features, generator=generator)
        )
        test_features.append(
            centers[class_index]
            + 0.03 * torch.randn(test_per_class, in_features, generator=generator)
        )
        train_labels.extend([class_index] * train_per_class)
        test_labels.extend([class_index] * test_per_class)

    train_features = torch.cat(train_features, dim=0)
    test_features = torch.cat(test_features, dim=0)
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    train_vectors = encoder(train_features)
    prototypes = torch.zeros(classes, dimensions, dtype=torch.complex64)
    prototypes.index_add_(0, train_labels, train_vectors)

    test_vectors = encoder(test_features)
    scores = fhrr_similarity(test_vectors, prototypes)
    accuracy = (scores.argmax(dim=1) == test_labels).float().mean().item()
    assert accuracy > 1.0 / classes


def main():
    set_seed(0)
    assert_fhrr_unit_phasors()
    assert_bind_unbind_recovery()
    assert_projection_shape_and_unit_magnitude()
    assert_similarity_is_real_inner_product_not_magnitude()
    assert_gradients_flow_to_frontend_only()
    assert_prototype_sanity_beats_chance()
    print("all_fhrr_smoke_tests_passed")


if __name__ == "__main__":
    main()
