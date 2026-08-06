import csv
import json
import math
import os
from pathlib import Path

import snntorch as snn
import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
import torchhd
import torchvision
from snntorch import surrogate
from snntorch import utils
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader


SEED = 0
BETA = 0.9
NUM_CLASSES = 11
DEFAULT_FHRR_DIMENSIONS = 1024
DEFAULT_SNN_FEATURES = 512
DEFAULT_SIGMA = 0.31
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_LR = 0.001
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LOGIT_SCALE = 10.0


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_loader(dataset, batch_size, shuffle, pin_memory, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "collate_fn": tonic.collation.PadTensors(batch_first=False),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def build_dataloaders(
    batch_size=DEFAULT_BATCH_SIZE,
    test_batch_size=DEFAULT_TEST_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
    device=None,
):
    if device is None:
        device = get_device()

    train_frame_transform = transforms.Compose(
        [
            transforms.CropTime(max=1500000),
            transforms.Denoise(filter_time=5000),
            transforms.Downsample(spatial_factor=0.25),
            transforms.ToFrame(sensor_size=(32, 32, 2), time_window=1000),
        ]
    )

    online_transform = transforms.Compose(
        [
            torch.from_numpy,
            torchvision.transforms.RandomAffine(
                degrees=[-10, 10], translate=(0.1, 0.1)
            ),
        ]
    )

    test_frame_transform = transforms.Compose(
        [
            transforms.CropTime(max=1500000),
            transforms.Denoise(filter_time=5000),
            transforms.Downsample(spatial_factor=0.25),
            transforms.ToFrame(sensor_size=(32, 32, 2), time_window=1000),
        ]
    )

    trainset = tonic.datasets.DVSGesture(
        save_to="./data", transform=train_frame_transform, train=True
    )
    testset = tonic.datasets.DVSGesture(
        save_to="./data", transform=test_frame_transform, train=False
    )

    cached_trainset = DiskCachedDataset(
        trainset, cache_path="./cache/gestures-1.5s/train"
    )
    cached_testset = DiskCachedDataset(
        testset, cache_path="./cache/gestures-1.5s/test"
    )
    cached_trainset.transform = online_transform

    pin_memory = device.type == "cuda"
    trainloader = make_loader(
        cached_trainset, batch_size, True, pin_memory, num_workers
    )
    testloader = make_loader(
        cached_testset, test_batch_size, False, pin_memory, num_workers
    )

    return trainloader, testloader


def build_snn(output_features=DEFAULT_SNN_FEATURES, beta=BETA, device=None):
    if device is None:
        device = get_device()

    spike_grad = surrogate.atan()
    return nn.Sequential(
        nn.Conv2d(2, 16, 5),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2),
        snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            init_hidden=True,
            reset_mechanism="zero",
            reset_delay=True,
        ),
        nn.Dropout(0.2),
        nn.Conv2d(16, 32, 5),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),
        snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            init_hidden=True,
            reset_mechanism="zero",
            reset_delay=True,
        ),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, output_features),
        snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
            reset_mechanism="zero",
            reset_delay=True,
        ),
    ).to(device)


def load_snn_checkpoint(net, checkpoint_path, device):
    if checkpoint_path is None:
        return False

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    return True


def forward_spikes(net, data):
    spk_rec = []
    utils.reset(net)

    for step in range(data.size(0)):
        spk_out, _ = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)


def spike_rate_features(net, data):
    spk_rec = forward_spikes(net, data)
    return torch.mean(spk_rec, dim=0)


class RandomFHRRProjection(nn.Module):
    """Fixed random Fourier/FHRR phase map: exp(i * (Rz + beta))."""

    def __init__(
        self,
        in_features,
        dimensions,
        sigma=DEFAULT_SIGMA,
        seed=SEED + 1,
        dtype=torch.float32,
    ):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        generator = torch.Generator()
        generator.manual_seed(seed)

        projection = torch.empty(dimensions, in_features, dtype=dtype)
        projection.normal_(mean=0.0, std=1.0 / sigma, generator=generator)

        phase_bias = torch.empty(dimensions, dtype=dtype)
        phase_bias.uniform_(-math.pi, math.pi, generator=generator)

        self.in_features = in_features
        self.dimensions = dimensions
        self.sigma = sigma
        self.register_buffer("projection", projection)
        self.register_buffer("phase_bias", phase_bias)

    def forward(self, features):
        phases = features.matmul(self.projection.t()) + self.phase_bias
        phasors = torch.complex(torch.cos(phases), torch.sin(phases))
        return phasors.as_subclass(torchhd.FHRRTensor)


@torch.no_grad()
def make_fhrr_codebook(num_classes, dimensions, device, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)

    label_role = torchhd.random(
        1, dimensions, "FHRR", generator=generator, dtype=torch.complex64
    )[0].to(device)
    class_symbols = torchhd.random(
        num_classes, dimensions, "FHRR", generator=generator, dtype=torch.complex64
    ).to(device)
    class_targets = torchhd.bind(label_role, class_symbols)

    return label_role, class_symbols, class_targets.as_subclass(torchhd.FHRRTensor)


def fhrr_similarity(vectors, prototypes):
    """Real-part complex inner product Re(A^H v) / D.

    This intentionally does not use the magnitude of the complex inner product.
    Prototype magnitudes are preserved so raw bundled sums remain meaningful.
    """

    vectors = vectors.as_subclass(torchhd.FHRRTensor)
    prototypes = prototypes.as_subclass(torchhd.FHRRTensor)
    dimensions = vectors.size(-1)
    return torch.real(vectors.matmul(torch.conj(prototypes).transpose(-2, -1))) / dimensions


@torch.no_grad()
def bundle_fhrr_prototypes(
    net,
    encoder,
    loader,
    num_classes=NUM_CLASSES,
    device=None,
    normalize_by_count=False,
):
    if device is None:
        device = get_device()

    prototypes = torch.zeros(
        num_classes,
        encoder.dimensions,
        dtype=torch.complex64,
        device=device,
    )
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    net.eval()
    encoder.eval()

    for data, targets in iter(loader):
        data = data.float().to(device)
        targets_device = targets.to(device)
        features = spike_rate_features(net, data)
        encoded = encoder(features)

        prototypes.index_add_(0, targets_device, encoded)
        counts.index_add_(0, targets_device, torch.ones_like(targets_device))

    if normalize_by_count:
        divisor = counts.clamp_min(1).to(prototypes.dtype).unsqueeze(1)
        prototypes = prototypes / divisor

    return prototypes.as_subclass(torchhd.FHRRTensor), counts


@torch.no_grad()
def evaluate_fhrr_prototypes(net, encoder, prototypes, loader, device=None):
    if device is None:
        device = get_device()

    total = 0
    correct = 0
    loss_sum = 0.0
    target_similarity = []

    net.eval()
    encoder.eval()

    for data, targets in iter(loader):
        data = data.float().to(device)
        targets_device = targets.to(device)
        features = spike_rate_features(net, data)
        encoded = encoder(features)
        scores = fhrr_similarity(encoded, prototypes)
        preds = scores.argmax(dim=1)

        total += targets_device.numel()
        correct += (preds == targets_device).sum().item()
        loss_sum += nn.functional.cross_entropy(scores, targets_device, reduction="sum").item()
        target_similarity.extend(
            scores.gather(1, targets_device.unsqueeze(1)).squeeze(1).cpu().tolist()
        )

    accuracy = correct / max(total, 1)
    mean_loss = loss_sum / max(total, 1)
    return accuracy, mean_loss, target_similarity


def save_fhrr_artifacts(
    save_dir,
    *,
    label_role,
    class_symbols,
    prototypes,
    encoder,
    counts=None,
    metadata=None,
):
    ensure_dir(save_dir)
    payload = {
        "label_role": label_role.detach().cpu(),
        "class_symbols": class_symbols.detach().cpu(),
        "prototypes": prototypes.detach().cpu(),
        "projection": encoder.projection.detach().cpu(),
        "phase_bias": encoder.phase_bias.detach().cpu(),
        "sigma": encoder.sigma,
        "snn_features": encoder.in_features,
        "fhrr_dimensions": encoder.dimensions,
    }
    if counts is not None:
        payload["prototype_counts"] = counts.detach().cpu()
    if metadata is not None:
        payload["metadata"] = metadata

    torch.save(payload, os.path.join(save_dir, "fhrr_codebook.pt"))


def write_history(path, history):
    with open(path, "w") as f:
        for value in history:
            f.write("%s\n" % value)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
