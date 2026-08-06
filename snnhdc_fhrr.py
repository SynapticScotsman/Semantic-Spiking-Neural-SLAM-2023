import math
import os

import tonic
import tonic.transforms as transforms
import torch
import torch.nn as nn
import torchhd
import torchmetrics
import torchvision
from snntorch import surrogate
from snntorch import utils
import snntorch as snn
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader


seed = 0

beta = 0.9
DIMENSIONS = 1024
SNN_FEATURES = 512
num_classes = 11
spike_grad = surrogate.atan()
lr = 0.001
num_epochs = 1000

# The SNN emits real spike-rate features z. A fixed random projection maps z
# into FHRR phase space: v = exp(i * (Rz + beta)).
FHRR_PROJECTION_SCALE = math.pi
FHRR_LOGIT_SCALE = 10.0
FHRR_ALIGNMENT_LOSS_WEIGHT = 0.1

batch_size = 32
test_batch_size = 32


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dataloaders(device):
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
    trainloader = DataLoader(
        cached_trainset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
        prefetch_factor=2,
    )
    testloader = DataLoader(
        cached_testset,
        batch_size=test_batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=4,
        pin_memory=pin_memory,
        prefetch_factor=2,
    )

    return trainloader, testloader


def build_snn(device):
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
        nn.Linear(32 * 5 * 5, SNN_FEATURES),
        snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
            reset_mechanism="zero",
            reset_delay=True,
        ),
    ).to(device)


@torch.no_grad()
def make_fhrr_codebook(num_labels, dimensions, device):
    generator = torch.Generator()
    generator.manual_seed(seed)

    label_role = torchhd.random(
        1, dimensions, "FHRR", generator=generator, dtype=torch.complex64
    )[0].to(device)
    class_symbols = torchhd.random(
        num_labels, dimensions, "FHRR", generator=generator, dtype=torch.complex64
    ).to(device)

    # FHRR binding is element-wise complex multiplication, which is equivalent
    # to adding phases in the Fourier domain.
    class_targets = torchhd.normalize(torchhd.bind(label_role, class_symbols))

    return label_role, class_symbols, class_targets


class RandomFHRRProjection(nn.Module):
    def __init__(self, in_features, dimensions, phase_scale=math.pi):
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed + 1)

        projection = torch.empty(dimensions, in_features)
        projection.normal_(
            mean=0.0,
            std=phase_scale / math.sqrt(in_features),
            generator=generator,
        )

        phase_bias = torch.empty(dimensions)
        phase_bias.uniform_(-math.pi, math.pi, generator=generator)

        self.register_buffer("projection", projection)
        self.register_buffer("phase_bias", phase_bias)

    def forward(self, features):
        phases = features.matmul(self.projection.t()) + self.phase_bias
        phasors = torch.complex(torch.cos(phases), torch.sin(phases))

        return phasors.as_subclass(torchhd.FHRRTensor)


class FHRRClassifier(nn.Module):
    def __init__(self, label_role, class_symbols):
        super().__init__()
        self.register_buffer("inverse_label_role", torchhd.inverse(label_role).clone())
        self.register_buffer("class_symbols", torchhd.normalize(class_symbols).clone())

    def forward(self, encoded_fhrr):
        encoded_fhrr = torchhd.normalize(encoded_fhrr.as_subclass(torchhd.FHRRTensor))
        inverse_label_role = self.inverse_label_role.as_subclass(torchhd.FHRRTensor)
        class_symbols = self.class_symbols.as_subclass(torchhd.FHRRTensor)

        decoded_symbols = torchhd.normalize(
            torchhd.bind(encoded_fhrr, inverse_label_role)
        )
        return torchhd.cosine_similarity(decoded_symbols, class_symbols)


def forward_pass(net, data):
    spk_rec = []
    mem_rec = []
    utils.reset(net)

    for step in range(data.size(0)):
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


@torch.no_grad()
def batch_accuracy_fhrr(net, encoder, classifier, loader, loss_fn, device):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    test_loss_hist = []
    test_cosine_hist = []

    for data, targets in iter(loader):
        data = data.to(device)
        targets_device = targets.to(device)

        spk_rec, _ = forward_pass(net, data)
        feature_rates = torch.mean(spk_rec, dim=0)

        encoded_fhrr = encoder(feature_rates)
        outputs = classifier(encoded_fhrr)
        logits = outputs * FHRR_LOGIT_SCALE
        loss_val = loss_fn(logits, targets_device)
        test_loss_hist.append(loss_val.item())

        accuracy.update(logits.cpu(), targets)

        target_cosines = outputs.gather(1, targets_device.unsqueeze(1)).squeeze(1)
        test_cosine_hist.extend(target_cosines.detach().cpu().tolist())

    return accuracy.compute().item(), test_loss_hist, test_cosine_hist


@torch.no_grad()
def forward_pass_for_spike_counting(net, data):
    spk_rec = []
    utils.reset(net)

    for step in range(data.size(0)):
        spk_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)


@torch.no_grad()
def forward_pass_for_full_model(net, data):
    spk_rec = []
    utils.reset(net)

    for step in range(data.size(0)):
        spk_out, _ = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)


@torch.no_grad()
def spike_counter_per_layer(net, testloader, device, leaky_indexes):
    layer_spike_counts = []
    layer_synaptic_operations = []
    s_ops = 0

    conv1 = torch.zeros(32, 32).to(device)
    for x in range(28):
        for y in range(28):
            for i in range(5):
                for j in range(5):
                    conv1[x + i, y + j] += 1
    conv1 *= 16

    conv2 = torch.zeros(14, 14).to(device)
    for x in range(10):
        for y in range(10):
            for i in range(5):
                for j in range(5):
                    conv2[x + i, y + j] += 1
    conv2 *= 32

    for data, _ in iter(testloader):
        data = data.to(device)
        data = torch.sum(data, dim=[0, 1, 2])
        s_ops += torch.sum(conv1 * data).item()
    layer_synaptic_operations.append(s_ops)

    for j in range(len(leaky_indexes)):
        s_ops = 0
        counter = 0
        sub_net = nn.Sequential(*[net[i] for i in range(leaky_indexes[j])])

        for data, _ in iter(testloader):
            data = data.to(device)
            spk_rec = forward_pass_for_spike_counting(sub_net, data)
            if j == 0:
                spk_rec = torch.sum(spk_rec, dim=[0, 1, 2])
                s_ops += torch.sum(conv2 * spk_rec).item()
            counter += torch.sum(spk_rec).item()
        layer_spike_counts.append(counter)
        if j == 0:
            layer_synaptic_operations.append(s_ops)

    full_net_counter = 0
    for data, _ in iter(testloader):
        data = data.to(device)
        spk_rec = forward_pass_for_full_model(net, data)
        full_net_counter += torch.sum(spk_rec).item()
    layer_spike_counts.append(full_net_counter)

    layer_synaptic_operations.append(layer_spike_counts[1] * SNN_FEATURES)

    return layer_spike_counts, layer_synaptic_operations


def write_history(path, history):
    with open(path, "w") as f:
        for line in history:
            f.write("%s\n" % line)


def main():
    torch.manual_seed(seed)
    device = get_device()

    for i in range(torch.cuda.device_count()):
        print("device", i, torch.cuda.get_device_properties(i).name)

    trainloader, testloader = build_dataloaders(device)

    snn_name = f"{DIMENSIONS}-D_{beta}-B_FHRR_seed({seed})"
    folder = "./snnhdc-models/"
    save_dir = os.path.join(folder, snn_name)
    os.makedirs(save_dir, exist_ok=True)

    net = build_snn(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    label_role, class_symbols, class_targets = make_fhrr_codebook(
        num_classes, DIMENSIONS, device
    )
    encoder = RandomFHRRProjection(
        SNN_FEATURES, DIMENSIONS, phase_scale=FHRR_PROJECTION_SCALE
    ).to(device)
    classifier = FHRRClassifier(label_role, class_symbols).to(device)

    torch.save(
        {
            "label_role": label_role.detach().cpu(),
            "class_symbols": class_symbols.detach().cpu(),
            "class_targets": class_targets.detach().cpu(),
            "projection": encoder.projection.detach().cpu(),
            "phase_bias": encoder.phase_bias.detach().cpu(),
            "snn_features": SNN_FEATURES,
            "fhrr_dimensions": DIMENSIONS,
            "projection_scale": FHRR_PROJECTION_SCALE,
        },
        os.path.join(save_dir, "fhrr_codebook.pt"),
    )

    train_loss_hist = []
    train_cosine_hist = []
    test_acc_hist = []
    test_cosine_hist = []

    classification_loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()

        for data, targets in iter(trainloader):
            data = data.to(device)
            targets_device = targets.to(device)

            spk_rec, _ = forward_pass(net, data)
            feature_rates = torch.mean(spk_rec, dim=0)

            encoded_fhrr = encoder(feature_rates)
            outputs = classifier(encoded_fhrr)
            logits = outputs * FHRR_LOGIT_SCALE
            target_cosines = outputs.gather(1, targets_device.unsqueeze(1)).squeeze(1)
            alignment_loss = 1.0 - target_cosines.mean()

            loss_val = classification_loss(
                logits, targets_device
            ) + FHRR_ALIGNMENT_LOSS_WEIGHT * alignment_loss
            loss_val.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss_hist.append(loss_val.item())
            train_cosine_hist.extend(target_cosines.detach().cpu().tolist())

        with torch.no_grad():
            net.eval()
            test_acc, test_loss, test_cosine = batch_accuracy_fhrr(
                net, encoder, classifier, testloader, classification_loss, device
            )
            test_acc_hist.append(test_acc)
            test_cosine_hist.extend(test_cosine)

        print(
            f"epoch {epoch + 1}/{num_epochs} "
            f"test_acc={test_acc:.4f} "
            f"train_loss={train_loss_hist[-1]:.4f}"
        )

    torch.save(net.state_dict(), os.path.join(save_dir, "snn_weights.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_state.pt"))

    write_history(os.path.join(save_dir, "train_loss_hist.txt"), train_loss_hist)
    write_history(os.path.join(save_dir, "train_cosine_hist.txt"), train_cosine_hist)
    write_history(os.path.join(save_dir, "test_acc_hist.txt"), test_acc_hist)
    write_history(os.path.join(save_dir, "test_cosine_hist.txt"), test_cosine_hist)

    layer_spike_counts, layer_synaptic_operations = spike_counter_per_layer(
        net, testloader, device, [4, 9]
    )

    write_history(
        os.path.join(save_dir, "spike_counts_per_layer.txt"), layer_spike_counts
    )
    write_history(
        os.path.join(save_dir, "synaptic_operations_per_layer.txt"),
        layer_synaptic_operations,
    )


if __name__ == "__main__":
    main()
