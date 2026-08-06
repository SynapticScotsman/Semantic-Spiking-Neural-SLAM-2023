import tonic
import tonic.transforms as transforms
import torchhd
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import torch
import torchvision
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch.nn as nn
from torchhd.models import Centroid
import torchmetrics
import os

seed = 0
torch.manual_seed(seed)

train_frame_transform = transforms.Compose([transforms.CropTime(max=1500000),
                                      transforms.Denoise(filter_time=5000),
                                      transforms.Downsample(spatial_factor=0.25),
                                      transforms.ToFrame(sensor_size=(32, 32, 2), time_window=1000),
                                      ])

online_transform = transforms.Compose([torch.from_numpy,
                                       torchvision.transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1))
                                    ])

test_frame_transform = transforms.Compose([transforms.CropTime(max=1500000),
                                      transforms.Denoise(filter_time=5000),
                                      transforms.Downsample(spatial_factor=0.25),
                                      transforms.ToFrame(sensor_size=(32, 32, 2), time_window=1000),
                                      ])

trainset = tonic.datasets.DVSGesture(save_to='./data', transform=train_frame_transform, train=True)
testset = tonic.datasets.DVSGesture(save_to="./data", transform=test_frame_transform, train=False)

batch_size = 32
test_batch_size = 32

cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/gestures-1.5s/train')
cached_testset = DiskCachedDataset(testset, cache_path='./cache/gestures-1.5s/test')

cached_trainset.transform = online_transform

trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True, pin_memory=True)
testloader = DataLoader(cached_testset, batch_size=test_batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), pin_memory=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

for i in range(torch.cuda.device_count()):
   print('device', i, torch.cuda.get_device_properties(i).name)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def hard_quantize(hypervector):
    positive = torch.tensor(1.0, dtype=hypervector.dtype, device=hypervector.device)
    negative = torch.tensor(-1.0, dtype=hypervector.dtype, device=hypervector.device)

    return torch.where(hypervector > 0, positive, negative)

beta = 0.9
DIMENSIONS = 1024
spike_grad = surrogate.atan()
lr = 0.001
num_epochs = 1000


snn_name = str(DIMENSIONS) + "-D_" + str(beta) + "-B_seed(" + str(seed) + ")"
folder = "./snnhdc-models/"
save_dir = folder + snn_name
net = nn.Sequential(nn.Conv2d(2, 16, 5),
                    nn.BatchNorm2d(16),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero', reset_delay=True),
                    nn.Dropout(0.2),
                    nn.Conv2d(16, 32, 5),
                    nn.BatchNorm2d(32),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero', reset_delay=True),
                    nn.Dropout(0.2),
                    nn.Flatten(),
                    nn.Linear(32*5*5, DIMENSIONS),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism='zero', reset_delay=True)
                    ).to(device)


try:
    os.mkdir(folder)
    print(f"Directory '{folder}' created successfully.")
except FileExistsError:
    print(f"Directory '{folder}' already exists.")

try:
    os.mkdir(save_dir)
    print(f"Directory '{save_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{save_dir}' already exists.")

def forward_pass(net, data):
  spk_rec = []
  utils.reset(net)

  for step in range(data.size(0)):
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)

num_classes = 11

model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)


class_targets = torchhd.random(num_classes, DIMENSIONS).to(device)
torch.save(class_targets, str(save_dir + "/class_targets.pt"))

model.add(class_targets.to(device), torch.arange(0, num_classes).to(device))
class_targets = class_targets.clamp(min=0)

positive_desired_dimensions_per_class = torch.zeros(num_classes, DIMENSIONS, dtype=torch.bool).to(device)

for i in range(model.weight.size()[0]):
    positive_desired_dimensions_per_class[i] = (model.weight[i] == 1)

@torch.no_grad()
def batch_accuracy_combined_model(net, hyper, loader, use_dot):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    test_mse_loss = nn.MSELoss()
    test_loss_hist = []
    test_hamming_hist = []

    with torch.no_grad():
        for data, targets in iter(loader):
            data = data.to(device)

            spk_rec = forward_pass(net, data)
            encoded_spks = torch.sum(spk_rec, dim=0)

            batch_target_vectors = torch.zeros(targets.size()[0], DIMENSIONS).to(device)

            for i in range(targets.size()[0]):
                current_target = targets[i].item()
                batch_target_vectors[i] = class_targets[current_target]

                positive_desired_mask = positive_desired_dimensions_per_class[current_target]
                encoded_spks[i][positive_desired_mask] = encoded_spks[i][positive_desired_mask].clamp(max=1)

                test_hamming_hist.append(torch.sum(torch.abs(torch.sub(encoded_spks[i].clamp(max=1), class_targets[current_target]))).item())

            outputs = hyper(hard_quantize(encoded_spks), dot=use_dot)
            accuracy.update(outputs.cpu(), targets)

            test_loss_val = test_mse_loss(encoded_spks, batch_target_vectors)

            test_loss_hist.append(test_loss_val.item())

    return accuracy.compute().item(), test_loss_hist, test_hamming_hist

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))


train_hamming_hist = []
test_acc_hist = []
test_hamming_hist = []

mse_loss = nn.MSELoss()

for epoch in range(num_epochs):
    for data, targets in iter(trainloader):
        data = data.to(device)

        net.train()
        spk_rec = forward_pass(net, data)

        encoded_spks = torch.sum(spk_rec, dim=0)
        batch_target_vectors = torch.zeros(targets.size()[0], DIMENSIONS).to(device)

        for i in range(targets.size()[0]):
            current_target = targets[i].item()
            batch_target_vectors[i] = class_targets[current_target]

            positive_desired_mask = positive_desired_dimensions_per_class[current_target]
            encoded_spks[i][positive_desired_mask] = encoded_spks[i][positive_desired_mask].clamp(max=1)

            train_hamming_hist.append(torch.sum(torch.abs(torch.sub(encoded_spks[i].clamp(max=1), class_targets[current_target]))).item())

        loss_val = mse_loss(encoded_spks, batch_target_vectors)

        loss_val.backward()

        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        net.eval()
        test_acc, test_loss, test_hamming = batch_accuracy_combined_model(net, model, testloader, True)
        test_acc_hist.append(test_acc)
        test_hamming_hist.extend(test_hamming)

torch.save(net.state_dict(), str(save_dir + "/snn_weights.pt"))

torch.save(optimizer.state_dict(), str(save_dir + "/optimizer_state.pt"))

with open(str(save_dir + "/train_hamming_hist.txt"), 'w') as f:
    for line in train_hamming_hist:
        f.write("%s\n" % line)

with open(str(save_dir + "/test_acc_hist.txt"), "w") as f:
    for line in test_acc_hist:
        f.write("%s\n" % line)

with open(str(save_dir + "/test_hamming_hist.txt"), "w") as f:
    for line in test_hamming_hist:
        f.write("%s\n" % line)


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
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)


@torch.no_grad()
def spike_counter_per_layer(leaky_indexes):
    sOps = 0

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

    for data, targets in iter(testloader):
        data = data.to(device)
        data = torch.sum(data, dim=[0, 1, 2])
        sOps += torch.sum(conv1 * data).item()
    layer_synaptic_operations.append(sOps)

    for j in range(len(leaky_indexes)):
        sOps = 0
        counter = 0
        sub_net = nn.Sequential(*[net[i] for i in range(leaky_indexes[j])])

        for data, targets in iter(testloader):
            data = data.to(device)
            spk_rec = forward_pass_for_spike_counting(sub_net, data)
            if j == 0:
                spk_rec = torch.sum(spk_rec, dim=[0, 1, 2])
                sOps += torch.sum(conv2 * spk_rec).item()
            counter += torch.sum(spk_rec).item()
        layer_spike_counts.append(counter)
        if j == 0:
            layer_synaptic_operations.append(sOps)

    full_net_counter = 0
    for data, targets in iter(testloader):
        data = data.to(device)
        spk_rec = forward_pass_for_full_model(net, data)
        full_net_counter += torch.sum(spk_rec).item()
    layer_spike_counts.append(full_net_counter)

    layer_synaptic_operations.append(layer_spike_counts[1] * DIMENSIONS)

layer_spike_counts = []
layer_synaptic_operations = []

with torch.no_grad():
    leaky_indexes = [4, 9]
    spike_counter_per_layer(leaky_indexes)

with open(str(save_dir + "/spike_counts_per_layer.txt"), "w") as f:
    for line in layer_spike_counts:
        f.write("%s\n" % line)

with open(str(save_dir + "/synaptic_operations_per_layer.txt"), "w") as f:
    for line in layer_synaptic_operations:
        f.write("%s\n" % line)

