import argparse
import os

import torch
import torch.nn as nn

from snnhdc_fhrr_common import (
    BETA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FHRR_DIMENSIONS,
    DEFAULT_LOGIT_SCALE,
    DEFAULT_LR,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SIGMA,
    DEFAULT_SNN_FEATURES,
    DEFAULT_TEST_BATCH_SIZE,
    NUM_CLASSES,
    RandomFHRRProjection,
    build_dataloaders,
    build_snn,
    ensure_dir,
    fhrr_similarity,
    get_device,
    load_snn_checkpoint,
    make_fhrr_codebook,
    save_fhrr_artifacts,
    set_seed,
    spike_rate_features,
    write_history,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inline-trained FHRR backend for the DVS Gesture SNN."
    )
    parser.add_argument("--dimensions", type=int, default=DEFAULT_FHRR_DIMENSIONS)
    parser.add_argument("--snn-features", type=int, default=DEFAULT_SNN_FEATURES)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--logit-scale", type=float, default=DEFAULT_LOGIT_SCALE)
    parser.add_argument("--alignment-loss-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--test-batch-size", type=int, default=DEFAULT_TEST_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Defaults to ./snnhdc-models/<D>-D_<beta>-B_FHRR_train_seed(<seed>)",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_trained_fhrr(net, encoder, prototypes, loader, loss_fn, logit_scale, device):
    net.eval()
    encoder.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    target_similarity = []

    for data, targets in iter(loader):
        data = data.float().to(device)
        targets_device = targets.to(device)

        features = spike_rate_features(net, data)
        encoded = encoder(features)
        scores = fhrr_similarity(encoded, prototypes)
        logits = scores * logit_scale
        loss_sum += loss_fn(logits, targets_device).item() * targets_device.numel()

        preds = logits.argmax(dim=1)
        total += targets_device.numel()
        correct += (preds == targets_device).sum().item()
        target_similarity.extend(
            scores.gather(1, targets_device.unsqueeze(1)).squeeze(1).cpu().tolist()
        )

    return correct / max(total, 1), loss_sum / max(total, 1), target_similarity


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = (
            f"./snnhdc-models/{args.dimensions}-D_{BETA}-B_"
            f"FHRR_train_sigma({args.sigma})_seed({args.seed})"
        )
    ensure_dir(save_dir)

    trainloader, testloader = build_dataloaders(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    net = build_snn(output_features=args.snn_features, beta=BETA, device=device)
    checkpoint_loaded = load_snn_checkpoint(net, args.checkpoint, device)
    encoder = RandomFHRRProjection(
        args.snn_features, args.dimensions, sigma=args.sigma, seed=args.seed + 1
    ).to(device)

    label_role, class_symbols, class_targets = make_fhrr_codebook(
        NUM_CLASSES, args.dimensions, device, seed=args.seed
    )
    prototypes = class_targets.detach()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    train_loss_hist = []
    train_ce_loss_hist = []
    train_alignment_loss_hist = []
    train_target_similarity_hist = []
    test_acc_hist = []
    test_loss_hist = []
    test_target_similarity_hist = []

    for epoch in range(args.epochs):
        net.train()
        encoder.eval()

        for data, targets in iter(trainloader):
            data = data.float().to(device)
            targets_device = targets.to(device)

            features = spike_rate_features(net, data)
            encoded = encoder(features)
            scores = fhrr_similarity(encoded, prototypes)
            logits = scores * args.logit_scale

            ce_loss = loss_fn(logits, targets_device)
            target_scores = scores.gather(1, targets_device.unsqueeze(1)).squeeze(1)
            alignment_loss = 1.0 - target_scores.mean()
            loss = ce_loss + args.alignment_loss_weight * alignment_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_hist.append(loss.item())
            train_ce_loss_hist.append(ce_loss.item())
            train_alignment_loss_hist.append(alignment_loss.item())
            train_target_similarity_hist.extend(target_scores.detach().cpu().tolist())

        test_acc, test_loss, test_target_similarity = evaluate_trained_fhrr(
            net,
            encoder,
            prototypes,
            testloader,
            loss_fn,
            args.logit_scale,
            device,
        )
        test_acc_hist.append(test_acc)
        test_loss_hist.append(test_loss)
        test_target_similarity_hist.extend(test_target_similarity)

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"test_acc={test_acc:.4f} "
            f"test_loss={test_loss:.4f} "
            f"train_loss={train_loss_hist[-1]:.4f}"
        )

    torch.save(net.state_dict(), os.path.join(save_dir, "snn_weights.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_state.pt"))

    metadata = {
        "mode": "inline_trained_fhrr",
        "checkpoint": args.checkpoint,
        "checkpoint_loaded": checkpoint_loaded,
        "dimensions": args.dimensions,
        "snn_features": args.snn_features,
        "sigma": args.sigma,
        "logit_scale": args.logit_scale,
        "alignment_loss_weight": args.alignment_loss_weight,
        "lr": args.lr,
        "epochs": args.epochs,
        "similarity": "real_part_complex_inner_product_divided_by_D",
        "prototype_rule": "fixed_bound_fhrr_class_targets",
        "final_test_accuracy": test_acc_hist[-1] if test_acc_hist else None,
        "final_test_loss": test_loss_hist[-1] if test_loss_hist else None,
    }

    save_fhrr_artifacts(
        save_dir,
        label_role=label_role,
        class_symbols=class_symbols,
        prototypes=prototypes,
        encoder=encoder,
        metadata=metadata,
    )
    write_json(os.path.join(save_dir, "metrics.json"), metadata)
    write_history(os.path.join(save_dir, "train_loss_hist.txt"), train_loss_hist)
    write_history(os.path.join(save_dir, "train_ce_loss_hist.txt"), train_ce_loss_hist)
    write_history(
        os.path.join(save_dir, "train_alignment_loss_hist.txt"),
        train_alignment_loss_hist,
    )
    write_history(
        os.path.join(save_dir, "train_target_similarity_hist.txt"),
        train_target_similarity_hist,
    )
    write_history(os.path.join(save_dir, "test_acc_hist.txt"), test_acc_hist)
    write_history(os.path.join(save_dir, "test_loss_hist.txt"), test_loss_hist)
    write_history(
        os.path.join(save_dir, "test_target_similarity_hist.txt"),
        test_target_similarity_hist,
    )

    print(f"saved={save_dir}")


if __name__ == "__main__":
    main()
