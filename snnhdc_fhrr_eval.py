import argparse
import os

import torch

from snnhdc_fhrr_common import (
    BETA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FHRR_DIMENSIONS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SIGMA,
    DEFAULT_SNN_FEATURES,
    DEFAULT_TEST_BATCH_SIZE,
    NUM_CLASSES,
    RandomFHRRProjection,
    build_dataloaders,
    build_snn,
    bundle_fhrr_prototypes,
    ensure_dir,
    evaluate_fhrr_prototypes,
    get_device,
    load_snn_checkpoint,
    make_fhrr_codebook,
    save_fhrr_artifacts,
    set_seed,
    write_history,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen FHRR evaluation for the DVS Gesture SNN-HDC backend."
    )
    parser.add_argument("--dimensions", type=int, default=DEFAULT_FHRR_DIMENSIONS)
    parser.add_argument("--snn-features", type=int, default=DEFAULT_SNN_FEATURES)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--test-batch-size", type=int, default=DEFAULT_TEST_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--normalize-by-count", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Defaults to ./snnhdc-models/<D>-D_<beta>-B_FHRR_frozen_seed(<seed>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = (
            f"./snnhdc-models/{args.dimensions}-D_{BETA}-B_"
            f"FHRR_frozen_sigma({args.sigma})_seed({args.seed})"
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

    label_role, class_symbols, _ = make_fhrr_codebook(
        NUM_CLASSES, args.dimensions, device, seed=args.seed
    )

    with torch.no_grad():
        prototypes, counts = bundle_fhrr_prototypes(
            net,
            encoder,
            trainloader,
            num_classes=NUM_CLASSES,
            device=device,
            normalize_by_count=args.normalize_by_count,
        )
        train_acc, train_loss, train_target_similarity = evaluate_fhrr_prototypes(
            net, encoder, prototypes, trainloader, device=device
        )
        test_acc, test_loss, test_target_similarity = evaluate_fhrr_prototypes(
            net, encoder, prototypes, testloader, device=device
        )

    metadata = {
        "mode": "frozen_fhrr_eval",
        "checkpoint": args.checkpoint,
        "checkpoint_loaded": checkpoint_loaded,
        "dimensions": args.dimensions,
        "snn_features": args.snn_features,
        "sigma": args.sigma,
        "normalize_by_count": args.normalize_by_count,
        "similarity": "real_part_complex_inner_product_divided_by_D",
        "prototype_rule": "raw_class_sum" if not args.normalize_by_count else "class_mean",
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "prototype_counts": counts.detach().cpu().tolist(),
    }

    save_fhrr_artifacts(
        save_dir,
        label_role=label_role,
        class_symbols=class_symbols,
        prototypes=prototypes,
        encoder=encoder,
        counts=counts,
        metadata=metadata,
    )
    write_json(os.path.join(save_dir, "metrics.json"), metadata)
    write_history(
        os.path.join(save_dir, "train_target_similarity_hist.txt"),
        train_target_similarity,
    )
    write_history(
        os.path.join(save_dir, "test_target_similarity_hist.txt"),
        test_target_similarity,
    )

    print(f"frozen_fhrr_train_acc={train_acc:.4f}")
    print(f"frozen_fhrr_test_acc={test_acc:.4f}")
    print(f"saved={save_dir}")


if __name__ == "__main__":
    main()
