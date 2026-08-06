import argparse
import os

import torch

from snnhdc_fhrr_common import (
    DEFAULT_SIGMA,
    DEFAULT_SNN_FEATURES,
    RandomFHRRProjection,
    ensure_dir,
    fhrr_similarity,
    set_seed,
    write_csv,
    write_json,
)


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic FHRR dimension/sigma capacity sweep."
    )
    parser.add_argument(
        "--dimensions",
        type=parse_int_list,
        default=parse_int_list("1024,2048,5000,10000"),
    )
    parser.add_argument(
        "--sigmas",
        type=parse_float_list,
        default=parse_float_list("0.1,0.2,0.31,0.5"),
    )
    parser.add_argument("--snn-features", type=int, default=DEFAULT_SNN_FEATURES)
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./snnhdc-models/fhrr_capacity_sweep",
    )
    return parser.parse_args()


@torch.no_grad()
def run_trial(dimensions, sigma, snn_features, items, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    features = torch.randn(items, snn_features, generator=generator)
    encoder = RandomFHRRProjection(
        snn_features,
        dimensions,
        sigma=sigma,
        seed=seed + 1000,
    )
    vectors = encoder(features)
    bundle = vectors.sum(dim=0, keepdim=True)

    recovered = fhrr_similarity(vectors, bundle).squeeze(1)
    errors = recovered - 1.0
    off_diag = fhrr_similarity(vectors, vectors)
    off_diag = off_diag[~torch.eye(items, dtype=torch.bool)]

    top_index = fhrr_similarity(vectors, bundle).argmax().item()
    return {
        "mean_recovered_weight": recovered.mean().item(),
        "std_recovered_weight": recovered.std(unbiased=False).item(),
        "mean_abs_error": errors.abs().mean().item(),
        "max_abs_error": errors.abs().max().item(),
        "off_diag_mean": off_diag.mean().item(),
        "off_diag_std": off_diag.std(unbiased=False).item(),
        "top_item_index": top_index,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)

    rows = []
    for dimensions in args.dimensions:
        for sigma in args.sigmas:
            trial_rows = []
            for trial in range(args.trials):
                result = run_trial(
                    dimensions,
                    sigma,
                    args.snn_features,
                    args.items,
                    args.seed + trial + dimensions,
                )
                row = {
                    "dimensions": dimensions,
                    "sigma": sigma,
                    "snn_features": args.snn_features,
                    "items": args.items,
                    "trial": trial,
                    **result,
                }
                rows.append(row)
                trial_rows.append(row)

            mean_abs_error = sum(r["mean_abs_error"] for r in trial_rows) / len(trial_rows)
            off_diag_std = sum(r["off_diag_std"] for r in trial_rows) / len(trial_rows)
            print(
                f"D={dimensions} sigma={sigma} "
                f"mean_abs_error={mean_abs_error:.4f} "
                f"off_diag_std={off_diag_std:.4f}"
            )

    fieldnames = [
        "dimensions",
        "sigma",
        "snn_features",
        "items",
        "trial",
        "mean_recovered_weight",
        "std_recovered_weight",
        "mean_abs_error",
        "max_abs_error",
        "off_diag_mean",
        "off_diag_std",
        "top_item_index",
    ]
    write_csv(os.path.join(args.save_dir, "capacity_sweep.csv"), rows, fieldnames)
    write_json(
        os.path.join(args.save_dir, "capacity_sweep_config.json"),
        {
            "dimensions": args.dimensions,
            "sigmas": args.sigmas,
            "snn_features": args.snn_features,
            "items": args.items,
            "trials": args.trials,
            "seed": args.seed,
            "default_sigma_reference": DEFAULT_SIGMA,
            "similarity": "real_part_complex_inner_product_divided_by_D",
            "bundle_rule": "raw_sum_no_component_normalization",
        },
    )

    print(f"saved={args.save_dir}")


if __name__ == "__main__":
    main()
