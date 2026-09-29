from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from formal_dataset import (
    load_npz_dataset,
    FormalDarcyDataset,
)

from fno_model import FNO2d


ENSEMBLE_DIRECTORY = Path(
    "outputs/fno_ensemble"
)

OUTPUT_DIRECTORY = Path(
    "outputs/fno_uncertainty"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

ENSEMBLE_SEEDS = [
    2026,
    2027,
    2028,
    2029,
    2030,
]

DATASETS = {
    "iid": Path(
        "data/formal/darcy_iid_test_300.npz"
    ),
    "ood_010": Path(
        "data/ood/darcy_ood_length_010_300.npz"
    ),
    "ood_040": Path(
        "data/ood/darcy_ood_length_040_300.npz"
    ),
}

DISPLAY_NAMES = {
    "iid": "IID length 0.15",
    "ood_010": "OOD length 0.10",
    "ood_040": "OOD length 0.40",
}


def build_fno(device):
    return FNO2d(
        observation_channels=2,
        output_channels=1,
        width=32,
        modes_x=12,
        modes_y=12,
        number_of_layers=4,
        padding=8,
        use_coordinates=True,
    ).to(device)


def checkpoint_path(seed):
    return (
        ENSEMBLE_DIRECTORY
        / f"seed_{seed}"
        / "fno_best.pt"
    )


def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{path.resolve()}"
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def verify_checkpoint_statistics(
    checkpoints,
):
    reference = checkpoints[0][
        "statistics"
    ]

    for checkpoint in checkpoints[1:]:
        candidate = checkpoint[
            "statistics"
        ]

        for key in reference:
            if not np.isclose(
                reference[key],
                candidate[key],
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    "Normalization statistics "
                    f"differ for {key}."
                )

    print(
        "All ensemble members use the same "
        "normalization statistics."
    )

    return reference


def create_data_loader(
    dataset_path,
    statistics,
):
    raw_data = load_npz_dataset(
        dataset_path
    )

    dataset = FormalDarcyDataset(
        data=raw_data,
        statistics=statistics,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return raw_data, data_loader


@torch.no_grad()
def predict_one_member(
    checkpoint,
    data_loader,
    statistics,
    device,
):
    model = build_fno(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    target_mean = statistics[
        "target_mean"
    ]

    target_std = statistics[
        "target_std"
    ]

    predictions = []
    targets = []

    for model_input, normalized_target in (
        data_loader
    ):
        model_input = model_input.to(
            device,
            non_blocking=True,
        )

        normalized_target = (
            normalized_target.to(
                device,
                non_blocking=True,
            )
        )

        normalized_prediction = model(
            model_input
        )

        prediction = (
            normalized_prediction * target_std
            + target_mean
        )

        target = (
            normalized_target * target_std
            + target_mean
        )

        predictions.append(
            prediction.cpu().numpy()
        )

        targets.append(
            target.cpu().numpy()
        )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    targets = np.concatenate(
        targets,
        axis=0,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions, targets


def relative_l2_per_sample(
    prediction,
    target,
):
    difference = prediction - target

    numerator = np.sqrt(
        np.sum(
            difference**2,
            axis=(1, 2, 3),
        )
    )

    denominator = np.sqrt(
        np.sum(
            target**2,
            axis=(1, 2, 3),
        )
    )

    return numerator / (
        denominator + 1e-8
    )


def calculate_risk_coverage(
    errors,
    uncertainty,
    number_of_points=19,
):
    """
    Retain samples from low to high uncertainty.

    coverage=0.5 means only the 50% least uncertain
    samples are retained.
    """

    order = np.argsort(uncertainty)

    coverages = np.linspace(
        0.1,
        1.0,
        number_of_points,
    )

    risks = []

    for coverage in coverages:
        count = max(
            1,
            int(
                np.ceil(
                    coverage * len(errors)
                )
            ),
        )

        selected = order[:count]

        risks.append(
            errors[selected].mean()
        )

    return coverages, np.array(risks)


def sampled_pixel_correlations(
    uncertainty,
    absolute_error,
    maximum_pixels=200000,
    seed=2026,
):
    uncertainty_flat = uncertainty.reshape(-1)
    error_flat = absolute_error.reshape(-1)

    number_of_pixels = len(
        uncertainty_flat
    )

    if number_of_pixels > maximum_pixels:
        rng = np.random.default_rng(seed)

        indices = rng.choice(
            number_of_pixels,
            size=maximum_pixels,
            replace=False,
        )

        uncertainty_flat = (
            uncertainty_flat[indices]
        )

        error_flat = error_flat[indices]

    return {
        "pixel_pearson": float(
            pearsonr(
                uncertainty_flat,
                error_flat,
            ).statistic
        ),
        "pixel_spearman": float(
            spearmanr(
                uncertainty_flat,
                error_flat,
            ).statistic
        ),
    }


def evaluate_ensemble(
    member_predictions,
    target,
):
    """
    member_predictions shape:
        (members, samples, 1, height, width)
    """

    ensemble_mean = np.mean(
        member_predictions,
        axis=0,
    )

    # ddof=0 follows the population-variance formula
    # used in the project definition.
    ensemble_std = np.std(
        member_predictions,
        axis=0,
        ddof=0,
    )

    ensemble_errors = relative_l2_per_sample(
        ensemble_mean,
        target,
    )

    member_errors = np.stack(
        [
            relative_l2_per_sample(
                member_prediction,
                target,
            )
            for member_prediction
            in member_predictions
        ],
        axis=0,
    )

    absolute_error = np.abs(
        ensemble_mean - target
    )

    # One uncertainty score for each sample.
    sample_uncertainty = np.mean(
        ensemble_std,
        axis=(1, 2, 3),
    )

    sample_pearson = pearsonr(
        sample_uncertainty,
        ensemble_errors,
    ).statistic

    sample_spearman = spearmanr(
        sample_uncertainty,
        ensemble_errors,
    ).statistic

    pixel_correlations = (
        sampled_pixel_correlations(
            uncertainty=ensemble_std,
            absolute_error=absolute_error,
        )
    )

    coverages, risks = (
        calculate_risk_coverage(
            errors=ensemble_errors,
            uncertainty=sample_uncertainty,
        )
    )

    # Diagnostic only: ensemble std is not yet
    # calibrated as a probabilistic interval.
    lower = (
        ensemble_mean
        - 1.96 * ensemble_std
    )

    upper = (
        ensemble_mean
        + 1.96 * ensemble_std
    )

    empirical_coverage_95 = np.mean(
        (target >= lower)
        & (target <= upper)
    )

    mean_interval_width_95 = np.mean(
        upper - lower
    )

    metrics = {
        "ensemble_relative_l2_mean": float(
            ensemble_errors.mean()
        ),
        "ensemble_relative_l2_std": float(
            ensemble_errors.std(ddof=1)
        ),
        "ensemble_relative_l2_median": float(
            np.median(ensemble_errors)
        ),
        "mean_individual_relative_l2": float(
            member_errors.mean()
        ),
        "ensemble_gain": float(
            member_errors.mean()
            - ensemble_errors.mean()
        ),
        "mean_sample_uncertainty": float(
            sample_uncertainty.mean()
        ),
        "sample_uncertainty_std": float(
            sample_uncertainty.std(ddof=1)
        ),
        "sample_pearson": float(
            sample_pearson
        ),
        "sample_spearman": float(
            sample_spearman
        ),
        "pixel_pearson":
            pixel_correlations[
                "pixel_pearson"
            ],
        "pixel_spearman":
            pixel_correlations[
                "pixel_spearman"
            ],
        "empirical_coverage_95": float(
            empirical_coverage_95
        ),
        "mean_interval_width_95": float(
            mean_interval_width_95
        ),
    }

    arrays = {
        "ensemble_mean": ensemble_mean,
        "ensemble_std": ensemble_std,
        "target": target,
        "absolute_error": absolute_error,
        "ensemble_errors": ensemble_errors,
        "member_errors": member_errors,
        "sample_uncertainty":
            sample_uncertainty,
        "coverages": coverages,
        "risks": risks,
    }

    return metrics, arrays


def plot_uncertainty_error(
    all_results,
):
    fig, axes = plt.subplots(
        1,
        len(all_results),
        figsize=(15, 4.5),
        squeeze=False,
    )

    for column, (
        dataset_name,
        result,
    ) in enumerate(all_results.items()):
        axis = axes[0, column]

        x = result["arrays"][
            "sample_uncertainty"
        ]

        y = result["arrays"][
            "ensemble_errors"
        ]

        axis.scatter(
            x,
            y,
            s=20,
            alpha=0.65,
        )

        axis.set_xlabel(
            "Mean ensemble uncertainty"
        )

        axis.set_ylabel(
            "Relative L2 error"
        )

        axis.set_title(
            DISPLAY_NAMES[dataset_name]
            + "\n"
            + "Spearman = "
            + f"{result['metrics']['sample_spearman']:.3f}"
        )

        axis.grid(alpha=0.25)

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "uncertainty_vs_error.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(f"Saved: {output_path}")


def plot_risk_coverage(
    all_results,
):
    plt.figure(figsize=(7, 5))

    for dataset_name, result in (
        all_results.items()
    ):
        plt.plot(
            result["arrays"]["coverages"],
            result["arrays"]["risks"],
            marker="o",
            markersize=3,
            label=DISPLAY_NAMES[dataset_name],
        )

    plt.xlabel("Coverage")
    plt.ylabel("Mean relative L2 risk")
    plt.title("Risk–coverage curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "risk_coverage.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(f"Saved: {output_path}")


def plot_uncertainty_distribution(
    all_results,
):
    labels = []
    values = []

    for dataset_name, result in (
        all_results.items()
    ):
        labels.append(
            DISPLAY_NAMES[dataset_name]
        )

        values.append(
            result["arrays"][
                "sample_uncertainty"
            ]
        )

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        values,
        tick_labels=labels,
        showfliers=True,
    )

    plt.ylabel("Mean ensemble uncertainty")
    plt.title(
        "Predictive uncertainty across distributions"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "uncertainty_distributions.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(f"Saved: {output_path}")


def plot_uncertainty_maps(
    all_results,
    raw_data_by_name,
):
    number_of_rows = len(all_results)

    fig, axes = plt.subplots(
        number_of_rows,
        5,
        figsize=(16, 3.3 * number_of_rows),
        squeeze=False,
    )

    titles = [
        "True log(a)",
        "Sparse pressure",
        "Ensemble mean",
        "Absolute error",
        "Ensemble std",
    ]

    for column, title in enumerate(titles):
        axes[0, column].set_title(title)

    for row, (
        dataset_name,
        result,
    ) in enumerate(all_results.items()):
        arrays = result["arrays"]

        # Display the highest-error sample for each
        # distribution.
        sample_index = int(
            np.argmax(
                arrays["ensemble_errors"]
            )
        )

        target = arrays["target"][
            sample_index,
            0,
        ]

        prediction = arrays[
            "ensemble_mean"
        ][
            sample_index,
            0,
        ]

        absolute_error = arrays[
            "absolute_error"
        ][
            sample_index,
            0,
        ]

        uncertainty = arrays[
            "ensemble_std"
        ][
            sample_index,
            0,
        ]

        observation = raw_data_by_name[
            dataset_name
        ]["observed_pressure"][
            sample_index
        ]

        field_min = min(
            target.min(),
            prediction.min(),
        )

        field_max = max(
            target.max(),
            prediction.max(),
        )

        fields = [
            (
                target,
                "viridis",
                field_min,
                field_max,
            ),
            (
                observation,
                "coolwarm",
                0.0,
                1.0,
            ),
            (
                prediction,
                "viridis",
                field_min,
                field_max,
            ),
            (
                absolute_error,
                "magma",
                0.0,
                absolute_error.max(),
            ),
            (
                uncertainty,
                "magma",
                0.0,
                uncertainty.max(),
            ),
        ]

        for column, (
            field,
            colour_map,
            minimum,
            maximum,
        ) in enumerate(fields):
            image = axes[
                row,
                column,
            ].imshow(
                field,
                origin="lower",
                cmap=colour_map,
                vmin=minimum,
                vmax=maximum,
            )

            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

            fig.colorbar(
                image,
                ax=axes[row, column],
                fraction=0.046,
                pad=0.03,
            )

        axes[row, 0].set_ylabel(
            DISPLAY_NAMES[dataset_name]
            + "\n"
            + f"index {sample_index}\n"
            + "error "
            + f"{arrays['ensemble_errors'][sample_index]:.3f}"
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "uncertainty_maps.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(f"Saved: {output_path}")


def print_summary(all_results):
    print()
    print("Ensemble uncertainty summary")
    print("----------------------------")

    print(
        f"{'Dataset':<20}"
        f"{'Ens RelL2':>12}"
        f"{'Gain':>10}"
        f"{'Uncertainty':>14}"
        f"{'Samp rho':>11}"
        f"{'Pixel rho':>11}"
        f"{'95% cov':>10}"
    )

    for dataset_name, result in (
        all_results.items()
    ):
        metrics = result["metrics"]

        print(
            f"{DISPLAY_NAMES[dataset_name]:<20}"
            f"{metrics['ensemble_relative_l2_mean']:>12.6f}"
            f"{metrics['ensemble_gain']:>10.6f}"
            f"{metrics['mean_sample_uncertainty']:>14.6f}"
            f"{metrics['sample_spearman']:>11.4f}"
            f"{metrics['pixel_spearman']:>11.4f}"
            f"{metrics['empirical_coverage_95']:>10.4f}"
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    checkpoints = [
        load_checkpoint(
            checkpoint_path(seed),
            device,
        )
        for seed in ENSEMBLE_SEEDS
    ]

    statistics = (
        verify_checkpoint_statistics(
            checkpoints
        )
    )

    all_results = {}
    raw_data_by_name = {}

    for dataset_name, dataset_path in (
        DATASETS.items()
    ):
        print()
        print(
            f"Evaluating ensemble on "
            f"{DISPLAY_NAMES[dataset_name]}"
        )

        raw_data, data_loader = (
            create_data_loader(
                dataset_path,
                statistics,
            )
        )

        raw_data_by_name[
            dataset_name
        ] = raw_data

        member_predictions = []
        reference_target = None

        for member_index, checkpoint in (
            enumerate(checkpoints)
        ):
            print(
                f"  Member "
                f"{member_index + 1}"
                f"/{len(checkpoints)}"
            )

            prediction, target = (
                predict_one_member(
                    checkpoint=checkpoint,
                    data_loader=data_loader,
                    statistics=statistics,
                    device=device,
                )
            )

            member_predictions.append(
                prediction
            )

            if reference_target is None:
                reference_target = target
            elif not np.allclose(
                reference_target,
                target,
            ):
                raise RuntimeError(
                    "Targets changed between "
                    "ensemble members."
                )

        member_predictions = np.stack(
            member_predictions,
            axis=0,
        )

        metrics, arrays = evaluate_ensemble(
            member_predictions=
                member_predictions,
            target=reference_target,
        )

        all_results[dataset_name] = {
            "metrics": metrics,
            "arrays": arrays,
        }

        np.savez_compressed(
            OUTPUT_DIRECTORY
            / f"{dataset_name}_results.npz",
            **arrays,
        )

    print_summary(all_results)

    summary = {
        dataset_name: result["metrics"]
        for dataset_name, result
        in all_results.items()
    }

    with open(
        OUTPUT_DIRECTORY
        / "uncertainty_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    plot_uncertainty_error(all_results)
    plot_risk_coverage(all_results)
    plot_uncertainty_distribution(
        all_results
    )
    plot_uncertainty_maps(
        all_results,
        raw_data_by_name,
    )

    print()
    print(
        "Saved all results to:",
        OUTPUT_DIRECTORY,
    )


if __name__ == "__main__":
    main()