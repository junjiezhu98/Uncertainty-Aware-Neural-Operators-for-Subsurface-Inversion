from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from formal_dataset import (
    load_npz_dataset,
    FormalDarcyDataset,
)

from train_unet_overfit import UNet
from fno_model import FNO2d


DATA_PATH = Path(
    "data/ood/darcy_ood_length_040_300.npz"
)

UNET_CHECKPOINT_PATH = Path(
    "outputs/formal_unet/unet_best.pt"
)

FNO_CHECKPOINT_PATH = Path(
    "outputs/formal_fno/fno_best.pt"
)

OUTPUT_DIRECTORY = Path(
    "outputs/failure_analysis"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}"
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def build_unet(device):
    return UNet(
        input_channels=2,
        output_channels=1,
        base_channels=32,
    ).to(device)


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


@torch.no_grad()
def predict_dataset(
    model,
    data_loader,
    statistics,
    device,
):
    model.eval()

    target_mean = statistics["target_mean"]
    target_std = statistics["target_std"]

    predictions = []
    targets = []
    relative_errors = []

    for model_input, normalized_target in data_loader:
        model_input = model_input.to(
            device,
            non_blocking=True,
        )

        normalized_target = normalized_target.to(
            device,
            non_blocking=True,
        )

        normalized_prediction = model(
            model_input
        )

        physical_prediction = (
            normalized_prediction * target_std
            + target_mean
        )

        physical_target = (
            normalized_target * target_std
            + target_mean
        )

        numerator = torch.linalg.vector_norm(
            physical_prediction
            - physical_target,
            dim=(1, 2, 3),
        )

        denominator = torch.linalg.vector_norm(
            physical_target,
            dim=(1, 2, 3),
        )

        relative_error = numerator / (
            denominator + 1e-8
        )

        predictions.append(
            physical_prediction.cpu().numpy()
        )

        targets.append(
            physical_target.cpu().numpy()
        )

        relative_errors.append(
            relative_error.cpu().numpy()
        )

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(relative_errors, axis=0),
    )


def paired_bootstrap(
    unet_errors,
    fno_errors,
    number_of_bootstrap_samples=10000,
    seed=2026,
):
    """
    Positive difference means FNO has higher error,
    so U-Net is better.
    """

    differences = (
        fno_errors - unet_errors
    )

    rng = np.random.default_rng(seed)

    sample_indices = rng.integers(
        low=0,
        high=len(differences),
        size=(
            number_of_bootstrap_samples,
            len(differences),
        ),
    )

    bootstrap_mean_differences = (
        differences[sample_indices].mean(axis=1)
    )

    lower, upper = np.percentile(
        bootstrap_mean_differences,
        [2.5, 97.5],
    )

    result = {
        "mean_difference": float(
            differences.mean()
        ),
        "median_difference": float(
            np.median(differences)
        ),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "unet_win_fraction": float(
            np.mean(unet_errors < fno_errors)
        ),
        "fno_win_fraction": float(
            np.mean(fno_errors < unet_errors)
        ),
    }

    return result, bootstrap_mean_differences


def plot_error_distributions(
    unet_errors,
    fno_errors,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    bins = np.linspace(
        0.0,
        max(
            unet_errors.max(),
            fno_errors.max(),
        ),
        30,
    )

    axes[0].hist(
        unet_errors,
        bins=bins,
        alpha=0.6,
        label="U-Net",
    )

    axes[0].hist(
        fno_errors,
        bins=bins,
        alpha=0.6,
        label="FNO",
    )

    axes[0].set_xlabel("Relative L2 error")
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title("Error distributions")
    axes[0].legend()

    maximum_error = max(
        unet_errors.max(),
        fno_errors.max(),
    )

    axes[1].scatter(
        unet_errors,
        fno_errors,
        s=18,
        alpha=0.65,
    )

    axes[1].plot(
        [0, maximum_error],
        [0, maximum_error],
        color="black",
        linestyle="--",
        linewidth=1,
    )

    axes[1].set_xlabel("U-Net relative L2")
    axes[1].set_ylabel("FNO relative L2")
    axes[1].set_title("Paired sample errors")
    axes[1].set_aspect(
        "equal",
        adjustable="box",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "error_distributions.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(f"Saved: {output_path}")


def plot_hardest_samples(
    true_fields,
    observed_pressure,
    unet_predictions,
    fno_predictions,
    unet_errors,
    fno_errors,
    number_of_samples=5,
):
    joint_error = (
        unet_errors + fno_errors
    ) / 2.0

    hardest_indices = np.argsort(
        joint_error
    )[::-1][:number_of_samples]

    fig, axes = plt.subplots(
        number_of_samples,
        6,
        figsize=(18, 3 * number_of_samples),
        squeeze=False,
    )

    column_titles = [
        "True log(a)",
        "Sparse pressure",
        "U-Net prediction",
        "U-Net error",
        "FNO prediction",
        "FNO error",
    ]

    for column, title in enumerate(
        column_titles
    ):
        axes[0, column].set_title(title)

    for row, sample_index in enumerate(
        hardest_indices
    ):
        true_field = true_fields[
            sample_index,
            0,
        ]

        unet_prediction = unet_predictions[
            sample_index,
            0,
        ]

        fno_prediction = fno_predictions[
            sample_index,
            0,
        ]

        unet_absolute_error = np.abs(
            unet_prediction - true_field
        )

        fno_absolute_error = np.abs(
            fno_prediction - true_field
        )

        field_min = min(
            true_field.min(),
            unet_prediction.min(),
            fno_prediction.min(),
        )

        field_max = max(
            true_field.max(),
            unet_prediction.max(),
            fno_prediction.max(),
        )

        error_max = max(
            unet_absolute_error.max(),
            fno_absolute_error.max(),
        )

        fields = [
            (
                true_field,
                "viridis",
                field_min,
                field_max,
            ),
            (
                observed_pressure[sample_index],
                "coolwarm",
                0.0,
                1.0,
            ),
            (
                unet_prediction,
                "viridis",
                field_min,
                field_max,
            ),
            (
                unet_absolute_error,
                "magma",
                0.0,
                error_max,
            ),
            (
                fno_prediction,
                "viridis",
                field_min,
                field_max,
            ),
            (
                fno_absolute_error,
                "magma",
                0.0,
                error_max,
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

            axes[
                row,
                column,
            ].set_xticks([])

            axes[
                row,
                column,
            ].set_yticks([])

            fig.colorbar(
                image,
                ax=axes[row, column],
                fraction=0.046,
                pad=0.03,
            )

        axes[row, 0].set_ylabel(
            f"Index {sample_index}\n"
            f"U={unet_errors[sample_index]:.3f}\n"
            f"F={fno_errors[sample_index]:.3f}"
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "hardest_ood_samples.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(f"Saved: {output_path}")

    print()
    print("Hardest sample indices")
    print("----------------------")

    for sample_index in hardest_indices:
        print(
            f"Index {sample_index:3d} | "
            f"U-Net: "
            f"{unet_errors[sample_index]:.6f} | "
            f"FNO: "
            f"{fno_errors[sample_index]:.6f}"
        )

    return hardest_indices


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    raw_data = load_npz_dataset(
        DATA_PATH
    )

    unet_checkpoint = load_checkpoint(
        UNET_CHECKPOINT_PATH,
        device,
    )

    fno_checkpoint = load_checkpoint(
        FNO_CHECKPOINT_PATH,
        device,
    )

    unet_statistics = unet_checkpoint[
        "statistics"
    ]

    fno_statistics = fno_checkpoint[
        "statistics"
    ]

    # Both models should have used the same training
    # data and normalization.
    for key in unet_statistics:
        difference = abs(
            unet_statistics[key]
            - fno_statistics[key]
        )

        if difference > 1e-7:
            raise ValueError(
                f"Normalization differs for {key}."
            )

    dataset = FormalDarcyDataset(
        data=raw_data,
        statistics=unet_statistics,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    unet = build_unet(device)
    unet.load_state_dict(
        unet_checkpoint["model_state_dict"]
    )

    fno = build_fno(device)
    fno.load_state_dict(
        fno_checkpoint["model_state_dict"]
    )

    (
        unet_predictions,
        true_fields,
        unet_errors,
    ) = predict_dataset(
        model=unet,
        data_loader=data_loader,
        statistics=unet_statistics,
        device=device,
    )

    (
        fno_predictions,
        second_true_fields,
        fno_errors,
    ) = predict_dataset(
        model=fno,
        data_loader=data_loader,
        statistics=fno_statistics,
        device=device,
    )

    if not np.allclose(
        true_fields,
        second_true_fields,
    ):
        raise RuntimeError(
            "Targets differ between model evaluations."
        )

    error_correlation = np.corrcoef(
        unet_errors,
        fno_errors,
    )[0, 1]

    bootstrap_result, bootstrap_samples = (
        paired_bootstrap(
            unet_errors=unet_errors,
            fno_errors=fno_errors,
            number_of_bootstrap_samples=10000,
            seed=2026,
        )
    )

    print()
    print("Paired comparison")
    print("-----------------")
    print(
        "U-Net mean error:",
        f"{unet_errors.mean():.6f}",
    )
    print(
        "FNO mean error:",
        f"{fno_errors.mean():.6f}",
    )
    print(
        "Mean FNO minus U-Net:",
        f"{bootstrap_result['mean_difference']:.6f}",
    )
    print(
        "95% bootstrap confidence interval:",
        f"[{bootstrap_result['ci_lower']:.6f}, "
        f"{bootstrap_result['ci_upper']:.6f}]",
    )
    print(
        "U-Net win fraction:",
        f"{bootstrap_result['unet_win_fraction']:.3f}",
    )
    print(
        "FNO win fraction:",
        f"{bootstrap_result['fno_win_fraction']:.3f}",
    )
    print(
        "Per-sample error correlation:",
        f"{error_correlation:.3f}",
    )

    plot_error_distributions(
        unet_errors,
        fno_errors,
    )

    hardest_indices = plot_hardest_samples(
        true_fields=true_fields,
        observed_pressure=raw_data[
            "observed_pressure"
        ],
        unet_predictions=unet_predictions,
        fno_predictions=fno_predictions,
        unet_errors=unet_errors,
        fno_errors=fno_errors,
        number_of_samples=5,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "failure_analysis_results.npz"
    )

    np.savez(
        output_path,
        unet_errors=unet_errors,
        fno_errors=fno_errors,
        error_difference=(
            fno_errors - unet_errors
        ),
        hardest_indices=hardest_indices,
        bootstrap_mean_differences=(
            bootstrap_samples
        ),
        unet_predictions=unet_predictions,
        fno_predictions=fno_predictions,
        true_fields=true_fields,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()