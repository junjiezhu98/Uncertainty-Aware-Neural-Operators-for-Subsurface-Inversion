from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from formal_dataset import FormalDarcyDataset, load_npz_dataset
from train_unet_overfit import UNet


ROOT = Path(".")
OUTPUT = ROOT / "outputs" / "application_figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "iid": ROOT / "data" / "formal" / "darcy_iid_test_300.npz",
    "ood_010": ROOT / "data" / "ood" / "darcy_ood_length_010_300.npz",
    "ood_040": ROOT / "data" / "ood" / "darcy_ood_length_040_300.npz",
}

ENSEMBLE_RESULTS = {
    name: ROOT / "outputs" / "fno_uncertainty" / f"{name}_results.npz"
    for name in DATASETS
}

UNET_CHECKPOINT = ROOT / "outputs" / "formal_unet" / "unet_best.pt"
FAILURE_RESULTS = (
    ROOT / "outputs" / "failure_analysis" / "failure_analysis_results.npz"
)

DISPLAY_NAMES = {
    "iid": "IID  l = 0.15",
    "ood_010": "Fine OOD  l = 0.10",
    "ood_040": "Smooth OOD  l = 0.40",
}

MODEL_COLORS = {
    "U-Net": "#2B6CB0",
    "FNO": "#DD6B20",
    "FNO ensemble": "#2F855A",
}

# Exact metrics from the completed experiments.
RELATIVE_L2 = {
    "U-Net": [0.527558, 0.597468, 0.592921],
    "FNO": [0.529490, 0.597545, 0.609506],
    "FNO ensemble": [0.518535, 0.588452, 0.600736],
}


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def require_files(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required files are missing:\n{formatted}")


def save_figure(fig, stem):
    png_path = OUTPUT / f"{stem}.png"
    pdf_path = OUTPUT / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def load_result(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def build_unet(device):
    return UNet(
        input_channels=2,
        output_channels=1,
        base_channels=32,
    ).to(device)


@torch.no_grad()
def predict_unet(dataset_path, checkpoint, device):
    raw_data = load_npz_dataset(dataset_path)
    statistics = checkpoint["statistics"]
    dataset = FormalDarcyDataset(raw_data, statistics)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_unet(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    target_mean = statistics["target_mean"]
    target_std = statistics["target_std"]

    for model_input, _ in loader:
        normalized_prediction = model(model_input.to(device, non_blocking=True))
        prediction = normalized_prediction * target_std + target_mean
        predictions.append(prediction.cpu().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(predictions, axis=0), raw_data


def add_panel_label(axis, label):
    axis.text(
        -0.10,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def figure_1_inversion_examples(device):
    checkpoint = torch.load(
        UNET_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    rows = []
    for dataset_name, dataset_path in DATASETS.items():
        ensemble = load_result(ENSEMBLE_RESULTS[dataset_name])
        unet_prediction, raw_data = predict_unet(
            dataset_path, checkpoint, device
        )

        errors = ensemble["ensemble_errors"]
        median_error = np.median(errors)
        sample_index = int(np.argmin(np.abs(errors - median_error)))

        rows.append(
            {
                "name": dataset_name,
                "index": sample_index,
                "true": ensemble["target"][sample_index, 0],
                "observation": raw_data["observed_pressure"][sample_index],
                "unet": unet_prediction[sample_index, 0],
                "ensemble": ensemble["ensemble_mean"][sample_index, 0],
                "error": ensemble["absolute_error"][sample_index, 0],
                "uncertainty": ensemble["ensemble_std"][sample_index, 0],
                "relative_error": errors[sample_index],
            }
        )

    fig, axes = plt.subplots(3, 6, figsize=(17.0, 8.4), constrained_layout=True)
    titles = [
        "True log permeability",
        "Sparse pressure",
        "U-Net prediction",
        "FNO ensemble mean",
        "Absolute error",
        "Ensemble uncertainty",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="semibold")

    for row_index, row in enumerate(rows):
        field_min = min(row["true"].min(), row["unet"].min(), row["ensemble"].min())
        field_max = max(row["true"].max(), row["unet"].max(), row["ensemble"].max())
        fields = [
            (row["true"], "viridis", field_min, field_max),
            (row["observation"], "coolwarm", 0.0, 1.0),
            (row["unet"], "viridis", field_min, field_max),
            (row["ensemble"], "viridis", field_min, field_max),
            (row["error"], "magma", 0.0, row["error"].max()),
            (row["uncertainty"], "magma", 0.0, row["uncertainty"].max()),
        ]

        for column, (field, cmap, vmin, vmax) in enumerate(fields):
            image = axes[row_index, column].imshow(
                field,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.025)

        axes[row_index, 0].set_ylabel(
            f"{DISPLAY_NAMES[row['name']]}\n"
            f"sample {row['index']}  |  Rel. L2 = {row['relative_error']:.3f}",
            fontweight="semibold",
        )

    fig.suptitle(
        "Subsurface inversion across in-distribution and shifted geology",
        fontweight="bold",
    )
    save_figure(fig, "figure_1_inversion_examples")
    plt.close(fig)


def figure_2_iid_ood_performance():
    categories = [DISPLAY_NAMES[name] for name in DATASETS]
    x = np.arange(len(categories))

    fig, axis = plt.subplots(figsize=(8.7, 5.2), constrained_layout=True)

    markers = {"U-Net": "o", "FNO": "s", "FNO ensemble": "D"}
    for model, values in RELATIVE_L2.items():
        axis.plot(
            x,
            values,
            marker=markers[model],
            markersize=7,
            linewidth=2.0,
            color=MODEL_COLORS[model],
            label=model,
        )
        for x_value, value in zip(x, values):
            axis.annotate(
                f"{value:.3f}",
                (x_value, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=MODEL_COLORS[model],
            )

    axis.set_xticks(x, categories)
    axis.set_ylabel("Mean relative L2 error  (lower is better)")
    axis.set_ylim(0.48, 0.64)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.set_title(
        "Reconstruction accuracy degrades under geological distribution shift",
        fontweight="bold",
    )
    axis.text(
        0.01,
        -0.18,
        "All test sets contain 300 independently generated samples with 64 sensors and 1% noise.",
        transform=axis.transAxes,
        fontsize=9,
        color="#4A5568",
    )

    save_figure(fig, "figure_2_iid_ood_performance")
    plt.close(fig)


def figure_3_uncertainty_evaluation():
    results = {name: load_result(path) for name, path in ENSEMBLE_RESULTS.items()}

    fig = plt.figure(figsize=(15.5, 8.2), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.05])

    for column, dataset_name in enumerate(DATASETS):
        axis = fig.add_subplot(grid[0, column])
        uncertainty = results[dataset_name]["sample_uncertainty"]
        error = results[dataset_name]["ensemble_errors"]
        rank_correlation = float(spearmanr(uncertainty, error).statistic)
        axis.scatter(uncertainty, error, s=17, alpha=0.58, color="#2F855A", edgecolors="none")
        axis.set_xlabel("Mean ensemble uncertainty")
        axis.set_ylabel("Relative L2 error")
        axis.set_title(f"{DISPLAY_NAMES[dataset_name]}\nrank correlation = {rank_correlation:.3f}")
        axis.grid(alpha=0.22)
        add_panel_label(axis, chr(ord("a") + column))

    risk_axis = fig.add_subplot(grid[1, :2])
    colors = {"iid": "#2B6CB0", "ood_010": "#DD6B20", "ood_040": "#805AD5"}
    for dataset_name in DATASETS:
        risk_axis.plot(
            results[dataset_name]["coverages"],
            results[dataset_name]["risks"],
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=colors[dataset_name],
            label=DISPLAY_NAMES[dataset_name],
        )
    risk_axis.set_xlabel("Coverage retained from lowest to highest uncertainty")
    risk_axis.set_ylabel("Mean relative L2 risk")
    risk_axis.set_title("Risk–coverage curves reveal weak uncertainty ranking")
    risk_axis.grid(alpha=0.25)
    risk_axis.legend(frameon=False)
    add_panel_label(risk_axis, "d")

    box_axis = fig.add_subplot(grid[1, 2])
    uncertainty_values = [results[name]["sample_uncertainty"] for name in DATASETS]
    box = box_axis.boxplot(
        uncertainty_values,
        labels=["IID", "Fine OOD", "Smooth OOD"],
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(box["boxes"], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    box_axis.set_ylabel("Mean ensemble uncertainty")
    box_axis.set_title("Smooth OOD appears less uncertain")
    box_axis.grid(axis="y", alpha=0.25)
    add_panel_label(box_axis, "e")

    fig.suptitle(
        "Ensemble disagreement improves accuracy but is not a reliable error indicator",
        fontweight="bold",
    )
    save_figure(fig, "figure_3_uncertainty_evaluation")
    plt.close(fig)


def figure_4_failure_cases():
    results = load_result(FAILURE_RESULTS)
    unet_errors = results["unet_errors"]
    fno_errors = results["fno_errors"]
    true_fields = results["true_fields"]
    unet_predictions = results["unet_predictions"]
    fno_predictions = results["fno_predictions"]

    joint_error = 0.5 * (unet_errors + fno_errors)
    indices = np.argsort(joint_error)[::-1][:3]

    fig, axes = plt.subplots(3, 5, figsize=(14.5, 8.4), constrained_layout=True)
    titles = [
        "True log permeability",
        "U-Net prediction",
        "U-Net absolute error",
        "FNO prediction",
        "FNO absolute error",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="semibold")

    for row, index in enumerate(indices):
        true_field = true_fields[index, 0]
        unet_prediction = unet_predictions[index, 0]
        fno_prediction = fno_predictions[index, 0]
        unet_error_map = np.abs(unet_prediction - true_field)
        fno_error_map = np.abs(fno_prediction - true_field)

        field_min = min(true_field.min(), unet_prediction.min(), fno_prediction.min())
        field_max = max(true_field.max(), unet_prediction.max(), fno_prediction.max())
        error_max = max(unet_error_map.max(), fno_error_map.max())
        fields = [
            (true_field, "viridis", field_min, field_max),
            (unet_prediction, "viridis", field_min, field_max),
            (unet_error_map, "magma", 0.0, error_max),
            (fno_prediction, "viridis", field_min, field_max),
            (fno_error_map, "magma", 0.0, error_max),
        ]

        for column, (field, cmap, vmin, vmax) in enumerate(fields):
            image = axes[row, column].imshow(
                field,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.025)

        axes[row, 0].set_ylabel(
            f"Sample {index}\nU-Net {unet_errors[index]:.3f}  |  FNO {fno_errors[index]:.3f}",
            fontweight="semibold",
        )

    correlation = float(np.corrcoef(unet_errors, fno_errors)[0, 1])
    fig.suptitle(
        "Shared failure cases under smooth geological shift"
        f"  |  paired error correlation = {correlation:.3f}",
        fontweight="bold",
    )
    save_figure(fig, "figure_4_failure_cases")
    plt.close(fig)


def main():
    configure_style()
    require_files(
        [
            *DATASETS.values(),
            *ENSEMBLE_RESULTS.values(),
            UNET_CHECKPOINT,
            FAILURE_RESULTS,
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    figure_1_inversion_examples(device)
    figure_2_iid_ood_performance()
    figure_3_uncertainty_evaluation()
    figure_4_failure_cases()

    print(f"\nAll application figures saved to: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()