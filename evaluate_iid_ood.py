from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from formal_dataset import (
    load_npz_dataset,
    FormalDarcyDataset,
)

from train_unet_overfit import UNet
from fno_model import FNO2d


UNET_CHECKPOINT = Path(
    "outputs/formal_unet/unet_best.pt"
)

FNO_CHECKPOINT = Path(
    "outputs/formal_fno/fno_best.pt"
)

TEST_DATASETS = {
    "IID length 0.15": Path(
        "data/formal/darcy_iid_test_300.npz"
    ),
    "OOD length 0.10": Path(
        "data/ood/darcy_ood_length_010_300.npz"
    ),
    "OOD length 0.40": Path(
        "data/ood/darcy_ood_length_040_300.npz"
    ),
}


def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}"
        )

    # Use weights_only=False only for checkpoints
    # created locally by your own training code.
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    return checkpoint


def create_loader(
    dataset_path,
    statistics,
    batch_size=32,
):
    data = load_npz_dataset(dataset_path)

    dataset = FormalDarcyDataset(
        data=data,
        statistics=statistics,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return loader


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    statistics,
    device,
):
    model.eval()

    target_mean = statistics["target_mean"]
    target_std = statistics["target_std"]

    normalized_squared_errors = []
    physical_squared_errors = []
    relative_l2_errors = []

    for model_input, normalized_target in data_loader:
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

        # MSE for each sample in normalized space.
        normalized_mse = torch.mean(
            (
                normalized_prediction
                - normalized_target
            ) ** 2,
            dim=(1, 2, 3),
        )

        physical_target = (
            normalized_target * target_std
            + target_mean
        )

        physical_prediction = (
            normalized_prediction * target_std
            + target_mean
        )

        physical_mse = torch.mean(
            (
                physical_prediction
                - physical_target
            ) ** 2,
            dim=(1, 2, 3),
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

        relative_l2 = numerator / (
            denominator + 1e-8
        )

        normalized_squared_errors.append(
            normalized_mse.cpu()
        )

        physical_squared_errors.append(
            physical_mse.cpu()
        )

        relative_l2_errors.append(
            relative_l2.cpu()
        )

    normalized_squared_errors = torch.cat(
        normalized_squared_errors
    ).numpy()

    physical_squared_errors = torch.cat(
        physical_squared_errors
    ).numpy()

    relative_l2_errors = torch.cat(
        relative_l2_errors
    ).numpy()

    metrics = {
        "normalized_mse_mean": float(
            normalized_squared_errors.mean()
        ),
        "physical_mse_mean": float(
            physical_squared_errors.mean()
        ),
        "relative_l2_mean": float(
            relative_l2_errors.mean()
        ),
        "relative_l2_std": float(
            relative_l2_errors.std(ddof=1)
        ),
        "relative_l2_median": float(
            np.median(relative_l2_errors)
        ),
    }

    return metrics


def build_unet(device):
    model = UNet(
        input_channels=2,
        output_channels=1,
        base_channels=32,
    )

    return model.to(device)


def build_fno(device):
    model = FNO2d(
        observation_channels=2,
        output_channels=1,
        width=32,
        modes_x=12,
        modes_y=12,
        number_of_layers=4,
        padding=8,
        use_coordinates=True,
    )

    return model.to(device)


def print_results(results):
    print()
    print(
        f"{'Model':<8}"
        f"{'Dataset':<20}"
        f"{'MSE':>12}"
        f"{'Relative L2':>15}"
        f"{'Std':>12}"
        f"{'Median':>12}"
    )

    print("-" * 79)

    for row in results:
        print(
            f"{row['model']:<8}"
            f"{row['dataset']:<20}"
            f"{row['normalized_mse_mean']:>12.6f}"
            f"{row['relative_l2_mean']:>15.6f}"
            f"{row['relative_l2_std']:>12.6f}"
            f"{row['relative_l2_median']:>12.6f}"
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model_configurations = [
        {
            "name": "U-Net",
            "checkpoint_path": UNET_CHECKPOINT,
            "build_model": build_unet,
        },
        {
            "name": "FNO",
            "checkpoint_path": FNO_CHECKPOINT,
            "build_model": build_fno,
        },
    ]

    results = []

    for configuration in model_configurations:
        checkpoint = load_checkpoint(
            configuration["checkpoint_path"],
            device,
        )

        statistics = checkpoint["statistics"]

        model = configuration[
            "build_model"
        ](device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print()
        print(
            f"Evaluating "
            f"{configuration['name']}"
        )

        for dataset_name, dataset_path in (
            TEST_DATASETS.items()
        ):
            print(f"  {dataset_name}")

            data_loader = create_loader(
                dataset_path=dataset_path,
                statistics=statistics,
                batch_size=32,
            )

            metrics = evaluate_model(
                model=model,
                data_loader=data_loader,
                statistics=statistics,
                device=device,
            )

            results.append(
                {
                    "model":
                        configuration["name"],
                    "dataset": dataset_name,
                    **metrics,
                }
            )

    print_results(results)

    output_path = Path(
        "outputs/iid_ood_results.npz"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,
        model=np.array(
            [row["model"] for row in results]
        ),
        dataset=np.array(
            [row["dataset"] for row in results]
        ),
        normalized_mse=np.array(
            [
                row["normalized_mse_mean"]
                for row in results
            ]
        ),
        physical_mse=np.array(
            [
                row["physical_mse_mean"]
                for row in results
            ]
        ),
        relative_l2_mean=np.array(
            [
                row["relative_l2_mean"]
                for row in results
            ]
        ),
        relative_l2_std=np.array(
            [
                row["relative_l2_std"]
                for row in results
            ]
        ),
        relative_l2_median=np.array(
            [
                row["relative_l2_median"]
                for row in results
            ]
        ),
    )

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()