from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from formal_dataset import (
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    build_formal_dataloaders,
)

from fno_model import (
    FNO2d,
    count_parameters,
)

from train_unet_formal import (
    train_model,
    evaluate,
)


ENSEMBLE_DIRECTORY = Path(
    "outputs/fno_ensemble"
)

ENSEMBLE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

# Existing FNO trained with seed 2026.
EXISTING_CHECKPOINT = Path(
    "outputs/formal_fno/fno_best.pt"
)

ENSEMBLE_SEEDS = [
    2026,
    2027,
    2028,
    2029,
    2030,
]


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These settings improve reproducibility.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def save_existing_member(
    destination_path,
    device,
):
    if not EXISTING_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Existing seed-2026 checkpoint not found: "
            f"{EXISTING_CHECKPOINT.resolve()}"
        )

    checkpoint = torch.load(
        EXISTING_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    checkpoint["seed"] = 2026

    checkpoint["architecture"] = {
        "name": "FNO2d",
        "observation_channels": 2,
        "output_channels": 1,
        "width": 32,
        "modes_x": 12,
        "modes_y": 12,
        "number_of_layers": 4,
        "padding": 8,
        "use_coordinates": True,
    }

    torch.save(
        checkpoint,
        destination_path,
    )

    print(
        "Copied existing seed 2026 checkpoint to:",
        destination_path,
    )


def train_member(
    seed,
    train_loader,
    validation_loader,
    test_loader,
    statistics,
    device,
):
    set_random_seed(seed)

    print()
    print("=" * 64)
    print(f"Training FNO ensemble member with seed {seed}")
    print("=" * 64)

    model = build_fno(device)

    print(
        "Trainable complex/real parameter elements:",
        f"{count_parameters(model):,}",
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        number_of_epochs=150,
        learning_rate=1e-3,
        patience=20,
    )

    loss_function = nn.MSELoss()

    validation_metrics = evaluate(
        model=model,
        data_loader=validation_loader,
        loss_function=loss_function,
        device=device,
    )

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    member_directory = (
        ENSEMBLE_DIRECTORY
        / f"seed_{seed}"
    )

    member_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        member_directory
        / "fno_best.pt"
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),
        "statistics": statistics,
        "history": history,
        "validation_metrics":
            validation_metrics,
        "test_metrics": test_metrics,
        "seed": seed,
        "architecture": {
            "name": "FNO2d",
            "observation_channels": 2,
            "output_channels": 1,
            "width": 32,
            "modes_x": 12,
            "modes_y": 12,
            "number_of_layers": 4,
            "padding": 8,
            "use_coordinates": True,
        },
        "dataset_paths": {
            "train": str(TRAIN_PATH),
            "validation":
                str(VALIDATION_PATH),
            "test": str(TEST_PATH),
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print()
    print(f"Seed {seed} complete")
    print(
        "Best epoch:",
        history["best_epoch"],
    )
    print(
        "Validation relative L2:",
        f"{validation_metrics['relative_l2_mean']:.6f}",
    )
    print(
        "Test relative L2:",
        f"{test_metrics['relative_l2_mean']:.6f}",
    )
    print(
        "Saved checkpoint:",
        checkpoint_path,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "best_epoch":
            history["best_epoch"],
        "best_validation_mse":
            history["best_validation_loss"],
        "validation_relative_l2":
            validation_metrics[
                "relative_l2_mean"
            ],
        "test_mse":
            test_metrics["mse"],
        "test_relative_l2":
            test_metrics[
                "relative_l2_mean"
            ],
        "checkpoint":
            str(checkpoint_path),
    }


def load_member_summary(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    history = checkpoint.get(
        "history",
        {}
    )

    validation_metrics = checkpoint.get(
        "validation_metrics",
        {}
    )

    test_metrics = checkpoint.get(
        "test_metrics",
        {}
    )

    return {
        "seed": int(
            checkpoint.get("seed", 2026)
        ),
        "best_epoch": int(
            history.get("best_epoch", -1)
        ),
        "best_validation_mse": float(
            history.get(
                "best_validation_loss",
                validation_metrics.get(
                    "mse",
                    np.nan,
                ),
            )
        ),
        "validation_relative_l2": float(
            validation_metrics.get(
                "relative_l2_mean",
                np.nan,
            )
        ),
        "test_mse": float(
            test_metrics.get(
                "mse",
                np.nan,
            )
        ),
        "test_relative_l2": float(
            test_metrics.get(
                "relative_l2_mean",
                np.nan,
            )
        ),
        "checkpoint": str(
            checkpoint_path
        ),
    }


def print_summary(summaries):
    print()
    print("Ensemble member summary")
    print("-----------------------")

    print(
        f"{'Seed':>6}"
        f"{'Epoch':>8}"
        f"{'Val MSE':>12}"
        f"{'Val RelL2':>12}"
        f"{'Test MSE':>12}"
        f"{'Test RelL2':>13}"
    )

    for result in summaries:
        print(
            f"{result['seed']:>6d}"
            f"{result['best_epoch']:>8d}"
            f"{result['best_validation_mse']:>12.6f}"
            f"{result['validation_relative_l2']:>12.6f}"
            f"{result['test_mse']:>12.6f}"
            f"{result['test_relative_l2']:>13.6f}"
        )

    test_errors = np.array(
        [
            result["test_relative_l2"]
            for result in summaries
        ],
        dtype=float,
    )

    finite_errors = test_errors[
        np.isfinite(test_errors)
    ]

    if len(finite_errors) > 1:
        print()
        print(
            "Individual-member test relative L2:",
            f"{finite_errors.mean():.6f}",
            "±",
            f"{finite_errors.std(ddof=1):.6f}",
        )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Load the formal data once.
    set_random_seed(2026)

    (
        train_loader,
        validation_loader,
        test_loader,
        statistics,
    ) = build_formal_dataloaders(
        train_path=TRAIN_PATH,
        validation_path=VALIDATION_PATH,
        test_path=TEST_PATH,
        batch_size=32,
    )

    summaries = []

    for seed in ENSEMBLE_SEEDS:
        member_directory = (
            ENSEMBLE_DIRECTORY
            / f"seed_{seed}"
        )

        member_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            member_directory
            / "fno_best.pt"
        )

        if checkpoint_path.exists():
            print()
            print(
                f"Seed {seed} already exists; "
                "skipping training."
            )

            summaries.append(
                load_member_summary(
                    checkpoint_path,
                    device,
                )
            )

            continue

        if seed == 2026:
            save_existing_member(
                destination_path=
                    checkpoint_path,
                device=device,
            )

            summaries.append(
                load_member_summary(
                    checkpoint_path,
                    device,
                )
            )

            continue

        result = train_member(
            seed=seed,
            train_loader=train_loader,
            validation_loader=
                validation_loader,
            test_loader=test_loader,
            statistics=statistics,
            device=device,
        )

        summaries.append(result)

    summaries = sorted(
        summaries,
        key=lambda item: item["seed"],
    )

    print_summary(summaries)

    summary_path = (
        ENSEMBLE_DIRECTORY
        / "member_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summaries,
            file,
            indent=2,
        )

    print()
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()