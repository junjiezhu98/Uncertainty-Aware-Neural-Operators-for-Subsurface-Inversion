from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from pytorch_dataset import (
    DATA_PATH,
    build_dataloaders,
)

from train_unet_overfit import (
    UNet,
    count_parameters,
)


OUTPUT_DIRECTORY = Path("outputs/full_training")
OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def relative_l2_per_sample(
    prediction,
    target,
):
    numerator = torch.linalg.vector_norm(
        prediction - target,
        dim=(1, 2, 3),
    )

    denominator = torch.linalg.vector_norm(
        target,
        dim=(1, 2, 3),
    )

    return numerator / (
        denominator + 1e-8
    )


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    loss_function,
    device,
):
    model.train()

    total_loss = 0.0
    number_of_samples = 0

    for model_input, target in data_loader:
        model_input = model_input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        prediction = model(model_input)

        loss = loss_function(
            prediction,
            target,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = model_input.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        number_of_samples += batch_size

    return total_loss / number_of_samples


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    loss_function,
    device,
):
    model.eval()

    total_loss = 0.0
    all_relative_errors = []
    number_of_samples = 0

    for model_input, target in data_loader:
        model_input = model_input.to(device)
        target = target.to(device)

        prediction = model(model_input)

        loss = loss_function(
            prediction,
            target,
        )

        batch_size = model_input.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        number_of_samples += batch_size

        relative_errors = relative_l2_per_sample(
            prediction,
            target,
        )

        all_relative_errors.append(
            relative_errors.cpu()
        )

    mean_loss = total_loss / number_of_samples

    all_relative_errors = torch.cat(
        all_relative_errors
    )

    metrics = {
        "mse": mean_loss,
        "relative_l2_mean":
            all_relative_errors.mean().item(),
        "relative_l2_std":
            all_relative_errors.std().item(),
        "relative_l2_median":
            all_relative_errors.median().item(),
    }

    return metrics


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    number_of_epochs=200,
    learning_rate=1e-3,
    patience=25,
):
    loss_function = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )
    )

    training_losses = []
    validation_losses = []

    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        number_of_epochs + 1,
    ):
        training_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        validation_metrics = evaluate(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        validation_loss = validation_metrics["mse"]

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        scheduler.step(validation_loss)

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        improved = (
            validation_loss
            < best_validation_loss - 1e-6
        )

        if improved:
            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            best_model_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):
            print(
                f"Epoch {epoch:3d} | "
                f"Train MSE: "
                f"{training_loss:.6f} | "
                f"Validation MSE: "
                f"{validation_loss:.6f} | "
                f"Validation relative L2: "
                f"{validation_metrics['relative_l2_mean']:.4f} | "
                f"LR: {current_learning_rate:.2e}"
            )

        if epochs_without_improvement >= patience:
            print()
            print(
                "Early stopping at epoch "
                f"{epoch}."
            )
            break

    if best_model_state is None:
        raise RuntimeError(
            "Training did not produce a valid model."
        )

    model.load_state_dict(best_model_state)

    history = {
        "training_loss": training_losses,
        "validation_loss": validation_losses,
        "best_epoch": best_epoch,
        "best_validation_loss":
            best_validation_loss,
    }

    return model, history


def plot_training_history(history):
    epochs = np.arange(
        1,
        len(history["training_loss"]) + 1,
    )

    plt.figure(figsize=(7, 4.5))

    plt.semilogy(
        epochs,
        history["training_loss"],
        label="Training",
    )

    plt.semilogy(
        epochs,
        history["validation_loss"],
        label="Validation",
    )

    plt.axvline(
        history["best_epoch"],
        color="black",
        linestyle="--",
        alpha=0.6,
        label=(
            f"Best epoch "
            f"{history['best_epoch']}"
        ),
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("U-Net training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "training_history.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(
        f"Training plot saved to: "
        f"{output_path}"
    )


@torch.no_grad()
def plot_test_predictions(
    model,
    test_loader,
    statistics,
    device,
    number_of_examples=4,
):
    model.eval()

    model_input, target = next(
        iter(test_loader)
    )

    model_input = model_input.to(device)
    target = target.to(device)

    prediction = model(model_input)

    target_mean = statistics["target_mean"]
    target_std = statistics["target_std"]

    true_log_permeability = (
        target * target_std + target_mean
    )

    predicted_log_permeability = (
        prediction * target_std + target_mean
    )

    model_input = (
        model_input.cpu().numpy()
    )

    true_log_permeability = (
        true_log_permeability.cpu().numpy()
    )

    predicted_log_permeability = (
        predicted_log_permeability
        .cpu()
        .numpy()
    )

    number_of_examples = min(
        number_of_examples,
        len(model_input),
    )

    fig, axes = plt.subplots(
        number_of_examples,
        4,
        figsize=(14, 3.2 * number_of_examples),
        squeeze=False,
    )

    for sample_index in range(
        number_of_examples
    ):
        observation = model_input[
            sample_index,
            0,
        ]

        true_field = true_log_permeability[
            sample_index,
            0,
        ]

        predicted_field = (
            predicted_log_permeability[
                sample_index,
                0,
            ]
        )

        absolute_error = np.abs(
            predicted_field - true_field
        )

        shared_min = min(
            true_field.min(),
            predicted_field.min(),
        )

        shared_max = max(
            true_field.max(),
            predicted_field.max(),
        )

        image = axes[
            sample_index,
            0,
        ].imshow(
            observation,
            origin="lower",
            cmap="coolwarm",
        )

        axes[
            sample_index,
            0,
        ].set_title(
            "Sparse observations"
        )

        fig.colorbar(
            image,
            ax=axes[sample_index, 0],
            fraction=0.046,
        )

        image = axes[
            sample_index,
            1,
        ].imshow(
            true_field,
            origin="lower",
            cmap="viridis",
            vmin=shared_min,
            vmax=shared_max,
        )

        axes[
            sample_index,
            1,
        ].set_title(
            "True log-permeability"
        )

        fig.colorbar(
            image,
            ax=axes[sample_index, 1],
            fraction=0.046,
        )

        image = axes[
            sample_index,
            2,
        ].imshow(
            predicted_field,
            origin="lower",
            cmap="viridis",
            vmin=shared_min,
            vmax=shared_max,
        )

        axes[
            sample_index,
            2,
        ].set_title(
            "Predicted log-permeability"
        )

        fig.colorbar(
            image,
            ax=axes[sample_index, 2],
            fraction=0.046,
        )

        image = axes[
            sample_index,
            3,
        ].imshow(
            absolute_error,
            origin="lower",
            cmap="magma",
        )

        axes[
            sample_index,
            3,
        ].set_title("Absolute error")

        fig.colorbar(
            image,
            ax=axes[sample_index, 3],
            fraction=0.046,
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "test_predictions.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(
        f"Test predictions saved to: "
        f"{output_path}"
    )


def main():
    torch.manual_seed(2026)
    np.random.seed(2026)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(2026)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    (
        train_loader,
        validation_loader,
        test_loader,
        statistics,
        split_indices,
    ) = build_dataloaders(
        data_path=DATA_PATH,
        batch_size=8,
        split_seed=2026,
    )

    model = UNet(
        input_channels=2,
        output_channels=1,
        base_channels=32,
    ).to(device)

    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}",
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        number_of_epochs=200,
        learning_rate=1e-3,
        patience=25,
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

    print()
    print("Best model")
    print("----------")
    print(
        "Best epoch:",
        history["best_epoch"],
    )
    print(
        "Best validation MSE:",
        f"{history['best_validation_loss']:.6f}",
    )

    print()
    print("Validation metrics")
    print("------------------")

    for name, value in validation_metrics.items():
        print(f"{name}: {value:.6f}")

    print()
    print("Test metrics")
    print("------------")

    for name, value in test_metrics.items():
        print(f"{name}: {value:.6f}")

    checkpoint_path = (
        OUTPUT_DIRECTORY
        / "unet_best.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "statistics": statistics,
            "split_indices": split_indices,
            "history": history,
            "validation_metrics":
                validation_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )

    print()
    print(
        f"Best checkpoint saved to: "
        f"{checkpoint_path}"
    )

    plot_training_history(history)

    plot_test_predictions(
        model=model,
        test_loader=test_loader,
        statistics=statistics,
        device=device,
        number_of_examples=4,
    )


if __name__ == "__main__":
    main()