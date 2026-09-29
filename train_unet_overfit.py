from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from pytorch_dataset import (
    DATA_PATH,
    build_dataloaders,
)


OUTPUT_DIRECTORY = Path("outputs")
OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


class DoubleConvolution(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
    ):
        super().__init__()

        # GroupNorm is stable for small batch sizes.
        number_of_groups = min(
            8,
            output_channels,
        )

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                number_of_groups,
                output_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                number_of_groups,
                output_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(
        self,
        input_channels=2,
        output_channels=1,
        base_channels=32,
    ):
        super().__init__()

        self.encoder_1 = DoubleConvolution(
            input_channels,
            base_channels,
        )

        self.encoder_2 = DoubleConvolution(
            base_channels,
            base_channels * 2,
        )

        self.encoder_3 = DoubleConvolution(
            base_channels * 2,
            base_channels * 4,
        )

        self.bottleneck = DoubleConvolution(
            base_channels * 4,
            base_channels * 8,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.up_3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )

        self.decoder_3 = DoubleConvolution(
            base_channels * 8,
            base_channels * 4,
        )

        self.up_2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )

        self.decoder_2 = DoubleConvolution(
            base_channels * 4,
            base_channels * 2,
        )

        self.up_1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )

        self.decoder_1 = DoubleConvolution(
            base_channels * 2,
            base_channels,
        )

        self.output_layer = nn.Conv2d(
            base_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(self, x):
        encoder_1 = self.encoder_1(x)

        encoder_2 = self.encoder_2(
            self.pool(encoder_1)
        )

        encoder_3 = self.encoder_3(
            self.pool(encoder_2)
        )

        bottleneck = self.bottleneck(
            self.pool(encoder_3)
        )

        decoder_3 = self.up_3(bottleneck)

        decoder_3 = torch.cat(
            [decoder_3, encoder_3],
            dim=1,
        )

        decoder_3 = self.decoder_3(
            decoder_3
        )

        decoder_2 = self.up_2(decoder_3)

        decoder_2 = torch.cat(
            [decoder_2, encoder_2],
            dim=1,
        )

        decoder_2 = self.decoder_2(
            decoder_2
        )

        decoder_1 = self.up_1(decoder_2)

        decoder_1 = torch.cat(
            [decoder_1, encoder_1],
            dim=1,
        )

        decoder_1 = self.decoder_1(
            decoder_1
        )

        prediction = self.output_layer(
            decoder_1
        )

        return prediction


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def train_on_single_batch(
    model,
    model_input,
    target,
    device,
    number_of_steps=1000,
    learning_rate=1e-3,
):
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_function = nn.MSELoss()

    losses = []

    for step in range(1, number_of_steps + 1):
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

        losses.append(loss.item())

        if (
            step == 1
            or step % 100 == 0
            or step == number_of_steps
        ):
            print(
                f"Step {step:4d} | "
                f"Loss: {loss.item():.8f}"
            )

    return losses


def calculate_relative_l2(
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


def plot_loss(losses):
    plt.figure(figsize=(6, 4))

    plt.semilogy(
        np.arange(1, len(losses) + 1),
        losses,
    )

    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("Single-batch overfitting test")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "single_batch_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(f"Loss plot saved to: {output_path}")


def plot_predictions(
    model_input,
    target,
    prediction,
    statistics,
    number_of_examples=3,
):
    target_mean = statistics["target_mean"]
    target_std = statistics["target_std"]

    # Return predictions to physical log-permeability
    # units.
    target_physical = (
        target * target_std + target_mean
    )

    prediction_physical = (
        prediction * target_std + target_mean
    )

    model_input = (
        model_input.detach().cpu().numpy()
    )

    target_physical = (
        target_physical.detach().cpu().numpy()
    )

    prediction_physical = (
        prediction_physical.detach().cpu().numpy()
    )

    number_of_examples = min(
        number_of_examples,
        len(target_physical),
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

        true_field = target_physical[
            sample_index,
            0,
        ]

        predicted_field = prediction_physical[
            sample_index,
            0,
        ]

        absolute_error = np.abs(
            predicted_field - true_field
        )

        fields = [
            (
                observation,
                "Normalized observations",
                "coolwarm",
            ),
            (
                true_field,
                "True log-permeability",
                "viridis",
            ),
            (
                predicted_field,
                "Predicted log-permeability",
                "viridis",
            ),
            (
                absolute_error,
                "Absolute error",
                "magma",
            ),
        ]

        shared_min = min(
            true_field.min(),
            predicted_field.min(),
        )

        shared_max = max(
            true_field.max(),
            predicted_field.max(),
        )

        for column, (
            field,
            title,
            colour_map,
        ) in enumerate(fields):
            if column in (1, 2):
                image = axes[
                    sample_index,
                    column,
                ].imshow(
                    field,
                    origin="lower",
                    cmap=colour_map,
                    vmin=shared_min,
                    vmax=shared_max,
                )
            else:
                image = axes[
                    sample_index,
                    column,
                ].imshow(
                    field,
                    origin="lower",
                    cmap=colour_map,
                )

            axes[
                sample_index,
                column,
            ].set_title(title)

            fig.colorbar(
                image,
                ax=axes[
                    sample_index,
                    column,
                ],
                fraction=0.046,
            )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "single_batch_predictions.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print(
        f"Prediction plot saved to: "
        f"{output_path}"
    )


def main():
    torch.manual_seed(2026)
    np.random.seed(2026)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    (
        train_loader,
        _,
        _,
        statistics,
        _,
    ) = build_dataloaders(
        data_path=DATA_PATH,
        batch_size=8,
        split_seed=2026,
    )

    # Extract one fixed batch and repeatedly train
    # on exactly these eight samples.
    model_input, target = next(
        iter(train_loader)
    )

    model_input = model_input.to(device)
    target = target.to(device)

    model = UNet(
        input_channels=2,
        output_channels=1,
        base_channels=32,
    ).to(device)

    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}",
    )

    with torch.no_grad():
        initial_prediction = model(model_input)

    print(
        "Initial prediction shape:",
        initial_prediction.shape,
    )

    assert (
        initial_prediction.shape
        == target.shape
    )

    losses = train_on_single_batch(
        model=model,
        model_input=model_input,
        target=target,
        device=device,
        number_of_steps=1000,
        learning_rate=1e-3,
    )

    model.eval()

    with torch.no_grad():
        final_prediction = model(model_input)

        relative_errors = calculate_relative_l2(
            final_prediction,
            target,
        )

    print()
    print(
        "Initial loss:",
        f"{losses[0]:.8f}",
    )

    print(
        "Final loss:",
        f"{losses[-1]:.8f}",
    )

    print(
        "Mean relative L2 error:",
        f"{relative_errors.mean().item():.6f}",
    )

    checkpoint_path = (
        OUTPUT_DIRECTORY
        / "unet_single_batch.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "statistics": statistics,
            "final_loss": losses[-1],
        },
        checkpoint_path,
    )

    print(
        f"Checkpoint saved to: "
        f"{checkpoint_path}"
    )

    plot_loss(losses)

    plot_predictions(
        model_input=model_input,
        target=target,
        prediction=final_prediction,
        statistics=statistics,
        number_of_examples=3,
    )


if __name__ == "__main__":
    main()