from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DATA_PATH = Path(
    "data/darcy_sparse_64sensors_noise001.npz"
)


def create_splits(
    number_of_samples,
    train_fraction=0.70,
    validation_fraction=0.15,
    seed=2026,
):
    """
    Create reproducible random train, validation,
    and test indices.
    """

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "Train and validation fractions must "
            "sum to less than one."
        )

    rng = np.random.default_rng(seed)

    indices = rng.permutation(number_of_samples)

    train_size = int(
        train_fraction * number_of_samples
    )

    validation_size = int(
        validation_fraction * number_of_samples
    )

    train_indices = indices[:train_size]

    validation_indices = indices[
        train_size:
        train_size + validation_size
    ]

    test_indices = indices[
        train_size + validation_size:
    ]

    return (
        train_indices,
        validation_indices,
        test_indices,
    )


def calculate_normalization_statistics(
    log_permeability,
    observed_pressure,
    sensor_mask,
    train_indices,
):
    """
    Calculate all normalization statistics using
    training data only.

    This prevents information leakage from validation
    and test data.
    """

    training_targets = log_permeability[
        train_indices
    ]

    target_mean = training_targets.mean()
    target_std = training_targets.std()

    boolean_mask = sensor_mask.astype(bool)

    training_observations = observed_pressure[
        train_indices
    ]

    # Use only actual sensor values. Missing zeros must
    # not influence the observation statistics.
    sensor_values = training_observations[
        :,
        boolean_mask,
    ]

    observation_mean = sensor_values.mean()
    observation_std = sensor_values.std()

    if target_std < 1e-8:
        raise ValueError(
            "Target standard deviation is too small."
        )

    if observation_std < 1e-8:
        raise ValueError(
            "Observation standard deviation is too small."
        )

    statistics = {
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "observation_mean": float(
            observation_mean
        ),
        "observation_std": float(
            observation_std
        ),
    }

    return statistics


class DarcyInverseDataset(Dataset):
    """
    Input:
        channel 0: normalized sparse pressure
        channel 1: sensor mask

    Target:
        normalized log-permeability
    """

    def __init__(
        self,
        log_permeability,
        observed_pressure,
        sensor_mask,
        indices,
        statistics,
    ):
        self.log_permeability = (
            log_permeability
        )

        self.observed_pressure = (
            observed_pressure
        )

        self.sensor_mask = sensor_mask.astype(
            np.float32
        )

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.target_mean = statistics[
            "target_mean"
        ]

        self.target_std = statistics[
            "target_std"
        ]

        self.observation_mean = statistics[
            "observation_mean"
        ]

        self.observation_std = statistics[
            "observation_std"
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        sample_index = self.indices[item]

        observation = self.observed_pressure[
            sample_index
        ].astype(np.float32)

        target = self.log_permeability[
            sample_index
        ].astype(np.float32)

        # Normalize observed values.
        normalized_observation = (
            observation - self.observation_mean
        ) / self.observation_std

        # Ensure missing positions remain zero after
        # normalization.
        normalized_observation *= self.sensor_mask

        normalized_target = (
            target - self.target_mean
        ) / self.target_std

        model_input = np.stack(
            [
                normalized_observation,
                self.sensor_mask,
            ],
            axis=0,
        )

        model_target = normalized_target[
            np.newaxis,
            :,
            :,
        ]

        return (
            torch.from_numpy(model_input),
            torch.from_numpy(model_target),
        )


def build_dataloaders(
    data_path=DATA_PATH,
    batch_size=8,
    split_seed=2026,
):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find dataset: "
            f"{data_path.resolve()}"
        )

    with np.load(
        data_path,
        allow_pickle=False,
    ) as data:
        log_permeability = data[
            "log_permeability"
        ].astype(np.float32)

        observed_pressure = data[
            "observed_pressure"
        ].astype(np.float32)

        sensor_mask = data[
            "sensor_mask"
        ].astype(np.float32)

    number_of_samples = len(log_permeability)

    (
        train_indices,
        validation_indices,
        test_indices,
    ) = create_splits(
        number_of_samples=number_of_samples,
        train_fraction=0.70,
        validation_fraction=0.15,
        seed=split_seed,
    )

    statistics = (
        calculate_normalization_statistics(
            log_permeability=log_permeability,
            observed_pressure=observed_pressure,
            sensor_mask=sensor_mask,
            train_indices=train_indices,
        )
    )

    train_dataset = DarcyInverseDataset(
        log_permeability=log_permeability,
        observed_pressure=observed_pressure,
        sensor_mask=sensor_mask,
        indices=train_indices,
        statistics=statistics,
    )

    validation_dataset = DarcyInverseDataset(
        log_permeability=log_permeability,
        observed_pressure=observed_pressure,
        sensor_mask=sensor_mask,
        indices=validation_indices,
        statistics=statistics,
    )

    test_dataset = DarcyInverseDataset(
        log_permeability=log_permeability,
        observed_pressure=observed_pressure,
        sensor_mask=sensor_mask,
        indices=test_indices,
        statistics=statistics,
    )

    # num_workers=0 is the safest initial setting,
    # particularly on Windows.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }

    return (
        train_loader,
        validation_loader,
        test_loader,
        statistics,
        split_indices,
    )


def validate_batch(
    data_loader,
    sensor_count=64,
):
    model_input, target = next(
        iter(data_loader)
    )

    print("Batch input shape:", model_input.shape)
    print("Batch target shape:", target.shape)

    print("Input dtype:", model_input.dtype)
    print("Target dtype:", target.dtype)

    assert model_input.ndim == 4
    assert target.ndim == 4

    assert model_input.shape[1] == 2
    assert target.shape[1] == 1

    assert torch.all(
        torch.isfinite(model_input)
    )

    assert torch.all(
        torch.isfinite(target)
    )

    pressure_channel = model_input[:, 0]
    mask_channel = model_input[:, 1]

    unique_mask_values = torch.unique(
        mask_channel
    )

    print(
        "Unique mask values:",
        unique_mask_values.tolist(),
    )

    assert set(
        unique_mask_values.tolist()
    ).issubset({0.0, 1.0})

    sensor_counts = mask_channel.sum(
        dim=(1, 2)
    )

    print(
        "Sensors in each batch sample:",
        sensor_counts.tolist(),
    )

    assert torch.all(
        sensor_counts == sensor_count
    )

    # Missing pressure positions should remain zero.
    missing_values = pressure_channel[
        mask_channel == 0
    ]

    maximum_missing_value = (
        missing_values.abs().max().item()
    )

    print(
        "Maximum absolute value at missing positions:",
        maximum_missing_value,
    )

    assert maximum_missing_value == 0.0

    print("Batch validation passed.")


def save_split_information(
    output_path,
    statistics,
    split_indices,
):
    np.savez(
        output_path,
        train_indices=split_indices["train"],
        validation_indices=split_indices[
            "validation"
        ],
        test_indices=split_indices["test"],
        target_mean=np.float32(
            statistics["target_mean"]
        ),
        target_std=np.float32(
            statistics["target_std"]
        ),
        observation_mean=np.float32(
            statistics["observation_mean"]
        ),
        observation_std=np.float32(
            statistics["observation_std"]
        ),
    )

    print(
        f"Split information saved to: {output_path}"
    )


if __name__ == "__main__":
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

    print("Dataset sizes")
    print("-------------")
    print(
        "Training:",
        len(train_loader.dataset),
    )
    print(
        "Validation:",
        len(validation_loader.dataset),
    )
    print(
        "Test:",
        len(test_loader.dataset),
    )

    print()
    print("Normalization statistics")
    print("------------------------")

    for name, value in statistics.items():
        print(f"{name}: {value:.6f}")

    print()
    validate_batch(train_loader)

    save_split_information(
        output_path="data/debug_split_and_stats.npz",
        statistics=statistics,
        split_indices=split_indices,
    )