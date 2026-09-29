from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


TRAIN_PATH = Path(
    "data/formal/darcy_train_2000.npz"
)

VALIDATION_PATH = Path(
    "data/formal/darcy_validation_300.npz"
)

TEST_PATH = Path(
    "data/formal/darcy_iid_test_300.npz"
)


def load_npz_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        result = {
            "log_permeability":
                data["log_permeability"].copy(),
            "observed_pressure":
                data["observed_pressure"].copy(),
            "sensor_mask":
                data["sensor_mask"].copy(),
            "pressure":
                data["pressure"].copy(),
        }

    return result


def calculate_training_statistics(
    training_data,
):
    log_permeability = training_data[
        "log_permeability"
    ]

    observed_pressure = training_data[
        "observed_pressure"
    ]

    sensor_mask = training_data[
        "sensor_mask"
    ].astype(bool)

    sensor_values = observed_pressure[
        :,
        sensor_mask,
    ]

    statistics = {
        "target_mean": float(
            log_permeability.mean()
        ),
        "target_std": float(
            log_permeability.std()
        ),
        "observation_mean": float(
            sensor_values.mean()
        ),
        "observation_std": float(
            sensor_values.std()
        ),
    }

    if statistics["target_std"] < 1e-8:
        raise ValueError(
            "Target standard deviation is too small."
        )

    if statistics["observation_std"] < 1e-8:
        raise ValueError(
            "Observation standard deviation is too small."
        )

    return statistics


class FormalDarcyDataset(Dataset):
    def __init__(
        self,
        data,
        statistics,
    ):
        self.log_permeability = data[
            "log_permeability"
        ].astype(
            np.float32,
            copy=False,
        )

        self.observed_pressure = data[
            "observed_pressure"
        ].astype(
            np.float32,
            copy=False,
        )

        self.sensor_mask = data[
            "sensor_mask"
        ].astype(
            np.float32,
            copy=False,
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
        return len(self.log_permeability)

    def __getitem__(self, index):
        observation = self.observed_pressure[
            index
        ]

        target = self.log_permeability[
            index
        ]

        normalized_observation = (
            observation - self.observation_mean
        ) / self.observation_std

        # Missing locations must remain zero.
        normalized_observation = (
            normalized_observation
            * self.sensor_mask
        )

        normalized_target = (
            target - self.target_mean
        ) / self.target_std

        model_input = np.stack(
            [
                normalized_observation,
                self.sensor_mask,
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        model_target = normalized_target[
            np.newaxis,
            :,
            :,
        ].astype(
            np.float32,
            copy=False,
        )

        return (
            torch.from_numpy(model_input),
            torch.from_numpy(model_target),
        )


def verify_dataset_compatibility(
    training_data,
    validation_data,
    test_data,
):
    datasets = {
        "training": training_data,
        "validation": validation_data,
        "test": test_data,
    }

    reference_shape = training_data[
        "log_permeability"
    ].shape[1:]

    reference_mask = training_data[
        "sensor_mask"
    ]

    for name, data in datasets.items():
        field_shape = data[
            "log_permeability"
        ].shape[1:]

        if field_shape != reference_shape:
            raise ValueError(
                f"{name} grid shape differs: "
                f"{field_shape}"
            )

        if not np.array_equal(
            data["sensor_mask"],
            reference_mask,
        ):
            raise ValueError(
                f"{name} sensor mask differs."
            )

        if (
            data["log_permeability"].shape
            != data["observed_pressure"].shape
        ):
            raise ValueError(
                f"{name} input and target "
                "shapes differ."
            )

        if not np.all(
            np.isfinite(
                data["log_permeability"]
            )
        ):
            raise ValueError(
                f"{name} targets contain "
                "invalid values."
            )

        if not np.all(
            np.isfinite(
                data["observed_pressure"]
            )
        ):
            raise ValueError(
                f"{name} observations contain "
                "invalid values."
            )

    print("Dataset compatibility check passed.")


def build_formal_dataloaders(
    train_path=TRAIN_PATH,
    validation_path=VALIDATION_PATH,
    test_path=TEST_PATH,
    batch_size=32,
):
    print("Loading formal datasets...")

    training_data = load_npz_dataset(
        train_path
    )

    validation_data = load_npz_dataset(
        validation_path
    )

    test_data = load_npz_dataset(
        test_path
    )

    verify_dataset_compatibility(
        training_data,
        validation_data,
        test_data,
    )

    statistics = (
        calculate_training_statistics(
            training_data
        )
    )

    train_dataset = FormalDarcyDataset(
        training_data,
        statistics,
    )

    validation_dataset = FormalDarcyDataset(
        validation_data,
        statistics,
    )

    test_dataset = FormalDarcyDataset(
        test_data,
        statistics,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        statistics,
    )


def validate_batch(data_loader):
    model_input, target = next(
        iter(data_loader)
    )

    print(
        "Input shape:",
        model_input.shape,
    )

    print(
        "Target shape:",
        target.shape,
    )

    assert model_input.shape[1] == 2
    assert target.shape[1] == 1

    mask = model_input[:, 1]
    observations = model_input[:, 0]

    assert torch.all(
        torch.isfinite(model_input)
    )

    assert torch.all(
        torch.isfinite(target)
    )

    assert torch.all(
        mask.sum(dim=(1, 2)) == 64
    )

    assert torch.all(
        observations[mask == 0] == 0
    )

    print("Formal batch validation passed.")


if __name__ == "__main__":
    (
        train_loader,
        validation_loader,
        test_loader,
        statistics,
    ) = build_formal_dataloaders(
        batch_size=32
    )

    print()
    print("Dataset sizes")
    print("-------------")
    print("Training:", len(train_loader.dataset))
    print(
        "Validation:",
        len(validation_loader.dataset),
    )
    print("Test:", len(test_loader.dataset))

    print()
    print("Training statistics")
    print("-------------------")

    for name, value in statistics.items():
        print(f"{name}: {value:.6f}")

    print()
    validate_batch(train_loader)