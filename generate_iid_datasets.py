from pathlib import Path
from time import perf_counter

import numpy as np

from darcy_solver import solve_darcy

from generate_dataset import (
    generate_log_permeability,
)

from create_sparse_observations import (
    create_regular_sensor_mask,
)


OUTPUT_DIRECTORY = Path("data/formal")
OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def generate_split(
    output_path,
    split_name,
    number_of_samples,
    nx,
    ny,
    correlation_length,
    field_seed_start,
    observation_seed,
    noise_level=0.01,
    sensors_per_axis=8,
):
    print()
    print("=" * 60)
    print(f"Generating split: {split_name}")
    print(f"Number of samples: {number_of_samples}")
    print(
        f"Correlation length: "
        f"{correlation_length}"
    )
    print(f"Field seed start: {field_seed_start}")
    print(f"Observation seed: {observation_seed}")
    print("=" * 60)

    log_permeability_data = np.empty(
        (number_of_samples, ny, nx),
        dtype=np.float32,
    )

    pressure_data = np.empty(
        (number_of_samples, ny, nx),
        dtype=np.float32,
    )

    observed_pressure_data = np.zeros(
        (number_of_samples, ny, nx),
        dtype=np.float32,
    )

    sensor_mask, sensor_coordinates = (
        create_regular_sensor_mask(
            ny=ny,
            nx=nx,
            sensors_per_axis=sensors_per_axis,
        )
    )

    boolean_mask = sensor_mask.astype(bool)

    number_of_sensors = int(
        sensor_mask.sum()
    )

    observation_rng = np.random.default_rng(
        observation_seed
    )

    start_time = perf_counter()

    for sample_index in range(
        number_of_samples
    ):
        field_seed = (
            field_seed_start + sample_index
        )

        (
            log_permeability,
            permeability,
        ) = generate_log_permeability(
            nx=nx,
            ny=ny,
            correlation_length=correlation_length,
            log_mean=0.0,
            log_std=1.0,
            seed=field_seed,
        )

        pressure = solve_darcy(
            permeability,
            verbose=False,
        )

        pressure_range = (
            pressure.max() - pressure.min()
        )

        noise_standard_deviation = (
            noise_level * pressure_range
        )

        noise = observation_rng.normal(
            loc=0.0,
            scale=noise_standard_deviation,
            size=number_of_sensors,
        )

        noisy_sensor_values = (
            pressure[boolean_mask] + noise
        )

        noisy_sensor_values = np.clip(
            noisy_sensor_values,
            0.0,
            1.0,
        )

        observed_pressure = np.zeros(
            (ny, nx),
            dtype=np.float32,
        )

        observed_pressure[boolean_mask] = (
            noisy_sensor_values
        )

        log_permeability_data[
            sample_index
        ] = log_permeability

        pressure_data[
            sample_index
        ] = pressure

        observed_pressure_data[
            sample_index
        ] = observed_pressure

        completed = sample_index + 1

        if (
            completed == 1
            or completed % 100 == 0
            or completed == number_of_samples
        ):
            elapsed = (
                perf_counter() - start_time
            )

            samples_per_second = (
                completed / elapsed
            )

            remaining = (
                number_of_samples - completed
            )

            estimated_remaining = (
                remaining / samples_per_second
                if samples_per_second > 0
                else float("inf")
            )

            print(
                f"{split_name}: "
                f"{completed}/{number_of_samples} | "
                f"{samples_per_second:.2f} samples/s | "
                f"approximately "
                f"{estimated_remaining / 60:.1f} min remaining"
            )

    validate_split(
        split_name=split_name,
        log_permeability=log_permeability_data,
        pressure=pressure_data,
        observed_pressure=observed_pressure_data,
        sensor_mask=sensor_mask,
    )

    np.savez_compressed(
        output_path,
        log_permeability=log_permeability_data,
        pressure=pressure_data,
        observed_pressure=observed_pressure_data,
        sensor_mask=sensor_mask.astype(
            np.float32
        ),
        sensor_coordinates=sensor_coordinates,
        correlation_length=np.float32(
            correlation_length
        ),
        noise_level=np.float32(noise_level),
        field_seed_start=np.int64(
            field_seed_start
        ),
        observation_seed=np.int64(
            observation_seed
        ),
        number_of_samples=np.int64(
            number_of_samples
        ),
        nx=np.int64(nx),
        ny=np.int64(ny),
    )

    elapsed = perf_counter() - start_time

    file_size_mb = (
        output_path.stat().st_size
        / (1024**2)
    )

    print()
    print(f"Saved: {output_path}")
    print(
        f"Generation time: "
        f"{elapsed / 60:.2f} minutes"
    )
    print(
        f"Compressed file size: "
        f"{file_size_mb:.2f} MB"
    )


def validate_split(
    split_name,
    log_permeability,
    pressure,
    observed_pressure,
    sensor_mask,
):
    print()
    print(f"Validating {split_name}")

    assert (
        log_permeability.shape
        == pressure.shape
        == observed_pressure.shape
    )

    assert np.all(
        np.isfinite(log_permeability)
    )

    assert np.all(
        np.isfinite(pressure)
    )

    assert np.all(
        np.isfinite(observed_pressure)
    )

    assert int(sensor_mask.sum()) == 64

    boolean_mask = sensor_mask.astype(bool)

    assert np.all(
        observed_pressure[:, ~boolean_mask]
        == 0.0
    )

    tolerance = 1e-5

    assert pressure.min() >= -tolerance
    assert pressure.max() <= 1.0 + tolerance

    assert np.max(
        np.abs(pressure[:, :, 0] - 1.0)
    ) < tolerance

    assert np.max(
        np.abs(pressure[:, :, -1])
    ) < tolerance

    sample_variation = np.mean(
        np.std(
            log_permeability,
            axis=0,
        )
    )

    assert sample_variation > 1e-6

    print(
        "log-permeability range:",
        f"[{log_permeability.min():.3f}, "
        f"{log_permeability.max():.3f}]",
    )

    print(
        "pressure range:",
        f"[{pressure.min():.6f}, "
        f"{pressure.max():.6f}]",
    )

    print(
        "average cross-sample variation:",
        f"{sample_variation:.6f}",
    )

    print(f"{split_name} validation passed.")


def check_seed_ranges_do_not_overlap(
    split_configurations,
):
    used_seeds = set()

    for configuration in split_configurations:
        start = configuration[
            "field_seed_start"
        ]

        count = configuration[
            "number_of_samples"
        ]

        seeds = set(
            range(start, start + count)
        )

        overlap = used_seeds.intersection(
            seeds
        )

        if overlap:
            raise ValueError(
                "Field seed ranges overlap."
            )

        used_seeds.update(seeds)

    print("Field seed ranges do not overlap.")


def main():
    common_configuration = {
        "nx": 64,
        "ny": 64,
        "correlation_length": 0.15,
        "noise_level": 0.01,
        "sensors_per_axis": 8,
    }

    split_configurations = [
        {
            "split_name": "train",
            "number_of_samples": 2000,
            "field_seed_start": 10000,
            "observation_seed": 110000,
            "output_path":
                OUTPUT_DIRECTORY
                / "darcy_train_2000.npz",
        },
        {
            "split_name": "validation",
            "number_of_samples": 300,
            "field_seed_start": 20000,
            "observation_seed": 120000,
            "output_path":
                OUTPUT_DIRECTORY
                / "darcy_validation_300.npz",
        },
        {
            "split_name": "iid_test",
            "number_of_samples": 300,
            "field_seed_start": 30000,
            "observation_seed": 130000,
            "output_path":
                OUTPUT_DIRECTORY
                / "darcy_iid_test_300.npz",
        },
    ]

    check_seed_ranges_do_not_overlap(
        split_configurations
    )

    for configuration in split_configurations:
        generate_split(
            **configuration,
            **common_configuration,
        )


if __name__ == "__main__":
    main()