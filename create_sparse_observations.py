from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


INPUT_PATH = Path("data/darcy_test_100.npz")
OUTPUT_PATH = Path(
    "data/darcy_sparse_64sensors_noise001.npz"
)


def create_regular_sensor_mask(
    ny,
    nx,
    sensors_per_axis=8,
):
    """
    Create a regular sensor grid.

    An 8 x 8 arrangement gives 64 sensors.
    Sensors are placed inside the domain rather than
    directly on the fixed-pressure boundaries.
    """

    if sensors_per_axis > min(nx - 2, ny - 2):
        raise ValueError(
            "Too many sensors for the grid."
        )

    x_indices = np.linspace(
        1,
        nx - 2,
        sensors_per_axis,
        dtype=int,
    )

    y_indices = np.linspace(
        1,
        ny - 2,
        sensors_per_axis,
        dtype=int,
    )

    mask = np.zeros(
        (ny, nx),
        dtype=np.float32,
    )

    for j in y_indices:
        for i in x_indices:
            mask[j, i] = 1.0

    coordinates = np.array(
        [
            [j, i]
            for j in y_indices
            for i in x_indices
        ],
        dtype=np.int64,
    )

    return mask, coordinates


def create_sparse_observations(
    pressure,
    sensor_mask,
    noise_level=0.01,
    seed=2026,
):
    """
    Add Gaussian noise only at sensor locations.

    noise_level=0.01 means the noise standard deviation
    is 1% of each sample's pressure range.
    """

    rng = np.random.default_rng(seed)

    number_of_samples, ny, nx = pressure.shape

    if sensor_mask.shape != (ny, nx):
        raise ValueError(
            "Sensor mask and pressure dimensions differ."
        )

    sparse_observations = np.zeros_like(
        pressure,
        dtype=np.float32,
    )

    clean_sparse_observations = np.zeros_like(
        pressure,
        dtype=np.float32,
    )

    noise_standard_deviations = np.zeros(
        number_of_samples,
        dtype=np.float32,
    )

    boolean_mask = sensor_mask.astype(bool)

    for sample_index in range(number_of_samples):
        sample_pressure = pressure[sample_index]

        pressure_range = (
            sample_pressure.max()
            - sample_pressure.min()
        )

        noise_std = noise_level * pressure_range
        noise_standard_deviations[sample_index] = (
            noise_std
        )

        sensor_noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=int(sensor_mask.sum()),
        )

        clean_values = sample_pressure[boolean_mask]
        noisy_values = clean_values + sensor_noise

        # Pressure is theoretically bounded by 0 and 1
        # for the current boundary conditions.
        noisy_values = np.clip(
            noisy_values,
            0.0,
            1.0,
        )

        clean_sparse_observations[
            sample_index,
            boolean_mask,
        ] = clean_values

        sparse_observations[
            sample_index,
            boolean_mask,
        ] = noisy_values

    return (
        clean_sparse_observations,
        sparse_observations,
        noise_standard_deviations,
    )


def validate_sparse_data(
    pressure,
    clean_sparse,
    noisy_sparse,
    sensor_mask,
):
    boolean_mask = sensor_mask.astype(bool)
    missing_mask = ~boolean_mask

    number_of_sensors = int(sensor_mask.sum())

    print(f"Number of sensors: {number_of_sensors}")
    print(
        "Sensor fraction:",
        f"{100 * sensor_mask.mean():.4f}%",
    )

    assert number_of_sensors == 64

    # No values should be stored outside sensors
    assert np.all(
        clean_sparse[:, missing_mask] == 0.0
    )

    assert np.all(
        noisy_sparse[:, missing_mask] == 0.0
    )

    print("✓ Missing positions are zero")

    # Clean sensor values must equal the full pressure
    clean_difference = np.max(
        np.abs(
            clean_sparse[:, boolean_mask]
            - pressure[:, boolean_mask]
        )
    )

    print(
        "Maximum clean-observation error:",
        f"{clean_difference:.3e}",
    )

    assert clean_difference < 1e-7

    print("✓ Clean sensor values match pressure")

    # Confirm that noise was actually added
    noise = (
        noisy_sparse[:, boolean_mask]
        - clean_sparse[:, boolean_mask]
    )

    print(
        "Observed noise mean:",
        f"{noise.mean():.6f}",
    )

    print(
        "Observed noise standard deviation:",
        f"{noise.std():.6f}",
    )

    assert noise.std() > 0.0

    print("✓ Noise was added")
    print("All sparse-observation checks passed.")


def plot_example(
    pressure,
    clean_sparse,
    noisy_sparse,
    sensor_mask,
    sample_index=0,
):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    plots = [
        (
            pressure[sample_index],
            "Complete pressure",
            "coolwarm",
        ),
        (
            sensor_mask,
            "Sensor mask",
            "gray_r",
        ),
        (
            clean_sparse[sample_index],
            "Clean sparse observations",
            "coolwarm",
        ),
        (
            noisy_sparse[sample_index],
            "Noisy sparse observations",
            "coolwarm",
        ),
    ]

    for axis, (field, title, colour_map) in zip(
        axes,
        plots,
    ):
        image = axis.imshow(
            field,
            origin="lower",
            cmap=colour_map,
            vmin=0.0,
            vmax=1.0,
        )

        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

        fig.colorbar(
            image,
            ax=axis,
            fraction=0.046,
        )

    plt.tight_layout()
    plt.show()


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH.resolve()}"
        )

    with np.load(
        INPUT_PATH,
        allow_pickle=False,
    ) as data:
        log_permeability = data[
            "log_permeability"
        ].copy()

        pressure = data["pressure"].copy()

        correlation_length = data[
            "correlation_length"
        ].item()

        base_seed = data["base_seed"].item()

    number_of_samples, ny, nx = pressure.shape

    sensor_mask, sensor_coordinates = (
        create_regular_sensor_mask(
            ny=ny,
            nx=nx,
            sensors_per_axis=8,
        )
    )

    (
        clean_sparse,
        noisy_sparse,
        noise_std,
    ) = create_sparse_observations(
        pressure=pressure,
        sensor_mask=sensor_mask,
        noise_level=0.01,
        seed=2026,
    )

    validate_sparse_data(
        pressure=pressure,
        clean_sparse=clean_sparse,
        noisy_sparse=noisy_sparse,
        sensor_mask=sensor_mask,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        log_permeability=log_permeability,
        pressure=pressure,
        clean_sparse_pressure=clean_sparse,
        observed_pressure=noisy_sparse,
        sensor_mask=sensor_mask,
        sensor_coordinates=sensor_coordinates,
        noise_standard_deviation=noise_std,
        noise_level=np.float32(0.01),
        correlation_length=np.float32(
            correlation_length
        ),
        base_seed=np.int64(base_seed),
        observation_seed=np.int64(2026),
    )

    print()
    print(f"Saved dataset to: {OUTPUT_PATH}")
    print(
        "log_permeability:",
        log_permeability.shape,
    )
    print("pressure:", pressure.shape)
    print("observed_pressure:", noisy_sparse.shape)
    print("sensor_mask:", sensor_mask.shape)
    print(
        "sensor_coordinates:",
        sensor_coordinates.shape,
    )

    plot_example(
        pressure=pressure,
        clean_sparse=clean_sparse,
        noisy_sparse=noisy_sparse,
        sensor_mask=sensor_mask,
        sample_index=0,
    )


if __name__ == "__main__":
    main()