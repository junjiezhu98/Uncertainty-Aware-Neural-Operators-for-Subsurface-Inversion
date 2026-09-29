from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from darcy_solver import solve_darcy


def generate_log_permeability(
    nx=64,
    ny=64,
    correlation_length=0.15,
    log_mean=0.0,
    log_std=1.0,
    seed=None,
):
    """python generate_dataset.py
    Generate a smooth random log-permeability field.

    correlation_length:
        Approximate spatial correlation length relative
        to the unit-square domain.

    Returns:
        log_permeability: shape (ny, nx)
        permeability:     shape (ny, nx)
    """

    rng = np.random.default_rng(seed)

    # Independent Gaussian noise
    white_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(ny, nx),
    )

    # Convert physical correlation length to grid units
    sigma_x = correlation_length * (nx - 1)
    sigma_y = correlation_length * (ny - 1)

    # Smooth the white noise to create spatial correlation
    smooth_field = gaussian_filter(
        white_noise,
        sigma=(sigma_y, sigma_x),
        mode="reflect",
    )

    # Standardize the field
    smooth_field -= smooth_field.mean()

    field_std = smooth_field.std()

    if field_std < 1e-12:
        raise ValueError(
            "Generated field has almost zero variance."
        )

    smooth_field /= field_std

    # Construct log-permeability
    log_permeability = (
        log_mean + log_std * smooth_field
    )

    # Avoid extreme permeability values
    log_permeability = np.clip(
        log_permeability,
        -3.0,
        3.0,
    )

    # Permeability must be positive
    permeability = np.exp(log_permeability)

    return log_permeability, permeability


def generate_dataset(
    number_of_samples=100,
    nx=64,
    ny=64,
    correlation_length=0.15,
    base_seed=1234,
):
    """
    Generate permeability-pressure pairs.
    """

    log_permeability_data = np.zeros(
        (number_of_samples, ny, nx),
        dtype=np.float32,
    )

    pressure_data = np.zeros(
        (number_of_samples, ny, nx),
        dtype=np.float32,
    )

    residuals = np.zeros(
        number_of_samples,
        dtype=np.float64,
    )

    for sample_index in range(number_of_samples):
        seed = base_seed + sample_index

        log_a, permeability = (
            generate_log_permeability(
                nx=nx,
                ny=ny,
                correlation_length=correlation_length,
                log_mean=0.0,
                log_std=1.0,
                seed=seed,
            )
        )

        # The previous solve_darcy function returns pressure.
        pressure = solve_darcy(permeability, verbose=False)

        log_permeability_data[sample_index] = log_a
        pressure_data[sample_index] = pressure

        if (
            not np.all(np.isfinite(log_a))
            or not np.all(np.isfinite(pressure))
        ):
            raise RuntimeError(
                f"Invalid values in sample {sample_index}"
            )

        if pressure.min() < -1e-6:
            raise RuntimeError(
                f"Pressure below zero in sample {sample_index}"
            )

        if pressure.max() > 1.0 + 1e-6:
            raise RuntimeError(
                f"Pressure above one in sample {sample_index}"
            )

        if (sample_index + 1) % 10 == 0:
            print(
                f"Generated {sample_index + 1}"
                f"/{number_of_samples} samples"
            )

    return log_permeability_data, pressure_data


def save_dataset(
    output_path,
    log_permeability,
    pressure,
    correlation_length,
    base_seed,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        log_permeability=log_permeability,
        pressure=pressure,
        correlation_length=np.float32(
            correlation_length
        ),
        base_seed=np.int64(base_seed),
    )

    print(f"Dataset saved to: {output_path}")
    print(
        "log_permeability shape:",
        log_permeability.shape,
    )
    print("pressure shape:", pressure.shape)


def plot_samples(
    log_permeability,
    pressure,
    number_to_plot=3,
):
    number_to_plot = min(
        number_to_plot,
        len(log_permeability),
    )

    fig, axes = plt.subplots(
        number_to_plot,
        2,
        figsize=(8, 3.4 * number_to_plot),
        squeeze=False,
    )

    for i in range(number_to_plot):
        image_1 = axes[i, 0].imshow(
            log_permeability[i],
            origin="lower",
            cmap="viridis",
        )
        axes[i, 0].set_title(
            f"Sample {i}: log-permeability"
        )
        fig.colorbar(
            image_1,
            ax=axes[i, 0],
            fraction=0.046,
        )

        image_2 = axes[i, 1].imshow(
            pressure[i],
            origin="lower",
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
        )
        axes[i, 1].set_title(
            f"Sample {i}: pressure"
        )
        fig.colorbar(
            image_2,
            ax=axes[i, 1],
            fraction=0.046,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    number_of_samples = 100
    nx = 64
    ny = 64
    correlation_length = 0.15
    base_seed = 1234

    log_a_data, pressure_data = generate_dataset(
        number_of_samples=number_of_samples,
        nx=nx,
        ny=ny,
        correlation_length=correlation_length,
        base_seed=base_seed,
    )

    save_dataset(
        output_path="data/darcy_test_100.npz",
        log_permeability=log_a_data,
        pressure=pressure_data,
        correlation_length=correlation_length,
        base_seed=base_seed,
    )

    plot_samples(
        log_permeability=log_a_data,
        pressure=pressure_data,
        number_to_plot=3,
    )