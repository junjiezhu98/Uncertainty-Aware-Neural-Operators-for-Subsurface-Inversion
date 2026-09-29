from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DATASET_PATH = Path("data/darcy_test_100.npz")


def inspect_dataset(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path.resolve()}"
        )

    # allow_pickle=False is safer and足够读取纯数值数组
    with np.load(dataset_path, allow_pickle=False) as data:
        print("Variables stored in the file:")
        print(data.files)
        print()

        # Check that required arrays exist
        required_keys = {
            "log_permeability",
            "pressure",
        }

        missing_keys = required_keys - set(data.files)

        if missing_keys:
            raise KeyError(
                f"Missing required variables: {missing_keys}"
            )

        # Copy arrays before closing the NPZ file
        log_permeability = data[
            "log_permeability"
        ].copy()

        pressure = data["pressure"].copy()

        correlation_length = (
            data["correlation_length"].item()
            if "correlation_length" in data.files
            else None
        )

        base_seed = (
            data["base_seed"].item()
            if "base_seed" in data.files
            else None
        )
    lower_fraction = np.mean(
        log_permeability <= -3.0 + 1e-6
    )

    upper_fraction = np.mean(
        log_permeability >= 3.0 - 1e-6
    )

    print(
        f"Fraction clipped at -3: "
        f"{100 * lower_fraction:.4f}%"
    )

    print(
        f"Fraction clipped at +3: "
        f"{100 * upper_fraction:.4f}%"
    )
    print("Basic information")
    print("-----------------")
    print(
        "log_permeability shape:",
        log_permeability.shape,
    )
    print("pressure shape:", pressure.shape)

    print(
        "log_permeability dtype:",
        log_permeability.dtype,
    )
    print("pressure dtype:", pressure.dtype)

    print(
        "correlation_length:",
        correlation_length,
    )
    print("base_seed:", base_seed)
    print()

    run_validation_checks(
        log_permeability,
        pressure,
    )

    plot_random_samples(
        log_permeability,
        pressure,
        number_of_samples=4,
        seed=2026,
    )


def run_validation_checks(
    log_permeability,
    pressure,
):
    print("Validation checks")
    print("-----------------")

    # 1. Shape checks
    assert log_permeability.ndim == 3, (
        "log_permeability must have shape "
        "(samples, height, width)"
    )

    assert pressure.ndim == 3, (
        "pressure must have shape "
        "(samples, height, width)"
    )

    assert log_permeability.shape == pressure.shape, (
        "Permeability and pressure arrays "
        "must have identical shapes"
    )

    number_of_samples, ny, nx = pressure.shape

    assert number_of_samples > 0
    assert nx > 2 and ny > 2

    print("✓ Array shapes are valid")

    # 2. NaN and infinity checks
    assert np.all(np.isfinite(log_permeability)), (
        "log_permeability contains NaN or infinity"
    )

    assert np.all(np.isfinite(pressure)), (
        "pressure contains NaN or infinity"
    )

    print("✓ No NaN or infinity found")

    # 3. Pressure range
    pressure_min = pressure.min()
    pressure_max = pressure.max()

    print(
        f"Pressure range: "
        f"[{pressure_min:.8f}, "
        f"{pressure_max:.8f}]"
    )

    tolerance = 1e-5

    assert pressure_min >= -tolerance, (
        "Pressure is significantly below zero"
    )

    assert pressure_max <= 1.0 + tolerance, (
        "Pressure is significantly above one"
    )

    print("✓ Pressure range is physically reasonable")

    # 4. Boundary-condition checks
    left_boundary = pressure[:, :, 0]
    right_boundary = pressure[:, :, -1]

    left_error = np.max(
        np.abs(left_boundary - 1.0)
    )

    right_error = np.max(
        np.abs(right_boundary - 0.0)
    )

    print(
        f"Maximum left-boundary error: "
        f"{left_error:.3e}"
    )

    print(
        f"Maximum right-boundary error: "
        f"{right_error:.3e}"
    )

    assert left_error < tolerance, (
        "Left Dirichlet boundary is incorrect"
    )

    assert right_error < tolerance, (
        "Right Dirichlet boundary is incorrect"
    )

    print("✓ Dirichlet boundary conditions are correct")

    # 5. Statistical information
    permeability = np.exp(log_permeability)

    print()
    print("Dataset statistics")
    print("------------------")

    print(
        "Log-permeability:"
        f" min={log_permeability.min():.4f},"
        f" max={log_permeability.max():.4f},"
        f" mean={log_permeability.mean():.4f},"
        f" std={log_permeability.std():.4f}"
    )

    print(
        "Permeability:"
        f" min={permeability.min():.4f},"
        f" max={permeability.max():.4f},"
        f" mean={permeability.mean():.4f},"
        f" std={permeability.std():.4f}"
    )

    print(
        "Pressure:"
        f" min={pressure.min():.4f},"
        f" max={pressure.max():.4f},"
        f" mean={pressure.mean():.4f},"
        f" std={pressure.std():.4f}"
    )

    # 6. Confirm samples are not all identical
    log_a_sample_variation = np.mean(
        np.std(log_permeability, axis=0)
    )

    pressure_sample_variation = np.mean(
        np.std(pressure, axis=0)
    )

    print()
    print(
        "Average variation across permeability samples:",
        f"{log_a_sample_variation:.6f}",
    )

    print(
        "Average variation across pressure samples:",
        f"{pressure_sample_variation:.6f}",
    )

    assert log_a_sample_variation > 1e-6, (
        "Permeability samples appear identical"
    )

    assert pressure_sample_variation > 1e-8, (
        "Pressure samples appear identical"
    )

    print("✓ Samples are not all identical")

    # 7. Check for duplicate consecutive samples
    duplicate_count = 0

    for i in range(1, number_of_samples):
        if np.array_equal(
            log_permeability[i],
            log_permeability[i - 1],
        ):
            duplicate_count += 1

    print(
        f"Consecutive duplicate fields: "
        f"{duplicate_count}"
    )

    assert duplicate_count == 0, (
        "Duplicate consecutive samples detected"
    )

    print("✓ No consecutive duplicate fields")
    print()
    print("All automatic checks passed.")


def plot_random_samples(
    log_permeability,
    pressure,
    number_of_samples=4,
    seed=2026,
):
    rng = np.random.default_rng(seed)

    number_of_samples = min(
        number_of_samples,
        len(log_permeability),
    )

    selected_indices = rng.choice(
        len(log_permeability),
        size=number_of_samples,
        replace=False,
    )

    fig, axes = plt.subplots(
        number_of_samples,
        3,
        figsize=(11, 3 * number_of_samples),
        squeeze=False,
    )

    for row, sample_index in enumerate(
        selected_indices
    ):
        log_a = log_permeability[sample_index]
        permeability = np.exp(log_a)
        sample_pressure = pressure[sample_index]

        image_1 = axes[row, 0].imshow(
            log_a,
            origin="lower",
            cmap="viridis",
        )
        axes[row, 0].set_title(
            f"Sample {sample_index}: log(a)"
        )
        fig.colorbar(
            image_1,
            ax=axes[row, 0],
            fraction=0.046,
        )

        image_2 = axes[row, 1].imshow(
            permeability,
            origin="lower",
            cmap="viridis",
        )
        axes[row, 1].set_title(
            f"Sample {sample_index}: a"
        )
        fig.colorbar(
            image_2,
            ax=axes[row, 1],
            fraction=0.046,
        )

        image_3 = axes[row, 2].imshow(
            sample_pressure,
            origin="lower",
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 2].set_title(
            f"Sample {sample_index}: pressure"
        )
        fig.colorbar(
            image_3,
            ax=axes[row, 2],
            fraction=0.046,
        )

        for column in range(3):
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inspect_dataset(DATASET_PATH)