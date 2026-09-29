from pathlib import Path

from generate_iid_datasets import (
    generate_split,
    check_seed_ranges_do_not_overlap,
)


OUTPUT_DIRECTORY = Path("data/ood")
OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def main():
    common_configuration = {
        "nx": 64,
        "ny": 64,
        "noise_level": 0.01,
        "sensors_per_axis": 8,
    }

    configurations = [
        {
            "split_name": "ood_length_010",
            "number_of_samples": 300,
            "correlation_length": 0.10,
            "field_seed_start": 40000,
            "observation_seed": 140000,
            "output_path":
                OUTPUT_DIRECTORY
                / "darcy_ood_length_010_300.npz",
        },
        {
            "split_name": "ood_length_040",
            "number_of_samples": 300,
            "correlation_length": 0.40,
            "field_seed_start": 50000,
            "observation_seed": 150000,
            "output_path":
                OUTPUT_DIRECTORY
                / "darcy_ood_length_040_300.npz",
        },
    ]

    check_seed_ranges_do_not_overlap(
        configurations
    )

    for configuration in configurations:
        generate_split(
            **configuration,
            **common_configuration,
        )


if __name__ == "__main__":
    main()