import torch
from torch import nn
import torch.nn.functional as F


class SpectralConvolution2d(nn.Module):
    """
    Two-dimensional Fourier layer.

    The layer transforms the input into Fourier space,
    applies learned complex weights to selected modes,
    and transforms the result back to physical space.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        modes_x,
        modes_y,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / (
            input_channels * output_channels
        )

        # Positive and negative frequencies along
        # the first spatial dimension.
        self.weights_positive = nn.Parameter(
            scale
            * torch.randn(
                input_channels,
                output_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

        self.weights_negative = nn.Parameter(
            scale
            * torch.randn(
                input_channels,
                output_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def complex_multiply(
        input_fourier,
        weights,
    ):
        return torch.einsum(
            "bixy,ioxy->boxy",
            input_fourier,
            weights,
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Real-valued two-dimensional FFT.
        x_fourier = torch.fft.rfft2(
            x,
            norm="ortho",
        )

        output_fourier = torch.zeros(
            batch_size,
            self.output_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        modes_x = min(
            self.modes_x,
            height // 2,
        )

        modes_y = min(
            self.modes_y,
            width // 2 + 1,
        )

        output_fourier[
            :,
            :,
            :modes_x,
            :modes_y,
        ] = self.complex_multiply(
            x_fourier[
                :,
                :,
                :modes_x,
                :modes_y,
            ],
            self.weights_positive[
                :,
                :,
                :modes_x,
                :modes_y,
            ],
        )

        output_fourier[
            :,
            :,
            -modes_x:,
            :modes_y,
        ] = self.complex_multiply(
            x_fourier[
                :,
                :,
                -modes_x:,
                :modes_y,
            ],
            self.weights_negative[
                :,
                :,
                :modes_x,
                :modes_y,
            ],
        )

        output = torch.fft.irfft2(
            output_fourier,
            s=(height, width),
            norm="ortho",
        )

        return output


class FourierLayer(nn.Module):
    def __init__(
        self,
        width,
        modes_x,
        modes_y,
    ):
        super().__init__()

        self.spectral_convolution = (
            SpectralConvolution2d(
                input_channels=width,
                output_channels=width,
                modes_x=modes_x,
                modes_y=modes_y,
            )
        )

        # Local pointwise transformation.
        self.local_convolution = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        self.normalization = nn.GroupNorm(
            num_groups=8,
            num_channels=width,
        )

    def forward(self, x):
        spectral_output = (
            self.spectral_convolution(x)
        )

        local_output = self.local_convolution(
            x
        )

        output = (
            spectral_output + local_output
        )

        output = self.normalization(output)

        return output


class FNO2d(nn.Module):
    def __init__(
        self,
        observation_channels=2,
        output_channels=1,
        width=32,
        modes_x=12,
        modes_y=12,
        number_of_layers=4,
        padding=8,
        use_coordinates=True,
    ):
        super().__init__()

        self.use_coordinates = use_coordinates
        self.padding = padding

        coordinate_channels = (
            2 if use_coordinates else 0
        )

        input_channels = (
            observation_channels
            + coordinate_channels
        )

        self.lifting = nn.Conv2d(
            input_channels,
            width,
            kernel_size=1,
        )

        self.fourier_layers = nn.ModuleList(
            [
                FourierLayer(
                    width=width,
                    modes_x=modes_x,
                    modes_y=modes_y,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.projection_1 = nn.Conv2d(
            width,
            128,
            kernel_size=1,
        )

        self.projection_2 = nn.Conv2d(
            128,
            output_channels,
            kernel_size=1,
        )

    @staticmethod
    def create_coordinate_grid(
        batch_size,
        height,
        width,
        device,
        dtype,
    ):
        x_coordinates = torch.linspace(
            0.0,
            1.0,
            width,
            device=device,
            dtype=dtype,
        )

        y_coordinates = torch.linspace(
            0.0,
            1.0,
            height,
            device=device,
            dtype=dtype,
        )

        y_grid, x_grid = torch.meshgrid(
            y_coordinates,
            x_coordinates,
            indexing="ij",
        )

        x_grid = x_grid[
            None,
            None,
            :,
            :,
        ].expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        y_grid = y_grid[
            None,
            None,
            :,
            :,
        ].expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        return torch.cat(
            [x_grid, y_grid],
            dim=1,
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        if self.use_coordinates:
            coordinates = (
                self.create_coordinate_grid(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    device=x.device,
                    dtype=x.dtype,
                )
            )

            x = torch.cat(
                [x, coordinates],
                dim=1,
            )

        x = self.lifting(x)

        # Padding helps with the non-periodic Darcy
        # boundary conditions.
        if self.padding > 0:
            x = F.pad(
                x,
                [
                    0,
                    self.padding,
                    0,
                    self.padding,
                ],
            )

        for layer_index, layer in enumerate(
            self.fourier_layers
        ):
            x = layer(x)

            if (
                layer_index
                < len(self.fourier_layers) - 1
            ):
                x = F.gelu(x)

        if self.padding > 0:
            x = x[
                :,
                :,
                :height,
                :width,
            ]

        x = F.gelu(
            self.projection_1(x)
        )

        output = self.projection_2(x)

        return output


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
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

    test_input = torch.randn(
        4,
        2,
        64,
        64,
    )

    test_output = model(test_input)

    print(
        "Input shape:",
        test_input.shape,
    )

    print(
        "Output shape:",
        test_output.shape,
    )

    print(
        "Parameter count:",
        f"{count_parameters(model):,}",
    )

    assert test_output.shape == (
        4,
        1,
        64,
        64,
    )

    print("FNO forward test passed.")