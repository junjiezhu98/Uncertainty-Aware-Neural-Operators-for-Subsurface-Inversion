import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def harmonic_mean(a1, a2, epsilon=1e-12):
    """Harmonic mean of two positive permeability values."""
    return 2.0 * a1 * a2 / (a1 + a2 + epsilon)


def create_permeability(nx, ny):
    """
    Create a permeability field containing a low-permeability
    circular inclusion.
    """
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    permeability = np.ones((ny, nx))

    # Circular low-permeability region
    radius = 0.18
    distance = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)

    permeability[distance < radius] = 0.05

    return x, y, permeability


def solve_darcy(permeability, verbose=False):
    """
    Solve:
        -div(a grad u) = 0

    Boundary conditions:
        u = 1 on the left
        u = 0 on the right
        zero flux on the top and bottom
    """

    ny, nx = permeability.shape

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    number_of_unknowns = nx * ny

    A = lil_matrix(
        (number_of_unknowns, number_of_unknowns)
    )
    b = np.zeros(number_of_unknowns)

    def index(i, j):
        return j * nx + i

    for j in range(ny):
        for i in range(nx):
            k = index(i, j)

            # Left boundary: u = 1
            if i == 0:
                A[k, k] = 1.0
                b[k] = 1.0
                continue

            # Right boundary: u = 0
            if i == nx - 1:
                A[k, k] = 1.0
                b[k] = 0.0
                continue

            diagonal = 0.0

            # West interface
            a_west = harmonic_mean(
                permeability[j, i],
                permeability[j, i - 1],
            )
            coefficient_west = a_west / dx**2

            A[k, index(i - 1, j)] = -coefficient_west
            diagonal += coefficient_west

            # East interface
            a_east = harmonic_mean(
                permeability[j, i],
                permeability[j, i + 1],
            )
            coefficient_east = a_east / dx**2

            A[k, index(i + 1, j)] = -coefficient_east
            diagonal += coefficient_east

            # South interface
            if j > 0:
                a_south = harmonic_mean(
                    permeability[j, i],
                    permeability[j - 1, i],
                )
                coefficient_south = a_south / dy**2

                A[k, index(i, j - 1)] = -coefficient_south
                diagonal += coefficient_south

            # North interface
            if j < ny - 1:
                a_north = harmonic_mean(
                    permeability[j, i],
                    permeability[j + 1, i],
                )
                coefficient_north = a_north / dy**2

                A[k, index(i, j + 1)] = -coefficient_north
                diagonal += coefficient_north

            A[k, k] = diagonal

    A = A.tocsr()

    pressure_vector = spsolve(A, b)
    pressure = pressure_vector.reshape(ny, nx)

    relative_residual = (
        np.linalg.norm(A @ pressure_vector - b)
        / np.linalg.norm(b)
    )
    if verbose:
        print(f"Grid: {nx} x {ny}")
        print(
            "Linear-system residual: "
            f"{relative_residual:.3e}"
        )
        print(f"Minimum pressure: {pressure.min():.6f}")
        print(f"Maximum pressure: {pressure.max():.6f}")

    return pressure


def compute_flux(permeability, pressure, dx, dy):
    """
    Approximate Darcy velocity:
        q = -a grad(u)
    """

    du_dy, du_dx = np.gradient(pressure, dy, dx)

    flux_x = -permeability * du_dx
    flux_y = -permeability * du_dy

    flux_magnitude = np.sqrt(
        flux_x**2 + flux_y**2
    )

    return flux_x, flux_y, flux_magnitude


def plot_results(
    x,
    y,
    permeability,
    pressure,
    flux_x,
    flux_y,
    flux_magnitude,
):
    X, Y = np.meshgrid(x, y)
    extent = [0.0, 1.0, 0.0, 1.0]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
    )

    image_1 = axes[0].imshow(
        permeability,
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axes[0].set_title("Permeability")
    fig.colorbar(image_1, ax=axes[0])

    image_2 = axes[1].imshow(
        pressure,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
    )
    axes[1].set_title("Pressure")
    fig.colorbar(image_2, ax=axes[1])

    image_3 = axes[2].imshow(
        flux_magnitude,
        origin="lower",
        extent=extent,
        cmap="magma",
    )

    # Plot fewer arrows for readability
    step = 3
    axes[2].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        flux_x[::step, ::step],
        flux_y[::step, ::step],
        color="white",
        scale=25,
    )

    axes[2].set_title("Darcy flux")
    fig.colorbar(image_3, ax=axes[2])

    for axis in axes:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    nx = 64
    ny = 64

    x, y, permeability = create_permeability(nx, ny)

    pressure = solve_darcy(permeability, verbose=False)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    flux_x, flux_y, flux_magnitude = compute_flux(
        permeability,
        pressure,
        dx,
        dy,
    )

    plot_results(
        x,
        y,
        permeability,
        pressure,
        flux_x,
        flux_y,
        flux_magnitude,
    )