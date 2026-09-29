import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def solve_darcy(nx=32, ny=32):
    """
    Solve:
        -div(a grad u) = 0

    Boundary conditions:
        u(0, y) = 1
        u(1, y) = 0
        du/dn = 0 on top and bottom

    Initial test:
        a(x, y) = 1

    Exact solution:
        u(x, y) = 1 - x
    """

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Constant permeability for the first validation test
    permeability = np.ones((ny, nx))

    number_of_unknowns = nx * ny
    A = lil_matrix((number_of_unknowns, number_of_unknowns))
    b = np.zeros(number_of_unknowns)

    def index(i, j):
        """Convert grid coordinate (i, j) to vector index."""
        return j * nx + i

    for j in range(ny):
        for i in range(nx):
            k = index(i, j)

            # Left Dirichlet boundary: u = 1
            if i == 0:
                A[k, k] = 1.0
                b[k] = 1.0
                continue

            # Right Dirichlet boundary: u = 0
            if i == nx - 1:
                A[k, k] = 1.0
                b[k] = 0.0
                continue

            diagonal = 0.0

            # West neighbor
            coefficient = 1.0 / dx**2
            A[k, index(i - 1, j)] = -coefficient
            diagonal += coefficient

            # East neighbor
            coefficient = 1.0 / dx**2
            A[k, index(i + 1, j)] = -coefficient
            diagonal += coefficient

            # South neighbor
            if j > 0:
                coefficient = 1.0 / dy**2
                A[k, index(i, j - 1)] = -coefficient
                diagonal += coefficient

            # North neighbor
            if j < ny - 1:
                coefficient = 1.0 / dy**2
                A[k, index(i, j + 1)] = -coefficient
                diagonal += coefficient

            # Omitting neighbors outside the top and bottom boundaries
            # implements the zero-flux condition.
            A[k, k] = diagonal

    A = A.tocsr()

    pressure_vector = spsolve(A, b)
    pressure = pressure_vector.reshape(ny, nx)

    X, Y = np.meshgrid(x, y)
    exact_pressure = 1.0 - X

    absolute_error = np.abs(pressure - exact_pressure)

    residual = np.linalg.norm(A @ pressure_vector - b) / np.linalg.norm(b)
    relative_error = (
        np.linalg.norm(pressure - exact_pressure)
        / np.linalg.norm(exact_pressure)
    )

    print(f"Grid: {nx} x {ny}")
    print(f"Linear-system residual: {residual:.3e}")
    print(f"Relative solution error: {relative_error:.3e}")

    return x, y, permeability, pressure, exact_pressure, absolute_error


def plot_results(
    x,
    y,
    permeability,
    pressure,
    exact_pressure,
    absolute_error,
):
    extent = [x.min(), x.max(), y.min(), y.max()]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

    images = [
        (permeability, "Permeability $a(x,y)$"),
        (pressure, "Numerical pressure"),
        (exact_pressure, "Exact pressure $1-x$"),
        (absolute_error, "Absolute error"),
    ]

    for axis, (field, title) in zip(axes, images):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, fraction=0.046)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = solve_darcy(nx=32, ny=32)
    plot_results(*results)