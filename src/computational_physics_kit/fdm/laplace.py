"""Relaxation solvers for the Laplace and Poisson equations on a uniform 2D grid.

Arrays follow the convention u[i, j] = u(x_i, y_j). The outer frame is always Dirichlet; `fixed`
marks extra nodes (conductors, internal boundaries) that keep their value.
"""
import numpy as np


def optimal_omega(nx, ny):
    """Optimal SOR factor for a rectangle of nx by ny cells, from the spectral radius of Jacobi."""
    rho = 0.5*(np.cos(np.pi/nx) + np.cos(np.pi/ny))
    return 2.0/(1.0 + np.sqrt(1.0 - rho*rho))


def _neighbour_average(u, h, source):
    """Five-point average of the four neighbours, plus the Poisson source term if any."""
    average = np.zeros_like(u)
    average[1:-1, 1:-1] = 0.25*(u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
    if source is not None:
        average[1:-1, 1:-1] += 0.25*h*h*source[1:-1, 1:-1]
    return average


def _free_nodes(shape, fixed):
    """Interior nodes that are not fixed."""
    free = np.zeros(shape, dtype=bool)
    free[1:-1, 1:-1] = True
    if fixed is not None:
        free &= ~np.asarray(fixed, dtype=bool)
    return free


def jacobi(u, fixed=None, tol=1e-6, max_sweeps=200000, source=None, h=1.0):
    """Jacobi relaxation of -laplacian(u) = source until the largest change drops below tol.

    Returns the relaxed array and the number of sweeps.
    """
    u = np.array(u, dtype=float, copy=True)
    free = _free_nodes(u.shape, fixed)
    for sweep in range(1, max_sweeps + 1):
        new = np.where(free, _neighbour_average(u, h, source), u)
        change = np.abs(new - u).max()
        u = new
        if change < tol:
            return u, sweep
    return u, max_sweeps


def sor(u, fixed=None, omega=None, tol=1e-8, max_sweeps=200000, source=None, h=1.0):
    """Red-black successive over-relaxation of -laplacian(u) = source.

    With omega=None the optimal factor for the grid is used. Returns the relaxed array and the
    number of sweeps.
    """
    u = np.array(u, dtype=float, copy=True)
    if omega is None:
        omega = optimal_omega(u.shape[0] - 1, u.shape[1] - 1)
    free = _free_nodes(u.shape, fixed)
    i, j = np.indices(u.shape)
    colours = (free & ((i + j) % 2 == 0), free & ((i + j) % 2 == 1))
    for sweep in range(1, max_sweeps + 1):
        change = 0.0
        for colour in colours:
            delta = omega*(_neighbour_average(u, h, source) - u)
            u[colour] += delta[colour]
            change = max(change, np.abs(delta[colour]).max(initial=0.0))
        if change < tol:
            return u, sweep
    return u, max_sweeps


def rectangle_fourier_series(x, y, width, height, potential, n_modes):
    """Exact solution in a rectangle with u = potential on the side y = 0 and zero on the others.

    Sums the first n_modes odd modes in a form that does not overflow for large mode numbers.
    """
    grid_x, grid_y = np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="ij")
    total = np.zeros_like(grid_x)
    for mode in range(1, 2*n_modes, 2):
        k = mode*np.pi/width
        ratio = (np.exp(-k*grid_y)*(1.0 - np.exp(-2.0*k*(height - grid_y)))
                 / (1.0 - np.exp(-2.0*k*height)))
        total += (4.0*potential/(mode*np.pi))*np.sin(k*grid_x)*ratio
    return total
