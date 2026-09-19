import numpy as np

from computational_physics_kit.fdm import jacobi, optimal_omega, rectangle_fourier_series, sor


def square_with_hot_side(n, potential=100.0):
    grid = np.zeros((n + 1, n + 1))
    grid[1:-1, 0] = potential
    return grid


def test_centre_is_a_quarter_of_the_hot_side():
    relaxed, _ = sor(square_with_hot_side(40), tol=1e-10)
    assert abs(relaxed[20, 20] - 25.0) < 1e-8


def test_jacobi_and_sor_reach_the_same_discrete_solution():
    by_jacobi, jacobi_sweeps = jacobi(square_with_hot_side(20), tol=1e-10)
    by_sor, sor_sweeps = sor(square_with_hot_side(20), tol=1e-10)
    assert np.abs(by_jacobi - by_sor).max() < 1e-7
    assert sor_sweeps < jacobi_sweeps/10


def test_fixed_nodes_keep_their_value():
    grid = np.zeros((21, 21))
    fixed = np.zeros_like(grid, dtype=bool)
    grid[10, 5:16] = 7.0
    fixed[10, 5:16] = True
    relaxed, _ = sor(grid, fixed, tol=1e-9)
    assert np.all(relaxed[10, 5:16] == 7.0)
    assert 0.0 < relaxed[5, 10] < 7.0


def test_poisson_source_is_exact_for_a_quadratic():
    n = 16
    h = 1.0/n
    x = np.arange(n + 1)*h
    exact = np.outer(x*(1.0 - x), np.ones(n + 1))
    start = exact.copy()
    start[1:-1, 1:-1] = 0.0
    relaxed, _ = sor(start, tol=1e-12, source=np.full_like(exact, 2.0), h=h)
    assert np.abs(relaxed - exact).max() < 1e-9


def test_optimal_omega_matches_the_square_formula():
    assert abs(optimal_omega(50, 50) - 2.0/(1.0 + np.sin(np.pi/50))) < 1e-12


def test_fourier_series_agrees_with_relaxation_away_from_the_corners():
    n = 60
    axis = np.linspace(0.0, 1.0, n + 1)
    relaxed, _ = sor(square_with_hot_side(n), tol=1e-10)
    series = rectangle_fourier_series(axis, axis, 1.0, 1.0, 100.0, 2000)
    assert np.abs(relaxed[15:46, 15:46] - series[15:46, 15:46]).max() < 0.05
