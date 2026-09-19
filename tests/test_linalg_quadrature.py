import numpy as np

from computational_physics_kit import gauss_legendre, solve_tridiagonal


def test_tridiagonal_solver_matches_a_dense_solve():
    rng = np.random.default_rng(7)
    n = 50
    lower = rng.uniform(-1.0, 1.0, n - 1)
    upper = rng.uniform(-1.0, 1.0, n - 1)
    diag = 4.0 + rng.uniform(0.0, 1.0, n)
    rhs = rng.uniform(-1.0, 1.0, n)
    dense = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
    assert np.abs(solve_tridiagonal(lower, diag, upper, rhs) - np.linalg.solve(dense, rhs)).max() < 1e-12


def test_three_point_rule_is_exact_up_to_degree_five():
    nodes, weights = gauss_legendre(3)
    for degree in range(6):
        exact = (1.0 - (-1.0)**(degree + 1))/(degree + 1)
        assert abs(np.sum(weights*nodes**degree) - exact) < 1e-14
    assert abs(np.sum(weights*nodes**6) - 2.0/7.0) > 1e-3
