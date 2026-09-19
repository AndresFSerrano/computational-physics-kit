"""Linear algebra helpers backed by LAPACK."""
import numpy as np
from scipy.linalg import solve_banded


def solve_tridiagonal(lower, diag, upper, rhs):
    """Solve a tridiagonal system with LAPACK's banded solver.

    `lower` and `upper` have length n - 1; `diag` and `rhs` have length n.
    """
    diag = np.asarray(diag, dtype=float)
    bands = np.zeros((3, diag.size))
    bands[0, 1:] = upper
    bands[1] = diag
    bands[2, :-1] = lower
    return solve_banded((1, 1), bands, np.asarray(rhs, dtype=float))
