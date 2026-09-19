"""Finite differences, finite elements and figures for computational physics courses."""
from computational_physics_kit.linalg import solve_tridiagonal
from computational_physics_kit.quadrature import gauss_legendre

__version__ = "0.1.0"

__all__ = ["gauss_legendre", "solve_tridiagonal", "__version__"]
