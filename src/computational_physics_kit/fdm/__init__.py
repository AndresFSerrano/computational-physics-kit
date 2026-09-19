"""Finite-difference solvers."""
from computational_physics_kit.fdm.laplace import jacobi, optimal_omega, rectangle_fourier_series, sor

__all__ = ["jacobi", "optimal_omega", "rectangle_fourier_series", "sor"]
