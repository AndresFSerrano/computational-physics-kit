"""Finite-difference solvers."""
from computational_physics_kit.fdm.electrostatics import Conductor, Electrostatics2D, parallel_plate_capacitor
from computational_physics_kit.fdm.laplace import jacobi, optimal_omega, rectangle_fourier_series, sor

__all__ = ["Conductor", "Electrostatics2D", "jacobi", "optimal_omega", "parallel_plate_capacitor",
           "rectangle_fourier_series", "sor"]
