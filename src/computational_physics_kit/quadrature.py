"""Gauss-Legendre quadrature."""
import numpy as np


def gauss_legendre(n):
    """Nodes and weights of the n-point Gauss-Legendre rule on [-1, 1]; exact up to degree 2n - 1."""
    return np.polynomial.legendre.leggauss(n)
