"""Two-dimensional electrostatics: conductors of any shape inside a grounded box."""
from dataclasses import dataclass

import numpy as np

from computational_physics_kit.fdm.laplace import sor


@dataclass
class Conductor:
    """A set of grid nodes held at a fixed potential."""

    mask: np.ndarray
    potential: float
    name: str


class Electrostatics2D:
    """Laplace problem in a rectangular box with conductors held at fixed potentials.

    The box walls are held at `boundary`. Arrays follow u[i, j] = u(x_i, y_j).
    """

    def __init__(self, width, height, h, boundary=0.0):
        self.h = float(h)
        nx = int(round(width/h))
        ny = int(round(height/h))
        self.x = np.arange(nx + 1)*self.h
        self.y = np.arange(ny + 1)*self.h
        self.grid_x, self.grid_y = np.meshgrid(self.x, self.y, indexing="ij")
        self.boundary = float(boundary)
        self.phi = np.full((nx + 1, ny + 1), self.boundary)
        self.fixed = np.zeros((nx + 1, ny + 1), dtype=bool)
        self.conductors = []
        self.sweeps = 0

    def add_mask(self, mask, potential, name=None):
        """Hold every node where `mask` is true at `potential`."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("the conductor does not cover any grid node; refine h or enlarge it")
        self.phi[mask] = potential
        self.fixed |= mask
        self.conductors.append(Conductor(mask, float(potential), name or f"conductor {len(self.conductors) + 1}"))
        return self

    def add_plate(self, start, end, potential, thickness=None, name=None):
        """Straight plate between two points, at any angle; default thickness is about one cell."""
        thickness = 1.5*self.h if thickness is None else thickness
        x0, y0 = start
        x1, y1 = end
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx*dx + dy*dy
        s = np.clip(((self.grid_x - x0)*dx + (self.grid_y - y0)*dy)/length_sq, 0.0, 1.0)
        distance = np.hypot(self.grid_x - (x0 + s*dx), self.grid_y - (y0 + s*dy))
        return self.add_mask(distance <= thickness/2.0, potential, name)

    def add_disk(self, center, radius, potential, name=None):
        """Solid disk (the cross-section of a long cylinder)."""
        distance = np.hypot(self.grid_x - center[0], self.grid_y - center[1])
        return self.add_mask(distance <= radius, potential, name)

    def add_ring(self, center, inner_radius, outer_radius, potential, name=None):
        """Annulus between two radii (the cross-section of a cylindrical shell)."""
        distance = np.hypot(self.grid_x - center[0], self.grid_y - center[1])
        return self.add_mask((distance >= inner_radius) & (distance <= outer_radius), potential, name)

    def add_rectangle(self, corner, opposite_corner, potential, name=None):
        """Filled axis-aligned rectangle."""
        x_low, x_high = sorted((corner[0], opposite_corner[0]))
        y_low, y_high = sorted((corner[1], opposite_corner[1]))
        inside = ((self.grid_x >= x_low) & (self.grid_x <= x_high)
                  & (self.grid_y >= y_low) & (self.grid_y <= y_high))
        return self.add_mask(inside, potential, name)

    def solve(self, tol=1e-8, omega=None, max_sweeps=200000):
        """Relax the potential with red-black SOR; returns the number of sweeps."""
        self.phi, self.sweeps = sor(self.phi, self.fixed, omega=omega, tol=tol, max_sweeps=max_sweeps)
        return self.sweeps

    def electric_field(self):
        """E = -grad(phi) with second-order differences, one-sided at the walls."""
        grad_x, grad_y = np.gradient(self.phi, self.h, self.h, edge_order=2)
        return -grad_x, -grad_y

    def field_magnitude(self):
        """Norm of the electric field on the grid."""
        e_x, e_y = self.electric_field()
        return np.hypot(e_x, e_y)


def parallel_plate_capacitor(box=100.0, h=1.0, plate_length=20.0, separation=10.0, voltage=50.0, angle=0.0):
    """Two plates at +voltage and -voltage centred in a grounded square box.

    `angle` rotates the pair, in degrees; 0 gives vertical plates separated along x.
    """
    problem = Electrostatics2D(box, box, h)
    centre = np.array([box/2.0, box/2.0])
    theta = np.radians(angle)
    across = np.array([np.cos(theta), np.sin(theta)])
    along = np.array([-np.sin(theta), np.cos(theta)])
    for sign, name in ((1.0, "positive plate"), (-1.0, "negative plate")):
        middle = centre - sign*across*separation/2.0
        problem.add_plate(middle - along*plate_length/2.0, middle + along*plate_length/2.0, sign*voltage,
                          name=name)
    return problem
