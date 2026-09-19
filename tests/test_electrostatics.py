import numpy as np
import pytest

from computational_physics_kit.fdm import Electrostatics2D, parallel_plate_capacitor


def test_parallel_plates_are_antisymmetric():
    capacitor = parallel_plate_capacitor(box=60.0, h=1.0, plate_length=16.0, separation=8.0, voltage=50.0)
    capacitor.solve(tol=1e-9)
    centre = 30
    assert abs(capacitor.phi[centre, centre]) < 1e-6
    assert np.abs(capacitor.phi + capacitor.phi[::-1, :]).max() < 1e-6


def test_field_between_the_plates_points_from_plus_to_minus():
    capacitor = parallel_plate_capacitor(box=60.0, h=0.5, plate_length=20.0, separation=8.0, voltage=50.0)
    capacitor.solve(tol=1e-9)
    e_x, e_y = capacitor.electric_field()
    centre = 60
    assert 0.7*100.0/8.0 < e_x[centre, centre] < 1.05*100.0/8.0
    assert abs(e_y[centre, centre]) < 1e-6


def test_rotated_capacitor_keeps_zero_potential_at_the_centre():
    capacitor = parallel_plate_capacitor(box=60.0, h=1.0, plate_length=16.0, separation=10.0, angle=45.0)
    capacitor.solve(tol=1e-9)
    assert abs(capacitor.phi[30, 30]) < 1e-6
    assert len(capacitor.conductors) == 2


def test_coaxial_cable_follows_the_logarithm():
    inner, outer, voltage = 4.0, 20.0, 10.0
    cable = Electrostatics2D(50.0, 50.0, 0.25)
    cable.add_disk((25.0, 25.0), inner, voltage)
    cable.add_ring((25.0, 25.0), outer, 30.0, 0.0)
    cable.solve(tol=1e-8)
    radius = np.sqrt(inner*outer)
    i = int(round((25.0 + radius)/0.25))
    expected = voltage*np.log(outer/radius)/np.log(outer/inner)
    assert abs(cable.phi[i, 100] - expected) < 0.03*voltage


def test_a_conductor_that_misses_the_grid_is_rejected():
    problem = Electrostatics2D(10.0, 10.0, 1.0)
    with pytest.raises(ValueError):
        problem.add_disk((5.3, 5.3), 0.1, 1.0)
