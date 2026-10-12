"""Validation checks for Phase 3: temperature-activated hopping."""

import numpy as np
import pytest

from thin_film import random_deposition as rd
from thin_film import thermal as th
from thin_film.units import K_B, thermal_parameters


def test_unit_conversion():
    eps, hops = thermal_parameters(300, e_bond=0.2, e_diff=0.8, flux=1.0, max_hops=1e9)
    assert eps == pytest.approx(0.2 / (K_B * 300))
    assert hops == pytest.approx(1e13 * np.exp(-0.8 / (K_B * 300)))
    assert thermal_parameters(2000, 0.2, 0.8, 1.0, max_hops=50)[1] == 50  # cap applies


def test_hops_conserve_atoms():
    heights = np.zeros(64, np.int64)
    th.thermal_grow(heights, 5000, 1.0, 20.0, 1)
    assert heights.sum() == 5000


def test_no_hops_reduces_to_random_deposition():
    s = th.ThermalSettings(n_columns=200, n_monolayers=100, n_runs=20, n_points=10)
    df = th.run_ensemble(s, epsilon=5.0, hops=0.0)
    theory = rd.theory_roughness(df["monolayers"], 200)
    assert np.all(np.abs(df["roughness_mean"] / theory - 1) < 0.07)


def test_same_seed_is_reproducible():
    a, b = np.zeros(32, np.int64), np.zeros(32, np.int64)
    th.thermal_grow(a, 2000, 2.0, 5.5, 9)
    th.thermal_grow(b, 2000, 2.0, 5.5, 9)
    assert np.array_equal(a, b)


@pytest.fixture(scope="module")
def base():
    return th.ThermalSettings(n_columns=100, n_monolayers=30, n_runs=4, n_points=5)


def test_hotter_is_smoother(base):
    df = th.sweep_temperature(base, [260, 360])
    assert df["roughness"].iloc[1] < 0.6 * df["roughness"].iloc[0]


def test_faster_deposition_is_rougher(base):
    df = th.sweep_flux(th.ThermalSettings(**{**base.__dict__, "temperature": 360}), [0.1, 100])
    assert df["roughness"].iloc[1] > 1.5 * df["roughness"].iloc[0]
