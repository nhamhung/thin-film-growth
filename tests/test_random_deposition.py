"""Validation checks for Phase 1: random deposition must reproduce its exact theory."""

import numpy as np
import pytest

from thin_film import random_deposition as rd
from thin_film.roughness import fit_growth_exponent, roughness


@pytest.fixture(scope="module")
def ensemble():
    return rd.run_ensemble(rd.RDSettings(n_columns=200, n_monolayers=500, n_runs=30, seed=0))


def test_flat_surface_has_zero_roughness():
    assert roughness(np.full(50, 7)) == 0.0


def test_roughness_is_population_std():
    assert roughness(np.array([0, 2])) == pytest.approx(1.0)


def test_deposit_one_adds_exactly_one_atom():
    heights = np.zeros(10, dtype=np.int64)
    rng = np.random.default_rng(1)
    for _ in range(100):
        rd.deposit_one(heights, rng)
    assert heights.sum() == 100


def test_growth_exponent_is_one_half(ensemble):
    beta, err = fit_growth_exponent(ensemble["monolayers"], ensemble["roughness_mean"])
    assert beta == pytest.approx(0.5, abs=0.03)
    assert err < 0.01


def test_matches_exact_theory(ensemble):
    ratio = ensemble["roughness_mean"] / ensemble["roughness_theory"]
    assert np.all(np.abs(ratio - 1) < 0.05)


def test_same_seed_is_reproducible():
    s = rd.RDSettings(n_columns=50, n_monolayers=20, n_runs=3, seed=7)
    assert rd.run_ensemble(s).equals(rd.run_ensemble(s))


def test_fit_recovers_known_exponent():
    t = np.logspace(0, 3, 20)
    beta, _ = fit_growth_exponent(t, 2.0 * t**0.33)
    assert beta == pytest.approx(0.33, abs=1e-9)
