"""Validation checks for Phase 2: surface relaxation (Edwards-Wilkinson behaviour)."""

import numpy as np
import pytest

from thin_film import surface_relaxation as sr
from thin_film.roughness import fit_growth_exponent


def test_atom_rolls_into_lower_neighbour():
    heights = np.array([0, 5, 7], dtype=np.int64)  # only the left neighbour is lower
    sr.relax_deposit_python(heights, np.array([1]), np.array([0.9]))
    assert heights.tolist() == [1, 5, 7]


def test_atom_stays_when_no_neighbour_is_lower():
    heights = np.array([5, 2, 5], dtype=np.int64)
    sr.relax_deposit_python(heights, np.array([1]), np.array([0.5]))
    assert heights.tolist() == [5, 3, 5]


def test_tie_break_uses_coin():
    for coin, expected in [(0.1, [1, 5, 0]), (0.9, [0, 5, 1])]:
        heights = np.array([0, 5, 0], dtype=np.int64)
        sr.relax_deposit_python(heights, np.array([1]), np.array([coin]))
        assert heights.tolist() == expected


def test_periodic_boundary():
    heights = np.array([4, 4, 0], dtype=np.int64)  # column 0's left neighbour is column 2
    sr.relax_deposit_python(heights, np.array([0]), np.array([0.5]))
    assert heights.tolist() == [4, 4, 1]


def test_numba_matches_python_exactly():
    rng = np.random.default_rng(3)
    cols, coins = rng.integers(50, size=20_000), rng.random(20_000)
    a, b = np.zeros(50, np.int64), np.zeros(50, np.int64)
    sr.relax_deposit_python(a, cols, coins)
    sr.relax_deposit_numba(b, cols, coins)
    assert np.array_equal(a, b) and a.sum() == 20_000


def test_growth_exponent_is_about_one_quarter():
    df = sr.run_ensemble(sr.RelaxSettings(n_columns=200, n_monolayers=100, n_runs=20, n_points=20))
    beta, _ = fit_growth_exponent(df["monolayers"], df["roughness_mean"])
    assert beta == pytest.approx(0.25, abs=0.05)


def test_roughness_saturates_and_scales_with_size():
    w = {}
    for L in (25, 100):
        df = sr.run_ensemble(sr.RelaxSettings(n_columns=L, n_monolayers=10_000, n_runs=40, n_points=30))
        late = df["roughness_mean"].iloc[-6:]
        assert late.max() / late.min() < 1.2  # flat at late times (up to sampling noise)
        w[L] = sr.saturation_width(df)
    alpha = np.log(w[100] / w[25]) / np.log(4)
    assert alpha == pytest.approx(0.5, abs=0.1)  # W_sat ~ L^(1/2)
