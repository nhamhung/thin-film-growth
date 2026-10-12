"""Validation checks for Phase 4: trench filling and void detection."""

import numpy as np

from thin_film import trench as tr
from thin_film.trench import EMPTY, FILM, SUBSTRATE


def test_initial_trench_geometry():
    s = tr.TrenchSettings(width=40, trench_width=6, trench_depth=10, base=2)
    g = tr.make_trench(s)
    x0, x1 = s.trench_x
    assert (g[:2] == SUBSTRATE).all()                        # base under everything
    assert (g[2:12, x0:x1] == EMPTY).all()                   # trench is open
    assert (g[2:12, :x0] == SUBSTRATE).all() and (g[2:12, x1:] == SUBSTRATE).all()
    assert (g[12:] == EMPTY).all()


def test_flood_fill_finds_sealed_pocket_only():
    g = np.zeros((8, 8), np.int8)
    g[:5] = SUBSTRATE
    g[1:4, 2:5] = EMPTY      # pocket sealed by the substrate on all sides: 9 cells
    g[2:5, 6] = EMPTY        # open slot reaching the surface: not a void
    voids = tr.find_voids(g)
    assert voids.sum() == 9 and voids[1:4, 2:5].all()


def test_flood_fill_wraps_periodically():
    g = np.zeros((4, 6), np.int8)
    g[:3] = SUBSTRATE
    g[1, 0] = g[1, 5] = EMPTY   # pocket split across the left/right edge
    g[2, 5] = EMPTY             # ...connected to the open top through column 5
    assert tr.find_voids(g).sum() == 0


def test_atoms_conserved():
    s = tr.TrenchSettings(width=64, n_monolayers=10, temperature=400)
    g = tr.grow(s)
    assert (g == FILM).sum() == 640
    assert (g == SUBSTRATE).sum() == (tr.make_trench(s) == SUBSTRATE).sum()


def test_cold_narrow_trench_traps_a_void():
    r = tr.run_ensemble(tr.TrenchSettings(trench_width=6, temperature=280, n_runs=3))
    assert r["void_probability"] == 1.0 and r["trench_void_area"] > 30


def test_hot_wide_trench_fills_almost_completely():
    s = tr.TrenchSettings(trench_width=24, n_runs=3)
    cold = tr.run_ensemble(tr.TrenchSettings(**{**s.__dict__, "temperature": 280}))
    hot = tr.run_ensemble(tr.TrenchSettings(**{**s.__dict__, "temperature": 450}))
    trench_area = s.trench_width * s.trench_depth
    assert hot["trench_void_area"] < 0.05 * trench_area < cold["trench_void_area"]


def test_heating_improves_step_coverage():
    cold = tr.run_ensemble(tr.TrenchSettings(temperature=280, n_runs=3))
    hot = tr.run_ensemble(tr.TrenchSettings(temperature=450, n_runs=3))
    assert hot["step_coverage"] > cold["step_coverage"] + 0.15
    assert hot["film_density"] > cold["film_density"]
