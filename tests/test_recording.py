"""Recordings must replay exactly what the simulation produced."""

import numpy as np
import pytest

from thin_film import fab
from thin_film import trench as tr
from thin_film.plotting import trench_image
from thin_film.recording import final_metrics, record_1d, record_trench, trench_frame


@pytest.mark.parametrize("model", ["random_deposition", "surface_relaxation", "thermal"])
def test_1d_recording_conserves_atoms(model):
    rec = record_1d(model, 50, 20, 10, seed=1, epsilon=3.0, hops=5.0)
    assert rec["heights"].shape == (11, 50)
    assert rec["heights"][0].sum() == 0
    assert np.array_equal(rec["heights"].sum(axis=1), np.arange(11) * 100)
    assert rec["rq"][-1] == pytest.approx(rec["heights"][-1].std())


@pytest.mark.parametrize("temperature", [280.0, 420.0])
def test_trench_frames_rebuild_exactly(temperature):
    s = tr.TrenchSettings(temperature=temperature, trench_width=10, n_monolayers=12)
    n_frames = 15
    rec = record_trench(s, n_frames)
    # replay the same run step by step and compare every frame
    grid = tr.make_trench(s)
    eps, hops = s.thermal()
    per_frame = int(round(s.n_monolayers * s.width / n_frames))
    x0, x1 = s.trench_x
    lo = max(0, x0 - 30)
    rows, cols = rec["arrival"].shape
    for k in range(1, n_frames + 1):
        tr.deposit(grid, per_frame, eps, hops, s.seed * 1000 + k)
        expected = trench_image(grid, tr.find_voids(grid))[:rows, lo:lo + cols]
        assert np.array_equal(trench_frame(rec, k), expected), f"frame {k}"


def test_trench_final_metrics_match_measure():
    s = tr.TrenchSettings(temperature=300.0, n_monolayers=10)
    rec = record_trench(s, 8)
    m = final_metrics(rec)
    assert m["trench_void_area"] == rec["trench_void_area"][-1]
    assert fab.spec_check("trench", m) in {"PASS", "FAIL"}


def test_unit_conversions_and_specs():
    assert fab.kelvin_to_celsius(273.15) == pytest.approx(0)
    assert fab.rate_nm_per_min(1.0) == pytest.approx(18.0)
    assert fab.recipe_name("trench", 420, 1.0, 24, (16, 16)) == "PM4-TRF-420K-1ML/s-W16D16"
    assert fab.spec_check("thermal", {"rq": 1.9}) == "PASS"
    assert fab.spec_check("thermal", {"rq": 2.1}) == "FAIL"
    assert fab.spec_check("trench", {"trench_void_area": 0, "step_coverage": 0.6}) == "PASS"
    assert fab.spec_check("trench", {"trench_void_area": 3, "step_coverage": 0.6}) == "FAIL"
    assert fab.spec_check("random_deposition", {"rq": 9.0}) == "MONITOR"
