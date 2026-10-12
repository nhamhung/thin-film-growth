"""Streamlit app: loads, every process module runs, POR loading, comparison and slider persistence work."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from thin_film import fab
from thin_film.ideal import IDEAL

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
MODULES = ["random_deposition", "surface_relaxation", "thermal", "trench"]


def fresh(module: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.selectbox(key="module").set_value(module).run()
    at.slider(key="n_frames").set_value(20).run()
    return at


def test_app_loads_idle():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.session_state["last_run"] is None


@pytest.mark.parametrize("module", MODULES)
def test_each_module_processes_a_wafer(module):
    at = fresh(module)
    at.button(key="run_button").click().run()
    assert not at.exception
    row = at.session_state["history"][-1]
    assert row["wafer"] == "TF-0001-W01" and row["result"] in {"PASS", "FAIL", "MONITOR"}
    assert len(at.session_state["last_run"]["wafers"]) == 1


def test_por_button_sets_thermal_recipe():
    at = fresh("thermal")
    at.button(key="por_button").click().run()
    s = IDEAL["thermal"]["settings"]
    assert at.slider(key="temperature").value == s["temperature"]
    assert at.slider(key="log_flux").value == pytest.approx(np.log10(s["flux"]))


def test_compare_runs_recipe_and_por_side_by_side():
    at = fresh("trench")
    at.slider(key="temperature").set_value(300).run()
    at.button(key="split_button").click().run()
    assert not at.exception
    rows = at.session_state["history"]
    assert [r["label"] for r in rows] == ["RECIPE", "POR"]
    s = IDEAL["trench"]["settings"]
    assert rows[1]["recipe"] == fab.recipe_name("trench", s["temperature"], s["flux"], s["n_monolayers"],
                                                (s["trench_width"], s["trench_depth"]))
    wafers = at.session_state["last_run"]["wafers"]
    assert len(wafers) == 2 and wafers[0]["n_frames"] == wafers[1]["n_frames"]


def test_trench_sliders_survive_switching_module():
    at = fresh("trench")
    at.slider(key="trench_width").set_value(20).run()
    at.selectbox(key="module").set_value("thermal").run()
    at.selectbox(key="module").set_value("trench").run()
    assert at.slider(key="trench_width").value == 20
