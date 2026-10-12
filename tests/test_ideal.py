"""The recommended settings must actually beat ordinary settings."""

from thin_film import thermal as th
from thin_film import trench as tr
from thin_film.ideal import IDEAL


def test_thermal_ideal_is_smoother_than_default():
    s = IDEAL["thermal"]["settings"]
    common = dict(n_columns=200, n_monolayers=50, n_runs=4, n_points=5)
    ideal = th.run_ensemble(th.ThermalSettings(temperature=s["temperature"], flux=s["flux"], **common)).iloc[-1]
    default = th.run_ensemble(th.ThermalSettings(**common)).iloc[-1]   # 350 K, 1 ML/s
    assert ideal["roughness_mean"] < 0.5 * default["roughness_mean"]


def test_trench_ideal_is_void_free():
    s = IDEAL["trench"]["settings"]
    r = tr.run_ensemble(tr.TrenchSettings(**{k: s[k] for k in ("temperature", "flux", "trench_width",
                                                                 "trench_depth", "n_monolayers")}, n_runs=8))
    assert r["trench_void_area"] < 1 and r["step_coverage"] > 0.5
