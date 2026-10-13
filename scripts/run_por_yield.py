"""Single-wafer pass rates of the POR recipes (and one off-POR recipe each) over many seeds.

Specs apply to individual wafers, so the question is "what fraction of wafers
made with this recipe pass?", not "does the average pass?". This script runs
each recipe with N_SEEDS different seeds exactly as the app does (60 frames)
and saves every wafer's result to results/por_yield/.
"""

import pandas as pd

from thin_film import fab
from thin_film import trench as tr
from thin_film.ideal import IDEAL
from thin_film.recording import final_metrics, record_1d
from thin_film.results import save_result
from thin_film.units import thermal_parameters

N_SEEDS = 30
N_FRAMES = 60
RECIPES = [
    ("thermal", "POR", 375.0, 0.01),
    ("thermal", "375 K, 1 ML/s", 375.0, 1.0),
    ("thermal", "450 K, 1 ML/s", 450.0, 1.0),
    ("trench", "POR", 420.0, 1.0),
    ("trench", "460 K, 10 ML/s", 460.0, 10.0),
    ("trench", "370 K, 1 ML/s", 370.0, 1.0),
]


def main():
    rows = []
    for module, label, temperature, flux in RECIPES:
        for seed in range(N_SEEDS):
            if module == "thermal":
                eps, hops = thermal_parameters(temperature, 0.2, 0.8, flux, max_hops=1e4)
                m = final_metrics(record_1d("thermal", 200, IDEAL["thermal"]["settings"]["n_monolayers"],
                                            N_FRAMES, seed, eps, hops))
            else:
                p = IDEAL["trench"]["settings"]
                s = tr.TrenchSettings(trench_width=p["trench_width"], trench_depth=p["trench_depth"],
                                      n_monolayers=p["n_monolayers"], temperature=temperature, flux=flux, seed=seed)
                m = tr.measure(tr.grow(s), s)
            rows.append({"module": module, "recipe": label, "temperature": temperature, "flux": flux, "seed": seed,
                         **{k: v for k, v in m.items() if k in ("rq", "trench_void_area", "step_coverage")},
                         "result": fab.spec_check(module, m)})
    df = pd.DataFrame(rows)
    summary = (df.assign(passed=df["result"] == "PASS").groupby(["module", "recipe"], sort=False)["passed"]
               .agg(["sum", "count"]).reset_index())
    save_result("por_yield", {"n_seeds": N_SEEDS, "n_frames": N_FRAMES, "recipes": RECIPES,
                              "specs": {k: v for k, v in fab.SPECS.items() if v}},
                df, {"pass_counts": summary.to_dict(orient="records")})
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
