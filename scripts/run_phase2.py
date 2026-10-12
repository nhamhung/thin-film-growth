"""Phase 2: surface relaxation. Growth exponent, saturation vs substrate size, Numba speed-up."""

import time
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thin_film import random_deposition as rd  # noqa: E402
from thin_film import surface_relaxation as sr  # noqa: E402
from thin_film.plotting import plot_roughness_loglog  # noqa: E402
from thin_film.results import save_result  # noqa: E402
from thin_film.roughness import fit_growth_exponent  # noqa: E402

SIZES = [25, 50, 100, 200]
EARLY_ML = 100  # fit beta before saturation sets in (t_x ~ L^2 is far beyond 100 ML for L = 200)


def scaling_experiment():
    base = sr.RelaxSettings()
    curves = {L: sr.run_ensemble(sr.RelaxSettings(**{**asdict(base), "n_columns": L})) for L in SIZES}
    table = pd.concat([df.assign(n_columns=L) for L, df in curves.items()], ignore_index=True)

    early = curves[200][curves[200]["monolayers"] <= EARLY_ML]
    beta, beta_err = fit_growth_exponent(early["monolayers"], early["roughness_mean"])
    w_sat = {L: sr.saturation_width(df) for L, df in curves.items()}
    alpha, alpha_err = fit_growth_exponent(np.array(SIZES), np.array([w_sat[L] for L in SIZES]))

    rd_curve = rd.run_ensemble(rd.RDSettings(n_monolayers=base.n_monolayers, n_runs=base.n_runs))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    plot_roughness_loglog({"random deposition": rd_curve, "with relaxation (L=200)": curves[200]},
                          reference_slope=0.25, ax=axes[0])
    axes[0].set_title(f"Relaxation slows roughening: beta = {beta:.3f} (theory 0.25)")
    for L, df in curves.items():
        axes[1].loglog(df["monolayers"], df["roughness_mean"], "o-", ms=3, label=f"L = {L}")
    axes[1].set(xlabel="monolayers", ylabel="W", title="Roughness saturates; plateau rises with L")
    axes[1].legend()
    for L, df in curves.items():  # Family-Vicsek collapse: W / L^alpha vs t / L^z, alpha = 1/2, z = 2
        axes[2].loglog(df["monolayers"] / L**2, df["roughness_mean"] / L**0.5, "o", ms=3, label=f"L = {L}")
    axes[2].set(xlabel="t / L^2", ylabel="W / L^0.5", title="Data collapse (alpha = 1/2, z = 2)")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)

    summary = {"beta": beta, "beta_std_error": beta_err, "beta_expected": 0.25, "beta_fit_window_ml": EARLY_ML,
               "w_sat": {str(L): w for L, w in w_sat.items()}, "alpha": alpha, "alpha_std_error": alpha_err,
               "alpha_expected": 0.5}
    out = save_result("phase2_surface_relaxation", {**base.to_dict(), "n_columns": SIZES}, table, summary, fig)
    print(f"beta = {beta:.3f} +/- {beta_err:.3f}, alpha = {alpha:.3f} +/- {alpha_err:.3f} -> {out}")


def timing_experiment(n_atoms: int = 200_000, repeats: int = 3):
    rng = np.random.default_rng(0)
    cols, coins = rng.integers(200, size=n_atoms), rng.random(n_atoms)
    sr.relax_deposit_numba(np.zeros(200, np.int64), cols[:10], coins[:10])  # compile before timing
    rows = []
    for name, kernel in [("python", sr.relax_deposit_python), ("numba", sr.relax_deposit_numba)]:
        best = min(_time(kernel, cols, coins) for _ in range(repeats))
        rows.append({"implementation": name, "atoms": n_atoms, "seconds": best,
                     "atoms_per_second": n_atoms / best})
    df = pd.DataFrame(rows)
    speedup = df["seconds"].iloc[0] / df["seconds"].iloc[1]
    out = save_result("phase2_numba_timing", {"n_columns": 200, "n_atoms": n_atoms, "repeats": repeats,
                                              "timing": "best of repeats, compile excluded"},
                      df, {"speedup": speedup})
    print(df.to_string(index=False), f"\nspeed-up = {speedup:.0f}x -> {out}")


def _time(kernel, cols, coins):
    heights = np.zeros(200, np.int64)
    t0 = time.perf_counter()
    kernel(heights, cols, coins)
    return time.perf_counter() - t0


if __name__ == "__main__":
    scaling_experiment()
    timing_experiment()
