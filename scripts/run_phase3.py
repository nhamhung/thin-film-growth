"""Phase 3: temperature-activated hopping. Roughness vs temperature, deposition rate and epsilon."""

from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thin_film import thermal as th  # noqa: E402
from thin_film.ideal import IDEAL  # noqa: E402
from thin_film.plotting import plot_heatmap, plot_surface, plot_sweep  # noqa: E402
from thin_film.results import save_result  # noqa: E402

TEMPERATURES = np.linspace(250, 475, 10)
FLUXES = [0.01, 0.1, 1, 10, 100, 1000]
FLUX_SWEEP_T = 375.0
EPSILONS = np.linspace(0, 8, 9)
EPSILON_SWEEP_HOPS = 100.0


def main():
    base = th.ThermalSettings()

    # 1. Dimensionless "single mobility parameter" version: vary epsilon only
    eps_df = th.sweep_epsilon(base, EPSILONS, EPSILON_SWEEP_HOPS)
    ax = plot_sweep(eps_df, "epsilon", "roughness", "roughness_sem",
                    xlabel="epsilon = E_bond / kT", ylabel=f"roughness after {base.n_monolayers:g} ML")
    ax.set_title(f"Dimensionless model, {EPSILON_SWEEP_HOPS:g} hops per landing")
    save_result("phase3_epsilon_sweep", {**base.to_dict(), "epsilons": EPSILONS.tolist(),
                                         "hops_per_landing": EPSILON_SWEEP_HOPS}, eps_df, figure=ax.figure)
    plt.close(ax.figure)

    # 2. Real units: temperature sweep at fixed flux
    t_df = th.sweep_temperature(base, TEMPERATURES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_sweep(t_df, "temperature", "roughness", "roughness_sem", ax=axes[0],
               xlabel="substrate temperature (K)", ylabel=f"roughness after {base.n_monolayers:g} ML")
    axes[0].set_title(f"Hotter substrate -> smoother film (flux {base.flux:g} ML/s)")
    plot_sweep(t_df, "temperature", "hops_per_landing", ax=axes[1], xlabel="substrate temperature (K)",
               ylabel="hop attempts per landing")
    axes[1].set_yscale("log")
    axes[1].set_title("Arrhenius: hops = nu0 exp(-E_diff/kT) / F")
    save_result("phase3_temperature_sweep", {**base.to_dict(), "temperatures": TEMPERATURES.tolist()},
                t_df, figure=fig)
    plt.close(fig)

    # 3. Real units: deposition-rate sweep at fixed temperature
    s = th.ThermalSettings(**{**asdict(base), "temperature": FLUX_SWEEP_T})
    f_df = th.sweep_flux(s, FLUXES)
    ax = plot_sweep(f_df, "flux", "roughness", "roughness_sem", logx=True,
                    xlabel="deposition rate (ML/s)", ylabel=f"roughness after {base.n_monolayers:g} ML")
    ax.set_title(f"Faster deposition -> rougher film (T = {FLUX_SWEEP_T:g} K)")
    save_result("phase3_flux_sweep", {**s.to_dict(), "fluxes": FLUXES}, f_df, figure=ax.figure)
    plt.close(ax.figure)

    # 4. What the films look like: cold vs hot profile
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, temp in zip(axes, (275.0, 425.0)):
        eps, hops = th.ThermalSettings(**{**asdict(base), "temperature": temp}).thermal()
        h = np.zeros(base.n_columns, np.int64)
        th.thermal_grow(h, int(base.n_monolayers * base.n_columns), eps, hops, base.seed)
        plot_surface(h, ax=ax, title=f"T = {temp:g} K (epsilon = {eps:.1f}, {hops:.3g} hops/landing)")
    fig.tight_layout()
    save_result("phase3_profiles", base.to_dict() | {"temperatures": [275, 425]}, figure=fig)

    print(t_df[["temperature", "epsilon", "hops_per_landing", "roughness"]].round(3).to_string(index=False))
    print(f_df[["flux", "hops_per_landing", "roughness"]].round(3).to_string(index=False))


def process_window():
    """Grid search over temperature x deposition rate; basis for the app's ideal config."""
    temps, fluxes = list(range(300, 551, 25)), [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    base = th.ThermalSettings(n_runs=4, n_points=5)
    rows = []
    for t in temps:
        for f in fluxes:
            s = th.ThermalSettings(**{**asdict(base), "temperature": float(t), "flux": f})
            eps, hops = s.thermal()
            last = th.run_ensemble(s).iloc[-1]
            rows.append({"temperature": t, "flux": f, "epsilon": eps, "hops_per_landing": hops,
                         "roughness": last["roughness_mean"], "roughness_sem": last["roughness_sem"]})
    df = pd.DataFrame(rows)
    ideal = IDEAL["thermal"]["settings"]
    ax = plot_heatmap(df.pivot(index="temperature", columns="flux", values="roughness"), cmap="viridis_r",
                      title=f"Roughness after {base.n_monolayers:g} ML (red box = recommended)",
                      highlight=(ideal["temperature"], ideal["flux"]))
    ax.figure.set_size_inches(8, 6)
    out = save_result("phase3_process_window", {**base.to_dict(), "temperatures": temps, "fluxes": fluxes},
                      df, {"recommended": ideal}, ax.figure)
    plt.close(ax.figure)
    print(df.sort_values("roughness").head(5).round(3).to_string(index=False), f"\n-> {out}")


if __name__ == "__main__":
    main()
    process_window()
