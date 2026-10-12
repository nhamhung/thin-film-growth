"""Phase 4: trench filling. Step coverage and voids vs temperature and trench width, plus animations."""

from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

from thin_film import trench as tr  # noqa: E402
from thin_film.ideal import IDEAL  # noqa: E402
from thin_film.plotting import animate_trench, plot_heatmap, plot_sweep, plot_trench, trench_image  # noqa: E402
from thin_film.results import RESULTS_DIR, save_result  # noqa: E402

TEMPERATURES = np.arange(280, 521, 30)
WIDTHS = [4, 6, 8, 12, 16, 20, 24, 32]
WIDTH_SWEEP_TEMPS = [370.0, 450.0]
GALLERY_TEMPS = [280.0, 370.0, 430.0, 500.0]


def main():
    base = tr.TrenchSettings()

    # 1. Temperature sweep at fixed trench width
    t_df = tr.sweep(base, "temperature", TEMPERATURES)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    plot_sweep(t_df, "temperature", "step_coverage", "step_coverage_sem", ax=axes[0],
               xlabel="temperature (K)", ylabel="step coverage (bottom / top)")
    plot_sweep(t_df, "temperature", "trench_void_area", "trench_void_area_sem", ax=axes[1],
               xlabel="temperature (K)", ylabel="void area in trench (cells)")
    plot_sweep(t_df, "temperature", "film_density", ax=axes[2], xlabel="temperature (K)",
               ylabel="film density (field)")
    fig.suptitle(f"Trench {base.trench_width} wide x {base.trench_depth} deep, flux {base.flux:g} ML/s")
    save_result("phase4_temperature_sweep", {**base.to_dict(), "temperatures": TEMPERATURES.tolist()},
                t_df, figure=fig)
    plt.close(fig)

    # 2. Trench-width sweep at two temperatures
    frames = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for temp in WIDTH_SWEEP_TEMPS:
        s = tr.TrenchSettings(**{**asdict(base), "temperature": temp})
        frames.append(tr.sweep(s, "trench_width", WIDTHS).assign(temperature=temp))
        plot_sweep(frames[-1], "trench_width", "step_coverage", "step_coverage_sem", ax=axes[0],
                   label=f"{temp:g} K", xlabel="trench width (cells)", ylabel="step coverage")
        plot_sweep(frames[-1], "trench_width", "void_probability", ax=axes[1], label=f"{temp:g} K",
                   xlabel="trench width (cells)", ylabel="fraction of runs with a trench void")
    for ax in axes:
        ax.legend()
    w_df = pd.concat(frames, ignore_index=True)
    save_result("phase4_width_sweep", {**base.to_dict(), "trench_widths": WIDTHS,
                                       "temperatures": WIDTH_SWEEP_TEMPS}, w_df, figure=fig)
    plt.close(fig)

    # 3. Gallery of cross-sections
    fig, axes = plt.subplots(1, len(GALLERY_TEMPS), figsize=(4.2 * len(GALLERY_TEMPS), 3.6))
    for ax, temp in zip(axes, GALLERY_TEMPS):
        s = tr.TrenchSettings(**{**asdict(base), "temperature": temp})
        g = tr.grow(s)
        m = tr.measure(g, s)
        x0, x1 = s.trench_x
        plot_trench(g[:, x0 - 24: x1 + 24], tr.find_voids(g)[:, x0 - 24: x1 + 24], ax=ax,
                    title=f"{temp:g} K: coverage {m['step_coverage']:.2f}, void {m['trench_void_area']}")
    fig.tight_layout()
    save_result("phase4_cross_sections", {**base.to_dict(), "temperatures": GALLERY_TEMPS}, figure=fig)
    plt.close(fig)

    # 4. Growth animations (GIF), cold and hot
    for temp in (280.0, 470.0):
        s = tr.TrenchSettings(**{**asdict(base), "temperature": temp})
        anim = animate_trench(growth_frames(s))
        anim.save(RESULTS_DIR / "phase4_cross_sections" / f"growth_{temp:g}K.gif", writer=PillowWriter(fps=8))
        plt.close("all")

    print(t_df[["temperature", "hops_per_atom", "step_coverage", "trench_void_area", "film_density"]]
          .round(3).to_string(index=False))
    print(w_df[["temperature", "trench_width", "step_coverage", "trench_void_area", "void_probability"]]
          .round(3).to_string(index=False))


def process_window():
    """Grid search over temperature x deposition rate x trench width; basis for the app's ideal config."""
    widths, temps, fluxes = [8, 12, 16], list(range(380, 521, 20)), [0.1, 1.0, 10.0, 100.0]
    base = tr.TrenchSettings(n_runs=8)
    rows = []
    for w in widths:
        for t in temps:
            for f in fluxes:
                s = tr.TrenchSettings(**{**asdict(base), "trench_width": w, "temperature": float(t), "flux": f})
                eps, hops = s.thermal()
                rows.append({"trench_width": w, "temperature": t, "flux": f, "epsilon": eps,
                             "hops_per_atom": hops, **tr.run_ensemble(s)})
    df = pd.DataFrame(rows)
    ideal = IDEAL["trench"]["settings"]
    fig, axes = plt.subplots(2, len(widths), figsize=(6 * len(widths), 10))
    for k, w in enumerate(widths):
        d = df[df["trench_width"] == w]
        hl = (ideal["temperature"], ideal["flux"]) if w == ideal["trench_width"] else None
        plot_heatmap(d.pivot(index="temperature", columns="flux", values="trench_void_area"), ax=axes[0, k],
                     title=f"width {w}: void area (cells)", fmt="{:.0f}", cmap="Reds", highlight=hl)
        plot_heatmap(d.pivot(index="temperature", columns="flux", values="step_coverage"), ax=axes[1, k],
                     title=f"width {w}: step coverage", highlight=hl)
    fig.suptitle(f"Trench process window (depth {base.trench_depth}); red box = recommended setting")
    fig.tight_layout()
    out = save_result("phase4_process_window", {**base.to_dict(), "trench_widths": widths,
                                                "temperatures": temps, "fluxes": fluxes},
                      df, {"recommended": ideal}, fig)
    plt.close(fig)
    best = df.sort_values(["void_probability", "trench_void_area", "step_coverage"],
                          ascending=[True, True, False]).groupby("trench_width").head(3)
    print(best[["trench_width", "temperature", "flux", "trench_void_area", "void_probability", "step_coverage"]]
          .round(3).to_string(index=False), f"\n-> {out}")


def growth_frames(s: tr.TrenchSettings, n_frames: int = 40) -> list[np.ndarray]:
    """Snapshots of one growing film, voids marked, cropped around the trench."""
    grid = tr.make_trench(s)
    eps, hops = s.thermal()
    per_frame = int(s.n_monolayers * s.width / n_frames)
    x0, x1 = s.trench_x
    frames = []
    for k in range(n_frames):
        tr.deposit(grid, per_frame, eps, hops, s.seed + k)
        frames.append(trench_image(grid, tr.find_voids(grid))[:, x0 - 24: x1 + 24])
    return frames


if __name__ == "__main__":
    main()
    process_window()
