"""Phase 1: run random deposition, fit the growth exponent, save results/phase1_random_deposition/."""

import matplotlib

matplotlib.use("Agg")

from thin_film import random_deposition as rd  # noqa: E402
from thin_film.plotting import plot_roughness_loglog  # noqa: E402
from thin_film.results import save_result  # noqa: E402
from thin_film.roughness import fit_growth_exponent  # noqa: E402


def main():
    settings = rd.RDSettings()
    df = rd.run_ensemble(settings)
    beta, beta_err = fit_growth_exponent(df["monolayers"], df["roughness_mean"])

    ax = plot_roughness_loglog({"random deposition": df}, {"random deposition": beta})
    ax.plot(df["monolayers"], df["roughness_theory"], "r-", lw=1, alpha=0.6, label="exact theory")
    ax.legend()

    out = save_result("phase1_random_deposition", settings.to_dict(), df,
                      summary={"beta": beta, "beta_std_error": beta_err, "beta_expected": 0.5},
                      figure=ax.figure)
    print(f"beta = {beta:.4f} +/- {beta_err:.4f} (expected 0.5); saved to {out}")


if __name__ == "__main__":
    main()
