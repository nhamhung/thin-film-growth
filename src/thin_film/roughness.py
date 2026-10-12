"""Shared surface measurements used by every 1D growth model.

A 1D film is stored as an array of column heights: heights[i] is how many
atoms are stacked in column i. Everything here works on that array.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def roughness(heights: np.ndarray) -> float:
    """Interface width W: how far a typical column sits from the average height.

    W = sqrt( mean( (h_i - mean(h))^2 ) ), i.e. the population standard
    deviation of the heights. A perfectly flat film has W = 0; a jagged film
    has large W. This is the single number we track to call a film "rough".
    """
    return float(np.std(heights))


def power_law(t, amplitude, beta):
    """W(t) = amplitude * t^beta, the scaling law rough surfaces follow while growing.

    beta is the "growth exponent". Random deposition predicts beta = 1/2;
    models with surface diffusion predict smaller values.
    """
    return amplitude * np.power(t, beta)


def fit_growth_exponent(times: np.ndarray, widths: np.ndarray) -> tuple[float, float]:
    """Fit W = A * t^beta and return (beta, beta_std_error).

    The fit is done on log W = log A + beta * log t, which is a straight line,
    so beta is just the slope of the log-log plot. Fitting in log space weights
    early and late times evenly (a fit on raw W would be dominated by the
    largest, latest values). Points with W = 0 are dropped since log(0) is
    undefined.
    """
    times = np.asarray(times, dtype=float)
    widths = np.asarray(widths, dtype=float)
    keep = (times > 0) & (widths > 0)
    log_t, log_w = np.log(times[keep]), np.log(widths[keep])

    def line(x, log_amplitude, beta):
        return log_amplitude + beta * x

    (_, beta), cov = curve_fit(line, log_t, log_w)
    return float(beta), float(np.sqrt(cov[1, 1]))


def measurement_times(n_columns: int, n_monolayers: float, n_points: int) -> np.ndarray:
    """Atom counts at which to record roughness, evenly spaced on a log scale.

    Log spacing (e.g. 1, 2, 5, 10, 20, 50 ... ML) gives equally spaced points on
    the log-log plot we use to read off the growth exponent. Starts at 1 ML
    because below that the surface is mostly bare substrate and still noisy.
    """
    monolayers = np.logspace(0, np.log10(n_monolayers), n_points)
    return np.unique(np.round(monolayers * n_columns).astype(np.int64))


def ensemble_roughness(simulate_one, times: np.ndarray, n_columns: int, n_runs: int,
                       seed: int) -> pd.DataFrame:
    """Run `simulate_one(times, rng)` n_runs times and average the roughness curves.

    One film's W(t) curve is noisy; averaging many films gives the smooth
    "expected" curve that theory talks about. Each run gets its own
    statistically independent random stream spawned from the master seed, so
    the whole ensemble is reproducible from a single number.
    """
    streams = np.random.SeedSequence(seed).spawn(n_runs)
    curves = np.array([simulate_one(times, np.random.default_rng(s)) for s in streams])
    return pd.DataFrame({
        "atoms": times,
        "monolayers": times / n_columns,
        "roughness_mean": curves.mean(axis=0),
        "roughness_sem": curves.std(axis=0, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0.0,
    })
