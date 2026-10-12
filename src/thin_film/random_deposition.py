"""Phase 1: random deposition in 1D, the simplest possible film-growth model.

The physics in plain language: atoms rain straight down onto a substrate made
of N side-by-side columns. Each atom picks a column completely at random and
sticks wherever it lands, on top of that column. It never moves sideways and
never cares about its neighbours. Real films are not like this (atoms do slide
around, see Phase 2/3), but this model is useful as a baseline because it can
be solved exactly, so it lets us check that the simulation code and
roughness measurement are correct.

Exact result: after t atoms, each column's height is Binomial(t, 1/N), so
    W(t)^2 = (t/N) * (1 - 1/N)   ->   W grows like sqrt(time), beta = 1/2,
and it never saturates, because columns never talk to each other.

Time is measured in monolayers (ML): 1 ML = N atoms = on average one layer of
atoms over the whole surface.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .roughness import ensemble_roughness, measurement_times, roughness

__all__ = ["RDSettings", "deposit_one", "deposit", "measurement_times", "simulate", "run_ensemble",
           "theory_roughness"]


@dataclass
class RDSettings:
    """Everything needed to reproduce a Phase 1 run. Saved next to every result."""

    n_columns: int = 200        # substrate width N
    n_monolayers: float = 1000  # how long to grow, in layers
    n_runs: int = 30            # independent films averaged together
    n_points: int = 40          # log-spaced times at which roughness is recorded
    seed: int = 42              # master random seed

    def to_dict(self) -> dict:
        return {"model": "random_deposition", **asdict(self)}


def deposit_one(heights: np.ndarray, rng: np.random.Generator) -> None:
    """Drop a single atom: pick a random column and add 1 to its height.

    This is the literal rule of the model. `deposit` below does the same thing
    in bulk for speed; this function is kept so the rule is visible and testable.
    """
    column = rng.integers(len(heights))
    heights[column] += 1


def deposit(heights: np.ndarray, n_atoms: int, rng: np.random.Generator) -> None:
    """Drop n_atoms at once (same statistics as n_atoms calls to deposit_one).

    Shortcut: because atoms in this model never interact, the order they arrive
    in does not matter. So instead of looping atom by atom, we draw all the
    random columns at once and count how many landed in each (np.bincount).
    Phase 2 cannot use this trick, since there each atom's move depends on the
    surface it lands on, which is why Numba comes in there.
    """
    heights += np.bincount(rng.integers(len(heights), size=n_atoms), minlength=len(heights))


def simulate(n_columns: int, times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Grow one film and return its roughness at each atom count in `times`."""
    heights = np.zeros(n_columns, dtype=np.int64)
    widths = np.empty(len(times))
    deposited = 0
    for k, target in enumerate(times):
        deposit(heights, target - deposited, rng)
        deposited = target
        widths[k] = roughness(heights)
    return widths


def run_ensemble(settings: RDSettings) -> pd.DataFrame:
    """Average `n_runs` independent films and add the exact theory for comparison.

    Returns one row per measurement time with the mean roughness, its standard
    error across runs, and the exact theoretical value.
    """
    times = measurement_times(settings.n_columns, settings.n_monolayers, settings.n_points)
    df = ensemble_roughness(lambda t, rng: simulate(settings.n_columns, t, rng),
                            times, settings.n_columns, settings.n_runs, settings.seed)
    df["roughness_theory"] = theory_roughness(df["monolayers"], settings.n_columns)
    return df


def theory_roughness(monolayers: np.ndarray, n_columns: int) -> np.ndarray:
    """Exact expected width sqrt(theta * (1 - 1/N)) for theta monolayers.

    (Strictly this is the root-mean-square width; the ensemble mean of W is a
    hair below it, but the difference is negligible for N = 200.)
    """
    return np.sqrt(np.asarray(monolayers) * (1 - 1 / n_columns))
