"""Phase 2: random deposition with surface relaxation (1D).

The physics in plain language: an atom still lands in a random column, but
before it settles it looks at the two neighbouring columns. If a neighbour is
lower, the atom rolls down into it (if both neighbours are lower it picks one
at random). This is the simplest form of surface diffusion: atoms prefer to
fill dips instead of piling up on peaks.

That one local rule changes the large-scale behaviour completely:
  * roughness grows more slowly, W ~ t^(1/4) instead of t^(1/2)
    (the Edwards-Wilkinson universality class);
  * roughness stops growing (saturates) once height information has spread
    across the whole substrate. The saturated width grows with substrate
    size, W_sat ~ L^(1/2), and the time to reach it grows as L^2.

Periodic boundaries: column N-1's right neighbour is column 0, so the
substrate behaves like a ring with no edges (no artificial wall effects).

Speed: unlike Phase 1, each atom's final column depends on the current
surface, so atoms must be processed one at a time in order. A Python loop over
millions of atoms is slow, so the same loop is compiled to machine code with
Numba's @njit, which typically gives a 50-200x speed-up.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numba import njit

from .roughness import ensemble_roughness, measurement_times, roughness


@dataclass
class RelaxSettings:
    """Everything needed to reproduce a Phase 2 run. Saved next to every result."""

    n_columns: int = 200
    n_monolayers: float = 20000  # long enough to see saturation for L <= 200
    n_runs: int = 20
    n_points: int = 45
    seed: int = 42

    def to_dict(self) -> dict:
        return {"model": "surface_relaxation", **asdict(self)}


def relax_deposit_python(heights: np.ndarray, columns: np.ndarray, coins: np.ndarray) -> None:
    """Deposit one atom per entry of `columns`, letting each roll to a lower neighbour.

    heights : column heights, modified in place
    columns : landing column of each atom (random integers in [0, N))
    coins   : one uniform random number per atom, used only to break the tie
              when both neighbours are lower

    The random numbers are passed in rather than drawn inside the loop so that
    the plain-Python and Numba versions get exactly the same inputs and must
    produce exactly the same film (a strong correctness check).
    """
    n = heights.shape[0]
    for k in range(columns.shape[0]):
        i = columns[k]
        left = i - 1 if i > 0 else n - 1      # periodic boundary: (i - 1) % N
        right = i + 1 if i < n - 1 else 0     # periodic boundary: (i + 1) % N
        here = heights[i]
        left_lower = heights[left] < here
        right_lower = heights[right] < here
        if left_lower and right_lower:
            target = left if coins[k] < 0.5 else right
        elif left_lower:
            target = left
        elif right_lower:
            target = right
        else:
            target = i                         # no lower neighbour: stay put
        heights[target] += 1


# Same function, compiled. cache=True stores the machine code on disk so later
# imports skip the ~1 s compile step.
relax_deposit_numba = njit(cache=True)(relax_deposit_python)


def deposit(heights: np.ndarray, n_atoms: int, rng: np.random.Generator, kernel=relax_deposit_numba,
            chunk: int = 1_000_000) -> None:
    """Deposit n_atoms with relaxation, drawing random numbers in memory-friendly chunks."""
    n = len(heights)
    while n_atoms > 0:
        m = min(chunk, n_atoms)
        kernel(heights, rng.integers(n, size=m), rng.random(m))
        n_atoms -= m


def simulate(n_columns: int, times: np.ndarray, rng: np.random.Generator, kernel=relax_deposit_numba) -> np.ndarray:
    """Grow one film and return its roughness at each atom count in `times`."""
    heights = np.zeros(n_columns, dtype=np.int64)
    widths = np.empty(len(times))
    deposited = 0
    for k, target in enumerate(times):
        deposit(heights, int(target - deposited), rng, kernel)
        deposited = target
        widths[k] = roughness(heights)
    return widths


def run_ensemble(settings: RelaxSettings) -> pd.DataFrame:
    """Average n_runs independent films (one row per measurement time)."""
    times = measurement_times(settings.n_columns, settings.n_monolayers, settings.n_points)
    return ensemble_roughness(lambda t, rng: simulate(settings.n_columns, t, rng),
                              times, settings.n_columns, settings.n_runs, settings.seed)


def saturation_width(df: pd.DataFrame, tail_fraction: float = 0.2) -> float:
    """Saturated roughness: the mean of W over the last `tail_fraction` of log-time points."""
    n_tail = max(1, int(len(df) * tail_fraction))
    return float(df["roughness_mean"].iloc[-n_tail:].mean())
