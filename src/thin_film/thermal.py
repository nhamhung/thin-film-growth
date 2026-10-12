"""Phase 3: temperature-activated surface hopping (1D solid-on-solid model).

The physics in plain language: atoms land in random columns as before. Between
landings, surface atoms try to hop sideways. Whether a hop succeeds depends on
how tightly the atom is bound:

    P(hop) = exp(-E / kT),   E = n * E_bond,

where n is the number of side neighbours (0, 1 or 2) at the atom's own height.
An isolated atom on a flat terrace (n = 0) hops freely; an atom tucked against
a step (n = 1) or sitting in a groove (n = 2) is held more strongly. The bond
to the atom below is the same for every surface atom, so it is folded into the
base hop rate and does not appear here.

Two knobs (see units.py for how real temperatures and fluxes map onto them):
  epsilon = E_bond / kT : "how sticky are bonds" (the single mobility parameter)
  hops                  : hop attempts per landing (deposition rate control:
                          slow deposition = many hops between landings)

Each hop attempt picks a random column, tries to move its top atom one column
left or right (chosen at random), and succeeds with probability exp(-n*epsilon).
Periodic boundaries as in Phase 2.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numba import njit

from .roughness import ensemble_roughness, measurement_times, roughness
from .units import thermal_parameters


@njit(cache=True)
def thermal_grow(heights, n_landings, epsilon, hops_per_landing, seed):
    """Deposit n_landings atoms with `hops_per_landing` hop attempts after each (in place).

    hops_per_landing may be fractional: e.g. 2.3 means 2 attempts, plus a 3rd
    with probability 0.3, so the average is exactly 2.3.
    Uses Numba's internal random generator, seeded here for reproducibility.
    """
    np.random.seed(seed)
    n = heights.shape[0]
    p_hop = np.exp(-epsilon * np.arange(3.0))   # P(hop) for 0, 1, 2 neighbours
    whole = int(hops_per_landing)
    frac = hops_per_landing - whole
    for _ in range(n_landings):
        heights[np.random.randint(n)] += 1       # a new atom arrives
        n_hops = whole + (1 if np.random.random() < frac else 0)
        for _ in range(n_hops):
            i = np.random.randint(n)
            h = heights[i]
            if h == 0:
                continue                         # bare substrate: nothing to move
            left = i - 1 if i > 0 else n - 1
            right = i + 1 if i < n - 1 else 0
            neighbours = (heights[left] >= h) + (heights[right] >= h)
            if np.random.random() < p_hop[neighbours]:
                j = left if np.random.random() < 0.5 else right
                heights[i] -= 1
                heights[j] += 1


@dataclass
class ThermalSettings:
    """Everything needed to reproduce a Phase 3 run. Saved next to every result."""

    n_columns: int = 200
    n_monolayers: float = 50
    n_runs: int = 8
    n_points: int = 20
    temperature: float = 350.0  # substrate temperature, K
    flux: float = 1.0           # deposition rate, monolayers per second
    e_bond: float = 0.2         # bond energy per side neighbour, eV
    e_diff: float = 0.8         # diffusion barrier of a free atom, eV
    nu0: float = 1e13           # attempt frequency, 1/s
    max_hops: float = 1e4       # computational cap on hops per landing
    seed: int = 42

    def thermal(self) -> tuple[float, float]:
        return thermal_parameters(self.temperature, self.e_bond, self.e_diff, self.flux,
                                  self.nu0, self.max_hops)

    def to_dict(self) -> dict:
        epsilon, hops = self.thermal()
        return {"model": "thermal_hopping", **asdict(self), "epsilon": epsilon, "hops_per_landing": hops}


def simulate(n_columns: int, times: np.ndarray, rng: np.random.Generator, epsilon: float,
             hops: float) -> np.ndarray:
    """Grow one film with fixed (epsilon, hops) and return W at each atom count in `times`."""
    heights = np.zeros(n_columns, dtype=np.int64)
    widths = np.empty(len(times))
    deposited = 0
    for k, target in enumerate(times):
        thermal_grow(heights, int(target - deposited), epsilon, hops, int(rng.integers(2**31 - 1)))
        deposited = target
        widths[k] = roughness(heights)
    return widths


def run_ensemble(settings: ThermalSettings, epsilon: float | None = None,
                 hops: float | None = None) -> pd.DataFrame:
    """Roughness curve averaged over runs.

    By default (epsilon, hops) come from the real-unit settings; pass them
    directly to use the dimensionless model instead.
    """
    if epsilon is None or hops is None:
        epsilon, hops = settings.thermal()
    times = measurement_times(settings.n_columns, settings.n_monolayers, settings.n_points)
    return ensemble_roughness(lambda t, rng: simulate(settings.n_columns, t, rng, epsilon, hops),
                              times, settings.n_columns, settings.n_runs, settings.seed)


def _final_row(settings: ThermalSettings, **extra) -> dict:
    epsilon, hops = settings.thermal()
    last = run_ensemble(settings).iloc[-1]
    return {**extra, "epsilon": epsilon, "hops_per_landing": hops,
            "roughness": last["roughness_mean"], "roughness_sem": last["roughness_sem"]}


def sweep_temperature(settings: ThermalSettings, temperatures) -> pd.DataFrame:
    """Final roughness at each substrate temperature, flux fixed."""
    rows = []
    for t in temperatures:
        s = ThermalSettings(**{**asdict(settings), "temperature": float(t)})
        rows.append(_final_row(s, temperature=float(t), flux=s.flux))
    return pd.DataFrame(rows)


def sweep_flux(settings: ThermalSettings, fluxes) -> pd.DataFrame:
    """Final roughness at each deposition rate, temperature fixed."""
    rows = []
    for f in fluxes:
        s = ThermalSettings(**{**asdict(settings), "flux": float(f)})
        rows.append(_final_row(s, temperature=s.temperature, flux=float(f)))
    return pd.DataFrame(rows)


def sweep_epsilon(settings: ThermalSettings, epsilons, hops: float) -> pd.DataFrame:
    """Dimensionless version: final roughness vs epsilon (bond strength / kT) at fixed hops."""
    rows = []
    for eps in epsilons:
        last = run_ensemble(settings, float(eps), hops).iloc[-1]
        rows.append({"epsilon": float(eps), "hops_per_landing": hops,
                     "roughness": last["roughness_mean"], "roughness_sem": last["roughness_sem"]})
    return pd.DataFrame(rows)
