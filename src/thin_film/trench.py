"""Phase 4: filling a trench (2D side view) and detecting trapped voids.

The picture: a cross-section through a wafer. The grid is rows (height y,
y = 0 at the bottom) by columns (x). Each cell is EMPTY, SUBSTRATE or FILM.
The substrate has a narrow rectangular trench etched into it, like the
features chips are made of. Atoms fall straight down from above.

"Sticky" ballistic deposition: a falling atom stops the moment it touches any
filled cell, below it or beside it. Atoms falling past the trench's top corners
touch the corner from the side and stick there, so the film grows sideways
from both corners (an "overhang"). If the two overhangs meet before the
trench bottom has filled, they seal off an empty pocket: a void. Voids are a
real defect in chip wiring (higher resistance, reliability failures).

Temperature: after sticking, the newly arrived atom gets a number of hop
attempts (set by temperature and deposition rate, see units.py). Each attempt
succeeds with probability exp(-n * epsilon), where n = number of filled side,
top or bottom neighbours (its coordination). A successful hop moves it to the
empty neighbouring cell (including diagonals) that has the most filled
neighbours, i.e. the most bonds. So mobile atoms drift into corners and
along the trench floor, which fills the trench from the bottom up and prevents
voids. Only the newest atom hops, because older atoms are quickly buried and
locked in. This is a common simplification in such models.

Void detection: flood-fill (breadth-first search) from the open space above
the film through empty cells. Any empty cell the flood cannot reach is sealed
off, i.e. a void.

Periodic boundaries in x, as in the 1D models.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numba import njit

from .units import thermal_parameters

EMPTY, SUBSTRATE, FILM = 0, 1, 2

# 4-neighbour offsets (bonds) and 8-neighbour offsets (allowed hop moves)
DY4 = np.array([1, -1, 0, 0])
DX4 = np.array([0, 0, 1, -1])
DY8 = np.array([1, 1, 1, 0, 0, -1, -1, -1])
DX8 = np.array([-1, 0, 1, -1, 1, -1, 0, 1])


@dataclass
class TrenchSettings:
    """Everything needed to reproduce a Phase 4 run. Saved next to every result."""

    width: int = 128            # domain width (cells)
    trench_width: int = 12
    trench_depth: int = 16
    base: int = 3               # substrate thickness under the trench
    n_monolayers: float = 24    # atoms deposited = n_monolayers * width
    temperature: float = 300.0  # K
    flux: float = 1.0           # ML/s
    e_bond: float = 0.1         # eV per neighbour (weaker than in 1D so bonded atoms can still move)
    e_diff: float = 0.8         # eV
    nu0: float = 1e13
    max_hops: float = 3000      # cap on hops per atom (cost control)
    n_runs: int = 6
    seed: int = 42

    def thermal(self) -> tuple[float, float]:
        return thermal_parameters(self.temperature, self.e_bond, self.e_diff, self.flux,
                                  self.nu0, self.max_hops)

    @property
    def height(self) -> int:
        # sticky films are porous (roughly half dense), so leave room for ~3x the nominal thickness
        return int(self.base + self.trench_depth + 3 * self.n_monolayers + 10)

    @property
    def trench_x(self) -> tuple[int, int]:
        x0 = (self.width - self.trench_width) // 2
        return x0, x0 + self.trench_width

    def to_dict(self) -> dict:
        epsilon, hops = self.thermal()
        return {"model": "trench_filling", **asdict(self), "epsilon": epsilon, "hops_per_atom": hops}


def make_trench(s: TrenchSettings) -> np.ndarray:
    """Build the initial grid: a flat substrate with a rectangular trench in the middle."""
    grid = np.zeros((s.height, s.width), dtype=np.int8)
    grid[: s.base + s.trench_depth, :] = SUBSTRATE
    x0, x1 = s.trench_x
    grid[s.base: s.base + s.trench_depth, x0:x1] = EMPTY
    return grid


@njit(cache=True)
def _coordination(grid, y, x, skip_y, skip_x):
    """Number of filled 4-neighbours of (y, x), ignoring cell (skip_y, skip_x)."""
    h, w = grid.shape
    n = 0
    for k in range(4):
        yy = y + DY4[k]
        xx = (x + DX4[k]) % w
        if yy < 0:
            n += 1
        elif yy < h and not (yy == skip_y and xx == skip_x) and grid[yy, xx] != EMPTY:
            n += 1
    return n


@njit(cache=True)
def deposit(grid, n_atoms, epsilon, hops_per_atom, seed):
    """Drop n_atoms sticky atoms with thermal hopping into `grid` (in place).

    Returns the number actually placed (fewer only if the grid fills to the top).
    """
    np.random.seed(seed)
    h, w = grid.shape
    ymax = 0                                   # highest filled row so far
    for y in range(h):
        for x in range(w):
            if grid[y, x] != EMPTY:
                ymax = y
    p_hop = np.exp(-epsilon * np.arange(5.0))  # coordination 0..4
    whole = int(hops_per_atom)
    frac = hops_per_atom - whole
    cand_y = np.empty(8, np.int64)
    cand_x = np.empty(8, np.int64)
    placed = 0
    for _ in range(n_atoms):
        x = np.random.randint(w)
        y = ymax + 1
        if y >= h - 1:
            break                              # grid full
        # fall until touching something below or beside
        while not (grid[y - 1, x] != EMPTY or grid[y, (x - 1) % w] != EMPTY
                   or grid[y, (x + 1) % w] != EMPTY):
            y -= 1
        grid[y, x] = FILM
        # thermal hops of the newly landed atom
        n_hops = whole + (1 if np.random.random() < frac else 0)
        for _ in range(n_hops):
            if np.random.random() >= p_hop[_coordination(grid, y, x, -1, -1)]:
                continue
            best, n_best = 0, 0
            for k in range(8):
                yy = y + DY8[k]
                xx = (x + DX8[k]) % w
                if yy < 1 or yy >= h - 1 or grid[yy, xx] != EMPTY:
                    continue
                c = _coordination(grid, yy, xx, y, x)  # bonds it would have after moving
                if c > best:
                    best, n_best = c, 0
                if c == best and c > 0:
                    cand_y[n_best] = yy
                    cand_x[n_best] = xx
                    n_best += 1
            if n_best == 0:
                continue
            k = np.random.randint(n_best)
            grid[y, x] = EMPTY
            y, x = cand_y[k], cand_x[k]
            grid[y, x] = FILM
        if y > ymax:
            ymax = y
        placed += 1
    return placed


@njit(cache=True)
def find_voids(grid):
    """Boolean mask of empty cells sealed off from the open space above.

    Breadth-first flood fill: start from every empty cell in the top row,
    repeatedly step to empty 4-neighbours, mark everything reached. Empty cells
    never reached are voids.
    """
    h, w = grid.shape
    reached = np.zeros((h, w), np.bool_)
    queue_y = np.empty(h * w, np.int64)
    queue_x = np.empty(h * w, np.int64)
    head, tail = 0, 0
    for x in range(w):
        if grid[h - 1, x] == EMPTY:
            reached[h - 1, x] = True
            queue_y[tail], queue_x[tail] = h - 1, x
            tail += 1
    while head < tail:
        y, x = queue_y[head], queue_x[head]
        head += 1
        for k in range(4):
            yy = y + DY4[k]
            xx = (x + DX4[k]) % w
            if 0 <= yy < h and grid[yy, xx] == EMPTY and not reached[yy, xx]:
                reached[yy, xx] = True
                queue_y[tail], queue_x[tail] = yy, xx
                tail += 1
    return (grid == EMPTY) & ~reached


def measure(grid: np.ndarray, s: TrenchSettings) -> dict:
    """Step coverage, void areas and film density of a finished trench.

    Thicknesses are "mass thicknesses" (film cells per column), which stay
    meaningful even when the film is porous.

    bottom thickness : film cells per column inside the trench (floor up to the
                       trench mouth), averaged over the central third of the trench
    top thickness    : film cells per column above the flat field, far from the trench
    step coverage    : bottom / top. Near 1 = the trench bottom gets as much
                       film as the top; near 0 = the film piles up on top and
                       barely reaches the bottom
    trench void area : sealed-off empty cells inside the trench (the defect)
    total void area  : all sealed-off empty cells, including pores in the
                       film itself (cold sticky films are porous)
    film density     : fraction of cells below the field surface that are film
    """
    voids = find_voids(grid)
    x0, x1 = s.trench_x
    floor, field = s.base, s.base + s.trench_depth
    third = max(1, s.trench_width // 3)
    start = x0 + (s.trench_width - third) // 2
    centre = np.arange(start, start + third)
    margin = min(s.trench_width, (s.width - s.trench_width) // 4)
    far = np.array([x for x in range(s.width) if not (x0 - margin <= x < x1 + margin)], dtype=np.int64)

    film = grid == FILM
    bottom = float(film[floor:field, centre].sum(axis=0).mean())
    field_film = film[field:, far]
    top = float(field_film.sum(axis=0).mean())
    tops = np.array([np.flatnonzero(col)[-1] + 1 if col.any() else 0 for col in field_film.T])
    return {
        "bottom_thickness": bottom,
        "top_thickness": top,
        "step_coverage": bottom / top if top > 0 else np.nan,
        "trench_void_area": int(voids[floor:field, x0:x1].sum()),
        "total_void_area": int(voids.sum()),
        "film_density": float(field_film.sum() / max(tops.sum(), 1)),
    }


def grow(s: TrenchSettings, seed: int | None = None) -> np.ndarray:
    """Build the trench and deposit the full film; returns the final grid."""
    grid = make_trench(s)
    epsilon, hops = s.thermal()
    deposit(grid, int(s.n_monolayers * s.width), epsilon, hops, s.seed if seed is None else seed)
    return grid


def run_ensemble(s: TrenchSettings) -> dict:
    """Average the measurements over n_runs seeded runs."""
    seeds = np.random.SeedSequence(s.seed).generate_state(s.n_runs)
    rows = pd.DataFrame([measure(grow(s, int(sd) & 0x7FFFFFFF), s) for sd in seeds])
    out = rows.mean().to_dict()
    out.update({f"{k}_sem": rows[k].std(ddof=1) / np.sqrt(s.n_runs) for k in
                ("step_coverage", "trench_void_area")} if s.n_runs > 1 else {})
    out["void_probability"] = float((rows["trench_void_area"] > 0).mean())
    return out


def sweep(s: TrenchSettings, field: str, values) -> pd.DataFrame:
    """Run the ensemble for each value of one setting (e.g. 'temperature' or 'trench_width')."""
    rows = []
    for v in values:
        si = TrenchSettings(**{**asdict(s), field: type(getattr(s, field))(v)})
        epsilon, hops = si.thermal()
        rows.append({field: v, "epsilon": epsilon, "hops_per_atom": hops, **run_ensemble(si)})
    return pd.DataFrame(rows)
