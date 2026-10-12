"""Record a growing film as a compact sequence of frames for replay and animation.

The app simulates a whole run first, then plays it back, so you can pause and
scrub to any moment (for example the frame where a trench seals). Recordings are
kept small enough to send to the browser:

  1D models : column heights after every frame, shape (n_frames + 1, N).
              Frame 0 is the bare substrate.
  trench    : two maps instead of a grid per frame:
                arrival[y, x]    = frame in which the cell became film (255 = never)
                void_start[y, x] = first frame in which the cell was a sealed void (255 = never)
              This works because in the trench model only the newest atom moves,
              and only right after it lands. Once a frame ends, every film cell
              stays film for good, so cell (y, x) at frame k is
                film      if arrival <= k
                void      if void_start <= k < arrival
                substrate if it was substrate at the start
                empty     otherwise.
              `trench_frame` rebuilds any frame and the tests check it against
              the real grids.
"""

import numpy as np

from . import random_deposition as rd
from . import surface_relaxation as sr
from . import thermal as th
from . import trench as tr
from .roughness import roughness

NEVER = 255
MAX_FRAMES = 254


def record_1d(model: str, n_columns: int, n_monolayers: float, n_frames: int, seed: int,
              epsilon: float = 0.0, hops: float = 0.0) -> dict:
    """Grow one 1D film (model = 'random_deposition' | 'surface_relaxation' | 'thermal')."""
    rng = np.random.default_rng(seed)
    heights = np.zeros(n_columns, np.int64)
    per_frame = max(1, int(round(n_monolayers * n_columns / n_frames)))
    frames = [heights.copy()]
    for _ in range(n_frames):
        if model == "random_deposition":
            rd.deposit(heights, per_frame, rng)
        elif model == "surface_relaxation":
            sr.deposit(heights, per_frame, rng)
        else:
            th.thermal_grow(heights, per_frame, epsilon, hops, int(rng.integers(2**31 - 1)))
        frames.append(heights.copy())
    frames = np.array(frames)
    ml = np.arange(n_frames + 1) * per_frame / n_columns
    return {"kind": "1d", "model": model, "heights": frames, "monolayers": ml,
            "rq": np.array([roughness(h) for h in frames])}


def record_trench(s: tr.TrenchSettings, n_frames: int) -> dict:
    """Grow one trench film, recording arrival and void-sealing frames for every cell."""
    if n_frames > MAX_FRAMES:
        raise ValueError(f"at most {MAX_FRAMES} frames (frame indices are stored as uint8)")
    grid = tr.make_trench(s)
    substrate = grid == tr.SUBSTRATE
    arrival = np.full(grid.shape, NEVER, np.uint8)
    void_start = np.full(grid.shape, NEVER, np.uint8)
    epsilon, hops = s.thermal()
    per_frame = max(1, int(round(s.n_monolayers * s.width / n_frames)))
    metrics = [tr.measure(grid, s)]
    for k in range(1, n_frames + 1):
        tr.deposit(grid, per_frame, epsilon, hops, s.seed * 1000 + k)
        arrival[(grid == tr.FILM) & (arrival == NEVER)] = k
        voids = tr.find_voids(grid)
        void_start[voids & (void_start == NEVER)] = k
        metrics.append(tr.measure(grid, s))

    # crop to the trench neighbourhood and the rows that ever hold anything
    x0, x1 = s.trench_x
    lo, hi = max(0, x0 - 30), min(s.width, x1 + 30)
    used = np.flatnonzero((grid != tr.EMPTY).any(axis=1))
    top = min(grid.shape[0], used[-1] + 6)
    crop = (slice(0, top), slice(lo, hi))
    return {
        "kind": "trench", "substrate": substrate[crop], "arrival": arrival[crop], "void_start": void_start[crop],
        "monolayers": np.arange(n_frames + 1) * per_frame / s.width,
        "step_coverage": np.array([m["step_coverage"] for m in metrics]),
        "trench_void_area": np.array([m["trench_void_area"] for m in metrics]),
        "film_density": np.array([m["film_density"] for m in metrics]),
        "trench_box": {"x0": x0 - lo, "x1": x1 - lo, "floor": s.base, "field": s.base + s.trench_depth},
        "final_grid": grid[crop],
    }


def trench_frame(rec: dict, k: int) -> np.ndarray:
    """Rebuild frame k's cell codes (0 empty, 1 substrate, 2 film, 3 void) from a trench recording."""
    out = np.zeros(rec["arrival"].shape, np.int8)
    out[rec["substrate"]] = 1
    out[rec["arrival"] <= k] = 2
    out[(rec["void_start"] <= k) & (rec["arrival"] > k)] = 3
    return out


def final_metrics(rec: dict) -> dict:
    """Metrics of the last frame, in the form fab.spec_check expects."""
    if rec["kind"] == "1d":
        return {"rq": float(rec["rq"][-1]), "monolayers": float(rec["monolayers"][-1])}
    return {"step_coverage": float(rec["step_coverage"][-1]),
            "trench_void_area": int(rec["trench_void_area"][-1]),
            "film_density": float(rec["film_density"][-1]),
            "monolayers": float(rec["monolayers"][-1])}
