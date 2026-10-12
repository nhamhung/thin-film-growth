"""Fab-style presentation helpers: unit conversions, recipe names, wafer IDs and spec checks.

Nothing here changes the physics. It translates simulation quantities into the
vocabulary a process engineer would use, and keeps the translations honest:
  * 1 monolayer (ML) is taken as ~0.3 nm, a typical metal atomic spacing. This is
    illustrative only, since the lattice model has no real length scale.
  * Rq (RMS roughness) is the metrology name for exactly the W we compute.
  * Specs are pass/fail limits set from the process-window results (see ideal.py).
"""

ML_NM = 0.3  # nm per monolayer (illustrative)

MODULES = {
    "random_deposition": "PM1 · Random deposition (1D)",
    "surface_relaxation": "PM2 · Surface relaxation (1D)",
    "thermal": "PM3 · Thermal hopping (1D)",
    "trench": "PM4 · Trench fill (2D)",
}

# Pass/fail limits. None = monitor only (the model has no process knobs to tune).
SPECS = {
    "random_deposition": None,
    "surface_relaxation": None,
    # single-wafer limit, 100 ML film, 30 seeds each: POR (375 K, 0.01 ML/s) Rq 0.95-1.86 (always passes);
    # 375 K at 1 ML/s: 5th-95th percentile Rq 1.87-3.29, so it fails almost every wafer
    "thermal": {"rq_max": 2.0, "text": "Rq ≤ 2.0 atoms (100 ML film)"},
    # set from the trench process window: POR gives void 0, coverage ~0.59
    "trench": {"void_max": 0, "coverage_min": 0.5, "text": "trench void = 0 and step coverage ≥ 0.50"},
}


def kelvin_to_celsius(t_k: float) -> float:
    return t_k - 273.15


def rate_nm_per_min(flux_ml_per_s: float) -> float:
    """Deposition rate in nm/min for a flux in ML/s (1 ML ≈ 0.3 nm)."""
    return flux_ml_per_s * ML_NM * 60


def thickness_nm(monolayers: float) -> float:
    return monolayers * ML_NM


def recipe_name(module: str, temperature: float | None = None, flux: float | None = None,
                monolayers: float | None = None, trench: tuple[int, int] | None = None) -> str:
    """Short, readable recipe ID, e.g. 'PM4-TRF-420K-1ML/s-W16D16'."""
    code = {"random_deposition": "PM1-RD", "surface_relaxation": "PM2-SR", "thermal": "PM3-TH",
            "trench": "PM4-TRF"}[module]
    parts = [code]
    if temperature is not None:
        parts.append(f"{temperature:g}K")
    if flux is not None:
        parts.append(f"{flux:.3g}ML/s")
    if trench is not None:
        parts.append(f"W{trench[0]}D{trench[1]}")
    if monolayers is not None and trench is None:
        parts.append(f"{monolayers:g}ML")
    return "-".join(parts)


def wafer_id(lot: str, number: int) -> str:
    return f"{lot}-W{number:02d}"


def spec_check(module: str, metrics: dict) -> str:
    """'PASS', 'FAIL' or 'MONITOR' (no spec) for a finished wafer's metrics."""
    spec = SPECS[module]
    if spec is None:
        return "MONITOR"
    if module == "thermal":
        return "PASS" if metrics["rq"] <= spec["rq_max"] else "FAIL"
    ok = metrics["trench_void_area"] <= spec["void_max"] and metrics["step_coverage"] >= spec["coverage_min"]
    return "PASS" if ok else "FAIL"
