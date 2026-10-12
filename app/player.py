"""Package recorded runs for the in-browser tool player (app/fab_player.html).

The player is a self-contained HTML page: a process stepper and chamber
schematic (plasma, atom flux, heated chuck), tool readouts with every wafer's
metrics side by side, the in-situ / SEM-style cross-sections, the AFM strip
(1D films) and a metric-vs-time trace. It plays the recorded frames entirely in
the browser, so playback is smooth and can be paused and scrubbed. The same
HTML can be downloaded as a standalone replay.
"""

import base64
import json
from pathlib import Path

import numpy as np

from thin_film import fab

TEMPLATE = Path(__file__).with_name("fab_player.html")


def _b64(arr: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def wafer_payload(rec: dict, *, wafer: str, label: str, module: str, recipe: str, status: str,
                  temperature: float | None = None, flux: float | None = None) -> dict:
    """One wafer's recording plus everything the player shows about it."""
    spec = fab.SPECS[module]
    base = {
        "kind": rec["kind"], "wafer": wafer, "label": label, "module": fab.MODULES[module].split(" · ")[0],
        "recipe": recipe, "status": status, "temperature": temperature, "flux": flux,
        "rate_nm_min": fab.rate_nm_per_min(flux) if flux else None,
        "spec_text": spec["text"] if spec else "none (monitor only)",
        "spec_limit": (spec.get("rq_max") if rec["kind"] == "1d" else spec.get("void_max")) if spec else None,
        "monolayers": rec["monolayers"].round(4).tolist(),
    }
    if rec["kind"] == "1d":
        heights = rec["heights"]
        if heights.max() > np.iinfo(np.uint16).max:
            raise ValueError("heights exceed uint16 range")
        n_frames, n_columns = heights.shape[0] - 1, heights.shape[1]
        return {**base, "n_frames": n_frames, "n_columns": n_columns, "heights_b64": _b64(heights, "<u2"),
                "rq": rec["rq"].round(4).tolist()}
    rows, cols = rec["arrival"].shape
    return {**base, "n_frames": len(rec["monolayers"]) - 1, "rows": rows, "cols": cols,
            "substrate_b64": _b64(rec["substrate"], "u1"), "arrival_b64": _b64(rec["arrival"], "u1"),
            "void_start_b64": _b64(rec["void_start"], "u1"),
            "step_coverage": np.nan_to_num(rec["step_coverage"]).round(4).tolist(),
            "trench_void_area": rec["trench_void_area"].astype(int).tolist(),
            "film_density": rec["film_density"].round(4).tolist()}


def build_html(wafers: list[dict], lot: str, autoplay: bool = True) -> str:
    """Full player page with the payload embedded (also usable as a downloadable replay)."""
    if len({w["n_frames"] for w in wafers}) != 1:
        raise ValueError("wafers shown side by side must have the same number of frames")
    payload = {"wafers": wafers, "lot": lot, "ml_nm": fab.ML_NM, "autoplay": autoplay}
    # "</" is escaped so the JSON can never close the <script> tag early
    return TEMPLATE.read_text().replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/"))


def player_height(wafers: list[dict]) -> int:
    """Pixel height for the embedded player: stepper + chamber row + wafer views + controls (+ slack)."""
    canvas = 240 + 40 if wafers[0]["kind"] == "1d" else 300
    return 36 + 340 + (32 + canvas + 30) + 52 + 24 + 30
