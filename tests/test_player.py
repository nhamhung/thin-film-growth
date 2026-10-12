"""The browser player receives exactly the recorded data."""

import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from thin_film import trench as tr
from thin_film.recording import record_1d, record_trench

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import player  # noqa: E402


def payload_of(html: str) -> dict:
    return json.loads(re.search(r"const P = (.*?);\n", html).group(1).replace("<\\/", "</"))


def test_1d_heights_round_trip():
    rec = record_1d("random_deposition", 40, 10, 5, seed=0)
    w = player.wafer_payload(rec, wafer="W", label="</script>", module="random_deposition", recipe="r", status="MONITOR")
    html = player.build_html([w], "LOT")
    assert "</script>\"" not in html          # payload cannot terminate the script tag
    p = payload_of(html)["wafers"][0]
    heights = np.frombuffer(base64.b64decode(p["heights_b64"]), "<u2").reshape(6, 40)
    assert np.array_equal(heights, rec["heights"]) and p["label"] == "</script>"


def test_trench_maps_round_trip():
    rec = record_trench(tr.TrenchSettings(n_monolayers=6), 6)
    w = player.wafer_payload(rec, wafer="W", label="POR", module="trench", recipe="r", status="PASS",
                             temperature=420, flux=1.0)
    p = payload_of(player.build_html([w], "LOT"))["wafers"][0]
    arrival = np.frombuffer(base64.b64decode(p["arrival_b64"]), "u1").reshape(p["rows"], p["cols"])
    assert np.array_equal(arrival, rec["arrival"])
    assert p["spec_limit"] == 0 and p["rate_nm_min"] == pytest.approx(18.0)


def test_split_lot_needs_equal_frame_counts():
    a = player.wafer_payload(record_1d("random_deposition", 20, 5, 4, 0), wafer="A", label="", module="random_deposition",
                             recipe="", status="MONITOR")
    b = player.wafer_payload(record_1d("random_deposition", 20, 5, 6, 0), wafer="B", label="", module="random_deposition",
                             recipe="", status="MONITOR")
    with pytest.raises(ValueError):
        player.build_html([a, b], "LOT")
