"""Thin-film growth simulator: set a recipe, run a wafer, replay it, compare against the POR.

Run with:  streamlit run app/streamlit_app.py   (from the project root, so .streamlit/config.toml applies)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

import player  # noqa: E402
from thin_film import fab  # noqa: E402
from thin_film import trench as tr  # noqa: E402
from thin_film.ideal import IDEAL  # noqa: E402
from thin_film.recording import final_metrics, record_1d, record_trench  # noqa: E402
from thin_film.units import thermal_parameters  # noqa: E402

LOT = "TF-0001"
ML_OPTIONS_1D = [10, 30, 100, 300, 1000, 3000]
THERMAL = {"thermal", "trench"}
E_BOND = {"thermal": 0.2, "trench": 0.1}
MAX_HOPS = {"thermal": 1e4, "trench": 3000}
E_DIFF = 0.8

st.set_page_config(page_title="Thin Film Growth", page_icon="🔬", layout="wide")
st.html("""<style>
  @import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap");
  h1, h2, h3, h4, [data-baseweb="tab"], [data-testid="stMetricValue"] { font-family: "JetBrains Mono", ui-monospace, monospace !important; }
  [data-testid="stSidebar"] { border-right: 1px solid #243140; }
  .recipe { font-family: "JetBrains Mono", monospace; font-size: 12px; color: #38d6e8; word-break: break-all; }
</style>""")

# ---------------------------------------------------------------- state
DEFAULTS = {"module": "trench", "temperature": 375, "log_flux": 0.0, "trench_width": 12, "trench_depth": 16,
            "ml_trench": 24, "n_columns": 200, "ml_1d": 100, "n_frames": 60, "seed": 42}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
    # Re-assigning keeps the value when its widget is not drawn on this run (e.g. trench sliders while a
    # 1D module is selected); otherwise Streamlit discards it and switching back resets the slider.
    st.session_state[key] = st.session_state[key]
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_run", None)


def por_settings(module: str) -> dict:
    """POR (process of record) settings, mapped onto widget keys."""
    out = {}
    for name, value in IDEAL[module]["settings"].items():
        if name == "temperature":
            out["temperature"] = int(value)
        elif name == "flux":
            out["log_flux"] = float(np.log10(value))
        elif name == "n_monolayers":
            out["ml_trench" if module == "trench" else "ml_1d"] = int(value)
        else:
            out[name] = int(value)
    return out


def load_por():
    st.session_state.update(por_settings(st.session_state["module"]))


# ---------------------------------------------------------------- recipe (sidebar)
with st.sidebar:
    st.markdown("### Recipe")
    module = st.selectbox("Process module", list(fab.MODULES), format_func=fab.MODULES.get, key="module")
    st.button("⭐ Load POR", key="por_button", on_click=load_por, width="stretch",
              help="Load the process of record: the recommended recipe for this module")
    thermal = module in THERMAL

    temperature = st.slider("Temperature (K)", 250, 550, step=5, key="temperature", disabled=not thermal)
    log_flux = st.slider("Deposition rate, log10(ML/s)", -2.0, 3.0, step=0.25, key="log_flux", disabled=not thermal)
    flux = float(10 ** log_flux)
    if thermal:
        st.caption(f"{fab.kelvin_to_celsius(temperature):.0f} °C · {flux:.3g} ML/s ≈ {fab.rate_nm_per_min(flux):.3g} nm/min")
    else:
        st.caption("Temperature and rate have no effect in this module.")

    if module == "trench":
        trench_width = st.slider("Trench width (cells)", 4, 32, key="trench_width")
        trench_depth = st.slider("Trench depth (cells)", 8, 32, key="trench_depth")
        n_monolayers = st.slider("Film thickness (ML)", 4, 40, key="ml_trench")
    else:
        n_columns = st.slider("Substrate width (columns)", 50, 400, step=50, key="n_columns")
        n_monolayers = st.select_slider("Film thickness (ML)", ML_OPTIONS_1D, key="ml_1d")

    with st.expander("Advanced"):
        n_frames = st.slider("Frames recorded", 20, 120, step=10, key="n_frames")
        seed = int(st.number_input("Random seed", step=1, key="seed"))
        if thermal:
            eps, hops = thermal_parameters(temperature, E_BOND[module], E_DIFF, flux, max_hops=MAX_HOPS[module])
            st.caption(f"Model inputs: ε = E_bond/kT = {eps:.2f}, {hops:.3g} hops per atom"
                       + (" (capped)" if hops >= MAX_HOPS[module] else ""))

    is_por = all(np.isclose(st.session_state[k], v) for k, v in por_settings(module).items())
    recipe = fab.recipe_name(module, temperature if thermal else None, flux if thermal else None, n_monolayers,
                             (trench_width, trench_depth) if module == "trench" else None)
    st.markdown(f'<div class="recipe">{"⭐ POR · " if is_por else ""}{recipe}</div>', unsafe_allow_html=True)
    spec = fab.SPECS[module]
    st.caption(f"Spec: {spec['text'] if spec else 'none (monitor only)'}")
    run = st.button("▶ Run wafer", key="run_button", type="primary", width="stretch")
    split = st.button("⇄ Compare with POR", key="split_button", width="stretch",
                      help="Run your recipe and the POR side by side with the same seed")


# ---------------------------------------------------------------- simulation
@st.cache_data(show_spinner=False, max_entries=32)
def simulate(module: str, temperature: float, flux: float, n_monolayers: float, n_frames: int, seed: int,
             n_columns: int = 200, trench_width: int = 12, trench_depth: int = 16) -> dict:
    """Record one wafer (cached, so an identical recipe is instant)."""
    if module == "trench":
        s = tr.TrenchSettings(trench_width=trench_width, trench_depth=trench_depth, n_monolayers=n_monolayers,
                              temperature=temperature, flux=flux, seed=seed)
        return record_trench(s, n_frames)
    eps, hps = thermal_parameters(temperature, E_BOND["thermal"], E_DIFF, flux, max_hops=MAX_HOPS["thermal"])
    return record_1d(module, n_columns, n_monolayers, n_frames, seed, eps, hps)


def current_recipe() -> dict:
    r = {"module": module, "temperature": float(temperature), "flux": flux, "n_monolayers": float(n_monolayers),
         "n_frames": n_frames, "seed": seed}
    if module == "trench":
        r.update(trench_width=trench_width, trench_depth=trench_depth)
    else:
        r.update(n_columns=n_columns)
    return r


def por_recipe() -> dict:
    """The current recipe with the POR settings swapped in (same seed and frame count)."""
    por = {k: float(v) if k in ("temperature", "flux", "n_monolayers") else v
           for k, v in IDEAL[module]["settings"].items()}
    return {**current_recipe(), **por}


def process_wafer(r: dict, label: str) -> dict:
    """Simulate, spec-check and log one wafer; returns its player payload."""
    rec = simulate(**r)
    metrics = final_metrics(rec)
    status = fab.spec_check(module, metrics)
    wid = fab.wafer_id(LOT, len(st.session_state["history"]) + 1)
    trench_dims = (r["trench_width"], r["trench_depth"]) if module == "trench" else None
    name = fab.recipe_name(module, r["temperature"] if thermal else None, r["flux"] if thermal else None,
                           r["n_monolayers"], trench_dims)
    row = {"wafer": wid, "module": fab.MODULES[module].split(" · ")[0], "module_key": module, "recipe": name,
           "label": label, "result": status, "seed": r["seed"]}
    if module == "trench":
        row.update({"void (cells)": metrics["trench_void_area"], "step coverage": metrics["step_coverage"],
                    "density": metrics["film_density"]})
    else:
        row.update({"Rq (atoms)": metrics["rq"]})
    st.session_state["history"].append(row)
    return player.wafer_payload(rec, wafer=wid, label=label, module=module, recipe=name, status=status,
                                temperature=r["temperature"] if thermal else None, flux=r["flux"] if thermal else None)


if run or split:
    with st.spinner("Simulating…"):
        wafers, settings = [process_wafer(current_recipe(), "POR" if is_por else "RECIPE")], [current_recipe()]
        if split:
            wafers.append(process_wafer(por_recipe(), "POR"))
            settings.append(por_recipe())
        st.session_state["last_run"] = {"wafers": wafers, "settings": settings}

hist = pd.DataFrame(st.session_state["history"])

# ---------------------------------------------------------------- page
st.markdown("### Thin film growth simulator")
tab_run, tab_lot, tab_about = st.tabs(["RUN", "LOT HISTORY", "ABOUT"])

with tab_run:
    last = st.session_state["last_run"]
    if last is None:
        st.info("Set a recipe in the sidebar (or press **⭐ Load POR**), then **▶ Run wafer**. "
                "**⇄ Compare with POR** runs your recipe and the POR side by side.")
    else:
        html = player.build_html(last["wafers"], LOT)
        components.html(html, height=player.player_height(last["wafers"]))
        with st.popover("⬇ Export"):
            name = last["wafers"][0]["wafer"]
            st.download_button("Replay (standalone HTML)", html, f"{name}_replay.html", mime="text/html")
            st.download_button("Recipe & settings (JSON)", json.dumps(last["settings"], indent=2), f"{name}_recipe.json")
            st.download_button("Lot history (CSV)", hist.drop(columns="module_key").to_csv(index=False),
                               f"{LOT}_history.csv")

COLOR = alt.Scale(domain=["PASS", "FAIL", "MONITOR"], range=["#3ddc84", "#ff5a4e", "#38d6e8"])


def trend_chart(df: pd.DataFrame, metric: str, limit: float | None, title: str):
    """One point per wafer, coloured by result, with the spec limit dashed."""
    base = alt.Chart(df).encode(x=alt.X("wafer:N", sort=None, title=None))
    layers = [base.mark_line(color="#4b5866").encode(y=alt.Y(f"{metric}:Q", title=title)),
              base.mark_circle(size=80, opacity=1).encode(
                  y=f"{metric}:Q", color=alt.Color("result:N", scale=COLOR, legend=None),
                  tooltip=["wafer", "recipe", "label", alt.Tooltip(f"{metric}:Q", format=".2f"), "result"])]
    if limit is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [limit]})).mark_rule(color="#ff5a4e", strokeDash=[6, 4]).encode(y="y:Q"))
    return (alt.layer(*layers).properties(height=220).configure(background="transparent")
            .configure_axis(gridColor="#243140", labelColor="#7b8a99", titleColor="#7b8a99", domainColor="#243140"))


with tab_lot:
    if hist.empty:
        st.info("No wafers yet.")
    else:
        judged = hist[hist["result"] != "MONITOR"]
        st.caption(f"Lot {LOT} · {len(hist)} wafers · {int((judged['result'] == 'PASS').sum())} pass, "
                   f"{int((judged['result'] == 'FAIL').sum())} fail"
                   + (f" · yield {100 * (judged['result'] == 'PASS').mean():.0f}%" if len(judged) else ""))
        mod = hist[hist["module_key"] == module]
        if not mod.empty:
            st.markdown(f"**Trend · {fab.MODULES[module]}** (switch module in the sidebar to see another)")
        if module == "trench" and not mod.empty:
            a, b = st.columns(2)
            a.altair_chart(trend_chart(mod, "void (cells)", 0, "trench void (cells)"), width="stretch")
            b.altair_chart(trend_chart(mod, "step coverage", 0.5, "step coverage"), width="stretch")
        elif not mod.empty:
            st.altair_chart(trend_chart(mod, "Rq (atoms)", (fab.SPECS[module] or {}).get("rq_max"), "Rq (atoms)"),
                            width="stretch")
        # wafers from different modules have different metrics: show missing ones as "—"
        fmt = {"step coverage": "{:.2f}", "density": "{:.2f}", "Rq (atoms)": "{:.2f}", "void (cells)": "{:.0f}"}
        show = hist.drop(columns="module_key")
        for col in show:
            show[col] = show[col].map(lambda v, f=fmt.get(col, "{}"): "—" if pd.isna(v) else f.format(v))
        st.dataframe(show.style.map(lambda v: {"PASS": "color:#3ddc84", "FAIL": "color:#ff5a4e",
                                               "MONITOR": "color:#38d6e8"}.get(v, ""), subset=["result"]),
                     hide_index=True, width="stretch")
        if st.button("Clear lot", key="clear_button"):
            st.session_state["history"], st.session_state["last_run"] = [], None
            st.rerun()

with tab_about:
    st.markdown(f"""
**Views.** *In-situ*: the film coloured by when it grew (early teal → late blue); trench voids in red.
*SEM*: the same cross-section drawn like an SEM image (grey levels, bright edges, scale bar). It is a
rendering of the simulation, not microscope data. *AFM strip* (1D modules): surface height above/below the mean.

**Metrics.** Rq is RMS roughness (the interface width W). Step coverage = film on the trench bottom ÷ film on top.
A void is empty space sealed off from the surface.

**POR.** {IDEAL[module]["why"]}

**Limits of the model.** 1 cell = 1 atom; 1 ML ≈ {fab.ML_NM} nm is illustrative. Temperatures are relative to the
model's energy barriers (E_bond {E_BOND['thermal']} eV in 1D, {E_BOND['trench']} eV in 2D; E_diff {E_DIFF} eV), not a real material.
Hops are capped ({MAX_HOPS['thermal']:.0e} per landing in 1D, {MAX_HOPS['trench']:.0f} per atom in 2D). 1D cross-sections are stretched
so roughness is visible; the stretch factor is printed in the view.
""")
    for name, title in [("phase4_process_window", "Trench process window (red box = POR)"),
                        ("phase3_process_window", "Thermal process window (red box = POR)")]:
        fig = ROOT / "results" / name / "figure.png"
        if fig.exists():
            with st.expander(title):
                st.image(str(fig), width="stretch")
