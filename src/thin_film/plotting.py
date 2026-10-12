"""Shared plots and animations, reused by every phase."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.colors import ListedColormap

# colours for trench cells: empty, substrate, film, void
TRENCH_CMAP = ListedColormap(["#ffffff", "#7f7f7f", "#4a7ab5", "#e4572e"])


def plot_roughness_loglog(curves: dict[str, pd.DataFrame], beta_fits: dict[str, float] | None = None,
                          reference_slope: float | None = 0.5, ax=None):
    """Log-log plot of roughness W versus time (monolayers), one line per model.

    On log-log axes a power law W = A t^beta is a straight line with slope beta,
    so the growth exponent can be read straight off the picture. A dashed
    reference line with the expected slope is drawn for comparison.

    `curves` maps a label to a DataFrame with columns monolayers, roughness_mean
    and (optionally) roughness_sem, as returned by run_ensemble.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    for label, df in curves.items():
        if beta_fits and label in beta_fits:
            label = f"{label} (fit beta = {beta_fits[label]:.3f})"
        ax.errorbar(df["monolayers"], df["roughness_mean"], yerr=df.get("roughness_sem"),
                    fmt="o", ms=4, capsize=2, label=label)
    if reference_slope is not None:
        first = next(iter(curves.values()))
        t = np.array([first["monolayers"].min(), first["monolayers"].max()])
        w0 = first["roughness_mean"].iloc[0]
        ax.plot(t, w0 * (t / t[0]) ** reference_slope, "k--", lw=1,
                label=f"slope {reference_slope} reference")
    ax.set(xscale="log", yscale="log", xlabel="time (monolayers deposited)",
           ylabel="roughness W (atoms)", title="Surface roughness growth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def plot_surface(heights: np.ndarray, ax=None, title: str = "Film profile"):
    """Draw a film's height profile as a bar chart, one bar per column."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(heights)), heights, width=1.0, color="#4a7ab5")
    ax.axhline(np.mean(heights), color="k", ls="--", lw=1, label="mean height")
    ax.set(xlabel="column", ylabel="height (atoms)", title=title, xlim=(-0.5, len(heights) - 0.5))
    ax.legend(loc="upper right")
    return ax


def plot_sweep(df: pd.DataFrame, x: str, y: str, yerr: str | None = None, ax=None, logx: bool = False,
               xlabel: str | None = None, ylabel: str | None = None, label: str | None = None):
    """Line plot of one measured quantity against one swept setting."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df[x], df[y], yerr=df[yerr] if yerr else None, fmt="o-", ms=5, capsize=3, label=label)
    ax.set(xlabel=xlabel or x, ylabel=ylabel or y, xscale="log" if logx else "linear")
    ax.grid(True, alpha=0.3)
    return ax


def plot_heatmap(pivot: pd.DataFrame, ax=None, title: str = "", fmt: str = "{:.2f}", cmap: str = "viridis",
                 highlight: tuple | None = None):
    """Annotated heatmap of a pivot table (rows = index, columns = columns), e.g. a process window.

    `highlight` = (row_value, column_value) draws a box around one cell (the recommended setting).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(pivot.shape[1]), [f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [f"{r:g}" for r in pivot.index])
    ax.set(xlabel=pivot.columns.name, ylabel=pivot.index.name, title=title)
    lo, hi = np.nanmin(pivot.values), np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if (v - lo) < 0.5 * (hi - lo) else "black")
    if highlight is not None:
        i, j = list(pivot.index).index(highlight[0]), list(pivot.columns).index(highlight[1])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2.5))
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def trench_image(grid: np.ndarray, voids: np.ndarray | None = None) -> np.ndarray:
    """Cell codes for display: 0 empty, 1 substrate, 2 film, 3 void."""
    img = grid.astype(np.int8).copy()
    if voids is not None:
        img[voids] = 3
    return img


def plot_trench(grid: np.ndarray, voids: np.ndarray | None = None, ax=None, title: str = "Trench cross-section",
                crop_top: bool = True):
    """Show a trench grid (y = 0 at the bottom) with voids highlighted in red."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    img = trench_image(grid, voids)
    if crop_top:
        filled_rows = np.flatnonzero((grid != 0).any(axis=1))
        img = img[: filled_rows[-1] + 4] if len(filled_rows) else img
    ax.imshow(img, origin="lower", cmap=TRENCH_CMAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set(title=title, xticks=[], yticks=[])
    return ax


def gif_bytes(anim, fps: int = 8) -> bytes:
    """Render a matplotlib animation to GIF bytes (compact, unlike to_jshtml's frame-by-frame PNGs)."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "anim.gif")
        anim.save(path, writer=animation.PillowWriter(fps=fps))
        with open(path, "rb") as f:
            return f.read()


def animate_profiles(frames: list[np.ndarray], interval: int = 100):
    """Animate a sequence of 1D height profiles (matplotlib FuncAnimation)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(np.arange(len(frames[0])), frames[0], width=1.0, color="#4a7ab5")
    ax.set(xlim=(-0.5, len(frames[0]) - 0.5), ylim=(0, max(f.max() for f in frames) * 1.05),
           xlabel="column", ylabel="height")

    def update(k):
        for bar, h in zip(bars, frames[k]):
            bar.set_height(h)
        ax.set_title(f"frame {k + 1}/{len(frames)}")
        return bars

    return animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)


def animate_trench(frames: list[np.ndarray], interval: int = 120):
    """Animate a sequence of trench grids (pass trench_image(...) arrays to show voids)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = max(np.flatnonzero((f != 0).any(axis=1))[-1] for f in frames) + 4
    im = ax.imshow(frames[0][:rows], origin="lower", cmap=TRENCH_CMAP, vmin=0, vmax=3,
                   interpolation="nearest")
    ax.set(xticks=[], yticks=[])

    def update(k):
        im.set_data(frames[k][:rows])
        ax.set_title(f"frame {k + 1}/{len(frames)}")
        return (im,)

    return animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
