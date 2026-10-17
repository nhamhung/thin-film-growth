#!/usr/bin/env python
"""One-click launcher: open the notebooks (JupyterLab) or the Streamlit app in your browser.

You normally don't run this yourself. Double-click `start_windows.bat` (Windows)
or `start_mac.command` (macOS): they install uv if needed, let uv set up the
exact Python and packages from `uv.lock`, then run this script.

Usage (from the project folder):
    uv run python scripts/launch.py            # menu
    uv run python scripts/launch.py notebooks  # straight to JupyterLab
    uv run python scripts/launch.py app        # straight to the Streamlit app
    uv run python scripts/launch.py check      # only check the setup, then exit
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "Thin-Film Growth"
APP = PROJECT_ROOT / "app" / "streamlit_app.py"
NOTEBOOKS = PROJECT_ROOT / "notebooks"


# --- Project-specific -------------------------------------------------------

NOTEBOOK_DATA: list[Path] = []  # nothing to download: the simulations generate their own data


def ensure_notebook_data() -> bool:
    return True


# --- Shared ------------------------------------------------------------------


def _free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found.")


def _open_when_ready(url: str, health: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(1)
    print(f"\nThe app is taking a while to start. Open {url} in your browser when it's ready.")


def _env() -> dict:
    env = os.environ.copy()
    # Keep Jupyter's kernels and settings inside the project, so an older
    # Jupyter install elsewhere on the computer can't take over the kernel.
    env["JUPYTER_DATA_DIR"] = str(PROJECT_ROOT / ".jupyter_data")
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    return env


def run_notebooks() -> None:
    if not ensure_notebook_data():
        return
    print("\nOpening JupyterLab in your browser...")
    print("Open a notebook from the list on the left, then use Run > Run All Cells.")
    print("To stop: close the browser tab, then press Ctrl+C in this window.\n")
    port = _free_port(8888)
    subprocess.run([sys.executable, "-m", "jupyter", "lab", f"--notebook-dir={NOTEBOOKS}", f"--port={port}"],
                   cwd=PROJECT_ROOT, env=_env())


def run_app() -> None:
    port = _free_port(8501)
    url = f"http://localhost:{port}"
    print(f"\nStarting the app at {url} (your browser opens by itself)...")
    print("To stop: press Ctrl+C in this window.\n")
    threading.Thread(target=_open_when_ready, args=(url, f"{url}/_stcore/health"), daemon=True).start()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless=true",
                    f"--server.port={port}"], cwd=PROJECT_ROOT, env=_env())


def check() -> None:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    for name in ("streamlit", "jupyterlab", "ipykernel"):
        __import__(name)
        print(f"  {name}: ok")
    print("Setup looks good.")


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else None
    actions = {"notebooks": run_notebooks, "app": run_app, "check": check}
    if choice:
        actions[choice]()
        return
    while True:
        print(f"\n=== {PROJECT_NAME} ===")
        print("  1  Open the notebooks (JupyterLab)")
        print("  2  Open the app (Streamlit)")
        print("  3  Quit")
        picked = input("Type 1, 2 or 3 and press Enter: ").strip()
        try:
            if picked == "1":
                run_notebooks()
            elif picked == "2":
                run_app()
            elif picked == "3":
                return
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
