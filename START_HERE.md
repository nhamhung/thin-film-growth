# Start here: run the Thin-Film Growth notebooks and app on your computer

This guide is for anyone, including people who have never installed Python. You **don't** need
Python, Git, VS Code or a Streamlit account. You double-click one file, and it
sets everything up.

**What you get**

- **The notebooks**: five step-by-step walk-throughs of how the film-growth simulation works,
  from single atoms landing on a surface to filling trenches on a chip, that you can run, change
  and re-run in your web browser.
- **The app**: the fab-style process tool (pick a recipe, run a wafer, inspect it and compare it
  with the standard recipe) running on your own computer.

**What you need**

- Windows 10 or 11, or macOS 12 or newer on an Apple-silicon Mac (M1 or later). Intel Macs are
  not supported: the simulation's speed-up library no longer provides ready-made files for them.
- An internet connection the first time (it downloads a few hundred MB).
- About 2 GB of free disk space.

---

## Windows

### 1. Download the project

1. Go to <https://github.com/nhamhung/thin-film-growth>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder, right-click `thin-film-growth-main.zip` and choose **Extract All...**.
   Extract it to a simple place such as `C:\Users\<you>\Documents\thin-film-growth-main`.

> Don't run it from inside the ZIP file, and avoid folders synced by OneDrive if you can
> (they work, but setup is slower).

### 2. Start it

1. Open the extracted folder and double-click **`start_windows.bat`**.
2. If Windows shows **"Windows protected your PC"**, click **More info**, then **Run anyway**.
   (This appears because the file was downloaded from the internet.)
3. A black window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds.
4. When it's ready, you see a menu:

   ```
   === Thin-Film Growth ===
     1  Open the notebooks (JupyterLab)
     2  Open the app (Streamlit)
     3  Quit
   Type 1, 2 or 3 and press Enter:
   ```

5. If Windows Firewall asks whether to allow Python, click **Allow** (it only talks to your own
   computer).

### 3. Stop it

Click the black window and press **Ctrl + C**. If it asks *"Terminate batch job (Y/N)?"*, type
**Y** and press Enter. Or simply close the black window.

---

## macOS

### 1. Download the project

1. Go to <https://github.com/nhamhung/thin-film-growth>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder and double-click `thin-film-growth-main.zip` to unzip it. Move the
   `thin-film-growth-main` folder somewhere simple, such as your **Documents** folder.

### 2. Start it

1. Open the folder. **The first time**, right-click (or Control-click) **`start_mac.command`**,
   choose **Open**, then click **Open** again in the warning. (macOS asks because the file was
   downloaded from the internet. After this, a normal double-click works.)
   - If macOS says it *"cannot be opened because Apple cannot check it"* and there is no Open
     button: open **System Settings → Privacy & Security**, scroll down, click **Open Anyway**
     next to `start_mac.command`, then try again.
2. A Terminal window opens. **The first time only**, it installs *uv* (a small tool that sets up
   Python for this project) and downloads Python and the packages. This takes about 2–5 minutes.
   Later starts take a few seconds.
3. When it's ready, you see the same menu as on Windows: type **1** (notebooks) or **2** (app)
   and press Return.

### 3. Stop it

Click the Terminal window and press **Control + C**, then close the window.

---

## Using the notebooks (choice 1)

JupyterLab opens in your web browser.

1. In the file list on the left, open the notebooks in order, starting with
   **`01_random_deposition.ipynb`**:

   | Notebook | What it explains |
   |---|---|
   | `01_random_deposition` | atoms falling at random, and how roughness is measured |
   | `02_surface_relaxation` | atoms sliding into dips, and the speed-up trick (Numba) |
   | `03_temperature_model` | how temperature and deposition speed change the film |
   | `04_trench_filling` | filling a trench, and how voids (empty pockets) are found |
   | `05_por_yield_and_app` | the standard recipes, pass rates and how the app works |

2. Run a notebook from top to bottom with **Run → Run All Cells**. Each takes under a minute or
   two. A `[*]` next to a cell means it is still running.
3. To run just one cell, click it and press **Shift + Enter**.
4. **Experiment freely.** Change a temperature or a size and run the cell again. Each notebook
   ends with suggested exercises ("try changing X and see Y").
5. If things get into a strange state, use **Kernel → Restart Kernel and Run All Cells**.

Your edits are saved in the notebook files. To keep an original, first use **File → Save
Notebook As...** and work on the copy (or download the ZIP again later).

## Using the app (choice 2)

The app opens in your browser by itself (if it doesn't, copy the `http://localhost:...` address
shown in the black/Terminal window into your browser). Pick a process module and a recipe in the
sidebar (or **⭐ Load POR** for the standard recipe), press **▶ Run wafer**, and inspect the
wafer. **⇄ Compare with POR** shows it side by side with the standard recipe. The full
walk-through with screenshots is in `report/user_guide.html` (open it in your browser).

Nothing you do in the app changes the project files.

## Next time

Just double-click `start_windows.bat` or `start_mac.command` again. No internet needed once the
first setup has finished.

---

## Optional: use VS Code instead of the browser

If you prefer VS Code for the notebook:

1. Run the start file once (above), so the project's Python environment (the `.venv` folder)
   exists. You can quit from the menu.
2. Install VS Code from <https://code.visualstudio.com>.
3. In VS Code: **File → Open Folder...** and choose the project folder. Accept the prompt to
   install the recommended **Python** and **Jupyter** extensions.
4. Open a notebook from the `notebooks` folder. Click **Select Kernel** (top right) →
   **Python Environments** → the one marked **`.venv`** (Recommended).

That is the only kernel step, and you do it once.

---

## If something goes wrong

| Problem | What to do |
|---|---|
| Installing uv fails (often on work computers that block scripts) | Install uv another way, then double-click the start file again. **Windows:** open *Command Prompt* and run `winget install --id=astral-sh.uv -e`. **macOS:** if you use Homebrew, run `brew install uv` in Terminal. Or follow <https://docs.astral.sh/uv/getting-started/installation/>. |
| macOS: *"permission denied"* or the file opens in a text editor | Open **Terminal**, type `bash ` (with a space), drag `start_mac.command` into the Terminal window, and press Return. |
| The browser didn't open | Copy the `http://localhost:...` address from the black/Terminal window into your browser. |
| *"Address already in use"* | Another copy is probably running. Close the other black/Terminal window, or just pick the option again: the launcher picks a free port. |
| Errors after editing the notebook | **Kernel → Restart Kernel and Run All Cells**. Still broken? Download the ZIP again for a fresh copy. |
| Something else is broken | Delete the `.venv` folder inside the project and double-click the start file again: it rebuilds a clean setup. |

## Uninstall

1. Delete the project folder.
2. Optional, to remove uv and the downloaded Python too:
   - **Windows:** delete `uv.exe` and `uvx.exe` from `C:\Users\<you>\.local\bin`, the folder
     `%APPDATA%\uv`, and the folder `%LOCALAPPDATA%\uv`.
   - **macOS:** in Terminal, run `rm -rf ~/.local/bin/uv ~/.local/bin/uvx ~/.local/share/uv ~/.cache/uv`.

---

## For developers

The project is a standard [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` lists the
direct dependencies at the exact versions it was tested with, and `uv.lock` pins everything else.

```bash
uv sync                                   # create .venv (add --no-dev to skip pytest/ruff)
uv run pytest                             # tests
uv run jupyter lab notebooks              # notebooks
uv run streamlit run app/streamlit_app.py # app
```

`scripts/launch.py` is what the start files run. `requirements.txt` and `app/requirements.txt`
stay for Streamlit Community Cloud. To publish your own changes to GitHub, see the setup manual at
<https://github.com/nhamhung/nhamhung.github.io/blob/main/README.md> (Git, VS Code, clone, push).
