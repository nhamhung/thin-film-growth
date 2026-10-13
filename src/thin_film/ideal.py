"""Recommended ("ideal") settings for each model, and the evidence behind them.

"Ideal" means something different per model:
  * random deposition and surface relaxation have no temperature or rate, so
    their ideal settings are the ones that best show the physics (a clean
    t^1/2 law; a full rise-then-saturate curve);
  * the thermal model's ideal is the smoothest film (lowest roughness);
  * the trench model's ideal is void-free filling with the highest step coverage.

The thermal and trench values come from the grid searches saved in
results/phase3_process_window and results/phase4_process_window (regenerate with
scripts/run_phase3.py and scripts/run_phase4.py).
"""

IDEAL = {
    "random_deposition": {
        "settings": {"n_columns": 200, "n_monolayers": 1000},
        "why": "Nothing to tune: roughness depends only on time. 1000 ML on 200 columns gives a clean "
               "t^1/2 curve (fitted beta = 0.503, theory 0.5).",
        "source": "results/phase1_random_deposition",
    },
    "surface_relaxation": {
        "settings": {"n_columns": 100, "n_monolayers": 3000},
        "why": "Nothing to tune physically. 100 columns saturate after about 1000 ML, so a 3000 ML run shows "
               "both the t^1/4 rise and the plateau.",
        "source": "results/phase2_surface_relaxation",
    },
    "thermal": {
        "settings": {"temperature": 375, "flux": 0.01, "n_columns": 200, "n_monolayers": 100},
        "why": "Smoothest film in the grid search (W = 0.91 after 50 ML vs 2.5 at 375 K / 1 ML/s). It is the "
               "coolest temperature at which slow deposition still maxes out the hops (1e4 per landing): "
               "atoms explore fully while bonds stay strong enough to hold them at good sites. "
               "All 30 test wafers passed the spec. Faster alternative: 450 K at 1 ML/s gives W = 1.35 at 100x "
               "the deposition rate, but only 24 of 30 wafers pass.",
        "source": "results/phase3_process_window, results/por_yield",
    },
    "trench": {
        "settings": {"temperature": 420, "flux": 1.0, "trench_width": 16, "trench_depth": 16, "n_monolayers": 24},
        "why": "Void-free in the grid search (void 0, coverage 0.59), within noise of the highest coverage "
               "found (0.60 at 520 K) but 100 K cooler. Single wafers vary: 24 of 30 seeds pass the spec "
               "(the rest have coverage just under 0.50 or a 1-6 cell void), so expect ~80% yield. "
               "Geometry matters most: a trench must be about as wide as it is "
               "deep. At width 8 / depth 16 no temperature or rate avoids a void. 460 K at 10 ML/s is also "
               "void-free on average and 10x faster, but only 16 of 30 wafers pass.",
        "source": "results/phase4_process_window, results/por_yield",
    },
}
