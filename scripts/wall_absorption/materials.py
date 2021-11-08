import numpy as np

# Source of all material absorption coefficients: 
# Wolfgang M. Willems, Kai Schild, Diana Stricker: Formeln und Tabellen Bauphysik. Fourth edition, p. 441.

class Materials:
    # Building materials
    concrete = np.array([
        [125, 0.02],
        [250, 0.02],
        [500, 0.03],
        [1000, 0.04],
        [2000, 0.05],
        [4000, 0.05]
    ])  

    carpet_10mm = np.array([
        [125, 0.04],
        [250, 0.07],
        [500, 0.12],
        [1000, 0.3],
        [2000, 0.5],
        [4000, 0.8]
    ])

    # Wood
    parquet_glued = np.array([
        [125, 0.04],
        [250, 0.04],
        [500, 0.05],
        [1000, 0.06],
        [2000, 0.06],
        [4000, 0.06]
    ])

    # Glass
    window = np.array([
        [125, 0.28],
        [250, 0.2],
        [500, 0.1],
        [1000, 0.06],
        [2000, 0.03],
        [4000, 0.02]
    ])

    mirror_on_wall = np.array([
        [125, 0.12],
        [250, 0.1],
        [500, 0.05],
        [1000, 0.04],
        [2000, 0.02],
        [4000, 0.02]
    ])

    # Special
    wallpaper_on_lime_cement_plaster = np.array([
        [125, 0.02],
        [250, 0.03],
        [500, 0.04],
        [1000, 0.05],
        [2000, 0.07],
        [4000, 0.08]
    ])

    bookshelf = np.array([
        [125, 0.3],
        [250, 0.4],
        [500, 0.4],
        [1000, 0.3],
        [2000, 0.3],
        [4000, 0.2]
    ])

    cinema_screen = np.array([
        [125, 0.1],
        [250, 0.1],
        [500, 0.2],
        [1000, 0.3],
        [2000, 0.5],
        [4000, 0.6]
    ])

    total_absorption = np.array([
        [125, 0.9999999],
        [250, 0.9999999],
        [500, 0.9999999],
        [1000, 0.9999999],
        [2000, 0.9999999],
        [4000, 0.9999999]
    ])

    total_reflection = np.array([
        [125, 0.000001],
        [250, 0.000001],
        [500, 0.000001],
        [1000, 0.000001],
        [2000, 0.000001],
        [4000, 0.000001]
    ])
