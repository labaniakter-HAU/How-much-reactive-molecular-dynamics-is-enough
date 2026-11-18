# file: src/metrics.py
"""Error metric from design section 2."""
import numpy as np


def normalized_rmse(pred, obs):
    """pred, obs (n_t, B, n_s) -> RMSE per run and class divided by that class's max |obs| in the run,
    averaged over (run, class) pairs where the class occurs."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    scale = np.abs(obs).max(axis=0)
    rmse = np.sqrt(((pred - obs) ** 2).mean(axis=0))
    ok = scale > 0
    return float((rmse[ok] / scale[ok]).mean())
