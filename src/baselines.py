# file: src/baselines.py
"""Simple baselines of H3(b) (design section 5). Definitions fixed before their preview (decision D4):
- single-step Arrhenius per class: y(t) = yinf + (y0 - yinf) exp(-k t) fitted per training temperature (seeds
  pooled), then ln k linear in 1/T and yinf linear in T;
- XGBoost: one regressor per class on (T, t).
runs: [(T, t (n_t,), y (n_t, n_classes))]; each fit returns predict(T, t, y0) -> (n_t, n_classes)."""
import numpy as np
from scipy.optimize import least_squares
from xgboost import XGBRegressor


def arrhenius_fit(runs):
    Ts = sorted({T for T, _, _ in runs})
    n_c = runs[0][2].shape[1]
    lnk, yinf = np.zeros((len(Ts), n_c)), np.zeros((len(Ts), n_c))
    for i, T in enumerate(Ts):
        rs = [(t, y) for TT, t, y in runs if TT == T]
        for c in range(n_c):
            def res(p):
                return np.concatenate([p[1] + (y[0, c] - p[1]) * np.exp(-np.exp(p[0]) * t) - y[:, c] for t, y in rs])
            lnk[i, c], yinf[i, c] = least_squares(res, [np.log(0.05), np.mean([y[-1, c] for _, y in rs])]).x
    pk = [np.polyfit(1 / np.array(Ts), lnk[:, c], 1) for c in range(n_c)]
    py = [np.polyfit(Ts, yinf[:, c], 1) for c in range(n_c)]

    def predict(T, t, y0):
        cols = []
        for c in range(n_c):
            y_inf = np.polyval(py[c], T)
            cols.append(y_inf + (y0[c] - y_inf) * np.exp(-np.exp(np.polyval(pk[c], 1 / T)) * t))
        return np.stack(cols, 1)
    return predict


def xgboost_fit(runs):
    X = np.concatenate([np.column_stack([np.full(len(t), T), t]) for T, t, _ in runs])
    Y = np.concatenate([y for _, _, y in runs])
    models = [XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05).fit(X, Y[:, c]) for c in range(Y.shape[1])]
    return lambda T, t, y0: np.stack([m.predict(np.column_stack([np.full(len(t), T), t])) for m in models], 1)
