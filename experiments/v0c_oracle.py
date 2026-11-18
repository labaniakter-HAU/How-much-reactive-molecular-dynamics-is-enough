# file: experiments/v0c_oracle.py
"""V0c diagnostic (descriptive, not pre-registered; run after V0c failed its design test): how much can any
information-based design gain on this menu? The greedy design is given the TRUE expected counts (true k x
noise-free exposure) and the true target weights, and compared with the pre-registered static design, the
uniform campaign and the median of 100 random campaigns, on V0c's 20 replicates.

  python -I experiments/v0c_oracle.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import design
from v0c_synthetic import RUNS, T_STAR, BUDGETS, N_REP, N_RANDOM, clean_exposure, rms
from synthetic import network
from kinetics import R_KJ
from v0b_reduction import EA, RX, SPECIES

torch.set_num_threads(1)
truth = network(SPECIES, [r[:2] for r in RX], EA, [r[3] for r in RX])
lnk = lambda T: np.log([r[3] for r in RX]) - EA / (R_KJ * T)
names = [f"{T}K_{s}" for T, s, _ in RUNS]
T_of = {nm: float(T) for nm, (T, _, _) in zip(names, RUNS)}
expo = {nm: clean_exposure(truth, T, b) for nm, (T, _, b) in zip(names, RUNS)}
menu_runs = [nm for nm in names if T_of[nm] != T_STAR]
menu = {nm: [L for L in (50, 100, 200) if L < len(expo[nm])] + [len(expo[nm])] for nm in menu_runs}
w = np.exp(lnk(T_STAR)) * sum(expo[nm].sum(0) for nm in names if T_of[nm] == T_STAR)
us = 1000 / T_STAR

def tvar(camp, mu_of):
    S = np.zeros((3, len(w)))
    for r, L in camp.items():
        if L:
            mu, u = mu_of(r, L), 1000 / T_of[r]
            S += np.array([mu, mu * u, mu * u * u])
    D = S[0] * S[2] - S[1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(D > 1e-9 * S[0] * S[2], (S[2] - 2 * us * S[1] + us ** 2 * S[0]) / D, np.inf)
    v = np.where(w > 0, v, 0.0)
    return float((w * v).sum() / w.sum()) if np.isfinite(v).all() else np.inf

def greedy_mu(B, mu_of):
    camp = {"5000K_1": 50}
    while mv := design._moves(menu, camp, B):
        q0 = tvar(camp, mu_of)
        def key(m):
            c, q1 = m[1] - camp.get(m[0], 0), tvar({**camp, m[0]: m[1]}, mu_of)
            g = (np.inf if q1 < np.inf else 0.0) if q0 == np.inf else q0 - q1
            return (-g / c, c, q1)
        r, L = min(mv, key=key); camp[r] = L
    return camp

true_mu = lambda r, L: np.exp(lnk(T_of[r])) * expo[r][:L].sum(0) + 1e-300
total = sum(len(expo[nm]) for nm in menu_runs)
rng, out = np.random.default_rng(0), {}
for f in BUDGETS:
    B = int(f * total)
    camps = {"static": design.greedy(menu, {"5000K_1": 50}, B, T_of, T_STAR), "oracle": greedy_mu(B, true_mu),
             "uniform": design.uniform(menu, {"5000K_1": 50}, B, T_of)}
    rand = [design.random_campaign(menu, {"5000K_1": 50}, B, rng) for _ in range(N_RANDOM)]
    err = {k: [] for k in camps}; med = []
    for rep in range(N_REP):
        r = np.random.default_rng(1000 + rep)
        ev = {nm: np.cumsum(r.poisson(np.exp(lnk(T_of[nm])) * expo[nm]), 0) for nm in names}
        data = {nm: {L: (ev[nm][L - 1], expo[nm][:L].sum(0)) for L in menu.get(nm, [len(expo[nm])])} for nm in names}
        def ev_of(camp):
            Ts, n, E = design.pool(camp, data, T_of)
            return rms(design.predict(design.fit_lines(n, E, Ts), T_STAR)[0], lnk(T_STAR), w)
        for k, c in camps.items(): err[k].append(ev_of(c))
        med.append(np.median([ev_of(c) for c in rand]))
    m = np.mean(med)
    print(f"{f:.0%}: median random {m:.3f} | " + " | ".join(f"{k} {np.mean(v):.3f} ({np.mean(v)/m:.2f}x)" for k, v in err.items()))
    out[str(f)] = {"median_random": float(m), **{k: float(np.mean(v)) for k, v in err.items()},
                   "oracle_campaign": {k: v for k, v in camps["oracle"].items() if v}}
    print("   oracle:", {k: v for k, v in camps["oracle"].items() if v}, " predicted var", round(tvar(camps["oracle"], true_mu), 5),
          "static's", round(tvar(camps["static"], true_mu), 5), "uniform's", round(tvar(camps["uniform"], true_mu), 5))
(ROOT / "results" / "v0c" / "oracle.json").write_text(json.dumps(out, indent=1))
