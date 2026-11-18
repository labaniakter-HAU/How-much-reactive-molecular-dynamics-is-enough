# file: experiments/round2_dev_calibration.py
"""Round 2 development (design section 12), C4: uncertainty variants on the V6 folds (development data).

  python -I experiments/round2_dev_calibration.py
Variants (all from the fold's training runs only):
  base     V6 as run (phi, normal)
  t        sd x t(0.975, m - 2) / 1.96 when phi is estimated
  psi      seed overdispersion: var x psi, psi = mean squared Poisson-standardised difference between the two seeds'
           ln k at training temperatures (reactions with >= 10 events in each seed); also on the observation
  loo      internal leave-one-temperature-out among the training temperatures: sd x kappa, kappa = the scale that
           makes the internal |z| median 0.674
  disc     sd^2 + tau^2 (u* - u_bar)^2, tau^2 from the internal leave-one-out residuals
  and combinations. Score: |coverage(|z| <= 2) - 0.954| + |median |z| - 0.674|.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import design  # noqa: E402
import fit  # noqa: E402
from data import load_stanford  # noqa: E402

TABLES = ROOT / "data" / "processed" / "stanford"


def psi_of(runs, data, T_of, A):
    """Seed overdispersion from the training temperatures that have two runs."""
    d = []
    for T in sorted({T_of[r] for r in runs}):
        rs = [r for r in runs if T_of[r] == T]
        if len(rs) != 2:
            continue
        (n1, e1), (n2, e2) = (data[r][max(data[r])] for r in rs)
        n1, e1, n2, e2 = n1[A], e1[A], n2[A], e2[A]
        ok = (n1 >= 10) & (n2 >= 10)
        d.extend(((np.log(n1[ok] / e1[ok]) - np.log(n2[ok] / e2[ok])) / np.sqrt(1 / n1[ok] + 1 / n2[ok])).tolist())
    return max(1.0, float(np.mean(np.square(d)))) if d else 1.0


def internal(train, data, T_of, A, psi):
    """Leave-one-temperature-out among the training temperatures -> (z values, residuals, distances u - u_bar)."""
    z, res, dist = [], [], []
    temps = sorted({T_of[r] for r in train})
    for T in temps:
        inner = {r: L for r, L in train.items() if T_of[r] != T}
        Ts, n, E = design.pool(inner, data, T_of)
        lines = design.fit_lines(n[:, A], E[:, A], Ts)
        _, nh, Eh = design.pool({r: L for r, L in train.items() if T_of[r] == T}, data, T_of)
        nh, Eh = nh[0, A], Eh[0, A]
        keep = nh >= 10
        mean, sd = design.predict(lines, T)
        r = mean[keep] - np.log(nh[keep] / Eh[keep])
        z.extend((r / np.sqrt(psi * sd[keep] ** 2 + psi / nh[keep])).tolist())
        res.extend(r.tolist())
        S0 = (np.where(n[:, A] > 0, n[:, A], 0.5) * (E[:, A] > 0)).sum(0)
        ub = (np.where(n[:, A] > 0, n[:, A], 0.5) * (E[:, A] > 0) * (1000 / Ts[:, None])).sum(0) / S0
        dist.extend((1000 / T - ub[keep]).tolist())
    return np.array(z), np.array(res), np.array(dist)


def main():
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g, O = {}, None
    for rid in butane.run_id:
        (g[rid],), _, O, _ = load_stanford(TABLES, [rid])
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    full = {r: g[r].ev.shape[1] for r in g}
    data = {r: design.truncations(g[r], O, [full[r]]) for r in g}
    names = ["base", "t", "psi", "loo", "disc", "psi+t", "psi+loo", "psi+disc", "psi+t+disc"]
    z = {k: [] for k in names}
    for T in sorted(set(T_of.values())):
        train = {r: L for r, L in full.items() if T_of[r] != T}
        A = fit.anchors([g[r] for r in train])
        Ts, n, E = design.pool(train, data, T_of)
        lines = design.fit_lines(n[:, A], E[:, A], Ts)
        _, nh, Eh = design.pool({r: L for r, L in full.items() if T_of[r] == T}, data, T_of)
        nh, Eh = nh[0, A], Eh[0, A]
        keep = nh >= 10
        mean, sd = design.predict(lines, T)
        r = mean[keep] - np.log(nh[keep] / Eh[keep])
        sd, nk = sd[keep], nh[keep]
        m = lines["m"][keep]
        tq = np.where(m >= 3, stats.t.ppf(0.975, np.maximum(m - 2, 1)) / 1.96, 1.0)
        psi = psi_of(list(train), data, T_of, A)
        S0 = (np.where(n[:, A] > 0, n[:, A], 0.5) * (E[:, A] > 0)).sum(0)
        ub = ((np.where(n[:, A] > 0, n[:, A], 0.5) * (E[:, A] > 0) * (1000 / Ts[:, None])).sum(0) / S0)[keep]
        for k in names:
            p = psi if "psi" in k else 1.0
            zi, ri, di = internal(train, data, T_of, A, p)
            kappa = float(np.median(np.abs(zi)) / 0.674) if "loo" in k else 1.0
            tau2 = max(0.0, float(np.mean(ri ** 2 - p * 0.0)) / max(float(np.mean(di ** 2)), 1e-12)) if "disc" in k else 0.0
            if "disc" in k:  # tau^2: residual variance beyond the predicted variance, per squared distance
                tau2 = max(0.0, float(np.mean(ri ** 2) - np.mean(np.square(ri / np.where(zi != 0, zi, np.inf))))) / max(float(np.mean(di ** 2)), 1e-12)
            var = (p * (sd * (tq if "+t" in k or k == "t" else 1.0)) ** 2) * kappa ** 2 + tau2 * (1000 / T - ub) ** 2
            z[k].extend((r / np.sqrt(var + p / nk)).tolist())
        print(f"fold {T:g} K: psi {psi:.2f}", flush=True)
    res = {}
    for k in names:
        a = np.abs(z[k])
        cov, med = float((a <= 2).mean()), float(np.median(a))
        res[k] = {"coverage": cov, "median_abs_z": med, "score": abs(cov - 0.954) + abs(med - 0.674)}
        print(f"{k:>12}: |z| <= 2 for {cov:.3f}, median |z| {med:.3f}, score {res[k]['score']:.3f}")
    out = ROOT / "results" / "round2" / "dev_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
