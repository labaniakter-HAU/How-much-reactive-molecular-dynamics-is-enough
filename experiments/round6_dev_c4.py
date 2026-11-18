# file: experiments/round6_dev_c4.py
"""Round 6 development for C4 (design section 17): variants of C4's predictive variance, scored by H5' on development
data only (synthetic seeds below 40000, butane, HFO; never MEA). The mean prediction is C4's (random-effects line);
only the variance changes:
  frozen   line variance (phi) + tau^2 (u* - u_bar)^2, tau^2 by internal leave-one-temperature-out (species.calibrated_re_k)
  chord    line variance (phi) + gamma^2 chord(u*)^2, gamma^2 from the campaign (design.curvature2) or the prior 70^2
  max      the larger of the two discrepancy terms, per reaction
  sum      both discrepancy terms
  chord_j  as chord, with each reaction's own curvature where its quadratic is estimable (>= 3 temperatures)

  python -I -X utf8 experiments/round6_dev_c4.py --synthetic 20000 20100 --target 3100 --workers 2
  python -I -X utf8 experiments/round6_dev_c4.py --hfo
Outputs: results/round6/dev_c4_<set>.json (z values per variant)
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import design  # noqa: E402
import species  # noqa: E402

OUT = ROOT / "results" / "round6"
VARIANTS = ("frozen", "chord", "max", "sum", "chord_j")


def variances(camp, data, T_of, A, T_star, min_events=10):
    """-> C4 mean (len(A),), s2, {variant: predictive sd (len(A),)}."""
    mean, sd_frozen, s2, tau2, lines = species.calibrated_re_k(camp, data, T_of, A, T_star)
    Ts, n, E = design.pool(camp, data, T_of)
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    n, E = n[:, A], E[:, A]
    us = 1000.0 / T_star
    linevar = np.square(design.predict(lines, T_star)[1])  # x*' Sigma x*, phi included
    g2 = design.curvature2(camp, data, T_of, A, T_star, s2)
    g2 = design.GAMMA_PRIOR ** 2 if g2 is None else g2
    use = E > 0
    w = np.where(use, 1.0 / (1.0 / np.where(n > 0, n, 0.5) + s2 / runs[:, None]), 0.0)
    x = (1000.0 / Ts - us)[:, None]
    S0 = w.sum(0)
    S0s = np.where(S0 > 0, S0, 1.0)
    m1, m2, m3 = ((w * x ** p).sum(0) / S0s for p in (1, 2, 3))
    var_x = m2 - m1 ** 2
    two = (use.sum(0) >= 2) & (var_x > 1e-12)
    chord2 = np.where(two, (m2 - m1 * (m3 - m1 * m2) / np.where(two, var_x, 1.0)) ** 2, np.inf)
    tau_term = np.where(np.isfinite(sd_frozen), sd_frozen ** 2 - linevar, np.inf)
    c2j = np.full(len(A), np.nan)
    if len(Ts) >= 3:
        X = np.stack([np.ones(len(Ts)), x[:, 0], x[:, 0] ** 2], 1)
        for j in range(len(A)):
            ok = (n[:, j] >= min_events) & (E[:, j] > 0)
            if ok.sum() < 3:
                continue
            W = 1.0 / (1.0 / n[ok, j] + s2 / runs[ok])
            M = X[ok].T @ (W[:, None] * X[ok])
            try:
                Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                continue
            cj = (Minv @ (X[ok].T @ (W * np.log(n[ok, j] / E[ok, j]))))[2]
            c2j[j] = max(0.0, cj * cj - Minv[2, 2])
    chord_term = g2 * chord2
    out = {"frozen": sd_frozen, "chord": np.sqrt(linevar + chord_term), "max": np.sqrt(linevar + np.maximum(chord_term, tau_term)),
           "sum": np.sqrt(linevar + chord_term + tau_term), "chord_j": np.sqrt(linevar + np.where(np.isfinite(c2j), c2j, g2) * chord2)}
    return mean, s2, out


def synthetic_one(args):
    """Development: seeds below 40000 only."""
    seed, T_STAR = args
    assert seed < 40000, "development seeds only"
    return synthetic_core(seed, T_STAR)


def synthetic_core(seed, T_STAR):
    """As round2_synthetic.one (same network and data per seed); z of the full campaign against the truth."""
    from round5_flag import kind5
    from synthetic import random_network, run_data, true_lnk
    from v0b_reduction import RX, SPECIES, V, X0
    from v0c_synthetic import RUNS
    net = random_network(RX, seed, kind5(seed))
    rng = np.random.default_rng(seed + 100000)
    names = [f"{T}K_{s}" for T, s, _ in RUNS]
    T_of = {nm: float(T) for nm, (T, _, _) in zip(names, RUNS)}
    bins = {nm: b for nm, (_, _, b) in zip(names, RUNS)}
    menu = {nm: [L for L in (50, 100, 200) if L < bins[nm]] + [bins[nm]] for nm in names if T_of[nm] != T_STAR}
    data = {}
    for nm in names:
        data[nm], _ = run_data(net, SPECIES, T_of[nm], bins[nm], X0, V, menu.get(nm, [bins[nm]]), rng, n_sub=4)
    truth = true_lnk(net, T_STAR)
    A = np.arange(len(RX))
    full = {nm: bins[nm] for nm in menu}
    Ts, n, E = design.pool(full, data, T_of)
    keep = n.sum(0) >= 10
    mean, s2, sds = variances(full, data, T_of, A, T_STAR)
    return {"seed": seed, "kind": kind5(seed), "target": T_STAR,
            "z": {k: ((mean - truth) / v)[keep].round(4).tolist() for k, v in sds.items()}}


def hfo():
    """Every non-full campaign of the HFO folds (C5's campaigns), z as round 4 (observation variance 1/n* + s2/4)."""
    import round4_lab as r4
    out = {k: [] for k in VARIANTS}
    for T_star in r4.TEMPS:
        T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup("HFO", T_star)
        camps = r4.campaigns("HFO", T_star, T_of, data, menu, A, pilot, test)
        obs, w = r4.observed(test, data, T_of, A)
        keep = w >= 10
        for name, camp in camps.items():
            if name == "full":
                continue
            mean, s2, sds = variances(camp, data, T_of, A, T_star)
            for k, sd in sds.items():
                out[k] += ((mean - obs) / np.sqrt(sd ** 2 + 1.0 / np.maximum(w, 1.0) + s2 / len(test)))[keep].round(4).tolist()
    return out


def butane_target(T_star):
    """Round 2's real C4 test (round2_confirm.design_target: same menu, pilot, campaigns and rng), every non-full
    campaign, z with observation variance 1/n* + s2/runs*."""
    import pandas as pd
    import torch
    import fit
    from data import load_stanford
    from round2_confirm import BUDGETS, TABLES
    torch.set_num_threads(1)
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g = {}
    for rid in butane.run_id:
        (g[rid],), S, O, classes = load_stanford(TABLES, [rid])
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    nb = {r: g[r].ev.shape[1] for r in g}
    target = [r for r in g if T_of[r] == T_star]
    menu = {r: [L for L in (50, 100, 200) if L < nb[r]] + [nb[r]] for r in g if T_of[r] != T_star}
    data = {r: design.truncations(g[r], O, menu.get(r, [nb[r]])) for r in g}
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in target}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))
    keep = w >= 10
    pilot_T = 4500.0 if T_star == 5000.0 else 5000.0
    pilot = {[r for r in menu if T_of[r] == pilot_T and "_1_" in r][0]: 50}
    total, rng = sum(nb[r] for r in menu), np.random.default_rng(int(T_star))
    camps = []
    for f in BUDGETS:
        Bud = int(f * total)
        camps += [design.greedy(menu, pilot, Bud, T_of, T_star), design.hot_only(menu, pilot, Bud, T_of), design.uniform(menu, pilot, Bud, T_of)]
        camps += [design.random_campaign(menu, pilot, Bud, rng) for _ in range(100)]
    out = {k: [] for k in VARIANTS}
    for camp in camps:
        mean, s2, sds = variances(camp, data, T_of, A, T_star)
        for k, sd in sds.items():
            out[k] += ((mean - obs) / np.sqrt(sd ** 2 + 1.0 / np.maximum(w, 1.0) + s2 / len(target)))[keep].round(4).tolist()
    return out


def h5(z):
    a = np.abs(np.asarray(z, float))
    a = a[np.isfinite(a)]
    cov, med = float((a <= 2).mean()), float(np.median(a))
    return {"n": int(a.size), "coverage": round(cov, 4), "median": round(med, 3), "pass": bool(cov >= 0.85 and 0.5 <= med <= 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, nargs=2)
    ap.add_argument("--target", type=float, default=3100.0)
    ap.add_argument("--hfo", action="store_true")
    ap.add_argument("--butane", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        with Pool(a.workers) as pool:
            rows = pool.map(synthetic_one, [(s, a.target) for s in range(*a.synthetic)])
        path = OUT / f"dev_c4_syn_{a.synthetic[0]}_{a.synthetic[1]}_T{a.target:g}.json"
        path.write_text(json.dumps(rows))
        kinds = sorted({r["kind"] for r in rows})
        print(f"synthetic {a.synthetic} T {a.target:g} ({', '.join(kinds)}):")
        for k in VARIANTS:
            print(f"   {k:>8}: {h5([z for r in rows for z in r['z'][k]])}")
    if a.butane:
        with Pool(a.workers) as pool:
            parts = pool.map(butane_target, [3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 4000.0, 4500.0, 5000.0])
        out = {k: [z for p_ in parts for z in p_[k]] for k in VARIANTS}
        (OUT / "dev_c4_butane.json").write_text(json.dumps(out))
        print("butane, round 2's 8 design targets, non-full campaigns:")
        for k in VARIANTS:
            print(f"   {k:>8}: {h5(out[k])}")
    if a.hfo:
        out = hfo()
        (OUT / "dev_c4_hfo.json").write_text(json.dumps(out))
        print("HFO, non-full campaigns:")
        for k in VARIANTS:
            print(f"   {k:>8}: {h5(out[k])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
