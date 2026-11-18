# file: experiments/round5_cache_c6.py
"""Round 5, C6 development cache (design section 16; no MEA file is read). For every campaign of a development set,
the raw ingredients any foresight score may use, so that score variants can be compared offline:
  per campaign: target T*, campaign temperatures Ts, runs per temperature, events n and exposures E per (temperature,
  anchor), the predicted flux per anchor at T* (the campaign's own model, as round 2/4), the domain ratio, and the label
  (event-weighted RMS of the frozen C4 prediction against counted ln k at T*; untrustworthy if > ln 2).

  python -I -X utf8 experiments/round5_cache_c6.py --hfo
  python -I -X utf8 experiments/round5_cache_c6.py --butane --workers 2
  python -I -X utf8 experiments/round5_cache_c6.py --synthetic 10000 10050 --target 3100 --workers 4
Outputs: results/round5/cache_c6_<set>.pkl (a list of campaign dicts).
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import design  # noqa: E402
import fit  # noqa: E402

OUT = ROOT / "results" / "round5"
# The .pkl caches are written and read only by this project's own round-5 scripts (local development data), never
# downloaded or shared, so pickle is safe here.
R2 = json.loads((ROOT / "results" / "round2" / "FROZEN.json").read_text())


def record(camp, data, T_of, A, T_star, flux, rms, dom, name, extra=None):
    Ts, n, E = design.pool(camp, data, T_of)
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    rr = [r for r, L in camp.items() if L > 0]
    out = {"name": name, "T": float(T_star), "Ts": Ts, "n": n[:, A], "E": E[:, A], "runs": runs,
           "flux": np.asarray(flux, float), "rms": float(rms), "dom": float(dom),
           "run_T": np.array([T_of[r] for r in rr]), "run_n": np.array([data[r][camp[r]][0][A] for r in rr]),
           "run_E": np.array([data[r][camp[r]][1][A] for r in rr])}
    out.update(extra or {})
    return out


def hfo():
    import round4_lab as r4
    import species
    res = json.loads((OUT / "dev_c6_hfo.json").read_text())
    rows = []
    for fold in res:
        T_star = fold["T"]
        T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup("HFO", T_star)
        camps = r4.campaigns("HFO", T_star, T_of, data, menu, A, pilot, test)
        for c in fold["campaigns"]:
            rows.append(record(camps[c["name"]], data, T_of, A, T_star, c["flux"], c["rms"], 1.0, c["name"],
                               {"sd_c4": np.array(c["sd"], float), "seen": np.array(c["seen"])}))
    return rows


def butane_target(T_star):
    import pandas as pd
    from data import load_stanford
    from kinetics import R_KJ, LumpedKinetics
    from round2_confirm import BUDGETS, CLIP, TABLES, uncertainty
    torch.set_num_threads(1)
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g = {}
    for rid in butane.run_id:
        (g[rid],), S, O, classes = load_stanford(TABLES, [rid])
    carbon = np.array(json.loads((TABLES / "classes.json").read_text())["composition"])[:, 0]
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    nb = {r: g[r].ev.shape[1] for r in g}
    target = [r for r in g if T_of[r] == T_star]
    menu = {r: [L for L in (50, 100, 200) if L < nb[r]] + [nb[r]] for r in g if T_of[r] != T_star}
    data = {r: design.truncations(g[r], O, menu.get(r, [nb[r]])) for r in g}
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in target}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))
    pilot_T = 4500.0 if T_star == 5000.0 else 5000.0
    pilot = {[r for r in menu if T_of[r] == pilot_T and "_1_" in r][0]: 50}
    total, rng = sum(nb[r] for r in menu), np.random.default_rng(int(T_star))
    camps = {"full": {r: nb[r] for r in menu}}
    for f in BUDGETS:
        Bud = int(f * total)
        camps.update({f"{f}/designed": design.greedy(menu, pilot, Bud, T_of, T_star), f"{f}/hot_only": design.hot_only(menu, pilot, Bud, T_of),
                      f"{f}/uniform": design.uniform(menu, pilot, Bud, T_of)})
        camps.update({f"{f}/random_{i}": design.random_campaign(menu, pilot, Bud, rng) for i in range(100)})
    rows = []
    for name, camp in camps.items():
        mean, sd, obs_var, lines = uncertainty(R2, camp, data, T_of, A, T_star)
        with torch.no_grad():
            flux = fit.simulate(LumpedKinetics(S[:, A], O[:, A], lines["a"], 10.0 * lines["b"] * R_KJ, carbon), g[target[0]])[1][-1, 0].numpy()
        rms = float(np.sqrt((w * np.minimum(np.abs(mean - obs), CLIP) ** 2).sum() / w.sum()))
        rows.append(record(camp, data, T_of, A, T_star, flux, rms, 1.0, name, {"sd_c4": sd, "seen": lines["seen"]}))
    return rows


def synthetic_one(args):
    seed, T_STAR = args
    from data import Group
    from round2_synthetic import BUDGETS, CLIP, kind_of
    from synthetic import network, random_network, run_data, true_lnk
    from v0b_reduction import RX, SPECIES, V, X0
    from v0c_synthetic import RUNS
    torch.set_num_threads(1)
    net = random_network(RX, seed, kind_of(seed))
    rng = np.random.default_rng(seed + 100000)
    names = [f"{T}K_{s}" for T, s, _ in RUNS]
    T_of = {nm: float(T) for nm, (T, _, _) in zip(names, RUNS)}
    bins = {nm: b for nm, (_, _, b) in zip(names, RUNS)}
    menu = {nm: [L for L in (50, 100, 200) if L < bins[nm]] + [bins[nm]] for nm in names if T_of[nm] != T_STAR}
    data = {}
    for nm in names:
        data[nm], _ = run_data(net, SPECIES, T_of[nm], bins[nm], X0, V, menu.get(nm, [bins[nm]]), rng, n_sub=4)
    truth = true_lnk(net, T_STAR)
    w = np.exp(truth) * sum(data[nm][bins[nm]][1] for nm in names if T_of[nm] == T_STAR)
    A = np.arange(len(RX))
    g0 = Group(torch.linspace(0, 1.2 * 500, 1001, dtype=torch.float64), torch.as_tensor(np.asarray(X0, float) / V)[None, None],
               torch.tensor([T_STAR], dtype=torch.float64), torch.tensor([V], dtype=torch.float64), None)
    total, pilot, crng = sum(bins[nm] for nm in menu), {"4500K_1" if T_STAR == 5000.0 else "5000K_1": 50}, np.random.default_rng(0)
    camps = {"full": {nm: bins[nm] for nm in menu}}
    for f in BUDGETS:
        Bud = int(f * total)
        cs = {"designed": design.greedy(menu, pilot, Bud, T_of, T_STAR), "hot_only": design.hot_only(menu, pilot, Bud, T_of),
              "uniform": design.uniform(menu, pilot, Bud, T_of)}
        rand = [design.random_campaign(menu, pilot, Bud, crng) for _ in range(100)]
        camps.update({f"{f}/{k}": c for k, c in cs.items()})
        camps.update({f"{f}/random_{i}": c for i, c in enumerate(rand[:3])})
    rows = []
    for name, camp in camps.items():
        _, _, _, _, lines = design.calibrated(camp, data, T_of, A, T_STAR)
        model = network(SPECIES, net["rx"], 1000.0 * 8.314462618e-3 * lines["b"], np.exp(lines["a"]))
        with torch.no_grad():
            flux = fit.simulate(model, g0, n_sub=4)[1][-1, 0].numpy()
        mean_re, sd_re = design.calibrated_re(camp, data, T_of, A, T_STAR)[:2]
        rms = float(np.sqrt((w * np.minimum(np.abs(mean_re - truth), CLIP) ** 2).sum() / w.sum()))
        rows.append(record(camp, data, T_of, A, T_STAR, flux, rms, 1.0, f"{seed}/{name}",
                           {"sd_c4": sd_re, "seen": lines["seen"], "seed": seed, "kind": kind_of(seed)}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hfo", action="store_true")
    ap.add_argument("--butane", action="store_true")
    ap.add_argument("--synthetic", type=int, nargs=2)
    ap.add_argument("--target", type=float, default=3100.0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.hfo:
        rows = hfo()
        pickle.dump(rows, open(OUT / "cache_c6_hfo.pkl", "wb"))
        print(f"HFO: {len(rows)} campaigns")
    if a.butane:
        with Pool(a.workers) as pool:
            rows = [r for rs in pool.map(butane_target, [3100.0] + [float(T) for T in R2["design_targets"]]) for r in rs]
        pickle.dump(rows, open(OUT / "cache_c6_butane.pkl", "wb"))
        print(f"butane: {len(rows)} campaigns")
    if a.synthetic:
        seeds = list(range(*a.synthetic))
        assert all(s < 24000 or 30000 <= s < 40000 for s in seeds), "development seeds only"
        with Pool(a.workers) as pool:
            rows = [r for rs in pool.map(synthetic_one, [(s, a.target) for s in seeds]) for r in rs]
        path = OUT / f"cache_c6_syn_{a.synthetic[0]}_{a.synthetic[1]}_T{a.target:g}.pkl"
        pickle.dump(rows, open(path, "wb"))
        print(f"synthetic {a.synthetic}: {len(rows)} campaigns -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
