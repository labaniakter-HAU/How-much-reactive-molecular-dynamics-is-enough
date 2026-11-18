# file: experiments/round2_dev_design.py
"""Round 2 development (design section 12), C5: design variants on DEVELOPMENT data only, i.e. the 3100 K target
of the butane runs (V7) and V0c's synthetic network (replicate seeds 1000-1019). The confirmation targets
(every other butane temperature) are never read here.

  python -I experiments/round2_dev_design.py
Prints, per budget, each variant's error relative to the median of 100 random campaigns.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import design  # noqa: E402
import fit  # noqa: E402
from data import load_stanford  # noqa: E402

TABLES, T_STAR, BUDGETS, CLIP = ROOT / "data" / "processed" / "stanford", 3100.0, (0.10, 0.25, 0.50), 10.0
TAUS = (0.0, 3.0, 10.0, 30.0, 100.0)


def variants(menu, pilot, B, T_of, data, A):
    out = {"static": design.greedy(menu, pilot, B, T_of, T_STAR), "nearest_first": design.nearest_first(menu, pilot, B, T_of, T_STAR),
           "hot_only": design.hot_only(menu, pilot, B, T_of), "uniform": design.uniform(menu, pilot, B, T_of)}
    out.update({f"sequential_tau{t:g}": design.sequential(menu, pilot, B, T_of, T_STAR, data, A, tau2=t) for t in TAUS})
    return out


def rms(camp, data, T_of, A, truth, w):
    Ts, n, E = design.pool(camp, data, T_of)
    pred = design.predict(design.fit_lines(n[:, A], E[:, A], Ts), T_STAR)[0]
    return float(np.sqrt((w * np.minimum(np.abs(pred - truth), CLIP) ** 2).sum() / w.sum()))


def real():
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g = {}
    for rid in butane.run_id:
        (g[rid],), S, O, _ = load_stanford(TABLES, [rid])
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    nb = {r: g[r].ev.shape[1] for r in g}
    target = [r for r in g if T_of[r] == T_STAR]
    menu = {r: [L for L in (50, 100, 200) if L < nb[r]] + [nb[r]] for r in g if T_of[r] != T_STAR}
    data = {r: design.truncations(g[r], O, menu.get(r, [nb[r]])) for r in g}
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in target}, data, T_of)
    w, truth = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))
    total, pilot, rng, res = sum(nb[r] for r in menu), {"C4H10_896at_1_5000K_40GPa_1Ms": 50}, np.random.default_rng(0), {}
    for f in BUDGETS:
        B = int(f * total)
        ev = {k: rms(c, data, T_of, A, truth, w) for k, c in variants(menu, pilot, B, T_of, data, A).items()}
        med = float(np.median([rms(design.random_campaign(menu, pilot, B, rng), data, T_of, A, truth, w) for _ in range(100)]))
        res[str(f)] = {"median_random": med, **{k: v / med for k, v in ev.items()}}
    res["full_over_random_50"] = rms({r: nb[r] for r in menu}, data, T_of, A, truth, w) / res["0.5"]["median_random"]
    return res


def synthetic():
    from kinetics import R_KJ
    from synthetic import network
    from v0b_reduction import EA, RX, SPECIES
    from v0c_synthetic import RUNS, clean_exposure
    torch.set_num_threads(1)
    truth_net = network(SPECIES, [r[:2] for r in RX], EA, [r[3] for r in RX])
    lnk = lambda T: np.log([r[3] for r in RX]) - EA / (R_KJ * T)
    names = [f"{T}K_{s}" for T, s, _ in RUNS]
    T_of = {nm: float(T) for nm, (T, _, _) in zip(names, RUNS)}
    expo = {nm: clean_exposure(truth_net, T, b) for nm, (T, _, b) in zip(names, RUNS)}
    menu = {nm: [L for L in (50, 100, 200) if L < len(expo[nm])] + [len(expo[nm])] for nm in names if T_of[nm] != T_STAR}
    w = np.exp(lnk(T_STAR)) * sum(expo[nm].sum(0) for nm in names if T_of[nm] == T_STAR)
    A, total, pilot = np.arange(len(RX)), sum(len(expo[nm]) for nm in menu), {"5000K_1": 50}
    acc = {str(f): {} for f in BUDGETS}
    for rep in range(20):
        r = np.random.default_rng(1000 + rep)
        ev = {nm: np.cumsum(r.poisson(np.exp(lnk(T_of[nm])) * expo[nm]), 0) for nm in names}
        data = {nm: {L: (ev[nm][L - 1], expo[nm][:L].sum(0)) for L in menu.get(nm, [len(expo[nm])])} for nm in names}
        rng = np.random.default_rng(0)
        for f in BUDGETS:
            B = int(f * total)
            evs = {k: rms(c, data, T_of, A, lnk(T_STAR), w) for k, c in variants(menu, pilot, B, T_of, data, A).items()}
            evs["median_random"] = float(np.median([rms(design.random_campaign(menu, pilot, B, rng), data, T_of, A, lnk(T_STAR), w) for _ in range(100)]))
            for k, v in evs.items():
                acc[str(f)].setdefault(k, []).append(v)
    return {f: {"median_random": float(np.mean(d["median_random"])), **{k: float(np.mean(v)) / float(np.mean(d["median_random"]))
                                                                           for k, v in d.items() if k != "median_random"}} for f, d in acc.items()}


def main():
    res = {"real_3100K": real(), "synthetic_V0c": synthetic()}
    out = ROOT / "results" / "round2" / "dev_design.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for src, r in res.items():
        print(src)
        for f in map(str, BUDGETS):
            print(f"  {float(f):.0%}: " + ", ".join(f"{k} {v:.2f}" for k, v in r[f].items() if k != "median_random"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
