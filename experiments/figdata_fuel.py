# file: experiments/figdata_fuel.py
"""Descriptive data for the Fuel figures (results/figdata/*.json). Nothing here is a test: it re-applies the frozen
code (round-6 C4 model, the counting of round 4, the Firth line of round 8) to data already used, and checks itself
against the stored official results where they overlap.

  python -I -X utf8 experiments/figdata_fuel.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import design  # noqa: E402
import firth  # noqa: E402
import species  # noqa: E402

OUT = ROOT / "results" / "figdata"
TABLES = ROOT / "data" / "processed" / "stanford"


def butane():
    """Round 2's butane data and anchors (as round6_dev_c4.butane_target), whole runs only."""
    import torch
    import fit
    from data import load_stanford
    torch.set_num_threads(1)
    runs = pd.read_csv(TABLES / "runs.csv")
    runs = runs[runs.molecule == "C4H10"]
    g, O = {}, None
    for rid in runs.run_id:
        (g[rid],), S, O, classes = load_stanford(TABLES, [rid])
    T_of = dict(zip(runs.run_id, runs.T_K.astype(float)))
    nb = {r: g[r].ev.shape[1] for r in g}
    data = {r: design.truncations(g[r], O, [nb[r]]) for r in g}
    return g, T_of, nb, data, fit.anchors(list(g.values())), classes


def label_butane(j, classes):
    tp = json.loads((TABLES / "event_types.json").read_text())[j]
    name = lambda c: {"C3_C4": "C3–C4", "C3_C4_rad": "C3–C4•", "C2_rad": "C2•", "C5_C10": "C5–C10", "C5_C10_rad": "C5–C10•",
                      "C11plus": "C11+", "radical_small": "R•", "H": "H•"}.get(c, c)
    nu = dict(zip(classes, tp["nu"]))
    prod = [name(c) for c in tp["reactants"] if nu[c] == 0] + [name(c) for c in classes if nu[c] > 0]  # a class with no net change persists
    return " + ".join(name(c) for c in tp["reactants"]) + " → " + " + ".join(prod)


def label_lab(rx):
    side = lambda s: " + ".join((f"{nu} " if nu > 1 else "") + sp for sp, nu in s)
    return side(rx[0]) + " → " + side(rx[1])


def arrhenius(camp, data, T_of, A, T_star, obs, w, n_test, labels, pick, grid):
    """Per-run training points, the C4 line (round 2's random-effects line) and the round-6 chord band on a grid of
    temperatures, and the held-out rate, for the anchors in `pick` (positions in A)."""
    from round6_dev_c4 import variances
    mean, s2, sds = variances(camp, data, T_of, A, T_star)
    sd = sds["chord"]
    _, _, _, _, lines = species.calibrated_re_k(camp, data, T_of, A, T_star)
    band = []
    for T in grid:
        _, _, s = variances(camp, data, T_of, A, T)
        band.append(s["chord"][pick].tolist())
    band = np.array(band)
    out = []
    for k, j in enumerate(pick):
        pts = []
        for r, L in camp.items():
            _, n, E = design.pool({r: L}, data, T_of)
            nj, Ej = float(n[0, A[j]]), float(E[0, A[j]])
            if nj > 0 and Ej > 0:
                pts.append({"T": T_of[r], "lnk": float(np.log(nj / Ej)), "n": nj})
        z = float((mean[j] - obs[j]) / np.sqrt(sd[j] ** 2 + 1.0 / max(w[j], 1.0) + s2 / n_test))
        out.append({"label": labels[j], "train": pts, "a": float(lines["a"][j]), "b": float(lines["b"][j]),
                    "grid": list(map(float, grid)), "line": [float(lines["a"][j] - lines["b"][j] * 1000.0 / T) for T in grid],
                    "sd": band[:, k].tolist(), "obs": float(obs[j]), "obs_n": float(w[j]), "pred": float(mean[j]),
                    "pred_sd": float(sd[j]), "z": z, "train_events": None})
    return out


def pick3(w, events):
    """Three anchors with >= 10 held-out events: the most frequent in training, one a third and one two thirds down."""
    ok = np.flatnonzero(w >= 10)
    order = ok[np.argsort(-events[ok], kind="stable")]
    return [int(order[0]), int(order[len(order) // 3]), int(order[2 * len(order) // 3])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import round4_lab as r4

    # --- lab, MEA, held-out 2500 K, full campaign (C4 as confirmed in round 6) ---
    res = {}
    for chem, T_star in (("MEA", 2500.0), ("HFO", 2500.0)):
        T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup(chem, T_star)
        obs, w = r4.observed(test, data, T_of, A)
        camp = {r: 1 for r in menu}
        mean, _, s2, _, _ = species.calibrated_re_k(camp, data, T_of, A, T_star)
        rms = float(np.sqrt((w * np.minimum(np.abs(mean - obs), 10.0) ** 2).sum() / w.sum()))
        stored = json.loads((ROOT / "results" / "round4" / f"fold_{chem}_{T_star:g}.json").read_text())["campaigns"]["full"]["rms"]
        assert np.isclose(rms, stored, rtol=0, atol=1e-10), (chem, rms, stored)
        ev_train = sum(data[r][1][0] for r in menu)[A]
        if chem == "MEA":
            import lab
            from c5_lab import CHEMS
            _, rx, _ = lab.tables(CHEMS[chem])
            labels = [label_lab(rx[j]) for j in A]
            pick = pick3(w, ev_train)
            res["MEA_arrhenius"] = arrhenius(camp, data, T_of, A, T_star, obs, w, len(test), labels, pick, np.linspace(2400.0, 3500.0, 23))
            for k, j in enumerate(pick):
                res["MEA_arrhenius"][k]["train_events"] = float(ev_train[j])
        res[f"{chem}_runs"] = [[T_of[r], float(data[r][1][0].sum()), 200.0] for r in T_of]  # T, events, ps
        res[f"{chem}_rank"] = sorted((sum(data[r][1][0] for r in T_of)).tolist(), reverse=True)
        res[f"{chem}_anchors"] = int(len(A))
        print(chem, "done; full-campaign RMS matches round 4:", round(rms, 4))

    # --- butane, held-out 3100 K, full campaign ---
    g, T_of, nb, data, A_all, classes = butane()
    T_star = 3100.0
    target = [r for r in g if T_of[r] == T_star]
    menu = {r: nb[r] for r in g if T_of[r] != T_star}
    import fit
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in target}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))
    ev_train = sum(data[r][nb[r]][0] for r in menu)[A]
    pick = pick3(w, ev_train)
    labels = [label_butane(int(j), classes) for j in A]
    res["butane_arrhenius"] = arrhenius(menu, data, T_of, A, T_star, obs, w, len(target), labels, pick, np.linspace(3000.0, 5000.0, 26))
    for k, j in enumerate(pick):
        res["butane_arrhenius"][k]["train_events"] = float(ev_train[j])
    res["butane_runs"] = [[T_of[r], float(data[r][nb[r]][0].sum()), 1.2 * nb[r]] for r in g]  # 1.2 ps per event bin
    res["butane_rank"] = sorted(sum(data[r][nb[r]][0] for r in g).tolist(), reverse=True)
    res["butane_anchors"] = int(len(A_all))
    print("butane done")

    # --- zero counts and the Firth line (the round-7 Monte Carlo, tests/test_firth.py's reaction) ---
    T = np.array([3000.0, 3300.0, 3600.0, 4000.0, 4500.0, 5000.0])
    U, RUNS = 1000.0 / T, np.ones(6)
    a, b = 1.0, 20.0
    lam = np.array([0.003, 0.01, 0.05, 5.0, 15.0, 40.0])
    E = (lam / np.exp(a - b * U))[:, None]
    targets = np.linspace(2800.0, 5000.0, 23)
    rng = np.random.default_rng(1)
    est = {"current": [], "poisson_ml": [], "firth": []}
    example = None
    for i in range(400):
        n = rng.poisson(lam)[:, None].astype(float)
        fits = {"current": design.fit_lines_re(n, E, T, RUNS, 0.0), "poisson_ml": design.fit_lines(n, E, T),
                "firth": firth.fit_lines_firth(n, E, T, RUNS, 0.0)}
        for k, f in fits.items():
            est[k].append([float(f["a"][0] - f["b"][0] * 1000.0 / Ts) - (a - b * 1000.0 / Ts) for Ts in targets])
        if example is None and (n[:3, 0] == 0).sum() >= 2 and n[3:, 0].min() > 0:
            example = {"n": n[:, 0].tolist(), "lnE": np.log(E[:, 0]).tolist(),
                       "lines": {k: [float(f["a"][0]), float(f["b"][0])] for k, f in fits.items()}}
    res["zero_count"] = {"T": T.tolist(), "a": a, "b": b, "lam": lam.tolist(), "targets": targets.tolist(), "example": example,
                         "bias": {k: np.mean(v, 0).tolist() for k, v in est.items()},
                         "q25": {k: np.percentile(v, 25, 0).tolist() for k, v in est.items()},
                         "q75": {k: np.percentile(v, 75, 0).tolist() for k, v in est.items()}}
    mid = list(targets).index(3600.0)
    assert res["zero_count"]["bias"]["current"][mid] > 1.0 and abs(res["zero_count"]["bias"]["firth"][mid]) < 0.2  # as tests/test_firth.py
    (OUT / "figdata.json").write_text(json.dumps(res), encoding="utf-8")
    print("wrote results/figdata/figdata.json")


if __name__ == "__main__":
    main()
