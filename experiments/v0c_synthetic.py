# file: experiments/v0c_synthetic.py
"""V0c (design section 11, first Part II test): the uncertainty and design machinery on V0b's 27-reaction
network with known rates (no MD, no lumping). A synthetic twin of the Stanford menu: the same temperatures,
seeds and lengths, 64 butane molecules in 3.3 nm3. Exposures come from the noise-free trajectory (MD knows
its own trajectory exactly) and events per 1.2 ps bin are Poisson with mean k x exposure. 20 replicates.

  python -I experiments/v0c_synthetic.py
Calibration: rates from every temperature except 3100 K, predicted at 3100 K for each reaction with >= 10
training events; z = (predicted - true ln k) / predicted sd. Pass: >= 85 % of |z| <= 2 and
0.5 <= median |z| <= 1.0.
Design: target 3100 K, budgets 10, 25, 50 % of the menu, pilot 60 ps of 5000 K seed 1; metric: RMS of
predicted - true ln k at 3100 K weighted by expected events there (|error| clipped at 10). Pass: designed
<= 0.8 x median random at every budget (both averaged over replicates).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import design  # noqa: E402
import fit  # noqa: E402
from data import Group  # noqa: E402
from kinetics import R_KJ  # noqa: E402
from synthetic import network  # noqa: E402
from v0b_reduction import EA, RX, SPECIES, V, X0  # noqa: E402

RUNS = [(T, s, 500) for T in (3100, 3200, 3300, 3400, 3500) for s in (1, 2)] + [(3600, 1, 500)] + \
       [(4000, s, 200) for s in (1, 2)] + [(4500, s, 100) for s in (1, 2)] + [(5000, 1, 100), (5000, 2, 50)]
T_STAR, BUDGETS, N_REP, N_RANDOM, CLIP = 3100.0, (0.10, 0.25, 0.50), 20, 100, 10.0


def clean_exposure(truth, T, bins):
    """-> exposure per 1.2 ps bin and reaction (bins, n_r) along the noise-free trajectory."""
    t = torch.linspace(0, 1.2 * bins, 2 * bins + 1, dtype=torch.float64)
    c0 = torch.as_tensor(np.asarray(X0, float) / V)[None]
    g = Group(t, c0[None], torch.tensor([float(T)], dtype=torch.float64), torch.tensor([V], dtype=torch.float64), None)
    with torch.no_grad():
        c = fit.simulate(truth, g, n_sub=8)[0][:, 0].numpy()
    rate = np.exp(np.log(np.clip(c, 1e-12, None)) @ truth.orders.numpy()) * V
    cum = np.concatenate([[np.zeros(rate.shape[1])], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t.numpy())[:, None], 0)])
    return np.diff(cum[::2], axis=0)


def rms(pred, true, w):
    return float(np.sqrt((w * np.minimum(np.abs(pred - true), CLIP) ** 2).sum() / w.sum()))


def main():
    torch.set_num_threads(1)
    truth = network(SPECIES, [r[:2] for r in RX], EA, [r[3] for r in RX])
    lnk = lambda T: np.log([r[3] for r in RX]) - EA / (R_KJ * T)
    names = [f"{T}K_{s}" for T, s, _ in RUNS]
    T_of = {nm: float(T) for nm, (T, _, _) in zip(names, RUNS)}
    expo = {nm: clean_exposure(truth, T, b) for nm, (T, _, b) in zip(names, RUNS)}
    menu_runs = [nm for nm in names if T_of[nm] != T_STAR]
    menu = {nm: [L for L in (50, 100, 200) if L < len(expo[nm])] + [len(expo[nm])] for nm in menu_runs}
    target = [nm for nm in names if T_of[nm] == T_STAR]
    w = np.exp(lnk(T_STAR)) * sum(expo[nm].sum(0) for nm in target)  # expected events at 3100 K
    total, pilot = sum(len(expo[nm]) for nm in menu_runs), {"5000K_1": 50}
    rng = np.random.default_rng(0)
    camps = {}
    for f in BUDGETS:
        B = int(f * total)
        camps[f] = {"designed": design.greedy(menu, pilot, B, T_of, T_STAR), "hot_only": design.hot_only(menu, pilot, B, T_of),
                    "uniform": design.uniform(menu, pilot, B, T_of)}
        camps[f].update({f"random_{i}": design.random_campaign(menu, pilot, B, rng) for i in range(N_RANDOM)})
    full = {nm: max(menu[nm]) for nm in menu_runs}

    z, z0, err = [], [], {f: {k: [] for k in camps[f]} for f in BUDGETS}
    for rep in range(N_REP):
        r = np.random.default_rng(1000 + rep)
        ev = {nm: np.cumsum(r.poisson(np.exp(lnk(T_of[nm])) * expo[nm]), 0) for nm in names}
        data = {nm: {L: (ev[nm][L - 1], expo[nm][:L].sum(0)) for L in menu.get(nm, [len(expo[nm])])} for nm in names}
        Ts, n, E = design.pool(full, data, T_of)
        lines = design.fit_lines(n, E, Ts)
        keep = n.sum(0) >= 10
        for out, phi in ((z, True), (z0, False)):
            mean, sd = design.predict(lines, T_STAR, with_phi=phi)
            out.extend(((mean - lnk(T_STAR)) / sd)[keep].tolist())
        for f in BUDGETS:
            for k, camp in camps[f].items():
                Ts, n, E = design.pool(camp, data, T_of)
                err[f][k].append(rms(design.predict(design.fit_lines(n, E, Ts), T_STAR)[0], lnk(T_STAR), w))

    az, az0 = np.abs(z), np.abs(z0)
    res = {"calibration": {"n_z": len(z), "frac_abs_z_le_2": float((az <= 2).mean()), "median_abs_z": float(np.median(az)),
                           "without_phi": {"frac_abs_z_le_2": float((az0 <= 2).mean()), "median_abs_z": float(np.median(az0))}},
           "design": {}}
    cal = res["calibration"]
    cal["pass"] = bool(cal["frac_abs_z_le_2"] >= 0.85 and 0.5 <= cal["median_abs_z"] <= 1.0)
    for f in BUDGETS:
        mean_err = {k: float(np.mean(v)) for k, v in err[f].items()}
        med_random = float(np.mean([np.median([err[f][f"random_{i}"][rep] for i in range(N_RANDOM)]) for rep in range(N_REP)]))
        res["design"][str(f)] = {"designed": mean_err["designed"], "hot_only": mean_err["hot_only"], "uniform": mean_err["uniform"],
                                 "median_random": med_random, "designed_campaign": {k: v for k, v in camps[f]["designed"].items() if v},
                                 "pass": bool(mean_err["designed"] <= 0.8 * med_random)}
    res["pass"] = bool(cal["pass"] and all(d["pass"] for d in res["design"].values()))
    out = ROOT / "results" / "v0c" / "v0c.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"calibration: {len(z)} z values, |z| <= 2 for {cal['frac_abs_z_le_2']:.3f}, median |z| {cal['median_abs_z']:.3f} "
          f"(without phi {cal['without_phi']['frac_abs_z_le_2']:.3f}, {cal['without_phi']['median_abs_z']:.3f}): {'PASS' if cal['pass'] else 'FAIL'}")
    for f, d in res["design"].items():
        print(f"budget {float(f):.0%}: designed {d['designed']:.3f}, median random {d['median_random']:.3f}, hot-only {d['hot_only']:.3f}, "
              f"uniform {d['uniform']:.3f}: {'PASS' if d['pass'] else 'FAIL'}  {d['designed_campaign']}")
    print(f"**V0c {'PASS' if res['pass'] else 'FAIL'}**")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
