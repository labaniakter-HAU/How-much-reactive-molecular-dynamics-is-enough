# file: experiments/v7_design.py
"""V7 / H6 (C5) and the primary part of V8 / H7 (C6), design section 11.

  python -I experiments/v7_design.py
Target 3100 K; both 3100 K runs are held out. Menu: the butane runs at 3200-5000 K, each at 60, 120 and 240 ps
where shorter than the run, and at full length. Pilot: the first 60 ps of the 5000 K seed-1 run, included in
every campaign. Budgets: 10, 25 and 50 % of the menu's picoseconds. Campaigns: designed (design.greedy),
hot-only, uniform, 100 random, and the full campaign. Rates per campaign from design.fit_lines on the anchors
of the full campaign; metric: RMS of predicted - counted ln k at 3100 K weighted by the events there
(|error| clipped at 10). H6: designed <= 0.8 x median random at every budget, and designed at 50 % <= 1.1 x
the full campaign.
V8 primary: every campaign's foresight flag, computed from its own data before its error is looked at,
against untrustworthy = RMS > 1. The flux is the anchored model with the campaign's rates, integrated at
3100 K from the target's initial state (pure butane). H7: >= 90 % correct overall and >= 80 % within each
class; each class needs >= 5 targets, else untested.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import design  # noqa: E402
import fit  # noqa: E402
from data import load_stanford  # noqa: E402
from kinetics import R_KJ, LumpedKinetics  # noqa: E402

TABLES = ROOT / "data" / "processed" / "stanford"
T_STAR, BUDGETS, N_RANDOM, CLIP = 3100.0, (0.10, 0.25, 0.50), 100, 10.0
PILOT = {"C4H10_896at_1_5000K_40GPa_1Ms": 50}


def main():
    torch.set_num_threads(1)
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g = {}
    for rid in butane.run_id:
        (g[rid],), S, O, classes = load_stanford(TABLES, [rid])
    carbon = np.array(json.loads((TABLES / "classes.json").read_text())["composition"])[:, 0]
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    nb = {r: g[r].ev.shape[1] for r in g}
    target = [r for r in g if T_of[r] == T_STAR]
    menu = {r: [L for L in (50, 100, 200) if L < nb[r]] + [nb[r]] for r in g if T_of[r] != T_STAR}
    data = {r: design.truncations(g[r], O, menu.get(r, [nb[r]])) for r in g}
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in target}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))
    g0 = g[target[0]]  # only its t = 0 state, box and time grid are used
    present = g0.obs[0, 0].numpy() > 0
    SA, OA = S[:, A], O[:, A]

    def evaluate(camp):
        Ts, n, E = design.pool(camp, data, T_of)
        lines = design.fit_lines(n[:, A], E[:, A], Ts)
        model = LumpedKinetics(SA, OA, lines["a"], 10.0 * lines["b"] * R_KJ, carbon)
        with torch.no_grad():
            flux = fit.simulate(model, g0)[1][-1, 0].numpy()
        fa, fb = design.flags(lines, T_STAR, SA, OA, present, flux)  # before the error is computed
        err = np.minimum(np.abs(design.predict(lines, T_STAR)[0] - obs), CLIP)
        rms = float(np.sqrt((w * err ** 2).sum() / w.sum()))
        per_T = {f"{T:g}": round(1.2 * sum(L for r, L in camp.items() if T_of[r] == T), 1) for T in Ts}
        return {"rms": rms, "flag_channel": fa, "flag_flux": fb, "ps": round(1.2 * sum(camp.values()), 1), "ps_per_T": per_T}

    total = sum(nb[r] for r in menu)
    rng, res = np.random.default_rng(0), {"anchors": int(len(A)), "menu_ps": round(1.2 * total, 1), "budgets": {}}
    res["full"] = evaluate({r: nb[r] for r in menu})
    print(f"full campaign ({res['full']['ps']} ps): RMS {res['full']['rms']:.3f}", flush=True)
    for f in BUDGETS:
        B = int(f * total)
        camps = {"designed": design.greedy(menu, PILOT, B, T_of, T_STAR), "hot_only": design.hot_only(menu, PILOT, B, T_of),
                 "uniform": design.uniform(menu, PILOT, B, T_of)}
        camps.update({f"random_{i}": design.random_campaign(menu, PILOT, B, rng) for i in range(N_RANDOM)})
        ev = {k: evaluate(c) for k, c in camps.items()}
        med = float(np.median([ev[f"random_{i}"]["rms"] for i in range(N_RANDOM)]))
        res["budgets"][str(f)] = {"campaigns": ev, "median_random": med, "pass": bool(ev["designed"]["rms"] <= 0.8 * med)}
        print(f"budget {f:.0%} ({1.2 * B:.0f} ps): designed {ev['designed']['rms']:.3f}, median random {med:.3f}, "
              f"hot-only {ev['hot_only']['rms']:.3f}, uniform {ev['uniform']['rms']:.3f}; designed ps per T {ev['designed']['ps_per_T']}", flush=True)
    d50 = res["budgets"]["0.5"]["campaigns"]["designed"]["rms"]
    res["H6"] = bool(all(b["pass"] for b in res["budgets"].values()) and d50 <= 1.1 * res["full"]["rms"])

    cases = [c for b in res["budgets"].values() for c in b["campaigns"].values()] + [res["full"]]
    bad = np.array([c["rms"] > 1.0 for c in cases])
    flagged = np.array([c["flag_channel"] or c["flag_flux"] for c in cases])
    v8 = {"targets": len(cases), "untrustworthy": int(bad.sum()), "flagged": int(flagged.sum()),
          "accuracy": float((bad == flagged).mean()),
          "accuracy_untrustworthy": float((flagged[bad]).mean()) if bad.any() else None,
          "accuracy_trustworthy": float((~flagged[~bad]).mean()) if (~bad).any() else None}
    v8["testable"] = bool(bad.sum() >= 5 and (~bad).sum() >= 5)
    v8["H7_campaigns"] = bool(v8["testable"] and v8["accuracy"] >= 0.9 and v8["accuracy_untrustworthy"] >= 0.8
                              and v8["accuracy_trustworthy"] >= 0.8) if v8["testable"] else None
    res["V8_campaigns"] = v8
    out = ROOT / "results" / "v7" / "v7.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"  H6 designed <= 0.8 x median random at every budget, and at 50 % <= 1.1 x full: {'PASS' if res['H6'] else 'FAIL'}")
    print(f"V8 on {v8['targets']} campaign targets: {v8['untrustworthy']} untrustworthy, {v8['flagged']} flagged, accuracy {v8['accuracy']:.3f} "
          f"(untrustworthy {v8['accuracy_untrustworthy']}, trustworthy {v8['accuracy_trustworthy']}); "
          f"H7 on campaigns: {'untested (a class has < 5 targets)' if not v8['testable'] else 'PASS' if v8['H7_campaigns'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
