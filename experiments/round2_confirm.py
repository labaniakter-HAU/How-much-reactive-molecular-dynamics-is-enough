# file: experiments/round2_confirm.py
"""Round 2 confirmation (design section 12): run ONCE, after results/round2/FROZEN.json fixes the method and
thresholds. Real part: the 8 butane design targets other than 3100 K (C4, C5, C6) and C8H18 at 2900 K (C6).
The synthetic part is experiments/round2_synthetic.py on seeds 10000-11999.

  python -I experiments/round2_confirm.py
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
from kinetics import R_KJ, LumpedKinetics  # noqa: E402

TABLES, OCT, OUT = ROOT / "data" / "processed" / "stanford", ROOT / "data" / "processed" / "octane2900", ROOT / "results" / "round2"
BUDGETS, CLIP = (0.10, 0.25, 0.50), 10.0


def uncertainty(frozen, camp, data, T_of, A, T):
    """-> mean, predictive sd, observation-variance function of (events, runs), lines."""
    if frozen["uncertainty"] == "re":
        mean, sd, s2, tau2, lines = design.calibrated_re(camp, data, T_of, A, T)
        return mean, sd, (lambda n, r: 1.0 / n + s2 / r), lines
    mean, sd, psi, tau2, lines = design.calibrated(camp, data, T_of, A, T)
    return mean, sd, (lambda n, r: psi / n), lines


def flag(frozen, sd, flux, lines, dom):
    """The frozen foresight rule: predicted error, domain and single-temperature flux parts."""
    f = np.asarray(flux, float)
    ok = np.isfinite(sd) & np.isfinite(f)
    if not ok.all() or f.sum() <= 0:
        return True
    err = float(np.sqrt((f * sd ** 2).sum() / f.sum()))
    single = f[lines["seen"] <= 1].sum() / f.sum()
    return bool(err > frozen["flag_err"] or single > 0.5 or (frozen["flag_dom"] is not None and dom > frozen["flag_dom"]))


def design_target(frozen, T_star, g, S, O, carbon, T_of):
    nb = {r: g[r].ev.shape[1] for r in g}
    target = [r for r in g if T_of[r] == T_star]
    menu = {r: [L for L in (50, 100, 200) if L < nb[r]] + [nb[r]] for r in g if T_of[r] != T_star}
    data = {r: design.truncations(g[r], O, menu.get(r, [nb[r]])) for r in g}
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in target}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))
    g0 = g[target[0]]
    pilot_T = 4500.0 if T_star == 5000.0 else 5000.0
    pilot = {[r for r in menu if T_of[r] == pilot_T and "_1_" in r][0]: 50}
    total, rng = sum(nb[r] for r in menu), np.random.default_rng(int(T_star))
    res, z = {"budgets": {}}, []

    def evaluate(camp, zs):
        mean, sd, obs_var, lines = uncertainty(frozen, camp, data, T_of, A, T_star)
        model = LumpedKinetics(S[:, A], O[:, A], lines["a"], 10.0 * lines["b"] * R_KJ, carbon)
        with torch.no_grad():
            flux = fit.simulate(model, g0)[1][-1, 0].numpy()
        flagged = flag(frozen, sd, flux, lines, 1.0)  # flag computed before the error
        rms = float(np.sqrt((w * np.minimum(np.abs(mean - obs), CLIP) ** 2).sum() / w.sum()))  # C6 label: the frozen prediction
        Ts, n, E = design.pool(camp, data, T_of)
        p = design.predict(design.fit_lines(n[:, A], E[:, A], Ts), T_star)[0]
        rms_c5 = float(np.sqrt((w * np.minimum(np.abs(p - obs), CLIP) ** 2).sum() / w.sum()))  # C5 metric: as round 1
        if zs is not None:
            keep = w >= 10
            zs.extend(((mean - obs) / np.sqrt(sd ** 2 + obs_var(np.maximum(w, 1.0), len(target))))[keep].tolist())
        return {"rms": rms_c5, "rms_frozen": rms, "flagged": flagged, "bad": bool(rms > frozen["label_rms"])}

    res["full"] = evaluate({r: nb[r] for r in menu}, None)  # the full campaign equals a V6 fold: no z values
    for f in BUDGETS:
        B = int(f * total)
        camps = {"designed": design.greedy(menu, pilot, B, T_of, T_star), "hot_only": design.hot_only(menu, pilot, B, T_of),
                 "uniform": design.uniform(menu, pilot, B, T_of)}
        camps.update({f"random_{i}": design.random_campaign(menu, pilot, B, rng) for i in range(100)})
        ev = {k: evaluate(c, z) for k, c in camps.items()}
        res["budgets"][str(f)] = {"campaigns": ev, "median_random": float(np.median([ev[f"random_{i}"]["rms"] for i in range(100)]))}
    res["z"] = [round(x, 4) for x in z]
    return res


def octane_target(frozen):
    """C8H18 at 2900 K (sr637qt9337, read only now): the H4 model (as experiments/v8_companion.py), the frozen flag
    from its training events, the target's t = 0 state and T, written before the error is computed; label as the
    companion targets (error > 2 x single-run noise)."""
    from v2_extrapolation import seed_scatter, stacked_error
    runs = pd.read_csv(TABLES / "runs.csv")
    butane = runs[runs.molecule == "C4H10"]
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    ids = lambda Ts: butane[butane.T_K.isin(Ts)].run_id.tolist()
    fit_ids, sel_ids = ids([4500.0, 5000.0]), ids([4000.0])
    train_g, S, O, classes = load_stanford(TABLES, fit_ids + sel_ids)
    fit_g, sel_g = load_stanford(TABLES, fit_ids)[0], load_stanford(TABLES, sel_ids)[0]
    carbon = np.array(json.loads((TABLES / "classes.json").read_text())["composition"])[:, 0]
    m = fit.build(train_g, S, O, carbon)
    fit.set_counted_rates(m, fit_g, O)
    m, _ = fit.reduce(m, fit_g, sel_g, 0.0, rounds=0, log=lambda s: None)
    fit.set_counted_rates(m, train_g, O)
    kept = m.active.numpy() > 0
    A = m.types.numpy()[kept]
    data, cmax = {}, np.zeros(len(classes))
    for rid in fit_ids + sel_ids:
        (g,), *_ = load_stanford(TABLES, [rid])
        data[rid] = design.truncations(g, O, [g.ev.shape[1]])
        cmax = np.maximum(cmax, g.obs[:, 0].numpy().max(0))
    camp = {r: max(d) for r, d in data.items()}
    oc = pd.read_csv(OCT / "runs.csv")
    rids = sorted(oc.run_id)
    (g0,), *_ = load_stanford(OCT, rids[:1])
    T = float(g0.T[0])
    mean, sd, obs_var, lines = uncertainty(frozen, camp, data, T_of, A, T)
    with torch.no_grad():
        flux = fit.simulate(m, g0)[1][-1, 0].numpy()[kept]
    c0 = g0.obs[0, 0].numpy()
    present = c0 > 0
    dom = float(np.max(c0[present] / np.maximum(cmax[present], 1e-12)))
    flagged = flag(frozen, sd, flux, lines, dom)
    (OUT / "confirm_octane_flag.json").write_text(json.dumps({"flagged": flagged, "dom": dom}, indent=1))
    per_run = [(rid, load_stanford(OCT, [rid])[0][0]) for rid in rids]
    with torch.no_grad():
        err = stacked_error(lambda rid, g: fit.simulate(m, g)[0][:, 0].numpy(), per_run)
    noise = seed_scatter(OCT, oc, classes) / np.sqrt(2)
    res = {"flagged": flagged, "dom": dom, "error": err, "single_run_noise": noise, "bad": bool(err > 2 * noise)}
    print(f"C8H18 2900 K: {'FLAGGED' if flagged else 'not flagged'} (start / training max {dom:.2f}); error {err:.3f} vs 2 x noise {2 * noise:.3f} -> "
          f"{'untrustworthy' if res['bad'] else 'trustworthy'}", flush=True)
    return res


def main():
    frozen_path = OUT / "FROZEN.json"
    if not frozen_path.exists():
        raise SystemExit("freeze the method and thresholds first: results/round2/FROZEN.json")
    frozen = json.loads(frozen_path.read_text())
    torch.set_num_threads(1)
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g = {}
    for rid in butane.run_id:
        (g[rid],), S, O, classes = load_stanford(TABLES, [rid])
    carbon = np.array(json.loads((TABLES / "classes.json").read_text())["composition"])[:, 0]
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    out = {"frozen": frozen, "targets": {}}
    for T in frozen["design_targets"]:
        out["targets"][f"{T:g}"] = design_target(frozen, float(T), g, S, O, carbon, T_of)
        r = out["targets"][f"{T:g}"]
        print(f"target {T:g} K: " + "; ".join(f"{float(f):.0%} designed {b['campaigns']['designed']['rms']:.3f} / hot {b['campaigns']['hot_only']['rms']:.3f} "
              f"/ uniform {b['campaigns']['uniform']['rms']:.3f} / random {b['median_random']:.3f}" for f, b in r["budgets"].items())
              + f"; full {r['full']['rms']:.3f}", flush=True)
        (OUT / "confirm_real.json").write_text(json.dumps(out, indent=1))
    out["octane"] = octane_target(frozen)
    (OUT / "confirm_real.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
