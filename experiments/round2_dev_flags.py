# file: experiments/round2_dev_flags.py
"""Round 2 development (design section 12), C6: flag variants on DEVELOPMENT cases only: the V7 campaigns
(target 3100 K), methane at 3300 K and the 5 companion targets (all labels already known).

  python -I experiments/round2_dev_flags.py
Variants (each also flags a target whose model puts > half its flux on reactions seen at <= 1 temperature, the
round-1 flag (b), which caught the full network):
  r1        round 1: no identified consumption channel (sd <= 1)
  err(c)    predicted error: flux-weighted RMS of the calibrated sd of ln k at the target > c
  dom(f)    domain: a class present at t = 0 starts above f x its highest concentration in the training runs
  err(c)|dom(f)
Score: balanced accuracy over the development cases.
"""
import itertools
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

TABLES, COMP = ROOT / "data" / "processed" / "stanford", ROOT / "data" / "processed" / "companion"
CS, FS = (0.5, 0.75, 1.0, 1.5), (1.0, 1.5, 2.0)


def single_frac(lines, flux):
    f = np.asarray(flux, float)
    return f[lines["seen"] <= 1].sum() / f.sum() if np.isfinite(f).all() and f.sum() > 0 else np.inf


def h4_targets():
    """Methane 3300 K and the companion targets, with the H4 model (as experiments/v8_companion.py)."""
    torch.set_num_threads(1)
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
    comp = pd.read_csv(COMP / "runs.csv")
    tg = {f"{mol} {T:g} K": (COMP, sorted(df.run_id)) for (mol, T), df in comp.groupby(["molecule", "T_K"])}
    tg["CH4 3300 K (Part I)"] = (TABLES, sorted(runs[runs.molecule == "CH4"].run_id))
    labels = json.loads((ROOT / "results" / "v8" / "companion.json").read_text())["targets"]
    out = []
    for key, (folder, rids) in tg.items():
        (g0,), *_ = load_stanford(folder, rids[:1])
        T = float(g0.T[0])
        sd_psi = design.calibrated(camp, data, T_of, A, T)[1]
        mean, sd, s2, tau2, lines = design.calibrated_re(camp, data, T_of, A, T)
        with torch.no_grad():
            flux = fit.simulate(m, g0)[1][-1, 0].numpy()[kept]
        c0 = g0.obs[0, 0].numpy()
        present = c0 > 0
        bad = labels[key]["untrustworthy"] if key in labels else True  # Part I methane: error 0.559 > 2 x 0.152
        r1 = design.flags(lines, T, S[:, A], O[:, A], present, flux)
        out.append({"case": key, "bad": bool(bad), "r1": bool(r1[0] or r1[1]), "single": single_frac(lines, flux) > 0.5,
                    "err": float(np.sqrt((flux * sd ** 2).sum() / flux.sum())) if np.isfinite(sd).all() else np.inf,
                    "err_psi": float(np.sqrt((flux * sd_psi ** 2).sum() / flux.sum())) if np.isfinite(sd_psi).all() else np.inf,
                    "dom": float(np.max(c0[present] / np.maximum(cmax[present], 1e-12)))})
        print(f"{key}: bad {bad}, predicted RMS sd {out[-1]['err']:.2f}, start / training max {out[-1]['dom']:.2f}, round-1 flag {out[-1]['r1']}", flush=True)
    return out


def campaign_targets():
    """The V7 campaigns (same menu, pilot and random seed as experiments/v7_design.py)."""
    torch.set_num_threads(1)
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g = {}
    for rid in butane.run_id:
        (g[rid],), S, O, _ = load_stanford(TABLES, [rid])
    carbon = np.array(json.loads((TABLES / "classes.json").read_text())["composition"])[:, 0]
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    nb = {r: g[r].ev.shape[1] for r in g}
    target = [r for r in g if T_of[r] == 3100.0]
    menu = {r: [L for L in (50, 100, 200) if L < nb[r]] + [nb[r]] for r in g if T_of[r] != 3100.0}
    data = {r: design.truncations(g[r], O, menu.get(r, [nb[r]])) for r in g}
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in target}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))
    g0 = g[target[0]]
    total, pilot, rng = sum(nb[r] for r in menu), {"C4H10_896at_1_5000K_40GPa_1Ms": 50}, np.random.default_rng(0)
    camps = [{r: nb[r] for r in menu}]
    for f in (0.10, 0.25, 0.50):
        B = int(f * total)
        camps += [design.greedy(menu, pilot, B, T_of, 3100.0), design.hot_only(menu, pilot, B, T_of), design.uniform(menu, pilot, B, T_of)]
        camps += [design.random_campaign(menu, pilot, B, rng) for _ in range(100)]
    out = []
    for c in camps:
        mean, sd, s2, tau2, lines = design.calibrated_re(c, data, T_of, A, 3100.0)
        model = LumpedKinetics(S[:, A], O[:, A], lines["a"], 10.0 * lines["b"] * R_KJ, carbon)
        with torch.no_grad():
            flux = fit.simulate(model, g0)[1][-1, 0].numpy()
        r1 = design.flags(lines, 3100.0, S[:, A], O[:, A], g0.obs[0, 0].numpy() > 0, flux)
        rms = float(np.sqrt((w * np.minimum(np.abs(mean - obs), 10.0) ** 2).sum() / w.sum()))
        ok = np.isfinite(sd) & np.isfinite(flux)
        out.append({"case": "campaign", "bad": rms > np.log(2.0), "r1": bool(r1[0] or r1[1]), "single": single_frac(lines, flux) > 0.5,
                    "err": float(np.sqrt((flux[ok] * sd[ok] ** 2).sum() / flux[ok].sum())) if ok.any() else np.inf, "dom": 1.0, "rms": rms})
    print(f"{len(out)} campaigns: predicted RMS sd median {np.median([o['err'] for o in out]):.2f}, max {max(o['err'] for o in out):.2f}; "
          f"realized RMS error median {np.median([o['rms'] for o in out]):.2f}, max {max(o['rms'] for o in out):.2f}", flush=True)
    return out


def main():
    cases = h4_targets() + campaign_targets()
    bad = np.array([c["bad"] for c in cases])
    rules = {"r1": lambda c: c["r1"]}
    rules.update({f"err{cc:g}": (lambda cc: lambda c: c["single"] or c["err"] > cc)(cc) for cc in CS})
    rules.update({f"dom{ff:g}": (lambda ff: lambda c: c["single"] or c["dom"] > ff)(ff) for ff in FS})
    rules.update({f"err{cc:g}|dom{ff:g}": (lambda cc, ff: lambda c: c["single"] or c["err"] > cc or c["dom"] > ff)(cc, ff)
                  for cc, ff in itertools.product(CS, FS)})
    res = {}
    for k, rule in rules.items():
        fl = np.array([rule(c) for c in cases])
        sens, spec = float(fl[bad].mean()), float((~fl[~bad]).mean())
        res[k] = {"sensitivity": sens, "specificity": spec, "balanced": (sens + spec) / 2, "accuracy": float((fl == bad).mean())}
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["balanced"]):
        print(f"{k:>14}: sensitivity {v['sensitivity']:.2f}, specificity {v['specificity']:.3f}, balanced {v['balanced']:.3f}")
    out = ROOT / "results" / "round2" / "dev_flags.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": [{k: v for k, v in c.items()} for c in cases if c["case"] != "campaign"], "rules": res}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
