# file: experiments/v8_companion.py
"""V8 / H7 companion targets (design section 11, pre-registered 2026-11-05 before any companion file was read)
and the descriptive H4 extension. Run experiments/companion_tables.py and V7 first.

  python -I experiments/v8_companion.py
1. Part I's compact model exactly as for H4 (v3_transfer.py): butane, fit 4500-5000 K, select 4000 K, final
   rates counted at 4000-5000 K.
2. For each target (composition, temperature): the foresight flag from the model's training events
   (design.fit_lines on the 4000-5000 K counts of its kept reactions), the target's t = 0 state and T. All flags
   are written to results/v8/companion_flags.json before any target trajectory is compared with a prediction.
3. Label: untrustworthy if the model's error (Part I metric, pooled over both seeds) > 2 x the target's
   single-run noise (seed-to-seed scatter / sqrt 2).
4. H7 primary on the V7 campaign targets plus these: >= 90 % correct overall and >= 80 % within each class,
   each class >= 5 targets, else untested.
"""
import argparse
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
from v2_extrapolation import seed_scatter, stacked_error  # noqa: E402

TABLES, COMP, OUT = ROOT / "data" / "processed" / "stanford", ROOT / "data" / "processed" / "companion", ROOT / "results" / "v8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--companion", default=str(COMP), help="table folder of the targets")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    comp_dir, out_dir = Path(a.companion), Path(a.out)
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
    data = {}
    for rid in fit_ids + sel_ids:
        (g,), *_ = load_stanford(TABLES, [rid])
        data[rid] = design.truncations(g, O, [g.ev.shape[1]])
    Ts, n, E = design.pool({r: max(d) for r, d in data.items()}, data, T_of)
    lines = design.fit_lines(n[:, A], E[:, A], Ts)

    comp = pd.read_csv(comp_dir / "runs.csv")
    targets = {f"{mol} {T:g} K": sorted(df.run_id) for (mol, T), df in comp.groupby(["molecule", "T_K"])}
    flags = {}
    for key, rids in targets.items():
        (g0,), *_ = load_stanford(comp_dir, rids[:1])
        with torch.no_grad():
            flux = fit.simulate(m, g0)[1][-1, 0].numpy()[kept]  # from the t = 0 state only
        fa, fb = design.flags(lines, float(g0.T[0]), S[:, A], O[:, A], g0.obs[0, 0].numpy() > 0, flux)
        flags[key] = {"flag_channel": fa, "flag_flux": fb, "flagged": fa or fb,
                      "start": comp.loc[comp.run_id == rids[0], "start"].item() if "start" in comp else ""}
        print(f"{key}: flag (a) {fa}, flag (b) {fb}  [t = 0: {flags[key]['start']}]", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "companion_flags.json").write_text(json.dumps({"reactions": int(kept.sum()), "flags": flags}, indent=1))

    in_domain = json.loads((ROOT / "results" / "v2" / "v2.json").read_text())["compact"]["test_error"]
    res = {"reactions": int(kept.sum()), "in_domain_error": in_domain, "targets": {}}
    for key, rids in targets.items():
        per_run = [(rid, load_stanford(comp_dir, [rid])[0][0]) for rid in rids]
        with torch.no_grad():
            err = stacked_error(lambda rid, g: fit.simulate(m, g)[0][:, 0].numpy(), per_run)
        noise = seed_scatter(comp_dir, comp[comp.run_id.isin(rids)], classes) / np.sqrt(2)
        res["targets"][key] = {**flags[key], "error": err, "single_run_noise": noise,
                               "untrustworthy": bool(err > 2 * noise), "H4_rule_pass": bool(err <= 2 * in_domain)}
        r = res["targets"][key]
        print(f"{key}: error {err:.3f}, 2 x noise {2 * noise:.3f} -> {'untrustworthy' if r['untrustworthy'] else 'trustworthy'}; "
              f"{'FLAGGED' if r['flagged'] else 'not flagged'}; H4 rule (<= {2 * in_domain:.3f}) {'PASS' if r['H4_rule_pass'] else 'FAIL'}")

    v7 = json.loads((ROOT / "results" / "v7" / "v7.json").read_text())
    cases = [(c["rms"] > 1.0, c["flag_channel"] or c["flag_flux"]) for b in v7["budgets"].values() for c in b["campaigns"].values()]
    cases += [(v7["full"]["rms"] > 1.0, v7["full"]["flag_channel"] or v7["full"]["flag_flux"])]
    cases += [(r["untrustworthy"], r["flagged"]) for r in res["targets"].values()]
    bad, fl = np.array([c[0] for c in cases]), np.array([c[1] for c in cases])
    h7 = {"targets": len(cases), "untrustworthy": int(bad.sum()), "flagged": int(fl.sum()), "accuracy": float((bad == fl).mean()),
          "accuracy_untrustworthy": float(fl[bad].mean()) if bad.any() else None,
          "accuracy_trustworthy": float((~fl[~bad]).mean()) if (~bad).any() else None,
          "testable": bool(bad.sum() >= 5 and (~bad).sum() >= 5)}
    h7["H7"] = bool(h7["accuracy"] >= 0.9 and h7["accuracy_untrustworthy"] >= 0.8 and h7["accuracy_trustworthy"] >= 0.8) if h7["testable"] else None
    res["H7_primary"] = h7
    (out_dir / "companion.json").write_text(json.dumps(res, indent=1))
    print(f"H7 primary on {h7['targets']} targets: {h7['untrustworthy']} untrustworthy, {h7['flagged']} flagged, accuracy {h7['accuracy']:.3f} "
          f"(untrustworthy {h7['accuracy_untrustworthy']}, trustworthy {h7['accuracy_trustworthy']}): "
          f"{'untested (a class has < 5 targets)' if not h7['testable'] else 'PASS' if h7['H7'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
