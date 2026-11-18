# file: experiments/v2_extrapolation.py
"""V2 (design section 6): downward extrapolation on the Stanford butane runs. Checks H1, H2 and the
full-network part of H3 (thresholds fixed 2026-11-04, before any real-data fit).

  python -I experiments/v2_extrapolation.py --tables data/processed/stanford
Models, all from the same training runs (default: fit 4500-5000 K, select on 4000 K, test 3100-3500 K):
  - compact (main method): anchors; rates counted at the fit temperatures; backward elimination judged on
    the selection temperature; final rates counted at all training temperatures;
  - concentration fit (lam = 0, as Reax-Lump fits): the same anchors, L-BFGS on concentrations with refits;
  - full network (baseline 5) and flux-ranked skeletons of it (baseline 6), rates from all training runs.
Checks:
  H1: training error <= 1.5 x seed-to-seed scatter of the training runs; firing score on the training runs
      <= 1.5; carbon balance of every kept reaction exact;
  H2: on the test runs, firing score <= 0.9 x the concentration fit's, and error <= 1.10 x its error;
  H3, full-network part: test error <= 1.5 x the full network's, with >= 10x fewer reactions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fit  # noqa: E402
from data import load_stanford  # noqa: E402
from fullnet import Network, load_run  # noqa: E402
from metrics import normalized_rmse  # noqa: E402

SKELETON_SIZES, SKELETON_FRACS = (300, 1000, 3000), (0.5,)


def temps(s):
    return [float(x) for x in s.split(",")]


def firing(model, groups):
    """Mean squared difference of 2 sqrt(count + 1) between predicted and counted firings, per kept reaction
    and 1.2 ps bin, pooled over runs."""
    act, total, n = model.active.bool(), 0.0, 0
    with torch.no_grad():
        for g in groups:
            xi = fit.simulate(model, g)[1]
            d = (2 * (fit.per_bin_counts(xi, 2)[..., act].clamp_min(0) + 1).sqrt()
                 - 2 * (g.ev[..., model.types][..., act] + 1).sqrt()) ** 2
            total, n = total + d.sum().item(), n + d.numel()
    return total / n


def stacked_error(predict, per_run):
    """Design metric pooled over all (run, class) pairs of the runs: predict(rid, g) -> (n_t, n_c). Runs may
    differ in length (some dumps end early: 4917 or 4988 frames instead of 5000)."""
    errs, n = [], []
    for rid, g in per_run:
        obs = g.obs.numpy()
        errs.append(normalized_rmse(predict(rid, g)[:, None], obs))
        n.append(int((np.abs(obs).max(0) > 0).sum()))
    return float(np.average(errs, weights=n))


def seed_scatter(tables, runs, classes):
    """Mean over training temperatures of the normalized RMSE between the two seeds' class trajectories."""
    out = []
    for T, ids in runs.groupby("T_K").run_id:
        if len(ids) == 2:
            a, b = (pd.read_csv(Path(tables) / f"{r}.csv")[classes].to_numpy() for r in ids)
            n = min(len(a), len(b))
            out.append(normalized_rmse(a[:n, None], b[:n, None]))
    return float(np.mean(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "data" / "processed" / "stanford"))
    ap.add_argument("--fit-T", default="4500,5000")
    ap.add_argument("--select-T", default="4000")
    ap.add_argument("--test-T", default="3100,3200,3300,3400,3500")
    ap.add_argument("--out", default=str(ROOT / "results" / "v2" / "v2.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f} s] {s}", flush=True)
    runs = pd.read_csv(Path(a.tables) / "runs.csv")
    butane = runs[runs.molecule == "C4H10"]
    pick = lambda Ts: butane[butane.T_K.isin(Ts)]
    fit_ids, sel_ids = pick(temps(a.fit_T)).run_id.tolist(), pick(temps(a.select_T)).run_id.tolist()
    train_runs = pick(temps(a.fit_T) + temps(a.select_T))
    test_ids = pick(temps(a.test_T)).run_id.tolist()
    if set(test_ids) & set(train_runs.run_id):
        log("WARNING: test runs overlap the training runs (development check only)")
    train_g, S, O, classes = load_stanford(a.tables, train_runs.run_id.tolist())
    fit_g, sel_g = load_stanford(a.tables, fit_ids)[0], load_stanford(a.tables, sel_ids)[0]
    per_run = [(rid, load_stanford(a.tables, [rid])[0][0]) for rid in test_ids]
    meta = json.loads((Path(a.tables) / "classes.json").read_text())
    carbon, units = np.array(meta["composition"])[:, 0], dict(zip(meta["classes"], meta["units"]))
    res = {"split": {"fit_T": a.fit_T, "select_T": a.select_T, "test_T": a.test_T}}

    models = {}
    for name, lam in (("compact", None), ("concentration_fit", 0.0)):
        m = fit.build(train_g, S, O, carbon)
        fit.set_counted_rates(m, fit_g, O)
        if lam is not None:
            fit.lbfgs(m, fit_g, lam, rounds=5)
        m, path = fit.reduce(m, fit_g, sel_g, 0.0, rounds=0 if lam is None else 3, log=log)
        fit.set_counted_rates(m, train_g, O)
        if lam is not None:
            fit.lbfgs(m, train_g, lam, rounds=5)
        with torch.no_grad():
            err = stacked_error(lambda rid, g: fit.simulate(m, g)[0][:, 0].numpy(), per_run)
        models[name] = m
        res[name] = {"reactions": m.n_active(), "test_error": err, "test_firing": firing(m, [g for _, g in per_run]),
                     "train_error": fit.error(m, train_g), "train_firing": firing(m, train_g), "path": path,
                     "carbon_imbalance": float((m.nu().detach().numpy().T @ carbon).__abs__().max())}
        log(f"{name}: {m.n_active()} reactions, test error {err:.4f}, test firing {res[name]['test_firing']:.4f}")

    train = [(float(r.T_K), *load_run(Path(a.tables) / "species", r.run_id)) for r in train_runs.itertuples()]
    test = {rid: load_run(Path(a.tables) / "species", rid) for rid in test_ids}
    net = Network.from_runs(train, extra_species=set().union(*(set(w.columns) for w, _, _ in test.values())))
    n = net.fit_rates(train)
    T_of = dict(zip(runs.run_id, runs.T_K))
    predict_net = lambda sub: (lambda rid, g: sub.predict_classes(test[rid][0], test[rid][1], float(T_of[rid]), g.t.numpy(), units))
    res["full_network"] = {"reactions": len(net.reactions), "seen_once": float((n == 1).mean()),
                           "test_error": stacked_error(predict_net(net), per_run)}
    log(f"full network: {len(net.reactions)} reactions, test error {res['full_network']['test_error']:.4f}")
    order = np.argsort(-n, kind="stable")
    res["skeletons"] = []
    for size in (res["compact"]["reactions"],) + SKELETON_SIZES:
        sub = net.subnet(np.sort(order[:size]))
        res["skeletons"].append({"rule": f"top {size} by events", "reactions": size, "test_error": stacked_error(predict_net(sub), per_run)})
    for frac in SKELETON_FRACS:
        keep = net.window_skeleton(train, frac)
        res["skeletons"].append({"rule": f"window {frac}", "reactions": int(len(keep)),
                                 "test_error": stacked_error(predict_net(net.subnet(keep)), per_run)})
    for s in res["skeletons"]:
        log(f"skeleton {s['rule']}: {s['reactions']} reactions, test error {s['test_error']:.4f}")

    c, f, full = res["compact"], res["concentration_fit"], res["full_network"]
    scatter = seed_scatter(a.tables, train_runs, classes)
    res["checks"] = {
        "H1 training error <= 1.5 x seed scatter": bool(c["train_error"] <= 1.5 * scatter),
        "H1 training firing score <= 1.5": bool(c["train_firing"] <= 1.5),
        "H1 carbon balance exact": bool(c["carbon_imbalance"] < 1e-6),
        "H2a test firing <= 0.9 x concentration fit's": bool(c["test_firing"] <= 0.9 * f["test_firing"]),
        "H2b test error <= 1.10 x concentration fit's": bool(c["test_error"] <= 1.10 * f["test_error"]),
        "H3 (full-network part) error <= 1.5 x full network's, >= 10x fewer reactions": bool(
            c["test_error"] <= 1.5 * full["test_error"] and full["reactions"] >= 10 * c["reactions"])}
    res["seed_scatter"], res["minutes"] = scatter, round((time.time() - t0) / 60, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"seed-to-seed scatter of the training runs {scatter:.4f}; training error {c['train_error']:.4f}; "
          f"training firing {c['train_firing']:.4f}")
    for k, v in res["checks"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
