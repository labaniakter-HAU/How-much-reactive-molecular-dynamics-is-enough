# file: experiments/v8_known.py
"""V8 / H7 secondary (design section 11, C6), reported only: the foresight flag on the 17 cases whose outcome
was known when the flag rule was written (disclosed as after the fact): the 9 V1 folds, V2, the methane runs,
the full network and its 5 skeletons. Untrustworthy = realized error > 2 x the case's single-run noise
(seed-to-seed scatter of its test runs / sqrt(2); the one-run 3600 K fold takes the mean of the other folds').
Compact models are rebuilt exactly as in V1, V2 and V3; each model's own predicted flux, from each target's
t = 0 state; a case with several target temperatures is flagged if any of them is.

  python -I experiments/v8_known.py --part compact
  python -I experiments/v8_known.py --part network
  python -I experiments/v8_known.py --part score
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import design  # noqa: E402
import fit  # noqa: E402
from data import load_stanford  # noqa: E402
from fullnet import Network, load_run  # noqa: E402

TABLES, OUT = ROOT / "data" / "processed" / "stanford", ROOT / "results" / "v8"
RUNS = pd.read_csv(TABLES / "runs.csv")
BUTANE = RUNS[RUNS.molecule == "C4H10"]
T_OF = dict(zip(RUNS.run_id, RUNS.T_K.astype(float)))
ids = lambda Ts: BUTANE[BUTANE.T_K.isin(Ts)].run_id.tolist()
seed1 = lambda df: df[df.seed == 1].run_id.tolist()
V2_TRAIN, V2_TEST = [4000.0, 4500.0, 5000.0], [3100.0, 3200.0, 3300.0, 3400.0, 3500.0]


def compact_case(args):
    name, train_T, targets = args
    torch.set_num_threads(1)
    carbon = np.array(json.loads((TABLES / "classes.json").read_text())["composition"])[:, 0]
    train_g, S, O, _ = load_stanford(TABLES, ids(train_T))
    fit_g, sel_g = (load_stanford(TABLES, ids(Ts))[0] for Ts in (train_T[1:], train_T[:1]))  # lowest selects
    m = fit.build(train_g, S, O, carbon)
    fit.set_counted_rates(m, fit_g, O)
    m, _ = fit.reduce(m, fit_g, sel_g, 0.0, rounds=0, log=lambda s: None)
    fit.set_counted_rates(m, train_g, O)
    kept = m.active.numpy() > 0
    A = m.types.numpy()[kept]
    data = {}
    for rid in ids(train_T):
        (g,), *_ = load_stanford(TABLES, [rid])
        data[rid] = design.truncations(g, O, [g.ev.shape[1]])
    Ts, n, E = design.pool({r: max(d) for r, d in data.items()}, data, T_OF)
    lines = design.fit_lines(n[:, A], E[:, A], Ts)
    flags = []
    for rid in targets:
        (gt,), *_ = load_stanford(TABLES, [rid])
        with torch.no_grad():
            flux = fit.simulate(m, gt)[1][-1, 0].numpy()[kept]
        flags.append(design.flags(lines, T_OF[rid], S[:, A], O[:, A], gt.obs[0, 0].numpy() > 0, flux))
    print(f"{name}: {int(kept.sum())} reactions, flags {flags}", flush=True)
    return name, {"reactions": int(kept.sum()), "flags": flags, "flagged": any(a or b for a, b in flags)}


def flux_worker(args):
    net, keeps, T, c0, V, t = args
    out = {}
    for name, keep in keeps.items():
        sub = net if name == "full" else net.subnet(keep)
        sol = sub.integrate(c0, T, t)
        if not sol.success:
            out[name] = np.full(len(keep), np.nan)
            continue
        prev, total = sub.rhs(sol.y[:, 0], T)[1], 0.0
        for i in range(1, sol.y.shape[1]):
            cur = sub.rhs(sol.y[:, i], T)[1]
            total, prev = total + 0.5 * (prev + cur) * (sol.t[i] - sol.t[i - 1]), cur
        out[name] = total * V
    print(f"flux at {T:g} K done", flush=True)
    return T, out


def network_part():
    train = [(T_OF[r], *load_run(TABLES / "species", r)) for r in ids(V2_TRAIN)]
    test = {rid: load_run(TABLES / "species", rid) for rid in ids(V2_TEST)}
    net = Network.from_runs(train, extra_species=set().union(*(set(w.columns) for w, _, _ in test.values())))
    expo, replay = {}, net.replay_exposure
    net.replay_exposure = lambda w, V, ev: expo.setdefault(id(ev), replay(w, V, ev))
    n_tot = net.fit_rates(train)
    del net.replay_exposure
    Ts = sorted(V2_TRAIN)
    n = np.array([sum(net.counts(ev) for T2, _, _, ev in train if T2 == T) for T in Ts])
    E = np.array([sum(expo[id(ev)][0] for T2, _, _, ev in train if T2 == T) for T in Ts])
    lines = design.fit_lines(n, E, Ts)
    order = np.argsort(-n_tot, kind="stable")
    keeps = {"full": np.arange(len(net.reactions)), **{f"top {s} by events": np.sort(order[:s]) for s in (44, 300, 1000, 3000)},
             "window 0.5": net.window_skeleton(train, 0.5)}
    jobs = []
    for rid in seed1(BUTANE[BUTANE.T_K.isin(V2_TEST)]):
        w, V, _ = test[rid]
        c0 = np.zeros(len(net.species))
        c0[[net.index[f] for f in w.columns]] = w.iloc[0].to_numpy(float) / V
        (g,), *_ = load_stanford(TABLES, [rid])
        jobs.append((net, keeps, T_OF[rid], c0, V, g.t.numpy()))
    with Pool(len(jobs)) as p:
        fluxes = dict(p.map(flux_worker, jobs))
    res = {}
    for name, keep in keeps.items():
        sub = {k: v[keep] for k, v in lines.items()}
        Nk, Ok = net.N[:, keep], net.O[:, keep]
        flags = []
        for _, _, T, c0, _, _ in jobs:
            pres = np.flatnonzero(c0 > 0)
            S2 = np.vstack([Nk[pres].toarray(), np.zeros((1, len(keep)))])
            O2 = np.vstack([Ok[pres].toarray(), np.asarray(Ok.sum(0)) - Ok[pres].toarray().sum(0)])
            flags.append(design.flags(sub, T, S2, O2, np.r_[np.ones(len(pres), bool), False], fluxes[T][name]))
        res[name] = {"reactions": int(len(keep)), "flags": flags, "flagged": any(a or b for a, b in flags),
                     "single_temperature_reactions": float((sub["seen"] <= 1).mean())}
        print(f"{name}: {len(keep)} reactions, flags {flags}", flush=True)
    return res


def score():
    v1 = {T: json.loads((ROOT / "results" / "v1" / f"T{T:g}.json").read_text()) for T in sorted(set(BUTANE.T_K.astype(float)))}
    noise = {T: d["seed_scatter"] / np.sqrt(2) for T, d in v1.items() if d["seed_scatter"] is not None}
    noise = {T: noise.get(T, float(np.mean(list(noise.values())))) for T in v1}
    v2 = json.loads((ROOT / "results" / "v2" / "v2.json").read_text())
    v2_noise = json.loads((ROOT / "results" / "v2" / "h3b.json").read_text())["single_run_noise"]
    from v2_extrapolation import seed_scatter
    meta = json.loads((TABLES / "classes.json").read_text())
    ch4_noise = seed_scatter(TABLES, RUNS[RUNS.molecule == "CH4"], meta["classes"]) / np.sqrt(2)
    errors = {f"V1 {T:g} K": (d["test_error"], noise[T]) for T, d in v1.items()}
    errors["V2"] = (v2["compact"]["test_error"], v2_noise)
    errors["methane"] = (json.loads((ROOT / "results" / "v3" / "v3.json").read_text())["transfer_error"], ch4_noise)
    errors["full"] = (v2["full_network"]["test_error"], v2_noise)
    errors.update({s["rule"]: (s["test_error"], v2_noise) for s in v2["skeletons"]})
    flags = {**json.loads((OUT / "known_compact.json").read_text()), **json.loads((OUT / "known_network.json").read_text())}
    rows = [{"case": k, "error": e, "noise": nz, "untrustworthy": bool(e > 2 * nz), "flagged": flags[k]["flagged"]} for k, (e, nz) in errors.items()]
    bad, fl = np.array([r["untrustworthy"] for r in rows]), np.array([r["flagged"] for r in rows])
    res = {"cases": rows, "accuracy": float((bad == fl).mean()), "accuracy_untrustworthy": float(fl[bad].mean()) if bad.any() else None,
           "accuracy_trustworthy": float((~fl[~bad]).mean()) if (~bad).any() else None, "note": "after the fact; reported only"}
    (OUT / "known.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"  {r['case']:>20}: error {r['error']:.3f} vs 2 x noise {2 * r['noise']:.3f} -> {'untrustworthy' if r['untrustworthy'] else 'trustworthy'}; "
              f"{'FLAGGED' if r['flagged'] else 'not flagged'}")
    print(f"secondary (after the fact): {(bad == fl).sum()} of {len(rows)} correct; untrustworthy caught {fl[bad].sum()} of {bad.sum()}, "
          f"trustworthy passed {(~fl[~bad]).sum()} of {(~bad).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("compact", "network", "score"), required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    temps = sorted(set(BUTANE.T_K.astype(float)))
    if a.part == "compact":
        cases = [(f"V1 {T:g} K", [x for x in temps if x != T], seed1(BUTANE[BUTANE.T_K == T])) for T in temps]
        cases.append(("V2", V2_TRAIN, seed1(BUTANE[BUTANE.T_K.isin(V2_TEST)])))
        cases.append(("methane", V2_TRAIN, seed1(RUNS[RUNS.molecule == "CH4"])))
        with Pool(4) as p:  # about 0.5 GB per case
            (OUT / "known_compact.json").write_text(json.dumps(dict(p.map(compact_case, cases)), indent=1))
    elif a.part == "network":
        (OUT / "known_network.json").write_text(json.dumps(network_part(), indent=1))
    else:
        score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
