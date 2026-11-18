# file: experiments/round5_dev_c2.py
"""Round 5 development for C2 and C3 (design section 16): the complete counted model (src/closure.py) on the HFO
folds only. It reads no MEA file. Every variant run is appended to results/round5/dev_c2.jsonl.

  python -I -X utf8 experiments/round5_dev_c2.py --anchors 80 [--no-absorb] [--pooled] [--no-closure] [--no-volume] [--fit] [--full]
Per fold (held-out temperature): counted model -> concentration error on the 4 test runs (round 4's metric and
species) and firing score per run and reaction; with --fit the same model fitted to the training concentrations;
with --full the full network (every (reactants, products) reaction with model reactants, same closure).
"""
import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import closure  # noqa: E402
import design  # noqa: E402
import lab  # noqa: E402
import species  # noqa: E402

HFO, OUT = ROOT.parent / "HFO1", ROOT / "results" / "round5"


def pooled_error(terms):
    e, n = np.array(terms, float).T
    return float(np.inf if not np.isfinite(e).all() else np.average(e, weights=n))


def read(root):
    rs = lab.runs(root)
    return ({nm: lab.read_events(f) for nm, _, _, f in rs}, {nm: lab.read_counts(f) for nm, _, _, f in rs}, {nm: T for nm, T, _, _ in rs})


def evaluate(tab, cnt, test, make, A):
    """Models make(run) on the test runs -> pooled concentration error, firing terms, most negative count."""
    err, fire, neg = [], [], 0.0
    for r in test:
        g = tab.group({r: cnt[r]}, [r])
        c, xi = closure.predict(make(r), g)
        err += species.error_terms(c[..., tab.score], closure.scored(g, tab.score))
        fire += species.firing_terms(xi, tab.n[r][A][None]).tolist()
        neg = min(neg, float(np.nanmin(c)) if np.isfinite(c).any() else -np.inf)
    return {"error_terms": err, "firing_terms": fire, "error": pooled_error(err),
            "firing": float(np.mean(fire)) if np.isfinite(fire).all() else float("inf"), "most_negative_count": neg}


def fit_models(tab, cnt, train, lines, A, closed, rounds=5):
    """The concentration fit of the same model: one model per training temperature (its own net changes), one
    parameter set. -> the fitted model for any temperature, the final loss."""
    temps = tab.temps if tab.by_T else [tab.temps[0]]
    models = [tab.model(lines, A, T, closed) for T in temps]
    for m in models[1:]:
        m.lnA, m.Ea = models[0].lnA, models[0].Ea
    groups = [tab.group(cnt, [r for r in train if not tab.by_T or tab.T_of[r] == T]) for T in temps]
    final = closure.conc_fit(models, groups, tab.score, rounds)

    def at(T):
        m = tab.model(lines, A, T, closed)
        with torch.no_grad():
            m.lnA.copy_(models[0].lnA)
            m.Ea.copy_(models[0].Ea)
        return m
    return at, final, models[0]


def full_network(tab, ev, cnt, train, test, T_star, closed):
    """Every (reactants, products) reaction with model reactants and a rate, each with its own line; the closure of
    the same closed species, per training temperature and interpolated as the compact model's."""
    ix, xx = {s: i for i, s in enumerate(tab.M)}, {s: i for i, s in enumerate(tab.closed)}
    pooled = sum(tab.events.values(), Counter())
    keys = sorted(k for k in pooled if tab.explicit(k[0]))
    expo = {r: {} for r in train}
    S, O, Qp = np.zeros((len(tab.M), len(keys))), np.zeros((len(tab.M), len(keys))), np.zeros((len(tab.closed), len(keys)))
    Ts = np.array(tab.temps)
    n, E = np.zeros((len(Ts), len(keys))), np.zeros((len(Ts), len(keys)))
    for k, (lhs, rhs) in enumerate(keys):
        for s, nu in lhs:
            S[ix[s], k] -= nu
            O[ix[s], k] = nu
        for s, nu in rhs:
            if s in ix:
                S[ix[s], k] += nu
            elif s in xx:
                Qp[xx[s], k] += nu
        for r in train:
            t, c = cnt[r]
            if all(s in c for s, _ in lhs):
                if lhs not in expo[r]:
                    expo[r][lhs] = closure.exposure(t, c, lhs, tab.vol[r])
                i = int(np.searchsorted(Ts, tab.T_of[r]))
                n[i, k] += ev[r].get((lhs, rhs), 0)
                E[i, k] += expo[r][lhs]
    cols = np.flatnonzero((E > 0).any(0))
    lines = {k: v[cols] for k, v in design.fit_lines(n, E, Ts).items()}
    S_cl = 0.0
    if closed:
        w = tab.weights(T_star) if tab.by_T else None
        S_cl = (sum(x * closure.closure(tab.events[T], tab.M, tab.closed, Qp)[0] for T, x in w.items()) if w
                else closure.closure(pooled, tab.M, tab.closed, Qp)[0])
    err = []
    for r in test:
        g = tab.group({r: cnt[r]}, [r])
        c, _ = closure.predict(species.model(*tab.with_volume(S + S_cl, O), lines, cols), g, extents=False)
        err += species.error_terms(c[..., tab.score], closure.scored(g, tab.score))
    return {"reactions": int(len(cols)), "error_terms": err, "error": pooled_error(err)}


def fold(args):
    T_star, a, root = args
    torch.set_num_threads(1)
    ev, cnt, T_of = read(root)
    train, test = sorted(r for r in T_of if T_of[r] != T_star), sorted(r for r in T_of if T_of[r] == T_star)
    tab = closure.Tables(ev, cnt, T_of, train, a["min_temps"], a["anchors"], a["coverage"], not a["no_absorb"], not a["pooled"], not a["no_volume"], a["t0"])
    A, closed = tab.A, not a["no_closure"]
    lines = tab.lines(train, A)
    res = {"T": T_star, "species": len(tab.M), "anchors": len(tab.anchor_set), "reactions": int(len(A)),
           "coverage": float(tab.n_train[A].sum() / sum(k for T in tab.temps for (lhs, _), k in tab.events[T].items() if tab.explicit(lhs)))}
    res["counted"] = evaluate(tab, cnt, test, lambda r: tab.model(lines, A, T_star, closed), A)
    if a["fit"]:
        t0 = time.time()
        at, res["fit_loss"], m0 = fit_models(tab, cnt, train, lines, A, closed)
        res["fit"] = evaluate(tab, cnt, test, lambda r: at(T_star), A)
        res["Ea_shift_median"] = float(np.median(np.abs(100 * m0.Ea.detach().numpy() - 1000.0 * lines["b"] * 8.314462618e-3)))
        res["fit_minutes"] = (time.time() - t0) / 60
    if a["full"]:
        res["full"] = full_network(tab, ev, cnt, train, test, T_star, closed)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-temps", type=int, default=1)
    ap.add_argument("--anchors", type=int, default=80)
    ap.add_argument("--coverage", type=float, default=0.95)
    ap.add_argument("--no-absorb", action="store_true")
    ap.add_argument("--pooled", action="store_true", help="one net-change matrix for all temperatures")
    ap.add_argument("--no-closure", action="store_true")
    ap.add_argument("--no-volume", action="store_true")
    ap.add_argument("--t0", type=float, default=0.0, help="ps: model and metric start at the observed state at t0")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    a = vars(ap.parse_args())
    workers = a.pop("workers")
    with Pool(workers) as pool:
        folds = pool.map(fold, [(T, a, HFO) for T in (2500.0, 2750.0, 3000.0, 3250.0, 3500.0)])
    OUT.mkdir(parents=True, exist_ok=True)
    slim = [{k: ({kk: vv for kk, vv in v.items() if not kk.endswith("_terms")} if isinstance(v, dict) else v) for k, v in r.items()} for r in folds]
    with open(OUT / "dev_c2.jsonl", "a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "variant": a, "folds": slim}) + "\n")
    print(f"variant {a}")
    for r in folds:
        line = (f"  {r['T']:.0f} K: species {r['species']}, anchors {r['anchors']}, reactions {r['reactions']} (cover {r['coverage']:.0%}); "
                f"counted error {r['counted']['error']:.4f} firing {r['counted']['firing']:.1f} min count {r['counted']['most_negative_count']:.1f}")
        if "fit" in r:
            line += (f" | fit error {r['fit']['error']:.4f} firing {r['fit']['firing']:.1f}; ratio error {r['counted']['error'] / r['fit']['error']:.3f} "
                     f"firing {r['counted']['firing'] / r['fit']['firing']:.3f} ({r['fit_minutes']:.0f} min)")
        if "full" in r:
            line += f" | full {r['full']['reactions']} rx error {r['full']['error']:.4f}; compact/full {r['counted']['error'] / r['full']['error']:.3f}"
        print(line)
    pool_ = lambda key: pooled_error([x for r in folds for x in r[key]["error_terms"]])
    line = f"  pooled counted error {pool_('counted'):.4f}"
    if a["fit"]:
        fc, ff = (float(np.mean([x for r in folds for x in r[k]["firing_terms"]])) for k in ("counted", "fit"))
        line += f"; fit {pool_('fit'):.4f}; H2b ratio {pool_('counted') / pool_('fit'):.3f}; H2a firing ratio {fc / ff:.3f}"
    if a["full"]:
        line += f"; full {pool_('full'):.4f}; H3 ratio {pool_('counted') / pool_('full'):.3f}, size factor min {min(r['full']['reactions'] / r['reactions'] for r in folds):.1f}"
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
