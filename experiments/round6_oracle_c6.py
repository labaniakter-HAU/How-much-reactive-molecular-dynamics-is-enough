# file: experiments/round6_oracle_c6.py
"""Round 6, C6: the best accuracy ANY foresight flag can reach on a synthetic network kind (an oracle bound, like
round 1's design oracle). Development seeds only (below 40000).

For each campaign the oracle knows the true rates, the campaign's exposures and the estimator. It does not know the
Poisson draw of the events. Redrawing the events R times (same exposures, same true rates, and for overdispersed
networks fresh run factors) gives p = P(label untrustworthy). The best flag says "untrustworthy" when p > 1/2, and its
expected accuracy is mean(max(p, 1 - p)). The realized labels are reported beside it.

  python -I -X utf8 experiments/round6_oracle_c6.py --seeds 20000 20025 --target 3100 --reps 40 --workers 2
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import design  # noqa: E402

OUT = ROOT / "results" / "round6"
L2 = float(np.log(2))


def one(args):
    seed, T_STAR, reps = args
    from round2_synthetic import BUDGETS, CLIP
    from round5_flag import kind5
    from synthetic import random_network, run_data, true_lnk
    from v0b_reduction import RX, SPECIES, V, X0
    from v0c_synthetic import RUNS
    assert seed < 40000, "development seeds only"
    net = random_network(RX, seed, kind5(seed))
    rng = np.random.default_rng(seed + 100000)
    names = [f"{T}K_{s}" for T, s, _ in RUNS]
    T_of = {nm: float(T) for nm, (T, _, _) in zip(names, RUNS)}
    bins = {nm: b for nm, (_, _, b) in zip(names, RUNS)}
    menu = {nm: [L for L in (50, 100, 200) if L < bins[nm]] + [bins[nm]] for nm in names if T_of[nm] != T_STAR}
    data = {}
    for nm in names:
        data[nm], _ = run_data(net, SPECIES, T_of[nm], bins[nm], X0, V, menu.get(nm, [bins[nm]]), rng, n_sub=4)
    truth = true_lnk(net, T_STAR)
    w = np.exp(truth) * sum(data[nm][bins[nm]][1] for nm in names if T_of[nm] == T_STAR)
    A = np.arange(len(RX))
    total, pilot, crng = sum(bins[nm] for nm in menu), {"4500K_1" if T_STAR == 5000.0 else "5000K_1": 50}, np.random.default_rng(0)
    camps = [{nm: bins[nm] for nm in menu}]
    for f in BUDGETS:
        Bud = int(f * total)
        camps += [design.greedy(menu, pilot, Bud, T_of, T_STAR), design.hot_only(menu, pilot, Bud, T_of), design.uniform(menu, pilot, Bud, T_of)]
        camps += [design.random_campaign(menu, pilot, Bud, crng) for _ in range(100)][:3]
    lnk_T = {T: true_lnk(net, T) for T in set(T_of.values())}
    mrng = np.random.default_rng(seed + 900000)
    rows = []
    for camp in camps:
        rms = lambda d: float(np.sqrt((w * np.minimum(np.abs(design.calibrated_re(camp, d, T_of, A, T_STAR)[0] - truth), CLIP) ** 2).sum() / w.sum()))
        actual = rms(data)
        bad = 0
        for _ in range(reps):
            d = {}
            for r, L in camp.items():
                if L <= 0:
                    continue
                E = data[r][L][1]
                lnk = lnk_T[T_of[r]] + (mrng.normal(0.0, net["sd_run"], len(E)) if net["sd_run"] else 0.0)
                d[r] = {L: (mrng.poisson(np.exp(lnk) * E).astype(float), E)}
            bad += rms(d) > L2
        rows.append({"actual": actual, "p_bad": bad / reps, "temps": len({T_of[r] for r, L in camp.items() if L})})
    return {"seed": seed, "kind": kind5(seed), "target": T_STAR, "campaigns": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.map(one, [(s, a.target, a.reps) for s in range(*a.seeds)])
    (OUT / f"oracle_c6_{a.seeds[0]}_{a.seeds[1]}_T{a.target:g}.json").write_text(json.dumps(res))
    rows = [c for r in res for c in r["campaigns"]]
    p = np.array([c["p_bad"] for c in rows])
    bad = np.array([c["actual"] > L2 for c in rows])
    fl = p > 0.5
    print(f"seeds {a.seeds} T {a.target:g} ({sorted({r['kind'] for r in res})}): campaigns {len(rows)}, untrustworthy {bad.sum()} ({bad.mean():.0%})")
    print(f"   oracle expected accuracy mean(max(p, 1-p)) = {np.maximum(p, 1 - p).mean():.3f}; share of campaigns with 0.1 < p < 0.9: {np.mean((p > 0.1) & (p < 0.9)):.0%}")
    print(f"   oracle flag (p > 1/2) on the realized labels: accuracy {(fl == bad).mean():.3f}, caught {fl[bad].mean() if bad.any() else float('nan'):.2f}, passed {(~fl[~bad]).mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
