# file: experiments/round3_dev_real.py
"""Round 3 development (design section 13): C5 design variants on all 9 butane design targets. These targets were
used by rounds 1-2, so this is development (after the fact), never confirmation.

  python -I experiments/round3_dev_real.py
Per target and budget: RMS (Poisson mean, the C5 metric) of each design, hot-only, uniform, median of 100 random,
and the full campaign; then H6' quantities per design.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import design  # noqa: E402
import fit  # noqa: E402
from data import load_stanford  # noqa: E402

TABLES, BUDGETS, CLIP = ROOT / "data" / "processed" / "stanford", (0.10, 0.25, 0.50), 10.0
DESIGNS = ("static", "bias8", "seq8", "seq_est", "nearest", "curv35", "curv70", "curv140")


def target(args):
    T_star, only = args
    import torch
    torch.set_num_threads(1)
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g = {}
    for rid in butane.run_id:
        (g[rid],), S, O, _ = load_stanford(TABLES, [rid])
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    nb = {r: g[r].ev.shape[1] for r in g}
    tgt = [r for r in g if T_of[r] == T_star]
    menu = {r: [L for L in (50, 100, 200) if L < nb[r]] + [nb[r]] for r in g if T_of[r] != T_star}
    data = {r: design.truncations(g[r], O, menu.get(r, [nb[r]])) for r in g}
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in tgt}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))

    def rms(camp):
        Ts, n, E = design.pool(camp, data, T_of)
        p = design.predict(design.fit_lines(n[:, A], E[:, A], Ts), T_star)[0]
        return float(np.sqrt((w * np.minimum(np.abs(p - obs), CLIP) ** 2).sum() / w.sum()))

    pilot = {[r for r in menu if T_of[r] == (4500.0 if T_star == 5000.0 else 5000.0) and "_1_" in r][0]: 50}
    total, rng, out = sum(nb[r] for r in menu), np.random.default_rng(int(T_star)), {"full": None, "budgets": {}}
    out["full"] = rms({r: nb[r] for r in menu})
    for f in BUDGETS:
        B = int(f * total)
        full_menu = {r: [max(Ls)] for r, Ls in menu.items()}  # complete runs only: lumped rates drift within a run
        scale = design.pilot_scale(pilot, data, A)
        make = {"adapt": lambda: design.adaptive(menu, pilot, B, T_of, T_star, data, A),
                "static_full": lambda: design.greedy(full_menu, pilot, B, T_of, T_star),
                "curv70_full": lambda: design.greedy_curv(full_menu, pilot, B, T_of, T_star, 70.0, scale),
                "adapt_full": lambda: design.adaptive(full_menu, pilot, B, T_of, T_star, data, A),
                "nearest_full": lambda: design.nearest_first(full_menu, pilot, B, T_of, T_star),
                "adapt2": lambda: design.adaptive(menu, pilot, B, T_of, T_star, data, A, common_window=True),
                "anchored": lambda: design.anchored(menu, pilot, B, T_of, T_star, data, A)}
        if only:
            camps = {k: make[k]() for k in only}
            camps.update(hot_only=design.hot_only(menu, pilot, B, T_of), uniform=design.uniform(menu, pilot, B, T_of))
            r = {k: rms(c) for k, c in camps.items()}
            r["median_random"] = float(np.median([rms(design.random_campaign(menu, pilot, B, rng)) for _ in range(100)]))
            out["budgets"][str(f)] = r
            continue
        camps = {"static": design.greedy(menu, pilot, B, T_of, T_star),
                 "bias8": design.greedy_bias(menu, pilot, B, T_of, T_star, design.TAU2_PRIOR, design.pilot_scale(pilot, data, A)),
                 "seq8": design.sequential(menu, pilot, B, T_of, T_star, data, A, tau2=design.TAU2_PRIOR),
                 "seq_est": design.sequential(menu, pilot, B, T_of, T_star, data, A, tau2=None),
                 "nearest": design.nearest_first(menu, pilot, B, T_of, T_star),
                 **{f"curv{gm:g}": design.greedy_curv(menu, pilot, B, T_of, T_star, gm, design.pilot_scale(pilot, data, A)) for gm in (35.0, 70.0, 140.0)},
                 "hot_only": design.hot_only(menu, pilot, B, T_of), "uniform": design.uniform(menu, pilot, B, T_of)}
        r = {k: rms(c) for k, c in camps.items()}
        r["median_random"] = float(np.median([rms(design.random_campaign(menu, pilot, B, rng)) for _ in range(100)]))
        out["budgets"][str(f)] = r
    print(f"{T_star:g} K done", flush=True)
    return f"{T_star:g}", out


def main():
    only = [d for d in (sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else []) if d]
    temps = sorted(set(pd.read_csv(TABLES / "runs.csv").query("molecule == 'C4H10'").T_K.astype(float)))
    with Pool(3) as p:
        res = dict(p.map(target, [(T, only) for T in temps]))
    out = ROOT / "results" / "round3" / ("dev_real.json" if not only else f"dev_real_{'_'.join(only)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for d in (only or DESIGNS):
        near, gain, half = [], [], []
        for T, r in res.items():
            for f, b in r["budgets"].items():
                naive = (b["hot_only"], b["uniform"], b["median_random"])
                near.append(b[d] <= 1.1 * min(naive))
                gain.append(b[d] <= 0.9 * max(naive))
            half.append(r["budgets"]["0.5"][d] / r["full"])
        print(f"{d:>8}: near-best {np.mean(near):.2f}, gain {np.mean(gain):.2f}, median half/full {np.median(half):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
