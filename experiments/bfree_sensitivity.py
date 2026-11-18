# file: experiments/bfree_sensitivity.py
"""Decision D3: how much the V2 result depends on the Arrhenius form. The compact model is rebuilt as in V2;
its final rates then come from ln k = a + b ln T + c / T fitted through the training temperatures (count-
weighted; reactions seen at fewer than 3 temperatures keep the b = 0 rates). The test runs are isothermal, so
each temperature runs with its own ln k(T). Reported only; the main method keeps b = 0.

  python -I experiments/bfree_sensitivity.py
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fit  # noqa: E402
from data import load_stanford  # noqa: E402
from v2_extrapolation import stacked_error, temps  # noqa: E402


def counted_k(groups, O, types):
    """Events and exposure per temperature for the given types: (temperatures, events, k)."""
    o, num, den = torch.as_tensor(O[:, types]), {}, {}
    for g in groups:
        expo = torch.trapezoid(torch.exp(torch.log(g.obs.clamp_min(1e-12)) @ o), g.t, dim=0) * g.V[:, None]
        n = g.ev[..., types].sum(1)
        for b, T in enumerate(g.T.tolist()):
            num[T], den[T] = num.get(T, 0) + n[b], den.get(T, 0) + expo[b]
    Ts = np.array(sorted(num))
    n = torch.stack([num[T] for T in Ts]).numpy()
    return Ts, n, n / np.maximum(torch.stack([den[T] for T in Ts]).numpy(), 1e-300)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "data" / "processed" / "stanford"))
    ap.add_argument("--v2", default=str(ROOT / "results" / "v2" / "v2.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "v2" / "bfree.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    split = json.loads(Path(a.v2).read_text())["split"]
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(Path(a.tables) / "runs.csv"))
    ids = lambda Ts: butane[butane.T_K.isin(Ts)].run_id.tolist()
    train_g, S, O, _ = load_stanford(a.tables, ids(temps(split["fit_T"]) + temps(split["select_T"])))
    fit_g, sel_g = load_stanford(a.tables, ids(temps(split["fit_T"])))[0], load_stanford(a.tables, ids(temps(split["select_T"])))[0]
    carbon = np.array(json.loads((Path(a.tables) / "classes.json").read_text())["composition"])[:, 0]
    m = fit.build(train_g, S, O, carbon)
    fit.set_counted_rates(m, fit_g, O)
    m, _ = fit.reduce(m, fit_g, sel_g, 0.0, rounds=0, log=lambda s: None)
    fit.set_counted_rates(m, train_g, O)
    Ts, n, k = counted_k(train_g, O, m.types.numpy())
    coef, curved = {}, 0
    for j in range(len(m.types)):
        seen = n[:, j] > 0
        if m.active[j] > 0 and seen.sum() >= 3:
            X = np.column_stack([np.ones(seen.sum()), np.log(Ts[seen]), 1 / Ts[seen]]) * np.sqrt(n[seen, j])[:, None]
            coef[j] = np.linalg.lstsq(X, np.log(k[seen, j]) * np.sqrt(n[seen, j]), rcond=None)[0]
            curved += 1
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    per_run = [(rid, load_stanford(a.tables, [rid])[0][0]) for rid in ids(temps(split["test_T"]))]
    free = {}
    for T in sorted({T_of[rid] for rid, _ in per_run}):  # isothermal runs: one set of rates per temperature
        f = copy.deepcopy(m)
        with torch.no_grad():
            lnk = f.ln_k(torch.tensor([T], dtype=torch.float64))[0].clone()
            for j, (c0, b, c1) in coef.items():
                lnk[j] = c0 + b * np.log(T) + c1 / T
            f.lnA.copy_(lnk)
            f.Ea.zero_()
        free[T] = f
    with torch.no_grad():
        res = {"split": split, "reactions": m.n_active(), "reactions_with_b": curved,
               "test_error (b = 0)": stacked_error(lambda rid, g: fit.simulate(m, g)[0][:, 0].numpy(), per_run),
               "test_error (b free)": stacked_error(lambda rid, g: fit.simulate(free[T_of[rid]], g)[0][:, 0].numpy(), per_run)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{m.n_active()} reactions ({curved} with b fitted): test error b = 0 {res['test_error (b = 0)']:.4f}, "
          f"b free {res['test_error (b free)']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
