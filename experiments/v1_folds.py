# file: experiments/v1_folds.py
"""V1 (design section 6): leave one temperature out on the butane runs, compact model only (descriptive, no
hypothesis). Fold T: train on the other temperatures of the pool, select on the lowest of them, test on T.
Run after V2, which needs the 3100-3500 K runs unseen.

  python -I experiments/v1_folds.py --T 3300      (one fold; start the folds in parallel)
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

import fit  # noqa: E402
from data import load_stanford  # noqa: E402
from v2_extrapolation import seed_scatter, stacked_error  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "data" / "processed" / "stanford"))
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--pool", default=None, help="temperatures to use (default: all butane temperatures)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(Path(a.tables) / "runs.csv"))
    pool = sorted([float(x) for x in a.pool.split(",")] if a.pool else set(butane.T_K.astype(float)))
    train_T = [T for T in pool if T != a.T]
    sel_T, fit_T = train_T[0], train_T[1:]
    ids = lambda Ts: butane[butane.T_K.isin(Ts)].run_id.tolist()
    train_g, S, O, classes = load_stanford(a.tables, ids(train_T))
    fit_g, sel_g = (load_stanford(a.tables, ids(Ts))[0] for Ts in (fit_T, [sel_T]))
    carbon = np.array(json.loads((Path(a.tables) / "classes.json").read_text())["composition"])[:, 0]
    m = fit.build(train_g, S, O, carbon)
    fit.set_counted_rates(m, fit_g, O)
    m, _ = fit.reduce(m, fit_g, sel_g, 0.0, rounds=0, log=lambda s: None)
    fit.set_counted_rates(m, train_g, O)
    test = butane[butane.T_K == a.T]
    per_run = [(rid, load_stanford(a.tables, [rid])[0][0]) for rid in test.run_id]
    with torch.no_grad():
        err = stacked_error(lambda rid, g: fit.simulate(m, g)[0][:, 0].numpy(), per_run)
    res = {"T": a.T, "fit_T": fit_T, "select_T": sel_T, "reactions": m.n_active(), "test_error": err,
           "seed_scatter": seed_scatter(a.tables, test, classes) if len(test) == 2 else None}
    out = Path(a.out or ROOT / "results" / "v1" / f"T{a.T:g}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"V1 fold {a.T:g} K: {m.n_active()} reactions, test error {res['test_error']:.4f}"
          + (f", seed-to-seed scatter {res['seed_scatter']:.4f}" if res["seed_scatter"] is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
