# file: experiments/v4_h0r.py
"""V4 / H0r(a) (design sections 2 and 6), training runs only. The same anchored reactions (from all training
runs) in 10 bootstrap resamples of the training runs, drawn with replacement within each temperature. E_a comes
from counted rates and from the concentration-only fit (lam 0, L-BFGS from the counted start). H0r(a) passes if
the flux-weighted median width of the 90 % E_a intervals of counted rates is <= 0.5 x that of the fit.
(H0r(b), the free-structure model, was cut on 2026-11-04.)

  python -I experiments/v4_h0r.py
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import fit  # noqa: E402
from data import load_stanford  # noqa: E402

TRAIN_T = (4000, 4500, 5000)


def resample(args):
    tables, seed = args
    torch.set_num_threads(1)
    runs = pd.read_csv(Path(tables) / "runs.csv")
    train = runs[(runs.molecule == "C4H10") & runs.T_K.isin(TRAIN_T)]
    groups, S, O, _ = load_stanford(tables, train.run_id.tolist())
    carbon = np.array(json.loads((Path(tables) / "classes.json").read_text())["composition"])[:, 0]
    rng = np.random.default_rng(seed)
    ids = [rid for _, g in train.groupby("T_K").run_id for rid in rng.choice(g.tolist(), len(g), replace=True)]
    boot = load_stanford(tables, ids)[0]
    m = fit.build(groups, S, O, carbon)  # anchors from all training runs: the same reactions in every resample
    fit.set_counted_rates(m, boot, O)
    counted = 100 * m.Ea.detach().numpy()
    fit.lbfgs(m, boot, 0.0, rounds=5)
    events = sum(g.ev.sum((0, 1)) for g in groups).numpy()[m.types.numpy()]
    return counted, 100 * m.Ea.detach().numpy(), events


def weighted_median(x, w):
    o = np.argsort(x)
    return float(x[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "data" / "processed" / "stanford"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "v4" / "h0r.json"))
    a = ap.parse_args(argv)
    with Pool(a.workers) as pool:
        out = pool.map(resample, [(a.tables, s) for s in range(a.n)])
    counted, fitted, events = np.array([o[0] for o in out]), np.array([o[1] for o in out]), out[0][2]
    width = lambda e: np.percentile(e, 95, axis=0) - np.percentile(e, 5, axis=0)
    wc, wf = weighted_median(width(counted), events), weighted_median(width(fitted), events)
    res = {"resamples": a.n, "reactions": int(counted.shape[1]), "width_counted": wc, "width_fit": wf,
           "ratio": wc / wf, "H0r_a": bool(wc <= 0.5 * wf)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"flux-weighted median 90 % E_a interval: counted rates {wc:.1f} kJ/mol, concentration fit {wf:.1f} kJ/mol "
          f"-> ratio {wc / wf:.2f}")
    print(f"  H0r(a) counted width <= 0.5 x fitted width: {'PASS' if res['H0r_a'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
