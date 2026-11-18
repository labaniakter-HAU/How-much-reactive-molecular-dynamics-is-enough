# file: experiments/h3b_baselines.py
"""H3(b) (design section 2, restated as decision D2): on the V2 split, the compact model's error in excess of
single-run noise must be <= 0.5 x that of the best simple baseline (single-step Arrhenius, XGBoost).
Single-run noise = seed-to-seed scatter of the test runs / sqrt(2). Run V2 first: the compact model's test
error is read from its JSON.

  python -I experiments/h3b_baselines.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines import arrhenius_fit, xgboost_fit  # noqa: E402
from data import load_stanford  # noqa: E402
from v2_extrapolation import seed_scatter, stacked_error, temps  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "data" / "processed" / "stanford"))
    ap.add_argument("--v2", default=str(ROOT / "results" / "v2" / "v2.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "v2" / "h3b.json"))
    a = ap.parse_args(argv)
    v2 = json.loads(Path(a.v2).read_text())
    split = v2["split"]
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(Path(a.tables) / "runs.csv"))
    train = butane[butane.T_K.isin(temps(split["fit_T"]) + temps(split["select_T"]))]
    test = butane[butane.T_K.isin(temps(split["test_T"]))]

    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    run = lambda rid: load_stanford(a.tables, [rid])[0][0]
    tr = [(T_of[rid], g.t.numpy(), g.obs.numpy()[:, 0]) for rid, g in ((r, run(r)) for r in train.run_id)]
    per_run = [(rid, run(rid)) for rid in test.run_id]
    errors = {}
    for name, fitter in (("single-step Arrhenius", arrhenius_fit), ("XGBoost", xgboost_fit)):
        predict = fitter(tr)
        errors[name] = stacked_error(lambda rid, g: predict(T_of[rid], g.t.numpy(), g.obs.numpy()[0, 0]), per_run)
    classes = json.loads((Path(a.tables) / "classes.json").read_text())["classes"]
    noise = seed_scatter(a.tables, test, classes) / np.sqrt(2)
    compact, best = v2["compact"]["test_error"], min(errors.values())
    excess = lambda e: max(e - noise, 0.0)  # floored: a baseline at the noise level cannot be beaten 2x
    ratio = excess(compact) / excess(best) if excess(best) > 0 else (0.0 if excess(compact) == 0 else float("inf"))
    res = {"split": split, "baseline_errors": errors, "single_run_noise": noise, "compact_error": compact,
           "excess_ratio": ratio, "H3b": bool(excess(compact) <= 0.5 * excess(best))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"baselines: " + ", ".join(f"{k} {v:.4f}" for k, v in errors.items()) + f"; single-run noise {noise:.4f}; "
          f"compact {compact:.4f} -> excess ratio {ratio:.2f}")
    print(f"  H3(b) excess error <= 0.5 x best simple baseline's: {'PASS' if res['H3b'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
