# file: experiments/v3_transfer.py
"""V3 / H4 (design sections 2 and 6): the compact model built exactly as in V2 (butane, fit 4500-5000 K,
select 4000 K, final rates counted at all training temperatures) predicts the methane runs. H4 passes if
their error is <= 2 x the in-domain V2 test error. Run V2 first.

  python -I experiments/v3_transfer.py
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
from v2_extrapolation import stacked_error  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "data" / "processed" / "stanford"))
    ap.add_argument("--v2", default=str(ROOT / "results" / "v2" / "v2.json"))
    ap.add_argument("--test-molecule", default="CH4")
    ap.add_argument("--out", default=str(ROOT / "results" / "v3" / "v3.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    v2 = json.loads(Path(a.v2).read_text())
    runs = pd.read_csv(Path(a.tables) / "runs.csv")
    butane = runs[runs.molecule == "C4H10"]
    T = lambda s: [float(x) for x in s.split(",")]
    fit_ids = butane[butane.T_K.isin(T(v2["split"]["fit_T"]))].run_id.tolist()
    sel_ids = butane[butane.T_K.isin(T(v2["split"]["select_T"]))].run_id.tolist()
    train_g, S, O, _ = load_stanford(a.tables, fit_ids + sel_ids)
    fit_g, sel_g = load_stanford(a.tables, fit_ids)[0], load_stanford(a.tables, sel_ids)[0]
    carbon = np.array(json.loads((Path(a.tables) / "classes.json").read_text())["composition"])[:, 0]
    m = fit.build(train_g, S, O, carbon)
    fit.set_counted_rates(m, fit_g, O)
    m, _ = fit.reduce(m, fit_g, sel_g, 0.0, rounds=0, log=lambda s: None)
    fit.set_counted_rates(m, train_g, O)
    test = [(rid, load_stanford(a.tables, [rid])[0][0]) for rid in runs[runs.molecule == a.test_molecule].run_id]
    with torch.no_grad():
        err = stacked_error(lambda rid, g: fit.simulate(m, g)[0][:, 0].numpy(), test)
    in_domain = v2["compact"]["test_error"]
    res = {"test_molecule": a.test_molecule, "reactions": m.n_active(), "transfer_error": err,
           "in_domain_error": in_domain, "H4": bool(err <= 2 * in_domain)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{a.test_molecule} runs: error {err:.4f} with {m.n_active()} reactions; in-domain V2 error {in_domain:.4f}")
    print(f"  H4 transfer error <= 2 x in-domain: {'PASS' if res['H4'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
