# file: src/check_tables.py
"""Sanity checks on data/processed/stanford/ after build_tables.py.

  python -I src/check_tables.py data/processed/stanford
1. Coverage: every formula maps to a declared class.
2. Element balance of the lumping (design section 4): totals implied by class amounts x mean composition
   may drift from the true totals by at most 2 %; the classes contributing most are listed, so the
   candidate knows which rule in assign_class to refine.
3. True element totals are identical in every frame (complete species).
4. Carbon balance of every lumped event type (exact when the carbon-bearing classes count carbon).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lumping import CLASSES, assign_class, lump  # noqa: E402


def check(folder, tol=0.02):
    folder = Path(folder)
    meta = json.loads((folder / "classes.json").read_text())
    units, comp = dict(zip(meta["classes"], meta["units"])), np.array(meta["composition"])
    formulas, drift, contrib, ok = set(), 0.0, np.zeros(len(CLASSES)), True
    for f in sorted((folder / "species").glob("*.csv.gz")):
        if f.name.endswith("_events.csv.gz"):
            continue
        w = pd.read_csv(f).pivot_table(index="t_ps", columns="formula", values="count", fill_value=0)
        formulas |= set(w.columns)
        Y, A = lump(list(w.columns), w.to_numpy(float), units)
        true = A.sum(axis=1)
        if not (true == true[0]).all():
            ok = False
            print(f"{f.name}: element totals change between frames")
        scale = np.where(true[0] > 0, true[0], 1.0)
        drift = max(drift, float((np.abs(Y @ comp - true) / scale).max()))
        contrib = np.maximum(contrib, (np.abs(A - Y[:, :, None] * comp[None]) / scale).max(axis=(0, 2)))
    bad = sorted(f for f in formulas if assign_class(f) not in CLASSES)
    nu = np.array([t["nu"] for t in json.loads((folder / "event_types.json").read_text())])
    c_err = float(np.abs(nu @ comp[:, 0]).max()) if len(nu) else 0.0
    top = ", ".join(f"{CLASSES[i]} {contrib[i]:.3f}" for i in np.argsort(contrib)[::-1][:4])
    ok &= not bad and drift <= tol
    print(f"stanford: {len(formulas)} formulas, unmapped: {bad or 'none'}; element drift {drift:.3f} (limit {tol}); "
          f"largest class contributions: {top}; event types {len(nu)}, max carbon imbalance {c_err:.3f} "
          f"-> {'OK' if ok else 'PROBLEM'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([check(p) for p in sys.argv[1:]]) else 1)
