# file: experiments/round3_score.py
"""Round 3 scoring (design section 13), written before the confirmation ran: H6' for design.anchored on the lumped
confirmation set and on the standard set (possible + impossible), from results/round3/confirm_*.jsonl.

  python -I experiments/round3_score.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "round3"


def h6(rows, d, t):
    cells, half = [], []
    for T in sorted({r["target"] for r in rows}):
        sel = [r for r in rows if r["target"] == T]
        for f in ("0.1", "0.25", "0.5"):
            med = {k: float(np.median([r["c5"][f][k] for r in sel])) for k in (d, "hot_only", "uniform", "median_random")}
            cells.append((med[d], (med["hot_only"], med["uniform"], med["median_random"])))
        half.append(float(np.median([r["c5"]["0.5"][d] / r["full"] for r in sel])))
    near = float(np.mean([v <= t["near_best"] * min(n) for v, n in cells]))
    gain = float(np.mean([v <= t["gain_over_worst"] * max(n) for v, n in cells]))
    hm = float(np.median(half))
    return {"networks": len(rows), "near_best_fraction": near, "gain_fraction": gain, "half_vs_full_median": hm,
            "pass": bool(near >= t["cell_fraction"] and gain >= t["cell_fraction"] and hm <= t["half_vs_full_median"])}


def main():
    t = json.loads((OUT / "FROZEN.json").read_text())["H6"]
    rows = {k: [json.loads(l) for f in sorted(glob.glob(str(OUT / f"confirm_{k}_T*.jsonl"))) for l in open(f)]
            for k in ("lumped", "possible", "impossible")}
    res = {"lumped": h6(rows["lumped"], "anchored", t), "standard": h6(rows["possible"] + rows["impossible"], "anchored", t),
           "possible (descriptive)": h6(rows["possible"], "anchored", t), "impossible (descriptive)": h6(rows["impossible"], "anchored", t),
           "round-2 design on lumped (descriptive)": h6(rows["lumped"], "designed", t)}
    res["adopted"] = bool(res["lumped"]["pass"] and res["standard"]["pass"])
    (OUT / "score.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:>40}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
