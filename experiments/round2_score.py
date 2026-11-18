# file: experiments/round2_score.py
"""Round 2 scoring (design section 12): applies results/round2/FROZEN.json to the confirmation outputs,
results/round2/confirm_real.json and results/round2/confirm_synth_*.jsonl. Written before the confirmation ran.

  python -I experiments/round2_score.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "round2"


def h5(z, t):
    a = np.abs(np.asarray(z, float))
    a = a[np.isfinite(a)]
    cov, med = float((a <= 2).mean()), float(np.median(a))
    return {"n": int(a.size), "coverage": cov, "median_abs_z": med,
            "pass": bool(cov >= t["coverage_min"] and t["median_lo"] <= med <= t["median_hi"])}


def h6(cells, full_ratio, t):
    """cells: [(designed, hot, uniform, random)]; full_ratio: designed at 50 % / full per target."""
    near = [d <= t["near_best"] * min(h, u, r) for d, h, u, r in cells]
    gain = [d <= t["gain_over_worst"] * max(h, u, r) for d, h, u, r in cells]
    med = float(np.median(full_ratio))
    res = {"cells": len(cells), "near_best_fraction": float(np.mean(near)), "gain_fraction": float(np.mean(gain)),
           "half_vs_full_median": med}
    res["pass"] = bool(res["near_best_fraction"] >= t["cell_fraction"] and res["gain_fraction"] >= t["cell_fraction"]
                       and med <= t["half_vs_full_median"])
    return res


def h7(bad, flagged, t):
    bad, fl = np.asarray(bad, bool), np.asarray(flagged, bool)
    res = {"targets": int(bad.size), "untrustworthy": int(bad.sum()), "flagged": int(fl.sum()),
           "accuracy": float((bad == fl).mean()),
           "untrustworthy_caught": float(fl[bad].mean()) if bad.any() else None,
           "trustworthy_passed": float((~fl[~bad]).mean()) if (~bad).any() else None,
           "testable": bool(bad.sum() >= t["min_per_class"] and (~bad).sum() >= t["min_per_class"])}
    res["pass"] = bool(res["accuracy"] >= t["overall"] and res["untrustworthy_caught"] >= t["per_class"]
                       and res["trustworthy_passed"] >= t["per_class"]) if res["testable"] else None
    return res


def main():
    frozen = json.loads((OUT / "FROZEN.json").read_text())
    real = json.loads((OUT / "confirm_real.json").read_text())
    res = {"frozen_at": frozen["frozen_at"], "real": {}, "synthetic": {}}

    z, cells, full_ratio, bad, fl = [], [], [], [], []
    for T, r in real["targets"].items():
        z += r["z"]
        for f, b in r["budgets"].items():
            c = b["campaigns"]
            cells.append((c["designed"]["rms"], c["hot_only"]["rms"], c["uniform"]["rms"], b["median_random"]))
            for k in c.values():
                bad.append(k["bad"])
                fl.append(k["flagged"])
        full_ratio.append(r["budgets"]["0.5"]["campaigns"]["designed"]["rms"] / r["full"]["rms"])
        bad.append(r["full"]["bad"])
        fl.append(r["full"]["flagged"])
    if "octane" in real:
        bad.append(real["octane"]["bad"])
        fl.append(real["octane"]["flagged"])
    res["real"] = {"H5": h5(z, frozen["H5"]), "H6": h6(cells, full_ratio, frozen["H6"]), "H7": h7(bad, fl, frozen["H7"])}

    rows = [json.loads(l) for f in sorted(glob.glob(str(OUT / "confirm_synth_*.jsonl"))) for l in open(f)]
    if rows:
        for kind in ("possible", "impossible"):
            sel = [r for r in rows if (r["kind"] == "possible") == (kind == "possible")]
            res["synthetic"][f"H5_{kind}"] = h5(np.concatenate([r["z_round2re"] for r in sel]), frozen["H5"])
        cells, full_ratio = [], []
        for T in sorted({r["target"] for r in rows}):
            sel = [r for r in rows if r["target"] == T]
            for f in ("0.1", "0.25", "0.5"):
                cells.append(tuple(float(np.median([r["c5"][f][k] for r in sel])) for k in ("designed", "hot_only", "uniform", "median_random")))
            full_ratio.append(float(np.median([r["c5"]["0.5"]["designed"] / r["full"] for r in sel])))
        res["synthetic"]["H6"] = h6(cells, full_ratio, frozen["H6"])
        c6 = [c for r in rows for c in r["c6"]]
        res["synthetic"]["H7"] = h7([c["rms_re"] > frozen["label_rms"] for c in c6],
                                    [c["single"] > 0.5 or c["err_re"] > frozen["flag_err"] for c in c6], frozen["H7"])
        res["synthetic"]["networks"] = len(rows)
    (OUT / "score.json").write_text(json.dumps(res, indent=1))
    for part, d in res.items():
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f"{part:>9} {k:>13}: " + ", ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in v.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
