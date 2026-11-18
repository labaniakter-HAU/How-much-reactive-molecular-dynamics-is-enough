# file: experiments/v6_calibration.py
"""V6 / H5 (design section 11, C4): leave one temperature out over the 9 butane temperatures. Rates and their
covariance from the other temperatures' events (design.fit_lines, phi included); for each anchor of the
training runs with >= 10 events in the held-out runs, z = (predicted - counted ln k at T*) /
sqrt(sd_pred^2 + 1/n*), with n* and the exposure pooled over the held-out runs.
Pass: >= 85 % of |z| <= 2 and 0.5 <= median |z| <= 1.0 (calibrated: 0.67). Also reported without phi.

  python -I experiments/v6_calibration.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import design  # noqa: E402
import fit  # noqa: E402
from data import load_stanford  # noqa: E402

TABLES = ROOT / "data" / "processed" / "stanford"


def main():
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g, O = {}, None
    for rid in butane.run_id:
        (g[rid],), _, O, _ = load_stanford(TABLES, [rid])
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    full = {r: g[r].ev.shape[1] for r in g}
    data = {r: design.truncations(g[r], O, [full[r]]) for r in g}
    folds, z, z0 = [], [], []
    for T in sorted(set(T_of.values())):
        train = {r: L for r, L in full.items() if T_of[r] != T}
        A = fit.anchors([g[r] for r in train])
        Ts, n, E = design.pool(train, data, T_of)
        lines = design.fit_lines(n[:, A], E[:, A], Ts)
        _, nh, Eh = design.pool({r: L for r, L in full.items() if T_of[r] == T}, data, T_of)
        nh, Eh = nh[0, A], Eh[0, A]
        keep = nh >= 10
        obs = np.log(nh[keep] / Eh[keep])
        fold = {"T": T, "anchors": int(len(A)), "evaluated": int(keep.sum()), "median_phi": float(np.median(lines["phi"]))}
        for out, phi, tag in ((z, True, "z"), (z0, False, "z_without_phi")):
            mean, sd = design.predict(lines, T, with_phi=phi)
            zz = (mean[keep] - obs) / np.sqrt(sd[keep] ** 2 + 1.0 / nh[keep])
            out.extend(zz.tolist())
            fold[tag] = [round(x, 3) for x in zz.tolist()]
        fold["frac_abs_z_le_2"] = float((np.abs(fold["z"]) <= 2).mean())
        folds.append(fold)
        print(f"fold {T:g} K: {fold['evaluated']} of {fold['anchors']} anchors evaluated, |z| <= 2 for {fold['frac_abs_z_le_2']:.2f}, "
              f"median |z| {np.median(np.abs(fold['z'])):.2f}, median phi {fold['median_phi']:.2f}", flush=True)
    az, az0 = np.abs(z), np.abs(z0)
    res = {"n_z": len(z), "frac_abs_z_le_2": float((az <= 2).mean()), "median_abs_z": float(np.median(az)),
           "without_phi": {"frac_abs_z_le_2": float((az0 <= 2).mean()), "median_abs_z": float(np.median(az0))}, "folds": folds}
    res["H5"] = bool(res["frac_abs_z_le_2"] >= 0.85 and 0.5 <= res["median_abs_z"] <= 1.0)
    out = ROOT / "results" / "v6" / "v6.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"V6: {len(z)} z values; |z| <= 2 for {res['frac_abs_z_le_2']:.3f}, median |z| {res['median_abs_z']:.3f} "
          f"(without phi: {res['without_phi']['frac_abs_z_le_2']:.3f}, {res['without_phi']['median_abs_z']:.3f})")
    print(f"  H5 >= 85 % of |z| <= 2 and 0.5 <= median |z| <= 1.0: {'PASS' if res['H5'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
