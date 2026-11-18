# file: experiments/round6_scores_c6.py
"""Round 6, C6 development (design section 17): flag rules computed offline from the round-5 campaign caches
(HFO, butane, synthetic development seeds; never MEA). Per anchor the predicted error variance is round 5's
(phi x line variance + gamma^2 chord^2, flat-line term with one temperature, + 1/flux). Two rules:
  mean     flag if the flux-weighted RMS of the predicted sd > ln 2 (round 5's flag)
  median   flag if P(realized RMS > ln 2) > 1/2, the realized RMS^2 drawn as sum_j f_j var_j z_j^2 / sum_j f_j, z ~ N(0, 1)
           (more likely than not to fail; differs from 'mean' when a few reactions carry the flux)

  python -I -X utf8 experiments/round6_scores_c6.py
"""
import glob
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import round5_scores as rs  # noqa: E402

L2 = float(np.log(2))
OUT = ROOT / "results" / "round5"
# The .pkl caches are written and read only by this project's own round-5 scripts (local development data), never
# downloaded or shared, so pickle is safe here.


def per_anchor(c, comp, phi=True, noise=True):
    var = comp["linevar"] * (comp["phi"] if phi else 1.0) + comp["g2_used"] * comp["chord2"] + comp["B2"] * comp["flat2"]
    var = np.where(comp["exposed"], var, rs.CLIP2)
    f = np.where(np.isfinite(c["flux"]) & (c["flux"] > 0), c["flux"], 0.0)
    if noise:
        var = var + 1.0 / np.maximum(f, 1.0)
    return var, f


def rules(c, comp, rng, draws=400, **kw):
    var, f = per_anchor(c, comp, **kw)
    if f.sum() <= 0:
        return np.inf, 1.0
    P = float(np.sqrt((f * var).sum() / f.sum()))
    z2 = rng.standard_normal((draws, len(var))) ** 2
    rms = np.sqrt((z2 * (f * var)).sum(1) / f.sum())
    return P, float((rms > L2).mean())


def h7(bad, fl):
    bad, fl = np.asarray(bad, bool), np.asarray(fl, bool)
    return (bad == fl).mean(), (fl[bad].mean() if bad.any() else np.nan), (~fl[~bad]).mean(), int(bad.sum())


def main():
    syn = [r for p in sorted(glob.glob(str(OUT / "cache_c6_syn_*.pkl"))) for r in pickle.load(open(p, "rb"))]
    sets = {"HFO": pickle.load(open(OUT / "cache_c6_hfo.pkl", "rb")), "butane": pickle.load(open(OUT / "cache_c6_butane.pkl", "rb")),
            "syn lumped": [r for r in syn if r["kind"] == "lumped"], "syn standard": [r for r in syn if r["kind"] != "lumped"]}
    rng = np.random.default_rng(0)
    for kw in ({"phi": True, "noise": True}, {"phi": True, "noise": False}, {"phi": False, "noise": True}):
        print(kw)
        for k, rows in sets.items():
            comps = [rs.components(c) for c in rows]
            res = [rules(c, comp, rng, **kw) for c, comp in zip(rows, comps)]
            P, p = np.array([x[0] for x in res]), np.array([x[1] for x in res])
            bad = [c["rms"] > L2 for c in rows]
            a1, a2 = h7(bad, P > L2), h7(bad, p > 0.5)
            print(f"   {k:>12}: mean rule {a1[0]:.3f}/{a1[1]:.2f}/{a1[2]:.3f}   median rule {a2[0]:.3f}/{a2[1]:.2f}/{a2[2]:.3f}   (untrustworthy {a1[3]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
