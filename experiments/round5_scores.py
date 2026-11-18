# file: experiments/round5_scores.py
"""Round 5, C6 development (design section 16): foresight scores computed offline from the campaign caches of
experiments/round5_cache_c6.py (HFO, butane, synthetic development seeds; never MEA). Every variant is scored with the
label and the bar of H7' (untrustworthy if RMS > ln 2; >= 90 % correct, >= 80 % per class). The cut is always ln 2.

  python -I -X utf8 experiments/round5_scores.py
"""
import glob
import itertools
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import design  # noqa: E402

L2, CLIP2 = float(np.log(2)), 100.0
# The .pkl caches are written and read only by this project's own round-5 scripts (local development data), never
# downloaded or shared, so pickle is safe here.
OUT = ROOT / "results" / "round5"


def s2_k(c, min_events=10):
    """species.seed_s2_k on the cached per-run counts."""
    q = []
    for T in np.unique(c["run_T"]):
        m = c["run_T"] == T
        if m.sum() < 2:
            continue
        n, E = c["run_n"][m], c["run_E"][m]
        ok = ((n >= min_events) & (E > 0)).all(0)
        x = np.log(n[:, ok] / E[:, ok])
        q.extend((x.var(0, ddof=1) - (1.0 / n[:, ok]).mean(0)).tolist())
    return max(0.0, float(np.mean(q))) if q else None


def curv2(c, s2, min_events=10):
    """design.curvature2 on the cached pooled counts."""
    Ts, n, E, runs = c["Ts"], c["n"], c["E"], c["runs"]
    if len(Ts) < 3:
        return None
    x = 1000.0 / Ts - 1000.0 / c["T"]
    X = np.stack([np.ones_like(x), x, x * x], 1)
    num = den = 0.0
    for j in range(n.shape[1]):
        ok = (n[:, j] >= min_events) & (E[:, j] > 0)
        if ok.sum() < 3:
            continue
        y = np.log(n[ok, j] / E[ok, j])
        W = 1.0 / (1.0 / n[ok, j] + s2 / runs[ok])
        M = X[ok].T @ (W[:, None] * X[ok])
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
        cj = (Minv @ (X[ok].T @ (W * y)))[2]
        num, den = num + n[ok, j].sum() * (cj * cj - Minv[2, 2]), den + n[ok, j].sum()
    return max(0.0, num / den) if den > 0 else None


def components(c, B=10.0, g_prior=design.GAMMA_PRIOR, s2_prior=design.S2_PRIOR):
    """Per-anchor pieces of the predicted error at T*: line variance, chord bias^2 (unit curvature), lack-of-fit phi,
    flat-line term for one temperature, and the discrepancy tau^2 (u* - u_bar)^2 of C4 (leave one temperature out)."""
    s2 = s2_k(c)
    s2 = s2_prior if s2 is None else s2
    g2 = curv2(c, s2)
    g2_used = g_prior ** 2 if g2 is None else g2
    Ts, n, E, runs = c["Ts"], c["n"], c["E"], c["runs"]
    use = E > 0
    y = np.where(n > 0, n, 0.5)
    w = np.where(use, 1.0 / (1.0 / y + s2 / runs[:, None]), 0.0)
    x = (1000.0 / Ts - 1000.0 / c["T"])[:, None]
    S0 = w.sum(0)
    S0s = np.where(S0 > 0, S0, 1.0)
    m1, m2, m3 = ((w * x ** p).sum(0) / S0s for p in (1, 2, 3))
    var_x = m2 - m1 ** 2
    two = (use.sum(0) >= 2) & (var_x > 1e-12)
    vx = np.where(two, var_x, 1.0)
    linevar = np.where(two, 1.0 / S0s + m1 ** 2 / (S0s * vx), 1.0 / S0s)
    chord2 = np.where(two, (m2 - m1 * (m3 - m1 * m2) / vx) ** 2, 0.0)
    flat2 = np.where(two, 0.0, m2)  # one temperature: (distance)^2, times a slope prior B^2
    lines = design.fit_lines_re(n, E, Ts, runs, s2)
    phi = np.where(np.isfinite(lines["phi"]), lines["phi"], 1.0)
    # per-reaction curvature: c_j^2 - var c_j from the reaction's own quadratic (>= 3 temperatures with >= 10 events),
    # NaN where not estimable
    c2j = np.full(n.shape[1], np.nan)
    if len(Ts) >= 3:
        X = np.stack([np.ones(len(Ts)), x[:, 0], x[:, 0] ** 2], 1)
        for j in range(n.shape[1]):
            ok = (n[:, j] >= 10) & (E[:, j] > 0)
            if ok.sum() < 3:
                continue
            W = 1.0 / (1.0 / n[ok, j] + s2 / runs[ok])
            M = X[ok].T @ (W[:, None] * X[ok])
            try:
                Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                continue
            cj = (Minv @ (X[ok].T @ (W * np.log(n[ok, j] / E[ok, j]))))[2]
            c2j[j] = max(0.0, cj * cj - Minv[2, 2])
    return {"s2": s2, "g2": g2, "g2_used": g2_used, "linevar": linevar, "chord2": chord2, "flat2": flat2, "phi": phi,
            "exposed": S0 > 0, "two": two, "B2": B * B, "c2j": c2j}


def score(c, comp, v):
    """Variant v (dict) -> P."""
    total = comp["linevar"] * (comp["phi"] if v.get("phi") else 1.0)
    g2 = comp["g2_used"]
    if v.get("g2_floor") is not None and comp["g2"] is not None:
        g2 = max(g2, v["g2_floor"] ** 2)
    g2j = np.full(len(total), g2)
    if v.get("per_rx"):
        g2j = np.where(np.isfinite(comp["c2j"]), comp["c2j"], g2)
    total = total + g2j * comp["chord2"] + comp["B2"] * comp["flat2"]
    if v.get("tau_max") and "sd_c4" in c:
        total = np.maximum(total, np.where(np.isfinite(c["sd_c4"]), c["sd_c4"] ** 2, total))
    total = np.where(comp["exposed"], total, CLIP2)
    f = np.where(np.isfinite(c["flux"]) & (c["flux"] > 0), c["flux"], 0.0)
    if v.get("noise"):
        total = total + 1.0 / np.maximum(f, 1.0)
    return float(np.sqrt((f * total).sum() / f.sum())) if f.sum() > 0 else np.inf


def h7(bad, fl):
    bad, fl = np.asarray(bad, bool), np.asarray(fl, bool)
    cau = fl[bad].mean() if bad.any() else float("nan")
    pas = (~fl[~bad]).mean()
    acc = (bad == fl).mean()
    ok = acc >= 0.9 and (cau >= 0.8 if bad.sum() >= 5 else True) and pas >= 0.8
    return acc, cau, pas, int(bad.sum()), ok


def main():
    sets = {"HFO": pickle.load(open(OUT / "cache_c6_hfo.pkl", "rb"))}
    if (OUT / "cache_c6_butane.pkl").exists():
        sets["butane"] = pickle.load(open(OUT / "cache_c6_butane.pkl", "rb"))
    syn = [r for p in sorted(glob.glob(str(OUT / "cache_c6_syn_*.pkl"))) for r in pickle.load(open(p, "rb"))]
    if syn:
        sets["synthetic"] = syn
    comps = {k: [components(c) for c in rows] for k, rows in sets.items()}
    variants = [dict(zip(("phi", "noise", "tau_max", "g2_floor"), vals))
                for vals in itertools.product((False, True), (False, True), (False, True), (None, 35.0, 70.0))]
    table = []
    for v in variants:
        line, oks = [], []
        for k, rows in sets.items():
            P = [score(c, comp, v) for c, comp in zip(rows, comps[k])]
            acc, cau, pas, nb, ok = h7([c["rms"] > L2 for c in rows], [p > L2 for p in P])
            line.append(f"{k} {acc:.3f}/{cau:.2f}/{pas:.3f}")
            oks.append(ok)
        table.append((sum(oks), v, line))
    for n_ok, v, line in sorted(table, key=lambda t: -t[0]):
        print(f"meets bar on {n_ok}/{len(sets)}  {v}:  " + "  ".join(line))
    # the frozen flag (part (a), sd of C4) for reference
    for k, rows in sets.items():
        Pc4 = []
        for c in rows:
            f = np.where(np.isfinite(c["flux"]), c["flux"], 0.0)
            sd = c["sd_c4"]
            Pc4.append(np.sqrt((f * sd ** 2).sum() / f.sum()) if np.isfinite(sd).all() and f.sum() > 0 else np.inf)
        acc, cau, pas, nb, ok = h7([c["rms"] > L2 for c in rows], [p > 0.6 for p in Pc4])
        print(f"frozen score (C4 sd > 0.6, no single-T part) on {k}: {acc:.3f}/{cau:.2f}/{pas:.3f} (untrustworthy {nb})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
