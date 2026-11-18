# file: experiments/round7_rca_c6c.py
"""Round 7, C6 root cause test (design section 18), SPENT round-5 seeds only: is the missed-failure error made by the
random-effects line's log-space fit (design.fit_lines_re: zero counts enter as ln(0.5/E) observations, weights from
observed counts)? Refits each missed campaign's lines three ways and reports the campaign rms (label) of each:
  wls  design.fit_lines_re, as used by C4/C5/C6 now
  ml   design.fit_lines, Poisson maximum likelihood (zeros as 1/2 inside the likelihood)
  pql  quasi-likelihood random-effects line: mean exp(a - b u) E, variance mu + mu^2 s2/runs (IRLS from the ml fit)
Prototype only; nothing in src/ is changed (design.py is inside round 6's frozen hash).

  python -I -X utf8 experiments/round7_rca_c6c.py --workers 2
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import design  # noqa: E402


def fit_lines_pql(n, E, T, runs, s2, iters=50, firth=False):
    """Random-effects Arrhenius line by quasi-likelihood (IRLS working weights 1/(1/mu + s2/runs)), started from the
    Poisson ML line; with s2 = 0 it is that line. Zero counts: 1/2 each (Jeffreys, as fit_lines), or with firth=True
    the observed n plus h/2, h the temperature's leverage (Firth 1993; sum of h = 2). Same output keys as
    design.fit_lines_re (cov with phi applied)."""
    n, E = np.asarray(n, float), np.asarray(E, float)
    u = np.broadcast_to(1000.0 / np.asarray(T, float)[:, None], E.shape)
    use = E > 0
    y = (n if firth else np.where(n > 0, n, 0.5)) * use
    Es = np.where(use, E, 1.0)
    rv = s2 / np.asarray(runs, float)[:, None]
    m = use.sum(0)
    two = m >= 2
    ml = design.fit_lines(n, E, T)
    a, b = ml["a"].copy(), ml["b"].copy()
    for _ in range(iters):
        eta = a - b * u
        mu = np.where(use, Es * np.exp(np.clip(eta, -700, 700)), 1.0)
        w = np.where(use, 1.0 / (1.0 / np.maximum(mu, 1e-300) + rv), 0.0)
        S0 = w.sum(0)
        ub = (w * u).sum(0) / np.where(S0 > 0, S0, 1.0)
        S2c = (w * (u - ub) ** 2).sum(0)
        ok = two & (S2c > 0)
        h = w * (1.0 / np.where(S0 > 0, S0, 1.0) + np.where(ok, (u - ub) ** 2 / np.where(ok, S2c, 1.0), 0.0))
        z = eta + ((y + 0.5 * h * use if firth else y) - mu) / mu
        b = np.where(ok, -(w * (u - ub) * z).sum(0) / np.where(ok, S2c, 1.0), 0.0)
        a = np.where(m > 0, (w * z).sum(0) / np.where(S0 > 0, S0, 1.0) + b * ub, design.ABSENT)
    eta = a - b * u
    mu = np.where(use, Es * np.exp(np.clip(eta, -700, 700)), 1.0)
    w = np.where(use, 1.0 / (1.0 / np.maximum(mu, 1e-300) + rv), 0.0)
    S0 = w.sum(0)
    ub = (w * u).sum(0) / np.where(S0 > 0, S0, 1.0)
    S2c = (w * (u - ub) ** 2).sum(0)
    ok = two & (S2c > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        chi2 = np.where(use, (y - mu) ** 2 / (mu + mu ** 2 * rv), 0.0).sum(0)
        phi = np.where(m >= 3, np.maximum(1.0, chi2 / np.maximum(m - 2, 1)), 1.0)
        cov = np.zeros((len(m), 2, 2))
        cov[:, 0, 0] = np.where(m > 0, 1.0 / S0, np.inf) + np.where(ok, ub ** 2 / S2c, 0.0)
        cov[:, 0, 1] = cov[:, 1, 0] = np.where(ok, ub / S2c, 0.0)
        cov[:, 1, 1] = np.where(ok, 1.0 / S2c, np.inf)
        cov *= phi[:, None, None]
    return {"a": a, "b": b, "cov": cov, "phi": phi, "seen": ((n > 0) & use).sum(0), "m": m, "ubar": ub}


def one(args):
    from round2_synthetic import CLIP
    from round7_rca_c6b import rebuild
    seed, T_STAR, idx, stored_rms = args
    camp, data, T_of, A, flux, truth, w, err, sd_re, lines = rebuild(seed, T_STAR, idx)
    Ts, n, E = design.pool(camp, data, T_of)
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    s2 = design.seed_s2(camp, data, T_of, A)
    s2 = 0.0 if s2 is None else s2  # calibrated_re's default
    fits = {"wls": design.fit_lines_re(n[:, A], E[:, A], Ts, runs, s2), "ml": design.fit_lines(n[:, A], E[:, A], Ts),
            "pql": fit_lines_pql(n[:, A], E[:, A], Ts, runs, s2), "firth": fit_lines_pql(n[:, A], E[:, A], Ts, runs, s2, firth=True)}
    out = {"seed": seed, "T": T_STAR, "idx": idx, "stored_rms": stored_rms}
    for k, li in fits.items():
        mean = design.predict(li, T_STAR)[0]
        out[k] = float(np.sqrt((w * np.minimum(np.abs(mean - truth), CLIP) ** 2).sum() / w.sum()))
    j = int(np.argmax(w * err ** 2))
    zeros = int(((n[:, A][:, j] == 0) & (E[:, A][:, j] > 0)).sum())
    out["culprit"] = {"j": j, "temps_exposed": int((E[:, A][:, j] > 0).sum()), "temps_zero": zeros,
                      **{k: float(design.predict(li, T_STAR)[0][j] - truth[j]) for k, li in fits.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    res = json.loads((ROOT / "results" / "round7" / "rca_c6b.json").read_text())
    with Pool(a.workers) as pool:
        out = pool.map(one, [(r["seed"], r["T"], r["idx"], r["rms"]) for r in res])
    (ROOT / "results" / "round7" / "rca_c6c.json").write_text(json.dumps(out, indent=1))
    L2 = float(np.log(2))
    for o in out:
        c = o["culprit"]
        print(f"seed {o['seed']} T {o['T']:g} camp {o['idx']}: rms/ln2 wls {o['wls'] / L2:.2f} (stored {o['stored_rms'] / L2:.2f}), "
              f"ml {o['ml'] / L2:.2f}, pql {o['pql'] / L2:.2f}, firth {o['firth'] / L2:.2f} | culprit rx {c['j']}: zero-count temps {c['temps_zero']} of {c['temps_exposed']}, "
              f"error wls {c['wls']:+.2f} ml {c['ml']:+.2f} pql {c['pql']:+.2f} firth {c['firth']:+.2f}")
    for k in ("wls", "ml", "pql", "firth"):
        print(f"{k}: still failing {sum(o[k] > L2 for o in out)} of {len(out)}; median rms/ln2 {np.median([o[k] for o in out]) / L2:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
