# file: src/firth.py
"""A random-effects Arrhenius line that zero counts do not bend (design sections 18-19).

fit_lines_firth (round 8): quasi-likelihood line, mean E exp(a - b u), variance mu + mu^2 s2/runs (negative-binomial
form), with Firth's penalty 1/2 log det of the information. For this variance function and the log link, the
penalized score is the quasi-likelihood score with each count n raised by h/2 (h the temperature's leverage, summing
to 2), so each step is a reweighted least-squares fit to the adjusted counts; steps are halved until the penalized
objective does not fall. Started from design.fit_lines; a reaction whose iteration fails keeps that start.
use_firth(): within the block, design.fit_lines_re (C4's line, in which a zero count enters as an observation
ln(0.5/E)) is fit_lines_firth. Every caller reaches the line through the design namespace.

Round 7's undamped prototype and its flag term are kept below, unchanged, so that section 18's numbers reproduce.
"""
from contextlib import contextmanager

import numpy as np

import design
import foresight

_LINE_V1 = design.fit_lines_re


def fit_lines_firth(n, E, T, runs, s2, iters=100, tol=1e-10):
    """n, E (m, J) pooled per temperature T (m,), runs (m,) -> dict as design.fit_lines_re (cov with phi applied)."""
    n, E = np.asarray(n, float), np.asarray(E, float)
    u = np.broadcast_to(1000.0 / np.asarray(T, float)[:, None], E.shape)
    use = E > 0
    y = n * use
    Es = np.where(use, E, 1.0)
    rv = np.broadcast_to(s2 / np.asarray(runs, float)[:, None], E.shape)
    m = use.sum(0)
    two, live = m >= 2, m > 0
    start = design.fit_lines(n, E, T)

    def state(a, b):
        eta = np.clip(a - b * u, -700.0, 700.0)
        mu = np.where(use, Es * np.exp(eta), 1.0)
        W = np.where(use, mu / (1.0 + rv * mu), 0.0)
        S0 = W.sum(0)
        S0s = np.where(S0 > 0, S0, 1.0)
        ub = (W * u).sum(0) / S0s
        S2c = (W * (u - ub) ** 2).sum(0)
        ok = two & (S2c > 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ell = np.where(use & (y > 0), y * np.log(np.where(use, mu / (1.0 + rv * mu), 1.0)), 0.0) \
                - np.where(use, np.where(rv > 0, np.log1p(rv * mu) / np.where(rv > 0, rv, 1.0), mu), 0.0)
            L = ell.sum(0) + 0.5 * (np.log(S0) + np.where(ok, np.log(np.where(ok, S2c, 1.0)), 0.0))
        return eta, mu, W, S0s, ub, S2c, ok, np.where(np.isfinite(L), L, -np.inf)

    a, b = start["a"].copy(), start["b"].copy()
    eta, mu, W, S0s, ub, S2c, ok, L = state(a, b)
    for _ in range(iters):
        S2cs = np.where(ok, S2c, 1.0)
        h = W * (1.0 / S0s + np.where(ok, (u - ub) ** 2 / S2cs, 0.0))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            z = np.where(use, eta + (y + 0.5 * h - mu) / mu, 0.0)
            b_new = np.where(ok, -(W * (u - ub) * z).sum(0) / S2cs, 0.0)
            a_new = (W * z).sum(0) / S0s + b_new * ub
        da = np.where(live & np.isfinite(a_new), a_new - a, 0.0)
        db = np.where(live & np.isfinite(b_new), b_new - b, 0.0)
        t = np.ones_like(a)
        for _ in range(60):
            Lt = state(a + t * da, b + t * db)[-1]
            worse = live & ~(Lt >= L - 1e-12 * np.abs(L))
            if not worse.any():
                break
            t = np.where(worse, 0.5 * t, t)
        t = np.where(worse, 0.0, t)
        a, b = a + t * da, b + t * db
        eta, mu, W, S0s, ub, S2c, ok, L = state(a, b)
        if np.max(np.abs(np.concatenate([t * da, t * db])), initial=0.0) < tol:
            break
    bad = live & ~(np.isfinite(a) & np.isfinite(b) & np.isfinite(L))
    a, b = np.where(bad, start["a"], a), np.where(bad, start["b"], b)
    a = np.where(live, a, design.ABSENT)
    eta, mu, W, S0s, ub, S2c, ok, L = state(a, b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        chi2 = np.where(use, (y - mu) ** 2 / (mu + rv * mu ** 2), 0.0).sum(0)
        phi = np.where(m >= 3, np.maximum(1.0, chi2 / np.maximum(m - 2, 1)), 1.0)
        cov = np.zeros((len(m), 2, 2))
        cov[:, 0, 0] = np.where(live, 1.0 / S0s, np.inf) + np.where(ok, ub ** 2 / np.where(ok, S2c, 1.0), 0.0)
        cov[:, 0, 1] = cov[:, 1, 0] = np.where(ok, ub / np.where(ok, S2c, 1.0), 0.0)
        cov[:, 1, 1] = np.where(ok, 1.0 / np.where(ok, S2c, 1.0), np.inf)
        cov *= phi[:, None, None]
    return {"a": a, "b": b, "cov": cov, "phi": phi, "seen": ((n > 0) & use).sum(0), "m": m, "ubar": ub}


@contextmanager
def use_firth():
    """Round 8 (design section 19): within the block, C4's line design.fit_lines_re is fit_lines_firth."""
    design.fit_lines_re = fit_lines_firth
    try:
        yield
    finally:
        design.fit_lines_re = _LINE_V1


# ---- Round 7 (design section 18), kept unchanged for reproducibility: rejected at development ----

def fit_lines_firth_r7(n, E, T, runs, s2, iters=50):
    """Round 7's undamped prototype of the Firth line (same model, fixed 50 reweighting steps, no step control)."""
    n, E = np.asarray(n, float), np.asarray(E, float)
    u = np.broadcast_to(1000.0 / np.asarray(T, float)[:, None], E.shape)
    use = E > 0
    y = n * use
    Es = np.where(use, E, 1.0)
    rv = s2 / np.asarray(runs, float)[:, None]
    m = use.sum(0)
    two = m >= 2
    start = design.fit_lines(n, E, T)

    def weights(a, b):
        eta = a - b * u
        mu = np.where(use, Es * np.exp(np.clip(eta, -700, 700)), 1.0)
        w = np.where(use, 1.0 / (1.0 / np.maximum(mu, 1e-300) + rv), 0.0)
        S0 = w.sum(0)
        ub = (w * u).sum(0) / np.where(S0 > 0, S0, 1.0)
        S2c = (w * (u - ub) ** 2).sum(0)
        return eta, mu, w, S0, ub, S2c, two & (S2c > 0)

    a, b = start["a"].copy(), start["b"].copy()
    for _ in range(iters):
        eta, mu, w, S0, ub, S2c, ok = weights(a, b)
        h = w * (1.0 / np.where(S0 > 0, S0, 1.0) + np.where(ok, (u - ub) ** 2 / np.where(ok, S2c, 1.0), 0.0))
        z = eta + (y + 0.5 * h * use - mu) / mu
        b = np.where(ok, -(w * (u - ub) * z).sum(0) / np.where(ok, S2c, 1.0), 0.0)
        a = np.where(m > 0, (w * z).sum(0) / np.where(S0 > 0, S0, 1.0) + b * ub, design.ABSENT)
    bad = ~(np.isfinite(a) & np.isfinite(b))
    a, b = np.where(bad, start["a"], a), np.where(bad, start["b"], b)
    eta, mu, w, S0, ub, S2c, ok = weights(a, b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        chi2 = np.where(use, (y - mu) ** 2 / (mu + mu ** 2 * rv), 0.0).sum(0)
        phi = np.where(m >= 3, np.maximum(1.0, chi2 / np.maximum(m - 2, 1)), 1.0)
        cov = np.zeros((len(m), 2, 2))
        cov[:, 0, 0] = np.where(m > 0, 1.0 / S0, np.inf) + np.where(ok, ub ** 2 / S2c, 0.0)
        cov[:, 0, 1] = cov[:, 1, 0] = np.where(ok, ub / S2c, 0.0)
        cov[:, 1, 1] = np.where(ok, 1.0 / S2c, np.inf)
        cov *= phi[:, None, None]
    return {"a": a, "b": b, "cov": cov, "phi": phi, "seen": ((n > 0) & use).sum(0), "m": m, "ubar": ub}


def defect2_arrays(n, E, Ts, runs, s2, T_star):
    """Round 7: (ln k* of design.fit_lines_re - ln k* of the round-7 Firth line)^2 per reaction, capped at CLIP2."""
    old = design.predict(_LINE_V1(n, E, Ts, runs, s2), T_star)[0]
    new = design.predict(fit_lines_firth_r7(n, E, Ts, runs, s2), T_star)[0]
    d2 = (old - new) ** 2
    return np.where(np.isfinite(d2), np.minimum(d2, foresight.CLIP2), foresight.CLIP2)


def predicted_error7(camp, data, T_of, A, T_star, flux, B, phi=True, noise=True):
    """Round 7: -> (P7, P5): round 5's P (foresight.predicted_error) with the flux-weighted squared defect added, and P."""
    P, _, s2 = foresight.predicted_error(camp, data, T_of, A, T_star, flux, B, phi, noise)
    f = np.asarray(flux, float)
    f = np.where(np.isfinite(f) & (f > 0), f, 0.0)
    if f.sum() <= 0:
        return P, P
    Ts, n, E = design.pool(camp, data, T_of)
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    d2 = defect2_arrays(n[:, A], E[:, A], Ts, runs, s2, T_star)
    return float(np.sqrt(P ** 2 + (f * d2).sum() / f.sum())), P
