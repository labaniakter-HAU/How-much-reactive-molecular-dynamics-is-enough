# file: src/foresight.py
"""Round 5 (design section 16): the revised foresight flag for C6.

The frozen flag (round 2) scored a campaign by C4's predictive sd. It had no curvature term with two temperatures,
a noisy one with three, and none at all with one (every sd infinite, so it always fired). Round 3's design objective
already predicts the error of a straight line through a campaign; the flag now uses it:

  per anchor   >= 2 temperatures: phi x random-effects line variance + gamma^2 x (chord bias)^2
               1 temperature:     1 / information + B^2 x (distance to the target)^2   (a flat line; B: slope prior)
               never exposed:     CLIP^2 (the metric's clip)
               plus noise:        1 / predicted events at the target (the Poisson noise of the counted rate the
                                  prediction is checked against)
  campaign     P = flux-weighted RMS over anchors; flagged if P > ln 2 (the label's own threshold) or the target starts
               above 3 x the campaign's highest amount of a species (round 2's domain rule, unchanged)
phi is the lack-of-fit factor of the random-effects line (design.fit_lines_re: max(1, chi^2 / (m - 2)), m >= 3
temperatures), the term that sees rates drifting within runs. gamma^2 comes from the campaign (design.curvature2) or
the prior design.GAMMA_PRIOR; s2 from the campaign (species.seed_s2_k, equal to design.seed_s2 for two seeds) or the
prior design.S2_PRIOR.
"""
import numpy as np

import design
import species

CUT, CLIP2, DOM = float(np.log(2)), 100.0, 3.0


def predicted_error(camp, data, T_of, A, T_star, flux, B, phi=True, noise=True):
    """-> P, gamma^2, s2 for one campaign {run: length}; flux (len(A),): events per anchor that the campaign's own
    model predicts at the target (non-finite or non-positive entries carry no weight)."""
    Ts, n, E = design.pool(camp, data, T_of)
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    s2 = species.seed_s2_k(camp, data, T_of, A)
    s2 = design.S2_PRIOR if s2 is None else s2
    g2 = design.curvature2(camp, data, T_of, A, T_star, s2)
    g2 = design.GAMMA_PRIOR ** 2 if g2 is None else g2
    n, E = n[:, A], E[:, A]
    use = E > 0
    w = np.where(use, 1.0 / (1.0 / np.where(n > 0, n, 0.5) + s2 / runs[:, None]), 0.0)
    x = (1000.0 / Ts - 1000.0 / T_star)[:, None]
    S0 = w.sum(0)
    S0s = np.where(S0 > 0, S0, 1.0)
    m1, m2, m3 = ((w * x ** p).sum(0) / S0s for p in (1, 2, 3))
    var_x = m2 - m1 ** 2
    two = (use.sum(0) >= 2) & (var_x > 1e-12)
    vx = np.where(two, var_x, 1.0)
    linevar = np.where(two, 1.0 / S0s + m1 ** 2 / (S0s * vx), 1.0 / S0s)
    if phi:
        f_phi = design.fit_lines_re(n, E, Ts, runs, s2)["phi"]
        linevar = linevar * np.where(np.isfinite(f_phi), f_phi, 1.0)
    chord2 = np.where(two, (m2 - m1 * (m3 - m1 * m2) / vx) ** 2, 0.0)
    total = linevar + g2 * chord2 + np.where(two, 0.0, B ** 2 * m2)
    total = np.where(S0 > 0, total, CLIP2)
    f = np.asarray(flux, float)
    f = np.where(np.isfinite(f) & (f > 0), f, 0.0)
    if noise:
        total = total + 1.0 / np.maximum(f, 1.0)
    return (float(np.sqrt((f * total).sum() / f.sum())) if f.sum() > 0 else np.inf), float(g2), float(s2)


def flagged(P, dom=1.0):
    return bool(not P <= CUT or dom > DOM)
