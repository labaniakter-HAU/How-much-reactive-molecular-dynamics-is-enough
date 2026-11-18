# file: src/design.py
"""Part II (design section 11): rate uncertainty at a new temperature, campaign design and the foresight
flag, all from counted events and exposures. The Part I counted-rate model is unchanged.

Per reaction, events n and exposures E (V x integral of prod c^o dt, as in fit.start_values) are pooled per
temperature. ln k = a - b u with u = 1000 / T is fitted by Poisson regression with offset ln E (Newton). A
temperature with exposure but no events enters as n = 1/2 (Jeffreys posterior mean); one without exposure
carries no information. With m >= 3 temperatures the covariance is multiplied by
phi = max(1, Pearson chi2 / (m - 2)). Run lengths are in 1.2 ps bins. Fixed 2026-11-05, before any Part II run.
"""
import numpy as np

ABSENT = -30.0  # ln k of a reaction never exposed (fit.start_values' convention)


def fit_lines(n, E, T, iters=100):
    """n, E (m, J) pooled per temperature T (m,) -> dict a, b, cov (J, 2, 2) with phi applied, phi, seen (J,):
    temperatures with events. One exposed temperature: b = 0 (Part I's rule) and var(b) infinite."""
    n, E = np.asarray(n, float), np.asarray(E, float)
    u = np.broadcast_to(1000.0 / np.asarray(T, float)[:, None], E.shape)
    use = E > 0
    y = np.where(n > 0, n, 0.5) * use
    Es = np.where(use, E, 1.0)
    m = use.sum(0)
    two = m >= 2
    def centred(w):  # sum, weighted mean of u, centred second moment: no cancellation when one T dominates
        s0 = w.sum(0)
        ub = (w * u).sum(0) / np.where(s0 > 0, s0, 1.0)
        return s0, ub, (w * (u - ub) ** 2).sum(0)

    def loglik(a, b):  # Poisson log-likelihood up to a constant; unclipped, so overflow reads as -inf
        eta = a - b * u
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(use, y * eta - Es * np.exp(eta), 0.0).sum(0)
    s0, ub, s2c = centred(y)
    lnk = np.log(np.where(use, y, 1.0) / Es) * use
    ok = two & (s2c > 0)
    b = np.where(ok, -(y * (u - ub) * lnk).sum(0) / np.where(ok, s2c, 1.0), 0.0)  # weighted least squares start
    a = np.where(m > 0, (y * lnk).sum(0) / np.where(m > 0, s0, 1.0) + b * ub, ABSENT)
    flat = np.where(m > 0, np.log(np.where(m > 0, s0, 1.0) / (Es * use).sum(0).clip(1e-300)), ABSENT)  # b = 0: cannot overflow
    ll = loglik(a, b)
    lf = loglik(flat, 0.0)
    a, b, ll = np.where(ll >= lf, a, flat), np.where(ll >= lf, b, 0.0), np.maximum(ll, lf)  # else every step beats -inf
    for _ in range(iters):  # Newton in centred coordinates, where the information matrix is diagonal
        mu = np.where(use, Es * np.exp(np.clip(a - b * u, -700, 700)), 0.0)
        S0, ub, S2c = centred(mu)
        ok = two & (S2c > 0)
        db = np.where(ok, -((y - mu) * (u - ub)).sum(0) / np.where(ok, S2c, 1.0), 0.0)
        da = np.where(m > 0, (y - mu).sum(0) / np.where(S0 > 0, S0, 1.0), 0.0) + ub * db
        t = np.ones_like(a)
        for _ in range(60):  # damped: a full step can overshoot when exposures span many orders of magnitude
            worse = ~(loglik(a + t * da, b + t * db) >= ll - 1e-12 * np.abs(ll))
            if not worse.any():
                break
            t = np.where(worse, 0.5 * t, t)
        t = np.where(worse, 0.0, t)
        a, b = a + t * da, b + t * db
        ll = loglik(a, b)
    mu = np.where(use, Es * np.exp(np.clip(a - b * u, -700, 700)), 0.0)
    S0, ub, S2c = centred(mu)
    ok = two & (S2c > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        chi2 = np.where(use, (y - mu) ** 2 / np.where(use, mu, 1.0), 0.0).sum(0)
        phi = np.where(m >= 3, np.maximum(1.0, chi2 / np.maximum(m - 2, 1)), 1.0)
        cov = np.zeros((len(m), 2, 2))
        cov[:, 0, 0] = np.where(m > 0, 1.0 / S0, np.inf) + np.where(ok, ub ** 2 / S2c, 0.0)
        cov[:, 0, 1] = cov[:, 1, 0] = np.where(ok, ub / S2c, 0.0)
        cov[:, 1, 1] = np.where(ok, 1.0 / S2c, np.inf)
        cov *= phi[:, None, None]  # may reach inf: an unidentified rate
    return {"a": a, "b": b, "cov": cov, "phi": phi, "seen": ((n > 0) & use).sum(0), "m": m}


def predict(lines, T_star, with_phi=True):
    """-> ln k at T_star and its standard deviation (J,)."""
    us, c = 1000.0 / T_star, lines["cov"] / (1.0 if with_phi else lines["phi"][:, None, None])
    with np.errstate(invalid="ignore"):
        var = c[:, 0, 0] - 2 * us * c[:, 0, 1] + us ** 2 * c[:, 1, 1]
    return lines["a"] - lines["b"] * us, np.sqrt(var)


def c_value(campaign, T_of, T_star):
    """x*' M^-1 x* with M = sum over the campaign's runs of length x (1, -u)(1, -u)'. Expected counts per bin
    are the pilot's at every temperature (section 11), so every reaction's target variance is this value
    divided by its pilot count rate, and the design is the same for all reactions."""
    L = np.array([v for v in campaign.values() if v > 0], float)
    u = np.array([1000.0 / T_of[r] for r, v in campaign.items() if v > 0])
    S0, S1, S2 = ((L * u ** p).sum() for p in range(3))
    D, us = S0 * S2 - S1 ** 2, 1000.0 / T_star
    return (S2 - 2 * us * S1 + us ** 2 * S0) / D if D > 1e-9 * S0 * S2 else np.inf


def _moves(menu, camp, budget):
    """(run, length) steps that fit the budget; lengthening a chosen run costs the extra bins."""
    left = budget - sum(camp.values())
    return [(r, L) for r, Ls in menu.items() for L in Ls if camp.get(r, 0) < L <= camp.get(r, 0) + left]


def greedy(menu, start, budget, T_of, T_star):
    """Designed campaign: repeatedly take the step that lowers the target variance most per bin."""
    camp = dict(start)
    while moves := _moves(menu, camp, budget):
        q0 = c_value(camp, T_of, T_star)

        def key(mv):
            cost, q1 = mv[1] - camp.get(mv[0], 0), c_value({**camp, mv[0]: mv[1]}, T_of, T_star)
            gain = (np.inf if q1 < np.inf else 0.0) if q0 == np.inf else q0 - q1
            return (-gain / cost, cost, q1)
        r, L = min(moves, key=key)
        camp[r] = L
    return camp


def hot_only(menu, start, budget, T_of):
    """Hottest runs first, each at the longest length that fits."""
    camp = dict(start)
    for r in sorted(menu, key=lambda r: -T_of[r]):
        fits = [L for L in menu[r] if L <= budget - sum(camp.values()) + camp.get(r, 0)]
        if fits and max(fits) > camp.get(r, 0):
            camp[r] = max(fits)
    return camp


def uniform(menu, start, budget, T_of):
    """Equal shares per temperature as far as the menu allows: round robin from the coldest temperature, each
    turn taking that temperature's cheapest step."""
    camp, temps = dict(start), sorted({T_of[r] for r in menu})
    added = True
    while added:
        added = False
        for T in temps:
            steps = [(L - camp.get(r, 0), r, L) for r, L in _moves(menu, camp, budget) if T_of[r] == T]
            if steps:
                _, r, L = min(steps)
                camp[r], added = L, True
    return camp


def random_campaign(menu, start, budget, rng):
    """Uniformly random steps among those that fit, until none fits."""
    camp = dict(start)
    while moves := _moves(menu, camp, budget):
        r, L = moves[rng.integers(len(moves))]
        camp[r] = L
    return camp


def pool(campaign, data, T_of):
    """campaign {run: bins}, data[run][bins] = (events, exposure) per event type -> T (m,), n, E (m, types)."""
    temps = sorted({T_of[r] for r, L in campaign.items() if L > 0})
    part = lambda T, i: sum(data[r][L][i] for r, L in campaign.items() if L > 0 and T_of[r] == T)
    return np.array(temps), np.array([part(T, 0) for T in temps]), np.array([part(T, 1) for T in temps])


def truncations(g, O, lengths, per_bin=2):
    """One run's Group (B = 1) -> {bins: (events, exposure) per event type} after each length in bins."""
    rate = np.exp(np.log(np.clip(g.obs[:, 0].numpy(), 1e-12, None)) @ O) * float(g.V[0])
    t = g.t.numpy()
    cum = np.vstack([np.zeros(rate.shape[1]), np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t)[:, None], 0)])
    ev = np.cumsum(g.ev[0].numpy(), 0)
    return {L: (ev[L - 1], cum[per_bin * L]) for L in lengths}


def flags(lines, T_star, S, O, present, flux):
    """Foresight flag for one target -> (no class present at t = 0 has an identified consumption channel,
    more than half the model's predicted flux at T* runs through reactions seen at no more than one
    temperature). S, O: classes x reactions; present: classes present at t = 0 (bool); flux: events per
    reaction that the model predicts at T* (non-finite: the integration failed, which flags the target)."""
    sd = predict(lines, T_star)[1]
    channel = (S[present] < 0).any(0) & (O[~present] == 0).all(0) & (sd <= 1.0)
    f = np.asarray(flux, float)
    frac = f[lines["seen"] <= 1].sum() / f.sum() if np.isfinite(f).all() and f.sum() > 0 else np.inf
    return bool(not channel.any()), bool(not frac <= 0.5)


# Round 2 (design section 12): design variants. The functions above stay as pre-registered in round 1.

def nearest_first(menu, start, budget, T_of, T_star):
    """Runs nearest the target in 1000/T first, each at the longest length that fits."""
    camp = dict(start)
    for r in sorted(menu, key=lambda r: (abs(1000.0 / T_of[r] - 1000.0 / T_star), r)):
        fits = [L for L in menu[r] if L <= budget - sum(camp.values()) + camp.get(r, 0)]
        if fits and max(fits) > camp.get(r, 0):
            camp[r] = max(fits)
    return camp


def sequential(menu, start, budget, T_of, T_star, data, A, tau2=0.0, target_bins=500):
    """Adaptive design: each step's events are revealed before the next step is chosen. A candidate's expected
    counts per reaction: the chosen run's own count rate when it is lengthened; for a new run, the count rate of
    the nearest chosen run scaled by the current Arrhenius line. Objective: event-weighted mean over reactions of
    var(ln k*) + tau2 (u_bar - u*)^2, where u_bar is the information-weighted mean of 1000/T (tau2 > 0 penalises
    data far from the target, where lack of fit biases the line). Weights: predicted events at the target.
    tau2=None (round 3): re-estimated each step from the campaign's own data with calibrated_re's discrepancy
    term, TAU2_PRIOR while fewer than 3 temperatures are chosen."""
    us, camp, t2 = 1000.0 / T_star, dict(start), [tau2]

    def state(c):
        Ts, n, E = pool(c, data, T_of)
        n, E = n[:, A], E[:, A]
        y = np.where(n > 0, n, 0.5) * (E > 0)
        u = 1000.0 / Ts[:, None]
        rate = {r: np.maximum(data[r][L][0][A], 0.5) / L for r, L in c.items() if L > 0}  # counts per bin
        return fit_lines(n, E, Ts), (y.sum(0), (y * u).sum(0), (y * u * u).sum(0)), rate

    def objective(S, w):
        S0, S1, S2 = S
        ub = S1 / S0
        S2c = np.maximum(S2 - S0 * ub ** 2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(S2c > 1e-12 * S2, 1.0 / S0 + (ub - us) ** 2 / S2c, np.inf) + t2[0] * (ub - us) ** 2
        return float((w * v).sum() / w.sum()) if np.isfinite(v[w > 0]).all() else np.inf

    while moves := _moves(menu, camp, budget):
        lines, S, rate = state(camp)
        chosen = [r for r, L in camp.items() if L > 0]
        if tau2 is None:
            est = calibrated_re(camp, data, T_of, A, T_star, tau2_default=None)[3] if len({T_of[r] for r in chosen}) >= 3 else None
            t2[0] = TAU2_PRIOR if est is None else est
        near = lambda T: min(chosen, key=lambda r: abs(1000.0 / T_of[r] - 1000.0 / T))
        r0 = near(T_star)
        w = rate[r0] * np.exp(-lines["b"] * (us - 1000.0 / T_of[r0])) * target_bins
        q0 = objective(S, w)

        def key(mv):
            r, L = mv
            if camp.get(r, 0) > 0:
                mu = rate[r] * (L - camp[r])
            else:
                rn = near(T_of[r])
                mu = rate[rn] * np.exp(-lines["b"] * (1000.0 / T_of[r] - 1000.0 / T_of[rn])) * L
            u = 1000.0 / T_of[r]
            q1 = objective((S[0] + mu, S[1] + mu * u, S[2] + mu * u * u), w)
            cost = L - camp.get(r, 0)
            gain = (np.inf if q1 < np.inf else 0.0) if q0 == np.inf else q0 - q1
            return (-gain / cost, cost, q1)
        r, L = min(moves, key=key)
        camp[r] = L
    return camp


def seed_psi(camp, data, T_of, A, min_events=10):
    """Seed overdispersion: mean squared Poisson-standardised difference of ln k between the two seeds of each
    temperature where both are in the campaign (reactions with >= min_events in each); None if no pair."""
    d = []
    for T in sorted({T_of[r] for r, L in camp.items() if L > 0}):
        rs = [r for r, L in camp.items() if L > 0 and T_of[r] == T]
        if len(rs) != 2:
            continue
        (n1, e1), (n2, e2) = (data[r][camp[r]] for r in rs)
        n1, e1, n2, e2 = n1[A], e1[A], n2[A], e2[A]
        ok = (n1 >= min_events) & (n2 >= min_events) & (e1 > 0) & (e2 > 0)
        d.extend(((np.log(n1[ok] / e1[ok]) - np.log(n2[ok] / e2[ok])) / np.sqrt(1 / n1[ok] + 1 / n2[ok])).tolist())
    return max(1.0, float(np.mean(np.square(d)))) if d else None


def discrepancy(camp, data, T_of, A, psi, min_events=10):
    """tau^2 of the model-discrepancy term: leave each campaign temperature out, predict it from the others, and
    divide the residual variance beyond the predicted variance by the mean squared distance from the data
    (in 1000/T). None with fewer than 3 temperatures."""
    temps = sorted({T_of[r] for r, L in camp.items() if L > 0})
    if len(temps) < 3:
        return None
    r2, v, d2 = [], [], []
    for T in temps:
        inner = {r: L for r, L in camp.items() if L > 0 and T_of[r] != T}
        Ts, n, E = pool(inner, data, T_of)
        lines = fit_lines(n[:, A], E[:, A], Ts)
        _, nh, Eh = pool({r: L for r, L in camp.items() if L > 0 and T_of[r] == T}, data, T_of)
        nh, Eh = nh[0, A], Eh[0, A]
        keep = (nh >= min_events) & (Eh > 0)
        mean, sd = predict(lines, T)
        y = np.where(n[:, A] > 0, n[:, A], 0.5) * (E[:, A] > 0)
        ub = (y * (1000.0 / Ts[:, None])).sum(0) / y.sum(0)
        ok = keep & np.isfinite(sd)
        r2.extend(((mean - np.log(np.where(keep, nh, 1) / np.where(keep, Eh, 1))) ** 2)[ok].tolist())
        v.extend((psi * sd ** 2 + psi / np.maximum(nh, 1))[ok].tolist())
        d2.extend(((1000.0 / T - ub) ** 2)[ok].tolist())
    return max(0.0, float(np.mean(r2) - np.mean(v))) / max(float(np.mean(d2)), 1e-12) if r2 else None


# defaults when a campaign cannot estimate them: the development estimates on the V6 folds (round 2, 2026-11-05)
PSI_DEFAULT, TAU2_DEFAULT = 1.4, 5.4


def calibrated(camp, data, T_of, A, T_star, psi_default=None, tau2_default=None):
    """Round-2 predictive distribution of ln k at T_star: mean, sd with seed overdispersion (psi) and the
    model-discrepancy term tau^2 (u* - u_bar)^2, plus psi and tau^2 used."""
    Ts, n, E = pool(camp, data, T_of)
    lines = fit_lines(n[:, A], E[:, A], Ts)
    mean, sd = predict(lines, T_star)
    psi = seed_psi(camp, data, T_of, A)
    psi = psi if psi is not None else (psi_default if psi_default is not None else PSI_DEFAULT)
    tau2 = discrepancy(camp, data, T_of, A, psi)
    tau2 = tau2 if tau2 is not None else (tau2_default if tau2_default is not None else 0.0)
    y = np.where(n[:, A] > 0, n[:, A], 0.5) * (E[:, A] > 0)
    ub = (y * (1000.0 / Ts[:, None])).sum(0) / np.maximum(y.sum(0), 1e-300)
    with np.errstate(invalid="ignore"):
        sd2 = np.sqrt(psi * sd ** 2 + tau2 * (1000.0 / T_star - ub) ** 2)
    return mean, sd2, psi, tau2, lines


def seed_s2(camp, data, T_of, A, min_events=10):
    """Additive between-run variance of ln k (random effect): (d^2 - 1/n1 - 1/n2) / 2 averaged over the
    reactions with >= min_events in both seeds of every temperature where the campaign has two; None if no pair."""
    q = []
    for T in sorted({T_of[r] for r, L in camp.items() if L > 0}):
        rs = [r for r, L in camp.items() if L > 0 and T_of[r] == T]
        if len(rs) != 2:
            continue
        (n1, e1), (n2, e2) = (data[r][camp[r]] for r in rs)
        n1, e1, n2, e2 = n1[A], e1[A], n2[A], e2[A]
        ok = (n1 >= min_events) & (n2 >= min_events) & (e1 > 0) & (e2 > 0)
        q.extend(((np.log(n1[ok] / e1[ok]) - np.log(n2[ok] / e2[ok])) ** 2 - 1 / n1[ok] - 1 / n2[ok]).tolist())
    return max(0.0, float(np.mean(q)) / 2) if q else None


def fit_lines_re(n, E, T, runs, s2):
    """Random-effects line: weighted least squares of ln k on u = 1000/T with per-temperature variance
    1/y + s2/runs (y: events, 1/2 if none); phi from the weighted residuals as in fit_lines."""
    n, E = np.asarray(n, float), np.asarray(E, float)
    u = np.broadcast_to(1000.0 / np.asarray(T, float)[:, None], E.shape)
    use = E > 0
    y = np.where(n > 0, n, 0.5) * use
    lnk = np.log(np.where(use, y, 1.0) / np.where(use, E, 1.0)) * use
    w = np.where(use, 1.0 / (1.0 / np.where(use, y, 1.0) + s2 / np.asarray(runs, float)[:, None]), 0.0)
    m = use.sum(0)
    S0 = w.sum(0)
    ub = (w * u).sum(0) / np.where(S0 > 0, S0, 1.0)
    S2c = (w * (u - ub) ** 2).sum(0)
    ok = (m >= 2) & (S2c > 0)
    b = np.where(ok, -(w * (u - ub) * lnk).sum(0) / np.where(ok, S2c, 1.0), 0.0)
    a = np.where(m > 0, (w * lnk).sum(0) / np.where(S0 > 0, S0, 1.0) + b * ub, ABSENT)
    chi2 = (w * (lnk - (a - b * u)) ** 2 * use).sum(0)
    phi = np.where(m >= 3, np.maximum(1.0, chi2 / np.maximum(m - 2, 1)), 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cov = np.zeros((len(m), 2, 2))
        cov[:, 0, 0] = np.where(m > 0, 1.0 / S0, np.inf) + np.where(ok, ub ** 2 / S2c, 0.0)
        cov[:, 0, 1] = cov[:, 1, 0] = np.where(ok, ub / S2c, 0.0)
        cov[:, 1, 1] = np.where(ok, 1.0 / S2c, np.inf)
        cov *= phi[:, None, None]
    return {"a": a, "b": b, "cov": cov, "phi": phi, "seen": ((n > 0) & use).sum(0), "m": m, "ubar": ub}


def calibrated_re(camp, data, T_of, A, T_star, s2_default=0.0, tau2_default=0.0, min_events=10):
    """Round-2 variant: random-effects line plus the model-discrepancy term (internal leave-one-temperature-out)
    -> mean, predictive sd, s2, tau2, lines. The observation variance at T* is 1/n* + s2/runs*."""
    Ts, n, E = pool(camp, data, T_of)
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    s2 = seed_s2(camp, data, T_of, A)
    s2 = s2 if s2 is not None else s2_default
    lines = fit_lines_re(n[:, A], E[:, A], Ts, runs, s2)
    mean, sd = predict(lines, T_star)
    r2, v, d2 = [], [], []
    if len(Ts) >= 3:
        for k, T in enumerate(Ts):
            inner = np.arange(len(Ts)) != k
            li = fit_lines_re(n[inner][:, A], E[inner][:, A], Ts[inner], runs[inner], s2)
            mi, si = predict(li, T)
            nh, Eh = n[k, A], E[k, A]
            keep = (nh >= min_events) & (Eh > 0) & np.isfinite(si)
            r2.extend(((mi - np.log(np.where(keep, nh, 1) / np.where(keep, Eh, 1))) ** 2)[keep].tolist())
            v.extend((si ** 2 + 1.0 / np.maximum(nh, 1) + s2 / runs[k])[keep].tolist())
            d2.extend(((1000.0 / T - li["ubar"]) ** 2)[keep].tolist())
    tau2 = max(0.0, float(np.mean(r2) - np.mean(v))) / max(float(np.mean(d2)), 1e-12) if r2 else tau2_default
    with np.errstate(invalid="ignore"):
        sd2 = np.sqrt(sd ** 2 + tau2 * (1000.0 / T_star - lines["ubar"]) ** 2)
    return mean, sd2, s2, tau2, lines


# Round 3 (design section 13): the curvature prior, the median discrepancy tau^2 of the real V6 folds (random effects)
TAU2_PRIOR = 8.0


def greedy_bias(menu, start, budget, T_of, T_star, tau2, scale):
    """Static design with a curvature penalty: greedy on scale x c_value + tau2 (u* - u_bar)^2, u_bar the
    picosecond-weighted mean of 1000/T. scale: variance of ln k* per unit c_value (pilot-based, see
    pilot_scale); tau2 = 0 gives greedy()."""
    us, camp = 1000.0 / T_star, dict(start)

    def q(c):
        L = np.array([v for v in c.values() if v > 0], float)
        ub = (L * np.array([1000.0 / T_of[r] for r, v in c.items() if v > 0])).sum() / L.sum()
        return scale * c_value(c, T_of, T_star) + tau2 * (ub - us) ** 2

    while moves := _moves(menu, camp, budget):
        q0 = q(camp)

        def key(mv):
            cost, q1 = mv[1] - camp.get(mv[0], 0), q({**camp, mv[0]: mv[1]})
            gain = (np.inf if q1 < np.inf else 0.0) if q0 == np.inf else q0 - q1
            return (-gain / cost, cost, q1)
        r, L = min(moves, key=key)
        camp[r] = L
    return camp


def pilot_scale(pilot, data, A):
    """Event-weighted mean over reactions of 1 / (pilot count rate per bin): the factor that turns c_value into
    var(ln k*), counts per bin taken from the pilot (zero -> 1/2) as in section 11."""
    (r, L), = pilot.items()
    n = np.maximum(data[r][L][0][A], 0.5)
    return float(len(n) * L / n.sum())


# Round 3 (design section 13): curvature of real lumped rates, event-weighted median |c| of a quadratic in
# 1000/T over the 9 butane temperatures (development data, 2026-11-09)
GAMMA_PRIOR = 70.0


def chord_bias(camp, T_of, T_star):
    """Bias at the target of a straight line fitted to data with unit curvature (y = x^2, x = 1000/T - 1000/T*),
    picosecond weights: m2 - x_bar (m3 - x_bar m2) / (m2 - x_bar^2). Charges for spread on either side of the target
    as well as for a one-sided offset; inf with one temperature."""
    L = np.array([v for v in camp.values() if v > 0], float)
    x = np.array([1000.0 / T_of[r] for r, v in camp.items() if v > 0]) - 1000.0 / T_star
    w = L / L.sum()
    m1, m2, m3 = (w * x).sum(), (w * x * x).sum(), (w * x ** 3).sum()
    var = m2 - m1 ** 2
    return float(m2 - m1 * (m3 - m1 * m2) / var) if var > 1e-12 * max(m2, 1e-300) else np.inf


def greedy_curv(menu, start, budget, T_of, T_star, gamma, scale):
    """Static design with a curvature penalty: greedy on scale x c_value + (gamma x chord_bias)^2."""
    camp = dict(start)
    q = lambda c: scale * c_value(c, T_of, T_star) + (gamma * chord_bias(c, T_of, T_star)) ** 2

    while moves := _moves(menu, camp, budget):
        q0 = q(camp)

        def key(mv):
            cost, q1 = mv[1] - camp.get(mv[0], 0), q({**camp, mv[0]: mv[1]})
            gain = (np.inf if q1 < np.inf else 0.0) if q0 == np.inf else q0 - q1
            return (-gain / cost, cost, q1)
        r, L = min(moves, key=key)
        camp[r] = L
    return camp


S2_PRIOR = 0.007  # round 3: between-run variance of ln k, median of the real V6 folds (random effects)


def curvature2(camp, data, T_of, A, T_star, s2, min_events=10):
    """gamma^2: event-weighted mean of (c_j^2 - var c_j) over reactions with >= min_events at >= 3 temperatures,
    c_j the quadratic coefficient of ln k in 1000/T (weights 1 / (1/n + s2/runs)); None if not estimable."""
    Ts, n, E = pool(camp, data, T_of)
    if len(Ts) < 3:
        return None
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    x = 1000.0 / Ts - 1000.0 / T_star
    X = np.stack([np.ones_like(x), x, x * x], 1)
    num = den = 0.0
    for j in A:
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
        c = (Minv @ (X[ok].T @ (W * y)))[2]
        num, den = num + n[ok, j].sum() * (c * c - Minv[2, 2]), den + n[ok, j].sum()
    return max(0.0, num / den) if den > 0 else None


def seed_s2_window(camp, data, T_of, A, min_events=10):
    """seed_s2 over the two seeds' common time window (both cut to the shorter length): lumped rates drift within a
    run, so seeds of unequal length differ by more than run-to-run scatter. Design only; C4 keeps seed_s2."""
    q = []
    for T in sorted({T_of[r] for r, L in camp.items() if L > 0}):
        rs = [r for r, L in camp.items() if L > 0 and T_of[r] == T]
        if len(rs) != 2:
            continue
        L = min(camp[r] for r in rs)
        if not all(L in data[r] for r in rs):
            continue
        (n1, e1), (n2, e2) = (data[r][L] for r in rs)
        n1, e1, n2, e2 = n1[A], e1[A], n2[A], e2[A]
        ok = (n1 >= min_events) & (n2 >= min_events) & (e1 > 0) & (e2 > 0)
        q.extend(((np.log(n1[ok] / e1[ok]) - np.log(n2[ok] / e2[ok])) ** 2 - 1 / n1[ok] - 1 / n2[ok]).tolist())
    return max(0.0, float(np.mean(q)) / 2) if q else None


def adaptive(menu, start, budget, T_of, T_star, data, A, gamma0=GAMMA_PRIOR, s20=S2_PRIOR, target_bins=500,
             common_window=False):
    """Round 3: sequential design on the C4 predictive error. Each step's events are revealed; per reaction the
    objective is var (weighted least squares over runs, per-run variance 1/mu + s2) + (gamma x chord bias)^2 with
    information weights; event-weighted over reactions by predicted target events. gamma^2 and s2 are re-estimated
    from the campaign (curvature2, seed_s2) and fall back to the priors until estimable."""
    us, camp = 1000.0 / T_star, dict(start)
    while moves := _moves(menu, camp, budget):
        chosen = [r for r, L in camp.items() if L > 0]
        s2 = (seed_s2_window if common_window else seed_s2)(camp, data, T_of, A)
        s2 = s20 if s2 is None else s2
        g2 = curvature2(camp, data, T_of, A, T_star, s2)
        g2 = gamma0 ** 2 if g2 is None else g2
        Ts, n, E = pool(camp, data, T_of)
        b = fit_lines(n[:, A], E[:, A], Ts)["b"]
        rate = {r: np.maximum(data[r][camp[r]][0][A], 0.5) / camp[r] for r in chosen}  # counts per bin, per reaction
        near = lambda T: min(chosen, key=lambda r: abs(1000.0 / T_of[r] - 1000.0 / T))
        r0 = near(T_star)
        wt = rate[r0] * np.exp(-b * (us - 1000.0 / T_of[r0])) * target_bins

        def expected(r, L):
            if r in rate:
                return rate[r] * L
            rn = near(T_of[r])
            return rate[rn] * np.exp(-b * (1000.0 / T_of[r] - 1000.0 / T_of[rn])) * L

        def objective(c):
            xs = np.array([1000.0 / T_of[r] - us for r, L in c.items() if L > 0])
            w = np.array([1.0 / (1.0 / expected(r, L) + s2) for r, L in c.items() if L > 0])  # runs x reactions
            S0 = w.sum(0)
            m1, m2, m3 = ((w * xs[:, None] ** p).sum(0) / S0 for p in (1, 2, 3))
            var_x = m2 - m1 ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where(var_x > 1e-12, 1.0 / S0 + m1 ** 2 / (S0 * var_x), np.inf)
                bias = np.where(var_x > 1e-12, m2 - m1 * (m3 - m1 * m2) / var_x, np.inf)
            total = v + g2 * bias ** 2
            return float((wt * total).sum() / wt.sum()) if np.isfinite(total[wt > 0]).all() else np.inf

        q0 = objective(camp)

        def key(mv):
            cost, q1 = mv[1] - camp.get(mv[0], 0), objective({**camp, mv[0]: mv[1]})
            gain = (np.inf if q1 < np.inf else 0.0) if q0 == np.inf else q0 - q1
            return (-gain / cost, cost, q1)
        r, L = min(moves, key=key)
        camp[r] = L
    return camp


def anchored(menu, start, budget, T_of, T_star, data, A):
    """Round 3: anchor, then adapt. First the run nearest the target (in 1000/T) at the longest length that fits,
    then adaptive() with common-window scatter for the rest of the budget."""
    camp = dict(start)
    r = min((r for r in menu if r not in camp), key=lambda r: (abs(1000.0 / T_of[r] - 1000.0 / T_star), r))
    fits = [L for L in menu[r] if L <= budget - sum(camp.values())]
    if fits:
        camp[r] = max(fits)
    return adaptive(menu, camp, budget, T_of, T_star, data, A, common_window=True)
