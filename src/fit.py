# file: src/fit.py
"""Fitting the event-anchored model (design section 5).

anchors: the most frequent event types; rates (the main method): k = events / exposure per temperature,
then an Arrhenius line; fit (comparisons only): L-BFGS on the joint loss (normalized squared error of
class trajectories plus lam x squared difference of 2 sqrt(count + 1) between each reaction's predicted
and counted events per bin); reduce: backward elimination judged on held-out runs. No L1 and no
derivative matching (both distorted mechanisms in V0, 2026-11-04). Fitting to concentrations moved
activation energies to absorb lumping error in V0b, so the main method does not fit (2026-11-04).
"""
import copy

import numpy as np
import torch

from integrate import rk4
from kinetics import R_KJ, LumpedKinetics
from metrics import normalized_rmse


def anchors(groups, coverage=0.95, max_n=80):
    """Type indices in order of frequency until they cover `coverage` of all counted events, at most max_n.
    max_n 80, not 40: on real validation runs it cut the held-out error by 31 %; 120 gained 0.5 % (2026-11-04)."""
    n = sum(g.ev.sum((0, 1)) for g in groups).numpy()
    order = np.argsort(-n, kind="stable")
    order = order[n[order] > 0]
    cum = np.cumsum(n[order]) / n.sum()
    return order[: min(max_n, int(np.searchsorted(cum, coverage - 1e-12)) + 1)]


def start_values(groups, S, O, types):
    """lnA, Ea (100 kJ/mol) per anchor from events / (V * integral of prod c^o dt), pooled per temperature,
    then a count-weighted line of ln k against 1/T. Seen at one temperature only: Ea = 0."""
    num, den = {}, {}
    o = torch.as_tensor(O[:, types])
    for g in groups:
        expo = torch.trapezoid(torch.exp(torch.log(g.obs.clamp_min(1e-12)) @ o), g.t, dim=0) * g.V[:, None]
        n = g.ev[..., types].sum(1)
        for b, T in enumerate(g.T.tolist()):
            num[T] = num.get(T, 0) + n[b]
            den[T] = den.get(T, 0) + expo[b]
    Ts = np.array(sorted(num))
    n = torch.stack([num[T] for T in Ts]).numpy()
    k = n / np.maximum(torch.stack([den[T] for T in Ts]).numpy(), 1e-300)
    lnA, Ea = np.zeros(len(types)), np.zeros(len(types))
    for j in range(len(types)):
        seen = n[:, j] > 0
        if seen.sum() >= 2:
            slope, icept = np.polyfit(1.0 / Ts[seen], np.log(k[seen, j]), 1, w=np.sqrt(n[seen, j]))
            lnA[j], Ea[j] = icept, -slope * R_KJ / 100.0
        else:
            lnA[j] = np.log(k[seen, j].mean()) if seen.any() else -30.0
    return lnA, Ea


def build(groups, S, O, carbon=None, coverage=0.95, max_n=80):
    types = anchors(groups, coverage, max_n)
    lnA, Ea = start_values(groups, S, O, types)
    model = LumpedKinetics(S[:, types], O[:, types], lnA, Ea, carbon)
    model.types = torch.as_tensor(types)
    return model


def set_counted_rates(model, groups, O):
    """The main method: every reaction's rate from the events and exposure in `groups`, in place."""
    lnA, Ea = start_values(groups, None, O, model.types.numpy())
    with torch.no_grad():
        model.lnA.copy_(torch.as_tensor(lnA))
        model.Ea.copy_(torch.as_tensor(Ea))


def simulate(model, g, n_sub=2):
    """-> c (n_t, B, n_c) and cumulative events per reaction (n_t, B, n_r), integrated together."""
    n_c = g.obs.shape[-1]

    def f(y, T):
        r = model.rates(y[:, :n_c], T)
        return torch.cat([r @ model.nu().T, r * g.V[:, None]], 1)
    y0 = torch.cat([g.obs[0], torch.zeros(g.obs.shape[1], model.nu0.shape[1], dtype=torch.float64)], 1)
    y = rk4(f, y0, g.T, g.t, n_sub)
    return y[..., :n_c], y[..., n_c:]


def per_bin_counts(xi, per_bin):
    """Cumulative events at grid points -> events per bin (B, n_bins, n_r)."""
    e = xi[::per_bin]
    return (e[1:] - e[:-1]).transpose(0, 1)


def loss(model, groups, lam, per_bin=2, n_sub=2):
    total, act = 0.0, model.active.bool()
    for g in groups:
        c, xi = simulate(model, g, n_sub)
        scale = g.obs.abs().amax(0).clamp_min(1e-12)
        total = total + (((c - g.obs) / scale) ** 2).mean()
        if lam:
            pred = per_bin_counts(xi, per_bin)[..., act]
            obs = g.ev[..., model.types][..., act]
            total = total + lam * ((2 * (pred.clamp_min(0) + 1).sqrt() - 2 * (obs + 1).sqrt()) ** 2).mean()
    return total / len(groups)


def lbfgs(model, groups, lam, rounds=10, per_bin=2, n_sub=2):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(params, lr=1, max_iter=20, history_size=20, line_search_fn="strong_wolfe")
    first = [True]

    def closure():
        opt.zero_grad()
        L = loss(model, groups, lam, per_bin, n_sub)
        if not torch.isfinite(L):
            if first[0]:
                raise FloatingPointError("loss is not finite at the start values")
            return torch.tensor(1e10, dtype=torch.float64)  # rejected by the line search
        first[0] = False
        L.backward()
        return L
    for _ in range(rounds):
        opt.step(closure)
    with torch.no_grad():
        return loss(model, groups, lam, per_bin, n_sub).item()


def error(model, groups, n_sub=2):
    """Design metric (normalized RMSE per class, averaged), mean over groups."""
    with torch.no_grad():
        return float(np.mean([normalized_rmse(simulate(model, g, n_sub)[0].numpy(), g.obs.numpy()) for g in groups]))


def without(model, j):
    m = copy.deepcopy(model)
    m.drop(j)
    return m


def reduce(model, fit_groups, val_groups, lam, tol=0.02, n_try=2, rounds=3, per_bin=2, n_sub=2, log=print):
    """Backward elimination: drop the reaction whose removal, after a short refit, raises the held-out
    error least; stop when that error exceeds the best seen by more than tol. Returns model and path."""
    best = error(model, val_groups, n_sub)
    path = [(model.n_active(), best)]
    while model.n_active() > 1:
        act = [j for j in range(len(model.active)) if model.active[j] > 0]
        quick = sorted((error(without(model, j), val_groups, n_sub), j) for j in act)[:n_try]
        trials = []
        for _, j in quick:
            m = without(model, j)
            lbfgs(m, fit_groups, lam, rounds, per_bin, n_sub)
            trials.append((error(m, val_groups, n_sub), j, m))
        v, j, m = min(trials, key=lambda x: x[0])
        if v > best * (1 + tol):
            log(f"stop: dropping reaction {j} gives held-out error {v:.4f} > {best:.4f} x {1 + tol}")
            break
        model, best = m, min(best, v)
        path.append((model.n_active(), v))
        log(f"dropped reaction {j}: held-out error {v:.4f}, {model.n_active()} left")
    return model, path
