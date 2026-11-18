# file: src/species.py
"""Round 4 (design section 15): Part I's counted-rate model at species level, for the lab's HFO/MEA runs.

Species: the molecules tracked in every run's 1.csv. Reactions: lab.tables' (tracked reactants, order <= 3).
Amounts are molecule counts with V = 1 (count units, design section 14), so k * prod N^o is events per ps;
products outside the tracked species leave the model. Integration: Part I's RK4 (fit.simulate) on a 1 ps grid
with 8 substeps (h = 0.125 ps; the fastest consumption seen in any run is 10 /ps, the 0.1 ps frame limit).
Fixed 2026-11-10, before any round-4 quantity was computed on these data.
"""
import numpy as np
import torch

import design
import fit
import lab
from data import Group
from kinetics import R_KJ, LumpedKinetics
from metrics import normalized_rmse

EVERY, N_SUB = 10, 8


def load(root):
    """-> T_of, rx, data (as lab.tables), species, S, O (species x reactions), groups {run: Group with B = 1}."""
    T_of, rx, data = lab.tables(root)
    cnt = {nm: lab.read_counts(f) for nm, _, _, f in lab.runs(root)}
    species = sorted(set.intersection(*(set(c) for _, c in cnt.values())))
    ix = {s: i for i, s in enumerate(species)}
    S, O = np.zeros((len(species), len(rx))), np.zeros((len(species), len(rx)))
    for j, (lhs, rhs) in enumerate(rx):
        for s, nu in lhs:
            O[ix[s], j] += nu
            S[ix[s], j] -= nu
        for s, nu in rhs:
            if s in ix:
                S[ix[s], j] += nu
    groups = {}
    for nm, (t, c) in cnt.items():
        rows = np.r_[0:len(t) - 1:EVERY, len(t) - 1]  # 0, 1, ..., 199 ps and the last row (199.9 ps)
        obs = np.stack([c[s] for s in species], 1)[rows]
        groups[nm] = Group(torch.tensor(t[rows]), torch.tensor(obs)[:, None], torch.tensor([T_of[nm]], dtype=torch.float64),
                           torch.ones(1, dtype=torch.float64), None)
    return T_of, rx, data, species, S, O, groups


def batch(groups, names):
    """Runs on one time grid -> one Group (B = len(names))."""
    gs = [groups[n] for n in names]
    assert all(torch.equal(g.t, gs[0].t) for g in gs), "runs on different time grids"
    return Group(gs[0].t, torch.cat([g.obs for g in gs], 1), torch.cat([g.T for g in gs]), torch.cat([g.V for g in gs]), None)


def model(S, O, lines, cols):
    """Mass-action model of reactions `cols` with the Arrhenius lines ln k = a - b * 1000 / T (Ea in 100 kJ/mol)."""
    return LumpedKinetics(S[:, cols], O[:, cols], lines["a"], 10.0 * lines["b"] * R_KJ)


def lines_of(names, data, T_of, cols):
    """The counted-rate estimator (design.fit_lines) on runs `names`, repeats allowed (bootstrap)."""
    Ts = np.array(sorted({T_of[r] for r in names}))
    n = np.array([sum(data[r][1][0][cols] for r in names if T_of[r] == T) for T in Ts])
    E = np.array([sum(data[r][1][1][cols] for r in names if T_of[r] == T) for T in Ts])
    return design.fit_lines(n, E, Ts)


def simulate(m, g, n_sub=N_SUB):
    """fit.simulate with the constant parts (nu, ln k) hoisted out of the RK4 loop: the same arithmetic in the same
    order (tests/test_species.py), about 1.6x faster. -> counts (n_t, B, species), cumulative events (n_t, B, reactions)."""
    n_c, nuT, lnk, V = g.obs.shape[-1], m.nu().T, m.ln_k(g.T), g.V[:, None]

    def f(y):
        r = torch.exp((lnk + torch.log(y[:, :n_c].clamp_min(1e-12)) @ m.orders).clamp(max=m.ln_r_max)) * m.active
        return torch.cat([r @ nuT, r * V], 1)
    y = torch.cat([g.obs[0], torch.zeros(g.obs.shape[1], m.nu0.shape[1], dtype=torch.float64)], 1)
    out = [y]
    for i in range(len(g.t) - 1):
        h = (g.t[i + 1] - g.t[i]) / n_sub
        for _ in range(n_sub):
            k1 = f(y)
            k2 = f(y + 0.5 * h * k1)
            k3 = f(y + 0.5 * h * k2)
            k4 = f(y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out.append(y)
    y = torch.stack(out)
    return y[..., :n_c], y[..., n_c:]


def loss(m, g, n_sub=N_SUB):
    """Part I's concentration loss on counts: squared error over the species' largest count in the run, mean over
    time, then over the (run, species) pairs where the species occurs (the pairs the error metric scores)."""
    c = simulate(m, g, n_sub)[0]
    scale = g.obs.abs().amax(0)
    ok = scale > 0
    return (((c - g.obs) / torch.where(ok, scale, torch.ones_like(scale))) ** 2).mean(0)[ok].mean()


def conc_fit(m, g, rounds=5, n_sub=N_SUB):
    """Concentrations only (lam = 0), L-BFGS from the counted start as fit.lbfgs; in place. -> final loss, or None
    when the counted start itself does not integrate (the model is then left at the counted rates)."""
    with torch.no_grad():
        if not torch.isfinite(loss(m, g, n_sub)):
            return None
    opt = torch.optim.LBFGS([m.lnA, m.Ea], lr=1, max_iter=20, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L = loss(m, g, n_sub)
        if not torch.isfinite(L):
            return torch.tensor(1e10, dtype=torch.float64)  # rejected by the line search
        L.backward()
        return L
    for _ in range(rounds):
        opt.step(closure)
    with torch.no_grad():
        return loss(m, g, n_sub).item()


def predict(m, g, extents=True, rtol=1e-6, atol=1e-8):
    """The evaluator of every round-4 prediction: each run integrated by scipy's BDF with the analytic Jacobian
    (stiff-safe; Part I integrated its full network the same way), the rate law of LumpedKinetics.rates.
    -> counts (n_t, B, species) on g.t and predicted events per reaction over each run (B, reactions), or None
    without extents. A run whose integration fails is NaN."""
    from scipy.integrate import solve_ivp
    nu, O, act = m.nu().detach().numpy(), m.orders.numpy(), m.active.numpy()
    lnk_all, t, cap = m.ln_k(g.T).detach().numpy(), g.t.numpy(), m.ln_r_max
    n_s, n_r, B = nu.shape[0], nu.shape[1], g.obs.shape[1]
    c_out, xi_out = np.full((len(t), B, n_s), np.nan), np.full((B, n_r), np.nan)
    for b in range(B):
        lnk, V = lnk_all[b], float(g.V[b])

        def rates(c):
            x = lnk + np.log(np.maximum(c, 1e-12)) @ O
            return np.exp(np.minimum(x, cap)) * act, x

        def rhs(_, y):
            r, _ = rates(y[:n_s])
            return np.concatenate([nu @ r, r * V]) if extents else nu @ r

        def jac(_, y):
            c = y[:n_s]
            r, x = rates(c)
            dr = (r * (x < cap))[None, :] * O * np.where(c > 1e-12, 1.0 / np.maximum(c, 1e-12), 0.0)[:, None]
            J = np.zeros((n_s + n_r, n_s + n_r) if extents else (n_s, n_s))
            J[:n_s, :n_s] = nu @ dr.T
            if extents:
                J[n_s:, :n_s] = V * dr.T
            return J
        y0 = np.concatenate([g.obs[0, b].numpy(), np.zeros(n_r)]) if extents else g.obs[0, b].numpy()
        sol = solve_ivp(rhs, (t[0], t[-1]), y0, method="BDF", t_eval=t, jac=jac, rtol=rtol, atol=atol)
        if sol.status == 0:
            c_out[:, b] = sol.y[:n_s].T
            if extents:
                xi_out[b] = sol.y[n_s:, -1]
    return c_out, (xi_out if extents else None)


def error_terms(c, g):
    """Part I's metric per run (normalized RMSE per species, averaged over species present) and its weight."""
    obs = g.obs.numpy()
    return [(normalized_rmse(c[:, [b]], obs[:, [b]]), int((np.abs(obs[:, b]).max(0) > 0).sum())) for b in range(obs.shape[1])]


def firing_terms(pred, counted):
    """Squared difference of 2 sqrt(count + 1) per run and reaction (run totals: one bin per run)."""
    return ((2 * np.sqrt(np.clip(pred, 0, None) + 1) - 2 * np.sqrt(counted + 1)) ** 2).ravel()


def seed_s2_k(camp, data, T_of, A, min_events=10):
    """design.seed_s2 for k >= 2 runs per temperature: per reaction and temperature, the sample variance of ln k over
    the runs minus their mean Poisson variance 1/n, averaged; identical to seed_s2 when every k is 2."""
    q = []
    for T in sorted({T_of[r] for r, L in camp.items() if L > 0}):
        rs = [r for r, L in camp.items() if L > 0 and T_of[r] == T]
        if len(rs) < 2:
            continue
        n = np.array([data[r][camp[r]][0][A] for r in rs])
        E = np.array([data[r][camp[r]][1][A] for r in rs])
        ok = ((n >= min_events) & (E > 0)).all(0)
        x = np.log(n[:, ok] / E[:, ok])
        q.extend((x.var(0, ddof=1) - (1.0 / n[:, ok]).mean(0)).tolist())
    return max(0.0, float(np.mean(q))) if q else None


def calibrated_re_k(camp, data, T_of, A, T_star, s2_default=0.0, tau2_default=0.0, min_events=10):
    """design.calibrated_re (C4, frozen in round 2) with seed_s2_k in place of seed_s2; nothing else changes."""
    Ts, n, E = design.pool(camp, data, T_of)
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    s2 = seed_s2_k(camp, data, T_of, A, min_events)
    s2 = s2 if s2 is not None else s2_default
    lines = design.fit_lines_re(n[:, A], E[:, A], Ts, runs, s2)
    mean, sd = design.predict(lines, T_star)
    r2, v, d2 = [], [], []
    if len(Ts) >= 3:
        for k, T in enumerate(Ts):
            inner = np.arange(len(Ts)) != k
            li = design.fit_lines_re(n[inner][:, A], E[inner][:, A], Ts[inner], runs[inner], s2)
            mi, si = design.predict(li, T)
            nh, Eh = n[k, A], E[k, A]
            keep = (nh >= min_events) & (Eh > 0) & np.isfinite(si)
            r2.extend(((mi - np.log(np.where(keep, nh, 1) / np.where(keep, Eh, 1))) ** 2)[keep].tolist())
            v.extend((si ** 2 + 1.0 / np.maximum(nh, 1) + s2 / runs[k])[keep].tolist())
            d2.extend(((1000.0 / T - li["ubar"]) ** 2)[keep].tolist())
    tau2 = max(0.0, float(np.mean(r2) - np.mean(v))) / max(float(np.mean(d2)), 1e-12) if r2 else tau2_default
    with np.errstate(invalid="ignore"):
        sd2 = np.sqrt(sd ** 2 + tau2 * (1000.0 / T_star - lines["ubar"]) ** 2)
    return mean, sd2, s2, tau2, lines
