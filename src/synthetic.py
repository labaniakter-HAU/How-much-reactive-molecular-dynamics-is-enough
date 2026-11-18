# file: src/synthetic.py
"""Synthetic test data (V0a, V0b). A species-level network is integrated exactly; class counts and events
per bin get the Poisson noise of a box `mult` times larger than V; species can then be lumped the way
build_tables.py lumps MD species (single-member classes in molecules, others in carbon atoms; events
merged by sign pattern of the net class change and reactant classes)."""
import numpy as np
import torch

from data import Group
from fit import per_bin_counts, simulate as integrate
from formula import atoms
from kinetics import LumpedKinetics


def network(species, reactions, Ea_kJ, A):
    """reactions: [(reactant formulas, product formulas)] -> species-level model, orders = multiplicities."""
    nu, o = np.zeros((len(species), len(reactions))), np.zeros((len(species), len(reactions)))
    for j, (R, P) in enumerate(reactions):
        for f in R:
            nu[species.index(f), j] -= 1
            o[species.index(f), j] += 1
        for f in P:
            nu[species.index(f), j] += 1
    return LumpedKinetics(nu, o, np.log(A), np.asarray(Ea_kJ, float) / 100.0)


def simulate(truth, temps, seeds, t_end, n_t, V, x0, mult, rng, per_bin=2, n_sub=8):
    """-> (noisy Group, clean concentrations (n_t, B, n_s)); x0: molecules of each species at t = 0."""
    t = torch.linspace(0, t_end, n_t, dtype=torch.float64)
    T = torch.tensor([x for x in temps for _ in range(seeds)], dtype=torch.float64)
    c0 = torch.as_tensor(np.tile(np.asarray(x0, float) / V, (len(T), 1)))
    g = Group(t, c0[None], T, torch.full((len(T),), float(V), dtype=torch.float64), None)
    with torch.no_grad():
        c, xi = integrate(truth, g, n_sub)
    vol = V * mult
    obs = rng.poisson(np.clip(c.numpy(), 0, None) * vol) / vol
    ev = rng.poisson(np.clip(per_bin_counts(xi, per_bin).numpy(), 0, None) * mult) / mult
    return Group(t, torch.as_tensor(obs), T, g.V, torch.as_tensor(ev)), c.numpy()


def lumping(species, reactions, classes, assign):
    """-> W (species x classes amount per molecule), carbon per class unit, type key of each reaction
    (None if it changes no class amount), and the list of distinct keys in first-seen order."""
    members = {c: [s for s in species if assign(s) == c] for c in classes}
    W = np.zeros((len(species), len(classes)))
    for i, s in enumerate(species):
        c = classes.index(assign(s))
        W[i, c] = 1.0 if len(members[classes[c]]) <= 1 else atoms(s)["C"]
    carbon = np.array([1.0 if len(members[c]) > 1 else float(atoms(members[c][0])["C"]) if members[c] else 0.0
                       for c in classes])
    keys, rkey = [], []
    for R, P in reactions:
        s = sum(W[species.index(f)] for f in P) - sum(W[species.index(f)] for f in R)
        key = None if not s.any() else (tuple(int(x) for x in np.sign(s)), tuple(sorted(assign(f) for f in R)))
        rkey.append(key)
        if key is not None and key not in keys:
            keys.append(key)
    return W, carbon, rkey, keys


def lump_groups(groups, species, reactions, classes, assign, weights_from=None):
    """Lumped groups plus S (classes x types: event-weighted mean net change), O (reactant molecules per
    class), carbon per unit and the type index of each true reaction (None if invisible)."""
    W, carbon, rkey, keys = lumping(species, reactions, classes, assign)
    s = np.array([sum(W[species.index(f)] for f in P) - sum(W[species.index(f)] for f in R) for R, P in reactions]).T
    tot = sum(g.ev.sum((0, 1)) for g in (weights_from or groups)).numpy()
    rtype = [None if k is None else keys.index(k) for k in rkey]
    S, O = np.zeros((len(classes), len(keys))), np.zeros((len(classes), len(keys)))
    M = np.zeros((len(reactions), len(keys)))  # reaction -> type membership
    for j, k in enumerate(rtype):
        if k is not None:
            M[j, k] = 1.0
    for k, key in enumerate(keys):
        w = tot * M[:, k]
        S[:, k] = s @ w / w.sum() if w.sum() > 0 else s[:, M[:, k] > 0].mean(1)
        for c in key[1]:
            O[classes.index(c), k] += 1
    out = [Group(g.t, g.obs @ torch.as_tensor(W), g.T, g.V, g.ev @ torch.as_tensor(M)) for g in groups]
    return out, S, O, carbon, rtype, W


# Round 2 (design section 12): random networks, "possible" (Arrhenius) and "impossible" (assumptions broken).
KINDS = ("possible", "curved", "overdispersed", "both")
# Round 3 (design section 13): "lumped" rates are sums of two channels, the second with a barrier higher by
# dE ~ U(*LUMPED_DE) kJ/mol and equal to the first at Tc ~ U(*LUMPED_TC). dE is set on development seeds so the
# discrepancy tau^2 matches the real data.
LUMPED_DE, LUMPED_TC = (160.0, 375.0), (3300.0, 4800.0)  # dev: 150-350 gave median tau^2 5.9, 175-400 gave 11.4 (real 8)


def random_network(base, seed, kind):
    """A random variant of a base network [(reactants, products, Ea kJ/mol, A)]: Ea x U(0.8, 1.2), ln A + N(0, 0.5).
    'curved': k = A (T / 4000)^n exp(-Ea / RT) with n ~ U(-4, 4) per reaction; 'overdispersed': every run gets
    its own lognormal rate factors, sd ~ U(0.1, 0.4); 'both'. -> dict(rx, Ea, lnA, n, sd_run)."""
    rng = np.random.default_rng(seed)
    Ea = np.array([r[2] for r in base], float) * rng.uniform(0.8, 1.2, len(base))
    lnA = np.log([r[3] for r in base]) + rng.normal(0.0, 0.5, len(base))
    n = rng.uniform(-4, 4, len(base)) if kind in ("curved", "both") else np.zeros(len(base))
    sd = rng.uniform(0.1, 0.4) if kind in ("overdispersed", "both") else 0.0
    net = {"rx": [r[:2] for r in base], "Ea": Ea, "lnA": lnA, "n": n, "sd_run": sd}
    if kind == "lumped":
        from kinetics import R_KJ
        dE = rng.uniform(*LUMPED_DE, len(base))
        Tc = rng.uniform(*LUMPED_TC, len(base))
        net.update(Ea2=Ea + dE, lnA2=lnA + dE / (R_KJ * Tc))  # the two channels are equal at Tc
    return net


def true_lnk(net, T):
    from kinetics import R_KJ
    lnk = net["lnA"] + net["n"] * np.log(T / 4000.0) - net["Ea"] / (R_KJ * T)
    return np.logaddexp(lnk, net["lnA2"] - net["Ea2"] / (R_KJ * T)) if "Ea2" in net else lnk


def run_data(net, species, T, bins, x0, V, lengths, rng, per_bin=2, n_sub=8):
    """One synthetic run of `bins` 1.2 ps bins: its own rate factors (if overdispersed), the noise-free trajectory,
    exposure per bin and Poisson events -> {length: (events, exposure) per reaction} for each length."""
    from kinetics import R_KJ
    lnk = true_lnk(net, T) + (rng.normal(0.0, net["sd_run"], len(net["Ea"])) if net["sd_run"] else 0.0)
    model = network(species, net["rx"], np.zeros(len(lnk)), np.exp(lnk))  # rates at this T folded into A
    t = torch.linspace(0, 1.2 * bins, per_bin * bins + 1, dtype=torch.float64)
    g = Group(t, torch.as_tensor(np.asarray(x0, float) / V)[None, None], torch.tensor([T], dtype=torch.float64),
              torch.tensor([V], dtype=torch.float64), None)
    with torch.no_grad():
        c = integrate(model, g, n_sub)[0][:, 0].numpy()
    rate = np.exp(np.log(np.clip(c, 1e-12, None)) @ model.orders.numpy()) * V
    cum = np.concatenate([[np.zeros(rate.shape[1])], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t.numpy())[:, None], 0)])
    E = np.diff(cum[::per_bin], axis=0)
    ev = np.cumsum(rng.poisson(np.exp(lnk) * E), 0)
    return {L: (ev[L - 1], E[:L].sum(0)) for L in lengths}, c
