# file: src/closure.py
"""Round 5 (design section 16): a counted model that stays complete when the species table is truncated.

Round 4's species-level model kept only reactions whose reactants are tracked in every run. On the HFO development
runs that left 44 % of its species' production outside the model (experiments/round5_rca.py). Counted fixes, none of
which fits anything to concentrations:
  1. model species: tracked at >= min_temps of the training temperatures (per-run tracking, not "every run");
  2. reactions keyed by their reactant set (mass action depends on reactants only); the net change per firing is the
     event average over the set's product channels;
  3. tail: a reaction outside the n_anchor most frequent reactant sets is not dropped. It is counted as a
     pseudo-first-order channel of its rarest reactant (rate law N_rarest), provided its other reactants never run
     out in the training runs; so nearly every explicit event stays in the model;
  4. closure: every other species is a short-lived intermediate in quasi-steady state. What is produced is consumed
     through its observed channels in the observed proportions (firings / production), followed down to model species;
  5. the net changes of 2-4 depend on temperature (branching): they are built per training temperature and
     interpolated linearly in 1000/T (nearest temperature outside the range).
  6. volume: the runs are at constant pressure, and the molecule count grows by 23-66 % within an HFO run, so the
     box grows with it (V proportional to N at fixed T and P) and count-unit rates of bimolecular reactions drift.
     Exposure and rate law carry the factor (N_REF / N)^(order - 1), N the total count of tracked molecules. In the
     model N is one more state (last row): its net change is the column sum of the net changes, its order
     -(order - 1), so src/kinetics.py and the integrators are used unchanged.
  7. start-up: in the first frames every run relaxes from its packed initial box (at 0.1 ps about a fifth of the
     water has changed species, the total energy jumps). Trajectories are modelled and compared from t0 on, starting
     at the observed state at t0. Rates stay whole-run counts.
Rates stay counted: events / exposure per temperature, in the runs where every reactant is tracked, then
design.fit_lines. Amounts are molecule counts, as in src/species.py.
"""
import re
from collections import Counter

import numpy as np
import torch

import design
import lab
import species
from data import Group

MAX_ORDER, PRESENT, N_REF = 3, 0.9, 1000.0


def atoms(formula):
    """'C4H3F5' -> {'C': 4, 'H': 3, 'F': 5}."""
    out = Counter()
    for el, k in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] += int(k or 1)
    return dict(out)


def exposure(t, counts, lhs, vol=None):
    """lab.exposure with the volume factor: integral of prod N^nu / vol^(order - 1), vol = N_total / N_REF."""
    f = np.prod([counts[s] ** nu for s, nu in lhs], axis=0)
    return float(np.trapezoid(f if vol is None else f / vol ** (sum(nu for _, nu in lhs) - 1), t))


def model_species(tracked, T_of, train, min_temps=1):
    """Species tracked at >= min_temps of the training temperatures (in at least one run each), sorted."""
    temps = {}
    for r in train:
        for s in tracked[r]:
            temps.setdefault(s, set()).add(T_of[r])
    return sorted(s for s, ts in temps.items() if len(ts) >= min_temps)


def sums(events, M, closed, jx, key):
    """Events {(lhs, rhs): n} -> firings n (J,), and summed over events the net change of model species (|M|, J) and the
    gross production of closed species (|closed|, J), per reaction key(lhs) (None: not in the model)."""
    ix, xx = {s: i for i, s in enumerate(M)}, {s: i for i, s in enumerate(closed)}
    n, S, Qp = np.zeros(len(jx)), np.zeros((len(M), len(jx))), np.zeros((len(closed), len(jx)))
    for (lhs, rhs), k in events.items():
        j = jx.get(key(lhs))
        if j is None:
            continue
        n[j] += k
        for s, nu in lhs:
            S[ix[s], j] -= nu * k
        for s, nu in rhs:
            if s in ix:
                S[ix[s], j] += nu * k
            elif s in xx:
                Qp[xx[s], j] += nu * k
    return n, S, Qp


def closure(events, M, closed, Qp):
    """Quasi-steady-state closure from event totals -> S_cl (|M|, J): net change of model species per firing of each
    explicit reaction through the closed species it feeds, and acc (|closed|, J): what stays in closed species.
    A reaction with k closed reactants is attributed 1/k to each. p = gross production of closed species:
    p = Qp r + Np F, F = A p, so p = (I - Np A)^-1 Qp r and the model species change by W A p."""
    ix, xx = {s: i for i, s in enumerate(M)}, {s: i for i, s in enumerate(closed)}
    P, C = np.zeros(len(closed)), np.zeros(len(closed))
    keys = []
    for (lhs, rhs), n in events.items():
        for s, nu in rhs:
            if s in xx:
                P[xx[s]] += nu * n
        for s, nu in lhs:
            if s in xx:
                C[xx[s]] += nu * n
        if any(s in xx for s, _ in lhs):
            keys.append((lhs, rhs))
    den = np.maximum(np.maximum(P, C), 1e-300)  # production; consumption if the tables record more of it
    A, Np = np.zeros((len(keys), len(closed))), np.zeros((len(closed), len(keys)))
    Nm, W = np.zeros((len(closed), len(keys))), np.zeros((len(M), len(keys)))
    for c, (lhs, rhs) in enumerate(keys):
        drivers = [xx[s] for s, _ in lhs if s in xx]
        A[c, drivers] = events[(lhs, rhs)] / den[drivers] / len(drivers)
        for s, nu in lhs:
            if s in ix:
                W[ix[s], c] -= nu
            else:
                Nm[xx[s], c] += nu
        for s, nu in rhs:
            if s in ix:
                W[ix[s], c] += nu
            elif s in xx:
                Np[xx[s], c] += nu
    K = np.linalg.solve(np.eye(len(closed)) - Np @ A, Qp)  # closed species produced per explicit firing, chains included
    AK = A @ K
    return W @ AK, K - Nm @ AK


class Tables:
    """One fold's counted model, from the training runs.
    M: model species; rx: its reactions (reactant sets: the n_anchor most frequent, plus one pseudo-first-order
    channel per species for the tail when absorb); n[run], E[run] (J,): firings and exposure of every run (NaN where a
    reactant is untracked in that run); O (|M|, J); score: columns of M that the error metric uses (round 4's
    species: tracked in every run of the series)."""

    def __init__(self, ev, cnt, T_of, train, min_temps=1, n_anchor=80, coverage=0.95, absorb=True, by_T=True, volume=True, t0=0.0):
        tracked = {r: set(c) for r, (_, c) in cnt.items()}
        self.T_of, self.train, self.by_T, self.volume, self.t0 = T_of, list(train), by_T, volume, t0
        self.vol = {r: sum(c.values()) / N_REF for r, (_, c) in cnt.items()} if volume else {r: None for r in cnt}
        self.M = model_species(tracked, T_of, train, min_temps)
        Mset = set(self.M)
        self.explicit = lambda lhs: all(s in Mset for s, _ in lhs) and sum(nu for _, nu in lhs) <= MAX_ORDER
        self.temps = sorted({T_of[r] for r in train})
        self.events = {T: Counter() for T in self.temps}  # training events per temperature
        for r in train:
            self.events[T_of[r]].update(ev[r])
        pooled = sum(self.events.values(), Counter())
        # anchors: fit.anchors' rule on the reactant sets that have a rate (every reactant tracked in one training run)
        tot = Counter()
        for (lhs, _), k in pooled.items():
            if self.explicit(lhs):
                tot[lhs] += k
        ranked = [lhs for lhs, _ in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))
                  if any(all(s in tracked[r] for s, _ in lhs) for r in train)]
        cum = np.cumsum([tot[lhs] for lhs in ranked]) / max(sum(tot.values()), 1)
        self.anchor_set = set(ranked[: min(n_anchor, int(np.searchsorted(cum, coverage - 1e-12)) + 1)])
        mean = {s: np.mean([cnt[r][1][s].mean() for r in train if s in tracked[r]]) for s in self.M}
        # a co-reactant may sit outside the rate law only if it never runs out: present in >= PRESENT of the frames
        # of every training run. Otherwise the channel would go on consuming it at zero (O2 in O2-free runs, the parent
        # once it is used up).
        lasting = {s for s in self.M if all(s in tracked[r] and (cnt[r][1][s] > 0).mean() >= PRESENT for r in train)}

        def rare(lhs):
            sp = [s for s, _ in lhs]
            ok = [s for s in sp if all(o in lasting for o in sp if o != s)]
            return ((min(ok, key=lambda s: (mean[s], s)), 1),) if ok else None

        def key(lhs):
            if not self.explicit(lhs):
                return None
            return lhs if lhs in self.anchor_set else (rare(lhs) if absorb else None)
        self.key = key
        n = Counter()
        for (lhs, _), k in pooled.items():
            if key(lhs) is not None:
                n[key(lhs)] += k
        self.rx = [lhs for lhs, _ in sorted(n.items(), key=lambda kv: (-kv[1], kv[0]))]
        self.jx = {lhs: j for j, lhs in enumerate(self.rx)}
        self.closed = sorted({s for k in pooled for side in k for s, _ in side} - Mset)
        self.n_train, S, Qp = sums(pooled, self.M, self.closed, self.jx, key)
        nz = np.maximum(self.n_train, 1e-300)
        self.S_pooled = S / nz + closure(pooled, self.M, self.closed, Qp / nz)[0]
        self.S_open = S / nz  # without the closure (diagnostics)
        self.S_of = {}
        for T in self.temps if by_T else ():  # a reaction without events at T takes the pooled net change
            nT, ST, QT = sums(self.events[T], self.M, self.closed, self.jx, key)
            has = nT > 0
            ST, QT = np.where(has, ST / np.maximum(nT, 1e-300), S / nz), np.where(has, QT / np.maximum(nT, 1e-300), Qp / nz)
            self.S_of[T] = ST + closure(self.events[T], self.M, self.closed, QT)[0]
        ix = {s: i for i, s in enumerate(self.M)}
        self.O = np.zeros((len(self.M), len(self.rx)))
        for j, lhs in enumerate(self.rx):
            for s, nu in lhs:
                self.O[ix[s], j] = nu
        self.n, self.E = {}, {}
        for r in ev:
            k = np.zeros(len(self.rx))
            for (lhs, _), m in ev[r].items():
                if key(lhs) in self.jx:
                    k[self.jx[key(lhs)]] += m
            t, c = cnt[r]
            self.n[r] = k
            self.E[r] = np.array([exposure(t, c, lhs, self.vol[r]) if all(s in c for s, _ in lhs) else np.nan for lhs in self.rx])
        self.A = np.flatnonzero([any(self.E[r][j] > 0 for r in train) for j in range(len(self.rx))])  # reactions with a rate
        every = set.intersection(*tracked.values())
        self.score = np.array([i for i, s in enumerate(self.M) if s in every])

    def weights(self, T):
        """Linear interpolation in 1000/T between the neighbouring training temperatures; the nearest one outside."""
        u, us = 1000.0 / T, [1000.0 / x for x in self.temps]
        if T in self.temps:
            return {T: 1.0}
        lo = [x for x in self.temps if x < T]
        hi = [x for x in self.temps if x > T]
        if not lo or not hi:
            return {(hi[0] if hi else lo[-1]): 1.0}
        a, b = lo[-1], hi[0]
        w = (u - 1000.0 / b) / (1000.0 / a - 1000.0 / b)
        return {a: w, b: 1.0 - w}

    def S(self, T, closed=True):
        """Net change of model species per firing at temperature T (closure included)."""
        if not closed:
            return self.S_open
        return sum(w * self.S_of[x] for x, w in self.weights(T).items()) if self.by_T else self.S_pooled

    def with_volume(self, S, O):
        """Append the volume state (N_total / N_REF) as the last row of a net-change and an order matrix."""
        if not self.volume:
            return S, O
        return np.vstack([S, S.sum(0) / N_REF]), np.vstack([O, 1.0 - O.sum(0)])

    def pooled(self, runs, cols):
        """-> Ts, n, E (m, len(cols)) pooled per temperature over the runs where the reaction has an exposure."""
        Ts = np.array(sorted({self.T_of[r] for r in runs}))
        n, E = np.zeros((len(Ts), len(cols))), np.zeros((len(Ts), len(cols)))
        for r in runs:
            ok, k = np.isfinite(self.E[r][cols]), int(np.searchsorted(Ts, self.T_of[r]))
            n[k] += np.where(ok, self.n[r][cols], 0.0)
            E[k] += np.where(ok, self.E[r][cols], 0.0)
        return Ts, n, E

    def lines(self, runs, cols):
        """The counted-rate estimator (design.fit_lines), as round 4."""
        Ts, n, E = self.pooled(runs, cols)
        return design.fit_lines(n, E, Ts)

    def group(self, cnt, runs):
        """Runs -> Group (B = len(runs)) on round 4's grid; species untracked in a run start at 0 and are unobserved."""
        obs = []
        for r in runs:
            t, c = cnt[r]
            rows = np.r_[0:len(t) - 1:species.EVERY, len(t) - 1]
            rows = rows[t[rows] >= self.t0 - 1e-9]
            obs.append(np.stack([c[s][rows] if s in c else np.zeros(len(rows)) for s in self.M] + ([self.vol[r][rows]] if self.volume else []), 1))
        return Group(torch.tensor(t[rows]), torch.tensor(np.stack(obs, 1)), torch.tensor([self.T_of[r] for r in runs], dtype=torch.float64),
                     torch.ones(len(runs), dtype=torch.float64), None)

    def model(self, lines, cols, T, closed=True):
        return species.model(*self.with_volume(self.S(T, closed), self.O), lines, cols)


def predict(m, g, extents=True):
    """species.predict, with a failed integration (non-finite rates or Jacobian) returned as NaN instead of raised."""
    try:
        return species.predict(m, g, extents)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        B, n_s, n_r = g.obs.shape[1], m.nu0.shape[0], m.nu0.shape[1]
        return np.full((len(g.t), B, n_s), np.nan), (np.full((B, n_r), np.nan) if extents else None)


def full_size(ev, T_of, train, M):
    """Reactions of the full network: (reactants, products) pairs seen in the training runs whose reactants are model
    species, order <= MAX_ORDER (the rate-bearing subset is counted by the caller)."""
    Mset = set(M)
    return {k for r in train for k in ev[r] if all(s in Mset for s, _ in k[0]) and sum(nu for _, nu in k[0]) <= MAX_ORDER}


def compact_tables(ev, cnt, T_of, train, max_size, start=160, step=10, **kw):
    """C3's compact model: the largest anchor count (start, start - step, ...) whose model has at most max_size reactions
    with a rate (anchors plus tail channels). -> Tables, or None if even step anchors exceed max_size."""
    for n in range(start, 0, -step):
        tab = Tables(ev, cnt, T_of, train, n_anchor=n, **kw)
        if len(tab.A) <= max_size:
            return tab
    return None


def scored(g, score):
    return Group(g.t, g.obs[..., score], g.T, g.V, None)


def loss(m, g, score, n_sub=species.N_SUB):
    """species.loss on the scored species only (the model also carries species the metric does not score)."""
    c, obs = species.simulate(m, g, n_sub)[0][..., score], g.obs[..., score]
    scale = obs.abs().amax(0)
    ok = scale > 0
    return (((c - obs) / torch.where(ok, scale, torch.ones_like(scale))) ** 2).mean(0)[ok].mean()


def conc_fit(models, groups, score, rounds=5, n_sub=species.N_SUB):
    """species.conc_fit for models that share lnA and Ea (one per training temperature) with their run groups; in
    place. -> final loss, or None when the counted start does not integrate."""
    total = lambda: sum(loss(m, g, score, n_sub) * g.obs.shape[1] for m, g in zip(models, groups)) / sum(g.obs.shape[1] for g in groups)
    with torch.no_grad():
        if not torch.isfinite(total()):
            return None
    opt = torch.optim.LBFGS([models[0].lnA, models[0].Ea], lr=1, max_iter=20, history_size=20, line_search_fn="strong_wolfe")

    def closure_():
        opt.zero_grad()
        L = total()
        if not torch.isfinite(L):
            return torch.tensor(1e10, dtype=torch.float64)  # rejected by the line search
        L.backward()
        return L
    for _ in range(rounds):
        opt.step(closure_)
    with torch.no_grad():
        return total().item()
