# file: src/fullnet.py
"""Baselines 5 and 6 (design section 5): the full elementary event network rebuilt Reed-style, and its
flux-ranked skeleton as in RxMD-SN.

- Reactions: every species-level reaction seen in the rate runs, keyed by sorted reactant and product formulas.
- Exposure: V x integral of prod c^o dt, with species counts rebuilt at 12 fs by replaying the events from the
  first species frame. Species frames are 0.12 ps apart; 22-29 % of the reactions seen at 4500-5000 K involve
  a species that lives shorter than that and got almost no exposure from the frames alone (2026-11-04).
  Counts change only through events, so the replay must reproduce every stored frame exactly.
- Rates: k = events / exposure pooled per temperature, then an Arrhenius line through the temperatures where
  the reaction was seen; zero barrier if it was seen at one temperature only. E_a here is in kJ/mol.
- Prediction: mean-field ODEs (BDF with the sparse Jacobian), lumped to classes with assign_class.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.integrate import solve_ivp

import lumping
from kinetics import R_KJ


def load_run(cache, run_id):
    """Species table (frames x formulas, counts), mean volume (nm3) and events of one cached run."""
    stem = f"{run_id}_0.12ts_every100_ffield.reax.cho.2017"
    s = pd.read_csv(Path(cache) / f"{stem}.csv.gz")
    ev = pd.read_csv(Path(cache) / f"{stem}_events.csv.gz", keep_default_na=False)
    return s.pivot_table(index="t_ps", columns="formula", values="count", fill_value=0), float(s["V_nm3"].mean()), ev


def key(reactants, products):
    return tuple(sorted(reactants.split(" + "))), tuple(sorted(products.split(" + ")))


class Network:
    """Species-level mass-action network. O: reactant multiplicities, N: net change (species x reactions)."""

    def __init__(self, species, reactions):
        self.species, self.reactions = list(species), list(reactions)
        self.index = {f: i for i, f in enumerate(self.species)}
        o, n = ([], [], []), ([], [], [])
        for j, (re, pr) in enumerate(self.reactions):
            for f, k in Counter(re).items():
                o[0].append(self.index[f]); o[1].append(j); o[2].append(k)
            net = Counter(pr)
            net.subtract(Counter(re))
            for f, k in net.items():
                if k:
                    n[0].append(self.index[f]); n[1].append(j); n[2].append(k)
        shape = (len(self.species), len(self.reactions))
        self.O = sp.csr_matrix((o[2], (o[0], o[1])), shape=shape)
        self.N = sp.csr_matrix((n[2], (n[0], n[1])), shape=shape)
        self.OT = self.O.T.tocsr()
        self.lnA, self.Ea = np.full(shape[1], -np.inf), np.zeros(shape[1])

    @classmethod
    def from_runs(cls, runs, extra_species=()):
        """Every species and reaction seen in runs [(T, w, V, ev)], plus extra species (e.g. the starting
        species of runs to predict, inert if they never react)."""
        species, reactions = set(extra_species), set()
        for _, w, _, ev in runs:
            species |= set(w.columns)
            for a, b in zip(ev["reactants"], ev["products"]):
                reactions.add(key(a, b))
                species |= set(a.split(" + ")) | set(b.split(" + "))
        return cls(sorted(species), sorted(reactions))

    def counts(self, ev):
        j = {r: i for i, r in enumerate(self.reactions)}
        n = np.zeros(len(self.reactions))
        for a, b in zip(ev["reactants"], ev["products"]):
            n[j[key(a, b)]] += 1
        return n

    def _log_c(self, counts, V):
        return np.log(np.where(counts > 0, counts / V, 1e-300))

    def replay_exposure(self, w, V, ev):
        """V x integral of prod c^o dt per reaction from the counts rebuilt by replaying the events.
        Returns (exposure, largest count mismatch with the stored species frames)."""
        t_frames = w.index.to_numpy()
        stored = np.zeros((len(w), len(self.species)))
        stored[:, [self.index[f] for f in w.columns]] = w.to_numpy(float)
        counts = stored[0].copy()
        rx_of = [self.O.indices[self.O.indptr[i]:self.O.indptr[i + 1]] for i in range(len(self.species))]
        val = np.exp(self.OT @ self._log_c(counts, V))
        last, expo = np.full(len(self.reactions), t_frames[0]), np.zeros(len(self.reactions))
        times = ev["t_ps"].to_numpy()
        order = np.argsort(times, kind="stable")
        k, i, mismatch = 1, 0, 0.0
        while i < len(order):
            t = times[order[i]]
            if t <= t_frames[0]:
                i += 1
                continue
            while k < len(t_frames) and t_frames[k] < t:
                mismatch = max(mismatch, float(np.abs(counts - stored[k]).max()))
                k += 1
            change = Counter()
            while i < len(order) and times[order[i]] == t:  # the events of one 12 fs frame together
                change.subtract(Counter(ev["reactants"].iat[order[i]].split(" + ")))
                change.update(Counter(ev["products"].iat[order[i]].split(" + ")))
                i += 1
            moved = [self.index[f] for f, d in change.items() if d]
            for f, d in change.items():
                counts[self.index[f]] += d
            if moved:
                J = np.unique(np.concatenate([rx_of[s] for s in moved]))
                expo[J] += val[J] * (t - last[J])
                last[J] = t
                val[J] = np.exp(self.OT[J] @ self._log_c(counts, V))
        for k in range(k, len(t_frames)):
            mismatch = max(mismatch, float(np.abs(counts - stored[k]).max()))
        expo += val * (t_frames[-1] - last)
        return expo * V, mismatch

    def fit_rates(self, runs):
        """runs [(T, w, V, ev)]: pooled k per temperature, then Arrhenius (or zero barrier). Returns the event
        count per reaction."""
        num, den = {}, {}
        for T, w, V, ev in runs:
            expo, mismatch = self.replay_exposure(w, V, ev)
            if mismatch:
                raise ValueError(f"replayed counts differ from a stored species frame by {mismatch:g}")
            num[T] = num.get(T, 0) + self.counts(ev)
            den[T] = den.get(T, 0) + expo
        Ts = np.array(sorted(num))
        n = np.stack([num[T] for T in Ts])
        k = n / np.maximum(np.stack([den[T] for T in Ts]), 1e-300)
        for j in range(len(self.reactions)):
            seen = n[:, j] > 0
            if seen.sum() >= 2:
                slope, icept = np.polyfit(1.0 / Ts[seen], np.log(k[seen, j]), 1, w=np.sqrt(n[seen, j]))
                self.lnA[j], self.Ea[j] = icept, -slope * R_KJ
            elif seen.any():
                self.lnA[j], self.Ea[j] = np.log(k[seen, j][0]), 0.0
        return n.sum(0)

    def rhs(self, c, T):
        """dc/dt and the rates r at concentrations c (per nm3)."""
        lnr = self.lnA - self.Ea / (R_KJ * T) + self.OT @ np.log(np.maximum(c, 1e-30))
        r = np.exp(np.minimum(lnr, 50.0))
        return self.N @ r, r

    def integrate(self, c0, T, t_eval, rtol=1e-4, atol=1e-10):
        """Mean-field ODEs with BDF and the sparse Jacobian N diag(r) O^T diag(1/c)."""
        def jac(t, c):
            r = self.rhs(c, T)[1]
            return (self.N @ sp.diags(r) @ self.OT @ sp.diags(1.0 / np.maximum(c, 1e-30))).tocsc()
        return solve_ivp(lambda t, c: self.rhs(c, T)[0], (t_eval[0], t_eval[-1]), c0, method="BDF",
                         t_eval=t_eval, jac=jac, rtol=rtol, atol=atol)

    def predict_classes(self, w, V, T, t_eval, units):
        """Class amounts per nm3 (n_t, n_classes), starting from the first species frame of w."""
        c0 = np.zeros(len(self.species))
        c0[[self.index[f] for f in w.columns]] = w.iloc[0].to_numpy(float) / V
        sol = self.integrate(c0, T, t_eval)
        if not sol.success:
            raise RuntimeError(sol.message)
        return lumping.lump(self.species, sol.y.T, units)[0]

    def subnet(self, keep):
        """The network restricted to reactions `keep`, same species and rates."""
        sub = Network(self.species, [self.reactions[j] for j in keep])
        sub.lnA, sub.Ea = self.lnA[keep], self.Ea[keep]
        return sub

    def window_skeleton(self, runs, frac, window_ps=6.0):
        """RxMD-SN-style skeleton: in every time window of every run, the most frequent reactions that make up
        `frac` of the window's events; returns the indices of their union."""
        j_of = {r: i for i, r in enumerate(self.reactions)}
        keep = set()
        for _, w, _, ev in runs:
            j = np.array([j_of[key(a, b)] for a, b in zip(ev["reactants"], ev["products"])])
            win = ((ev["t_ps"].to_numpy() - w.index[0]) // window_ps).astype(int)
            for b in np.unique(win):
                jj, n = np.unique(j[win == b], return_counts=True)
                o = np.argsort(-n, kind="stable")
                keep |= set(jj[o[: int(np.searchsorted(np.cumsum(n[o]) / n.sum(), frac - 1e-12)) + 1]].tolist())
        return np.array(sorted(keep), dtype=int)
