# file: src/lab.py
"""The lab's HFO/MEA SCWO ReaxFF runs (design section 14), read-only from ../HFO1 and ../MEA1.

Per run: reaction totals over 0-200 ps (1.xlsx without a header row; MEA 2500 K without O2 has 1.txt, tab-separated)
and molecule counts every 0.1 ps (1.csv, the run's ~50 most common species). No box volume exists, so exposure is
the time integral of prod N^nu in molecule counts ("count units"), not of concentrations.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


def side(s):
    """'2 HO + H2O' -> (('H2O', 1), ('HO', 2))."""
    nu = {}
    for term in s.split("+"):
        coef, sp = re.fullmatch(r"\s*(?:(\d+)\s+)?(\S+)\s*", term).groups()
        nu[sp] = nu.get(sp, 0) + int(coef or 1)
    return tuple(sorted(nu.items()))


def read_events(run):
    """Run folder -> {(reactants, products): count}; identity events (same species on both sides) dropped."""
    run = Path(run)
    tab = (pd.read_excel(run / "1.xlsx", header=None) if (run / "1.xlsx").exists()
           else pd.read_csv(run / "1.txt", sep="\t", header=None, encoding="utf-8"))
    out = {}
    for rx, n in zip(tab[0], tab[1]):
        lhs, rhs = (side(s) for s in str(rx).split("→"))
        if lhs != rhs:
            out[(lhs, rhs)] = out.get((lhs, rhs), 0) + int(n)
    return out


def read_counts(run):
    """Run folder -> t (ps), {species: molecule counts}. The last row (t = 200 ps) of every run holds the energy only,
    so rows without species counts are dropped; any other gap is an error."""
    d = pd.read_csv(Path(run) / "1.csv")
    d = d.dropna(subset=d.columns[2:], how="all")
    assert not d.iloc[:, 2:].isna().any().any(), f"missing counts in {run}"
    return d.iloc[:, 0].to_numpy(float), {c.split(" / ")[0]: d[c].to_numpy(float) for c in d.columns[2:]}


def exposure(t, counts, reactants):
    """Integral of prod N_s^nu_s over the run (trapezoid), molecules^order x ps."""
    return float(np.trapezoid(np.prod([counts[s] ** nu for s, nu in reactants], axis=0), t))


def runs(root):
    """The 250 atm temperature series -> [(name, T, O2, folder)]; name '<T>K_O2_<nnn>', so O2 = 0 sorts first."""
    out = []
    for f in sorted(Path(root).glob("250atm/*/*/1.csv")) + sorted(Path(root).glob("2750K_250atm/*/1.csv")):
        T = 2750.0 if f.parent.parent.name == "2750K_250atm" else float(f.parent.parent.name)
        o2 = int(m.group(1)) if (m := re.search(r"_(\d+)O2_", f.parent.name)) else 0
        out.append((f"{T:g}K_O2_{o2:03d}", T, o2, f.parent))
    return out


def tables(root):
    """-> T_of {run: T}, reactions [(reactants, products)], data {run: {1: (events, exposure)}} in the design's
    format (one menu length: the whole run). Only reactions whose reactants are tracked in every run's 1.csv, of
    order <= 3 (higher orders are bond-analysis artifacts: one 115-body HFO event)."""
    rs = runs(root)
    ev = {nm: read_events(f) for nm, _, _, f in rs}
    cnt = {nm: read_counts(f) for nm, _, _, f in rs}
    tracked = set.intersection(*(set(c) for _, c in cnt.values()))
    rx = sorted({k for e in ev.values() for k in e
                 if all(s in tracked for s, _ in k[0]) and sum(nu for _, nu in k[0]) <= 3})
    data = {nm: {1: (np.array([ev[nm].get(k, 0) for k in rx], float),
                     np.array([exposure(*cnt[nm], k[0]) for k in rx]))} for nm, *_ in rs}
    return {nm: T for nm, T, _, _ in rs}, rx, data
