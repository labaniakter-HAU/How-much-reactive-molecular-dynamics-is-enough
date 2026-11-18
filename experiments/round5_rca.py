# file: experiments/round5_rca.py
"""Round 5 root-cause analysis (design section 16), development data only: the HFO runs of the 250 atm series.
It reads no MEA file. Descriptive tallies of the event tables against the tracked species:

  python -I -X utf8 experiments/round5_rca.py
1. which species are tracked in every run (the round-4 model's species), in some runs, or in none;
2. the share of events by what is known about their reactants;
3. whether the event tables balance the observed change of each tracked species (completeness of the tables);
4. per round-4 species, how much of its production and consumption runs through reactions outside the round-4 model.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

import lab  # noqa: E402

HFO = ROOT.parent / "HFO1"  # development chemistry; MEA is locked (confirmation)


def main():
    rs = lab.runs(HFO)
    ev = {nm: lab.read_events(f) for nm, _, _, f in rs}
    cnt = {nm: lab.read_counts(f) for nm, _, _, f in rs}
    tracked = {nm: set(c) for nm, (_, c) in cnt.items()}
    inter, union = set.intersection(*tracked.values()), set.union(*tracked.values())
    in_runs = Counter(s for t in tracked.values() for s in t)
    all_sp = {s for e in ev.values() for (lhs, rhs) in e for s, _ in lhs + rhs}
    print(f"runs {len(rs)}; species tracked per run {min(map(len, tracked.values()))}-{max(map(len, tracked.values()))}; "
          f"in every run {len(inter)}; in at least one {len(union)}; species in the event tables {len(all_sp)}")
    print("tracked in every run:", " ".join(sorted(inter)))
    part = sorted(((in_runs[s], s) for s in union - inter), reverse=True)
    print("tracked in some runs (runs, species):", " ".join(f"{s}:{k}" for k, s in part))
    for key in ("O2", "H2O", "C4H3F5", "HF", "CO2", "CO", "HO", "H", "O", "H2", "HO2", "H2O2"):
        print(f"   {key}: tracked in {in_runs.get(key, 0)} of {len(rs)} runs")

    # 2. events by what is known about their reactants
    cat = Counter()
    n_rx = defaultdict(set)
    for nm, e in ev.items():
        for (lhs, rhs), n in e.items():
            sp = {s for s, _ in lhs}
            order = sum(nu for _, nu in lhs)
            c = ("order > 3" if order > 3 else "reactants tracked in every run (round-4 model)" if sp <= inter
                 else "reactants tracked in this run" if sp <= tracked[nm] else "reactants tracked in another run"
                 if sp <= union else "a reactant never tracked")
            cat[c] += n
            n_rx[c].add((lhs, rhs))
    tot = sum(cat.values())
    print(f"\nevents (non-identity) {tot}; distinct reactions {len({k for e in ev.values() for k in e})}")
    for c, n in cat.most_common():
        print(f"   {n / tot:6.1%}  {c}  ({len(n_rx[c])} distinct reactions)")

    # 3. balance: observed change of each tracked species against the net change summed over its run's events
    num = den = 0.0
    worst = []
    for nm, e in ev.items():
        t, c = cnt[nm]
        net = Counter()
        for (lhs, rhs), n in e.items():
            for s, nu in lhs:
                net[s] -= nu * n
            for s, nu in rhs:
                net[s] += nu * n
        for s in tracked[nm]:
            obs = c[s][-1] - c[s][0]
            num, den = num + abs(obs - net[s]), den + abs(obs) + 1e-9
            worst.append((abs(obs - net[s]), nm, s, obs, net[s]))
    print(f"\nbalance over all (run, tracked species): sum |observed change - events' net change| / sum |observed change| = {num / den:.3f}")
    for d, nm, s, obs, net in sorted(worst, reverse=True)[:8]:
        print(f"   {nm} {s}: observed {obs:+.0f}, events {net:+.0f}")

    # 4. per round-4 species: gross production and consumption by reaction category (pooled over runs)
    pooled = Counter()
    for e in ev.values():
        pooled.update({k: n for k, n in e.items() if all(s in inter for s, _ in k[0]) and sum(nu for _, nu in k[0]) <= 3})
    anchors = {k for k, _ in pooled.most_common(80)}
    prod, cons = defaultdict(Counter), defaultdict(Counter)
    for nm, e in ev.items():
        for (lhs, rhs), n in e.items():
            sp = {s for s, _ in lhs}
            c = ("anchor" if (lhs, rhs) in anchors else "other round-4 reaction" if (lhs, rhs) in pooled
                 else "outside: reactants tracked in this run" if sp <= tracked[nm] else "outside: a reactant untracked in this run")
            for s, nu in rhs:
                prod[s][c] += nu * n
            for s, nu in lhs:
                cons[s][c] += nu * n
    cats = ("anchor", "other round-4 reaction", "outside: reactants tracked in this run", "outside: a reactant untracked in this run")
    print("\nper round-4 species: gross production | consumption, share by reaction category "
          "(anchor / other round-4 / outside, tracked here / outside, untracked)")
    rows = []
    for s in inter:
        p, c = sum(prod[s].values()), sum(cons[s].values())
        rows.append((p + c, s, p, [prod[s][k] / max(p, 1) for k in cats], c, [cons[s][k] / max(c, 1) for k in cats]))
    for _, s, p, ps, c, cs in sorted(rows, reverse=True):
        print(f"   {s:>10}  prod {p:>7.0f}: " + " ".join(f"{x:4.0%}" for x in ps) + f"   | cons {c:>7.0f}: " + " ".join(f"{x:4.0%}" for x in cs))
    out_p = sum(prod[s][k] for s in inter for k in cats[2:]) / sum(sum(prod[s].values()) for s in inter)
    out_c = sum(cons[s][k] for s in inter for k in cats[2:]) / sum(sum(cons[s].values()) for s in inter)
    print(f"\nround-4 species, all together: {out_p:.0%} of gross production and {out_c:.0%} of gross consumption run through "
          f"reactions outside the round-4 model")
    leave = sum(nu * n for nm, e in ev.items() for (lhs, rhs), n in e.items() if (lhs, rhs) in anchors for s, nu in rhs if s not in inter)
    stay = sum(nu * n for nm, e in ev.items() for (lhs, rhs), n in e.items() if (lhs, rhs) in anchors for s, nu in rhs if s in inter)
    print(f"anchor products: {leave / (leave + stay):.0%} of product molecules are species outside the round-4 model (they leave it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
