# file: experiments/round5_rca2.py
"""Round 5 root-cause analysis, part 2 (design section 16; HFO development runs only, no MEA file is read).
How much flux can an explicit model carry under three structural choices:
  - species: tracked in every run (round 4) or in at least K of a fold's training runs;
  - reaction key: (reactants, products) as in round 4, or the reactant set alone (mass action depends on reactants only);
  - anchors: the top N keys by events.

  python -I -X utf8 experiments/round5_rca2.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

import lab  # noqa: E402

HFO = ROOT.parent / "HFO1"


def main():
    rs = lab.runs(HFO)
    ev = {nm: lab.read_events(f) for nm, _, _, f in rs}
    tracked = {nm: set(lab.read_counts(f)[1]) for nm, _, _, f in rs}
    T_of = {nm: T for nm, T, _, _ in rs}
    for T_star in sorted(set(T_of.values())):
        train = [r for r in T_of if T_of[r] != T_star]
        runs_with = Counter(s for r in train for s in tracked[r])
        temps_with = {s: len({T_of[r] for r in train if s in tracked[r]}) for s in runs_with}
        total = sum(n for r in train for n in ev[r].values())
        print(f"== fold T* = {T_star:g} K: training events {total}")
        for label, M in (("every run (round 4)", set.intersection(*tracked.values())),
                         ("all 4 training temperatures", {s for s, k in temps_with.items() if k >= 4}),
                         (">= 2 training temperatures", {s for s, k in temps_with.items() if k >= 2}),
                         (">= 1 training run", set(runs_with))):
            full, merged, expo = Counter(), Counter(), 0
            for r in train:
                for (lhs, rhs), n in ev[r].items():
                    if sum(nu for _, nu in lhs) <= 3 and all(s in M for s, _ in lhs):
                        full[(lhs, rhs)] += n
                        merged[lhs] += n
                        expo += n * all(s in tracked[r] for s, _ in lhs)
            cov = lambda c, N: sum(n for _, n in c.most_common(N)) / total
            n95 = lambda c: int(np.searchsorted(np.cumsum(sorted(c.values(), reverse=True)) / sum(c.values()), 0.95)) + 1
            print(f"   species tracked in {label}: {len(M)}; events with model reactants {sum(full.values()) / total:.1%} "
                  f"(with exposure in their own run {expo / total:.1%})")
            print(f"      key (reactants, products): {len(full)} reactions; top 80 cover {cov(full, 80):.1%} of all events; 95 % of its own needs {n95(full)}")
            print(f"      key reactant set:          {len(merged)} reactions; top 80 cover {cov(merged, 80):.1%}, top 40 {cov(merged, 40):.1%}; 95 % of its own needs {n95(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
