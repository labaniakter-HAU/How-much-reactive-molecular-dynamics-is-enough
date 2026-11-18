# file: experiments/round7_rca_c6.py
"""Round 7, root cause of the round-5 C6 synthetic failure (design section 18). Descriptive only: reads the SPENT round-5
confirmation blocks (seeds 40000-42999, already scored once) and profiles the failures the revised flag missed, by kind,
sub-kind, campaign type, temperatures and target. Nothing here selects a method.

  python -I -X utf8 experiments/round7_rca_c6.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import foresight  # noqa: E402

OUT = ROOT / "results" / "round5"
L2 = foresight.CUT
TYPES = ["full"] + [f"{b}% {t}" for b in (10, 25, 50) for t in ("greedy", "hot", "uniform", "rand", "rand", "rand")]


def main():
    rows = []
    for p in sorted(OUT.glob("confirm_flag_syn_*_T*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            for i, c in enumerate(r["campaigns"]):
                rows.append({**c, "seed": r["seed"], "kind": r["kind"], "T": r["target"], "type": TYPES[i]})
    print(f"campaigns {len(rows)}; network kinds {Counter(r['kind'] for r in rows)}")
    for kind in ("possible", "curved", "overdispersed", "both", "lumped"):
        rs = [r for r in rows if r["kind"] == kind]
        bad = [r for r in rs if r["rms"] > L2]
        miss = [r for r in bad if not foresight.flagged(r["P"])]
        fa = [r for r in rs if r["rms"] <= L2 and foresight.flagged(r["P"])]
        print(f"\n== {kind}: campaigns {len(rs)}, failures {len(bad)}, missed {len(miss)}, false alarms {len(fa)}")
        if not bad:
            continue
        ratio = np.array([r["P"] / L2 for r in miss])
        if len(ratio):
            print(f"   missed: P/ln2 quartiles {np.percentile(ratio, [0, 25, 50, 75, 100]).round(2)}; realized rms/ln2 median {np.median([r['rms'] / L2 for r in miss]):.2f}")
            print(f"   missed by type {Counter(r['type'] for r in miss).most_common(8)}")
            print(f"   missed by temps {sorted(Counter(r['temps'] for r in miss).items())}; by target {sorted(Counter(r['T'] for r in miss).items())}")
            print(f"   missed and caught by the old flag: {sum(r['old'] for r in miss)} of {len(miss)}")
        print(f"   all failures by temps {sorted(Counter(r['temps'] for r in bad).items())}")
        for lo, hi in ((0, .5), (.5, .75), (.75, 1.0), (1.0, 1.5), (1.5, np.inf)):
            sel = [r for r in rs if lo <= r["P"] / L2 < hi]
            if sel:
                print(f"   P/ln2 in [{lo}, {hi}): campaigns {len(sel)}, failure rate {np.mean([r['rms'] > L2 for r in sel]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
