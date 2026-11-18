# file: experiments/round7_rca_c6d.py
"""Round 7, C6 root cause (design section 18), SPENT round-5 seeds only: would the method's anchor rule (fit.anchors:
the most frequent types covering 95 % of counted events, at most 80), applied to the target's training runs as on
real data, include the reaction that carries each missed failure? The synthetic test bed scores all 27 reactions.

  python -I -X utf8 experiments/round7_rca_c6d.py --workers 3
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402


def one(args):
    from round7_c6 import network_setup
    seed, T, idx, j = args
    net, data, T_of, A, g0, truth, w, camps = network_setup(seed, T)
    n = sum(data[r][L][0] for r, L in camps[0].items())  # campaign 0: every training run at full length
    order = np.argsort(-n, kind="stable")
    order = order[n[order] > 0]
    cum = np.cumsum(n[order]) / n.sum()
    anchors = order[: min(80, int(np.searchsorted(cum, 0.95 - 1e-12)) + 1)]
    return {"seed": seed, "T": T, "idx": idx, "culprit": j, "anchors": len(anchors), "culprit_is_anchor": bool(j in anchors),
            "culprit_event_share": float(n[j] / n.sum()), "culprit_target_flux_share": float(w[j] / w.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    res = json.loads((ROOT / "results" / "round7" / "rca_c6c.json").read_text())
    with Pool(a.workers) as pool:
        out = pool.map(one, [(r["seed"], r["T"], r["idx"], r["culprit"]["j"]) for r in res])
    (ROOT / "results" / "round7" / "rca_c6d.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(f"seed {o['seed']} T {o['T']:g} camp {o['idx']}: culprit rx {o['culprit']} {'IN' if o['culprit_is_anchor'] else 'OUT of'} "
              f"the anchors ({o['anchors']} of 27); training-event share {o['culprit_event_share']:.2%}, target-flux share {o['culprit_target_flux_share']:.1%}")
    print(f"culprits inside the anchor set: {sum(o['culprit_is_anchor'] for o in out)} of {len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
