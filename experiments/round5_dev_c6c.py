# file: experiments/round5_dev_c6c.py
"""Round 5 development for C6, part 3 (HFO campaigns and butane campaigns; no MEA file is read): the revised score P
(src/foresight.py) and the frozen flag's score per campaign, saved for combining.

  python -I -X utf8 experiments/round5_dev_c6c.py     # -> results/round5/dev_c6c_hfo.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import foresight  # noqa: E402


def main():
    import round4_lab as r4
    res = json.loads((ROOT / "results" / "round5" / "dev_c6_hfo.json").read_text())
    out = []
    for fold in res:
        T_star = fold["T"]
        T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup("HFO", T_star)
        camps = r4.campaigns("HFO", T_star, T_of, data, menu, A, pilot, test)
        for c in fold["campaigns"]:
            sd, f = np.array(c["sd"], float), np.array(c["flux"], float)
            pc4 = float(np.sqrt((f * sd ** 2).sum() / f.sum())) if np.isfinite(sd).all() and np.isfinite(f).all() and f.sum() > 0 else float("inf")
            seen = np.array(c["seen"])
            single = float(f[seen <= 1].sum() / f.sum()) if np.isfinite(f).all() and f.sum() > 0 else float("inf")
            P, g2, s2 = foresight.predicted_error(camps[c["name"]], data, T_of, A, T_star, c["flux"], 10.0)
            out.append({"T": T_star, "name": c["name"], "rms": c["rms"], "temps": c["temps"], "runs": c["runs"], "P": P, "P_c4": pc4,
                        "single": single, "g2": g2, "s2": s2, "tau2": c["tau2"]})
    (ROOT / "results" / "round5" / "dev_c6c_hfo.json").write_text(json.dumps(out))
    print(len(out), "HFO campaigns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
