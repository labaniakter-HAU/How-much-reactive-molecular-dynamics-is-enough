# file: experiments/round5_rca4.py
"""Round 5 root-cause analysis, part 4 (HFO only): where is the counted model's remaining error?
Per fold, 160 merged anchors with closure:
  line     rates from the Arrhenius lines of the training temperatures (the method)
  oracle   rates counted at the held-out temperature itself, from its 4 runs (no extrapolation: the model-structure floor)
  per-run  rates counted in each test run itself
and per scored species: the error of `line` and of `oracle`, to see which species carry the gap.

  python -I -X utf8 experiments/round5_rca4.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import closure  # noqa: E402
import lab  # noqa: E402
import species  # noqa: E402
from metrics import normalized_rmse  # noqa: E402
from round5_dev_c2 import HFO, pooled_error  # noqa: E402


def main():
    rs = lab.runs(HFO)
    ev = {nm: lab.read_events(f) for nm, _, _, f in rs}
    cnt = {nm: lab.read_counts(f) for nm, _, _, f in rs}
    T_of = {nm: T for nm, T, _, _ in rs}
    per_species = {}
    for T_star in sorted(set(T_of.values())):
        train, test = sorted(r for r in T_of if T_of[r] != T_star), sorted(r for r in T_of if T_of[r] == T_star)
        tab = closure.Tables(ev, cnt, T_of, train)
        A = tab.anchors(0.95, 160)
        lines = tab.lines(train, A)

        def const(n, E):  # rate constants without a temperature model
            ok = (E > 0) & (n > 0)
            return {"a": np.where(ok, np.log(np.where(ok, n, 1.0) / np.where(ok, E, 1.0)), lines["a"] - lines["b"] * 1000.0 / T_star), "b": np.zeros(len(A))}
        _, n4, E4 = tab.pooled(test, A)
        variants = {"line": lambda r: lines, "oracle": lambda r: const(n4[0], E4[0]),
                    "per-run": lambda r: const(np.where(np.isfinite(tab.E[r][A]), tab.n[r][A], 0.0), np.nan_to_num(tab.E[r][A]))}
        out = {}
        for name, get in variants.items():
            err = []
            for r in test:
                g = tab.group({r: cnt[r]}, [r])
                c, _ = species.predict(tab.model(get(r), A), g, extents=False)
                gs = closure.scored(g, tab.score)
                err += species.error_terms(c[..., tab.score], gs)
                if name in ("line", "oracle"):
                    obs = gs.obs.numpy()
                    for k, i in enumerate(tab.score):
                        if np.abs(obs[:, 0, k]).max() > 0:
                            per_species.setdefault(tab.M[i], {"line": [], "oracle": []})[name].append(normalized_rmse(c[:, :, [i]], obs[:, :, [k]]))
            out[name] = pooled_error(err)
        d = np.abs(lines["a"] - lines["b"] * 1000.0 / T_star - const(n4[0], E4[0])["a"])
        w = n4[0]
        print(f"{T_star:.0f} K: error with line rates {out['line']:.4f}, oracle rates {out['oracle']:.4f}, per-run rates {out['per-run']:.4f}; "
              f"event-weighted RMS of (line - oracle) ln k {np.sqrt((w * d ** 2).sum() / w.sum()):.3f}", flush=True)
    print("\nper scored species, mean over folds and runs: error with line rates / oracle rates (sorted by the difference)")
    rows = sorted(((np.mean(v["line"]) - np.mean(v["oracle"]), s, np.mean(v["line"]), np.mean(v["oracle"])) for s, v in per_species.items()), reverse=True)
    for d, s, a, b in rows:
        print(f"   {s:>10}: {a:.3f} / {b:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
