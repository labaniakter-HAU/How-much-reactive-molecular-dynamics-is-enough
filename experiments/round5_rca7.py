# file: experiments/round5_rca7.py
"""Round 5 root-cause analysis, part 7 (HFO only; no MEA file is read): where does the concentration fit gain over
counted rates? Per fold and scored species, the normalized RMSE of the counted and of the fitted model on the held-out
runs, and the fit's largest rate changes (ln k at the held-out temperature) with the species each reaction touches.

  python -I -X utf8 experiments/round5_rca7.py --anchors 160 --workers 3
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import closure  # noqa: E402
from metrics import normalized_rmse  # noqa: E402
from round5_dev_c2 import HFO, fit_models, read  # noqa: E402

OUT = ROOT / "results" / "round5"


def fold(args):
    T_star, anchors = args
    torch.set_num_threads(1)
    ev, cnt, T_of = read(HFO)
    train, test = sorted(r for r in T_of if T_of[r] != T_star), sorted(r for r in T_of if T_of[r] == T_star)
    tab = closure.Tables(ev, cnt, T_of, train, 1, anchors)
    A = tab.A
    lines = tab.lines(train, A)
    at, final, m0 = fit_models(tab, cnt, train, lines, A, True)
    fitted = at(T_star)
    per = {}
    for name, m in (("counted", tab.model(lines, A, T_star)), ("fit", fitted)):
        for r in test:
            g = tab.group({r: cnt[r]}, [r])
            c, _ = closure.predict(m, g, extents=False)
            obs = g.obs.numpy()
            for i in tab.score:
                if np.abs(obs[:, 0, i]).max() > 0:
                    per.setdefault(tab.M[i], {"counted": [], "fit": [], "mean": []})[name].append(normalized_rmse(c[:, :, [i]], obs[:, :, [i]]))
                    if name == "counted":
                        per[tab.M[i]]["mean"].append(float(obs[:, 0, i].mean()))
    with torch.no_grad():
        k_c = (lines["a"] - lines["b"] * 1000.0 / T_star)
        k_f = (fitted.lnA - 100.0 * fitted.Ea / (8.314462618e-3 * T_star)).numpy()
    d = k_f - k_c
    n_at = sum(tab.n[r][A] for r in test)
    order = np.argsort(-np.abs(d) * np.sqrt(n_at + 1))[:15]
    moves = [{"reactants": "+".join(f"{nu}{s}" if nu > 1 else s for s, nu in tab.rx[A[j]]), "dlnk": float(d[j]),
              "test_events": float(n_at[j]), "anchor": tab.rx[A[j]] in tab.anchor_set} for j in order]
    return {"T": T_star, "per_species": per, "moves": moves, "fit_loss": final,
            "dlnk_event_weighted_rms": float(np.sqrt((n_at * d ** 2).sum() / n_at.sum()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=160)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        res = pool.map(fold, [(T, a.anchors) for T in (2500.0, 2750.0, 3000.0, 3250.0, 3500.0)])
    (OUT / f"dev_rca7_hfo_{a.anchors}.json").write_text(json.dumps(res))
    agg = {}
    for r in res:
        for s, v in r["per_species"].items():
            for k in ("counted", "fit", "mean"):
                agg.setdefault(s, {"counted": [], "fit": [], "mean": []})[k] += v[k]
    tot_c = np.mean([x for v in agg.values() for x in v["counted"]])
    tot_f = np.mean([x for v in agg.values() for x in v["fit"]])
    print(f"mean over (fold, run, species): counted {tot_c:.4f}, fit {tot_f:.4f}, ratio {tot_c / tot_f:.3f}")
    gain = sorted(((np.sum(v["counted"]) - np.sum(v["fit"]), s, np.mean(v["counted"]), np.mean(v["fit"]), np.mean(v["mean"])) for s, v in agg.items()), reverse=True)
    total_gain = sum(g for g, *_ in gain)
    print("species by the fit's gain (share of the total gain): counted / fit error, mean count")
    for g, s, c, f, m in gain:
        print(f"   {s:>10} {g / total_gain:6.1%}  {c:.3f} / {f:.3f}  mean count {m:8.1f}")
    for r in res:
        print(f"\nfold {r['T']:.0f} K: event-weighted RMS of the fit's ln k change at T* {r['dlnk_event_weighted_rms']:.2f}; largest changes:")
        for mv in r["moves"][:10]:
            print(f"   {mv['reactants']:>22} {'anchor' if mv['anchor'] else 'tail  '} d ln k {mv['dlnk']:+.2f}  test events {mv['test_events']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
