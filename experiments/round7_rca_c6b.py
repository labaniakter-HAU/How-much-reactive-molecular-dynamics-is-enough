# file: experiments/round7_rca_c6b.py
"""Round 7, C6 root cause at the reaction level (design section 18), on SPENT round-5 confirmation seeds only.
Rebuilds each missed failure's campaign exactly as round5_flag.synthetic_one does (copied: that file is hash-frozen),
checks the stored rms is reproduced, and shows which reactions carry the realized error against the weight and
variance the revised flag gave them.

  python -I -X utf8 experiments/round7_rca_c6b.py --max 20 --workers 2
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

OUT = ROOT / "results" / "round5"


def rebuild(seed, T_STAR, idx):
    """-> campaign idx of network seed at T_STAR, with everything the flag and the label saw (as synthetic_one)."""
    from round2_synthetic import CLIP
    from round7_c6 import campaign_state, network_setup
    net, data, T_of, A, g0, truth, w, camps = network_setup(seed, T_STAR)
    camp = camps[idx]
    flux, lines, mean_re, sd_re = campaign_state(net, data, T_of, A, g0, T_STAR, camp)
    err = np.minimum(np.abs(mean_re - truth), CLIP)
    return camp, data, T_of, A, flux, truth, w, err, sd_re, lines


def one(args):
    import foresight
    import round5_scores as rs
    from round5_cache_c6 import record
    seed, T_STAR, idx, stored = args
    camp, data, T_of, A, flux, truth, w, err, sd_re, lines = rebuild(seed, T_STAR, idx)
    rms = float(np.sqrt((w * err ** 2).sum() / w.sum()))
    assert abs(rms - stored["rms"]) < 1e-6 * max(1.0, rms), (seed, idx, rms, stored["rms"])
    rec = record(camp, data, T_of, A, T_STAR, flux, rms, 1.0, f"{seed}/{idx}", {"sd_c4": sd_re, "seen": lines["seen"]})
    comp = rs.components(rec)
    var = comp["linevar"] * comp["phi"] + comp["g2_used"] * comp["chord2"] + comp["B2"] * comp["flat2"]
    var = np.where(comp["exposed"], var, foresight.CLIP2)
    f = np.where(np.isfinite(flux) & (flux > 0), flux, 0.0)
    share = w * err ** 2 / (w * err ** 2).sum()
    top = np.argsort(share)[::-1][:3]
    return {"seed": seed, "T": T_STAR, "idx": idx, "rms": rms, "P": stored["P"], "old": stored["old"], "temps": stored["temps"],
            "top": [{"j": int(j), "share_err2": float(share[j]), "true_flux_share": float(w[j] / w.sum()),
                     "pred_flux_share": float(f[j] / f.sum()) if f.sum() > 0 else float("nan"), "err": float(err[j]),
                     "pred_sd": float(np.sqrt(var[j])), "sd_c4": float(sd_re[j]), "seen": int(lines["seen"][j]),
                     "events": int(rec["n"][:, j].sum())} for j in top]}


def main():
    import foresight
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = {"possible": [], "impossible": []}
    for p in sorted(OUT.glob("confirm_flag_syn_*_T*.jsonl")):
        kind = "possible" if "_possible_" in p.name else "impossible" if "_impossible_" in p.name else None
        if kind is None:
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            assert 40000 <= r["seed"] < 43000, "spent round-5 confirmation seeds only"
            for i, c in enumerate(r["campaigns"]):
                if c["rms"] > foresight.CUT and not foresight.flagged(c["P"]):
                    jobs[kind].append((r["seed"], r["target"], i, c))
    todo = jobs["possible"] + jobs["impossible"][:: max(1, len(jobs["impossible"]) // a.max)][: a.max]
    with Pool(a.workers) as pool:
        res = pool.map(one, todo)
    (ROOT / "results" / "round7").mkdir(exist_ok=True)
    (ROOT / "results" / "round7" / "rca_c6b.json").write_text(json.dumps(res, indent=1))
    for r in res:
        print(f"seed {r['seed']} T {r['T']:g} camp {r['idx']} temps {r['temps']}: rms/ln2 {r['rms'] / foresight.CUT:.2f}, P/ln2 {r['P'] / foresight.CUT:.2f}, old {r['old']}")
        for t in r["top"]:
            print(f"     rx {t['j']:>3}: {t['share_err2']:.0%} of err^2 | true flux {t['true_flux_share']:.1%}, predicted {t['pred_flux_share']:.1%} | "
                  f"err {t['err']:.2f}, flag sd {t['pred_sd']:.2f}, C4 sd {t['sd_c4']:.2f} | temps seen {t['seen']}, events {t['events']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
