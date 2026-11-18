# file: experiments/round5_confirm.py
"""Round 5 confirmation (design section 16): the revised methods tested once on the locked real data, the MEA runs of
the 250 atm series, with round 4's folds, campaigns and seeds. Two freezes, each before its own confirmation:
  stage c2c3: C2 and C3, frozen in results/round5/FROZEN_c2c3.json
  stage c6:   C6, frozen in results/round5/FROZEN_c6.json; its synthetic part is experiments/round5_flag.py on seeds
              40000-42999 (outputs results/round5/confirm_flag_syn_*.jsonl)

  python -I -X utf8 experiments/round5_confirm.py --run c2c3   # once per stage; refuses without its freeze
  python -I -X utf8 experiments/round5_confirm.py --run c6
  python -I -X utf8 experiments/round5_confirm.py --score      # once, after both stages and the synthetic blocks
"""
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import closure  # noqa: E402
import foresight  # noqa: E402
import species  # noqa: E402

OUT, CHEM = ROOT / "results" / "round5", "MEA"  # CHEM: the locked confirmation chemistry (a smoke test may set HFO)
TEMPS = (2500.0, 2750.0, 3000.0, 3250.0, 3500.0)
COMMON = [ROOT / "src" / f for f in ("species.py", "design.py", "lab.py", "fit.py", "kinetics.py", "integrate.py", "metrics.py",
                                     "data.py")] + [ROOT / "experiments" / f for f in ("round5_confirm.py", "c5_lab.py")]
CODE = {"c2c3": COMMON + [ROOT / "src" / "closure.py", ROOT / "experiments" / "round5_dev_c2.py"],
        "c6": COMMON + [ROOT / "src" / "foresight.py"] + [ROOT / "experiments" / f for f in ("round5_flag.py", "round4_lab.py",
                                                                                              "round2_confirm.py", "round2_score.py")]}


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def check(stage):
    import c5_lab
    frozen = json.loads((OUT / f"FROZEN_{stage}.json").read_text())  # raises if the stage is not frozen
    assert frozen["code_sha256"] == sha(CODE[stage]), f"{stage}: code changed since the freeze"
    assert frozen["data_sha256"] == sha([c5_lab.MANIFEST]) and c5_lab.MANIFEST.read_text().splitlines()[2:] == c5_lab.manifest(), "data changed"
    return frozen


def c2_c3(T_star, frozen):
    """C2: the counted model against the same model fitted to concentrations; C3: compact against full network."""
    from round5_dev_c2 import evaluate, fit_models, full_network, read
    torch.set_num_threads(1)
    m = frozen["model"]
    ev, cnt, T_of = read(ROOT.parent / f"{CHEM}1")
    train, test = sorted(r for r in T_of if T_of[r] != T_star), sorted(r for r in T_of if T_of[r] == T_star)
    kw = dict(min_temps=1, coverage=m["coverage"], absorb=True, by_T=True, volume=True, t0=m["t0"])
    tab = closure.Tables(ev, cnt, T_of, train, n_anchor=m["anchors"], **kw)
    A = tab.A
    lines = tab.lines(train, A)
    res = {"T": T_star, "species": len(tab.M), "anchors": len(tab.anchor_set), "reactions": int(len(A))}
    res["counted"] = evaluate(tab, cnt, test, lambda r: tab.model(lines, A, T_star), A)
    t0 = time.time()
    at, res["fit_loss"], m0 = fit_models(tab, cnt, train, lines, A, True, m["fit_rounds"])
    res["fit"] = evaluate(tab, cnt, test, lambda r: at(T_star), A)
    res["fit_minutes"] = (time.time() - t0) / 60
    res["Ea_shift_median"] = float(np.median(np.abs(100 * m0.Ea.detach().numpy() - 1000.0 * lines["b"] * 8.314462618e-3)))
    # C3: the full network, and the compact model at most a tenth of its size (the counted model above if it fits)
    res["full"] = full_network(tab, ev, cnt, train, test, T_star, True)
    cap = res["full"]["reactions"] // m["c3_size_factor"]
    if len(A) <= cap:
        res["compact"] = {"reactions": int(len(A)), "error_terms": res["counted"]["error_terms"], "same_as_counted": True}
    else:
        small = closure.compact_tables(ev, cnt, T_of, train, cap, start=m["anchors"], step=10, **kw)
        if small is None:
            res["compact"] = {"reactions": None, "error_terms": None, "same_as_counted": False}
        else:
            ls = small.lines(train, small.A)
            e = evaluate(small, cnt, test, lambda r: small.model(ls, small.A, T_star), small.A)
            res["compact"] = {"reactions": int(len(small.A)), "anchors": len(small.anchor_set), "error_terms": e["error_terms"], "same_as_counted": False}
    return res


def c6(T_star, frozen):
    """C6: the revised flag on round 4's MEA campaigns; flux from round 4's campaign model, label as round 4."""
    import round4_lab as r4
    torch.set_num_threads(1)
    T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup(CHEM, T_star)
    camps = r4.campaigns(CHEM, T_star, T_of, data, menu, A, pilot, test)
    g_test = species.batch(groups, test)
    obs, w = r4.observed(test, data, T_of, A)
    f6 = frozen["flag"]
    rows = []
    for name, camp in camps.items():
        mean, sd, s2, tau2, lines = species.calibrated_re_k(camp, data, T_of, A, T_star)
        _, xi = species.predict(species.model(S, O, lines, A), g_test)
        P = foresight.predicted_error(camp, data, T_of, A, T_star, xi.sum(0), f6["B"], f6["phi"], f6["noise"])[0]
        flagged = foresight.flagged(P, 1.0)  # computed before the label; every training temperature has the same starts
        rms = float(np.sqrt((w * np.minimum(np.abs(mean - obs), r4.CLIP) ** 2).sum() / w.sum()))
        rows.append({"name": name, "P": P, "flagged": flagged, "rms": rms, "bad": bool(rms > foresight.CUT)})
    return {"T": T_star, "campaigns": rows}


def job(spec):
    kind, T = spec
    path = OUT / f"confirm_{kind}_MEA_{T:g}.json"
    if path.exists():
        return f"exists: {path.name}"
    frozen = json.loads((OUT / f"FROZEN_{kind}.json").read_text())
    t0 = time.time()
    res = c2_c3(T, frozen) if kind == "c2c3" else c6(T, frozen)
    res["minutes"] = (time.time() - t0) / 60
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(path)
    return f"{path.name}: {res['minutes']:.1f} min"


def run(stage):
    frozen = check(stage)
    specs = [(stage, T) for T in TEMPS]
    t0 = time.time()
    with Pool(int(frozen["workers"])) as pool:
        for msg in pool.imap_unordered(job, specs):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


def score():
    from round2_score import h7
    from round5_dev_c2 import pooled_error
    b = check("c2c3")["bars"] | check("c6")["bars"]
    assert not (OUT / "score.json").exists(), "scored once"
    c23 = [json.loads((OUT / f"confirm_c2c3_MEA_{T:g}.json").read_text()) for T in TEMPS]
    c6r = [json.loads((OUT / f"confirm_c6_MEA_{T:g}.json").read_text()) for T in TEMPS]
    fire = {k: [x for f in c23 for x in f[k]["firing_terms"]] for k in ("counted", "fit")}
    fc, ff = (float(np.mean(v)) if np.isfinite(v).all() else np.inf for v in (fire["counted"], fire["fit"]))
    ec, ef = (pooled_error([x for f in c23 for x in f[k]["error_terms"]]) for k in ("counted", "fit"))
    out = {"C2": {"firing_counted": fc, "firing_fit": ff, "firing_ratio": fc / ff, "error_counted": ec, "error_fit": ef,
                  "error_ratio": ec / ef, "Ea_shift_median_per_fold": [f["Ea_shift_median"] for f in c23]}}
    out["C2"]["pass"] = bool(out["C2"]["firing_ratio"] <= b["C2_firing_ratio"] and out["C2"]["error_ratio"] <= b["C2_error_ratio"])
    if any(f["compact"]["error_terms"] is None for f in c23):
        out["C3"] = {"pass": False, "reason": "no compact model within a tenth of the full network's size"}
    else:
        ecm, efu = (pooled_error([x for f in c23 for x in f[k]["error_terms"]]) for k in ("compact", "full"))
        sf = min(f["full"]["reactions"] / f["compact"]["reactions"] for f in c23)
        out["C3"] = {"error_compact": ecm, "error_full": efu, "error_ratio": ecm / efu, "size_factor_min": sf,
                     "sizes": [(f["compact"]["reactions"], f["full"]["reactions"]) for f in c23]}
        out["C3"]["pass"] = bool(out["C3"]["error_ratio"] <= b["C3_error_ratio"] and sf >= b["C3_size_factor"])
    rows = [c for f in c6r for c in f["campaigns"]]
    out["C6 real"] = h7([c["bad"] for c in rows], [c["flagged"] for c in rows], b["C6"])
    syn = [json.loads(x) for p in sorted(OUT.glob("confirm_flag_syn_*.jsonl")) for x in p.read_text().splitlines()]
    srows = [c for r in syn for c in r["campaigns"]]
    out["C6 synthetic"] = h7([c["rms"] > foresight.CUT for c in srows], [foresight.flagged(c["P"]) for c in srows], b["C6"]) | {"networks": len(syn)}
    (OUT / "score.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        print(f"{k}: {json.dumps(v, default=float)}")
    return 0


def main():
    if "--run" in sys.argv:
        run(sys.argv[sys.argv.index("--run") + 1])
    elif "--score" in sys.argv:
        return score()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
