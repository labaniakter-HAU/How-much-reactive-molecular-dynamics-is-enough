# file: experiments/round7_c6.py
"""Round 7 (design section 18): C6 with the zero-count defect term (src/firth.py): development, freeze, one confirmation.
Labels and C4's prediction are round 5's; only the flag's score changes (P7 instead of round 5's P).

  python -I -X utf8 experiments/round7_c6.py --cache-mea                       # MEA records, computed as round 5 confirmed them
  python -I -X utf8 experiments/round7_c6.py --cache-spent 40000 41000 --target 3100 --workers 11   # every 4th spent seed
  python -I -X utf8 experiments/round7_c6.py --dev                             # development scores and the rule
  python -I -X utf8 experiments/round7_c6.py --freeze
  python -I -X utf8 experiments/round7_c6.py --confirm 46000 46250 --target 3100 --workers 11
  python -I -X utf8 experiments/round7_c6.py --score                           # once
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

OUT = ROOT / "results" / "round7"
R5 = ROOT / "results" / "round5"
L2 = float(np.log(2))
KINDS, TARGETS = {"lumped": 46000, "possible": 47000, "impossible": 48000}, (3100, 3600, 4500, 5000)
OK_KIND = {"lumped": {"lumped"}, "possible": {"possible"}, "impossible": {"curved", "overdispersed", "both"}}
FLAG = json.loads((R5 / "FROZEN_c6.json").read_text())["flag"]  # round 5's frozen B, phi, noise
CODE = [ROOT / "src" / f for f in ("firth.py", "foresight.py", "design.py", "species.py", "synthetic.py", "fit.py", "integrate.py",
                                   "kinetics.py", "data.py")] + \
       [ROOT / "experiments" / f for f in ("round7_c6.py", "round5_flag.py", "round2_synthetic.py", "v0b_reduction.py", "v0c_synthetic.py")]
# The .pkl caches are written and read only by this project's own scripts (local development data), so pickle is safe here.


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def kind7(seed):
    """Network kind by seed: round 7's confirmation layout (design section 18), else round 5's (round5_flag.kind5)."""
    from round5_flag import kind5
    from synthetic import KINDS as NET_KINDS
    if 46000 <= seed < 47000:
        return "lumped"
    if 47000 <= seed < 48000:
        return "possible"
    if 48000 <= seed < 49000:
        return NET_KINDS[1 + seed % 3]
    return kind5(seed)


def network_setup(seed, T_STAR):
    """-> net, data, T_of, A, g0, truth, w, camps: network seed at T_STAR and its 19 campaigns, exactly as
    round5_flag.synthetic_one builds them (copied: that file is hash-frozen)."""
    import torch
    import design
    from data import Group
    from round2_synthetic import BUDGETS
    from synthetic import random_network, run_data, true_lnk
    from v0b_reduction import RX, SPECIES, V, X0
    from v0c_synthetic import RUNS
    torch.set_num_threads(1)
    net = random_network(RX, seed, kind7(seed))
    rng = np.random.default_rng(seed + 100000)
    names = [f"{T}K_{s}" for T, s, _ in RUNS]
    T_of = {nm: float(T) for nm, (T, _, _) in zip(names, RUNS)}
    bins = {nm: b for nm, (_, _, b) in zip(names, RUNS)}
    menu = {nm: [L for L in (50, 100, 200) if L < bins[nm]] + [bins[nm]] for nm in names if T_of[nm] != T_STAR}
    data = {}
    for nm in names:
        data[nm], _ = run_data(net, SPECIES, T_of[nm], bins[nm], X0, V, menu.get(nm, [bins[nm]]), rng, n_sub=4)
    truth = true_lnk(net, T_STAR)
    w = np.exp(truth) * sum(data[nm][bins[nm]][1] for nm in names if T_of[nm] == T_STAR)
    A = np.arange(len(RX))
    g0 = Group(torch.linspace(0, 1.2 * 500, 1001, dtype=torch.float64), torch.as_tensor(np.asarray(X0, float) / V)[None, None],
               torch.tensor([T_STAR], dtype=torch.float64), torch.tensor([V], dtype=torch.float64), None)
    total, pilot, crng = sum(bins[nm] for nm in menu), {"4500K_1" if T_STAR == 5000.0 else "5000K_1": 50}, np.random.default_rng(0)
    camps = [{nm: bins[nm] for nm in menu}]
    for f in BUDGETS:
        Bud = int(f * total)
        cs = [design.greedy(menu, pilot, Bud, T_of, T_STAR), design.hot_only(menu, pilot, Bud, T_of), design.uniform(menu, pilot, Bud, T_of)]
        rand = [design.random_campaign(menu, pilot, Bud, crng) for _ in range(100)]
        camps += cs + rand[:3]
    return net, data, T_of, A, g0, truth, w, camps


def campaign_state(net, data, T_of, A, g0, T_STAR, camp):
    """-> flux, lines, mean_re, sd_re of one campaign, as round5_flag.synthetic_one computes them."""
    import torch
    import design
    import fit
    from synthetic import network
    from v0b_reduction import SPECIES
    _, _, _, _, lines = design.calibrated(camp, data, T_of, A, T_STAR)
    model = network(SPECIES, net["rx"], 1000.0 * 8.314462618e-3 * lines["b"], np.exp(lines["a"]))
    with torch.no_grad():
        flux = fit.simulate(model, g0, n_sub=4)[1][-1, 0].numpy()
    mean_re, sd_re = design.calibrated_re(camp, data, T_of, A, T_STAR)[:2]
    return flux, lines, mean_re, sd_re


def synthetic_rows(seed, T_STAR, records=False):
    """Every campaign of network seed at T_STAR: label rms, round 5's P, round 7's P7, temperatures (or cache records)."""
    import firth
    from round2_synthetic import CLIP
    net, data, T_of, A, g0, truth, w, camps = network_setup(seed, T_STAR)
    rows = []
    for i, camp in enumerate(camps):
        flux, lines, mean_re, sd_re = campaign_state(net, data, T_of, A, g0, T_STAR, camp)
        rms = float(np.sqrt((w * np.minimum(np.abs(mean_re - truth), CLIP) ** 2).sum() / w.sum()))
        P7, P5 = firth.predicted_error7(camp, data, T_of, A, T_STAR, flux, FLAG["B"], FLAG["phi"], FLAG["noise"])
        row = {"rms": rms, "P5": P5, "P7": P7, "temps": len({T_of[r] for r, L in camp.items() if L})}
        if records:
            from round5_cache_c6 import record
            row = record(camp, data, T_of, A, T_STAR, flux, rms, 1.0, f"{seed}/{i}",
                         {"sd_c4": sd_re, "seen": lines["seen"], "seed": seed, "kind": kind7(seed), **row})
        rows.append(row)
    return {"seed": seed, "kind": kind7(seed), "target": T_STAR, "campaigns": rows}


def spent_one(args):
    seed, T_STAR = args
    assert 40000 <= seed < 43000 and (R5 / "score_syn.json").exists(), "spent round-5 seeds only, after their scoring"
    return synthetic_rows(seed, T_STAR, records=True)


def mea_fold(T_star):
    """Records of C5's MEA campaigns at T_star: flux, label and round 5's P exactly as round5_confirm.c6, plus P7."""
    import torch
    import firth
    import round4_lab as r4
    import species
    from round5_cache_c6 import record
    torch.set_num_threads(1)
    T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup("MEA", T_star)
    camps = r4.campaigns("MEA", T_star, T_of, data, menu, A, pilot, test)
    g_test = species.batch(groups, test)
    obs, w = r4.observed(test, data, T_of, A)
    out = []
    for name, camp in camps.items():
        mean, sd, s2, tau2, lines = species.calibrated_re_k(camp, data, T_of, A, T_star)
        _, xi = species.predict(species.model(S, O, lines, A), g_test)
        flux = np.asarray(xi.sum(0), float)
        P7, P5 = firth.predicted_error7(camp, data, T_of, A, T_star, flux, FLAG["B"], FLAG["phi"], FLAG["noise"])
        rms = float(np.sqrt((w * np.minimum(np.abs(mean - obs), r4.CLIP) ** 2).sum() / w.sum()))
        out.append(record(camp, data, T_of, A, T_star, flux, rms, 1.0, name, {"P5": P5, "P7": P7}))
    return out


def cache_mea(workers):
    with Pool(workers) as pool:
        folds = pool.map(mea_fold, [2500.0, 2750.0, 3000.0, 3250.0, 3500.0])
    for T, recs in zip((2500, 2750, 3000, 3250, 3500), folds):
        stored = {c["name"]: c for c in json.loads((R5 / f"confirm_c6_MEA_{T}.json").read_text())["campaigns"]}
        dP = max(abs(r["P5"] - stored[r["name"]]["P"]) for r in recs)
        drms = max(abs(r["rms"] - stored[r["name"]]["rms"]) for r in recs)
        print(f"MEA {T} K: {len(recs)} campaigns; max |P5 - round 5's P| {dP:.2e}, max |rms - round 5's| {drms:.2e}")
        assert len(recs) == len(stored) and dP < 1e-9 and drms < 1e-9, "MEA records do not reproduce round 5"
        pickle.dump(recs, open(OUT / f"cache_mea_{T}.pkl", "wb"))


def cache_spent(a, b, T, workers):
    seeds = list(range(a, b, 4))
    with Pool(workers) as pool:
        nets = pool.map(spent_one, [(s, T) for s in seeds])
    kind = "lumped" if a < 41000 else "possible" if a < 42000 else "impossible"
    stored = {r["seed"]: r for r in map(json.loads, (R5 / f"confirm_flag_syn_{kind}_T{T:g}.jsonl").read_text().splitlines())}
    dP = max(abs(c["P5"] - s["P"]) for n in nets for c, s in zip(n["campaigns"], stored[n["seed"]]["campaigns"]))
    drms = max(abs(c["rms"] - s["rms"]) for n in nets for c, s in zip(n["campaigns"], stored[n["seed"]]["campaigns"]))
    print(f"spent {a}-{b} T {T:g}: {len(nets)} networks; max |P5 - stored P| {dP:.2e}, max |rms - stored| {drms:.2e}")
    assert dP < 1e-9 and drms < 1e-9, "spent records do not reproduce round 5"
    pickle.dump([c for n in nets for c in n["campaigns"]], open(OUT / f"cache_spent_{a}_{b}_T{T:g}.pkl", "wb"))


def scores(c):
    """-> (P5, P7) of a cached record: online values when the record has them, else round 5's component path."""
    import firth
    import round5_scores as rs
    from round6_scores_c6 import per_anchor
    comp = rs.components(c)
    var, f = per_anchor(c, comp)
    if f.sum() <= 0:
        return np.inf, np.inf
    P5 = float(np.sqrt((f * var).sum() / f.sum()))
    d2 = firth.defect2_arrays(c["n"], c["E"], c["Ts"], c["runs"], comp["s2"], c["T"])
    P7 = float(np.sqrt(P5 ** 2 + (f * d2).sum() / f.sum()))
    if "P5" in c:  # computed online (spent seeds, MEA): the path the confirmation uses; check the component path agrees
        assert abs(c["P5"] - P5) < 1e-6 * max(1.0, P5) and abs(c["P7"] - P7) < 1e-6 * max(1.0, P7), (c["name"], c["P5"], P5, c["P7"], P7)
        return c["P5"], c["P7"]
    return P5, P7


def dev():
    import foresight
    from round2_score import h7
    bar = json.loads((R5 / "FROZEN_c6.json").read_text())["bars"]["C6"]
    load = lambda pattern, d=R5: [r for p in sorted(d.glob(pattern)) for r in pickle.load(open(p, "rb"))]
    syn = load("cache_c6_syn_*.pkl") + load("cache_spent_*.pkl", OUT)
    sets = {"HFO": load("cache_c6_hfo.pkl"), "butane": load("cache_c6_butane.pkl"), "MEA": load("cache_mea_*.pkl", OUT)}
    sets.update({f"synthetic {k}": [c for c in syn if c["kind"] in ok] for k, ok in OK_KIND.items()})
    out = {}
    for name, rows in sets.items():
        P = [scores(c) for c in rows]
        bad = [c["rms"] > L2 for c in rows]
        out[name] = {"campaigns": len(rows), "round 5": h7(bad, [foresight.flagged(p5, c["dom"]) for (p5, _), c in zip(P, rows)], bar),
                     "round 7": h7(bad, [foresight.flagged(p7, c["dom"]) for (_, p7), c in zip(P, rows)], bar)}
    std = [c for c in syn if c["kind"] in OK_KIND["possible"] | OK_KIND["impossible"]]
    P = [scores(c) for c in std]
    out["synthetic standard, pooled"] = {"campaigns": len(std), "round 7": h7([c["rms"] > L2 for c in std], [foresight.flagged(p7, c["dom"]) for (_, p7), c in zip(P, std)], bar)}
    harm = {k: v["round 7"]["accuracy"] >= v["round 5"]["accuracy"] for k, v in out.items() if "round 5" in v}
    gain = out["synthetic standard, pooled"]["round 7"]["untrustworthy_caught"] >= 0.80
    out["rule"] = {"no harm": harm, "gain (pooled standard catch >= 80 %)": bool(gain), "adopt": bool(all(harm.values()) and gain)}
    (OUT / "dev_c6.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        if k == "rule":
            print(f"rule: {v}")
            continue
        r5 = v.get("round 5")
        r7 = v["round 7"]
        s5 = f"{r5['accuracy']:.3f} / caught {r5['untrustworthy_caught']:.2f} / passed {r5['trustworthy_passed']:.3f}" if r5 else "—"
        print(f"{k:>28} ({v['campaigns']} campaigns, {r7['untrustworthy']} failures): round 5 {s5} | round 7 {r7['accuracy']:.3f} / "
              f"caught {r7['untrustworthy_caught']:.2f} / passed {r7['trustworthy_passed']:.3f}")
    return out


def check():
    frozen = json.loads((OUT / "FROZEN.json").read_text())  # raises if round 7 is not frozen
    assert frozen["code_sha256"] == sha(CODE), "code changed since the freeze"
    return frozen


def freeze():
    res = json.loads((OUT / "dev_c6.json").read_text())
    assert res["rule"]["adopt"], "the development rule failed: round 7 changes nothing"
    path = OUT / "FROZEN.json"
    assert not path.exists(), "frozen once"
    path.write_text(json.dumps({"frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "flag": {**FLAG, "defect": "firth (src/firth.py)"},
                                "code_sha256": sha(CODE), "development rule": res["rule"]}, indent=1))
    print(path.read_text())


def confirm_one(args):
    """Confirmation seeds 46000-48999 only, and only once round 7 is frozen."""
    seed, T_STAR = args
    assert 46000 <= seed < 49000 and (OUT / "FROZEN.json").exists()
    return synthetic_rows(seed, T_STAR)


def confirm(a, b, T, workers):
    check()
    seeds = list(range(a, b))
    kind = next(k for k, s0 in KINDS.items() if s0 <= seeds[0] < s0 + 1000)
    assert all(KINDS[kind] <= s < KINDS[kind] + 1000 for s in seeds)
    path = OUT / f"confirm_syn_{kind}_T{T:g}.jsonl"
    assert not path.exists() or len(path.read_text().splitlines()) < len(seeds), "run once"
    with Pool(workers) as pool, open(path, "w") as f:
        for r in pool.imap_unordered(confirm_one, [(s, T) for s in seeds]):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"{kind} {T:g}: {len(seeds)} networks")


def score():
    import foresight
    from round2_score import h7
    frozen = check()
    assert not (OUT / "score.json").exists(), "scored once"
    bar = json.loads((R5 / "FROZEN_c6.json").read_text())["bars"]["C6"]
    out, allrows = {"frozen": frozen["frozen"]}, []
    for kind, s0 in KINDS.items():
        rows = []
        for i, T in enumerate(TARGETS):
            nets = [json.loads(x) for x in (OUT / f"confirm_syn_{kind}_T{T}.jsonl").read_text().splitlines()]
            assert sorted(r["seed"] for r in nets) == list(range(s0 + 250 * i, s0 + 250 * (i + 1))), f"{kind} {T}: seeds"
            assert all(r["kind"] in OK_KIND[kind] and r["target"] == T for r in nets), f"{kind} {T}: kinds or targets"
            rows += [c for r in nets for c in r["campaigns"]]
        bad = [c["rms"] > L2 for c in rows]
        out[kind] = h7(bad, [foresight.flagged(c["P7"]) for c in rows], bar)
        out[kind]["round 5 flag (descriptive)"] = h7(bad, [foresight.flagged(c["P5"]) for c in rows], bar)
        allrows += rows
    out["pooled (descriptive)"] = h7([c["rms"] > L2 for c in allrows], [foresight.flagged(c["P7"]) for c in allrows], bar)
    (OUT / "score.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-mea", action="store_true")
    ap.add_argument("--cache-spent", type=int, nargs=2)
    ap.add_argument("--confirm", type=int, nargs=2)
    ap.add_argument("--target", type=float)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cache_mea:
        cache_mea(a.workers)
    if a.cache_spent:
        cache_spent(*a.cache_spent, a.target, a.workers)
    if a.dev:
        dev()
    if a.freeze:
        freeze()
    if a.confirm:
        confirm(*a.confirm, a.target, a.workers)
    if a.score:
        score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
