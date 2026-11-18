# file: experiments/round8.py
"""Round 8 (design section 19): C4 with the Firth line (firth.use_firth), and C6 on its labels. Every quantity is
computed with the current line (v1) and the Firth line (v2) on identical data and campaigns; v1 must reproduce the
stored round-5 and round-6 results exactly.

  python -I -X utf8 experiments/round8.py --dev-real --workers 9     # HFO and MEA folds (C4, C6), butane (C4, C6)
  python -I -X utf8 experiments/round8.py --dev-syn --workers 11     # every 10th spent seed of 40000-45999
  python -I -X utf8 experiments/round8.py --dev                      # development scores and the rule
  python -I -X utf8 experiments/round8.py --freeze
  python -I -X utf8 experiments/round8.py --confirm 46000 46250 --target 3100 --workers 11
  python -I -X utf8 experiments/round8.py --score                    # once
"""
import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

OUT = ROOT / "results" / "round8"
L2 = float(np.log(2))
TARGETS = (3100, 3600, 4500, 5000)
KINDS = {"lumped": 46000, "possible": 47000, "impossible": 48000}
OK_KIND = {"lumped": {"lumped"}, "possible": {"possible"}, "impossible": {"curved", "overdispersed", "both"}}
FLAG = json.loads((ROOT / "results" / "round5" / "FROZEN_c6.json").read_text())["flag"]  # round 5's frozen B, phi, noise
BAR_C6 = json.loads((ROOT / "results" / "round5" / "FROZEN_c6.json").read_text())["bars"]["C6"]
CODE = [ROOT / "src" / f for f in ("firth.py", "design.py", "species.py", "foresight.py", "synthetic.py", "fit.py", "integrate.py",
                                   "kinetics.py", "data.py")] + \
       [ROOT / "experiments" / f for f in ("round8.py", "round7_c6.py", "round6_dev_c4.py", "round5_flag.py", "round2_synthetic.py",
                                           "v0b_reduction.py", "v0c_synthetic.py")]


def line(v):
    import firth
    return firth.use_firth() if v == "v2" else nullcontext()


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def target_of(seed):
    return float(TARGETS[((seed - 40000) % 1000) // 250])


def network8(seed, T_STAR):
    """One synthetic network: C4's z on the full campaign (round 6's chord variance) and C6 on round 5's 19 campaigns,
    each with v1 and v2. The flag's flux comes from design.calibrated (Poisson line), the same under v1 and v2."""
    import design
    import foresight
    from round2_synthetic import CLIP
    from round6_dev_c4 import variances
    from round7_c6 import campaign_state, kind7, network_setup
    net, data, T_of, A, g0, truth, w, camps = network_setup(seed, T_STAR)
    _, n, _ = design.pool(camps[0], data, T_of)
    keep = n.sum(0) >= 10
    out = {"seed": seed, "kind": kind7(seed), "target": T_STAR, "z": {}, "c6": []}
    for v in ("v1", "v2"):
        with line(v):
            mean, s2, sds = variances(camps[0], data, T_of, A, T_STAR)
        out["z"][v] = ((mean - truth) / sds["chord"])[keep].round(4).tolist()
    for camp in camps:
        flux, _, mean1, _ = campaign_state(net, data, T_of, A, g0, T_STAR, camp)
        row = {"temps": len({T_of[r] for r, L in camp.items() if L})}
        for v in ("v1", "v2"):
            with line(v):
                mean = mean1 if v == "v1" else design.calibrated_re(camp, data, T_of, A, T_STAR)[0]
                row[f"P_{v}"] = foresight.predicted_error(camp, data, T_of, A, T_STAR, flux, FLAG["B"], FLAG["phi"], FLAG["noise"])[0]
            row[f"rms_{v}"] = float(np.sqrt((w * np.minimum(np.abs(mean - truth), CLIP) ** 2).sum() / w.sum()))
        out["c6"].append(row)
    return out


def dev_syn_one(seed):
    assert 40000 <= seed < 46000 and (ROOT / "results" / "round6" / "score.json").exists(), "spent seeds only, after round 6's scoring"
    return network8(seed, target_of(seed))


def lab8(chem, T_star):
    """One lab fold: C4's z on every non-full campaign (as round 6) and C6 on every campaign (as round 5), v1 and v2."""
    import torch
    import foresight
    import round4_lab as r4
    import species
    from round6_dev_c4 import variances
    torch.set_num_threads(1)
    T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup(chem, T_star)
    camps = r4.campaigns(chem, T_star, T_of, data, menu, A, pilot, test)
    g_test = species.batch(groups, test)
    obs, w = r4.observed(test, data, T_of, A)
    keep = w >= 10
    out = {"chem": chem, "T": T_star, "z": {"v1": [], "v2": []}, "c6": []}
    for name, camp in camps.items():
        row = {"name": name}
        for v in ("v1", "v2"):
            with line(v):
                if name != "full":
                    mean, s2, sds = variances(camp, data, T_of, A, T_star)
                    out["z"][v] += ((mean - obs) / np.sqrt(sds["chord"] ** 2 + 1.0 / np.maximum(w, 1.0) + s2 / len(test)))[keep].round(5).tolist()
                mean, sd, s2, tau2, lines = species.calibrated_re_k(camp, data, T_of, A, T_star)
                _, xi = species.predict(species.model(S, O, lines, A), g_test)
                row[f"P_{v}"] = foresight.predicted_error(camp, data, T_of, A, T_star, np.asarray(xi.sum(0), float), FLAG["B"], FLAG["phi"], FLAG["noise"])[0]
            row[f"rms_{v}"] = float(np.sqrt((w * np.minimum(np.abs(mean - obs), r4.CLIP) ** 2).sum() / w.sum()))
        out["c6"].append(row)
    return out


def butane_c4(T_star):
    from round6_dev_c4 import butane_target
    out = {"T": T_star, "z": {}}
    for v in ("v1", "v2"):
        with line(v):
            out["z"][v] = butane_target(T_star)["chord"]
    return out


def butane_c6(T_star):
    import round5_scores as rs
    from round5_cache_c6 import butane_target
    from round6_scores_c6 import per_anchor
    out = {"T": T_star, "c6": None}
    for v in ("v1", "v2"):
        with line(v):
            recs = butane_target(T_star)
            P = []
            for c in recs:
                var, f = per_anchor(c, rs.components(c))
                P.append(float(np.sqrt((f * var).sum() / f.sum())) if f.sum() > 0 else float("inf"))
        rows = [{"name": c["name"], f"rms_{v}": c["rms"], f"P_{v}": p} for c, p in zip(recs, P)]
        out["c6"] = rows if out["c6"] is None else [{**a, **b} for a, b in zip(out["c6"], rows)]
    return out


def real_job(job):
    kind, chem, T = job
    return {"lab": lambda: lab8(chem, T), "butane_c4": lambda: butane_c4(T), "butane_c6": lambda: butane_c6(T)}[kind]()


def dev_real(workers):
    import round4_lab as r4
    R2 = json.loads((ROOT / "results" / "round2" / "FROZEN.json").read_text())
    jobs = [("lab", c, float(T)) for c in ("MEA", "HFO") for T in r4.TEMPS] + \
           [("butane_c4", None, T) for T in (3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 4000.0, 4500.0, 5000.0)] + \
           [("butane_c6", None, T) for T in [3100.0] + [float(T) for T in R2["design_targets"]]]
    with Pool(workers) as pool:
        res = pool.map(real_job, jobs)
    lab = [r for r, j in zip(res, jobs) if j[0] == "lab"]
    # v1 must be round 5's and round 6's MEA results exactly
    mea6 = json.loads((ROOT / "results" / "round6" / "confirm_mea.json").read_text())["z"]
    z1 = [x for r in lab if r["chem"] == "MEA" for x in r["z"]["v1"]]
    assert len(z1) == len(mea6) and np.allclose(z1, mea6, atol=1e-5), "MEA C4 v1 does not reproduce round 6"
    for r in (r for r in lab if r["chem"] == "MEA"):
        st = {c["name"]: c for c in json.loads((ROOT / "results" / "round5" / f"confirm_c6_MEA_{int(r['T'])}.json").read_text())["campaigns"]}
        assert all(abs(c["P_v1"] - st[c["name"]]["P"]) < 1e-9 and abs(c["rms_v1"] - st[c["name"]]["rms"]) < 1e-9 for c in r["c6"]), \
            f"MEA {r['T']} C6 v1 does not reproduce round 5"
    print(f"v1 reproduces round 6's MEA z ({len(z1)}) and round 5's MEA C6 exactly")
    (OUT / "dev_real.json").write_text(json.dumps(res))


def dev_syn(workers):
    seeds = list(range(40000, 46000, 10))
    path = OUT / "dev_syn.jsonl"
    with Pool(workers) as pool, open(path, "w") as f:
        for r in pool.imap_unordered(dev_syn_one, seeds):
            f.write(json.dumps(r) + "\n")
            f.flush()
    nets = [json.loads(x) for x in path.read_text().splitlines()]
    # v1 must be the stored confirmation results exactly: round 5's C6 rows (40000-42999), round 6's C4 z (43000-45999)
    stored = {}
    for p in (ROOT / "results" / "round5").glob("confirm_flag_syn_*_T*.jsonl"):
        stored.update({("c6", r["seed"]): r for r in map(json.loads, p.read_text().splitlines())})
    for p in (ROOT / "results" / "round6").glob("confirm_syn_*_T*.jsonl"):
        stored.update({("c4", r["seed"]): r for r in map(json.loads, p.read_text().splitlines())})
    n5 = n6 = 0
    for r in nets:
        if ("c6", r["seed"]) in stored:
            st = stored[("c6", r["seed"])]["campaigns"]
            assert all(abs(c["P_v1"] - s["P"]) < 1e-9 and abs(c["rms_v1"] - s["rms"]) < 1e-9 for c, s in zip(r["c6"], st)), r["seed"]
            n5 += 1
        if ("c4", r["seed"]) in stored:
            assert np.allclose(r["z"]["v1"], stored[("c4", r["seed"])]["z"], atol=1e-4), r["seed"]
            n6 += 1
    print(f"{len(nets)} networks; v1 reproduces round 5's C6 on {n5} and round 6's C4 on {n6}")


def h5(z):
    from round6_confirm import h5 as h5_6
    return h5_6(z)


def h7(rows, v):
    import foresight
    from round2_score import h7 as h7_2
    return h7_2([c[f"rms_{v}"] > L2 for c in rows], [foresight.flagged(c[f"P_{v}"]) for c in rows], BAR_C6)


def dev():
    real = json.loads((OUT / "dev_real.json").read_text())
    syn = [json.loads(x) for x in (OUT / "dev_syn.jsonl").read_text().splitlines()]
    lab = [r for r in real if "chem" in r]
    c4 = {c: {v: [x for r in lab if r["chem"] == c for x in r["z"][v]] for v in ("v1", "v2")} for c in ("HFO", "MEA")}
    c4["butane"] = {v: [x for r in real if "z" in r and "chem" not in r for x in r["z"][v]] for v in ("v1", "v2")}
    c6 = {c: [row for r in lab if r["chem"] == c for row in r["c6"]] for c in ("HFO", "MEA")}
    c6["butane"] = [row for r in real if "c6" in r and "chem" not in r for row in r["c6"]]
    for k, ok in OK_KIND.items():
        nets = [r for r in syn if r["kind"] in ok]
        c4[f"synthetic {k}"] = {v: [x for r in nets for x in r["z"][v]] for v in ("v1", "v2")}
        c6[f"synthetic {k}"] = [row for r in nets for row in r["c6"]]
    out = {"C4": {k: {v: h5(z[v]) for v in ("v1", "v2")} for k, z in c4.items()},
           "C6": {k: {v: h7(rows, v) for v in ("v1", "v2")} for k, rows in c6.items()}}
    std = c6["synthetic possible"] + c6["synthetic impossible"]
    out["C6"]["synthetic standard, pooled"] = {v: h7(std, v) for v in ("v1", "v2")}
    rule = {"C4 no harm (H5' passes)": {k: v["v2"]["pass"] for k, v in out["C4"].items()},
            "C6 no harm (accuracy v2 >= v1)": {k: v["v2"]["accuracy"] >= v["v1"]["accuracy"] for k, v in out["C6"].items() if "pooled" not in k},
            "gain (pooled standard catch >= 80 %)": bool(out["C6"]["synthetic standard, pooled"]["v2"]["untrustworthy_caught"] >= 0.80)}
    rule["adopt"] = bool(all(rule["C4 no harm (H5' passes)"].values()) and all(rule["C6 no harm (accuracy v2 >= v1)"].values())
                         and rule["gain (pooled standard catch >= 80 %)"])
    out["rule"] = rule
    (OUT / "dev.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out["C4"].items():
        print(f"C4 {k:>22}: v1 {v['v1']['coverage']:.3f} / {v['v1']['median_abs_z']:.3f} ({v['v1']['n']})   v2 {v['v2']['coverage']:.3f} / {v['v2']['median_abs_z']:.3f}  pass {v['v2']['pass']}")
    for k, v in out["C6"].items():
        a, b = v["v1"], v["v2"]
        c = lambda x: "  - " if x is None else f"{x:.2f}"  # no failures left: nothing to catch
        print(f"C6 {k:>28}: v1 {a['accuracy']:.3f} caught {c(a['untrustworthy_caught'])} passed {a['trustworthy_passed']:.3f} ({a['untrustworthy']} failures)"
              f"   v2 {b['accuracy']:.3f} caught {c(b['untrustworthy_caught'])} passed {b['trustworthy_passed']:.3f} ({b['untrustworthy']} failures)")
    print("rule:", json.dumps(rule))


def check():
    frozen = json.loads((OUT / "FROZEN.json").read_text())  # raises if round 8 is not frozen
    assert frozen["code_sha256"] == sha(CODE), "code changed since the freeze"
    return frozen


def freeze():
    res = json.loads((OUT / "dev.json").read_text())
    assert res["rule"]["adopt"], "the development rule failed: round 8 changes nothing"
    path = OUT / "FROZEN.json"
    assert not path.exists(), "frozen once"
    path.write_text(json.dumps({"frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "line": "firth.fit_lines_firth (via firth.use_firth)",
                                "C4": "round 6 chord variance", "C6": {**FLAG, "cut": "ln 2"}, "code_sha256": sha(CODE),
                                "development rule": res["rule"]}, indent=1))
    print(path.read_text())


def confirm_one(seed):
    """Confirmation seeds 46000-48999 only, and only once round 8 is frozen."""
    assert 46000 <= seed < 49000 and (OUT / "FROZEN.json").exists()
    return network8(seed, float(TARGETS[((seed - 46000) % 1000) // 250]))


def confirm(a, b, T, workers):
    check()
    seeds = list(range(a, b))
    kind = next(k for k, s0 in KINDS.items() if s0 <= seeds[0] < s0 + 1000)
    assert all(KINDS[kind] <= s < KINDS[kind] + 1000 and TARGETS[((s - 46000) % 1000) // 250] == T for s in seeds)
    path = OUT / f"confirm_{kind}_T{T:g}.jsonl"
    assert not path.exists() or len(path.read_text().splitlines()) < len(seeds), "run once"
    with Pool(workers) as pool, open(path, "w") as f:
        for r in pool.imap_unordered(confirm_one, seeds):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"{kind} {T:g}: {len(seeds)} networks")


def score():
    frozen = check()
    assert not (OUT / "score.json").exists(), "scored once"
    out = {"frozen": frozen["frozen"], "C4": {}, "C6": {}}
    for kind, s0 in KINDS.items():
        nets = []
        for i, T in enumerate(TARGETS):
            rows = [json.loads(x) for x in (OUT / f"confirm_{kind}_T{T}.jsonl").read_text().splitlines()]
            assert sorted(r["seed"] for r in rows) == list(range(s0 + 250 * i, s0 + 250 * (i + 1))), f"{kind} {T}: seeds"
            assert all(r["kind"] in OK_KIND[kind] and r["target"] == T for r in rows), f"{kind} {T}: kinds or targets"
            nets += rows
        out["C4"][kind] = {"v2": h5([x for r in nets for x in r["z"]["v2"]]), "v1 (descriptive)": h5([x for r in nets for x in r["z"]["v1"]])}
        rows = [c for r in nets for c in r["c6"]]
        out["C6"][kind] = {"v2": h7(rows, "v2"), "v1 (descriptive)": h7(rows, "v1")}
    (OUT / "score.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-real", action="store_true")
    ap.add_argument("--dev-syn", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--confirm", type=int, nargs=2)
    ap.add_argument("--target", type=float)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.dev_real:
        dev_real(a.workers)
    if a.dev_syn:
        dev_syn(a.workers)
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
