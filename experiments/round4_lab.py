# file: experiments/round4_lab.py
"""Round 4 (design section 15): every claim tested once, on equal terms, on real data its development never saw:
the lab's HFO/MEA runs. The same 40 runs, the same 10 folds (chemistry x held-out temperature T*: trained on the 16
runs at the other four temperatures, tested on the 4 runs at T*), the same anchors, campaigns and random seeds as
C5's test (section 14), the same species-level counted-rate model (src/species.py) and 100 replicates per fold
wherever a claim resamples (C1: bootstrap resamples; C4-C6: C5's 100 random campaigns per budget).

  python -I -X utf8 experiments/round4_lab.py --prepare   # sizes, campaign identity with C5, integrator check on
                                                          # training runs only; computes no claim outcome
  python -I -X utf8 experiments/round4_lab.py --run       # every part once (refuses without FROZEN.json)
  python -I -X utf8 experiments/round4_lab.py --score     # the verdicts, once
"""
import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import c5_lab  # noqa: E402
import design  # noqa: E402
import species  # noqa: E402
from kinetics import R_KJ  # noqa: E402
from round2_confirm import flag  # noqa: E402

OUT, CHEMS, FRACTIONS, CLIP = ROOT / "results" / "round4", c5_lab.CHEMS, c5_lab.FRACTIONS, c5_lab.CLIP
TEMPS, N_REP, CHUNK, ROUNDS, WORKERS = (2500.0, 2750.0, 3000.0, 3250.0, 3500.0), 100, 5, 5, 8
ROUND2 = json.loads((ROOT / "results" / "round2" / "FROZEN.json").read_text())  # C4's model, C6's flag and label
C5 = json.loads((ROOT / "results" / "lab_c5" / "result.json").read_text())
CODE = [ROOT / "src" / f for f in ("data.py", "design.py", "fit.py", "integrate.py", "kinetics.py", "lab.py", "metrics.py",
                                   "species.py")] + [ROOT / "experiments" / f for f in ("c5_lab.py", "round2_confirm.py",
                                                                                         "round2_score.py", "v4_h0r.py", "round4_lab.py")]


def setup(chem, T_star):
    T_of, rx, data, sp, S, O, groups = species.load(CHEMS[chem])
    menu, A, _, pilot = c5_lab.setup(T_of, rx, data, T_star)  # C5's menu, anchors and pilot
    test = sorted(r for r in T_of if T_of[r] == T_star)
    return T_of, data, S, O, groups, menu, A, pilot, sorted(menu), test


def observed(test, data, T_of, A):
    """Counted ln k at T* over the 4 target runs and its event weights (C5's convention)."""
    _, nt, Et = design.pool({r: 1 for r in test}, data, T_of)
    return np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0)), nt[0, A]


def campaigns(chem, T_star, T_of, data, menu, A, pilot, test):
    """C5's campaigns exactly (the same calls with the same rng), checked against results/lab_c5/result.json."""
    rng = np.random.default_rng(int(T_star) + list(CHEMS).index(chem))
    obs, w = observed(test, data, T_of, A)

    def rms(camp):  # C5's metric (Poisson line)
        Ts, n, E = design.pool(camp, data, T_of)
        err = np.minimum(np.abs(design.predict(design.fit_lines(n[:, A], E[:, A], Ts), T_star)[0] - obs), CLIP)
        return float(np.sqrt((w * err ** 2).sum() / w.sum()))
    ref = C5["targets"][chem][f"{T_star:g}"]
    out = {"full": {r: 1 for r in menu}}
    assert np.isclose(rms(out["full"]), ref["full"], rtol=0, atol=1e-12), "full campaign differs from C5's"
    for f in FRACTIONS:
        B = math.ceil(f * len(menu) - 1e-9)
        cs = {"anchored": design.anchored(menu, pilot, B, T_of, T_star, data, A), "round2": design.greedy(menu, pilot, B, T_of, T_star),
              "hot_only": design.hot_only(menu, pilot, B, T_of), "uniform": design.uniform(menu, pilot, B, T_of)}
        rand = [design.random_campaign(menu, pilot, B, rng) for _ in range(100)]
        b = ref["budgets"][str(f)]
        assert all(sorted(nm for nm, L in c.items() if L) == b["campaigns"][k] for k, c in cs.items()), "campaigns differ from C5's"
        assert np.isclose(np.median([rms(c) for c in rand]), b["median_random"], rtol=0, atol=1e-12), "random campaigns differ from C5's"
        out.update({f"{f:g}/{k}": c for k, c in cs.items()})
        out.update({f"{f:g}/random_{i:02d}": c for i, c in enumerate(rand)})
    return out


def fold_job(chem, T_star):
    """C4 and C6 on every campaign; C2 (counted rates against the concentration fit) and C3 (compact against the
    full tracked network) on the full training set."""
    T_of, data, S, O, groups, menu, A, pilot, train, test = setup(chem, T_star)
    camps = campaigns(chem, T_star, T_of, data, menu, A, pilot, test)
    g_train, g_test = species.batch(groups, train), species.batch(groups, test)
    obs, w = observed(test, data, T_of, A)
    cmax = np.max([groups[r].obs.numpy().max((0, 1)) for r in train], 0)
    c0 = g_test.obs[0].numpy()
    dom = float(max(np.max(c0[b][c0[b] > 0] / np.maximum(cmax[c0[b] > 0], 1e-12)) for b in range(len(test))))
    res = {"chem": chem, "T": T_star, "train": train, "test": test, "anchors": [int(j) for j in A], "dom": dom, "campaigns": {}}
    for name, camp in camps.items():  # C4 and C6
        mean, sd, s2, tau2, lines = species.calibrated_re_k(camp, data, T_of, A, T_star)
        _, xi = species.predict(species.model(S, O, lines, A), g_test)
        flagged = flag(ROUND2, sd, xi.sum(0), lines, dom)  # the frozen flag, computed before the error
        rms = float(np.sqrt((w * np.minimum(np.abs(mean - obs), CLIP) ** 2).sum() / w.sum()))
        keep = w >= 10
        z = ((mean - obs) / np.sqrt(sd ** 2 + 1.0 / np.maximum(w, 1.0) + s2 / len(test)))[keep]
        res["campaigns"][name] = {"runs": sorted(r for r, L in camp.items() if L), "flagged": bool(flagged), "rms": rms,
                                  "bad": bool(rms > ROUND2["label_rms"]), "s2": s2, "tau2": tau2,
                                  "z": [round(float(x), 5) for x in z]}
    lines = species.lines_of(train, data, T_of, A)  # C2 and C3
    counted, fitted = species.model(S, O, lines, A), species.model(S, O, lines, A)
    t0 = time.time()
    res["fit_loss"] = species.conc_fit(fitted, g_train, ROUNDS)
    res["fit_minutes"] = (time.time() - t0) / 60
    ev = np.array([data[r][1][0][A] for r in test])
    for name, m in (("counted", counted), ("fit", fitted)):
        c, xi = species.predict(m, g_test)
        res[name] = {"error_terms": species.error_terms(c, g_test), "firing_terms": species.firing_terms(xi, ev).tolist(),
                     "Ea": (100 * m.Ea.detach().numpy()).tolist()}
    n_train = sum(data[r][1][0] for r in train)
    cols = np.flatnonzero(n_train > 0)
    c, _ = species.predict(species.model(S, O, species.lines_of(train, data, T_of, cols), cols), g_test, extents=False)
    res["full_network"] = {"reactions": int(len(cols)), "error_terms": species.error_terms(c, g_test)}
    res["anchor_events"] = n_train[A].tolist()
    return res


def boot_job(chem, T_star, i0):
    """C1: bootstrap resamples i0 .. i0 + CHUNK - 1 of the 16 training runs, drawn with replacement within each
    temperature (rng seeded by (int(T*) + chemistry index, i)); E_a from counted rates and from the concentration fit."""
    T_of, data, S, O, groups, menu, A, pilot, train, test = setup(chem, T_star)
    by_T = {T: [r for r in train if T_of[r] == T] for T in sorted({T_of[r] for r in train})}
    rows = []
    for i in range(i0, i0 + CHUNK):
        rng = np.random.default_rng([int(T_star) + list(CHEMS).index(chem), i])
        names = [str(r) for rs in by_T.values() for r in rng.choice(rs, len(rs), replace=True)]
        lines = species.lines_of(names, data, T_of, A)
        m = species.model(S, O, lines, A)
        loss = species.conc_fit(m, species.batch(groups, names), ROUNDS)
        rows.append({"i": i, "runs": names, "loss": loss, "Ea_counted": (1000 * lines["b"] * R_KJ).tolist(),
                     "Ea_fit": (100 * m.Ea.detach().numpy()).tolist()})
    return {"chem": chem, "T": T_star, "rows": rows}


def path_of(spec):
    kind, chem, T, *i = spec
    return OUT / (f"fold_{chem}_{T:g}.json" if kind == "fold" else f"boot_{chem}_{T:g}_{i[0]:03d}.json")


def job(spec):
    path = path_of(spec)
    if path.exists():
        return f"exists: {path.name}"
    torch.set_num_threads(1)
    t0 = time.time()
    res = fold_job(*spec[1:]) if spec[0] == "fold" else boot_job(*spec[1:])
    res["minutes"] = (time.time() - t0) / 60
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(path)
    return f"{path.name}: {res['minutes']:.1f} min"


def specs():
    return ([("fold", c, T) for c in CHEMS for T in TEMPS] +
            [("boot", c, T, i) for i in range(0, N_REP, CHUNK) for c in CHEMS for T in TEMPS])


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def check():
    frozen = json.loads((OUT / "FROZEN.json").read_text())  # raises if the protocol is not frozen
    assert frozen["data_sha256"] == sha([c5_lab.MANIFEST]) and c5_lab.MANIFEST.read_text().splitlines()[2:] == c5_lab.manifest(), "data changed"
    assert frozen["code_sha256"] == sha(CODE), "code changed since the freeze"
    return frozen


def run():
    check()
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for msg in pool.imap_unordered(job, specs()):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


def pooled_error(terms):
    e, n = np.array(terms, float).T
    e = np.where(np.isfinite(e), e, np.inf)  # a failed integration counts as an infinite error
    return float(np.inf if np.isinf(e).any() else np.average(e, weights=n))


def score():
    from round2_score import h5, h7
    from v4_h0r import weighted_median
    frozen = check()
    assert not (OUT / "score.json").exists(), "scored once"
    folds = {(c, T): json.loads(path_of(("fold", c, T)).read_text()) for c in CHEMS for T in TEMPS}
    boots = {(c, T): [r for i in range(0, N_REP, CHUNK) for r in json.loads(path_of(("boot", c, T, i)).read_text())["rows"]]
             for c in CHEMS for T in TEMPS}
    t, out = frozen["bars"], {}

    def by(keys):
        # C1: flux-weighted median 90 % E_a interval width, pooled over (fold, anchor)
        wc, wf, wt, n_fail = [], [], [], 0
        for k in keys:
            rows = boots[k]
            assert sorted(r["i"] for r in rows) == list(range(N_REP))
            ec, ef = np.array([r["Ea_counted"] for r in rows]), np.array([r["Ea_fit"] for r in rows])
            wc.append(np.percentile(ec, 95, 0) - np.percentile(ec, 5, 0))
            wf.append(np.percentile(ef, 95, 0) - np.percentile(ef, 5, 0))
            wt.append(np.array(folds[k]["anchor_events"]))
            n_fail += sum(r["loss"] is None for r in rows)
        wc, wf, wt = np.concatenate(wc), np.concatenate(wf), np.concatenate(wt)
        c1 = {"width_counted": weighted_median(wc, wt), "width_fit": weighted_median(wf, wt), "fits_not_started": n_fail}
        c1["ratio"] = c1["width_counted"] / c1["width_fit"]
        c1["pass"] = bool(c1["ratio"] <= t["C1_ratio"])
        # C2 and C3 on the 4 test runs of every fold, pooled
        f = {m: [x for k in keys for x in folds[k][m]["firing_terms"]] for m in ("counted", "fit")}
        e = {m: pooled_error([x for k in keys for x in folds[k][m]["error_terms"]]) for m in ("counted", "fit")}
        e["full"] = pooled_error([x for k in keys for x in folds[k]["full_network"]["error_terms"]])
        fire = {m: float(np.mean(v)) if np.isfinite(v).all() else np.inf for m, v in f.items()}
        c2 = {"firing_counted": fire["counted"], "firing_fit": fire["fit"], "firing_ratio": fire["counted"] / fire["fit"],
              "error_counted": e["counted"], "error_fit": e["fit"], "error_ratio": e["counted"] / e["fit"]}
        c2["pass_a"], c2["pass_b"] = bool(c2["firing_ratio"] <= t["C2_firing_ratio"]), bool(c2["error_ratio"] <= t["C2_error_ratio"])
        c2["pass"] = c2["pass_a"] and c2["pass_b"]
        sizes = [(folds[k]["full_network"]["reactions"], len(folds[k]["anchors"])) for k in keys]
        c3 = {"error_compact": e["counted"], "error_full": e["full"], "error_ratio": e["counted"] / e["full"],
              "size_factor_min": min(a / b for a, b in sizes)}
        c3["pass"] = bool(c3["error_ratio"] <= t["C3_error_ratio"] and c3["size_factor_min"] >= t["C3_size_factor"])
        # C4 on every campaign but the full one; C6 on every campaign
        camps = [(k, nm, cp) for k in keys for nm, cp in folds[k]["campaigns"].items()]
        c4 = h5(np.array([x for _, nm, cp in camps if nm != "full" for x in cp["z"]]), t["C4"])
        c6 = h7([cp["bad"] for *_, cp in camps], [cp["flagged"] for *_, cp in camps], t["C6"])
        return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C6": c6}
    keys = list(folds)
    out["primary"] = by(keys)
    out["primary"]["C5"] = C5["score"]["primary"]
    for c in CHEMS:
        out[f"{c} (descriptive)"] = by([k for k in keys if k[0] == c])
    zf = np.array([x for k in keys for x in folds[k]["campaigns"]["full"]["z"]])
    out["descriptive"] = {"C4 full campaigns": h5(zf, t["C4"]) if len(zf) else None,
                          "C2 |E_a fit - E_a counted| (kJ/mol), event-weighted median": weighted_median(
                              np.concatenate([np.abs(np.array(folds[k]["fit"]["Ea"]) - folds[k]["counted"]["Ea"]) for k in keys]),
                              np.concatenate([np.array(folds[k]["anchor_events"]) for k in keys])),
                          "fit minutes": [round(folds[k]["fit_minutes"], 1) for k in keys]}
    (OUT / "score.json").write_text(json.dumps(out, indent=1, default=float))
    for claim in ("C1", "C2", "C3", "C4", "C5", "C6"):
        print(f"{claim}: {json.dumps(out['primary'][claim], default=float)}")
    return 0


def prepare():
    """Sizes and checks that compute no claim outcome: campaign identity with C5's test, full-network sizes, and on
    the TRAINING runs of one fold only, the counted model integrated by RK4 (the fit) against BDF (the evaluator)."""
    for chem in CHEMS:
        for T in TEMPS:
            T_of, data, S, O, groups, menu, A, pilot, train, test = setup(chem, T)
            camps = campaigns(chem, T, T_of, data, menu, A, pilot, test)
            full = int((sum(data[r][1][0] for r in train) > 0).sum())
            print(f"{chem} {T:g} K: species {S.shape[0]}, reactions {S.shape[1]}, train {len(train)}, test {len(test)}, "
                  f"anchors {len(A)}, full network {full} ({full / len(A):.1f}x), campaigns {len(camps)} (identical to C5's)", flush=True)
        g = species.batch(groups, train)
        m = species.model(S, O, species.lines_of(train, data, T_of, A), A)
        t0 = time.time()
        c_bdf, xi_bdf = species.predict(m, g)
        t1 = time.time()
        with torch.no_grad():
            c_rk, xi_rk = (x.numpy() for x in species.simulate(m, g))
        scale = np.maximum(np.abs(c_bdf).max(0), 1.0)
        print(f"   integrator check, {chem} {T:g} K fold, 16 training runs: max |RK4 - BDF| / max count {np.nanmax(np.abs(c_rk - c_bdf) / scale):.2e}; "
              f"events {np.nanmax(np.abs(xi_rk[-1] - xi_bdf) / np.maximum(np.abs(xi_bdf), 1.0)):.2e}; BDF {t1 - t0:.1f} s", flush=True)
    print(f"jobs: {len(specs())} ({len(CHEMS) * len(TEMPS)} folds; C1 {N_REP} resamples per fold in chunks of {CHUNK})")


def main():
    if "--prepare" in sys.argv:
        prepare()
    elif "--run" in sys.argv:
        run()
    elif "--score" in sys.argv:
        return score()
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
