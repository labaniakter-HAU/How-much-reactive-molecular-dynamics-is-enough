# file: experiments/round5_flag.py
"""Round 5, C6 (design section 16): the revised flag (src/foresight.py) beside the frozen one, on campaigns of
  --butane                      the real butane design targets (development: the no-harm check)
  --synthetic A B --target T    synthetic networks, seeds A..B-1 (development seeds, or the locked confirmation seeds
                                40000-42999, which need results/round5/FROZEN.json)
Per campaign: the label (frozen C4 prediction's RMS error > ln 2), the frozen flag and the revised score P.

  python -I -X utf8 experiments/round5_flag.py --butane --B 10
  python -I -X utf8 experiments/round5_flag.py --synthetic 10000 10100 --target 3100 --B 10 --out results/round5/dev_flag_p3100.jsonl
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

import design  # noqa: E402
import fit  # noqa: E402
import foresight  # noqa: E402

OUT = ROOT / "results" / "round5"
SYN_CODE = [ROOT / "src" / f for f in ("synthetic.py", "design.py", "fit.py", "species.py", "foresight.py", "integrate.py", "kinetics.py",
                                       "data.py")] + [ROOT / "experiments" / f for f in ("round5_flag.py", "round2_synthetic.py",
                                                                                         "v0b_reduction.py", "v0c_synthetic.py")]


def sha(paths):
    import hashlib
    h = hashlib.sha256()
    for q in paths:
        h.update(Path(q).read_bytes())
    return h.hexdigest()
R2 = json.loads((ROOT / "results" / "round2" / "FROZEN.json").read_text())
LABEL = R2["label_rms"]


def butane_target(args):
    """As round2_confirm.design_target (same menu, pilot, campaigns and rng), with the revised score added."""
    T_star, B, phi, noise = args
    import pandas as pd
    from data import load_stanford
    from kinetics import R_KJ, LumpedKinetics
    from round2_confirm import BUDGETS, CLIP, TABLES, flag, uncertainty
    torch.set_num_threads(1)
    butane = (lambda r: r[r.molecule == "C4H10"])(pd.read_csv(TABLES / "runs.csv"))
    g = {}
    for rid in butane.run_id:
        (g[rid],), S, O, classes = load_stanford(TABLES, [rid])
    carbon = np.array(json.loads((TABLES / "classes.json").read_text())["composition"])[:, 0]
    T_of = dict(zip(butane.run_id, butane.T_K.astype(float)))
    nb = {r: g[r].ev.shape[1] for r in g}
    target = [r for r in g if T_of[r] == T_star]
    menu = {r: [L for L in (50, 100, 200) if L < nb[r]] + [nb[r]] for r in g if T_of[r] != T_star}
    data = {r: design.truncations(g[r], O, menu.get(r, [nb[r]])) for r in g}
    A = fit.anchors([g[r] for r in menu])
    _, nt, Et = design.pool({r: nb[r] for r in target}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))
    pilot_T = 4500.0 if T_star == 5000.0 else 5000.0
    pilot = {[r for r in menu if T_of[r] == pilot_T and "_1_" in r][0]: 50}
    total, rng = sum(nb[r] for r in menu), np.random.default_rng(int(T_star))
    full = {r: nb[r] for r in menu}
    camps = {"full": full}
    for f in BUDGETS:
        Bud = int(f * total)
        camps.update({f"{f}/designed": design.greedy(menu, pilot, Bud, T_of, T_star), f"{f}/hot_only": design.hot_only(menu, pilot, Bud, T_of),
                      f"{f}/uniform": design.uniform(menu, pilot, Bud, T_of)})
        camps.update({f"{f}/random_{i}": design.random_campaign(menu, pilot, Bud, rng) for i in range(100)})
    rows = []
    for name, camp in camps.items():
        mean, sd, obs_var, lines = uncertainty(R2, camp, data, T_of, A, T_star)
        with torch.no_grad():
            flux = fit.simulate(LumpedKinetics(S[:, A], O[:, A], lines["a"], 10.0 * lines["b"] * R_KJ, carbon), g[target[0]])[1][-1, 0].numpy()
        rms = float(np.sqrt((w * np.minimum(np.abs(mean - obs), CLIP) ** 2).sum() / w.sum()))
        P, g2, s2 = foresight.predicted_error(camp, data, T_of, A, T_star, flux, B, phi, noise)
        rows.append({"T": T_star, "name": name, "rms": rms, "old": bool(flag(R2, sd, flux, lines, 1.0)), "P": P, "g2": g2, "s2": s2,
                     "temps": len({T_of[r] for r, L in camp.items() if L})})
    Ts, n, E = design.pool(full, data, T_of)
    fl = design.fit_lines(n[:, A], E[:, A], Ts)
    wt = sum(data[r][nb[r]][0][A] for r in menu)
    return rows, float(np.sqrt((wt * fl["b"] ** 2).sum() / wt.sum()))


def kind5(seed):
    """Network kind by seed. round2_synthetic.kind_of knows only seeds below 24000 (anything else became an impossible
    kind), so the confirmation layout of design sections 16-17 is stated here: 40000-40999 and 43000-43999 lumped,
    41000-41999 and 44000-44999 possible, 42000-42999 and 45000-45999 impossible (cycling as kind_of does)."""
    from round2_synthetic import kind_of
    from synthetic import KINDS
    for lo in (40000, 43000):
        if lo <= seed < lo + 1000:
            return "lumped"
        if lo + 1000 <= seed < lo + 2000:
            return "possible"
        if lo + 2000 <= seed < lo + 3000:
            return KINDS[1 + seed % 3]
    return kind_of(seed)


def synthetic_one(args):
    """As round2_synthetic.one's C6 part (same network, data and campaigns per seed), with the revised score added."""
    seed, T_STAR, B, phi, noise = args
    from data import Group
    from round2_synthetic import BUDGETS, CLIP
    from synthetic import network, random_network, run_data, true_lnk
    from v0b_reduction import RX, SPECIES, V, X0
    from v0c_synthetic import RUNS
    torch.set_num_threads(1)
    net = random_network(RX, seed, kind5(seed))
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
    rows = []
    for camp in camps:
        _, _, _, _, lines = design.calibrated(camp, data, T_of, A, T_STAR)
        model = network(SPECIES, net["rx"], 1000.0 * 8.314462618e-3 * lines["b"], np.exp(lines["a"]))
        with torch.no_grad():
            flux = fit.simulate(model, g0, n_sub=4)[1][-1, 0].numpy()
        mean_re, sd_re = design.calibrated_re(camp, data, T_of, A, T_STAR)[:2]
        ok = np.isfinite(sd_re) & np.isfinite(flux)
        fr = np.where(ok, flux, 0.0)
        err_re = float(np.sqrt((fr * sd_re ** 2).sum() / fr.sum())) if fr.sum() > 0 and ok.all() else float("inf")
        f = np.where(np.isfinite(flux), flux, 0.0)
        single = float(f[lines["seen"] <= 1].sum() / f.sum()) if f.sum() > 0 else float("inf")
        rows.append({"rms": float(np.sqrt((w * np.minimum(np.abs(mean_re - truth), CLIP) ** 2).sum() / w.sum())),
                     "old": bool(single > 0.5 or err_re > R2["flag_err"]), "P": foresight.predicted_error(camp, data, T_of, A, T_STAR, flux, B, phi, noise)[0],
                     "temps": len({T_of[r] for r, L in camp.items() if L})})
    return {"seed": seed, "kind": kind5(seed), "target": T_STAR, "campaigns": rows}


def summary(rows, label):
    from round2_score import h7
    bad = [r["rms"] > LABEL for r in rows]
    print(f"{label}: campaigns {len(rows)}, untrustworthy {sum(bad)}")
    print(f"   frozen flag : {h7(bad, [r['old'] for r in rows], R2['H7'])}")
    print(f"   revised flag: {h7(bad, [foresight.flagged(r['P']) for r in rows], R2['H7'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--butane", action="store_true")
    ap.add_argument("--synthetic", type=int, nargs=2)
    ap.add_argument("--target", type=float, default=3100.0)
    ap.add_argument("--B", type=float, required=True)
    ap.add_argument("--no-phi", action="store_true")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    phi, noise = not a.no_phi, not a.no_noise
    OUT.mkdir(parents=True, exist_ok=True)
    if a.butane:
        with Pool(min(a.workers, 4)) as pool:
            res = pool.map(butane_target, [(float(T), a.B, phi, noise) for T in [3100] + R2["design_targets"]])
        rows = [r for rs, _ in res for r in rs]
        (OUT / f"dev_flag_butane_B{a.B:g}.json").write_text(json.dumps({"rows": rows, "rms_slope_per_target": [b for _, b in res]}))
        print("butane: event-weighted RMS slope b of the full-campaign lines per target:", np.round([b for _, b in res], 1))
        summary(rows, "butane, 9 targets")
    if a.synthetic:
        seeds = list(range(*a.synthetic))
        if any(40000 <= s < 43000 for s in seeds):  # confirmation: only with the frozen flag, its settings and its code
            fz = json.loads((OUT / "FROZEN_c6.json").read_text())
            if (a.B, phi, noise) != (fz["flag"]["B"], fz["flag"]["phi"], fz["flag"]["noise"]):
                raise SystemExit(f"confirmation seeds take the frozen flag only: {fz['flag']}")
            if fz["synthetic_code_sha256"] != sha(SYN_CODE):
                raise SystemExit("synthetic code changed since the freeze")
        with Pool(a.workers) as pool, open(a.out, "w") as f:
            allrows = []
            for res in pool.imap_unordered(synthetic_one, [(s, a.target, a.B, phi, noise) for s in seeds]):
                f.write(json.dumps(res) + "\n")
                f.flush()
                allrows += res["campaigns"]
        summary(allrows, f"synthetic seeds {a.synthetic[0]}-{a.synthetic[1]}, target {a.target:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
