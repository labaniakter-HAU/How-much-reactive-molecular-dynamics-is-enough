# file: experiments/round2_synthetic.py
"""Round 2 (design section 12): the "possible" and "impossible" synthetic networks.
Seeds < 5000 and 10000-10999: possible (Arrhenius). Seeds 5000-9999 and 11000-11999: impossible, cycling
curved / overdispersed / both. Development uses seeds < 10000; confirmation seeds 10000-11999 are run once,
after the method and thresholds are frozen.

  python -I experiments/round2_synthetic.py --seeds 0 50 --out results/round2/synth_dev_possible.jsonl
Per network, on a twin of the Stanford menu with target 3100 K (V0c's layout), truth = population ln k:
  C4  z = (predicted - true ln k) / sd for reactions with >= 10 training events, full campaign; round 1 (fit_lines,
      phi) and round 2 (design.calibrated)
  C5  RMS error of the frozen design, hot-only, uniform, median of 100 random, full; budgets 10/25/50 %
  C6  for the designed, hot-only, uniform and 3 random campaigns per budget plus the full campaign: RMS error,
      the flux-weighted RMS of the calibrated sd, the share of flux on reactions seen at <= 1 temperature
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
from data import Group  # noqa: E402
from synthetic import KINDS, network, random_network, run_data, true_lnk  # noqa: E402
from v0b_reduction import RX, SPECIES, V, X0  # noqa: E402
from v0c_synthetic import RUNS  # noqa: E402

BUDGETS, CLIP = (0.10, 0.25, 0.50), 10.0


def kind_of(seed):
    if 20000 <= seed < 22000:
        return "lumped"  # round 3: development 20000-20999, confirmation 21000-21999
    if seed < 5000 or 10000 <= seed < 11000 or 22000 <= seed < 23000:
        return "possible"
    return KINDS[1 + seed % 3]


def one(args):
    seed, T_STAR, extra, do_c6 = args
    torch.set_num_threads(1)
    kind = kind_of(seed)
    net = random_network(RX, seed, kind)
    rng = np.random.default_rng(seed + 100000)
    names = [f"{T}K_{s}" for T, s, _ in RUNS]
    T_of = {nm: float(T) for nm, (T, _, _) in zip(names, RUNS)}
    bins = {nm: b for nm, (_, _, b) in zip(names, RUNS)}
    menu = {nm: [L for L in (50, 100, 200) if L < bins[nm]] + [bins[nm]] for nm in names if T_of[nm] != T_STAR}
    data, conc0 = {}, None
    for nm in names:
        data[nm], c = run_data(net, SPECIES, T_of[nm], bins[nm], X0, V, menu.get(nm, [bins[nm]]), rng, n_sub=4)
    truth = true_lnk(net, T_STAR)
    w = np.exp(truth) * sum(data[nm][bins[nm]][1] for nm in names if T_of[nm] == T_STAR)
    A = np.arange(len(RX))
    out = {"seed": seed, "kind": kind, "sd_run": net["sd_run"], "target": T_STAR}
    full = {nm: bins[nm] for nm in menu}

    Ts, n, E = design.pool(full, data, T_of)
    keep = n.sum(0) >= 10
    m1, s1 = design.predict(design.fit_lines(n, E, Ts), T_STAR)
    m2, s2, psi, tau2, _ = design.calibrated(full, data, T_of, A, T_STAR)
    out["z_round1"] = ((m1 - truth) / s1)[keep].round(4).tolist()
    out["z_round2"] = ((m2 - truth) / s2)[keep].round(4).tolist()
    out["psi"], out["tau2"] = psi, tau2
    m3, s3, s2re, tau2re, _ = design.calibrated_re(full, data, T_of, A, T_STAR)
    out["z_round2re"] = ((m3 - truth) / s3)[keep].round(4).tolist()
    out["s2"], out["tau2_re"] = s2re, tau2re

    g0 = Group(torch.linspace(0, 1.2 * 500, 1001, dtype=torch.float64), torch.as_tensor(np.asarray(X0, float) / V)[None, None],
               torch.tensor([T_STAR], dtype=torch.float64), torch.tensor([V], dtype=torch.float64), None)
    species_net = network(SPECIES, net["rx"], np.zeros(len(RX)), np.ones(len(RX)))
    present = np.asarray(X0) > 0

    def rms_of(camp):
        Ts, n, E = design.pool(camp, data, T_of)
        pred = design.predict(design.fit_lines(n, E, Ts), T_STAR)[0]
        return float(np.sqrt((w * np.minimum(np.abs(pred - truth), CLIP) ** 2).sum() / w.sum()))

    def c6(camp):
        mean, sd, psi, tau2, lines = design.calibrated(camp, data, T_of, A, T_STAR)
        model = network(SPECIES, net["rx"], 1000.0 * 8.314462618e-3 * lines["b"], np.exp(lines["a"]))  # Ea in kJ/mol
        with torch.no_grad():
            flux = fit.simulate(model, g0, n_sub=4)[1][-1, 0].numpy()
        ok = np.isfinite(sd) & np.isfinite(flux)
        f = np.where(ok, flux, 0.0)
        r1 = design.flags(lines, T_STAR, species_net.nu0.numpy(), species_net.orders.numpy(), present, flux)
        mean_re, sd_re = design.calibrated_re(camp, data, T_of, A, T_STAR)[:2]
        okr = np.isfinite(sd_re) & np.isfinite(flux)
        fr = np.where(okr, flux, 0.0)
        return {"err_re": float(np.sqrt((fr * sd_re ** 2).sum() / fr.sum())) if fr.sum() > 0 and okr.all() else float("inf"),
                "rms_re": float(np.sqrt((w * np.minimum(np.abs(mean_re - truth), CLIP) ** 2).sum() / w.sum())),
                "rms": float(np.sqrt((w * np.minimum(np.abs(mean - truth), CLIP) ** 2).sum() / w.sum())),
                "err": float(np.sqrt((f * sd ** 2).sum() / f.sum())) if f.sum() > 0 and ok.all() else float("inf"),
                "single": float(f[lines["seen"] <= 1].sum() / f.sum()) if f.sum() > 0 else float("inf"),
                "r1": bool(r1[0] or r1[1])}

    total, pilot, crng = sum(bins[nm] for nm in menu), {"4500K_1" if T_STAR == 5000.0 else "5000K_1": 50}, np.random.default_rng(0)
    out["full"], out["c5"], out["c6"] = rms_of(full), {}, ([c6(full)] if do_c6 else [])
    for f in BUDGETS:
        B = int(f * total)
        camps = {"designed": design.greedy(menu, pilot, B, T_of, T_STAR), "hot_only": design.hot_only(menu, pilot, B, T_of),
                 "uniform": design.uniform(menu, pilot, B, T_of)}
        rand = [design.random_campaign(menu, pilot, B, crng) for _ in range(100)]
        for k in extra:  # round 3 design variants
            camps[k] = {"bias8": lambda: design.greedy_bias(menu, pilot, B, T_of, T_STAR, design.TAU2_PRIOR, design.pilot_scale(pilot, data, A)),
                        "seq8": lambda: design.sequential(menu, pilot, B, T_of, T_STAR, data, A, tau2=design.TAU2_PRIOR),
                        "seq_est": lambda: design.sequential(menu, pilot, B, T_of, T_STAR, data, A, tau2=None),
                        "nearest": lambda: design.nearest_first(menu, pilot, B, T_of, T_STAR),
                        "adapt": lambda: design.adaptive(menu, pilot, B, T_of, T_STAR, data, A),
                        "adapt2": lambda: design.adaptive(menu, pilot, B, T_of, T_STAR, data, A, common_window=True),
                        "anchored": lambda: design.anchored(menu, pilot, B, T_of, T_STAR, data, A),
                        **{f"curv{gm:g}": (lambda gm: lambda: design.greedy_curv(menu, pilot, B, T_of, T_STAR, gm, design.pilot_scale(pilot, data, A)))(gm)
                           for gm in (35.0, 70.0, 140.0)}}[k]()
        r = {k: rms_of(c) for k, c in camps.items()}
        r["median_random"] = float(np.median([rms_of(c) for c in rand]))
        out["c5"][str(f)] = r
        if do_c6:
            out["c6"] += [c6(c) for c in [camps[k] for k in ("designed", "hot_only", "uniform")] + rand[:3]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, required=True, help="first and one-past-last seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", type=float, default=3100.0)
    ap.add_argument("--designs", default="", help="round 3: extra designs, e.g. bias8,seq8,seq_est,nearest")
    ap.add_argument("--no-c6", action="store_true")
    a = ap.parse_args()
    seeds = list(range(*a.seeds))
    if any(10000 <= s < 12000 for s in seeds) and not (ROOT / "results" / "round2" / "FROZEN.json").exists():
        raise SystemExit("confirmation seeds need results/round2/FROZEN.json (method and thresholds frozen first)")
    if any(21000 <= s < 24000 for s in seeds) and not (ROOT / "results" / "round3" / "FROZEN.json").exists():
        raise SystemExit("round-3 confirmation seeds need results/round3/FROZEN.json")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as p, open(a.out, "w") as f:
        extra = [d for d in a.designs.split(",") if d]
        for res in p.imap_unordered(one, [(s_, a.target, extra, not a.no_c6) for s_ in seeds]):
            f.write(json.dumps(res) + "\n")
            f.flush()
    print(f"{len(seeds)} networks -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
