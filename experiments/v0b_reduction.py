# file: experiments/v0b_reduction.py
"""V0b: a 27-reaction species-level network (17 main reactions plus a tail of 10 slow ones), lumped into
radical-aware classes with counted events. Does the main method (anchors -> rates counted from events ->
backward elimination on held-out concentrations) give a compact model that extrapolates from 4000-5000 K
down to 3100-3500 K? --lam 0 / 0.1 / 1 runs the fitted comparisons (concentration-only and joint L-BFGS).

  python -I experiments/v0b_reduction.py --mult 100            # counted rates
  python -I experiments/v0b_reduction.py --mult 100 --lam 0.1  # joint fit, for comparison
Pass rule (design section 6, revised 2026-11-04):
  - test error <= 1.5 x max(noise floor, lumping floor), at mult 1 and mult 100; the lumping floor is the
    test error of the same method on noise-free data with every event type;
  - at mult 100, E_a within 10 % (or 5 kJ/mol) for every true reaction that the oracle (true network +
    events) recovers, that maps one-to-one onto a kept lumped reaction and whose reactant classes each
    have one member (exact lumping: a merged class has an effective E_a by construction);
  - at mult 1 (the Stanford box size), at most 13 lumped reactions (half of 27).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fit  # noqa: E402
from data import Group  # noqa: E402
from metrics import normalized_rmse  # noqa: E402
from synthetic import lump_groups, network, simulate  # noqa: E402

SPECIES = ["C4H10", "C4H9", "C2H6", "C2H5", "C2H4", "CH4", "CH3", "H2", "H"]
RX = [  # reactants, products, E_a (kJ/mol), A (1/ps or nm3/ps)
    (["C4H10"], ["C2H5", "C2H5"], 250, 100), (["C4H10", "H"], ["C4H9", "H2"], 40, 1.0),
    (["C4H10", "CH3"], ["C4H9", "CH4"], 60, 0.5), (["C4H9"], ["C2H4", "C2H5"], 120, 50),
    (["C2H5"], ["C2H4", "H"], 150, 50), (["C2H4", "H"], ["C2H5"], 20, 0.5),
    (["C2H6", "H"], ["C2H5", "H2"], 40, 0.5), (["C2H5", "H2"], ["C2H6", "H"], 60, 0.3),
    (["C2H5", "C2H5"], ["C2H6", "C2H4"], 0, 0.2), (["CH3", "H2"], ["CH4", "H"], 50, 0.5),
    (["CH4", "H"], ["CH3", "H2"], 60, 0.5), (["H", "H"], ["H2"], 0, 0.5),
    (["H2"], ["H", "H"], 300, 200), (["C2H6"], ["CH3", "CH3"], 280, 100),
    (["CH3", "CH3"], ["C2H6"], 0, 0.3), (["C2H5", "H"], ["C2H6"], 0, 0.3), (["C4H9", "H"], ["C4H10"], 0, 0.3),
    # tail: slow side reactions, as in the long tail of rare event types in MD
    (["C4H9", "H2"], ["C4H10", "H"], 60, 0.05), (["C2H6", "CH3"], ["C2H5", "CH4"], 50, 0.05),
    (["C2H4", "H2"], ["C2H6"], 150, 0.002), (["C4H10", "C2H5"], ["C4H9", "C2H6"], 50, 0.05),
    (["C2H5", "CH3"], ["C2H4", "CH4"], 0, 0.02), (["C4H9", "C2H5"], ["C4H10", "C2H4"], 0, 0.02),
    (["CH3", "H"], ["CH4"], 0, 0.05), (["C4H9", "CH4"], ["C4H10", "CH3"], 60, 0.02),
    (["C2H5", "CH4"], ["C2H6", "CH3"], 60, 0.02), (["C2H6", "C2H4"], ["C2H5", "C2H5"], 100, 0.002),
]
CLASSES = ["CH4", "radical_small", "C2", "C2_rad", "C3_C4", "C3_C4_rad", "H2", "H"]
ASSIGN = {"CH4": "CH4", "CH3": "radical_small", "C2H6": "C2", "C2H4": "C2", "C2H5": "C2_rad",
          "C4H10": "C3_C4", "C4H9": "C3_C4_rad", "H2": "H2", "H": "H"}
EXACT = [all(list(ASSIGN.values()).count(ASSIGN[s]) == 1 for s in r[0]) for r in RX]  # one-member reactant classes
EA = np.array([r[2] for r in RX], float)
V, X0, MAX_KEPT = 3.3, [64, 0, 0, 0, 0, 0, 0, 0, 0], 13


def sub(g, idx):
    return Group(g.t, g.obs[:, idx], g.T[idx], g.V[idx], g.ev[idx])


def ea_ok(fitted, true):
    return bool(abs(fitted - true) <= max(0.10 * true, 5.0))


def datasets(mult, seed=0):
    truth = network(SPECIES, [r[:2] for r in RX], EA, [r[3] for r in RX])
    rng = np.random.default_rng(seed)
    train, _ = simulate(truth, [4000.0, 4500.0, 5000.0], 2, 60.0, 201, V, X0, mult, rng)
    test, clean = simulate(truth, [3100.0, 3300.0, 3500.0], 1, 200.0, 201, V, X0, mult, rng)
    rx = [r[:2] for r in RX]
    (tr,), S, O, carbon, rtype, W = lump_groups([train], SPECIES, rx, CLASSES, ASSIGN.get)
    (te,), *_ = lump_groups([test], SPECIES, rx, CLASSES, ASSIGN.get, weights_from=[train])
    return train, tr, te, clean @ W, S, O, carbon, rtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--lam", default="counts", help="'counts' (main method: rates from event counts, no fit) or the event weight of a fit")
    ap.add_argument("--out", default=str(ROOT / "results" / "v0b"))
    a = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f} s] {s}", flush=True)

    # lumping floor: every event type, noise-free data
    _, tr0, te0, clean0, S0, O0, c0, _ = datasets(1e6)
    counts = a.lam == "counts"
    lam, rounds = (0.0, 0) if counts else (float(a.lam), 5)
    full = fit.build([tr0], S0, O0, c0, coverage=1.0)
    fit.lbfgs(full, [tr0], lam, rounds=rounds)
    with torch.no_grad():
        lump_floor = normalized_rmse(fit.simulate(full, te0)[0].numpy(), clean0)
    log(f"lumping floor {lump_floor:.4f} ({full.n_active()} types)")

    train, tr, te, clean, S, O, carbon, rtype = datasets(a.mult)
    oracle = network(SPECIES, [r[:2] for r in RX], EA, [r[3] for r in RX])
    oracle.types = torch.arange(len(RX))
    fit.set_counted_rates(oracle, [train], oracle.orders.numpy())
    fit.lbfgs(oracle, [train], max(lam, 0.1), rounds=5)
    recoverable = [j for j in range(len(RX)) if ea_ok(100 * oracle.Ea[j].item(), EA[j])]
    log(f"oracle recovers {len(recoverable)} of {len(RX)} activation energies")

    fit_g, val_g = [sub(tr, [2, 3, 4, 5])], [sub(tr, [0, 1])]  # the lowest training T (4000 K) selects
    model = fit.build([tr], S, O, carbon)
    fit.set_counted_rates(model, fit_g, O)  # rates from 4500-5000 K only, so the 4000 K check stays held out
    log(f"{model.n_active()} anchors of {S.shape[1]} types")
    fit.lbfgs(model, fit_g, lam, rounds=rounds)
    model, path = fit.reduce(model, fit_g, val_g, lam, rounds=0 if counts else 3, log=log)
    fit.set_counted_rates(model, [tr], O)  # final rates from all training runs
    fit.lbfgs(model, [tr], lam, rounds=rounds)
    with torch.no_grad():
        c_te, xi_te = fit.simulate(model, te)
    err = normalized_rmse(c_te.numpy(), te.obs.numpy())
    act = model.active.bool()  # firings at the test temperatures, per kept reaction and bin (design H2)
    firing = ((2 * (fit.per_bin_counts(xi_te, 2)[..., act].clamp_min(0) + 1).sqrt()
               - 2 * (te.ev[..., model.types][..., act] + 1).sqrt()) ** 2).mean().item()
    noise_floor = normalized_rmse(clean, te.obs.numpy())
    ref = max(noise_floor, lump_floor)
    kept = {int(model.types[j]): j for j in range(len(model.active)) if model.active[j] > 0}
    rows, ea_pass = [], True
    for j, k in enumerate(rtype):
        if k is None or rtype.count(k) != 1 or j not in recoverable or not EXACT[j]:
            continue
        ea = 100 * model.Ea[kept[k]].item() if k in kept else float("nan")
        ok = k in kept and ea_ok(ea, EA[j])
        if k in kept:
            ea_pass &= ok
        rows.append({"reaction": f"R{j + 1}", "kept": k in kept, "Ea_true": float(EA[j]), "Ea_fit": round(ea, 1), "pass": ok})
    res = {"mult": a.mult, "lam": a.lam, "error": err, "noise_floor": noise_floor, "lumping_floor": lump_floor,
           "ratio": err / ref, "firing_score": firing, "kept": model.n_active(), "path": path, "ea_checks": rows,
           "pass": bool(err <= 1.5 * ref and (model.n_active() <= MAX_KEPT or a.mult > 1) and (ea_pass or a.mult < 100)),
           "minutes": round((time.time() - t0) / 60, 1)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"mult{a.mult:g}_lam{a.lam}.json").write_text(json.dumps(res, indent=1))
    print(f"V0b mult {a.mult:g} lam {a.lam}: test error {err:.4f}; noise floor {noise_floor:.4f}, lumping floor "
          f"{lump_floor:.4f} -> ratio {err / ref:.2f} (pass <= 1.5); {model.n_active()} reactions kept ({f'pass <= {MAX_KEPT}' if a.mult <= 1 else 'size checked at mult 1'})")
    print(f"   firings at the test temperatures: mean squared difference of 2 sqrt(count + 1) {firing:.4f}")
    for r in rows:
        print("  ", r)
    print(f"**V0b {'PASS' if res['pass'] else 'FAIL'}** ({res['minutes']} min)")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
