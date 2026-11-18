# file: experiments/v0a_identifiability.py
"""V0a (hypothesis H0): what can small-box data identify?

The 6-reaction mechanism is fitted by an oracle that is given the true reactions, with event counts
(lam = 0.1) and without (lam = 0), at 64 and 6400 molecules: 5 replicate datasets, each fitted from an
uninformative start, so the spread shows what the data fix. The Fisher information (Gauss-Newton matrix
of the loss) at the truth shows which parameter combination concentrations cannot see.

  python -I experiments/v0a_identifiability.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fit  # noqa: E402
from metrics import normalized_rmse  # noqa: E402
from synthetic import network, simulate  # noqa: E402

SPECIES = ["C4H10", "C2H6", "CH4", "H2", "C2H4", "H"]
RX = [(["C4H10"], ["C2H4", "C2H4", "H2"]), (["C4H10"], ["C2H6", "C2H4"]), (["C2H6"], ["C2H4", "H2"]),
      (["H2"], ["H", "H"]), (["H", "H"], ["H2"]), (["C2H6", "H2"], ["CH4", "CH4"])]
LABELS = ["C4H10 -> 2 C2H4 + H2", "C4H10 -> C2H6 + C2H4", "C2H6 -> C2H4 + H2", "H2 -> 2 H", "2 H -> H2",
          "C2H6 + H2 -> 2 CH4"]
EA = np.array([150.0, 170.0, 200.0, 250.0, 10.0, 120.0])
A = np.array([10.0, 10.0, 20.0, 50.0, 1.0, 0.5])
V, X0, REPS, LAM = 3.3, [64, 0, 0, 0, 0, 0], 5, 0.1


def ok(ea, true):
    return abs(ea - true) <= max(0.10 * true, 5.0)


def data(mult, seed):
    truth = network(SPECIES, RX, EA, A)
    rng = np.random.default_rng(seed)
    train, _ = simulate(truth, [4000.0, 4500.0, 5000.0], 2, 60.0, 201, V, X0, mult, rng)
    test, clean = simulate(truth, [3100.0, 3300.0, 3500.0], 1, 200.0, 201, V, X0, mult, rng)
    return truth, train, test, clean


def oracle_fit(train, lam, seed):
    g = torch.Generator().manual_seed(seed)
    start_lnA = torch.randn(len(RX), generator=g, dtype=torch.float64)
    start_Ea = 1.0 + torch.rand(len(RX), generator=g, dtype=torch.float64)  # 100-200 kJ/mol
    m = network(SPECIES, RX, 100 * start_Ea.numpy(), np.exp(start_lnA.numpy()))
    m.types = torch.arange(len(RX))
    fit.lbfgs(m, [train], lam, rounds=15)
    return m


def fisher_eigs(truth, train, lam, h=1e-5):
    """Eigenvalues/vectors of J^T J for the residuals whose mean square is the loss, at the truth."""
    def residuals():
        c, xi = fit.simulate(truth, train)
        scale = train.obs.abs().amax(0).clamp_min(1e-12)
        r = [((c - train.obs) / scale).flatten() / np.sqrt(c.numel())]
        if lam:
            pred = fit.per_bin_counts(xi, 2)
            r.append(np.sqrt(lam) * (2 * (pred.clamp_min(0) + 1).sqrt() - 2 * (train.ev + 1).sqrt()).flatten() / np.sqrt(pred.numel()))
        return torch.cat(r)
    theta = torch.cat([truth.lnA.detach(), truth.Ea.detach()]).clone()
    cols = []
    with torch.no_grad():
        for i in range(len(theta)):
            for sgn in (1, -1):
                th = theta.clone()
                th[i] += sgn * h
                truth.lnA.copy_(th[:len(RX)])
                truth.Ea.copy_(th[len(RX):])
                cols.append(residuals())
        truth.lnA.copy_(theta[:len(RX)])
        truth.Ea.copy_(theta[len(RX):])
    J = torch.stack([(cols[2 * i] - cols[2 * i + 1]) / (2 * h) for i in range(len(theta))], 1)
    w, v = torch.linalg.eigh(J.T @ J)
    return w.numpy(), v.numpy()


def main():
    torch.set_num_threads(1)
    report, rows = {}, []
    for mult in (1.0, 100.0):
        for lam in (0.0, LAM):
            eas, ratios = [], []
            for rep in range(REPS):
                truth, train, test, clean = data(mult, rep)
                m = oracle_fit(train, lam, rep)
                with torch.no_grad():
                    pred = fit.simulate(m, test)[0].numpy()
                ratios.append(normalized_rmse(pred, test.obs.numpy()) / normalized_rmse(clean, test.obs.numpy()))
                eas.append(100 * m.Ea.detach().numpy())
            eas = np.array(eas)
            key = f"mult{mult:g}_lam{lam:g}"
            miss = float(np.mean([[not ok(e, EA[j]) for j, e in enumerate(row)] for row in eas]))
            report[key] = {"Ea_fits": eas.round(1).tolist(), "miss_share": miss,
                           "Ea_median": np.median(eas, 0).round(1).tolist(), "Ea_min": eas.min(0).round(1).tolist(),
                           "Ea_max": eas.max(0).round(1).tolist(), "pair_diff": (eas[:, 3] - eas[:, 4]).round(1).tolist(),
                           "ratio_median": float(np.median(ratios)),
                           "identified": [bool(all(ok(e, EA[j]) for e in eas[:, j])) for j in range(len(RX))]}
            rows.append((mult, lam, report[key]))
            print(f"mult {mult:g} lam {lam:g}: Ea median {report[key]['Ea_median']} range "
                  f"{list(zip(report[key]['Ea_min'], report[key]['Ea_max']))}; pair diff {report[key]['pair_diff']}; "
                  f"extrapolation ratio median {report[key]['ratio_median']:.2f}; share of single fits missing E_a {miss:.2f}", flush=True)
    truth, train, _, _ = data(100.0, 0)
    for lam in (0.0, LAM):
        w, v = fisher_eigs(truth, train, lam)
        names = [f"lnA{j + 1}" for j in range(len(RX))] + [f"Ea{j + 1}" for j in range(len(RX))]
        top = np.argsort(-np.abs(v[:, 0]))[:3]
        report[f"fisher_lam{lam:g}"] = {"smallest": float(w[0]), "largest": float(w[-1]),
                                        "weakest_direction": {names[i]: round(float(v[i, 0]), 2) for i in top}}
        print(f"Fisher lam {lam:g}: smallest eigenvalue {w[0]:.2e}, largest {w[-1]:.2e}; weakest direction "
              f"{report[f'fisher_lam{lam:g}']['weakest_direction']}", flush=True)
    c0, c1, e1 = report["mult1_lam0"], report["mult100_lam0"], report[f"mult100_lam{LAM:g}"]
    h0 = {"(a) 64 molecules: extrapolation within 1.5x while most single-fit E_a miss": bool(
              c0["ratio_median"] <= 1.5 and c0["miss_share"] > 0.5),
          "(b) concentrations give the fast pair's difference only": bool(
              not (c1["identified"][3] and c1["identified"][4]) and all(abs(d - 240) <= 24 for d in c1["pair_diff"])),
          "(c) with events at 6400, each E_a of the pair is identified": bool(e1["identified"][3] and e1["identified"][4])}
    report["H0"] = h0
    out = ROOT / "results" / "v0a"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for k, v in h0.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"**H0 {'PASS' if all(h0.values()) else 'FAIL'}**")
    return 0 if all(h0.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
