# file: experiments/c5_lab.py
"""C5 on real data that development never saw (design section 14): the lab's HFO and MEA SCWO runs.

  python -I experiments/c5_lab.py --prepare   # integrity summary + file manifest; computes no design outcome
  python -I experiments/c5_lab.py             # the one-shot test; refuses to run without results/lab_c5/FROZEN.json
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

import design  # noqa: E402
import lab  # noqa: E402

LAB = ROOT.parent  # ../HFO1 and ../MEA1 (read-only)
CHEMS = {"HFO": LAB / "HFO1", "MEA": LAB / "MEA1"}
OUT, MANIFEST = ROOT / "results" / "lab_c5", ROOT / "data" / "lab" / "hfo_mea_files.txt"
FRACTIONS, CLIP, COVERAGE, MAX_N = (0.10, 0.25, 0.50), 10.0, 0.95, 80
NAIVE = ("hot_only", "uniform", "median_random")
CODE = [ROOT / "src" / "design.py", ROOT / "src" / "lab.py", Path(__file__).resolve()]


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def manifest():
    return [f"{hashlib.md5(p.read_bytes()).hexdigest()} {p.stat().st_size} {p.relative_to(LAB).as_posix()}"
            for root in CHEMS.values() for *_, folder in lab.runs(root) for p in sorted(folder.glob("1.*"))]


def anchors(n):
    """fit.anchors' rule on pooled menu counts: most frequent reactions until 95 % of events, at most 80."""
    order = np.argsort(-n, kind="stable")
    order = order[n[order] > 0]
    cum = np.cumsum(n[order]) / n.sum()
    return order[: min(MAX_N, int(np.searchsorted(cum, COVERAGE - 1e-12)) + 1)]


def setup(T_of, rx, data, T_star):
    menu = {r: [1] for r in T_of if T_of[r] != T_star}  # whole runs only: the event tables are run totals
    A = anchors(sum(data[r][1][0] for r in menu))
    uni = np.array([sum(nu for _, nu in rx[j][0]) == 1 for j in A])
    pilot = {f"{max(T_of[r] for r in menu):g}K_O2_000": 1}  # hottest other temperature, O2 = 0
    return menu, A, uni, pilot


def target(T_of, rx, data, T_star, rng):
    menu, A, uni, pilot = setup(T_of, rx, data, T_star)
    _, nt, Et = design.pool({r: 1 for r in T_of if T_of[r] == T_star}, data, T_of)
    w, obs = nt[0, A], np.log(np.maximum(nt[0, A], 1.0) / np.where(Et[0, A] > 0, Et[0, A], 1.0))

    def rms(camp, sel=slice(None)):
        Ts, n, E = design.pool(camp, data, T_of)
        err = np.minimum(np.abs(design.predict(design.fit_lines(n[:, A], E[:, A], Ts), T_star)[0] - obs), CLIP)
        return float(np.sqrt((w[sel] * err[sel] ** 2).sum() / w[sel].sum()))

    full = {r: 1 for r in menu}
    out = {"anchors": int(len(A)), "unimolecular_anchors": int(uni.sum()), "pilot": next(iter(pilot)),
           "full": rms(full), "full_unimolecular": rms(full, uni), "budgets": {}}
    for f in FRACTIONS:
        B = math.ceil(f * len(menu) - 1e-9)
        camps = {"anchored": design.anchored(menu, pilot, B, T_of, T_star, data, A),
                 "round2": design.greedy(menu, pilot, B, T_of, T_star),
                 "hot_only": design.hot_only(menu, pilot, B, T_of), "uniform": design.uniform(menu, pilot, B, T_of)}
        rand = [design.random_campaign(menu, pilot, B, rng) for _ in range(100)]
        r = {k: rms(c) for k, c in camps.items()}
        r["median_random"] = float(np.median([rms(c) for c in rand]))
        u = {k: rms(c, uni) for k, c in camps.items()}
        u["median_random"] = float(np.median([rms(c, uni) for c in rand]))
        out["budgets"][str(f)] = {"runs": B, **r, "unimolecular": u,
                                  "campaigns": {k: sorted(nm for nm, L in c.items() if L) for k, c in camps.items()}}
    return out


def h6(res, design_key="anchored", uni=False, t=None):
    near, gain, half = [], [], []
    for chem in res.values():
        for tr in chem.values():
            for f in FRACTIONS:
                b = tr["budgets"][str(f)]
                b = b["unimolecular"] if uni else b
                naive = [b[k] for k in NAIVE]
                near.append(b[design_key] <= t["near_best"] * min(naive))
                gain.append(b[design_key] <= t["gain_over_worst"] * max(naive))
            b5 = tr["budgets"]["0.5"]
            half.append((b5["unimolecular"] if uni else b5)[design_key] / (tr["full_unimolecular"] if uni else tr["full"]))
    q = {"cells": len(near), "near_best_fraction": float(np.mean(near)), "gain_fraction": float(np.mean(gain)),
         "half_vs_full_median": float(np.median(half))}
    q["pass"] = bool(q["near_best_fraction"] >= t["cell_fraction"] and q["gain_fraction"] >= t["cell_fraction"]
                     and q["half_vs_full_median"] <= t["half_vs_full_median"])
    return q


def prepare():
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows = manifest()
    MANIFEST.write_text("# The lab's HFO/MEA SCWO ReaxFF runs, 250 atm series (design section 14); paths relative to the "
                        "folder holding HFO1/ and MEA1/\n# md5 size_bytes path\n" + "\n".join(rows) + "\n")
    print(f"manifest: {len(rows)} files -> {MANIFEST.relative_to(ROOT)}")
    for chem, root in CHEMS.items():
        T_of, rx, data = lab.tables(root)
        raw = {nm: sum(lab.read_events(f).values()) for nm, _, _, f in lab.runs(root)}
        kept = {nm: data[nm][1][0].sum() for nm in T_of}
        print(f"== {chem}: {len(T_of)} runs, T = {sorted(set(T_of.values()))}; reactions with tracked reactants: {len(rx)}")
        print(f"   events per run (non-identity) {min(raw.values())}-{max(raw.values())}; with tracked reactants "
              f"{sum(kept.values()) / sum(raw.values()):.0%} of all")
        for T in sorted(set(T_of.values())):
            menu, A, uni, pilot = setup(T_of, rx, data, T)
            n = sum(data[r][1][0] for r in menu)
            zero = sum(int((~(data[r][1][1][A] > 0)).sum()) for r in menu)  # also catches NaN
            print(f"   target {T:g} K: menu {len(menu)} runs, pilot {next(iter(pilot))}, anchors {len(A)} "
                  f"({uni.sum()} unimolecular) covering {n[A].sum() / n.sum():.0%}; anchor-run pairs with zero exposure {zero}")


def main():
    if "--prepare" in sys.argv:
        prepare()
        return 0
    frozen = json.loads((OUT / "FROZEN.json").read_text())  # raises if the protocol is not frozen
    assert frozen["data_sha256"] == sha([MANIFEST]) and MANIFEST.read_text().splitlines()[2:] == manifest(), "data changed"
    assert frozen["code_sha256"] == sha(CODE), "code changed since the freeze"
    assert not (OUT / "result.json").exists(), "the test runs once"
    res = {}
    for k, (chem, root) in enumerate(CHEMS.items()):
        T_of, rx, data = lab.tables(root)
        res[chem] = {f"{T:g}": target(T_of, rx, data, T, np.random.default_rng(int(T) + k))
                     for T in sorted(set(T_of.values()))}
    t = frozen["H6"]
    score = {"primary": h6(res, t=t),
             **{f"{c} (descriptive)": h6({c: res[c]}, t=t) for c in res},
             "unimolecular only (descriptive)": h6(res, uni=True, t=t),
             "round-2 design (descriptive)": h6(res, "round2", t=t)}
    (OUT / "result.json").write_text(json.dumps({"score": score, "targets": res}, indent=1))
    for chem, rs in res.items():
        for T, tr in rs.items():
            cells = "  ".join(f"{tr['budgets'][str(f)]['runs']} runs: {tr['budgets'][str(f)]['anchored'] / min(tr['budgets'][str(f)][k] for k in NAIVE):.2f}"
                              for f in FRACTIONS)
            print(f"{chem} {T:>5} K  anchored / best naive  {cells}   (50 % / full {tr['budgets']['0.5']['anchored'] / tr['full']:.2f})")
    for k, v in score.items():
        print(f"{k:>34}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
