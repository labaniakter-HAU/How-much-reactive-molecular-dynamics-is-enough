# file: experiments/round5_dev_c6b.py
"""Round 5 development for C6, part 2 (design section 16; HFO campaigns only, no MEA file is read): candidate
predicted-error scores for the flag, each compared with the realized error of the frozen C4 prediction.

  python -I -X utf8 experiments/round5_dev_c6b.py
Scores (flux-weighted RMS over anchors of a predicted error per anchor):
  old       the frozen flag's: C4's predictive sd; any non-finite sd flags the campaign
  design    round 3's design objective: random-effects line variance + (gamma x chord bias)^2, gamma^2 from the
            campaign (design.curvature2) or the prior 70; one temperature: 1/S0 + (B x distance)^2, B a slope prior
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import numpy as np  # noqa: E402

import design  # noqa: E402
import species  # noqa: E402

L2 = float(np.log(2))


def predicted(camp, data, T_of, A, T_star, flux, B, g_prior=design.GAMMA_PRIOR, s2_prior=design.S2_PRIOR):
    """-> flux-weighted RMS of the predicted error of a straight line through the campaign, at the target."""
    Ts, n, E = design.pool(camp, data, T_of)
    runs = np.array([sum(1 for r, L in camp.items() if L > 0 and T_of[r] == T) for T in Ts])
    s2 = species.seed_s2_k(camp, data, T_of, A)
    s2 = s2_prior if s2 is None else s2
    g2 = design.curvature2(camp, data, T_of, A, T_star, s2)
    g2 = g_prior ** 2 if g2 is None else g2
    n, E = n[:, A], E[:, A]
    use = E > 0
    y = np.where(n > 0, n, 0.5)
    w = np.where(use, 1.0 / (1.0 / y + s2 / runs[:, None]), 0.0)
    x = (1000.0 / Ts - 1000.0 / T_star)[:, None]
    S0 = w.sum(0)
    S0s = np.where(S0 > 0, S0, 1.0)
    m1, m2, m3 = ((w * x ** p).sum(0) / S0s for p in (1, 2, 3))
    var_x = m2 - m1 ** 2
    two = (use.sum(0) >= 2) & (var_x > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        line = 1.0 / S0s + m1 ** 2 / (S0s * np.where(two, var_x, 1.0)) + g2 * (m2 - m1 * (m3 - m1 * m2) / np.where(two, var_x, 1.0)) ** 2
        flat = 1.0 / S0s + B ** 2 * m2  # one temperature: the line is flat; the slope prior times the distance
    total = np.where(two, line, np.where(S0 > 0, flat, design.c5_clip2 if hasattr(design, "c5_clip2") else 100.0))
    f = np.asarray(flux, float)
    f = np.where(np.isfinite(f) & (f > 0), f, 0.0)
    return float(np.sqrt((f * total).sum() / f.sum())) if f.sum() > 0 else np.inf, float(g2), float(s2)


def report(name, score, bad):
    score, bad = np.asarray(score, float), np.asarray(bad, bool)
    order = np.sort(np.unique(score[np.isfinite(score)]))
    cuts = np.r_[0.0, (order[1:] + order[:-1]) / 2]
    acc = [((score > c) == bad).mean() for c in cuts]
    ok = [c for c in cuts if (score > c)[bad].mean() >= 0.85]
    best85 = max(ok, key=lambda c: ((score > c) == bad).mean())
    pos, neg = score[bad], score[~bad]
    auc = np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])
    line = f"{name:>28}: AUC {auc:.3f}; best accuracy {max(acc):.4f}; with caught >= 0.85: {((score > best85) == bad).mean():.4f} at cut {best85:.3f}"
    for c in (0.6, L2):
        fl = score > c
        line += f" | cut {c:.3f}: acc {(fl == bad).mean():.4f} caught {fl[bad].mean():.3f} passed {(~fl[~bad]).mean():.3f}"
    print(line)


def main():
    import round4_lab as r4
    res = json.loads((ROOT / "results" / "round5" / "dev_c6_hfo.json").read_text())
    rows, slopes = [], []
    for fold in res:
        T_star = fold["T"]
        T_of, data, S, O, groups, menu, A, pilot, train, test = r4.setup("HFO", T_star)
        camps = r4.campaigns("HFO", T_star, T_of, data, menu, A, pilot, test)
        full = species.lines_of(train, data, T_of, A)
        w = sum(data[r][1][0][A] for r in train)
        slopes.append(float(np.sqrt((w * full["b"] ** 2).sum() / w.sum())))
        for c in fold["campaigns"]:
            rows.append((T_star, c, camps[c["name"]], data, T_of, A))
    print("event-weighted RMS slope b of the full-campaign lines per fold:", np.round(slopes, 1))
    bad = np.array([c["rms"] > L2 for _, c, *_ in rows])
    old = []
    for _, c, *_ in rows:
        sd, f = np.array(c["sd"], float), np.array(c["flux"], float)
        old.append(np.sqrt((f * sd ** 2).sum() / f.sum()) if np.isfinite(sd).all() and np.isfinite(f).all() and f.sum() > 0 else np.inf)
    print(f"campaigns {len(rows)}, bad {bad.sum()}")
    report("old (inf -> flagged)", old, bad)
    for B in (10.0, 15.0, 20.0, 25.0, 30.0):
        for g in (35.0, 70.0, 140.0):
            sc = [predicted(camp, data, T_of, A, T, c["flux"], B, g)[0] for T, c, camp, data, T_of, A in rows]
            report(f"design, B {B:g}, gamma0 {g:g}", sc, bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
