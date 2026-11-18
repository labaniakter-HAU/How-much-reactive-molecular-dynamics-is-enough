# file: experiments/figures.py
"""Manuscript figures from the result files (no computation of new outcomes).

  python -I -X utf8 experiments/figures.py      # writes manuscript/figures/fig*.pdf and .png
Colours: the first categorical slots of the validated default palette (blue, orange, aqua, yellow; adjacent-pair
CVD dE >= 9.1), each series also carrying its own marker and a direct label (aqua and yellow sit below 3:1 contrast).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "manuscript" / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.5, "pdf.fonttype": 42})
MARK = dict(markersize=5.5, markeredgecolor="white", markeredgewidth=1.0)


def load(p):
    p = RES / p
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote manuscript/figures/{name}.pdf/.png")


def fig_identifiability():
    """V0a at 6400 molecules: E_a of the true reactions, concentrations only against concentrations plus events
    (5 fits each, median and range), with the truth; round 4: per-anchor 90 % interval widths if scored."""
    v0a, true = load("v0a/report.json"), [150.0, 170.0, 200.0, 250.0, 10.0, 120.0]
    widths = [(json.loads(p.read_text())["rows"], p) for p in sorted((RES / "round4").glob("boot_*.json"))] if (RES / "round4" / "score.json").exists() else []
    fig, axes = plt.subplots(1, 2 if widths else 1, figsize=(7.0 if widths else 3.4, 2.8), squeeze=False)
    ax = axes[0, 0]
    y = np.arange(6)
    for k, (key, col, mk, lab, dy) in enumerate([("mult100_lam0", ORANGE, "s", "concentrations only", 0.16),
                                                 ("mult100_lam0.1", BLUE, "o", "concentrations + events", -0.16)]):
        d = v0a[key]
        med, lo, hi = np.array(d["Ea_median"]), np.array(d["Ea_min"]), np.array(d["Ea_max"])
        ax.hlines(y + dy, lo, hi, color=col, lw=1.5)
        ax.plot(med, y + dy, mk, color=col, label=lab, **MARK)
    ax.plot(true, y, "|", color=INK, markersize=9, mew=1.6, label="true")
    ax.set_yticks(y, [f"R{i + 1}" + (" (fast pair)" if i in (3, 4) else "") for i in range(6)])
    ax.invert_yaxis()
    ax.set_xlabel("$E_a$ (kJ mol$^{-1}$), 5 fits: median and range")
    ax.set_title("a  Synthetic, 6400 molecules (V0a)", loc="left", fontsize=8, color=INK)
    ax.legend(loc="upper left", fontsize=7)
    if widths:  # round 4: per-anchor widths, counted against fit, all folds
        ax = axes[0, 1]
        pairs = {}
        for rows, p in widths:
            key = p.stem.rsplit("_", 1)[0]
            pairs.setdefault(key, []).extend(rows)
        wc, wf, chem = [], [], []
        for key, rows in pairs.items():
            ec, ef = np.array([r["Ea_counted"] for r in rows]), np.array([r["Ea_fit"] for r in rows], float)
            ok = np.isfinite(ef).all(1)  # the 8 diverged fits are left out, as in the text
            wc.append(np.percentile(ec, 95, 0) - np.percentile(ec, 5, 0))
            wf.append(np.percentile(ef[ok], 95, 0) - np.percentile(ef[ok], 5, 0))
            chem.append(np.full(ec.shape[1], key.split("_")[1]))
        wc, wf, chem = np.concatenate(wc), np.concatenate(wf), np.concatenate(chem)
        for c, col, mk in (("HFO", BLUE, "o"), ("MEA", ORANGE, "s")):
            s = chem == c
            ax.plot(wf[s], wc[s], mk, color=col, alpha=0.45, label=c, markersize=3.5, markeredgecolor="white", markeredgewidth=0.4)
        lo, hi = 0.7 * min(wf.min(), wc.min()), 1.4 * max(wf.max(), wc.max())
        ax.plot([lo, hi], [lo, hi], color=REF, lw=1, ls="--")
        ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], color=REF, lw=1, ls=":")
        ax.text(0.92 * hi, 0.92 * hi, "equal", ha="right", va="bottom", color=INK2, fontsize=7)
        ax.text(0.92 * hi, 0.46 * hi, "counted = 0.5 × fit", ha="right", va="top", color=INK2, fontsize=7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("90 % interval of $E_a$, concentration fit (kJ mol$^{-1}$)")
        ax.set_ylabel("90 % interval, counted rates (kJ mol$^{-1}$)")
        ax.set_title("b  Lab runs, 10 folds × 100 resamples (round 4)", loc="left", fontsize=8, color=INK)
        ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_identifiability")


def fig_calibration():
    """ECDF of |z| for C4: round 1 (Poisson + phi) and round 2 (random effects + discrepancy) on butane, and round 4
    on the lab runs when scored; reference |N(0, 1)|."""
    v6, r2 = load("v6/v6.json"), load("round2/confirm_real.json")
    sets = [("round 1, butane (Poisson + φ)", np.abs([z for f in v6["folds"] for z in f["z"]]), ORANGE),
            ("round 2, butane (+ RE + discrepancy)", np.abs([z for t in r2["targets"].values() for z in t["z"]]), BLUE)]
    if (RES / "round4" / "score.json").exists():
        z4 = [z for p in sorted((RES / "round4").glob("fold_*.json")) for nm, cp in json.loads(p.read_text())["campaigns"].items()
              if nm != "full" for z in cp["z"]]
        sets.append(("round 4, lab HFO/MEA (same model)", np.abs(z4), AQUA))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    x = np.linspace(0, 4, 400)
    from math import erf, sqrt
    ax.plot(x, [erf(v / sqrt(2)) for v in x], color=REF, ls="--", lw=1, label="calibrated |N(0, 1)|: 95.4%")
    for lab, a, col in sets:
        a = np.sort(a[np.isfinite(a)])
        ax.plot(a, np.arange(1, a.size + 1) / a.size, color=col, label=f"{lab}: {np.mean(a <= 2):.1%}")
    ax.axvline(2, color=REF, lw=0.8)
    ax.axhline(0.85, color=REF, lw=0.8, ls=":")
    ax.text(3.95, 0.86, "bar: 85 % within |z| ≤ 2", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("|z| of predicted ln k at the held-out temperature")
    ax.set_ylabel("fraction of rates")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04), fontsize=6.2, title="|z| ≤ 2 in", title_fontsize=6.2)
    save(fig, "fig3_calibration")


def fig_design():
    """C5: (a) butane, target 3100 K, RMS against budget (round 1 design); (b, c) lab runs: anchored / best naive per
    cell, by target and budget (pre-registered test, section 14)."""
    v7, lab = load("v7/v7.json"), load("lab_c5/result.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    f = [0.1, 0.25, 0.5]
    for key, col, mk, name in (("designed", BLUE, "o", "designed"), ("hot_only", ORANGE, "s", "hot-only"),
                              ("uniform", AQUA, "^", "uniform"), ("median_random", YELLOW, "D", "median random")):
        y = [v7["budgets"][str(b)]["campaigns"][key]["rms"] if key != "median_random" else v7["budgets"][str(b)]["median_random"] for b in f]
        ax.plot([100 * b for b in f], y, marker=mk, color=col, label=name, **MARK)  # end values collide: legend, no end labels
    ax.axhline(v7["full"]["rms"], color=REF, ls="--", lw=1)
    ax.text(54, v7["full"]["rms"] + 0.008, "full campaign", va="bottom", ha="right", fontsize=6.5, color=INK2)
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_xticks([10, 25, 50])
    ax.set_xlim(5, 55)
    ax.set_xlabel("budget (% of the full campaign's ps)")
    ax.set_ylabel("RMS error of ln k at 3100 K")
    ax.set_title("a  Butane, target 3100 K", loc="left", fontsize=8, color=INK)
    naive = ("hot_only", "uniform", "median_random")
    for ax, chem, title in ((axes[1], "HFO", "b  HFO, lab runs"), (axes[2], "MEA", "c  MEA, lab runs")):
        Ts = sorted(lab["targets"][chem], key=float)
        for k, (b, col, mk) in enumerate((("0.1", BLUE, "o"), ("0.25", ORANGE, "s"), ("0.5", AQUA, "^"))):
            r = [lab["targets"][chem][T]["budgets"][b]["anchored"] / min(lab["targets"][chem][T]["budgets"][b][n] for n in naive) for T in Ts]
            runs = lab["targets"][chem][Ts[0]]["budgets"][b]["runs"]
            ax.plot(np.arange(len(Ts)) + (k - 1) * 0.15, r, mk, color=col, label=f"{runs} runs", linestyle="none", **MARK)
        ax.axhline(1.1, color=REF, ls="--", lw=1)
        ax.text(-0.3, 1.13, "near-best bar 1.1", va="bottom", ha="left", fontsize=6.5, color=INK2)
        ax.set_xticks(range(len(Ts)), [f"{float(T):.0f}" for T in Ts])
        ax.set_xlabel("held-out temperature (K)")
        ax.set_ylim(0.3, 2.6)
        ax.set_title(title, loc="left", fontsize=8, color=INK)
    axes[1].set_ylabel("anchored / best naive (RMS)")
    axes[2].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig4_design")


if __name__ == "__main__":
    fig_identifiability()
    fig_calibration()
    fig_design()
