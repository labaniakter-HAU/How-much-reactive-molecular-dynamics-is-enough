# file: experiments/figures_fuel.py
"""Figures for the Fuel submission (manuscript/fuel/figures): Figure_1..10 as vector PDF and 600-dpi PNG, and the
graphical abstract. Styled after the journal's example article: a centred panel letter, the legend in a row above the
plot, markers with error bars and fitted curves, response surfaces with floor contours.
Reads only stored results: official score files, per-campaign result files, the processed open data tables and the
descriptive cache results/figdata/figdata.json (experiments/figdata_fuel.py). Numbers quoted in the text are
recomputed here and checked (assertions).

  python -I -X utf8 experiments/figures_fuel.py [1 2 ... 10 ga]
"""
import json
import re
import sys
from math import log, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "manuscript" / "fuel" / "figures"
STAN = ROOT / "data" / "processed" / "stanford"
FD = json.loads((RES / "figdata" / "figdata.json").read_text(encoding="utf-8"))
LN2 = log(2.0)

# Example-article palette (red diamonds, orange squares, teal triangles; blue and green as in its Fig. 5). Checked with
# the dataviz palette validator: adjacent-pair CVD dE >= 12.8; orange sits below 3:1 contrast, so every series also has
# its own marker shape and a legend entry.
RED, ORANGE, TEAL, BLUE, PURPLE = "#d7301f", "#f28e2b", "#2a9d8f", "#3b5ba5", "#8e44ad"
INK, INK2, GRID, REF = "#1a1a1a", "#595959", "#d9d9d9", "#8c8c8c"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": REF, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "ytick.major.size": 0, "ytick.minor.size": 0, "axes.labelcolor": INK,
    "legend.frameon": False, "legend.handletextpad": 0.35, "legend.columnspacing": 1.0, "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.2, "lines.linewidth": 1.3, "lines.markersize": 4.5, "errorbar.capsize": 2,
    "pdf.fonttype": 42, "axes.titlesize": 8.5, "axes.titleweight": "bold"})
MK = dict(markeredgecolor="white", markeredgewidth=0.6)
BOX = dict(fc="white", ec="none", pad=0.6, alpha=0.85)


def save(fig, name, dpi=600):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {name}")


def panel(ax, letter, pad, ncol=None, handles=None):
    """Centred letter above the plot and the legend (if any) in a row just above the plot, as in the example article;
    one `pad` per figure keeps the letters level."""
    h, lab = handles if handles else ax.get_legend_handles_labels()
    if lab:
        ax.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol or len(lab))
    ax.set_title(f"{letter})", pad=pad)


def chem(s):
    """'C2H7ON + HO' -> mathtext subscripts."""
    return re.sub(r"([A-Za-z)])(\d+)", r"\1$_{\2}$", s)


def load(p):
    return json.loads((RES / p).read_text(encoding="utf-8"))


def jl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def logticks(ax, axis, ticks):
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


# ------------------------------------------------------------------ data helpers
def butane_c34(T, seed=1):
    run = pd.read_csv(STAN / "runs.csv")
    rid = run[(run.molecule == "C4H10") & (run.seed == seed) & (run.T_K == T)].run_id.iloc[0]
    d = pd.read_csv(STAN / f"{rid}.csv", usecols=["t_ps", "C3_C4"])
    t = d.t_ps.to_numpy() - d.t_ps.iloc[0]
    return t, d.C3_C4.to_numpy() / d.C3_C4.iloc[0]


def binmean(t, y, n):
    """Means of y in n log-spaced time windows (frames every 0.12 ps flicker as molecules form and break)."""
    e = np.geomspace(0.5, t[-1], n + 1)
    k = np.digitize(t, e) - 1
    ok = [i for i in range(n) if np.any(k == i)]
    return np.array([np.sqrt(e[i] * e[i + 1]) for i in ok]), np.array([y[k == i].mean() for i in ok])


def decay_fit(t, y):
    f = lambda t, yinf, k: yinf + (1 - yinf) * np.exp(-k * t)
    p, _ = curve_fit(f, t, y, p0=(0.2, 0.01), bounds=([0, 1e-6], [1, 10]))
    return lambda tt: f(tt, *p)


def lab_folds(chem_):
    return {T: load(f"round4/fold_{chem_}_{T}.json")["campaigns"] for T in ("2500", "2750", "3000", "3250", "3500")}


# ------------------------------------------------------------------ Fig. 1: workflow
def fig1():
    fig = plt.figure(figsize=(6.9, 2.6))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    pale = ["#f2f2f2", "#e6eef9", "#e3f3f0", "#fdeee2"]
    edge = [REF, BLUE, TEAL, ORANGE]
    heads = ["1  ReaxFF trajectories", "2  Count reaction events", "3  Predict with calibrated error", "4  Design and flag"]
    subs = ["butane pyrolysis (19 runs);\nSCWO of MEA and HFO (40 runs)", "rate = events / exposure (Eq. (1));\nanchors: 95% of events, ≤ 80",
            "Poisson + between-run +\nchord bias (Eqs. (2)–(3))", "which runs to simulate (C5);\ntrust, before use (C6, Eq. (5))"]
    x0s, w = [0.01, 0.2575, 0.505, 0.7525], 0.2375
    minis = []
    for k in range(4):
        x0 = x0s[k]
        bg.add_patch(FancyBboxPatch((x0, 0.2), w, 0.77, boxstyle="round,pad=0.003,rounding_size=0.02", fc=pale[k], ec=edge[k], lw=1.0))
        bg.text(x0 + w / 2, 0.925, heads[k], ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK)
        bg.text(x0 + w / 2, 0.285, subs[k], ha="center", va="center", fontsize=6.2, color=INK2, linespacing=1.25)
        if k < 3:
            bg.add_patch(FancyArrowPatch((x0 + w + 0.001, 0.6), (x0s[k + 1] - 0.001, 0.6), arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1.1))
        minis.append(fig.add_axes([x0 + 0.045, 0.43, w - 0.065, 0.4]))
    for ax in minis:
        ax.tick_params(labelsize=5.2, length=2, pad=1)
        ax.set_facecolor("white")
    a = minis[0]  # butane: carbon left in C3-C4 molecules
    for T, col in ((3100, BLUE), (3600, TEAL), (4000, ORANGE)):
        t, y = butane_c34(T)
        x, m = binmean(t, y, 40)
        a.plot(x, 100 * m, color=col, lw=1.1, label=f"{T} K")
    a.set_xscale("log")
    a.set_xlim(0.5, 700)
    a.set_ylim(0, 105)
    a.set_xlabel("time (ps)", fontsize=5.4, labelpad=1)
    a.set_ylabel("C in C$_3$–C$_4$ (%)", fontsize=5.4, labelpad=1)
    a.legend(fontsize=5, loc="lower left", handlelength=1.0)
    a = minis[1]  # event types by frequency
    n = np.array(FD["butane_rank"][:150])
    A = FD["butane_anchors"]
    a.bar(np.arange(A), n[:A], width=1.0, color=BLUE, lw=0)
    a.bar(np.arange(A, len(n)), n[A:], width=1.0, color="#bdbdbd", lw=0)
    a.set_yscale("log")
    logticks(a, "y", [100, 1000])
    a.set_ylim(80, 1.3 * n[0])
    a.set_xlabel("event type, by frequency", fontsize=5.4, labelpad=1)
    a.set_ylabel("events", fontsize=5.4, labelpad=1)
    a.text(A * 0.55, n[0] * 0.3, "anchors", ha="center", fontsize=5.4, color=BLUE)
    a = minis[2]  # one Arrhenius line with its band
    r = FD["butane_arrhenius"][0]
    u = 1000 / np.array(r["grid"])
    a.fill_between(u, np.array(r["line"]) - 2 * np.array(r["sd"]), np.array(r["line"]) + 2 * np.array(r["sd"]), color=BLUE, alpha=0.18, lw=0)
    a.plot(u, r["line"], color=BLUE, lw=1.0)
    a.plot([1000 / p["T"] for p in r["train"]], [p["lnk"] for p in r["train"]], "o", color=TEAL, ms=2.6, **MK)
    a.plot(1000 / 3100, r["obs"], "D", color=RED, ms=3.6, **MK)
    a.set_xlabel("1000/T (K$^{-1}$)", fontsize=5.4, labelpad=1)
    a.set_ylabel(r"ln $\hat{k}$", fontsize=5.4, labelpad=1)
    a.text(1000 / 3100, r["obs"] + 0.9, "unseen", ha="center", fontsize=5.2, color=RED)
    a = minis[3]  # design at a quarter of the compute
    v7 = load("v7/v7.json")
    b = v7["budgets"]["0.25"]
    vals = [b["campaigns"]["designed"]["rms"], b["median_random"], b["campaigns"]["uniform"]["rms"], b["campaigns"]["hot_only"]["rms"]]
    a.bar(range(4), vals, color=[BLUE, ORANGE, TEAL, RED], width=0.65)
    a.axhline(v7["full"]["rms"], color=INK2, lw=0.8, ls="--")
    a.text(-0.45, v7["full"]["rms"] + 0.015, "full campaign", ha="left", va="bottom", fontsize=5.0, color=INK2)
    a.set_xticks(range(4), ["designed", "random", "uniform", "hot"], fontsize=5.0)
    a.set_ylabel("RMS error, ¼ compute", fontsize=5.4, labelpad=1)
    bg.add_patch(FancyBboxPatch((0.01, 0.025), 0.98, 0.12, boxstyle="round,pad=0.003,rounding_size=0.015", fc="white", ec=INK2, lw=0.8))
    bg.text(0.5, 0.085, "Every claim pre-registered  ·  methods frozen with SHA-256 hashes  ·  tested once on data never used in development"
            "  ·  every failure reported", ha="center", va="center", fontsize=6.5, color=INK)
    save(fig, "Figure_1")


# ------------------------------------------------------------------ Fig. 2: the data
def fig2():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.9, 2.5), gridspec_kw={"wspace": 0.42})
    pad = 25
    for T, col, mk in ((3100, BLUE, "o"), (3500, TEAL, "^"), (4000, ORANGE, "s"), (4500, RED, "D")):
        t, y = butane_c34(T)
        x, m = binmean(t, y, 12)
        a.plot(x, 100 * m, mk, color=col, label=f"{T} K", ms=4.2, **MK)
        f = decay_fit(t, y)
        tt = np.geomspace(0.3, t[-1], 200)
        a.plot(tt, 100 * f(tt), color=col, lw=1.1)
    a.set_xscale("log")
    logticks(a, "x", [1, 10, 100])
    a.set_xlim(0.4, 800)
    a.set_ylim(0, 105)
    a.set_yticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    a.set_xlabel("Time (ps)")
    a.set_ylabel("Carbon in C$_3$–C$_4$ molecules")
    panel(a, "a", pad, ncol=2)
    for s, col, mk, lab in (("butane", BLUE, "o", "butane"), ("HFO", ORANGE, "s", "HFO"), ("MEA", TEAL, "^", "MEA")):
        R = np.array(FD[f"{s}_runs"])
        u, rate = 1000 / R[:, 0], R[:, 1] / R[:, 2]
        b.plot(u, rate, mk, color=col, label=lab, ms=4.2, **MK)
        p = np.polyfit(u, np.log(rate), 1)
        uu = np.linspace(u.min() - 0.005, u.max() + 0.005, 50)
        b.plot(uu, np.exp(np.polyval(p, uu)), color=col, lw=1.1)
    b.set_yscale("log")
    logticks(b, "y", [3, 10, 30, 100, 300])
    b.set_xlabel("1000/T (K$^{-1}$)")
    b.set_ylabel("Reaction events per ps, per run")
    panel(b, "b", pad)
    shares = {}
    for s, col, mk, lab in (("butane", BLUE, "o", "butane"), ("HFO", ORANGE, "s", "HFO"), ("MEA", TEAL, "^", "MEA")):
        n = np.array(FD[f"{s}_rank"])
        n = n[n > 0]
        A = FD[f"{s}_anchors"]
        shares[s] = (round(n[:A].sum() / n.sum(), 3), round(float(np.mean(n == 1)), 3), len(n))
        c.plot(np.arange(1, len(n) + 1), n, color=col, lw=1.2, label=lab)
        c.plot(A, n[A - 1], mk, color=col, ms=5, **MK)
    c.set_xscale("log")
    c.set_yscale("log")
    logticks(c, "x", [1, 10, 100, 1000])
    logticks(c, "y", [1, 10, 100, 1000, 10000])
    c.axvline(80, color=REF, lw=0.8, ls="--")
    c.text(90, 3000, "80 anchors", fontsize=6.3, color=INK2, va="top")
    c.set_xlabel("Event type, ranked by frequency")
    c.set_ylabel("Events, all runs")
    panel(c, "c", pad)
    save(fig, "Figure_2")
    return shares


# ------------------------------------------------------------------ Fig. 3: counted rates, their line and band, the held-out rate
def fig3():
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.7), gridspec_kw={"hspace": 0.78, "wspace": 0.32})
    for row, (key, T_star, sysname) in enumerate((("butane_arrhenius", 3100.0, "butane"), ("MEA_arrhenius", 2500.0, "MEA"))):
        for col, r in enumerate(FD[key]):
            ax = axes[row, col]
            u = 1000 / np.array(r["grid"])
            line, sd = np.array(r["line"]), np.array(r["sd"])
            ax.fill_between(u, line - 2 * sd, line + 2 * sd, color=BLUE, alpha=0.16, lw=0, label="±2σ, Eq. (3)")
            ax.plot(u, line, color=BLUE, lw=1.3, label="fitted line, Eq. (2)")
            tu = np.array([1000 / p["T"] for p in r["train"]])
            tl = np.array([p["lnk"] for p in r["train"]])
            te = np.array([2 / np.sqrt(p["n"]) for p in r["train"]])
            ax.errorbar(tu, tl, yerr=te, fmt="o", color=TEAL, ms=3.6, elinewidth=0.7, capsize=1.4, label="training runs", **MK)
            ax.errorbar([1000 / T_star], [r["obs"]], yerr=[2 / np.sqrt(r["obs_n"])], fmt="D", color=RED, ms=5, elinewidth=1.0,
                        capsize=2, label="held-out rate (unseen)", **MK)
            ax.axvline(1000 / T_star, color=REF, lw=0.7, ls=":")
            ax.set_title(f"{'abcdef'[3 * row + col]})", pad=21)
            ax.text(0.5, 1.12, chem(r["label"]), transform=ax.transAxes, ha="center", va="bottom", fontsize=6.7, color=INK)
            ax.text(0.5, 1.02, f"{r['train_events']:,.0f} training events  ·  z = {r['z']:+.2f}", transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=6.2, color=INK2)
            if row == 1:
                ax.set_xlabel("1000/T (K$^{-1}$)")
            if col == 0:
                ax.set_ylabel(r"ln $\hat{k}$  (" + sysname + ")")
    h, lab = axes[0, 0].get_legend_handles_labels()
    order = [2, 1, 0, 3]
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.035))
    save(fig, "Figure_3")
    return [(r["label"], round(r["z"], 2)) for k in ("butane_arrhenius", "MEA_arrhenius") for r in FD[k]]


# ------------------------------------------------------------------ Fig. 4: identifiability (C1)
def fig4():
    v0a = load("v0a/report.json")
    true = [150.0, 170.0, 200.0, 250.0, 10.0, 120.0]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.9, 2.6), gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 1.1]})
    pad = 25
    y = np.arange(6)
    for key, col, mk, lab, dy in (("mult100_lam0", ORANGE, "s", "conc. only", 0.17), ("mult100_lam0.1", BLUE, "o", "conc. + events", -0.17)):
        d = v0a[key]
        med, lo, hi = np.array(d["Ea_median"]), np.array(d["Ea_min"]), np.array(d["Ea_max"])
        a.errorbar(med, y + dy, xerr=[med - lo, hi - med], fmt=mk, color=col, ms=4.2, elinewidth=1.0, capsize=1.6, label=lab, **MK)
    a.plot(true, y, "|", color=INK, ms=9, mew=1.6, label="true")
    a.set_yticks(y, [f"R{i + 1}" + ("*" if i in (3, 4) else "") for i in range(6)])
    a.invert_yaxis()
    a.grid(axis="x", color=GRID, lw=0.6)
    a.grid(axis="y", visible=False)
    a.set_xlabel("$E_a$ (kJ mol$^{-1}$)")
    panel(a, "a", pad)
    pairs = {}
    for p in sorted((RES / "round4").glob("boot_*.json")):
        pairs.setdefault(p.stem.rsplit("_", 1)[0], []).extend(json.loads(p.read_text())["rows"])
    wc, wf, ch = [], [], []
    for key, rows in pairs.items():
        ec, ef = np.array([r["Ea_counted"] for r in rows]), np.array([r["Ea_fit"] for r in rows], float)
        ok = np.isfinite(ef).all(1)
        wc.append(np.percentile(ec, 95, 0) - np.percentile(ec, 5, 0))
        wf.append(np.percentile(ef[ok], 95, 0) - np.percentile(ef[ok], 5, 0))
        ch.append(np.full(ec.shape[1], key.split("_")[1]))
    wc, wf, ch = np.concatenate(wc), np.concatenate(wf), np.concatenate(ch)
    for cname, col, mk in (("HFO", BLUE, "o"), ("MEA", ORANGE, "s")):
        s = ch == cname
        b.plot(wf[s], wc[s], mk, color=col, alpha=0.5, ms=3, mec="white", mew=0.3, label=cname)
    lo, hi = 3, 2000
    b.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls="--", label="1:1")
    b.plot([lo, hi], [0.5 * lo, 0.5 * hi], color=RED, lw=0.9, ls=":", label="bar 0.5×")
    b.set_xscale("log")
    b.set_yscale("log")
    logticks(b, "x", [10, 100, 1000])
    logticks(b, "y", [10, 100, 1000])
    b.set_xlim(lo, hi)
    b.set_ylim(lo, hi)
    b.set_xlabel("Fit: 90% interval of $E_a$ (kJ mol$^{-1}$)")
    b.set_ylabel("Counted: 90% interval (kJ mol$^{-1}$)")
    panel(b, "b", pad, ncol=2)
    fold = load("round4/fold_MEA_2500.json")
    rows = pairs["boot_MEA_2500"]
    ec, ef = np.array([r["Ea_counted"] for r in rows]), np.array([r["Ea_fit"] for r in rows], float)
    top = np.argsort(-np.array(fold["anchor_events"]), kind="stable")[:4]
    for k, j in enumerate(top):
        for off, data, col in ((-0.18, ec[:, j], BLUE), (0.18, ef[:, j], ORANGE)):
            c.boxplot(data[np.isfinite(data)], positions=[k + off], widths=0.3, patch_artist=True, whis=(5, 95), showfliers=False,
                      medianprops=dict(color="white", lw=1.2), boxprops=dict(facecolor=col, edgecolor=col), whiskerprops=dict(color=col),
                      capprops=dict(color=col))
            jit = np.random.default_rng(k).uniform(-0.09, 0.09, data.size)
            c.plot(k + off + jit, data, ".", color=col, ms=1.6, alpha=0.35)
    c.set_xticks(range(4), [f"#{k + 1}" for k in range(4)])
    c.set_xlabel("Anchor (MEA fold, 2500 K)")
    c.set_ylabel("$E_a$ in 100 resamples (kJ mol$^{-1}$)")
    panel(c, "c", pad, handles=([Patch(color=BLUE), Patch(color=ORANGE)], ["counted", "fit"]))
    save(fig, "Figure_4")


# ------------------------------------------------------------------ Fig. 5: count, do not fit (C2) and compactness (C3)
def fig5():
    r5 = load("round5/score.json")
    Ts = ["2500", "2750", "3000", "3250", "3500"]
    d = {T: load(f"round5/confirm_c2c3_MEA_{T}.json") for T in Ts}
    x = np.arange(5)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.9, 2.5), gridspec_kw={"wspace": 0.42})
    pad = 15
    for off, key, col, lab in ((-0.19, "counted", BLUE, "counted"), (0.19, "fit", ORANGE, "fitted")):
        a.bar(x + off, [d[T][key]["firing"] for T in Ts], 0.36, color=col, label=lab)
    a.set_yscale("log")
    logticks(a, "y", [1, 10, 100, 1000])
    a.set_ylim(0.6, 3000)
    a.set_xticks(x, Ts)
    a.set_xlabel("Held-out temperature (K)")
    a.set_ylabel("Firing error at the held-out T")
    a.text(0.5, 0.97, f"pooled ratio {r5['C2']['firing_ratio']:.3f}: {1 / r5['C2']['firing_ratio']:.0f}× better", transform=a.transAxes,
           ha="center", va="top", fontsize=6.5, color=INK)
    panel(a, "a", pad)
    for key, col, mk, lab in (("counted", BLUE, "o", "counted"), ("fit", ORANGE, "s", "fitted"), ("full", TEAL, "^", "full network")):
        b.plot(x, [d[T][key]["error"] for T in Ts], mk + "-", color=col, label=lab, ms=4.5, lw=1.1, **MK)
    b.set_xticks(x, Ts)
    b.set_ylim(0, 0.32)
    b.set_xlabel("Held-out temperature (K)")
    b.set_ylabel("Concentration error (normalized RMS)")
    b.text(0.5, 0.05, f"C2: counted/fitted {r5['C2']['error_ratio']:.3f} (bar 1.10)\nC3: compact/full {r5['C3']['error_ratio']:.3f} (bar 1.5)",
           transform=b.transAxes, ha="center", va="bottom", fontsize=6.3, color=INK)
    panel(b, "b", pad)
    v = [d[T]["Ea_shift_median"] for T in Ts]
    bars = c.bar(x, v, 0.55, color=ORANGE)
    for r_, val in zip(bars, v):
        c.text(r_.get_x() + r_.get_width() / 2, val + 0.6, f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)
    c.set_xticks(x, Ts)
    c.set_ylim(0, max(v) * 1.25)
    c.set_xlabel("Held-out temperature (K)")
    c.set_ylabel("Median |Δ$E_a$| made by the fit (kJ mol$^{-1}$)")
    panel(c, "c", pad)
    save(fig, "Figure_5")
    return (round(r5["C2"]["firing_ratio"], 3), round(r5["C2"]["error_ratio"], 3), round(r5["C3"]["error_ratio"], 3), [round(x_) for x_ in v])


# ------------------------------------------------------------------ Fig. 6: calibration (C4)
def zsets():
    v6, r2 = load("v6/v6.json"), load("round2/confirm_real.json")
    z4 = [z for p in sorted((RES / "round4").glob("fold_*.json")) for nm, cp in json.loads(p.read_text())["campaigns"].items() if nm != "full" for z in cp["z"]]
    z4l = [z for p in sorted((RES / "round4").glob("synth_lumped_T*.jsonl")) for r in jl(p) for z in r["z_round2re"]]
    z6l = [z for p in sorted((RES / "round6").glob("confirm_syn_lumped_T*.jsonl")) for r in jl(p) for z in r["z"]]
    return [("Butane: Poisson counting only", [z for f in v6["folds"] for z in f["z"]]),
            ("Butane: + between-run + discrepancy", [z for t in r2["targets"].values() for z in t["z"]]),
            ("Laboratory HFO and MEA (unseen)", z4),
            ("Synthetic lumped: first variance", z4l),
            ("Synthetic lumped: chord bias", z6l),
            ("MEA: chord bias (unseen)", load("round6/confirm_mea.json")["z"])]


def fig6():
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.1), gridspec_kw={"hspace": 0.68, "wspace": 0.3}, sharey=True)
    xs = np.linspace(-5, 5, 400)
    out, tops = [], []
    for k, (title, z) in enumerate(zsets()):
        ax = axes.flat[k]
        z = np.asarray(z, float)
        z = z[np.isfinite(z)]
        cov, med = np.mean(np.abs(z) <= 2), np.median(np.abs(z))
        ok = cov >= 0.85 and 0.5 <= med <= 1.0
        out.append((title, round(100 * cov, 1), round(med, 2), ok))
        hgt, _, _ = ax.hist(np.clip(z, -5, 5), bins=np.linspace(-5, 5, 41), density=True, color=BLUE if ok else ORANGE, alpha=0.85,
                            ec="white", lw=0.4)
        tops.append(hgt.max())
        ax.plot(xs, np.exp(-xs ** 2 / 2) / sqrt(2 * np.pi), color=RED, lw=1.3, ls="--")
        for v in (-2, 2):
            ax.axvline(v, color=INK2, lw=0.7, ls=":")
        ax.text(0.03, 0.96, f"|z| ≤ 2: {cov:.1%}\nmedian |z|: {med:.2f}\nn = {z.size:,}", transform=ax.transAxes, va="top", fontsize=6.2,
                color=INK, bbox=BOX)
        ax.text(0.97, 0.96, "PASS" if ok else "FAIL", transform=ax.transAxes, va="top", ha="right", fontsize=7, fontweight="bold",
                color=TEAL if ok else RED, bbox=BOX)
        ax.set_xlim(-5, 5)
        ax.set_title(f"{'abcdef'[k]})", pad=12)
        ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.6)
        if k >= 3:
            ax.set_xlabel("z of the predicted ln k (Eq. (4))")
        if k % 3 == 0:
            ax.set_ylabel("Density")
    axes.flat[0].set_ylim(0, 1.05 * max(tops))
    h = [Patch(color=BLUE), Patch(color=ORANGE), Line2D([], [], color=RED, ls="--")]
    fig.legend(h, ["passes H5′", "fails H5′", "calibrated N(0, 1)"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    save(fig, "Figure_6")
    return out


# ------------------------------------------------------------------ Fig. 7: response surfaces of the realized error
def cube(ax, x, y, z):
    """The example article's box: all twelve edges, the three behind the surface dashed (view elev 24, azim -57)."""
    (x0, x1), (y0, y1), (z0, z1) = x, y, z
    corners = [(a, b, c) for a in (x0, x1) for b in (y0, y1) for c in (z0, z1)]
    hidden = (x0, y1, z0)
    for p in corners:
        for q in corners:
            if p < q and sum(i != j for i, j in zip(p, q)) == 1:
                back = hidden in (p, q)
                ax.plot(*zip(p, q), color="#444444", lw=0.5, ls=(0, (3, 2)) if back else "-", zorder=0 if back else 10)


def fig7():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    c5 = load("lab_c5/result.json")
    fig = plt.figure(figsize=(6.9, 5.0))
    strategies = (("anchored", "designed"), ("hot_only", "hot-only"), ("random", "median random"))
    Tg, Lg = np.meshgrid(np.linspace(2500, 3500, 31), np.linspace(1, 4, 31))
    feats = lambda T, L: np.stack([np.ones_like(T), (T - 3000) / 500, L - 2.5, ((T - 3000) / 500) ** 2, (L - 2.5) ** 2, (T - 3000) / 500 * (L - 2.5)], -1)
    out = {}
    for row, chem_ in enumerate(("HFO", "MEA")):
        folds = lab_folds(chem_)
        pts = {s: [] for s, _ in strategies}
        for T, camps in folds.items():
            for b in ("0.1", "0.25", "0.5"):
                runs = c5["targets"][chem_][T]["budgets"][b]["runs"]
                for s, _ in strategies:
                    v = (np.median([cp["rms"] for nm, cp in camps.items() if nm.startswith(f"{b}/random_")]) if s == "random"
                         else camps[f"{b}/{s}"]["rms"])
                    pts[s].append((float(T), np.log2(runs), v))
            for s, _ in strategies:
                pts[s].append((float(T), np.log2(16), camps["full"]["rms"]))
        zmax = 1.05 * max(max(p[2] for p in v) for v in pts.values())
        for col, (s, name) in enumerate(strategies):
            P = np.array(pts[s])
            X = feats(P[:, 0], P[:, 1])
            beta, *_ = np.linalg.lstsq(X, P[:, 2], rcond=None)
            Z = np.clip(feats(Tg, Lg) @ beta, 0, None)
            out[(chem_, name)] = (round(float(np.mean(P[:, 2])), 3), round(float(np.sqrt(np.mean((X @ beta - P[:, 2]) ** 2))), 3))
            ax = fig.add_subplot(2, 3, 3 * row + col + 1, projection="3d")
            ax.plot_surface(Tg, Lg, Z, cmap="turbo", vmin=0, vmax=zmax, edgecolor="k", linewidth=0.12, antialiased=True, rstride=1, cstride=1)
            ax.contour(Tg, Lg, Z, zdir="z", offset=0, levels=np.linspace(0.1, zmax, 10), cmap="turbo", vmin=0, vmax=zmax, linewidths=0.6)
            ax.contour(Tg, Lg, Z, zdir="z", offset=0, levels=[LN2], colors="k", linewidths=1.1, linestyles="--")
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="k", s=4, depthshade=False)
            cube(ax, (2500, 3500), (1, 4), (0, zmax))
            ax.set_zlim(0, zmax)
            ax.set_xlim(2500, 3500)
            ax.set_ylim(1, 4)
            ax.set_xticks([2500, 3000, 3500])
            ax.set_yticks([1, 2, 3, 4], ["2", "4", "8", "16"])
            ax.set_zticks(np.arange(0, zmax, 0.4))
            ax.tick_params(labelsize=5.4, pad=-1)
            ax.set_xlabel("TARGET T (K)", fontsize=5.6, labelpad=-5)
            ax.set_ylabel("RUNS", fontsize=5.6, labelpad=-6)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.pane.set_edgecolor((1, 1, 1, 0))
                axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
            ax.view_init(elev=24, azim=-57)
            ax.set_box_aspect((1, 1, 0.8), zoom=0.97)
            ax.set_facecolor((1, 1, 1, 0))  # else the next panel's background hides this panel's z label
            ax.set_title(f"{'abcdef'[3 * row + col]})", pad=2, fontsize=8.5)
            ax.text2D(0.5, 0.985, f"{chem_}  ·  {name}", transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color=INK)
    fig.subplots_adjust(left=0.0, right=0.96, bottom=0.03, top=0.96, wspace=0.1, hspace=0.12)
    fig.text(0.985, 0.5, "VERTICAL AXIS: RMS ERROR OF ln k AT THE HELD-OUT TEMPERATURE", rotation=90, ha="center", va="center",
             fontsize=6.2, color=INK)  # mplot3d keeps the z axis on the right; one shared label instead of six clashing ones
    save(fig, "Figure_7")
    return out


# ------------------------------------------------------------------ Fig. 8: campaign design (C5)
def fig8():
    v7, c5 = load("v7/v7.json"), load("lab_c5/result.json")
    anc = load("round3/dev_real_anchored.json")["3100"]  # the final design of Section 2.4 (butane is its development data)
    assert abs(anc["full"] - v7["full"]["rms"]) < 1e-12
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.9, 2.6), gridspec_kw={"wspace": 0.42, "width_ratios": [1.15, 1.05, 1]})
    pad = 25
    f = [10, 25, 50]
    full = v7["full"]["rms"]
    for key, col, mk, lab in (("designed", BLUE, "D", "designed"), ("hot_only", RED, "s", "hot-only"), ("uniform", TEAL, "^", "uniform"),
                              ("random", ORANGE, "o", "random")):
        if key == "random":
            vals = [[cp["rms"] for nm, cp in v7["budgets"][str(x / 100)]["campaigns"].items() if nm.startswith("random_")] for x in f]
            y = [v7["budgets"][str(x / 100)]["median_random"] for x in f]
            q = np.array([np.percentile(v, [25, 75]) for v in vals])
            a.errorbar(f, y, yerr=[np.array(y) - q[:, 0], q[:, 1] - np.array(y)], fmt=mk, color=col, ms=4.5, elinewidth=0.9, label=lab, **MK)
        else:
            y = [anc["budgets"][str(x / 100)]["anchored" if key == "designed" else key] for x in f]
            if key != "designed":  # same menu, pilot and metric as the round-1 run: identical naive campaigns
                assert np.allclose(y, [v7["budgets"][str(x / 100)]["campaigns"][key]["rms"] for x in f])
            a.plot(f, y, mk, color=col, ms=4.8, label=lab, **MK)
        xx = np.geomspace(10, 100, 60)
        a.plot(xx, PchipInterpolator(np.log([10, 25, 50, 100]), y + [full])(np.log(xx)), color=col, lw=1.0)
    a.plot(100, full, "k*", ms=6, label="full")
    d25 = anc["budgets"]["0.25"]["anchored"]
    hot = [anc["budgets"][str(x / 100)]["hot_only"] / anc["budgets"][str(x / 100)]["anchored"] for x in f]
    gain = 1 - d25 / full
    a.annotate(f"{gain:.0%} below\nthe full campaign", xy=(25, d25), xytext=(40, 0.12), fontsize=6.2, color=BLUE, ha="center",
               arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8, mutation_scale=7))
    a.set_xscale("log")
    logticks(a, "x", [10, 25, 50, 100])
    a.set_xticklabels(["10%", "25%", "50%", "100%"])
    a.set_ylim(0.08, 0.75)
    a.set_xlabel("Compute budget (simulated ps)")
    a.set_ylabel("RMS error of ln k at 3100 K")
    panel(a, "a", pad, ncol=3)
    naive = ("hot_only", "uniform", "median_random")
    ratios = []
    for chem_, filled in (("HFO", True), ("MEA", False)):
        Ts = sorted(c5["targets"][chem_], key=float)
        for k, (bud, col, mk) in enumerate((("0.1", BLUE, "o"), ("0.25", ORANGE, "s"), ("0.5", TEAL, "^"))):
            r = [c5["targets"][chem_][T]["budgets"][bud]["anchored"] / min(c5["targets"][chem_][T]["budgets"][bud][n] for n in naive) for T in Ts]
            ratios += r
            runs = c5["targets"][chem_][Ts[0]]["budgets"][bud]["runs"]
            b.plot(np.arange(5) + (k - 1) * 0.14 + (0.05 if not filled else -0.05), r, mk, color=col, ms=4.2, mfc=col if filled else "white",
                   mec=col, mew=0.9, label=f"{runs} runs" if filled else None)
    b.axhline(1.1, color=RED, lw=0.9, ls="--")
    b.axhline(1.0, color=INK2, lw=0.6)
    b.text(4.45, 1.14, "near-best bar 1.1", ha="right", va="bottom", fontsize=6.2, color=RED)
    b.set_xticks(range(5), ["2500", "2750", "3000", "3250", "3500"])
    b.set_xlabel("Held-out temperature (K)")
    b.set_ylabel("Designed / best naive (RMS)")
    hb, lb = b.get_legend_handles_labels()
    hb += [Line2D([], [], ls="", marker="o", color=INK2, ms=4.2), Line2D([], [], ls="", marker="o", mfc="white", mec=INK2, ms=4.2)]
    panel(b, "b", pad, ncol=3, handles=(hb, lb + ["HFO", "MEA"]))
    rows = [(chem_, T, c5["targets"][chem_][T]["budgets"]["0.5"]["anchored"] / c5["targets"][chem_][T]["full"])
            for chem_ in ("HFO", "MEA") for T in sorted(c5["targets"][chem_], key=float)]
    hr = [r[2] for r in rows]
    c.bar(range(10), hr, 0.62, color=[BLUE] * 5 + [ORANGE] * 5)
    c.axhline(1.25, color=RED, lw=0.9, ls="--")
    c.axhline(np.median(hr), color=INK, lw=0.8, ls=":")
    trans = matplotlib.transforms.blended_transform_factory(c.transAxes, c.transData)
    c.text(1.01, 1.25, "bar\n1.25", transform=trans, va="center", fontsize=6, color=RED)
    c.text(1.01, np.median(hr) - 0.06, f"median\n{np.median(hr):.2f}", transform=trans, va="top", fontsize=6, color=INK)
    c.set_xticks(range(10), [f"{float(r[1]):.0f}" for r in rows], rotation=90, fontsize=5.8)
    c.set_xlabel("Held-out temperature (K)")
    c.set_ylabel("Half the runs / all runs (RMS)")
    panel(c, "c", pad, handles=([Patch(color=BLUE), Patch(color=ORANGE)], ["HFO", "MEA"]))
    save(fig, "Figure_8")
    return round(float(np.mean(np.array(ratios) <= 1.1)), 2), round(float(np.median(hr)), 2), round(gain, 3), (round(min(hot), 2), round(max(hot), 2))


# ------------------------------------------------------------------ Fig. 9: the failure flag (C6)
def confusion(h):
    n, bad, flagged = h["targets"], h["untrustworthy"], h["flagged"]
    tp = round((h["untrustworthy_caught"] or 0.0) * bad)
    fn, fp = bad - tp, flagged - tp
    tn = n - bad - fp
    assert abs((tp + tn) / n - h["accuracy"]) < 1e-9
    return tp, fn, fp, tn


def fig9():
    rows = [r for T in ("2500", "2750", "3000", "3250", "3500") for r in load(f"round5/confirm_c6_MEA_{T}.json")["campaigns"]]
    P, rms = np.array([r["P"] for r in rows]), np.array([r["rms"] for r in rows])
    fl, bad = np.array([r["flagged"] for r in rows]), np.array([r["bad"] for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.35]})
    pad = 25
    groups = (("caught", fl & bad, RED, "D"), ("missed", ~fl & bad, ORANGE, "s"), ("false alarm", fl & ~bad, PURPLE, "^"), ("passed", ~fl & ~bad, TEAL, "o"))
    for lab, s, col, mk in groups:
        a.plot(P[s], rms[s], mk, color=col, ms=2.8, alpha=0.6, mec="none", label=f"{lab} ({s.sum():,})")
    a.axvline(LN2, color=INK2, lw=0.8, ls="--")
    a.axhline(LN2, color=INK2, lw=0.8, ls="--")
    a.set_xscale("log")
    a.set_yscale("log")
    logticks(a, "x", [0.2, 0.5, 1, 2, 5])
    logticks(a, "y", [0.1, 0.2, 0.5, 1, 2, 5])
    a.text(LN2 * 1.04, a.get_ylim()[0] * 1.15, "ln 2", fontsize=6, color=INK2)
    a.set_xlabel("Flag score P before use (Eq. (5))")
    a.set_ylabel("Realized RMS error of ln k")
    a.text(0.98, 0.02, f"{np.mean(fl == bad):.1%} correct", transform=a.transAxes, ha="right", va="bottom", fontsize=6.8, fontweight="bold")
    panel(a, "a", pad, ncol=2)
    r4, r5, syn = load("round4/score.json")["MEA (descriptive)"]["C6"], load("round5/score.json")["C6 real"], load("round5/score_syn.json")
    sets = [("MEA\nfirst flag", r4), ("MEA\nrevised", r5), ("synthetic\nlumped", syn["lumped"]), ("synthetic\npossible", syn["possible"]),
            ("synthetic\nimpossible", syn["impossible"])]
    labels = []
    for k, (lab, h) in enumerate(sets):
        tp, fn, fp, tn = confusion(h)
        x1, x2 = 3 * k, 3 * k + 1
        b.bar(x1, tp / (tp + fn), 0.9, color=RED)
        b.bar(x1, fn / (tp + fn), 0.9, bottom=tp / (tp + fn), color=ORANGE)
        b.bar(x2, tn / (tn + fp), 0.9, color=TEAL)
        b.bar(x2, fp / (tn + fp), 0.9, bottom=tn / (tn + fp), color=PURPLE)
        ok = h["accuracy"] >= 0.9 and tp / (tp + fn) >= 0.8 and tn / (tn + fp) >= 0.8
        b.text(x1 + 0.5, 1.03, f"{h['accuracy']:.1%}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=TEAL if ok else RED)
        labels.append(f"{lab}\n{tp + fn:,} | {tn + fp:,}")
    b.axhline(0.8, color=INK, lw=0.8, ls=":")
    trans = matplotlib.transforms.blended_transform_factory(b.transAxes, b.transData)
    b.text(1.005, 0.8, "80%", transform=trans, va="center", fontsize=6, color=INK)
    b.set_xticks([3 * k + 0.5 for k in range(5)], labels, fontsize=6)
    b.set_ylim(0, 1.12)
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    b.set_ylabel("Share within each class")
    hb = [Patch(color=c_) for c_ in (RED, ORANGE, TEAL, PURPLE)]
    panel(b, "b", pad, ncol=2, handles=(hb, ["failures caught", "failures missed", "good passed", "false alarms"]))
    save(fig, "Figure_9")
    return float(np.mean(fl == bad)), int((fl & bad).sum()), int((~fl & bad).sum()), int((fl & ~bad).sum()), int((~fl & ~bad).sum())


# ------------------------------------------------------------------ Fig. 10: zero counts, the Firth line and round 8
def fig10():
    from round5_flag import kind5
    zc = FD["zero_count"]
    T, a0, b0 = np.array(zc["T"]), zc["a"], zc["b"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.8), gridspec_kw={"hspace": 0.75, "wspace": 0.3})
    a, b, c, d = axes.flat
    pad = 25
    ex = zc["example"]
    u = 1000 / T
    n, lnE = np.array(ex["n"]), np.array(ex["lnE"])
    seen = n > 0
    uu = np.linspace(0.19, 0.345, 50)
    a.plot(uu, a0 - b0 * uu, color=INK2, lw=1.0, ls="--", label="true")
    for key, col, lab in (("current", ORANGE, "current (WLS)"), ("poisson_ml", TEAL, "Poisson ML"), ("firth", BLUE, "Firth")):
        aa, bb = ex["lines"][key]
        a.plot(uu, aa - bb * uu, color=col, lw=1.3, label=lab)
    a.plot(u[seen], np.log(n[seen]) - lnE[seen], "o", color=INK, ms=5, label="counted", zorder=5, **MK)
    a.plot(u[~seen], np.log(0.5) - lnE[~seen], "o", mfc="white", mec=INK, ms=5, mew=1.0, label="0 events, as ½", zorder=5)
    a.set_ylim(-8.3, 0.5)
    for x, lam in zip(u, zc["lam"]):
        a.text(x, -7.9, f"{lam:g}", ha="center", fontsize=5.6, color=INK2)
    a.text(0.192, -7.25, "expected events:", ha="left", fontsize=5.6, color=INK2)
    a.set_xlabel("1000/T (K$^{-1}$)")
    a.set_ylabel("ln k")
    h, lab = a.get_legend_handles_labels()
    order = [4, 5, 0, 1, 2, 3]
    panel(a, "a", pad, ncol=3, handles=([h[i] for i in order], [lab[i] for i in order]))
    tg = np.array(zc["targets"])
    for key, col, mk, lab in (("current", ORANGE, "s", "current line"), ("poisson_ml", TEAL, "^", "Poisson ML"), ("firth", BLUE, "o", "Firth")):
        m, lo, hi = np.array(zc["bias"][key]), np.array(zc["q25"][key]), np.array(zc["q75"][key])
        b.fill_between(tg, lo, hi, color=col, alpha=0.15, lw=0)
        b.plot(tg, m, color=col, lw=1.3)
        b.plot(tg[::4], m[::4], mk, color=col, ms=4.2, **MK)
        b.plot([], [], mk + "-", color=col, label=lab, **MK)
    b.axhline(0, color=INK2, lw=0.7)
    for t_ in T:
        b.axvline(t_, color=REF, lw=0.6, ls=":", zorder=0)
    b.text(T[0], 3.6, "sampled temperatures", fontsize=5.8, color=INK2, va="bottom")
    b.set_ylim(-1.0, 4.0)
    b.set_xlabel("Target temperature (K)")
    b.set_ylabel("Mean bias of ln k (400 draws)")
    panel(b, "b", pad)
    rc = load("round7/rca_c6c.json")
    keys = (("wls", "current"), ("ml", "Poisson ML"), ("pql", "quasi-lik."), ("firth", "Firth"))
    kinds = {"possible": BLUE, "impossible": TEAL}
    seen_kind = set()
    for r in rc:
        k = kind5(r["seed"])
        k = k if k in ("lumped", "possible") else "impossible"  # curved, overdispersed or both
        ys = [r[k_] / LN2 for k_, _ in keys]
        c.plot(range(4), ys, "o-", color=kinds[k], lw=0.7, ms=2.6, alpha=0.75, label=k if k not in seen_kind else None)
        seen_kind.add(k)
    fails = [sum(r[k_] > LN2 for r in rc) for k_, _ in keys]
    assert fails == [24, 9, 10, 6], fails
    c.set_yscale("log")
    logticks(c, "y", [0.25, 0.5, 1, 2, 4, 8])
    c.set_ylim(0.2, 22)
    c.axhline(1, color=RED, lw=0.9, ls="--")
    for i, nf in enumerate(fails):
        c.text(i, 13, f"{nf} fail", ha="center", fontsize=6.4, color=RED, fontweight="bold")
    c.set_xticks(range(4), [k_[1] for k_ in keys])
    c.set_xlim(-0.4, 3.4)
    c.set_ylabel("RMS error / ln 2 (24 missed failures)")
    hc, lc = c.get_legend_handles_labels()
    panel(c, "c", pad, handles=(hc, [x + " networks" for x in lc]))
    dev = load("round8/dev_real.json")
    res = {}
    for name, rows in (("HFO", [x for e in dev if e.get("chem") == "HFO" for x in e["c6"]]),
                       ("MEA", [x for e in dev if e.get("chem") == "MEA" for x in e["c6"]]),
                       ("butane", [x for e in dev if "chem" not in e and "c6" in e for x in e["c6"]])):
        res[name] = {}
        for v in ("v1", "v2"):
            Pv, rv = np.array([x[f"P_{v}"] for x in rows]), np.array([x[f"rms_{v}"] for x in rows])
            res[name][v] = (round(float(np.mean((Pv > LN2) == (rv > LN2))), 4), int((rv > LN2).sum()), len(rows))
    for name, col, mk in (("HFO", BLUE, "o"), ("MEA", ORANGE, "s"), ("butane", TEAL, "^")):
        y = [100 * res[name][v][0] for v in ("v1", "v2")]
        d.plot([0, 1], y, mk + "-", color=col, ms=5, lw=1.3, label=name, **MK)
        d.text(1.07, y[1], f"{res[name]['v1'][1]} → {res[name]['v2'][1]} failures", va="center", fontsize=6.2, color=col)
    d.set_xlim(-0.25, 1.95)
    d.set_ylim(90.5, 99)
    d.set_xticks([0, 1], ["current line", "Firth line"])
    d.set_ylabel("Flag accuracy, development sets (%)")
    panel(d, "d", pad)
    save(fig, "Figure_10")
    return res


def graphical_abstract():
    fig = plt.figure(figsize=(5.9, 2.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.94, "How much reactive MD is enough?  Event-counted ReaxFF kinetics you can trust", ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=INK)
    tiles = [("Count events,\ndon't fit", "Eₐ intervals 3× narrower;\nwhich reactions fire,\npredicted 27× better", "#e6eef9", BLUE),
             ("Calibrated where\nnot simulated", "93.5–96.1% of held-out\nrates within 2σ on\nunseen runs", "#e6eef9", BLUE),
             ("Simulate where\nit matters", "¼ of the compute near\nbutane's 3100 K target:\nerror 39% below\nthe full campaign", "#e3f3f0", TEAL),
             ("Know before\nyou trust", "failure flag 91% correct\non unseen SCWO runs", "#fdeee2", ORANGE)]
    for k, (head, body, fc, ec) in enumerate(tiles):
        x0 = 0.02 + k * 0.245
        ax.add_patch(FancyBboxPatch((x0, 0.2), 0.225, 0.62, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=1.0))
        ax.text(x0 + 0.1125, 0.68, head, ha="center", va="center", fontsize=7.6, fontweight="bold", color=INK, linespacing=1.15)
        ax.text(x0 + 0.1125, 0.4, body, ha="center", va="center", fontsize=6.4, color=INK, linespacing=1.25)
        if k < 3:
            ax.add_patch(FancyArrowPatch((x0 + 0.227, 0.51), (x0 + 0.243, 0.51), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1.0))
    ax.text(0.5, 0.08, "Butane pyrolysis · SCWO of MEA and a hydrofluoroolefin · 3000+ synthetic networks — every claim pre-registered and tested once on unseen data",
            ha="center", va="center", fontsize=6.0, color=INK2)
    save(fig, "Graphical_abstract", dpi=300)


if __name__ == "__main__":
    only = set(sys.argv[1:])
    run = lambda name: not only or name in only
    for key, f, msg in (("1", fig1, None), ("2", fig2, "Fig 2 (anchor share, share seen once, types)"), ("3", fig3, "Fig 3 panels"),
                        ("4", fig4, None), ("5", fig5, "Fig 5 (firing, error, C3, Ea shifts)"), ("6", fig6, "Fig 6"),
                        ("7", fig7, "Fig 7 (mean RMS, RSM residual)"), ("8", fig8, "Fig 8 (near-best share, median half/full, gain at 25%)"),
                        ("9", fig9, "Fig 9 (accuracy, caught, missed, false alarms, passed)"), ("10", fig10, "Fig 10 round 8"),
                        ("ga", graphical_abstract, None)):
        if run(key):
            r = f()
            if msg:
                print(msg + ":", r)
