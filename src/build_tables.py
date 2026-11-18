# file: src/build_tables.py
"""Build per-run lumped class and event tables from the Stanford long runs (Plan 1 output contract).

  python -I src/build_tables.py --raw E:/open_data/stanford_sr637qt9337 --out data/processed/stanford
  python -I src/build_tables.py --raw E:/open_data/stanford_sr637qt9337 --out data/processed/stanford --species-only

Species and species-level events are cached in <out>/species/ (two .csv.gz per dump), so lumping can be
re-run in seconds after assign_class changes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lumping  # noqa: E402
from stanford import extract, is_long_run, parse_name  # noqa: E402


def cached(raw, out):
    """Yield (dump name, species table, event table), extracting each long-run dump once."""
    cache = Path(out) / "species"
    cache.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(raw).glob("*.atom")):
        if not is_long_run(path.name):
            continue
        sp, ev = cache / (path.stem + ".csv.gz"), cache / (path.stem + "_events.csv.gz")
        if not (sp.exists() and ev.exists()):
            print("extracting", path.name, flush=True)
            species, events = extract(path)
            species.to_csv(sp, index=False)
            events.to_csv(ev, index=False)
        yield path.name, pd.read_csv(sp), pd.read_csv(ev, keep_default_na=False)


def write(raw, out):
    out = Path(out)
    runs = list(cached(raw, out))
    formulas = set()
    for _, sp, _ in runs:
        formulas |= set(sp["formula"])
    units = lumping.class_units(formulas)
    n_c = len(lumping.CLASSES)
    types, A_sum, Y_sum, rows = {}, np.zeros((n_c, len(lumping.ELEMENTS))), np.zeros(n_c), []
    for name, sp, ev in runs:
        wide = sp.pivot_table(index=["t_ps", "V_nm3"], columns="formula", values="count", fill_value=0).reset_index()
        names = [c for c in wide.columns if c not in ("t_ps", "V_nm3")]
        t = wide["t_ps"].to_numpy()
        Y, A = lumping.lump(names, wide[names].to_numpy(float), units)
        A_sum += A.sum(axis=0)
        Y_sum += Y.sum(axis=0)
        run_id = name.split("_0.12ts")[0]
        table = pd.DataFrame(Y, columns=lumping.CLASSES)
        table.insert(0, "V_nm3", wide["V_nm3"].to_numpy())
        table.insert(0, "t_ps", t - t[0])
        table.to_csv(out / f"{run_id}.csv", index=False)
        # an event at time t belongs to the first species frame k with t[k] >= t, i.e. interval (t[k-1], t[k]]
        frame = np.ceil((ev["t_ps"].to_numpy() - t[0]) / np.median(np.diff(t)) - 1e-6).astype(int)
        counts = {}
        for k, r, p in zip(frame, ev["reactants"], ev["products"]):
            key = lumping.event_type(r.split(" + "), p.split(" + "), units)
            if key is None:
                continue
            s, react = key
            # one type per (sign pattern, reactant classes): size variants of a pathway (C2 joining clusters
            # of any size) merge; exact net changes gave 1447 types in one run, the top 40 covering 57 %
            entry = types.setdefault((tuple(int(x) for x in np.sign(s)), react), [len(types), np.zeros(n_c), 0])
            entry[1] += s
            entry[2] += 1
            counts[(k, entry[0])] = counts.get((k, entry[0]), 0) + 1
        pd.DataFrame([(k, j, n) for (k, j), n in sorted(counts.items())],
                     columns=["frame", "type", "count"]).to_csv(out / f"{run_id}_events.csv", index=False)
        info = parse_name(name)
        rows.append({"run_id": run_id, **{k: info[k] for k in ("molecule", "seed", "T_K", "P_GPa")},
                     "t0_ps": float(t[0]), "n_frames": len(t), "t_end_ps": float(t[-1] - t[0]),
                     "n_events": int(sum(counts.values()))})
    comp = np.divide(A_sum, Y_sum[:, None], out=np.zeros_like(A_sum), where=Y_sum[:, None] > 0)
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    (out / "classes.json").write_text(json.dumps(
        {"classes": lumping.CLASSES, "elements": lumping.ELEMENTS, "units": [units[c] for c in lumping.CLASSES],
         "composition": comp.round(6).tolist()}, indent=1))
    (out / "event_types.json").write_text(json.dumps(  # nu = mean net class change of the type's events
        [{"nu": (s / n).round(6).tolist(), "reactants": list(r), "n": n}
         for (_, r), (_, s, n) in sorted(types.items(), key=lambda kv: kv[1][0])]))
    print(f"stanford: {len(rows)} runs, {len(types)} lumped event types -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--species-only", action="store_true", help="only build the species and event cache")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.species_only:
        for name, _, _ in cached(args.raw, args.out):
            print("cached", name, flush=True)
    else:
        write(args.raw, args.out)


if __name__ == "__main__":
    main()
