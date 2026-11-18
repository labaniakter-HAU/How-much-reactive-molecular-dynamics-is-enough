# file: experiments/companion_tables.py
"""Companion dataset (doi:10.25740/qq718my6646, CC BY 4.0): lump its 10 runs that are not in our Stanford subset
with Part I's frozen class units and event-type table; data/processed/stanford is not touched.

  python -I experiments/companion_tables.py --raw E:/open_data/stanford_qq718my6646 --workers 5
Writes data/processed/companion/: the species and event cache (species/), {run}.csv, {run}_events.csv,
runs.csv, and Part I's classes.json and event_types.json unchanged, so data.load_stanford reads the folder as
it reads Part I's. Events of a type that Part I's table lacks are dropped and counted.
"""
import argparse
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lumping  # noqa: E402
from stanford import extract, is_long_run, parse_name  # noqa: E402

PART1, OUT = ROOT / "data" / "processed" / "stanford", ROOT / "data" / "processed" / "companion"


def _set_out(out):  # workers on Windows re-import the module, so they need the output folder passed in
    global OUT
    OUT = Path(out)


def cache_one(path):
    sp, ev = OUT / "species" / (path.stem + ".csv.gz"), OUT / "species" / (path.stem + "_events.csv.gz")
    if not (sp.exists() and ev.exists()):
        species, events = extract(path)
        species.to_csv(sp, index=False)
        events.to_csv(ev, index=False)
    print("cached", path.name, flush=True)
    return path


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=str(OUT), help="output folder (default data/processed/companion)")
    a = ap.parse_args()
    OUT = Path(a.out)
    (OUT / "species").mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(a.raw).glob("*.atom") if is_long_run(p.name))
    with Pool(a.workers, initializer=_set_out, initargs=(str(OUT),)) as p:
        p.map(cache_one, paths)
    meta = json.loads((PART1 / "classes.json").read_text())
    units = dict(zip(meta["classes"], meta["units"]))
    types = json.loads((PART1 / "event_types.json").read_text())
    index = {(tuple(int(x) for x in np.sign(t["nu"])), tuple(t["reactants"])): j for j, t in enumerate(types)}
    rows = []
    for path in paths:
        sp = pd.read_csv(OUT / "species" / (path.stem + ".csv.gz"))
        ev = pd.read_csv(OUT / "species" / (path.stem + "_events.csv.gz"), keep_default_na=False)
        unknown_el = {e for f in set(sp["formula"]) for e in lumping.atoms(f) if e not in lumping.ELEMENTS}
        if unknown_el:
            raise ValueError(f"{path.name}: elements {unknown_el} outside Part I's classes")
        wide = sp.pivot_table(index=["t_ps", "V_nm3"], columns="formula", values="count", fill_value=0).reset_index()
        names = [c for c in wide.columns if c not in ("t_ps", "V_nm3")]
        t = wide["t_ps"].to_numpy()
        Y, _ = lumping.lump(names, wide[names].to_numpy(float), units)
        run_id = path.name.split("_0.12ts")[0]
        table = pd.DataFrame(Y, columns=lumping.CLASSES)
        table.insert(0, "V_nm3", wide["V_nm3"].to_numpy())
        table.insert(0, "t_ps", t - t[0])
        table.to_csv(OUT / f"{run_id}.csv", index=False)
        frame = np.ceil((ev["t_ps"].to_numpy() - t[0]) / np.median(np.diff(t)) - 1e-6).astype(int)  # as build_tables
        counts, dropped = {}, 0
        for k, r, p in zip(frame, ev["reactants"], ev["products"]):
            key = lumping.event_type(r.split(" + "), p.split(" + "), units)
            if key is None:
                continue
            j = index.get((tuple(int(x) for x in np.sign(key[0])), key[1]))
            if j is None:
                dropped += 1
                continue
            counts[(k, j)] = counts.get((k, j), 0) + 1
        pd.DataFrame([(k, j, n) for (k, j), n in sorted(counts.items())],
                     columns=["frame", "type", "count"]).to_csv(OUT / f"{run_id}_events.csv", index=False)
        info = parse_name(path.name)
        rows.append({"run_id": run_id, **{k: info[k] for k in ("molecule", "seed", "T_K", "P_GPa")},
                     "t0_ps": float(t[0]), "n_frames": len(t), "t_end_ps": float(t[-1] - t[0]),
                     "n_events": int(sum(counts.values())), "dropped_events": dropped,
                     "start": ", ".join(f"{c} {v:g}" for c, v in zip(lumping.CLASSES, Y[0]) if v)})
        print(f"{run_id}: {len(t)} frames, {rows[-1]['n_events']} events kept, {dropped} of unknown type; t = 0: {rows[-1]['start']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    for f in ("classes.json", "event_types.json"):
        shutil.copy(PART1 / f, OUT / f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
