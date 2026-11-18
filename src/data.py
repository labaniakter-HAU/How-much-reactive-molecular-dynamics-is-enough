# file: src/data.py
"""Training data: groups of equally long runs (one batch each), as built by build_tables.py."""
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch

Group = namedtuple("Group", "t obs T V ev")
Group.__doc__ = """t (n_t,) ps; obs (n_t, B, n_c) class amounts per nm3; T, V (B,) in K and nm3;
ev (B, n_bins, n_types) counted events per bin, where bin b spans grid points per_bin*b .. per_bin*(b+1)."""


def load_stanford(folder, run_ids, every=5, per_bin=2):
    """Class tables on a grid of every `every`-th species frame (0.6 ps), events in bins of per_bin grid
    steps (1.2 ps). Returns groups, S (classes x types, mean net change), O (classes x types, reactant
    molecules), class names."""
    folder = Path(folder)
    meta = json.loads((folder / "classes.json").read_text())
    types = json.loads((folder / "event_types.json").read_text())
    runs = pd.read_csv(folder / "runs.csv").set_index("run_id")
    step, by_len = every * per_bin, {}
    for rid in run_ids:
        table = pd.read_csv(folder / f"{rid}.csv")
        n_bins = (len(table) - 1) // step
        rows = slice(0, n_bins * step + 1, every)
        V = float(table["V_nm3"].mean())
        e = pd.read_csv(folder / f"{rid}_events.csv")
        b = (e["frame"].to_numpy() - 1) // step  # species frame k belongs to the interval ending at frame k
        keep = (b >= 0) & (b < n_bins)
        ev = np.zeros((n_bins, len(types)))
        np.add.at(ev, (b[keep], e["type"].to_numpy()[keep]), e["count"].to_numpy()[keep])
        by_len.setdefault(n_bins, []).append(
            (table["t_ps"].to_numpy()[rows], table[meta["classes"]].to_numpy()[rows] / V, float(runs.loc[rid, "T_K"]), V, ev))
    groups = [Group(torch.tensor(items[0][0]), torch.tensor(np.stack([x[1] for x in items], 1)),
                    torch.tensor([x[2] for x in items]), torch.tensor([x[3] for x in items]),
                    torch.tensor(np.stack([x[4] for x in items])))
              for items in by_len.values()]
    S = np.array([tp["nu"] for tp in types]).T
    O = np.array([[tp["reactants"].count(c) for tp in types] for c in meta["classes"]], float)
    return groups, S, O, meta["classes"]
