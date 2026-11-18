# file: src/stanford.py
"""Species time series from the Stanford ReaxFF dumps (doi:10.25740/sr637qt9337, CC BY 4.0).

Bond rule (Yang, Sing-Long & Reed, Chem. Sci. 2017): a pair is bonded once its distance has stayed
below the pair cutoff for m consecutive frames (tau = 96 fs = 8 frames of 12 fs), and unbonded once it
has stayed above it for m frames. Tracking starts with no bonds, so the first m frames of every file are
a warm-up and are not reported; the long runs start mid-simulation (step 212000), where treating all
close pairs as bonded merges neighbouring molecules.

Events (as in ReacNetGenerator): molecules of two analysed frames are linked when they share atoms; each
linked group that is not one molecule on each side is one event, old molecules -> new molecules.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from formula import formula_string

ELEMENTS = ("C", "H")  # LAMMPS type 1 = C, type 2 = H in these dumps
CUT = np.zeros((3, 3))  # pair cutoffs in Angstrom, indexed by LAMMPS type
CUT[1, 1], CUT[1, 2], CUT[2, 1], CUT[2, 2] = 1.98, 1.57, 1.57, 1.09
_NAME = re.compile(
    r"(?P<molecule>[A-Za-z0-9]+)_(?P<n_atoms>\d+)at_(?P<seed>\d+)_(?P<T>\d+)K_(?P<P>\d+)GPa_"
    r"(?P<n>\d+)(?P<unit>ks|Ms)_(?P<dt>[\d.]+)ts_every(?P<every>\d+)_"
)


def parse_name(name):
    m = _NAME.match(name)
    steps = int(m["n"]) * (1_000 if m["unit"] == "ks" else 1_000_000)
    return {"molecule": m["molecule"], "n_atoms": int(m["n_atoms"]), "seed": int(m["seed"]),
            "T_K": int(m["T"]), "P_GPa": int(m["P"]), "steps": steps, "dt_fs": float(m["dt"]),
            "every": int(m["every"])}


def is_long_run(name):
    """Kinetics use the long runs; the 10ks/100ks files are 1-12 ps starts that include box compression."""
    return parse_name(name)["steps"] >= 500_000


def read_frames(path):
    """Yield (step, box lengths L[3] in A, LAMMPS types[n], wrapped xyz[n, 3] in A), atoms sorted by id.
    Handles both column styles in this dataset: 'xs ys zs' (scaled) and 'x y z' (box units, origin != 0)."""
    with open(path) as f:
        while f.readline():  # ITEM: TIMESTEP
            step = int(f.readline())
            f.readline()
            n = int(f.readline())
            f.readline()
            box = np.array([f.readline().split()[:2] for _ in range(3)], float)
            cols = f.readline().split()[2:]  # names after 'ITEM: ATOMS'
            a = np.loadtxt(f, max_rows=n, ndmin=2)
            a = a[np.argsort(a[:, cols.index("id")])]
            lo, L = box[:, 0], box[:, 1] - box[:, 0]
            if "xs" in cols:
                xyz = (a[:, [cols.index(c) for c in ("xs", "ys", "zs")]] % 1.0) * L
            else:
                xyz = (a[:, [cols.index(c) for c in ("x", "y", "z")]] - lo) % L
            xyz = np.where(xyz >= L, 0.0, xyz)  # float modulo can return exactly L; cKDTree needs [0, L)
            yield step, L, a[:, cols.index("type")].astype(int), xyz


def close_pairs(xyz, L, types):
    """Sorted unique pair keys i*n + j (i < j) closer than their pair cutoff, minimum image."""
    n = len(xyz)
    pairs = cKDTree(xyz, boxsize=L).query_pairs(CUT.max(), output_type="ndarray")
    if len(pairs) == 0:
        return np.empty(0, np.int64)
    d = xyz[pairs[:, 0]] - xyz[pairs[:, 1]]
    d -= L * np.round(d / L)
    keep = np.linalg.norm(d, axis=1) < CUT[types[pairs[:, 0]], types[pairs[:, 1]]]
    p = np.sort(pairs[keep], axis=1).astype(np.int64)
    return np.unique(p[:, 0] * n + p[:, 1])


class BondTracker:
    """Hysteresis on pair keys: bonded after m frames inside the cutoff, unbonded after m frames outside."""

    def __init__(self, m):
        self.m = m
        self.keys = np.empty(0, np.int64)
        self.bonded = np.empty(0, bool)
        self.inside = np.empty(0, int)  # consecutive frames inside the cutoff
        self.outside = np.empty(0, int)  # consecutive frames outside the cutoff

    def update(self, close):
        """close: sorted unique keys inside the cutoff this frame -> sorted keys bonded after this frame."""
        allk = np.union1d(self.keys, close)
        pos = np.searchsorted(allk, self.keys)  # self.keys is a subset of allk
        bonded = np.zeros(len(allk), bool)
        inside = np.zeros(len(allk), int)
        outside = np.zeros(len(allk), int)
        bonded[pos], inside[pos], outside[pos] = self.bonded, self.inside, self.outside
        present = np.isin(allk, close, assume_unique=True)
        inside = np.where(present, inside + 1, 0)
        outside = np.where(present, 0, outside + 1)
        bonded = np.where(bonded, outside < self.m, inside >= self.m)
        keep = bonded | present
        self.keys, self.bonded, self.inside, self.outside = allk[keep], bonded[keep], inside[keep], outside[keep]
        return self.keys[self.bonded]


def molecule_labels(bonded, n):
    """Bonded pair keys -> molecule (connected component) label per atom; single atoms included."""
    i, j = bonded // n, bonded % n
    return connected_components(coo_matrix((np.ones(len(bonded)), (i, j)), shape=(n, n)), directed=False)[1]


def molecule_formulas(label, types):
    """Formula of each molecule label."""
    counts = np.zeros((label.max() + 1, len(ELEMENTS)), int)
    np.add.at(counts, (label, types - 1), 1)
    return [formula_string(r, ELEMENTS) for r in counts]


def frame_events(old, new, types):
    """old, new: molecule label per atom in two frames -> [(sorted reactant formulas, sorted product formulas)]."""
    n_old, n_new = old.max() + 1, new.max() + 1
    graph = coo_matrix((np.ones(len(old)), (old, n_old + new)), shape=(n_old + n_new,) * 2)
    comp = connected_components(graph, directed=False)[1]
    side_old, side_new = comp[:n_old], comp[n_old:]
    n_r = np.bincount(side_old, minlength=comp.max() + 1)
    n_p = np.bincount(side_new, minlength=comp.max() + 1)
    changed = np.flatnonzero((n_r != 1) | (n_p != 1))
    if len(changed) == 0:
        return []
    f_old, f_new = molecule_formulas(old, types), molecule_formulas(new, types)
    return [(sorted(f_old[k] for k in np.flatnonzero(side_old == c)),
             sorted(f_new[k] for k in np.flatnonzero(side_new == c))) for c in changed]


def species_counts(bonded, n, types):
    """Bonded pair keys -> {formula: count} over molecules."""
    names, mult = np.unique(molecule_formulas(molecule_labels(bonded, n), types), return_counts=True)
    return {str(f): int(k) for f, k in zip(names, mult)}


def extract(path, stride=10, m=8, dt_fs=None):
    """Species table (t_ps file time, V_nm3, formula, count) for every stride-th frame after the m-frame
    warm-up, and event table (t_ps, reactants, products) from every pair of consecutive frames after it.
    Events need single-frame resolution: at a 120 fs stride, 27 % of events merged several elementary
    steps (four or more reactants). An event is reported at the frame where its products first appear;
    formulas within a side are sorted and joined by ' + '."""
    dt_fs = parse_name(Path(path).name)["dt_fs"] if dt_fs is None else dt_fs
    tracker, rows, ev, prev = BondTracker(m), [], [], None
    for k, (step, L, types, xyz) in enumerate(read_frames(path)):
        bonded = tracker.update(close_pairs(xyz, L, types))
        if k < m:
            continue
        t = step * dt_fs / 1000.0
        label = molecule_labels(bonded, len(types))
        if prev is not None:
            ev += [(t, " + ".join(r), " + ".join(q)) for r, q in frame_events(prev, label, types)]
        prev = label
        if (k - m) % stride == 0:
            names, mult = np.unique(molecule_formulas(label, types), return_counts=True)
            V = float(np.prod(L)) / 1000.0
            rows += [(t, V, str(f), int(c)) for f, c in zip(names, mult)]
    return (pd.DataFrame(rows, columns=["t_ps", "V_nm3", "formula", "count"]),
            pd.DataFrame(ev, columns=["t_ps", "reactants", "products"]))
