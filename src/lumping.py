# file: src/lumping.py
"""Map species formulas to lumped classes (design section 4).

assign_class() holds the chemistry judgement and is written by the candidate (Task 6).
Amounts: a class with a single member formula is counted in molecules; a class with several members is
counted in carbon atoms (hydrogen if some member has no carbon), so 2 x C2H6 and 1 x C4H10 weigh the
same. Molecule counts with a mean composition broke the element balance by up to 55 % in prototyping.
event_type() gives an event's net change of class amounts and its reactant classes; events that change
no class amount are invisible at the lumped level and return None. build_tables.py merges events with the
same sign pattern and reactant classes into one lumped type.
"""
import numpy as np

from formula import atoms, composition_matrix

ELEMENTS = ["C", "H"]
# radicals (odd H count for C/H species) are kept apart from closed-shell species: lumping C2H5 with C2H4
# and C2H6 broke mass action in V0b (test error 0.23-1.29 against 0.08 with radicals apart, 2026-11-04)
CLASSES = ["CH4", "H2", "H", "C2", "C2_rad", "C3_C4", "C3_C4_rad", "C5_C10", "C5_C10_rad", "C11plus",
           "radical_small"]


def assign_class(formula):
    """Return one name from CLASSES (Task 6; the user delegated these rules, decision D1 of 2026-11-04).
    Splitting C11+ at C20 or C5-C10 at C7 did not lower the validation error (design section 5)."""
    a = atoms(formula)
    c, h = a.get("C", 0), a.get("H", 0)
    if c == 0:  # H-only species: H atom on its own, H2 and H-only clusters (H3, ...) together
        return "H" if formula == "H" else "H2"
    if c == 1:
        return "CH4" if h == 4 else "radical_small"
    if c > 10:  # large clusters stay in one class whatever their H count
        return "C11plus"
    size = "C2" if c == 2 else "C3_C4" if c <= 4 else "C5_C10"
    return size + "_rad" if h % 2 else size  # odd H count: radical (bare C2 counts as closed shell)


def class_units(formulas):
    """Counting unit per class, fixed once from every formula seen: 'molecule' for a single-member class,
    otherwise 'C', or 'H' if some member has no carbon."""
    members = {}
    for f in formulas:
        members.setdefault(assign_class(f), set()).add(f)
    return {c: "molecule" if len(members.get(c, ())) <= 1 else
            "C" if all(atoms(f)["C"] for f in members[c]) else "H" for c in CLASSES}


def molecule_amounts(formulas, units):
    """Class index and amount (in class units) of one molecule of each formula."""
    idx = np.array([CLASSES.index(assign_class(f)) for f in formulas], dtype=int)
    amt = np.array([1.0 if units[CLASSES[i]] == "molecule" else float(atoms(f)[units[CLASSES[i]]])
                    for f, i in zip(formulas, idx)])
    return idx, amt


def lump(names, X, units):
    """names (species), X (frames x species counts) -> Y (frames x classes amounts),
    A (frames x classes x elements atoms)."""
    M = composition_matrix(names, ELEMENTS)
    idx, amt = molecule_amounts(names, units)
    Y = np.zeros((X.shape[0], len(CLASSES)))
    A = np.zeros((X.shape[0], len(CLASSES), len(ELEMENTS)))
    for c in np.unique(idx):
        sel = idx == c
        Y[:, c] = X[:, sel] @ amt[sel]
        A[:, c, :] = X[:, sel] @ M[sel]
    return Y, A


def event_type(reactants, products, units):
    """Formula lists of one event -> (net class change tuple, sorted reactant classes tuple), or None."""
    s = np.zeros(len(CLASSES))
    i, a = molecule_amounts(products, units)
    np.add.at(s, i, a)
    i, a = molecule_amounts(reactants, units)
    np.subtract.at(s, i, a)
    if not s.any():
        return None
    return tuple(float(x) for x in s), tuple(sorted(CLASSES[k] for k in i))
