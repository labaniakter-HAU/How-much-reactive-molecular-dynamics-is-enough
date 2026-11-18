# file: src/formula.py
"""Chemical formula helpers."""
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def atoms(formula):
    """'C4H10' -> Counter({'C': 4, 'H': 10})."""
    out = Counter()
    for el, n in _TOKEN.findall(formula):
        out[el] += int(n) if n else 1
    return out


def formula_string(counts, elements):
    """(4, 10), ('C', 'H') -> 'C4H10'. Zero counts are dropped; a count of 1 has no digit."""
    return "".join(f"{e}{n if n > 1 else ''}" for e, n in zip(elements, counts) if n)


def composition_matrix(names, elements):
    """Species x elements atom counts."""
    return np.array([[atoms(n)[e] for e in elements] for n in names], dtype=float)
