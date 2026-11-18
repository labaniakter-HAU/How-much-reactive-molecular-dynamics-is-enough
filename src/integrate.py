# file: src/integrate.py
"""Fixed-step RK4 for batched kinetic ODEs."""
import torch


def rk4(f, c0, T, t, n_sub=2):
    """Integrate dc/dt = f(c, T) on a shared grid t (n_t,) for a batch c0 (B, n_s), T (B,).
    Returns (n_t, B, n_s).
    shortcut: fixed step without stiffness control (rates are capped in CRNN.forward); switch to an
    implicit solver if real-data fits show oscillation or NaN."""
    out, c = [c0], c0
    for i in range(len(t) - 1):
        h = (t[i + 1] - t[i]) / n_sub
        for _ in range(n_sub):
            k1 = f(c, T)
            k2 = f(c + 0.5 * h * k1, T)
            k3 = f(c + 0.5 * h * k2, T)
            k4 = f(c + h * k3, T)
            c = c + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out.append(c)
    return torch.stack(out)
