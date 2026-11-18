# file: src/kinetics.py
"""Event-anchored lumped kinetic model in CRNN form (design section 5).

For reaction j: ln r_j = ln A_j - E_a,j / (R T) + sum_i o_ij ln c_i, and dc/dt = nu @ r.
Reactions start as observed lumped event types: nu_j is the type's mean net class change and o_ij counts
its reactant molecules. Orders stay fixed, because classes counted in atoms make |nu| differ from
molecularity. E_a is in units of 100 kJ/mol. A dropped reaction keeps its slot with active = 0.
Fine-tuning (dnu) may change nu only within each reaction's own classes and keeps carbon balanced.
"""
import torch

R_KJ = 8.314462618e-3  # kJ / (mol K)


class LumpedKinetics(torch.nn.Module):
    def __init__(self, nu, orders, lnA, Ea, carbon=None, ln_r_max=12.0):
        """nu, orders: classes x reactions; lnA, Ea: (reactions,); carbon: carbon atoms per class unit."""
        super().__init__()
        f64 = lambda x: torch.as_tensor(x, dtype=torch.float64)
        self.register_buffer("nu0", f64(nu).clone())
        self.register_buffer("orders", f64(orders).clone())
        self.register_buffer("active", torch.ones(self.nu0.shape[1], dtype=torch.float64))
        self.register_buffer("carbon", f64(carbon) if carbon is not None else torch.zeros(self.nu0.shape[0], dtype=torch.float64))
        self.lnA = torch.nn.Parameter(f64(lnA).clone())
        self.Ea = torch.nn.Parameter(f64(Ea).clone())
        self.dnu = torch.nn.Parameter(torch.zeros_like(self.nu0), requires_grad=False)
        self.ln_r_max = ln_r_max

    def nu(self):
        own = (self.nu0 != 0).to(self.nu0.dtype)
        d, e = self.dnu * own, self.carbon[:, None] * own
        d = d - e * ((e * d).sum(0) / (e * e).sum(0).clamp_min(1e-12))  # no net carbon change
        return (self.nu0 + d) * self.active

    def ln_k(self, T):
        """T (B,) -> ln k (B, n_r)."""
        return self.lnA - 100.0 * self.Ea / (R_KJ * T[:, None])

    def rates(self, c, T):
        """c (B, n_c) per volume, T (B,) -> r (B, n_r) per volume per time."""
        ln_r = self.ln_k(T) + torch.log(c.clamp_min(1e-12)) @ self.orders
        return torch.exp(ln_r.clamp(max=self.ln_r_max)) * self.active

    def forward(self, c, T):
        return self.rates(c, T) @ self.nu().T

    def drop(self, j):
        with torch.no_grad():
            self.active[j] = 0.0

    def n_active(self):
        return int(self.active.sum())
