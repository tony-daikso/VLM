import torch, torch.nn as nn, torch.nn.functional as F
class Probe(nn.Module):
    """modes: single | concat | gate (E3: weighted sum of window embeddings) | preset |
    late (E3_late: one linear head per window, logits mixed by the per-organ window gate) |
    randfeat (E0_rf capacity control: baseline + fixed random ReLU features, same width as concat)."""

    def __init__(self, mode, U, W, D, finding_units, win_idx=None, preset_idx=None, n_rf=0, seed=0, finding_sites=None):
        """finding_sites: per finding None (mean over units) or a list of unit-index lists (sites) -> MIL max."""
        super().__init__()
        fs = finding_sites or [None] * len(finding_units)
        self.fm = [f for f, x in enumerate(fs) if x is not None]          # MIL findings
        if self.fm:
            S = max(len(fs[f]) for f in self.fm)
            site_of = torch.full((len(self.fm), U), -1, dtype=torch.long)
            for i, f in enumerate(self.fm):
                for k, us in enumerate(fs[f]):
                    site_of[i, us] = k
            self.register_buffer("site_of", site_of)                       # [Fm,U], -1 = not a unit of f
            self.register_buffer("site_ok", torch.tensor([[k < len(fs[f]) for k in range(S)] for f in self.fm]))
            dsh = (W, D + 1) if mode == "late" else ({"concat": W * D, "randfeat": D + n_rf}.get(mode, D) + 1,)
            self.ws = nn.Parameter(torch.randn(len(self.fm), S, *dsh) * 0.01)   # one shared head per site
            self.ub = nn.Parameter(torch.zeros(len(self.fm), U))                # per-unit bias
            self.sn = nn.Parameter(torch.zeros(len(self.fm), S))                # site with no present unit
            self.bm = nn.Parameter(torch.zeros(len(self.fm)))
        self.mode, self.win_idx = mode, win_idx
        self.register_buffer("preset_idx", torch.tensor(preset_idx if preset_idx is not None else [0] * U))
        self.fu = finding_units
        Fn = len(finding_units)
        din = {"concat": W * D, "randfeat": D + n_rf}.get(mode, D)
        self.gate = nn.Parameter(torch.zeros(U, W)) if mode in ("gate", "late") else None
        if mode == "late":
            self.w = nn.Parameter(torch.randn(Fn, W, D + 1) * 0.01)
            self.b = nn.Parameter(torch.zeros(Fn, W))
        else:
            self.w = nn.Parameter(torch.randn(Fn, din + 1) * 0.01)
            self.b = nn.Parameter(torch.zeros(Fn))
        if mode == "randfeat":
            g = torch.Generator().manual_seed(1234 + seed)
            self.register_buffer("R", torch.randn(D, n_rf, generator=g) / D ** 0.5)
        Mu = torch.zeros(Fn, U)
        for f, us in enumerate(finding_units):
            Mu[f, us] = 1.0
        self.register_buffer("Mu", Mu)  # finding -> units incidence

    def unit_repr(self, E):
        N, U, W, D = E.shape
        if self.mode == "single":
            return E[:, :, self.win_idx]
        if self.mode == "concat":
            return E.reshape(N, U, W * D)
        if self.mode == "gate":
            return torch.einsum("nuwd,uw->nud", E, torch.softmax(self.gate, -1))
        if self.mode == "preset":
            return E[:, torch.arange(U, device=E.device), self.preset_idx]
        if self.mode == "randfeat":
            x = E[:, :, self.win_idx]
            return torch.cat([x, torch.relu(x @ self.R)], -1)
        if self.mode == "late":
            return E  # [N,U,W,D]
        raise ValueError(self.mode)

    def forward(self, E, P):
        out = self.forward_mean(E, P)
        if not self.fm:
            return out
        out = out.clone()
        out[:, self.fm] = self.forward_max(E, P)
        return out

    def forward_max(self, E, P):
        """MIL findings: each unit is scored by its site's shared head (+ per-unit bias); site logit = max over its
        present units; finding logit = logsumexp over sites (a single site = plain max)."""
        Z = self.unit_repr(E)
        M = self.Mu[self.fm]                                     # [Fm,U]
        A = P[:, None, :] * M[None]                              # [N,Fm,U]
        frac = A.sum(-1) / M.sum(-1)[None]                       # [N,Fm]
        Wu = self.ws[torch.arange(len(self.fm))[:, None], self.site_of.clamp(min=0)]   # [Fm,U,...]
        if self.mode == "late":                                  # Z [N,U,W,D], Wu [Fm,U,W,D+1]
            L = torch.einsum("nuwd,fuwd->nfuw", Z, Wu[..., :-1]) + frac[:, :, None, None] * Wu[None, ..., -1]
            G = (M @ torch.softmax(self.gate, -1)) / M.sum(-1, keepdim=True)            # [Fm,W]
            L = (L * G[None, :, None]).sum(-1)
        else:
            L = torch.einsum("nud,fud->nfu", Z, Wu[..., :-1]) + frac[..., None] * Wu[None, ..., -1]
        L = L + self.ub[None]
        S = self.site_ok.shape[1]
        site = []
        for k in range(S):
            mk = (A > 0) & (self.site_of == k)[None]
            v = L.masked_fill(~mk, -1e4).max(-1).values
            site.append(torch.where(mk.any(-1), v, self.sn[None, :, k].expand_as(v)))
        St = torch.stack(site, -1).masked_fill(~self.site_ok[None], -1e4)     # [N,Fm,S]
        return torch.logsumexp(St, -1) + self.bm

    def forward_mean(self, E, P):
        Z = self.unit_repr(E)
        A = P[:, None, :] * self.Mu[None]                      # [N,F,U] present units of each finding
        cnt = A.sum(-1, keepdim=True)
        frac = cnt / self.Mu.sum(-1)[None, :, None]
        if self.mode == "late":
            H = torch.einsum("nfu,nuwd->nfwd", A, Z) / cnt[..., None].clamp(min=1)
            H = torch.cat([H, frac[..., None].expand(-1, -1, H.shape[2], 1)], -1)
            logit_w = (H * self.w[None]).sum(-1) + self.b      # [N,F,W]
            G = (self.Mu @ torch.softmax(self.gate, -1)) / self.Mu.sum(-1, keepdim=True)  # [F,W]
            return (logit_w * G[None]).sum(-1)
        H = torch.einsum("nfu,nud->nfd", A, Z) / cnt.clamp(min=1)
        H = torch.cat([H, frac], -1)
        return (H * self.w[None]).sum(-1) + self.b


TRAP = None  # set in main(): dict(bin=LongTensor[N], X=FloatTensor[N,k], n_bins=int)


