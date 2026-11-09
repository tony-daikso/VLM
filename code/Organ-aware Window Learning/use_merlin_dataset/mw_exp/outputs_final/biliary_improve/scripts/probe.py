"""Step 5: linear probes E0-E4 on cached embeddings only.

E0 baseline window | E1_<w> single window | E2 concat all windows | E3 per-unit softmax gate | E4 fixed preset
All findings are trained jointly (one linear head per finding, class-balanced BCE); in E3 all findings of a
unit share that unit's gate. A finding mapped to several units uses the mean over its present units; the
fraction of its units that are present is appended as one extra feature (organ absence is information).
A finding with `agg: max` in finding_organ_map.yaml instead scores each of its present units with the shared
head (+ a per-unit bias) and takes the max over units (MIL; experiment B, fracture -> T8..L5).
Units of organ_map.yaml that are not in emb/ are read from emb_bones/ (src/extract_bones.py; `bone_pool`
mean|max picks emb / emb_max). emb_bones has only baseline / soft_tissue / bone windows; the other windows of a
bone unit are filled from `bone_window_fallback` (default soft_tissue). With --standardize, bone units get
per-unit statistics so the pooled statistics of the original 12 units are unchanged.
HP grid lr x wd selected on validation mean AUROC (seed 0), then 3 seeds with the chosen HP.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from common import load_cfg, log_progress


def auroc(y, p):
    from scipy.stats import rankdata
    m = ~np.isnan(y)
    y, p = y[m], p[m]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_data(cfg, drop_test=False):
    """Assemble [N, U, W, D] from the per-window caches emb/{window_key}/{id}.npz for the windows in windows.yaml.
    Only cases that have every window are used. The assembled matrix is cached in results/emb_matrix_*.npz."""
    from extract import window_key
    wins = list(cfg["windows"].keys())
    wkeys = [window_key(n, w) for n, w in cfg["windows"].items()]
    lab = pd.read_csv(cfg["labels_file"]).set_index("study_id")
    spl = pd.read_csv(cfg["split_file"]).set_index("study_id")["split"]
    findings = [f for f in cfg["finding_map"] if f in lab.columns]
    sets = [{p.stem for p in (cfg["out"] / "emb" / k).glob("*.npz") if not p.stem.endswith(".tmp")} for k in wkeys]
    want = list(cfg["organ_map"].keys())
    z0 = np.load(cfg["out"] / "emb" / wkeys[0] / f"{sorted(sets[0])[0]}.npz")
    main_units = list(z0["organ_names"])
    bone_units = [u for u in want if u not in main_units]
    bdir = Path(cfg.get("bone_emb_dir") or cfg["out"] / "emb_bones")
    fb = cfg.get("bone_window_fallback", {})
    bkey = {k: k if (bdir / k).is_dir() else window_key(fb.get(w, "soft_tissue"), cfg["windows"][fb.get(w, "soft_tissue")])
            for w, k in zip(wins, wkeys)}
    bpool = "emb_max" if cfg.get("bone_pool", "mean") == "max" else "emb"
    if bone_units:
        sets += [{p.stem for p in (bdir / k).glob("*.npz")} for k in set(bkey.values())]
    keep = spl[spl != "test"].index if drop_test else spl.index
    ids = sorted(set.intersection(*sets) & set(lab.index) & set(keep))
    tag = f"+bones_{bpool}" if bone_units else ""
    cache = cfg["out"] / "results" / f"emb_matrix_{'+'.join(wkeys)}{tag}{'_notest' if drop_test else ''}.npz"
    if cache.exists() and list(np.load(cache)["ids"]) == ids:
        z = np.load(cache)
        E, P, units = z["E"], z["P"], list(z["units"])
    else:
        E, P = [], []
        for cid in ids:
            zs = [np.load(cfg["out"] / "emb" / k / f"{cid}.npz") for k in wkeys]
            e, pr = np.stack([z["emb"] for z in zs], 1), zs[0]["organ_present"]
            if bone_units:
                zb = {k: np.load(bdir / k / f"{cid}.npz") for k in set(bkey.values())}
                bn = list(next(iter(zb.values()))["organ_names"])
                bi = [bn.index(u) for u in bone_units]
                eb = np.stack([zb[bkey[k]][bpool][bi] for k in wkeys], 1)
                e = np.concatenate([e, eb.astype(e.dtype)], 0)
                pr = np.concatenate([pr, next(iter(zb.values()))["organ_present"][bi]])
            E.append(e); P.append(pr)
        units = main_units + bone_units
        E, P = np.stack(E), np.stack(P)
        np.savez(cache, E=E, P=P, ids=np.array(ids), units=np.array(units))
    E = torch.from_numpy(E.astype(np.float32))
    P = torch.from_numpy(P.astype(np.float32))
    Y = lab.loc[ids, findings].values.astype(np.float32)
    S = spl.loc[ids].values
    return ids, E, P, Y, S, units, wins, findings


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


def build_trap_inputs(cfg, ids, age_bin_years=5):
    """Shortcut 'trap' inputs from Merlin metadata (training-only branch, after the RSNA-knee 33rd-place idea):
    age bins -> random-effect table a[bin, finding]; sex / contrast / phase / vendor / kVp / slice / mA -> small MLP g."""
    m = pd.read_csv(cfg["metadata_file"]).drop_duplicates("study id").set_index("study id").reindex(ids)
    age = m["Age"].fillna(m["Age"].median()).clip(0, 100).values
    b = np.minimum((age // age_bin_years).astype(int), 100 // age_bin_years)
    def z(c):
        v = m[c].astype(float); v = v.fillna(v.median()).values
        return (v - v.mean()) / (v.std() + 1e-6)
    X = np.c_[(m["Gender"] == "Male").astype(float), m["contrast"].fillna(True).astype(float),
              *[(m["phase"] == ph).astype(float) for ph in ("portal_venous", "arterial_late", "arterial_early", "native")],
              *[(m["manufacturer"] == v).astype(float) for v in ("GE MEDICAL SYSTEMS", "SIEMENS")],
              z("kvp"), z("slicethickness"), z("xraytubecurrent")]
    az = (age - age.mean()) / age.std()
    Xp = np.c_[X, az, az ** 2]
    return {"bin": torch.from_numpy(b).long(), "X": torch.from_numpy(X).float(), "X_plus": torch.from_numpy(Xp).float(),
            "n_bins": int(b.max()) + 1, "age": age}


def inlp_remove_age(E, P, S, age, n_iter=20, ridge=1.0):
    """Iterative null-space projection, separately per (organ unit, window): repeatedly fit a ridge regression
    age ~ embedding on train rows (present organs), and project the fitted direction out. Returns new E and
    a report of val R^2 for predicting age before/after (mean over unit x window)."""
    E = E.clone()
    tr, va = torch.from_numpy(S == "train"), torch.from_numpy(S == "val")
    a = torch.from_numpy((age - age.mean()) / age.std()).float()
    U, W, D = E.shape[1], E.shape[2], E.shape[3]
    def r2(X, y, w):
        pr = X @ w; return float(1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    before, after = [], []
    for u in range(U):
        mtr, mva = tr & (P[:, u] > 0), va & (P[:, u] > 0)
        for w in range(W):
            X = E[:, u, w]
            Xt, yt = X[mtr], a[mtr]
            fit = lambda Xa: torch.linalg.solve(Xa.T @ Xa + ridge * torch.eye(D), Xa.T @ yt)
            before.append(r2(X[mva], a[mva], fit(Xt)))
            Pm = torch.eye(D)
            for _ in range(n_iter):
                d = fit(Xt @ Pm)
                d = Pm @ d
                d = d / d.norm().clamp(min=1e-8)
                Pm = Pm - torch.outer(d, d)
            Xn = X @ Pm
            after.append(r2(Xn[mva], a[mva], fit(Xt @ Pm)))
            E[:, u, w] = Xn * (P[:, u:u + 1] > 0)
    return E, float(np.mean(before)), float(np.mean(after))


class Trap(nn.Module):
    """training-only bias branch: logit_train = logit_image + a[age_bin, f] + g(acquisition/demographics)[f]."""

    def __init__(self, n_bins, k, Fn, a_init):
        super().__init__()
        self.a = nn.Parameter(a_init.clone())                              # [n_bins, F], empirical-Bayes init
        self.g = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, Fn))
        nn.init.zeros_(self.g[2].weight); nn.init.zeros_(self.g[2].bias)  # zero-init output

    def forward(self, b, X):
        return self.a[b] + self.g(X)


def train_one(cfg, spec, data, lr, wd, seed, dev, max_epochs=100, patience=10, bs=256, return_state=False):
    ids, E, P, Y, S, units, wins, findings = data
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = S == "train", S == "val"
    fu = [[units.index(u) for u in cfg["finding_map"][f]["organs"]] for f in findings]
    def sites_of(f):
        fm = cfg["finding_map"][f]
        if fm.get("agg", "mean") != "max":
            return None
        return [[units.index(u) for u in us] for us in (fm.get("sites") or {"all": fm["organs"]}).values()]
    fsites = [sites_of(f) for f in findings]
    model = Probe(spec["mode"], E.shape[1], E.shape[2], E.shape[3], fu, spec.get("win_idx"),
                  spec.get("preset_idx"), n_rf=spec.get("n_rf", 0), seed=seed, finding_sites=fsites).to(dev)
    trap = None
    if spec.get("trap"):
        Ytr = torch.from_numpy(Y[tr]); vt = ~torch.isnan(Ytr); yt = Ytr.nan_to_num(0)
        btr = TRAP["bin"][torch.from_numpy(tr)]
        pf = (yt * vt).sum(0) / vt.sum(0).clamp(min=1)                       # overall rate per finding
        k_shrink = 20.0
        pos_b = torch.zeros(TRAP["n_bins"], len(findings)).index_add_(0, btr, yt * vt)
        n_b = torch.zeros(TRAP["n_bins"], len(findings)).index_add_(0, btr, vt.float())
        rate = (pos_b + k_shrink * pf) / (n_b + k_shrink)
        lg = lambda q: torch.log(q.clamp(1e-4, 1 - 1e-4) / (1 - q.clamp(1e-4, 1 - 1e-4)))
        trap = Trap(TRAP["n_bins"], TRAP["X_plus" if spec.get("trap_plus") else "X"].shape[1], len(findings),
                    lg(rate) - lg(pf)).to(dev)
        prior_sd = float(spec.get("trap_prior_sd", 0.5))
        Xkey = "X_plus" if spec.get("trap_plus") else "X"
        Bt, Xt = btr.to(dev), TRAP[Xkey][torch.from_numpy(tr)].to(dev)
    gate_params = [model.gate] if model.gate is not None else []
    head_params = [p for n, p in model.named_parameters() if n != "gate"]
    groups = [{"params": head_params, "lr": lr}]
    if gate_params:
        groups.append({"params": gate_params, "lr": lr * spec.get("gate_lr_mult", 1.0)})
    if trap is not None:
        groups.append({"params": list(trap.parameters()), "lr": lr, "weight_decay": 0.0})
    opt = torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    freeze_gate = int(spec.get("freeze_gate_epochs", 0))
    if gate_params and freeze_gate:
        model.gate.requires_grad_(False)
    Et, Pt, Yt = E[tr].to(dev), P[tr].to(dev), torch.from_numpy(Y[tr]).to(dev)
    Ev, Pv = E[va].to(dev), P[va].to(dev)
    valid = ~torch.isnan(Yt)
    npos = (Yt.nan_to_num(0) * valid).sum(0); nneg = valid.sum(0) - npos
    wpos = (valid.sum(0) / (2 * npos.clamp(min=1))); wneg = (valid.sum(0) / (2 * nneg.clamp(min=1)))
    best, best_state, bad = -1.0, None, 0
    n = len(Et)
    for ep in range(max_epochs):
        if gate_params and freeze_gate and ep == freeze_gate:
            model.gate.requires_grad_(True)
            bad = 0  # early stopping restarts once the gate starts learning
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            y, m = Yt[j], valid[j]
            logit = model(Et[j], Pt[j])
            if trap is not None:                       # shortcut absorbed here during training only
                logit = logit + trap(Bt[j], Xt[j])
            w = torch.where(y == 1, wpos, wneg) * m
            loss = (F.binary_cross_entropy_with_logits(logit, y.nan_to_num(0), reduction="none") * w).sum() / m.sum().clamp(min=1)
            if trap is not None:                       # Gaussian prior N(0, sd^2) on the bin effects, per-entry scale
                loss = loss + (trap.a ** 2).sum() / (2 * prior_sd ** 2) / valid.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Ev, Pv)).cpu().numpy()
        aucs = [auroc(Y[va][:, f], pv[:, f]) for f in range(len(findings))]
        score = float(np.nanmean(aucs)) if not all(np.isnan(aucs)) else 0.0
        if score > best + 1e-5:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience and not (gate_params and freeze_gate and ep < freeze_gate):
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(torch.cat([model(E[i:i + 1024].to(dev), P[i:i + 1024].to(dev))
                                        for i in range(0, len(E), 1024)])).cpu().numpy()
    gate = torch.softmax(model.gate, -1).detach().cpu().numpy() if model.gate is not None else None
    if return_state:  # --save-model: also hand back the restored best-epoch head + how to rebuild it
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return best, prob, gate, {"state_dict": state, "finding_units": fu, "finding_sites": fsites}
    return best, prob, gate


def BONE_UNITS(cfg, units):
    """units appended from emb_bones (they come after the units found in emb/)."""
    from extract import window_key
    k = window_key(*next(iter(cfg["windows"].items())))
    d = cfg["out"] / "emb" / k
    z = np.load(next(p for p in d.glob("*.npz") if not p.stem.endswith(".tmp")))
    main = list(z["organ_names"])
    return [u for u in units if u not in main]


def experiments(cfg, units, wins):
    b = wins.index("baseline")
    ex = {"E0": {"mode": "single", "win_idx": b}}
    for i, w in enumerate(wins):
        if w != "baseline":
            ex[f"E1_{w}"] = {"mode": "single", "win_idx": i}
    ex["E2"] = {"mode": "concat"}
    ex["E3"] = {"mode": "gate"}
    ex["E3_late"] = {"mode": "late"}                                             # per-window heads, gated logits
    ex["E0_rf"] = {"mode": "randfeat", "win_idx": b, "n_rf": (len(wins) - 1) * 448}  # capacity control for E2
    ex["E3_slow"] = {"mode": "gate", "gate_lr_mult": 0.1}                       # gate learns 10x slower
    ex["E3_2stage"] = {"mode": "gate", "gate_lr_mult": 0.1, "freeze_gate_epochs": 10}  # heads first, then gate
    ex["E0_trap"] = dict(ex["E0"], trap=True)
    ex["E3_late_trap"] = dict(ex["E3_late"], trap=True)
    ex["E0_trap2"] = dict(ex["E0"], trap=True, trap_plus=True, trap_prior_sd=1.0)          # stronger trap
    ex["E3_late_trap2"] = dict(ex["E3_late"], trap=True, trap_plus=True, trap_prior_sd=1.0)
    pre = cfg["presets"]
    ex["E4"] = {"mode": "preset", "preset_idx": [wins.index(pre.get(u, pre["default"])) for u in units]}
    return ex


def save_model(a, cfg, name, spec, hp, states, std_stats, E, units, wins, findings, ids, S):
    """--save-model: everything needed to re-run the head on new embeddings (see infer_head.py).
    <dir>/<name>.pt   torch dict: per-seed state_dicts (incl. gate logits), standardize mu/sd [U,W,D], meta
    <dir>/<name>_meta.json  the same meta (human readable, no tensors)"""
    from extract import window_key
    d = Path(a.save_model)
    d.mkdir(parents=True, exist_ok=True)
    bone = BONE_UNITS(cfg, units)
    fb = cfg.get("bone_window_fallback", {})
    bdir = Path(cfg.get("bone_emb_dir") or cfg["out"] / "emb_bones")
    wkeys = [window_key(n, w) for n, w in cfg["windows"].items()]
    bkey = {k: k if (bdir / k).is_dir() else window_key(fb.get(w, "soft_tissue"), cfg["windows"][fb.get(w, "soft_tissue")])
            for w, k in zip(wins, wkeys)}
    meta = {
        "experiment": name, "spec": spec, "mode": spec["mode"],
        "hp": hp, "seeds": [s["seed"] for s in states], "seed_val_auc": [s["val_auc"] for s in states],
        "max_epochs": a.max_epochs, "patience": 10, "batch_size": 256, "optimizer": "Adam (gate lr x gate_lr_mult)",
        "early_stopping": "best epoch by val mean AUROC (as in probe.py)",
        "standardize": bool(a.standardize), "inlp": a.inlp, "drop_test": a.drop_test,
        "standardize_stats": ("Merlin train split, present organs only; main units share pooled [W,D] stats, "
                              "bone units per-unit stats; absent units -> 0" if a.standardize else None),
        "n_train": int((S == "train").sum()), "n_val": int((S == "val").sum()), "n_test_loaded": int((S == "test").sum()),
        "shape_UWD": list(E.shape[1:]),
        "units": list(units), "main_units": [u for u in units if u not in bone], "bone_units": bone,
        "windows": list(wins), "window_keys": wkeys, "bone_window_keys": [bkey[k] for k in wkeys],
        "bone_pool": cfg.get("bone_pool", "mean"), "bone_emb_key": "emb_max" if cfg.get("bone_pool", "mean") == "max" else "emb",
        "findings": list(findings),
        "finding_units": states[0]["finding_units"], "finding_sites": states[0]["finding_sites"],
        "finding_map": {f: cfg["finding_map"][f] for f in findings},
        "config": a.config, "labels_file": str(cfg["labels_file"]), "split_file": str(cfg["split_file"]),
        "emb_dir": str(cfg["out"] / "emb"), "bone_emb_dir": str(bdir),
    }
    ckpt = {"meta": meta,
            "state_dicts": [s["state_dict"] for s in states],
            "std_mu": std_stats["mu"].float() if std_stats else None,   # [U,W,D] fp32
            "std_sd": std_stats["sd"].float() if std_stats else None}
    torch.save(ckpt, d / f"{name}.pt")
    meta["gate_softmax_per_seed"] = [torch.softmax(s["state_dict"]["gate"], -1).tolist()
                                     for s in states if "gate" in s["state_dict"]]
    (d / f"{name}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"saved model -> {d / (name + '.pt')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--standardize", action="store_true",
                    help="z-score each (window, dim) with train statistics of present organs; absent organs stay 0")
    ap.add_argument("--results-subdir", default="", help="write to results/<subdir>/ (keeps runs apart)")
    ap.add_argument("--inlp", type=int, default=0, help="INLP iterations removing linear age directions (0 = off)")
    ap.add_argument("--drop-test", action="store_true", help="exclude test ids before loading any embedding")
    ap.add_argument("--save-model", default=None, metavar="DIR",
                    help="also save per-seed head state_dicts + standardize stats + meta to DIR/<exp>.pt and "
                         "DIR/<exp>_meta.json (off by default; nothing else changes)")
    ap.add_argument("--hp", nargs=2, type=float, default=None, metavar=("LR", "WD"),
                    help="skip the lr x wd grid and use these hyper-parameters (e.g. the 'chosen' of a *_grid.json)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    dev = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    data = load_data(cfg, drop_test=a.drop_test)
    ids, E, P, Y, S, units, wins, findings = data
    std_stats = None
    if a.standardize:
        tr = torch.from_numpy(S == "train")
        main_idx = [i for i, u in enumerate(units) if i < len(units) - len(BONE_UNITS(cfg, units))]
        Et, Pt = E[tr][:, main_idx], P[tr][:, main_idx] > 0        # [n,U,W,D], [n,U]
        rows = Et[Pt]                                              # [m,W,D] present organs only (original units)
        mu, sd = rows.mean(0), rows.std(0).clamp(min=1e-6)
        E = E.clone()
        E[:, main_idx] = ((E[:, main_idx] - mu) / sd) * (P[:, main_idx][..., None, None] > 0)
        tri = torch.where(tr)[0]
        std_mu = torch.zeros(E.shape[1:]); std_sd = torch.ones(E.shape[1:])   # [U,W,D], kept for --save-model
        std_mu[main_idx], std_sd[main_idx] = mu, sd
        for i in range(len(main_idx), len(units)):                 # bone units: per-unit statistics
            r = E[tri, i][P[tri, i] > 0]
            std_mu[i], std_sd[i] = r.mean(0), r.std(0).clamp(min=1e-6)
            E[:, i] = ((E[:, i] - r.mean(0)) / r.std(0).clamp(min=1e-6)) * (P[:, i][:, None, None] > 0)
        std_stats = {"mu": std_mu, "sd": std_sd, "n_train": int(tr.sum())}
        data = (ids, E, P, Y, S, units, wins, findings)
    print(f"N={len(ids)} split={pd.Series(S).value_counts().to_dict()} E={tuple(E.shape)} findings={len(findings)} "
          f"standardize={a.standardize}")
    global TRAP
    need_trap = any(sp.get("trap") for nm, sp in experiments(cfg, units, wins).items() if not a.only or nm in a.only)
    if need_trap or a.inlp:
        TRAP = build_trap_inputs(cfg, ids)
        print(f"trap: {TRAP['n_bins']} age bins, {TRAP['X'].shape[1]} acquisition/demographic features")
    if a.inlp:
        E, r2b, r2a = inlp_remove_age(E, P, S, TRAP["age"], n_iter=a.inlp)
        data = (ids, E, P, Y, S, units, wins, findings)
        print(f"INLP x{a.inlp}: val R^2 for predicting age from organ embeddings {r2b:.3f} -> {r2a:.3f}", flush=True)
    lrs, wds = [1e-4, 1e-3, 1e-2], [0, 1e-4, 1e-3]
    out = cfg["out"] / "results" / a.results_subdir
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in experiments(cfg, units, wins).items():
        if a.only and name not in a.only:
            continue
        grid_wds = [1e-4, 1e-3, 1e-2] if spec["mode"] in ("concat", "late", "randfeat") else wds
        grid = []
        if a.hp:
            hp = {"lr": a.hp[0], "wd": a.hp[1], "val_auc": None, "source": "--hp"}
        else:
            for lr, wd in itertools.product(lrs, grid_wds):
                s, _, _ = train_one(cfg, spec, data, lr, wd, cfg["seed"], dev, a.max_epochs)
                grid.append({"lr": lr, "wd": wd, "val_auc": s})
            hp = max(grid, key=lambda g: g["val_auc"])
        probs, gates, vals, states = [], [], [], []
        for sd in range(a.seeds):
            r = train_one(cfg, spec, data, hp["lr"], hp["wd"], cfg["seed"] + sd, dev, a.max_epochs,
                          return_state=bool(a.save_model))
            s, p, g = r[:3]
            probs.append(p); vals.append(s)
            if g is not None:
                gates.append(g)
            if a.save_model:
                states.append(dict(r[3], seed=cfg["seed"] + sd, val_auc=s))
        if a.save_model:
            save_model(a, cfg, name, spec, hp, states, std_stats, E, units, wins, findings, ids, S)
        np.savez(out / f"{name}.npz", ids=np.array(ids), split=S, prob=np.stack(probs), findings=np.array(findings),
                 units=np.array(units), windows=np.array(wins), val_auc=np.array(vals),
                 gate=np.stack(gates) if gates else np.zeros(0))
        (out / f"{name}_grid.json").write_text(json.dumps({"grid": grid, "chosen": hp, "seed_val_auc": vals}, indent=1))
        print(f"{name}: hp={hp} val_auc seeds={np.round(vals, 4).tolist()}", flush=True)
    log_progress(f"probe.py: trained {a.only or 'all'} on N={len(ids)} -> {out}")


if __name__ == "__main__":
    main()
