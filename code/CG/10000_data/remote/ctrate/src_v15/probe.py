"""Step 5: linear probes E0-E4 on cached embeddings only.

E0 baseline window | E1_<w> single window | E2 concat all windows | E3 per-unit softmax gate | E4 fixed preset
All findings are trained jointly (one linear head per finding, class-balanced BCE); in E3 all findings of a
unit share that unit's gate. A finding mapped to several units uses the mean over its present units; the
fraction of its units that are present is appended as one extra feature (organ absence is information).
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


def load_data(cfg):
    """Assemble [N, U, W, D] from the per-window caches emb/{window_key}/{id}.npz for the windows in windows.yaml.
    Only cases that have every window are used. The assembled matrix is cached in results/emb_matrix_*.npz."""
    from extract import window_key
    wins = list(cfg["windows"].keys())
    wkeys = [window_key(n, w) for n, w in cfg["windows"].items()]
    lab = pd.read_csv(cfg["labels_file"]).set_index("study_id")
    spl = pd.read_csv(cfg["split_file"]).set_index("study_id")["split"]
    findings = [f for f in cfg["finding_map"] if f in lab.columns]
    sets = [{p.stem for p in (cfg["out"] / "emb" / k).glob("*.npz") if not p.stem.endswith(".tmp")} for k in wkeys]
    ids = sorted(set.intersection(*sets) & set(lab.index) & set(spl.index))
    cache = cfg["out"] / "results" / f"emb_matrix_{'+'.join(wkeys)}.npz"
    if cache.exists() and list(np.load(cache)["ids"]) == ids:
        z = np.load(cache)
        E, P, units = z["E"], z["P"], list(z["units"])
    else:
        E, P = [], []
        for cid in ids:
            zs = [np.load(cfg["out"] / "emb" / k / f"{cid}.npz") for k in wkeys]
            E.append(np.stack([z["emb"] for z in zs], 1)); P.append(zs[0]["organ_present"])
        units = list(zs[0]["organ_names"])
        E, P = np.stack(E), np.stack(P)
        np.savez(cache, E=E, P=P, ids=np.array(ids), units=np.array(units))
    E = torch.from_numpy(E.astype(np.float32))
    P = torch.from_numpy(P.astype(np.float32))
    Y = lab.loc[ids, findings].values.astype(np.float32)
    S = spl.loc[ids].values
    return ids, E, P, Y, S, units, wins, findings


class Probe(nn.Module):
    def __init__(self, mode, U, W, D, finding_units, win_idx=None, preset_idx=None):
        super().__init__()
        self.mode, self.win_idx = mode, win_idx
        self.register_buffer("preset_idx", torch.tensor(preset_idx if preset_idx is not None else [0] * U))
        self.fu = finding_units
        Fn = len(finding_units)
        din = W * D if mode == "concat" else D
        self.gate = nn.Parameter(torch.zeros(U, W)) if mode == "gate" else None
        self.w = nn.Parameter(torch.randn(Fn, din + 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(Fn))
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
        raise ValueError(self.mode)

    def forward(self, E, P):
        Z = self.unit_repr(E)                                  # [N,U,d]
        A = P[:, None, :] * self.Mu[None]                      # [N,F,U] present units of each finding
        cnt = A.sum(-1, keepdim=True)
        H = torch.einsum("nfu,nud->nfd", A, Z) / cnt.clamp(min=1)
        frac = cnt / self.Mu.sum(-1)[None, :, None]
        H = torch.cat([H, frac], -1)
        return (H * self.w[None]).sum(-1) + self.b


def train_one(cfg, spec, data, lr, wd, seed, dev, max_epochs=100, patience=10, bs=256):
    ids, E, P, Y, S, units, wins, findings = data
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = S == "train", S == "val"
    fu = [[units.index(u) for u in cfg["finding_map"][f]["organs"]] for f in findings]
    model = Probe(spec["mode"], E.shape[1], E.shape[2], E.shape[3], fu, spec.get("win_idx"),
                  spec.get("preset_idx")).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Et, Pt, Yt = E[tr].to(dev), P[tr].to(dev), torch.from_numpy(Y[tr]).to(dev)
    Ev, Pv = E[va].to(dev), P[va].to(dev)
    valid = ~torch.isnan(Yt)
    npos = (Yt.nan_to_num(0) * valid).sum(0); nneg = valid.sum(0) - npos
    wpos = (valid.sum(0) / (2 * npos.clamp(min=1))); wneg = (valid.sum(0) / (2 * nneg.clamp(min=1)))
    best, best_state, bad = -1.0, None, 0
    n = len(Et)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            y, m = Yt[j], valid[j]
            logit = model(Et[j], Pt[j])
            w = torch.where(y == 1, wpos, wneg) * m
            loss = (F.binary_cross_entropy_with_logits(logit, y.nan_to_num(0), reduction="none") * w).sum() / m.sum().clamp(min=1)
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
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(torch.cat([model(E[i:i + 1024].to(dev), P[i:i + 1024].to(dev))
                                        for i in range(0, len(E), 1024)])).cpu().numpy()
    gate = torch.softmax(model.gate, -1).detach().cpu().numpy() if model.gate is not None else None
    return best, prob, gate


def experiments(cfg, units, wins):
    b = wins.index("baseline")
    ex = {"E0": {"mode": "single", "win_idx": b}}
    for i, w in enumerate(wins):
        if w != "baseline":
            ex[f"E1_{w}"] = {"mode": "single", "win_idx": i}
    ex["E2"] = {"mode": "concat"}
    ex["E3"] = {"mode": "gate"}
    pre = cfg["presets"]
    ex["E4"] = {"mode": "preset", "preset_idx": [wins.index(pre.get(u, pre["default"])) for u in units]}
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=100)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    dev = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    data = load_data(cfg)
    ids, E, P, Y, S, units, wins, findings = data
    print(f"N={len(ids)} split={pd.Series(S).value_counts().to_dict()} E={tuple(E.shape)} findings={len(findings)}")
    lrs, wds = [1e-4, 1e-3, 1e-2], [0, 1e-4, 1e-3]
    out = cfg["out"] / "results"
    for name, spec in experiments(cfg, units, wins).items():
        if a.only and name not in a.only:
            continue
        grid_wds = [1e-4, 1e-3, 1e-2] if spec["mode"] == "concat" else wds
        grid = []
        for lr, wd in itertools.product(lrs, grid_wds):
            s, _, _ = train_one(cfg, spec, data, lr, wd, cfg["seed"], dev, a.max_epochs)
            grid.append({"lr": lr, "wd": wd, "val_auc": s})
        hp = max(grid, key=lambda g: g["val_auc"])
        probs, gates, vals = [], [], []
        for sd in range(a.seeds):
            s, p, g = train_one(cfg, spec, data, hp["lr"], hp["wd"], cfg["seed"] + sd, dev, a.max_epochs)
            probs.append(p); vals.append(s)
            if g is not None:
                gates.append(g)
        np.savez(out / f"{name}.npz", ids=np.array(ids), split=S, prob=np.stack(probs), findings=np.array(findings),
                 units=np.array(units), windows=np.array(wins), val_auc=np.array(vals),
                 gate=np.stack(gates) if gates else np.zeros(0))
        (out / f"{name}_grid.json").write_text(json.dumps({"grid": grid, "chosen": hp, "seed_val_auc": vals}, indent=1))
        print(f"{name}: hp={hp} val_auc seeds={np.round(vals, 4).tolist()}", flush=True)
    log_progress(f"probe.py: trained {a.only or 'all'} on N={len(ids)} -> {out}")


if __name__ == "__main__":
    main()
