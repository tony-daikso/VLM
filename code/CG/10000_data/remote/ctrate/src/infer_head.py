"""Standalone inference for a head saved by `probe.py --save-model` (OWL multi-window organ probe, e.g. E3_late expB).

Input = per-case embeddings exactly as written by the extraction scripts:
  --emb-dir       <dir>/<window_key>/<id>.npz   from src/extract.py        (emb [U_main,D], organ_present, organ_names)
  --bone-emb-dir  <dir>/<window_key>/<id>.npz   from src/extract_bones.py  (emb / emb_max [U_bone,D], organ_present, organ_names)
The window keys (5 windows; the bone dir only has baseline/soft_tissue/bone, the others fall back as in training),
unit order, standardize mu/sd (Merlin train) and finding order are all read from the checkpoint.
Output: CSV  study_id, <14 findings>  = sigmoid probability averaged over the saved seeds
        (+ <finding>__seed<k> columns with --per-seed).

Usage:
  python infer_head.py --model .../models/E3_late_expB/E3_late.pt \
      --emb-dir .../outputs_final/emb --bone-emb-dir .../outputs/emb_bones --out probs.csv [--ids ids.txt] [--per-seed]
Dependencies: numpy, pandas, torch (no project imports).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---- copied verbatim from src/probe.py (keep in sync) ----
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
# ---- end of copy ----


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    U, W, D = m["shape_UWD"]
    spec = m["spec"]
    models = []
    for sd in ck["state_dicts"]:
        net = Probe(spec["mode"], U, W, D, m["finding_units"], spec.get("win_idx"), spec.get("preset_idx"),
                    n_rf=spec.get("n_rf", 0), finding_sites=m["finding_sites"])
        net.load_state_dict(sd)
        models.append(net.to(device).eval())
    return ck, models


def standardize(E, P, ck):
    """E [N,U,W,D] raw, P [N,U] -> z-scored with the saved Merlin-train stats; absent units -> 0 (as in probe.py)."""
    if not ck["meta"]["standardize"]:
        return E
    return ((E - ck["std_mu"]) / ck["std_sd"]) * (P[..., None, None] > 0)


def case_ids(meta, emb_dir, bone_dir):
    dirs = [emb_dir / k for k in meta["window_keys"]]
    if meta["bone_units"]:
        dirs += [bone_dir / k for k in set(meta["bone_window_keys"])]
    sets = [{p.stem for p in d.glob("*.npz") if not p.stem.endswith(".tmp")} for d in dirs]
    return sorted(set.intersection(*sets))


def load_case(cid, meta, emb_dir, bone_dir):
    """-> E [U,W,D] float32 (raw), P [U] in the checkpoint's unit order."""
    zs = [np.load(emb_dir / k / f"{cid}.npz") for k in meta["window_keys"]]
    names = list(zs[0]["organ_names"])
    mi = [names.index(u) for u in meta["main_units"]]
    e = np.stack([z["emb"][mi] for z in zs], 1).astype(np.float32)
    p = zs[0]["organ_present"][mi].astype(np.float32)
    if meta["bone_units"]:
        zb = {k: np.load(bone_dir / k / f"{cid}.npz") for k in set(meta["bone_window_keys"])}
        z0 = next(iter(zb.values()))
        bn = list(z0["organ_names"])
        bi = [bn.index(u) for u in meta["bone_units"]]
        eb = np.stack([zb[k][meta["bone_emb_key"]][bi] for k in meta["bone_window_keys"]], 1).astype(np.float32)
        e = np.concatenate([e, eb], 0)
        p = np.concatenate([p, z0["organ_present"][bi].astype(np.float32)])
    return e, p


@torch.no_grad()
def predict(models, ck, E, P, device="cpu", bs=512):
    """E [N,U,W,D] raw float tensor, P [N,U] -> probs [S,N,F] (one row block per seed)."""
    out = []
    for i in range(0, len(E), bs):
        e = standardize(E[i:i + bs].float(), P[i:i + bs].float(), ck).to(device)
        p = P[i:i + bs].float().to(device)
        out.append(torch.stack([torch.sigmoid(m(e, p)) for m in models]).cpu())
    return torch.cat(out, 1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="<exp>.pt written by probe.py --save-model")
    ap.add_argument("--emb-dir", required=True, help="extract.py output: <dir>/<window_key>/<id>.npz")
    ap.add_argument("--bone-emb-dir", default=None, help="extract_bones.py output (required if the model has bone units)")
    ap.add_argument("--ids", default=None, help="text file, one study id per line (default: every id with all files)")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--per-seed", action="store_true", help="also write each seed's probability")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=1000, help="cases loaded into memory at a time")
    a = ap.parse_args()
    ck, models = load_model(a.model, a.device)
    meta = ck["meta"]
    emb_dir = Path(a.emb_dir)
    bone_dir = Path(a.bone_emb_dir) if a.bone_emb_dir else None
    if meta["bone_units"] and bone_dir is None:
        sys.exit(f"model uses {len(meta['bone_units'])} bone units: --bone-emb-dir is required")
    avail = case_ids(meta, emb_dir, bone_dir)
    if a.ids:
        want = [l.strip() for l in open(a.ids) if l.strip()]
        miss = sorted(set(want) - set(avail))
        if miss:
            print(f"WARNING: {len(miss)} ids skipped (missing a window / bone file), e.g. {miss[:5]}", file=sys.stderr)
        ids = [c for c in want if c in set(avail)]
    else:
        ids = avail
    print(f"model={meta['experiment']} seeds={meta['seeds']} units={len(meta['units'])} windows={meta['windows']} "
          f"findings={len(meta['findings'])} cases={len(ids)}", file=sys.stderr)
    probs = []
    for i in range(0, len(ids), a.chunk):
        ec, pc = zip(*[load_case(c, meta, emb_dir, bone_dir) for c in ids[i:i + a.chunk]])
        probs.append(predict(models, ck, torch.from_numpy(np.stack(ec)), torch.from_numpy(np.stack(pc)), a.device))
        print(f"  {min(i + a.chunk, len(ids))}/{len(ids)}", file=sys.stderr)
    pr = np.concatenate(probs, 1)                             # [S,N,F]
    df = pd.DataFrame(pr.mean(0), columns=meta["findings"])
    df.insert(0, "study_id", ids)
    if a.per_seed:
        for s, seed in enumerate(meta["seeds"]):
            for f, name in enumerate(meta["findings"]):
                df[f"{name}__seed{seed}"] = pr[s, :, f]
    df.to_csv(a.out, index=False, float_format="%.6f")
    print(f"wrote {a.out} ({len(df)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
