"""Experiment L: 1.5 mm per-rib units (rib15_full.py emb15/) in the bone site-wise MIL (E3_late late fusion,
bone_probe / vert15_probe / bone_expI recipe).

stages
  build : one matrix (train/val only; test never loaded) -> <out>/l_matrix.npz
            b17  [N,17,3,448]  3 mm bone units (bone_matrix.npz, mean pool)
            r3   [N,18,3,448]  3 mm single-rib units (Exp G rib_matrix.npz, mean pool)
            v15  [N,10,3,448]  1.5 mm T8..L5 (Exp J emb15, mean pool)
            r15  [N,18,3,448]  1.5 mm single-rib units (Exp L emb15, mean) ; r15x = max pool ; nvox15
  oof   : 5-fold OOF on TRAIN (inner 10 % early stop on hard-label AUROC; val never used to fit / stop / select),
          val = mean of the 5 fold models.
variant = '+'-joined tokens
  sites : Jadd  (default)  vert3 | rib3(2 big units) | pelvis | v15            (= Exp J add)
          L_add            Jadd + rib15 (18 units) as a 5th site
          L_repl           vert3 | rib15 | pelvis | v15                         (rib15 replaces the 3 mm rib units)
          G3               vert3 | rib3 per-rib (18 units, Exp G) | pelvis | v15 (fair 3 mm per-rib reference)
  rmax  : rib15 units use the max-pooled embedding ; rmm : mean & max averaged
  rlN   : rib instance loss weight N on the per-rib site (reported side/level of TRAIN rib positives = 1, side-only
          -> max over that side = 1; every present rib unit of a negative = 0)   (Exp I r18+rl recipe)
  lrc   : left-right contrast on the per-rib site: unit logit += <x_u - x_contralateral, h_c>
  vminK : rib15 unit present only if nvox15 >= K (default 200)
--subset tier0: train rows restricted to tier0 ids that have rib15 (interim); val = all val rows with rib15.
outputs <out>/oof/<subset>/oof_L_<variant>_s<k>.npz (ids, split, logit, y, site_logit, inst, inst_units, oof_train_auc)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import bone_probe as B  # noqa: E402
from probe import auroc  # noqa: E402

MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
LV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
RIBU = ["rib_left_1_4"] + [f"rib_left_{i}" for i in range(5, 13)] + ["rib_right_1_4"] + [f"rib_right_{i}" for i in range(5, 13)]


# ----------------------------------------------------------------------------------------------- build
def _emb_files(dirs):
    files = {}
    for d in dirs:
        for p in Path(d).glob("*.npz"):
            if ".tmp" not in p.name:
                files.setdefault(p.stem, p)
    return files


def build(a, out):
    zb = np.load(a.bone_matrix, allow_pickle=True, mmap_mode="r")
    ids, S = [str(i) for i in zb["ids"]], zb["split"].astype(str)
    assert set(S) <= {"train", "val"}
    N = len(ids)
    b17 = np.ascontiguousarray(zb["E"][:, :, :, 0]); Pb = np.array(zb["P"])
    zr = np.load(a.rib_matrix, allow_pickle=True, mmap_mode="r")
    assert [str(i) for i in zr["ids"]] == ids and [str(u) for u in zr["units"]] == RIBU
    r3 = np.ascontiguousarray(zr["E"][:, :, :, 0]); Pr3 = np.array(zr["P"])
    v15 = np.zeros((N, 10, 3, 448), np.float16); Pv = np.zeros((N, 10), bool)
    fv = _emb_files(a.v15_dirs)
    for i, cid in enumerate(ids):
        p = fv.get(cid)
        if p is not None:
            e = np.load(p)
            v15[i] = e["emb"]; Pv[i] = e["present"]
    r15 = np.zeros((N, 18, 3, 448), np.float16); r15x = np.zeros_like(r15)
    nv = np.zeros((N, 18), np.int64); has = np.zeros(N, bool)
    fr = _emb_files(a.r15_dirs)
    for i, cid in enumerate(ids):
        p = fr.get(cid)
        if p is None:
            continue
        e = np.load(p)
        assert list(e["units"]) == RIBU and list(e["windows"]) == ["baseline", "soft_tissue", "bone"]
        r15[i], r15x[i], nv[i] = e["emb"], e["emb_max"], e["nvox"]
        has[i] = True
    np.savez(out / "l_matrix.npz", ids=np.array(ids), split=S, b17=b17, Pb=Pb, b17_units=np.array(zb["units"]),
             r3=r3, Pr3=Pr3, v15=v15, Pv=Pv, r15=r15, r15x=r15x, nvox15=nv, has15=has)
    print(f"N={N} v15 found {int(Pv.any(1).sum())} r15 found {int(has.sum())}; r15 presence (>=200):",
          (nv >= 200).mean(0).round(3), flush=True)


# ----------------------------------------------------------------------------------------------- model
def parse_variant(v):
    c = dict(sites="Jadd", pool="mean", rl=0.0, lrc=False, vmin=200)
    for t in v.split("+"):
        if t in ("Jadd", "L_add", "L_repl", "G3"):
            c["sites"] = t
        elif t == "rmax":
            c["pool"] = "max"
        elif t == "rmm":
            c["pool"] = "mm"
        elif t.startswith("rl"):
            s = t[2:]; c["rl"] = float("0." + s[1:]) if s.startswith("0") else float(s)
        elif t == "lrc":
            c["lrc"] = True
        elif t.startswith("vmin"):
            c["vmin"] = int(t[4:])
        else:
            raise ValueError(t)
    if c["rl"] or c["lrc"]:
        assert c["sites"] != "Jadd"
    return c


def assemble(cfg, z, rows):
    """-> E [n,U,3,448] f16, P [n,U], sites, site group ('v'/'r'/'p'), names, per-rib site index or None."""
    b17, Pb = z["b17"][rows], z["Pb"][rows]
    parts, Ps, names = [b17[:, :10], b17[:, 12:17], z["v15"][rows]], [Pb[:, :10], Pb[:, 12:17], z["Pv"][rows]], \
        [f"v3_{l}" for l in LV] + ["sacrum", "hip_l", "hip_r", "femur_l", "femur_r"] + [f"v15_{l}" for l in LV]
    sites = [list(range(10)), list(range(10, 15)), list(range(15, 25))]
    grp = ["v", "p", "v"]
    if cfg["sites"] in ("Jadd", "L_add"):
        parts.append(b17[:, 10:12]); Ps.append(Pb[:, 10:12]); names += ["ribs_left", "ribs_right"]
        sites.append([25, 26]); grp.append("r")
    rib_site = None
    if cfg["sites"] in ("L_add", "L_repl"):
        if cfg["pool"] == "mean":
            R = z["r15"][rows]
        elif cfg["pool"] == "max":
            R = z["r15x"][rows]
        else:
            R = ((z["r15"][rows].astype(np.float32) + z["r15x"][rows].astype(np.float32)) / 2).astype(np.float16)
        parts.append(R); Ps.append(z["has15"][rows][:, None] & (z["nvox15"][rows] >= cfg["vmin"]))
        names += [f"r15_{u}" for u in RIBU]
    elif cfg["sites"] == "G3":
        parts.append(z["r3"][rows]); Ps.append(z["Pr3"][rows]); names += [f"r3_{u}" for u in RIBU]
    if cfg["sites"] != "Jadd":
        u0 = sum(p.shape[1] for p in parts[:-1])
        sites.append(list(range(u0, u0 + 18))); grp.append("r"); rib_site = len(sites) - 1
    E = np.concatenate(parts, 1); P = np.concatenate(Ps, 1)
    return E, P, sites, grp, names, rib_site


class KLate(nn.Module):
    """bone_expI.LateModel generalised to K sites (+ optional left-right contrast on the per-rib site)."""

    def __init__(self, W, D, U, sites, contra=None):
        super().__init__()
        K = len(sites); bd = 1 / D ** 0.5
        self.hw = nn.Parameter(torch.empty(K, W, D).uniform_(-bd, bd))
        self.hb = nn.Parameter(torch.empty(K, W).uniform_(-bd, bd))
        self.gate = nn.Parameter(torch.zeros(U, W))
        self.ub = nn.Parameter(torch.zeros(U)); self.none = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(1))
        self.sites = [torch.tensor(s) for s in sites]
        self.contra = contra
        if contra is not None:
            self.hc = nn.Parameter(torch.zeros(W, D))

    def forward(self, X, M):
        out, inst = [], []
        for k, idx in enumerate(self.sites):
            L = torch.einsum("nuwd,wd->nuw", X[:, idx], self.hw[k]) + self.hb[k]
            if self.contra is not None and k == self.contra[0]:
                ci = self.contra[1]
                mc = (M[:, idx] & M[:, ci]).float()[:, :, None, None]
                L = L + torch.einsum("nuwd,wd->nuw", (X[:, idx] - X[:, ci]) * mc, self.hc)
            s = (L * torch.softmax(self.gate[idx], -1)[None]).sum(-1) + self.ub[idx]
            inst.append(s)
            r, any_ = B.BoneModel.pool(s, M[:, idx], "max")
            out.append(torch.where(any_, r, self.none[k].expand_as(r)))
        St = torch.stack(out, 1)
        return torch.logsumexp(St, 1) + self.b, St, inst


def rib_targets(cfg, ids, S, y, names, sites, rib_site, P):
    """YL [n,18] instance targets for the per-rib site (NaN = unlabeled) and side-MIL targets YSd [n,2]."""
    n = len(ids)
    YL = np.full((n, 18), np.nan, np.float32); YSd = np.full((n, 2), np.nan, np.float32)
    if rib_site is None or cfg["rl"] <= 0:
        return None, None
    ucols = sites[rib_site]
    Pr = P[:, ucols]
    YL[y == 0] = 0.0
    rp = pd.read_csv(MW / "outputs_fracture/analysis_rib/rib_positives_parsed.csv").set_index("id")
    rp = rp[rp.split == "train"]                       # supervision from TRAIN reports only
    idx = {c: i for i, c in enumerate(ids)}
    SD = {"L": ["left"], "R": ["right"], "B": ["left", "right"]}
    for cid, r in rp.iterrows():
        i = idx.get(cid)
        if i is None or y[i] != 1 or S[i] != "train":
            continue
        sides = SD.get(r.side if isinstance(r.side, str) else "?", [])
        done = set()
        if isinstance(r.rib_levels, str) and r.rib_levels:
            for tok in r.rib_levels.split(";"):
                sd, lv = tok[0], int(tok[1:])
                for s_ in SD.get(sd, []):
                    j = RIBU.index(f"rib_{s_}_1_4" if lv <= 4 else f"rib_{s_}_{lv}")
                    if Pr[i, j]:
                        YL[i, j] = 1.0; done.add(s_)
        for s_ in sides:
            if s_ not in done:
                YSd[i, 0 if s_ == "left" else 1] = 1.0
    return YL, YSd


def make_X(E, P, fit):
    X = np.zeros(E.shape, np.float32)
    for u in range(E.shape[1]):
        for w in range(E.shape[2]):
            X[:, u, w] = B.std_unit(E[:, u, w], P[:, u], fit)
    return torch.from_numpy(X)


def train_l(cfg, args, X, M, y, ys, YL, YSd, rib_site, tr_idx, es_idx, lr, wd, seed, dev,
            max_epochs=100, patience=10, bs=512):
    torch.manual_seed(seed); np.random.seed(seed)
    model = KLate(*args).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt, Mt = X[tr_idx].to(dev), M[tr_idx].to(dev)
    yt = torch.from_numpy(y[tr_idx]).float().to(dev)
    yst = torch.from_numpy(ys[tr_idx]).float().to(dev); vs = ~torch.isnan(yst)

    def wts(v, valid):
        n1 = (v.nan_to_num(0) * valid).sum(0); n = valid.sum(0)
        return n / (2 * n1.clamp(min=1)), n / (2 * (n - n1).clamp(min=1))
    wp, wn = wts(yt, torch.ones_like(yt, dtype=torch.bool)); swp, swn = wts(yst, vs)
    use_inst = YL is not None
    if use_inst:
        rc = model.sites[rib_site]
        YLt = torch.from_numpy(YL[tr_idx]).to(dev); Mr = Mt[:, rc]
        lab = ~torch.isnan(YLt) & Mr
        v = YLt[lab]; n1, n0 = v.sum(), (1 - v).sum()
        cwp, cwn = (n1 + n0) / (2 * n1.clamp(min=1)), (n1 + n0) / (2 * n0.clamp(min=1))
        YSt = torch.from_numpy(YSd[tr_idx]).to(dev)
        side_cols = [torch.arange(0, 9), torch.arange(9, 18)]
    Xe, Me = X[es_idx].to(dev), M[es_idx].to(dev)
    best, state, bad, best_ep = -1.0, None, 0, 0
    n = len(Xt)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            la, ls, inst = model(Xt[j], Mt[j])
            loss = (F.binary_cross_entropy_with_logits(la, yt[j], reduction="none") * torch.where(yt[j] == 1, wp, wn)).mean()
            m = vs[j]
            w = torch.where(yst[j] == 1, swp, swn) * m
            loss = loss + (F.binary_cross_entropy_with_logits(ls, yst[j].nan_to_num(0), reduction="none") * w).sum() / m.sum().clamp(min=1)
            if use_inst:
                li = inst[rib_site]; lb = lab[j]
                denom = lb.sum().clamp(min=1)
                if lb.any():
                    vv = YLt[j].nan_to_num(0)
                    iw = torch.where(vv == 1, cwp, cwn) * lb
                    loss = loss + cfg["rl"] * (F.binary_cross_entropy_with_logits(li, vv, reduction="none") * iw).sum() / denom
                ysj = YSt[j]; mrj = Mr[j]
                for k in range(2):
                    ok = ~torch.isnan(ysj[:, k])
                    if ok.any():
                        mm = mrj[ok][:, side_cols[k]]
                        r = li[ok][:, side_cols[k]].masked_fill(~mm, -1e4).max(1).values
                        okm = mm.any(1)
                        if okm.any():
                            loss = loss + cfg["rl"] * F.binary_cross_entropy_with_logits(
                                r[okm], torch.ones_like(r[okm]), reduction="sum") * cwp / denom
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = auroc(y[es_idx], torch.cat([model(Xe[i:i + 2048], Me[i:i + 2048])[0]
                                            for i in range(0, len(Xe), 2048)]).cpu().numpy())
        if s > best + 1e-5:
            best, bad, best_ep = s, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model, best, best_ep


def predict(model, X, M, idx, dev, rib_site):
    a, s, li = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), 2048):
            j = idx[i:i + 2048]
            r = model(X[j].to(dev), M[j].to(dev))
            a.append(r[0].cpu()); s.append(r[1].cpu())
            if rib_site is not None:
                li.append(r[2][rib_site].cpu())
    return torch.cat(a).numpy(), torch.cat(s).numpy(), (torch.cat(li).numpy() if li else None)


def run(a, variant, so, dev):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    cfg = parse_variant(variant)
    out = Path(a.out)
    z = np.load(out / "l_matrix.npz", allow_pickle=True)
    ids_all, S_all = [str(i) for i in z["ids"]], z["split"].astype(str)
    assert set(S_all) <= {"train", "val"}
    keep = np.ones(len(ids_all), bool)
    if a.subset_file:
        sub = set(pd.read_csv(a.subset_file).query("tier == 0").study_id)
        keep = np.array([i in sub for i in ids_all]) & z["has15"]
    elif a.require15:
        keep = z["has15"].copy()
    lab = B.load_labels(B.LABELS, ids_all).loc[ids_all]
    T = B.targets(lab)
    keep &= ~np.isnan(T["any"])
    rows = np.where(keep)[0]
    ids, S = [ids_all[i] for i in rows], S_all[rows]
    y = T["any"][rows]
    E, P, sites, grp, names, rib_site = assemble(cfg, z, rows)
    tmap = {"v": T["vertebra"], "r": T["rib"], "p": T["pelvic"]}
    ys = np.stack([tmap[g][rows] for g in grp], 1)
    contra = None
    if cfg["lrc"]:
        ci = [sites[rib_site][RIBU.index(u.replace("left", "T").replace("right", "left").replace("T", "right"))]
              for u in RIBU]
        contra = (rib_site, torch.tensor(ci))
    YL, YSd = rib_targets(cfg, ids, S, y, names, sites, rib_site, P)
    M = torch.from_numpy(P)
    hp = {"lr": a.lr, "wd": a.wd}
    tr_all, va_all = np.where(S == "train")[0], np.where(S == "val")[0]
    N = len(ids)
    o = np.full(N, np.nan, np.float32); osite = np.full((N, len(sites)), np.nan, np.float32)
    oinst = np.full((N, 18), np.nan, np.float32)
    vl, vs_, vi, info = [], [], [], []
    print(f"{variant} s{so}: cfg={cfg} U={E.shape[1]} sites={[len(s) for s in sites]} train {len(tr_all)} val {len(va_all)}"
          + (f" inst pos={int(np.nansum(YL[tr_all] == 1))} side-MIL={int(np.nansum(YSd[tr_all] == 1))}" if YL is not None else ""),
          flush=True)
    for k, (fp, hpos) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr_all, y[tr_all])):
        t0 = time.time()
        fit, hold = tr_all[fp], tr_all[hpos]
        X = make_X(E, P, np.isin(np.arange(N), fit))
        inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
        args = (X.shape[2], X.shape[3], X.shape[1], sites, contra)
        m, s, ep = train_l(cfg, args, X, M, y, ys, YL, YSd, rib_site, inner, es, hp["lr"], hp["wd"], 42 + k + 100 * so, dev)
        o[hold], osite[hold], ih = predict(m, X, M, hold, dev, rib_site)
        av, sv, iv = predict(m, X, M, va_all, dev, rib_site)
        vl.append(av); vs_.append(sv)
        if rib_site is not None:
            oinst[hold] = ih; vi.append(iv)
        info.append({"fold": k, "es": s, "ep": ep, "held": auroc(y[hold], o[hold])})
        print(" ", info[-1], f"{time.time() - t0:.0f}s", flush=True)
        del X
    o[va_all], osite[va_all] = np.mean(vl, 0), np.mean(vs_, 0)
    if rib_site is not None:
        oinst[va_all] = np.mean(vi, 0)
        oinst[~P[:, sites[rib_site]]] = np.nan
    cv = auroc(y[tr_all], o[tr_all])
    print(f"OOF {variant} s{so}: train OOF={cv:.4f}", flush=True)       # val AUROC deliberately not printed
    sub = "tier0" if a.subset_file else ("has15" if a.require15 else "full")
    od = out / "oof" / sub; od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"oof_L_{variant.replace('+', '-')}_s{so}.npz", ids=np.array(ids), split=S, logit=o, y=y,
             site_logit=osite, site_grp=np.array(grp), inst=oinst,
             inst_units=np.array([names[u] for u in sites[rib_site]] if rib_site is not None else []),
             exp=f"L:{variant}", hp=json.dumps(hp), folds=json.dumps(info), oof_train_auc=cv, cfg=json.dumps(cfg))
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "oof"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bone-matrix", default=str(MW / "outputs_fracture/bones/bone_matrix.npz"))
    ap.add_argument("--rib-matrix", default=str(MW / "outputs_fracture/bones/rib_matrix.npz"))
    ap.add_argument("--v15-dirs", nargs="*", default=[str(MW / "outputs_vert15/merged_final/emb15")])
    ap.add_argument("--r15-dirs", nargs="*", default=[str(MW / "outputs_ribs15/emb15")])
    ap.add_argument("--variants", nargs="*", default=["Jadd"])
    ap.add_argument("--subset-file", default=None)
    ap.add_argument("--require15", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    if a.stage == "build":
        build(a, out)
        return
    for v in a.variants:
        for so in a.seeds:
            sub = "tier0" if a.subset_file else ("has15" if a.require15 else "full")
            f = out / "oof" / sub / f"oof_L_{v.replace('+', '-')}_s{so}.npz"
            if f.exists():
                continue
            run(a, v, so, torch.device(a.device))


if __name__ == "__main__":
    main()
