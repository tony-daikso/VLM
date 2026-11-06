"""Experiment I: level-supervised MIL + hard negatives + age matching + FOV-aware positives on the frozen bone-unit
embeddings, E3_late (3-window late fusion, site-wise max MIL) recipe of bone_probe.py --stage modes_oof.

Everything is 5-fold OOF on TRAIN with the folds / inner early-stop split / HP / seeds of modes_oof (val never used to
fit, stop or select; val logit = mean of the 5 fold models). The early-stop metric is the unchanged hard-label AUROC.
Inputs are train/val only: bone_matrix.npz / rib_matrix.npz (test never extracted into them), expI/meta.csv
(bone_expI_meta.py), analysis_rib/rib_positives_parsed.csv (train rows only are used for supervision).

variant spec = '+'-joined tokens (see TOK below), e.g.  ref | lv1 | lv1w | hn_b2 | age_full | fov03 | r18+rl1 | lv1+hn_n2
  units  : b17 (default: 17 bone units, 2 big rib units) | r18 (big rib units replaced by the 18 single-rib units)
  lvN    : vertebral-level instance loss, weight N (lv03 = 0.3): reported T8-L5 level(s) of train vertebral positives
           = 1; other levels of that case unlabeled; every present level of a negative case = 0
  lvNw   : as lvN, other (non-adjacent) levels of the positive case = weak negative (weight 0.2)
  rlN    : rib instance loss (weight N): r18 -> reported rib unit(s) (level+side) = 1, side-only cases: max over that
           side's rib units = 1; b17 -> reported side's big rib unit = 1; negatives' rib units = 0
  lrc    : rib left-right contrast term (r18): unit logit += <x_u - x_contralateral, h_c> per window (zero init)
  hn_bK / hn_nK / hn_mK : hard-negative weight K for train negatives with (b) age>=75 | osteoporosis | degenerative |
           scoliosis/kyphosis | hardware ; (n) same without plain degenerative ; (m) metal / implant / arthroplasty
  age_full / age_sqrt : reweight negatives by p_pos(age bin)/p_neg(age bin) (5-y bins, fit rows) (sqrt = tempered)
  pairN  : age-matched pairwise ranking loss (|age diff|<=2 y, positives vs negatives in the batch), weight N
  fov03 / fovsoft : positives whose reported site is outside the FOV / masks -> weight 0.3 / target 0.5
Sample weights are renormalised to mean 1 within each class on the fit rows (class balance unchanged).
usage: python bone_expI.py --variant lv1+hn_b2 --seed-off 0 [--threads 4]
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
from bone_probe import BoneModel, load, site_targets, std_unit  # noqa: E402
from probe import auroc  # noqa: E402

MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
B = MW / "outputs_fracture/bones"
OUT = B / "expI"
COV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]


def parse_variant(v):
    c = dict(units="b17", lv=0.0, lvw=False, rl=0.0, lrc=False, hn=None, hnw=1.0, age=None, pair=0.0, fov=None)
    for t in v.split("+"):
        num = lambda s: float("0." + s[1:]) if s.startswith("0") else float(s)
        if t in ("ref", ""):
            pass
        elif t in ("b17", "r18"):
            c["units"] = t
        elif t.startswith("lv"):
            c["lvw"] = t.endswith("w"); c["lv"] = num(t[2:].rstrip("w"))
        elif t.startswith("rl"):
            c["rl"] = num(t[2:])
        elif t == "lrc":
            c["lrc"] = True
        elif t.startswith("hn_"):
            c["hn"] = t[3]; c["hnw"] = float(t[4:])
        elif t in ("age_full", "age_sqrt"):
            c["age"] = t[4:]
        elif t.startswith("pair"):
            c["pair"] = num(t[4:])
        elif t in ("fov03", "fovsoft"):
            c["fov"] = t[3:]
        else:
            raise ValueError(t)
    if c["lrc"] or (c["rl"] and c["units"] == "r18"):
        assert c["units"] == "r18", "lrc needs r18"
    return c


class LateModel(nn.Module):
    """bone_probe.ModeModel(mode='late') with the same parameter order / init, generic unit sets, optional rib
    left-right contrast; also returns the per-unit (instance) logits."""

    def __init__(self, W, D, U, sites, contra=None):
        super().__init__()
        bd = 1 / D ** 0.5
        self.hw = nn.Parameter(torch.empty(3, W, D).uniform_(-bd, bd))
        self.hb = nn.Parameter(torch.empty(3, W).uniform_(-bd, bd))
        self.gate = nn.Parameter(torch.zeros(U, W))
        self.ub = nn.Parameter(torch.zeros(U)); self.none = nn.Parameter(torch.zeros(3))
        self.b = nn.Parameter(torch.zeros(1))
        self.sites = [torch.tensor(s) for s in sites]
        self.contra = contra                                  # None or (rib site k, tensor contralateral unit idx)
        if contra is not None:
            self.hc = nn.Parameter(torch.zeros(W, D))

    def forward(self, X, M):
        sites, inst = [], []
        for k, idx in enumerate(self.sites):
            L = torch.einsum("nuwd,wd->nuw", X[:, idx], self.hw[k]) + self.hb[k]
            if self.contra is not None and k == self.contra[0]:
                ci = self.contra[1]
                mc = (M[:, idx] & M[:, ci]).float()[:, :, None, None]
                L = L + torch.einsum("nuwd,wd->nuw", (X[:, idx] - X[:, ci]) * mc, self.hc)
            s = (L * torch.softmax(self.gate[idx], -1)[None]).sum(-1) + self.ub[idx]
            inst.append(s)
            r, any_ = BoneModel.pool(s, M[:, idx], "max")
            sites.append(torch.where(any_, r, self.none[k].expand_as(r)))
        St = torch.stack(sites, 1)
        return torch.logsumexp(St, 1) + self.b, St, torch.cat(inst, 1)


def unit_setup(cfg, ids, Eb, Pb):
    """-> E [N,U,3,448] fp16 (mean pool), P [N,U], sites (lists), unit names, contra spec, col group (0 v / 1 r / 2 p)."""
    zb = np.load(B / "bone_matrix.npz", allow_pickle=True, mmap_mode="r")
    bunits = list(zb["units"])
    if cfg["units"] == "b17":
        E, P, names = Eb[:, :, :, 0], Pb, bunits
        sites = [list(range(10)), [10, 11], [12, 13, 14, 15, 16]]
    else:
        zr = np.load(B / "rib_matrix.npz", allow_pickle=True)
        assert [str(i) for i in zr["ids"]] == ids
        keep = list(range(10)) + [12, 13, 14, 15, 16]
        E = np.concatenate([Eb[:, keep][:, :, :, 0], zr["E"][:, :, :, 0]], 1)
        P = np.concatenate([Pb[:, keep], zr["P"]], 1)
        names = [bunits[i] for i in keep] + [str(u) for u in zr["units"]]
        sites = [list(range(10)), list(range(15, 33)), list(range(10, 15))]
        del zr
    contra = None
    if cfg["lrc"]:
        rib = sites[1]
        ci = [names.index(names[u].replace("left", "TMP").replace("right", "left").replace("TMP", "right")) for u in rib]
        contra = (1, torch.tensor(ci))
    order = [u for s in sites for u in s]
    grp = np.array([k for k, s in enumerate(sites) for _ in s])
    return E, P, sites, names, contra, order, grp


def make_X(E, P, fit_mask):
    X = np.zeros(E.shape, np.float32)
    for u in range(E.shape[1]):
        for w in range(3):
            X[:, u, w] = std_unit(E[:, u, w], P[:, u], fit_mask)
    return torch.from_numpy(X)


def instance_targets(cfg, ids, y, meta, names, order, P):
    """YL [N, U(order)] NaN = unlabeled, WL weights; side-MIL targets YSd [N,2] (r18 only). Train positives only get
    positive labels (val rows are never used in the loss)."""
    N = len(ids); U = len(order)
    YL = np.full((N, U), np.nan, np.float32); WL = np.ones((N, U), np.float32)
    YSd = np.full((N, 2), np.nan, np.float32)
    col = {names[u]: j for j, u in enumerate(order)}
    Po = P[:, order]
    tr = (meta.split.values == "train")
    neg = (y == 0)
    grp_cols = {}
    for j, u in enumerate(order):
        n = names[u]
        g = "v" if n in COV else "r" if "rib" in n else "p"
        grp_cols.setdefault(g, []).append(j)
    if cfg["lv"] > 0:
        vc = grp_cols["v"]
        YL[np.ix_(neg, vc)] = 0.0
        for i in np.where(tr & (y == 1))[0]:
            L = [l for l in str(meta.levels.iloc[i]).split(";") if l in COV]
            L = [l for l in L if Po[i, col[l]]]
            if not L:
                continue
            for l in L:
                YL[i, col[l]] = 1.0
            if cfg["lvw"]:
                adj = {COV[k] for l in L for k in (COV.index(l) - 1, COV.index(l), COV.index(l) + 1) if 0 <= k < 10}
                for l in COV:
                    if l not in adj and Po[i, col[l]]:
                        YL[i, col[l]] = 0.0; WL[i, col[l]] = 0.2
    if cfg["rl"] > 0:
        rc = grp_cols["r"]
        YL[np.ix_(neg, rc)] = 0.0
        rp = pd.read_csv(MW / "outputs_fracture/analysis_rib/rib_positives_parsed.csv").set_index("id")
        rp = rp[rp.split == "train"]                                      # supervision from train reports only
        idx = {c: i for i, c in enumerate(ids)}
        for cid, r in rp.iterrows():
            i = idx.get(cid)
            if i is None or y[i] != 1:
                continue
            side = r.side if isinstance(r.side, str) else "?"
            sides = {"L": ["left"], "R": ["right"], "B": ["left", "right"]}.get(side, [])
            levs = []
            if isinstance(r.rib_levels, str) and r.rib_levels:
                for tok in r.rib_levels.split(";"):
                    sd, lv = tok[0], int(tok[1:])
                    for s_ in ({"L": ["left"], "R": ["right"], "B": ["left", "right"]}.get(sd, [])):
                        levs.append((s_, lv))
            if cfg["units"] == "b17":
                for s_ in set([s for s, _ in levs] + sides):
                    j = col[f"ribs_{s_}"]
                    if Po[i, j]:
                        YL[i, j] = 1.0
            else:
                done_sides = set()
                for s_, lv in levs:
                    n = f"rib_{s_}_1_4" if lv <= 4 else f"rib_{s_}_{lv}"
                    j = col[n]
                    if Po[i, j]:
                        YL[i, j] = 1.0; done_sides.add(s_)
                for s_ in sides:
                    if s_ not in done_sides:
                        YSd[i, 0 if s_ == "left" else 1] = 1.0
    side_cols = None
    if cfg["units"] == "r18":
        side_cols = [[j for j in grp_cols["r"] if "left" in names[order[j]]], [j for j in grp_cols["r"] if "right" in names[order[j]]]]
    gcol = np.array([{"v": 0, "r": 1, "p": 2}["v" if names[u] in COV else "r" if "rib" in names[u] else "p"] for u in order])
    return YL, WL, YSd, side_cols, gcol


def sample_weights(cfg, y, meta, fit):
    """per-row weight and soft target; renormalised within class over the fit rows."""
    N = len(y); w = np.ones(N, np.float32); t = y.astype(np.float32).copy()
    neg, pos = y == 0, y == 1
    m = meta
    if cfg["hn"]:
        if cfg["hn"] == "b":
            f = m.hn_age75 | m.hn_osteo | m.hn_degen | m.hn_scolio_kyph | m.hn_hardware
        elif cfg["hn"] == "n":
            f = m.hn_age75 | m.hn_osteo | m.hn_scolio_kyph | m.hn_hardware
        elif cfg["hn"] == "m":
            f = m.hn_metal
        w[neg & f.values] *= cfg["hnw"]
    if cfg["age"]:
        a = m.age.values
        bins = np.clip(np.nan_to_num(a, nan=-1) // 5, 0, 19).astype(int)
        ok = fit & ~np.isnan(a)
        pp = np.bincount(bins[ok & pos], minlength=20) + 1.0; pn = np.bincount(bins[ok & neg], minlength=20) + 1.0
        r = (pp / pp.sum()) / (pn / pn.sum())
        r = np.sqrt(r) if cfg["age"] == "sqrt" else r
        r = np.clip(r, 0.05, 20)
        w[neg & ~np.isnan(a)] *= r[bins[neg & ~np.isnan(a)]]
    if cfg["fov"]:
        f = m.fov_out.values & pos
        if cfg["fov"] == "03":
            w[f] *= 0.3
        else:
            t[f] = 0.5
    for c in (neg, pos):
        sel = c & fit
        if sel.any():
            w[c] /= w[sel].mean()
    return w, t


def train_i(cfg, model_args, X, M, y, t, w, ys, YL, WL, YSd, side_cols, gcol, age, tr_idx, es_idx, lr, wd, seed, dev,
            max_epochs=100, patience=10, bs=512):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LateModel(*model_args).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt, Mt = X[tr_idx].to(dev), M[tr_idx].to(dev)
    yt = torch.from_numpy(y[tr_idx]).float().to(dev)
    tt = torch.from_numpy(t[tr_idx]).float().to(dev)
    wt = torch.from_numpy(w[tr_idx]).float().to(dev)
    yst = torch.from_numpy(ys[tr_idx]).float().to(dev)
    vs = ~torch.isnan(yst)

    def wts(v, valid):
        n1 = (v.nan_to_num(0) * valid).sum(0); n = valid.sum(0)
        return n / (2 * n1.clamp(min=1)), n / (2 * (n - n1).clamp(min=1))
    wp, wn = wts(yt, torch.ones_like(yt, dtype=torch.bool))
    swp, swn = wts(yst, vs)
    order = torch.tensor([u for st in model_args[3] for u in st])
    use_inst = cfg["lv"] > 0 or cfg["rl"] > 0
    if use_inst:
        YLt = torch.from_numpy(YL[tr_idx]).to(dev); WLt = torch.from_numpy(WL[tr_idx]).to(dev)
        Mo = Mt[:, order]
        lab = ~torch.isnan(YLt) & Mo
        lam = torch.tensor([cfg["lv"], cfg["rl"], 0.0])[torch.from_numpy(gcol)]
        G = torch.from_numpy(gcol)
        # class balance per column group (weighted counts of labeled entries)
        gp, gn = torch.ones(3), torch.ones(3)
        for g in range(3):
            lg = lab[:, G == g]
            if lg.any():
                v = YLt[:, G == g][lg]; ww = WLt[:, G == g][lg]
                n1, n0 = (ww * v).sum(), (ww * (1 - v)).sum()
                gp[g], gn[g] = (n1 + n0) / (2 * n1.clamp(min=1)), (n1 + n0) / (2 * n0.clamp(min=1))
        cwp, cwn = gp[G], gn[G]
        if YSd is not None and side_cols is not None and cfg["rl"] > 0:
            YSt = torch.from_numpy(YSd[tr_idx]).to(dev)
            sc = [torch.tensor(c) for c in side_cols]
        else:
            YSt = None
    if cfg["pair"] > 0:
        at = torch.from_numpy(np.nan_to_num(age[tr_idx], nan=-1000)).float()
    Xe, Me = X[es_idx].to(dev), M[es_idx].to(dev)
    best, state, bad, best_ep = -1.0, None, 0, 0
    n = len(Xt)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            la, ls, li = model(Xt[j], Mt[j])
            cw = torch.where(yt[j] == 1, wp, wn) * wt[j]
            loss = (F.binary_cross_entropy_with_logits(la, tt[j], reduction="none") * cw).mean()
            m = vs[j]
            sw = torch.where(yst[j] == 1, swp, swn) * m * wt[j][:, None]
            loss = loss + (F.binary_cross_entropy_with_logits(ls, yst[j].nan_to_num(0), reduction="none") * sw).sum() / m.sum().clamp(min=1)
            if use_inst:
                lb = lab[j]
                if lb.any():
                    v = YLt[j].nan_to_num(0)
                    iw = torch.where(v == 1, cwp, cwn) * WLt[j] * lam * lb * wt[j][:, None]
                    loss = loss + (F.binary_cross_entropy_with_logits(li, v, reduction="none") * iw).sum() / lb.sum().clamp(min=1)
                if YSt is not None:
                    ysj = YSt[j]
                    for k in range(2):
                        ok = ~torch.isnan(ysj[:, k])
                        if ok.any():
                            mm = Mo[j][:, sc[k]][ok]
                            if mm.any():
                                r = li[ok][:, sc[k]].masked_fill(~mm, -1e4).max(1).values
                                okm = mm.any(1)
                                loss = loss + cfg["rl"] * F.binary_cross_entropy_with_logits(r[okm], torch.ones_like(r[okm]), reduction="sum") * cwp[G == 1][0] / lb.sum().clamp(min=1)
            if cfg["pair"] > 0:
                yj = yt[j]; aj = at[j]
                pi, ni = torch.where(yj == 1)[0], torch.where(yj == 0)[0]
                if len(pi) and len(ni):
                    D = ((aj[pi][:, None] - aj[ni][None, :]).abs() <= 2) & (aj[pi][:, None] > 0) & (aj[ni][None, :] > 0)
                    if D.any():
                        dl = la[pi][:, None] - la[ni][None, :]
                        loss = loss + cfg["pair"] * (F.softplus(-dl) * D).sum() / D.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = auroc(y[es_idx], torch.cat([model(Xe[i:i + 2048], Me[i:i + 2048])[0] for i in range(0, len(Xe), 2048)]).cpu().numpy())
        if s > best + 1e-5:
            best, bad, best_ep = s, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model, best, best_ep


def predict_i(model, X, M, idx, dev):
    a, s, li = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), 2048):
            j = idx[i:i + 2048]
            r = model(X[j].to(dev), M[j].to(dev))
            a.append(r[0].cpu()); s.append(r[1].cpu()); li.append(r[2].cpu())
    return torch.cat(a).numpy(), torch.cat(s).numpy(), torch.cat(li).numpy()


def run(variant, seed_off, dev, outdir, folds=5):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    cfg = parse_variant(variant)
    ids, S, Eb, Pb, units, lab, T = load(B)
    ids = [str(i) for i in ids]
    y, ys = T["any"], site_targets(T)
    meta = pd.read_csv(OUT / "meta.csv", index_col=0, keep_default_na=False, na_values=[""]).reindex(ids)
    assert (meta.split.values == S).all()
    for c in [c for c in meta.columns if c.startswith("hn_") or c in ("fov_out",)]:
        meta[c] = meta[c].astype(str).eq("True")
    meta["levels"] = meta["levels"].fillna("")
    E, P, sites, names, contra, order, grp = unit_setup(cfg, ids, Eb, Pb)
    del Eb
    M = torch.from_numpy(P)
    YL, WL, YSd, side_cols, gcol = instance_targets(cfg, ids, y, meta, names, order, P)
    hp = json.loads((B / "rerun/std/E3_late.json").read_text())["hp"]
    tr_all, va_all = np.where(S == "train")[0], np.where(S == "val")[0]
    N, U = len(ids), len(order)
    o = np.full(N, np.nan, np.float32); osite = np.full((N, 3), np.nan, np.float32); oinst = np.full((N, U), np.nan, np.float32)
    vl, vs_, vi, info = [], [], [], []
    print(f"{variant}: cfg={cfg} units={U} labeled inst pos={int(np.nansum(YL[tr_all] == 1))} side-MIL={int(np.nansum(YSd[tr_all] == 1))}", flush=True)
    for k, (fp, hpos) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=42).split(tr_all, y[tr_all])):
        t0 = time.time()
        fit, hold = tr_all[fp], tr_all[hpos]
        fm = np.isin(np.arange(N), fit)
        X = make_X(E, P, fm)
        inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
        w, t = sample_weights(cfg, y, meta, np.isin(np.arange(N), inner))
        args = (X.shape[2], X.shape[3], U, sites, contra)
        m, s, ep = train_i(cfg, args, X, M, y, t, w, ys, YL, WL, YSd, side_cols, gcol, meta.age.values.astype(float),
                           inner, es, hp["lr"], hp["wd"], 42 + k + 100 * seed_off, dev)
        o[hold], osite[hold], oinst[hold] = predict_i(m, X, M, hold, dev)
        a, st, li = predict_i(m, X, M, va_all, dev); vl.append(a); vs_.append(st); vi.append(li)
        info.append({"fold": k, "es": s, "ep": ep, "held": auroc(y[hold], o[hold])})
        print(info[-1], f"{time.time() - t0:.0f}s", flush=True)
        del X
    o[va_all], osite[va_all], oinst[va_all] = np.mean(vl, 0), np.mean(vs_, 0), np.mean(vi, 0)
    oinst[~P[:, order]] = np.nan
    cv = auroc(y[tr_all], o[tr_all])
    print(f"OOF {variant} s{seed_off}: train OOF={cv:.4f}", flush=True)       # val AUROC deliberately not printed
    od = Path(outdir); od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"oof_bone_E_{variant.replace('+', '-')}_s{seed_off}.npz", ids=np.array(ids), split=S, logit=o, y=y,
             site_logit=osite, inst=oinst, inst_units=np.array([names[u] for u in order]), exp=f"I:{variant}",
             hp=json.dumps(hp), folds=json.dumps(info), oof_train_auc=cv, cfg=json.dumps(cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed-off", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default=str(OUT / "members"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run(a.variant, a.seed_off, torch.device(a.device), a.outdir)


if __name__ == "__main__":
    main()
