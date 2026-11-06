"""Distillation C2: 1.5 mm (Exp J) -> 3 mm cross-resolution distillation. Train/val only; val never used to fit,
stop or select (val logit = mean of the 5 fold models); test never loaded.

Models are the E3_late site-wise MIL (bone_expI.LateModel recipe generalised to K sites; E3_late HP; per (unit,
window) z-score with fit-row stats; inner 10 % early stop on the HARD any-fracture AUROC; 5-fold StratifiedKFold rs=42).
  teacher  : 27 units (17 x 3 mm + T8..L5 x 1.5 mm), sites (vert3, ribs, pelvis, vert15), trained on tier0 TRAIN
             (Exp J 'add'); train OOF any-logit and vert15 site logit, Platt-calibrated on tier0 train.
  ref_t0   : 3 mm only (17 units), hard labels, tier0 train              <- plain 3 mm, same subset
  stu_t0_aX: 3 mm only, tier0 train, any target = a*y + (1-a)*q_any ; vertebra-site target = a*y_v + (1-a)*q_v
             (where y_v is unlabeled = fracture elsewhere: q_v)
  ref_full : 3 mm only, full train, hard                                 <- plain 3 mm (E3_late single seed)
  stu_full_aX : 3 mm only, full train; soft targets on tier0 train rows, hard elsewhere
  ad_full  : feature-level: ridge adapter 3 mm level emb (3x448) -> 1.5 mm level emb, fitted on TRAIN rows that
             have both (train rows get 5-fold cross-fitted predictions, val rows the all-train fit); the 27-unit
             'add' model then uses pseudo-1.5 mm units for EVERY row (train and val), full train, hard labels.
             -> needs only 3 mm inputs at inference.
usage: python distill_c2.py --exp teacher|ref_t0|stu_t0|ref_full|stu_full|ad_full [--alpha .5] --seed-off 0
"""
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
OUT = MW / "outputs_fracture/distill/c2"
S3 = [list(range(10)), [10, 11], [12, 13, 14, 15, 16]]
S4 = S3 + [list(range(17, 27))]


class GModel(nn.Module):
    def __init__(self, W, D, U, sites):
        super().__init__()
        K = len(sites); bd = 1 / D ** 0.5
        self.hw = nn.Parameter(torch.empty(K, W, D).uniform_(-bd, bd))
        self.hb = nn.Parameter(torch.empty(K, W).uniform_(-bd, bd))
        self.gate = nn.Parameter(torch.zeros(U, W))
        self.ub = nn.Parameter(torch.zeros(U)); self.none = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(1))
        self.sites = [torch.tensor(s) for s in sites]

    def forward(self, X, M):
        out = []
        for k, idx in enumerate(self.sites):
            L = torch.einsum("nuwd,wd->nuw", X[:, idx], self.hw[k]) + self.hb[k]
            s = (L * torch.softmax(self.gate[idx], -1)[None]).sum(-1) + self.ub[idx]
            r, any_ = BoneModel.pool(s, M[:, idx], "max")
            out.append(torch.where(any_, r, self.none[k].expand_as(r)))
        St = torch.stack(out, 1)
        return torch.logsumexp(St, 1) + self.b, St


def make_X(E, P, fit_mask):
    X = np.zeros(E.shape, np.float32)
    for u in range(E.shape[1]):
        for w in range(3):
            X[:, u, w] = std_unit(E[:, u, w], P[:, u], fit_mask)
    return torch.from_numpy(X)


def train(X, M, sites, y, t, ysh, yst, tr_idx, es_idx, lr, wd, seed, max_epochs=100, patience=10, bs=512):
    """y hard any (class weights + early stop), t any target (soft ok), ysh hard site labels (class weights, NaN =
    unlabeled), yst site targets (soft ok, NaN = no loss)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GModel(X.shape[2], X.shape[3], X.shape[1], sites)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt, Mt = X[tr_idx], M[tr_idx]
    yt = torch.from_numpy(y[tr_idx]).float(); tt = torch.from_numpy(t[tr_idx]).float()
    yh = torch.from_numpy(ysh[tr_idx]).float(); ys = torch.from_numpy(yst[tr_idx]).float()
    vs = ~torch.isnan(ys)
    n1 = yt.sum(); n = len(yt); wp, wn = n / (2 * n1), n / (2 * (n - n1))
    hv = ~torch.isnan(yh); h1 = (yh.nan_to_num(0) * hv).sum(0); hn = hv.sum(0)
    swp, swn = hn / (2 * h1.clamp(min=1)), hn / (2 * (hn - h1).clamp(min=1))
    swu = torch.ones_like(swp)                                            # rows with no hard site label (soft only)
    Xe, Me = X[es_idx], M[es_idx]
    best, state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            la, ls = model(Xt[j], Mt[j])
            loss = (F.binary_cross_entropy_with_logits(la, tt[j], reduction="none") * torch.where(yt[j] == 1, wp, wn)).mean()
            m = vs[j]
            w = torch.where(torch.isnan(yh[j]), swu, torch.where(yh[j] == 1, swp, swn)) * m
            loss = loss + (F.binary_cross_entropy_with_logits(ls, ys[j].nan_to_num(0), reduction="none") * w).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = auroc(y[es_idx], model(Xe, Me)[0].numpy())
        if s > best + 1e-5:
            best, bad, best_ep = s, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model, best, best_ep


def predict(model, X, M, idx):
    with torch.no_grad():
        a, s = zip(*[model(X[idx[i:i + 2048]], M[idx[i:i + 2048]]) for i in range(0, len(idx), 2048)])
    return torch.cat(a).numpy(), torch.cat(s).numpy()


def oof(E, P, sites, y, t, ysh, yst, rows, va, hp, seed_off, folds=5):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    N = len(y); K = len(sites)
    o = np.full(N, np.nan, np.float32); os_ = np.full((N, K), np.nan, np.float32)
    M = torch.from_numpy(P); vl, vs, info = [], [], []
    for k, (fp, hpos) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=42).split(rows, y[rows])):
        t0 = time.time()
        fit, hold = rows[fp], rows[hpos]
        X = make_X(E, P, np.isin(np.arange(N), fit))
        inner, es = train_test_split(fit, test_size=0.1, stratify=y[fit], random_state=42 + k)
        m, s, ep = train(X, M, sites, y, t, ysh, yst, inner, es, hp["lr"], hp["wd"], 42 + k + 100 * seed_off)
        o[hold], os_[hold] = predict(m, X, M, hold)
        a, st = predict(m, X, M, va); vl.append(a); vs.append(st)
        info.append({"fold": k, "es": s, "ep": ep, "held": auroc(y[hold], o[hold])})
        print(info[-1], f"{time.time() - t0:.0f}s", flush=True)
    o[va], os_[va] = np.mean(vl, 0), np.mean(vs, 0)
    return o, os_, info


def platt(z, yy):
    from sklearn.linear_model import LogisticRegression
    ok = ~np.isnan(z) & ~np.isnan(yy)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[ok, None], yy[ok].astype(int))
    return 1 / (1 + np.exp(-(lr.coef_[0, 0] * z + lr.intercept_[0])))


def adapter(E3, P3, E15, P15, S, n_folds=5):
    """ridge 3 mm level emb (3*448) -> 1.5 mm level emb (3*448), levels pooled; returns pseudo [N,10,3,448]."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    N = len(S); tr = np.where(S == "train")[0]
    both = P3[:, :10] & P15
    Xa = E3[:, :10].reshape(N, 10, -1).astype(np.float32); Ya = E15.reshape(N, 10, -1).astype(np.float32)
    mu_x, sd_x = Xa[tr][both[tr]].mean(0), Xa[tr][both[tr]].std(0) + 1e-6
    mu_y = Ya[tr][both[tr]].mean(0)
    lvl = np.eye(10, dtype=np.float32)

    def design(rows):
        r, l = np.nonzero(P3[rows, :10])
        return rows[r], l, np.c_[(Xa[rows[r], l] - mu_x) / sd_x, lvl[l]]

    def fit(rows):
        rr, ll, D = design(rows)
        k = both[rr, ll]
        return Ridge(alpha=ALPHA[0]).fit(D[k], Ya[rr[k], ll[k]] - mu_y)

    # alpha by 3-fold CV R^2 on train cases (labels unused)
    best = None
    for al in (10.0, 100.0, 1000.0, 3000.0):
        ALPHA[0] = al; sc = []
        for a_, b_ in KFold(3, shuffle=True, random_state=0).split(tr):
            m = fit(tr[a_]); rr, ll, D = design(tr[b_]); k = both[rr, ll]
            Yt = Ya[rr[k], ll[k]] - mu_y
            sc.append(1 - ((m.predict(D[k]) - Yt) ** 2).sum() / (Yt ** 2).sum())
        print(f"adapter ridge alpha={al}: CV R2 {np.mean(sc):.3f}", flush=True)
        if best is None or np.mean(sc) > best[1]:
            best = (al, np.mean(sc))
    ALPHA[0] = best[0]
    out = np.zeros((N, 10, 3 * 448), np.float32)
    for a_, b_ in KFold(n_folds, shuffle=True, random_state=1).split(tr):
        m = fit(tr[a_]); rr, ll, D = design(tr[b_]); out[rr, ll] = m.predict(D) + mu_y
    m = fit(tr); va = np.where(S != "train")[0]; rr, ll, D = design(va); out[rr, ll] = m.predict(D) + mu_y
    return out.reshape(N, 10, 3, 448), P3[:, :10].copy(), best


ALPHA = [100.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed-off", type=int, default=0); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--tsuf", default=None, help="teacher file suffix (e.g. _hpJ) independent of --hp")
    ap.add_argument("--hp", default="", help="lr,wd override (e.g. Exp J tier0 grid choice 0.01,0.01); adds _hpJ to the tag")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    ids, S, Eb, Pb, units, lab, T = load(B)
    ids = [str(i) for i in ids]; S = np.asarray(S).astype(str)
    E3 = np.ascontiguousarray(Eb[:, :, :, 0]); P3 = np.asarray(Pb); del Eb
    y = T["any"]; ysite = site_targets(T)                                  # [N,3] vert / rib / pelv (hard, NaN)
    v = np.load(MW / "outputs_vert15/v15_only.npz", allow_pickle=True)
    assert [str(i) for i in v["ids"]] == ids
    E15, P15 = v["E"], v["P"]
    t0ids = set(pd.read_csv(MW / "outputs_vert15/ids_prio.csv").query("tier == 0").study_id)
    tier0 = np.array([i in t0ids for i in ids]) & P15.any(1)
    lab_ok = ~np.isnan(y)
    tr_t0 = np.where((S == "train") & tier0 & lab_ok)[0]
    tr_full = np.where((S == "train") & lab_ok)[0]
    va = np.where((S == "val") & lab_ok)[0]
    hp = json.loads((B / "rerun/std/E3_late.json").read_text())["hp"]
    tag = a.exp if a.exp in ("teacher", "ref_t0", "ref_full", "ad_full") else f"{a.exp}_a{a.alpha:g}"
    if a.hp:
        hp = {"lr": float(a.hp.split(",")[0]), "wd": float(a.hp.split(",")[1])}; tag += "_hpJ"
    TSUF = "_hpJ" if a.hp else ""
    if a.tsuf is not None:
        TSUF = a.tsuf; tag += "_t" + (a.tsuf.strip("_") or "E3")
    f = OUT / f"oof_c2_{tag}_s{a.seed_off}.npz"
    if f.exists():
        print("exists", f); return
    t = y.astype(np.float32).copy()
    if a.exp == "teacher":
        E = np.concatenate([E3, E15], 1); P = np.concatenate([P3, P15], 1)
        ysh = np.c_[ysite, ysite[:, 0]]
        o, os_, info = oof(E, P, S4, y, t, ysh, ysh.copy(), tr_t0, va, hp, a.seed_off)
        rows = tr_t0
    elif a.exp in ("ref_t0", "ref_full"):
        rows = tr_t0 if a.exp == "ref_t0" else tr_full
        o, os_, info = oof(E3, P3, S3, y, t, ysite, ysite.copy(), rows, va, hp, a.seed_off)
    elif a.exp in ("stu_t0", "stu_full"):
        tz = [np.load(p, allow_pickle=True) for p in sorted(OUT.glob(f"oof_c2_teacher{TSUF}_s*.npz"))]
        assert len(tz) >= 1
        rx = lambda z, v: pd.Series(np.asarray(v, float), index=[str(i) for i in z["ids"]]).reindex(ids).values
        za = np.mean([rx(z, z["logit"]) for z in tz], 0); zv = np.mean([rx(z, z["site_logit"][:, 3]) for z in tz], 0)
        tm = np.zeros(len(y), bool); tm[tr_t0] = True
        za, zv = np.where(tm, za, np.nan), np.where(tm, zv, np.nan)           # teacher used on tier0 TRAIN rows only
        qa, qv = platt(za, np.where(tm, y, np.nan)), platt(zv, np.where(tm, ysite[:, 0], np.nan))
        t = np.where(tm, a.alpha * np.nan_to_num(y) + (1 - a.alpha) * np.nan_to_num(qa), y).astype(np.float32)
        yst = ysite.copy()
        hv = ysite[:, 0]
        yst[:, 0] = np.where(tm, np.where(np.isnan(hv), qv, a.alpha * np.nan_to_num(hv) + (1 - a.alpha) * np.nan_to_num(qv)), hv)
        print(f"teacher files {len(tz)}; q_any pos {np.nanmean(qa[tm & (y == 1)]):.3f} neg {np.nanmean(qa[tm & (y == 0)]):.3f}", flush=True)
        rows = tr_t0 if a.exp == "stu_t0" else tr_full
        o, os_, info = oof(E3, P3, S3, y, t, ysite, yst, rows, va, hp, a.seed_off)
    elif a.exp == "ad_full":
        cache = OUT / "adapter_pseudo15.npz"
        if cache.exists():
            z = np.load(cache); Ep, Pp, best = z["E"], z["P"], tuple(z["best"])
        else:
            Ep, Pp, best = adapter(E3, P3, E15, P15, S)
            np.savez(cache, E=Ep.astype(np.float16), P=Pp, best=np.array(best), ids=np.array(ids), split=S)
        print("adapter", best, flush=True)
        E = np.concatenate([E3, Ep.astype(np.float16)], 1); P = np.concatenate([P3, Pp], 1)
        ysh = np.c_[ysite, ysite[:, 0]]
        rows = tr_full
        o, os_, info = oof(E, P, S4, y, t, ysh, ysh.copy(), rows, va, hp, a.seed_off)
    else:
        raise ValueError(a.exp)
    keep = np.zeros(len(y), bool); keep[rows] = True; keep[va] = True
    cv = auroc(y[rows], o[rows])
    np.savez(f, ids=np.array(ids)[keep], split=S[keep], logit=o[keep], y=y[keep], site_logit=os_[keep],
             exp=f"c2:{tag}", hp=json.dumps(hp), folds=json.dumps(info), oof_train_auc=cv, n_train=len(rows))
    print(f"{tag} s{a.seed_off}: train OOF {cv:.4f} (train rows {len(rows)}, val rows {len(va)}) -> {f}", flush=True)


if __name__ == "__main__":
    main()
