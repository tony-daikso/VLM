"""Osteopenia x Exp J 1.5 mm per-vertebra embeddings (L1-L4). Train/val only; test never read.

Inputs (one dir, train/val-only copies): ov_emb15_L1L4_trval.npz (ov_build.py), labels_trval.csv, bundle_trval.csv
(Age/Gender for evaluation only, L*_hu_med = 3 mm vertebral HU), oof_train.csv + scores_val_descriptive.csv
(analysis_osteopenia 3 mm lumbar_spine window/E3_late scores), osteopenia_oof.csv + osteopenia_val.csv (integrated_best
formal model), geom15_levels.csv (1.5 mm mid-sagittal trabecular-core HU, hu_body), ib_core.py.

Selection uses train 5-fold OOF only (folds = oof_train.csv 'fold' = analysis_osteopenia folds). Val is reporting only.
Embedding models:
  LR on mean over L1-L4 (3 windows x 448; baseline-only; mean+max pooling; per-level concat), C grid by OOF
  MIL (torch): per-level linear logit -> mean / max across levels; gated attention pooling. class-balanced BCE,
  Adam lr 1e-3, wd grid, ES on an inner 15 % train holdout (like op_oof.py), 2 seeds.
Stacks (ib_core.select_and_fit: LR C .1/.3/1 (+ 1 GBM for HU sets), same folds): v15 (+E3_late 3 mm) (+HU) /
  integrated_best stack + v15 (+HU) / 5-window logits + v15 (+HU).
Age/sex are never model inputs.
usage: python ov_run.py DATA_DIR OUT_DIR"""
import sys, os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)
D, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, D)
from ib_core import (auc, age_matched_auc, spearman_neg, boot_ci, boot_delta, sens_at_spec, LR, CANDS,
                     select_and_fit, EDGES)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

LOG = open(OUT + "run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# ---------------------------------------------------------------- labels: split filter first
lab = pd.read_csv(D + "labels_trval.csv", usecols=["study_id", "split", "y_op"])
lab = lab[lab.split.isin(["train", "val"])].dropna(subset=["y_op"]).set_index("study_id")
fold = pd.read_csv(D + "oof_train.csv", usecols=["study_id", "fold"]).set_index("study_id").fold
meta = pd.read_csv(D + "bundle_trval.csv").set_index("study_id")
tr_ids = [i for i in lab.index[lab.split == "train"]]
va_ids = [i for i in lab.index[lab.split == "val"]]
assert all(i in fold.index for i in tr_ids)
yT = lab.loc[tr_ids, "y_op"].values.astype(int); yV = lab.loc[va_ids, "y_op"].values.astype(int)
fT = fold.loc[tr_ids].values
ageT = meta.Age.reindex(tr_ids).values.astype(float); ageV = meta.Age.reindex(va_ids).values.astype(float)
sexT = meta.Gender.reindex(tr_ids).values; sexV = meta.Gender.reindex(va_ids).values
log(f"train {len(tr_ids)} (pos {yT.sum()}), val {len(va_ids)} (pos {yV.sum()})")

# ---------------------------------------------------------------- embeddings
z = np.load(D + "ov_emb15_L1L4_trval.npz")
eid = {s: i for i, s in enumerate(z["ids"])}
def rows(ids): return np.array([eid.get(i, -1) for i in ids])
rT, rV = rows(tr_ids), rows(va_ids)
log(f"emb15 missing: train {(rT < 0).sum()}, val {(rV < 0).sum()}")
E = z["E"].astype(np.float32).reshape(len(eid), 4, -1)          # [N,4,1344] windows baseline|soft|bone
Em = z["Emax"].astype(np.float32).reshape(len(eid), 4, -1)
P = z["P"].copy(); del z
def take(A, r):
    out = np.zeros((len(r),) + A.shape[1:], A.dtype); ok = r >= 0; out[ok] = A[r[ok]]; return out
ET, EV, EmT, EmV = take(E, rT), take(E, rV), take(Em, rT), take(Em, rV)
PT, PV = take(P, rT), take(P, rV); del E, Em
# standardise per level-shared dim on train present level rows
def stdz(AT, AV, PT):
    rr = AT[PT]; mu, sd = rr.mean(0), rr.std(0) + 1e-6
    return ((AT - mu) / sd) * PT[..., None], ((AV - mu) / sd) * PV[..., None]
ET, EV = stdz(ET, EV, PT); EmT, EmV = stdz(EmT, EmV, PT)
def lmean(A, Pm): return A.sum(1) / np.maximum(Pm.sum(1, keepdims=True), 1)
MT, MV = lmean(ET, PT), lmean(EV, PV); MmT, MmV = lmean(EmT, PT), lmean(EmV, PV)
okT = PT.any(1)
log(f"train rows with >=1 of L1-L4 present: {okT.mean():.4f}; val {PV.any(1).mean():.4f}")

def cv_lr(XT, XV, Cs, tag):
    best = None
    for C in Cs:
        oof = np.full(len(XT), np.nan)
        for k in range(5):
            tr, te = (fT != k) & okT, fT == k
            m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(XT[tr], yT[tr])
            oof[te] = m.decision_function(XT[te])
        oof[~okT] = np.nan
        a = auc(yT, oof); log(f"  {tag} C={C}: OOF {a:.4f}")
        if best is None or a > best[0]: best = (a, C, oof)
    a, C, oof = best
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(XT[okT], yT[okT])
    return oof, m.decision_function(XV), dict(C=C, oof=a)

# ---------------------------------------------------------------- MIL
class MIL(nn.Module):
    def __init__(self, d, mode, h=64):
        super().__init__(); self.mode = mode; self.drop = nn.Dropout(0.1)
        self.lin = nn.Linear(d, 1); self.lvl = nn.Parameter(torch.zeros(4))
        if mode == "attn": self.V = nn.Linear(d, h); self.U = nn.Linear(d, h); self.w = nn.Linear(h, 1)
    def forward(self, X, Pm):
        X = self.drop(X)
        if self.mode == "attn":
            g = self.w(torch.tanh(self.V(X)) * torch.sigmoid(self.U(X)))[..., 0]
            g = g.masked_fill(~Pm, -1e9); a = torch.softmax(g, 1)
            return self.lin((a[..., None] * X).sum(1))[:, 0]
        s = self.lin(X)[..., 0] + self.lvl
        if self.mode == "mean": return (s * Pm).sum(1) / Pm.sum(1).clamp(min=1)
        return s.masked_fill(~Pm, -1e9).max(1).values                 # max

def fit_mil(X, Pm, y, i_tr, i_es, Xp, Pp, mode, wd, seed, lr=1e-3, max_ep=100, pat=10, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MIL(X.shape[2], mode); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Xt, Pt, Yt = X[i_tr], Pm[i_tr], torch.from_numpy(y[i_tr].astype(np.float32))
    npos = Yt.sum(); wpos = len(Yt) / (2 * npos); wneg = len(Yt) / (2 * (len(Yt) - npos))
    best, bst, bad = -1, None, 0
    for ep in range(max_ep):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]; yb = Yt[j]
            loss = (F.binary_cross_entropy_with_logits(m(Xt[j], Pt[j]), yb, reduction="none")
                    * torch.where(yb == 1, wpos, wneg)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): sc = auc(y[i_es], m(X[i_es], Pm[i_es]).numpy())
        if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad(): return m(Xp, Pp).numpy()

def cv_mil(mode, wds, seeds=(0, 1)):
    X = torch.from_numpy(ET); Pm = torch.from_numpy(PT); XV_, PV_ = torch.from_numpy(EV), torch.from_numpy(PV)
    PmS = Pm.clone(); PmS[~torch.from_numpy(okT), 0] = True       # avoid empty bags (scores set NaN after)
    PVS = PV_.clone(); PVS[~PV_.any(1), 0] = True
    best = None; rng = np.random.default_rng(0)
    for wd in wds:
        oof = np.full(len(yT), np.nan)
        for k in range(5):
            a = np.where((fT != k) & okT)[0]; b = np.where(fT == k)[0]
            a = rng.permutation(a); nes = int(0.15 * len(a))
            oof[b] = np.mean([fit_mil(X, PmS, yT, a[nes:], a[:nes], X[b], PmS[b], mode, wd, s) for s in seeds], 0)
        oof[~okT] = np.nan
        sc = auc(yT, oof); log(f"  MIL-{mode} wd={wd}: OOF {sc:.4f}")
        if best is None or sc > best[0]: best = (sc, wd, oof)
    sc, wd, oof = best
    a = rng.permutation(np.where(okT)[0]); nes = int(0.15 * len(a))
    v = np.mean([fit_mil(X, PmS, yT, a[nes:], a[:nes], XV_, PVS, mode, wd, s) for s in seeds], 0)
    return oof, v, dict(wd=wd, oof=sc)

# ---------------------------------------------------------------- base embedding variants
base = {}; info = {}
Cs = [0.0001, 0.0003, 0.001, 0.003, 0.01]
t0 = time.time()
log("== LR variants")
base["v15_LR_mean"] = cv_lr(MT, MV, Cs, "LR mean L1-L4 (3 win)")
base["v15_LR_mean_base"] = cv_lr(MT[:, :448], MV[:, :448], Cs, "LR mean L1-L4 (baseline win)")
base["v15_LR_meanmax"] = cv_lr(np.c_[MT, MmT], np.c_[MV, MmV], Cs, "LR mean+max-pool L1-L4")
base["v15_LR_levelcat"] = cv_lr(ET.reshape(len(ET), -1), EV.reshape(len(EV), -1), [0.00001, 0.00003, 0.0001, 0.0003, 0.001], "LR per-level concat")
log(f"LR done {time.time()-t0:.0f}s")
log("== MIL variants")
for mode in ["mean", "max", "attn"]:
    base[f"v15_MIL_{mode}"] = cv_mil(mode, [1e-4, 1e-2])
    log(f"MIL {mode} done {time.time()-t0:.0f}s")
for k, (o, v, i) in base.items(): info[k] = i
pd.DataFrame({"study_id": tr_ids, "fold": fT, "y": yT, **{k: v[0] for k, v in base.items()}}).to_csv(OUT + "v15_oof_train.csv", index=False)
pd.DataFrame({"study_id": va_ids, "y": yV, **{k: v[1] for k, v in base.items()}}).to_csv(OUT + "v15_val.csv", index=False)
best_v15 = max(base, key=lambda k: info[k]["oof"])
log(f"best v15 by OOF: {best_v15} {info[best_v15]}")

# ---------------------------------------------------------------- other inputs
o3 = pd.read_csv(D + "oof_train.csv").set_index("study_id"); v3 = pd.read_csv(D + "scores_val_descriptive.csv").set_index("study_id")
WIN = ["E3_late", "E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]
ibo = pd.read_csv(D + "osteopenia_oof.csv").set_index("study_id"); ibv = pd.read_csv(D + "osteopenia_val.csv").set_index("study_id")
g = pd.read_csv(D + "geom15_levels.csv", usecols=["id", "level", "hu_body"])
g = g[g.level.isin(["L1", "L2", "L3", "L4"])].pivot(index="id", columns="level", values="hu_body")
g.columns = [f"hu15_{c}" for c in g.columns]
g["hu15_mean"] = g.mean(1); g["hu15_min"] = g.min(1); g["hu15_L1L2"] = g[["hu15_L1", "hu15_L2"]].mean(1)
HU15 = list(g.columns)
HU3 = [f"{l}_hu_med" for l in ["L1", "L2", "L3", "L4"]]
def frame(ids, sc3, ib, vb, ibcol):
    Fm = pd.DataFrame(index=ids)
    for c in WIN: Fm[c] = sc3[c].reindex(ids).values
    Fm["ib_stack"] = ib[ibcol].reindex(ids).values
    for k, v in vb.items(): Fm[k] = v
    Fm = Fm.join(g.reindex(ids)); Fm = Fm.join(meta[HU3].reindex(ids))
    return Fm
T = frame(tr_ids, o3, ibo, {k: v[0] for k, v in base.items()}, "stack_oof")
V = frame(va_ids, v3, ibv, {k: v[1] for k, v in base.items()}, "stack")
log(f"hu15 coverage train {T.hu15_mean.notna().mean():.4f} val {V.hu15_mean.notna().mean():.4f}; ib val n/a {V.ib_stack.isna().sum()}")
HU = HU15 + HU3
B = best_v15
LRc = {k: CANDS[k] for k in ["LR_C0.1", "LR_C0.3", "LR_C1"]}
LRG = {**LRc, "GBM_lr.03_l7_m40_500": CANDS["GBM_lr.03_l7_m40_500"]}
stacks = {
    "HU_only(1.5mm+3mm)": ({"hu": HU}, LRG),
    "HU15_only": ({"hu15": HU15}, LRG),
    "v15+E3_late": ({"x": [B, "E3_late"]}, LRc),
    "v15+HU": ({"x": [B] + HU}, LRG),
    "v15+E3_late+HU": ({"x": [B, "E3_late"] + HU}, LRG),
    "v15(all variants)+E3_late": ({"x": list(base) + ["E3_late"]}, LRc),
    "ib_stack+v15": ({"x": ["ib_stack", B]}, LRc),
    "ib_stack+v15+HU": ({"x": ["ib_stack", B] + HU}, LRG),
    "5win+v15": ({"x": WIN + [B]}, LRc),
    "5win+v15+HU": ({"x": WIN + [B] + HU}, LRG),
}
res = {}
for name, (fs, cands) in stacks.items():
    log(f"== stack {name}")
    r = select_and_fit(T, V, yT, fT, fs, cands=cands, log=log)
    res[name] = r
pd.DataFrame({"study_id": tr_ids, "fold": fT, "y": yT, **{k: r["oof"] for k, r in res.items()}}).to_csv(OUT + "stack_oof_train.csv", index=False)
pd.DataFrame({"study_id": va_ids, "y": yV, **{k: r["val"] for k, r in res.items()}}).to_csv(OUT + "stack_val.csv", index=False)

# ---------------------------------------------------------------- evaluation
def strat_auc(y, s, age, sex):
    b = pd.cut(pd.Series(age), EDGES, right=False).astype(str).values
    key = np.array([f"{x}|{y}" for x, y in zip(b, pd.Series(sex).fillna("NA").astype(str))]); num = den = 0.0; per = {}
    for k in np.unique(key):
        m = key == k; n1 = y[m].sum(); n0 = m.sum() - n1
        if n1 and n0:
            a = auc(y[m], s[m]); num += a * n1 * n0; den += n1 * n0; per[k] = (a, int(n1), int(n0))
    return num / den, per

refV = V.ib_stack.values; refT = T.ib_stack.values
rows, strata_rows = [], []
models = {"integrated_best（5 window logit LR，正式最佳）": (refT, refV), "E3_late 3 mm（re-run）": (T.E3_late.values, V.E3_late.values)}
models.update({k: (v[0], v[1]) for k, v in base.items()})
models.update({k: (r["oof"], r["val"]) for k, r in res.items()})
for name, (o, v) in models.items():
    o = np.asarray(o, float); v = np.asarray(v, float)
    vm = ~np.isnan(v)
    r = dict(model=name, oof=auc(yT, o), oof_age=age_matched_auc(yT, o, ageT),
             val=auc(yV, v), val_ci=boot_ci(yV[vm], v[vm]))
    d = boot_delta(yV, refV, v); r.update(d_val=d["mean"], d_val_ci=d["ci"], d_val_p=d["p_le0"])
    r["val_age"] = age_matched_auc(yV, v, ageV)
    da = boot_delta(yV, refV, v, n=500, age=ageV); r.update(d_val_age=da["mean"], d_val_age_ci=da["ci"])
    r["val_agesex"], per = strat_auc(yV, v, ageV, sexV)
    r["oof_agesex"], _ = strat_auc(yT, o, ageT, sexT)
    r["spearman_neg_val"] = spearman_neg(yV, v, ageV); r["spearman_neg_oof"] = spearman_neg(yT, o, ageT)
    r["sens90_val"] = sens_at_spec(yV, v)[0]
    rows.append(r)
    for k, (a, n1, n0) in per.items(): strata_rows.append(dict(model=name, stratum=k, auc=a, n_pos=n1, n_neg=n0))
    log(f"{name:<45s} OOF {r['oof']:.4f} | val {r['val']:.4f} [{r['val_ci'][0]:.3f},{r['val_ci'][1]:.3f}] "
        f"d {r['d_val']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] | age {r['val_age']:.4f} agesex {r['val_agesex']:.4f} "
        f"rho {r['spearman_neg_val']:.3f}")
R = pd.DataFrame(rows); R.to_csv(OUT + "results.csv", index=False)
pd.DataFrame(strata_rows).to_csv(OUT + "strata_val.csv", index=False)
json.dump(dict(base_info=info, best_v15=best_v15,
               stacks={k: dict(featset=r["featset"], cand=r["cand"], cols=r["cols"], oof=r["oof_auc"], cv=r["cv"]) for k, r in res.items()}),
          open(OUT + "selection.json", "w"), indent=1, ensure_ascii=False)
log("ALL_DONE")
