"""Re-train the current-best osteopenia pipeline (osteopenia_v15 '5win+v15') under one label-handling scheme.
Components (same recipes / fixed hyper-parameters as the formal best, so only the label handling changes):
  * 6 window probes on the 3 mm lumbar_spine unit (E0, E3_late, E1 x4; probe.py Probe, lr 1e-3, wd 1e-4,
    class-balanced BCE, ES on an inner 15 % train holdout, seeds 42/43 for E0/E3_late, 42 otherwise)
  * 1.5 mm L1-L4 per-level concat LR (C = 1e-4, class balanced)
  * stack: StandardScaler + LR C = 0.1 (class balanced) on the 7 scores.
Folds = analysis_osteopenia folds. Train/val only. Test never read.
usage: python ol_scheme.py SCHEME OUT_DIR    SCHEME in A, B110, B130, C, D, Dmt"""
import sys, os, json, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OL_THREADS", "3")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ol_common import D, load_base, scheme_targets, balanced_weights, expand
sys.path.insert(0, D)
from probe_cls import Probe
from ib_core import auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCH, OUT = sys.argv[1], sys.argv[2].rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + f"{SCH}.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

B = load_base(); T, V = B["T"], B["V"]
tr_ids, va_ids = list(T.index), list(V.index); fT = T.fold.values; yT = T.y.values
nT = len(tr_ids)
info = dict(scheme=SCH)
if SCH == "C":
    a = pd.read_csv(OUT + "A_oof.csv").set_index("study_id").loc[tr_ids, "stack"].values
    # Platt calibration of scheme-A OOF stack score to P(s=1|x) (unweighted), Elkan-Noto c = mean g on labelled pos
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(a[:, None], yT)
    g = pl.predict_proba(a[:, None])[:, 1]; c = float(g[yT == 1].mean())
    tg, wg = scheme_targets(T, "C", pu_g=(g, c))
    info.update(c=c, prior_label=float(yT.mean()), prior_est=float(tg.mean()),
                n_unl_p_gt_0_5=int(((yT == 0) & (tg > 0.5)).sum()), sum_p_unl=float(tg[yT == 0].sum()))
    log("Elkan-Noto:", info)
    tT, wT = tg, wg
elif SCH == "Dmt":
    tT, wT = yT.astype(float), np.ones(nT)
else:
    tT, wT = scheme_targets(T, SCH)
info.update(n_weight0=int((wT == 0).sum()), sum_t=float((tT * wT).sum()))
# label used for early stopping / internal OOF diagnostics: report label, rows with w > 0
yES = yT.astype(float).copy(); yES[wT == 0] = np.nan
log(f"scheme {SCH}: train {nT}, weight-0 rows {int((wT == 0).sum())}, effective positives {(tT * wT).sum():.1f}")

def stdz_rows(A, Pm, trmask):
    rr = A[trmask & Pm]; mu, sd = rr.mean(0), rr.std(0) + 1e-6
    return ((A - mu) / sd) * Pm[:, None]

res_oof, res_val = {}, {}
t0 = time.time()
if SCH != "Dmt":
    # ------------------------------------------------ 3 mm window probes
    z = np.load(D + "emb_lumbar_L1_trval.npz"); ids = list(z["ids"]); units = [str(u) for u in z["units"]]
    pos = {s: i for i, s in enumerate(ids)}; ui = units.index("lumbar_spine")
    rT = np.array([pos[i] for i in tr_ids]); rV = np.array([pos[i] for i in va_ids])
    E = z["E"][:, ui].astype(np.float32); P = z["P"][:, ui].astype(np.float32); del z
    E = np.concatenate([E[rT], E[rV]]); P = np.concatenate([P[rT], P[rV]])
    isT = np.r_[np.ones(nT, bool), np.zeros(len(va_ids), bool)]
    pres = (P > 0) & isT; rows = E[pres]; mu, sd = rows.mean(0), np.clip(rows.std(0), 1e-6, None)
    E = ((E - mu) / sd) * (P[:, None, None] > 0)
    E = torch.from_numpy(E[:, None]); P = torch.from_numpy(P[:, None])        # [N,1,W,D], [N,1]
    W = E.shape[2]
    tAll = np.r_[tT, np.zeros(len(va_ids))]; wAll = np.r_[wT, np.zeros(len(va_ids))]
    yESall = np.r_[yES, np.full(len(va_ids), np.nan)]

    def fit(idx_tr, idx_es, idx_pred, spec, seed, lr=1e-3, wd=1e-4, max_ep=100, pat=10, bs=256):
        torch.manual_seed(seed); np.random.seed(seed)
        m = Probe(spec["mode"], 1, W, E.shape[3], [[0]], spec.get("win_idx"))
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        idx_tr = idx_tr[wAll[idx_tr] > 0]
        wp, wn = balanced_weights(tAll[idx_tr], wAll[idx_tr])
        Et, Pt = E[idx_tr], P[idx_tr]; WP, WN = torch.from_numpy(wp).float(), torch.from_numpy(wn).float()
        es = idx_es[~np.isnan(yESall[idx_es])]
        best, bst, bad = -1, None, 0
        for ep in range(max_ep):
            m.train(); perm = torch.randperm(len(Et))
            for i in range(0, len(Et), bs):
                j = perm[i:i + bs]; lg = m(Et[j], Pt[j])[:, 0]
                loss = (WP[j] * F.softplus(-lg) + WN[j] * F.softplus(lg)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): pv = m(E[es], P[es])[:, 0].numpy()
            sc = auc(yESall[es], pv)
            if sc > best + 1e-5: best, bad, bst = sc, 0, {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= pat: break
        m.load_state_dict(bst); m.eval()
        with torch.no_grad(): return m(E[idx_pred], P[idx_pred])[:, 0].numpy()

    WINS = ["baseline", "soft_tissue", "liver", "narrow_abd", "bone"]
    specs = {"E0": {"mode": "single", "win_idx": 0}, "E3_late": {"mode": "late"}}
    for i, w in enumerate(WINS[1:], 1): specs[f"E1_{w}"] = {"mode": "single", "win_idx": i}
    SO = int(os.environ.get("OL_SEED", "0")) * 100
    seeds = {"E0": [42 + SO, 43 + SO], "E3_late": [42 + SO, 43 + SO]}
    rng = np.random.default_rng(SO); vai = np.arange(nT, nT + len(va_ids))
    for name, spec in specs.items():
        col = np.zeros(nT)
        for k in range(5):
            a = np.where(fT != k)[0]; b = np.where(fT == k)[0]
            a = rng.permutation(a); nes = int(0.15 * len(a))
            col[b] = np.mean([fit(a[nes:], a[:nes], b, spec, s) for s in seeds.get(name, [42 + SO])], 0)
        a = rng.permutation(nT); nes = int(0.15 * nT)
        vv = np.mean([fit(a[nes:], a[:nes], vai, spec, s) for s in seeds.get(name, [42 + SO])], 0)
        res_oof[name], res_val[name] = col, vv
        log(f"  {name}: OOF(report) {auc(yT, col):.4f} | {time.time()-t0:.0f}s")
    del E, P

# ------------------------------------------------ 1.5 mm per-level concat LR
z = np.load(D + "ol_emb15_L1L4_trval.npz"); eid = {s: i for i, s in enumerate(z["ids"])}
rT = np.array([eid.get(i, -1) for i in tr_ids]); rV = np.array([eid.get(i, -1) for i in va_ids])
E15 = z["E"].astype(np.float32).reshape(len(eid), 4, -1); P15 = z["P"].copy(); del z
def take(A, r):
    o = np.zeros((len(r),) + A.shape[1:], A.dtype); ok = r >= 0; o[ok] = A[r[ok]]; return o
ET, EV, PT, PV = take(E15, rT), take(E15, rV), take(P15, rT), take(P15, rV); del E15
rr = ET[PT]; mu, sd = rr.mean(0), rr.std(0) + 1e-6
ET = (((ET - mu) / sd) * PT[..., None]).reshape(nT, -1); EV = (((EV - mu) / sd) * PV[..., None]).reshape(len(va_ids), -1)
okT = PT.any(1)

def lr_fit(X, t, w, C):
    Xe, ye, we = expand(X, t, w)
    return LogisticRegression(C=C, max_iter=3000).fit(Xe, ye, sample_weight=we)

def cv_levelcat(t, w, C=1e-4, tag="v15"):
    oof = np.full(nT, np.nan)
    for k in range(5):
        trm = (fT != k) & okT & (w > 0); te = fT == k
        oof[te] = lr_fit(ET[trm], t[trm], w[trm], C).decision_function(ET[te])
    oof[~okT] = np.nan
    m = lr_fit(ET[okT & (w > 0)], t[okT & (w > 0)], w[okT & (w > 0)], C)
    log(f"  {tag}: OOF(report) {auc(yT, oof):.4f} | {time.time()-t0:.0f}s")
    return oof, m.decision_function(EV)

if SCH == "Dmt":
    A_o = pd.read_csv(OUT + "A_oof.csv").set_index("study_id").loc[tr_ids]
    A_v = pd.read_csv(OUT + "A_val.csv").set_index("study_id").loc[va_ids]
    for c in ["E3_late", "E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone", "v15"]:
        res_oof[c], res_val[c] = A_o[c].values, A_v[c].values
    hu = T.hu.values; whu = (~np.isnan(hu)).astype(float); thu = np.nan_to_num((hu < 110).astype(float))
    o, v = cv_levelcat(thu, whu, tag="HU<110 head (target = L1 HU<110)")
    res_oof["hu_head"], res_val["hu_head"] = o, v
    hm = ~np.isnan(hu); log(f"  hu_head OOF AUROC vs HU<110 target: {auc((hu[hm] < 110).astype(int), o[hm]):.4f}")
else:
    res_oof["v15"], res_val["v15"] = cv_levelcat(tT, wT)

# ------------------------------------------------ stack
cols = list(res_oof)
XT = pd.DataFrame(res_oof, index=tr_ids); XV = pd.DataFrame(res_val, index=va_ids)
med = XT.median(); XT = XT.fillna(med).values; XV = XV.fillna(med).values
def st_fit(X, t, w):
    sc = StandardScaler().fit(X); return sc, lr_fit(sc.transform(X), t, w, 0.1)
oof = np.zeros(nT)
for k in range(5):
    trm = (fT != k) & (wT > 0); te = fT == k
    sc, m = st_fit(XT[trm], tT[trm], wT[trm]); oof[te] = m.decision_function(sc.transform(XT[te]))
sc, m = st_fit(XT[wT > 0], tT[wT > 0], wT[wT > 0]); sv = m.decision_function(sc.transform(XV))
res_oof["stack"], res_val["stack"] = oof, sv
info["stack_cols"] = cols; info["stack_coef"] = dict(zip(cols, m.coef_[0].round(4).tolist()))
log(f"  stack: OOF(report, all train) {auc(yT, oof):.4f}; OOF(report, w>0) {auc(yES, oof):.4f} | {time.time()-t0:.0f}s")
pd.DataFrame({"study_id": tr_ids, **res_oof}).to_csv(OUT + f"{SCH}_oof.csv", index=False)
pd.DataFrame({"study_id": va_ids, **res_val}).to_csv(OUT + f"{SCH}_val.csv", index=False)
json.dump(info, open(OUT + f"{SCH}_info.json", "w"), indent=1)
log("DONE", SCH)
