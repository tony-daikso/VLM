"""GS1: gallstone late stack = E3_late + GB HU (quickwins) + absent-GB (missing/volume/fossa/clip) + cholecystectomy-head OOF logit
+ extra 3 mm GB texture/shape. Model (LR vs GBM) chosen on train 5-fold OOF only (same folds as the E3_late OOF); val reported once.
No age/sex as inputs (age only for the age-matched AUROC). Test is never read.
usage: gs1_stack.py <in_dir> <gs1_feat.csv> <out_dir> [native.csv]   (native.csv = gs2_native.py output -> adds V5 variants)"""
import sys, os, json, warnings, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
I, F1, O = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(O, exist_ok=True)
NATF = sys.argv[4] if len(sys.argv) > 4 else None
NAT4 = ["n_md_max", "n_m_cc200", "n_md_cc400", "n_dog4_min"]; NAT = []

GS_HU = ["gb_vox", "gb_p50", "gb_p90", "gb_p95", "gb_p99", "gb_max", "gb_std", "gb_n60", "gb_n100", "gb_n130", "gb_n200",
         "gb_n_med40", "gb_n_lowfat", "shell_n100", "shell_p99", "gb_maxblob100", "liver_hu_med", "aorta_hu_med"]
AG = ["gb_ncc", "gb_lcc_frac", "gb_frac_z", "gb_frac_y", "gb_frac_x", "gb_hu_med", "gb_f_fluid", "gb_f_fat", "gb_f_air", "gb_f_dense",
      "gb_liver_contact", "gb_bowel_contact", "f30_n300", "f30_n500", "f30_n800", "f30_n1200", "f30_max", "f30_blobs500", "f30_nonliver",
      "f30_f_fat", "f30_f_fluid", "f30_f_soft", "f30_gb_frac", "f50_n300", "f50_n500", "f50_n800", "f50_n1200", "f50_max",
      "body_metal1500", "peri_n800", "peri_n1200", "liver_vox"]
G1 = None  # filled from file (all g1_* columns)

def logc(X, cols):
    for c in cols:
        if c in X and (c.startswith(("gb_n", "gb_vox", "shell_n", "gb_max", "f30_n", "f50_n", "peri_n", "body_metal", "liver_vox", "g1_vox", "g1_lcc", "g1_in_vox", "g1_dog_n", "g1_lc_n", "f30_blobs")) or c.endswith("_vox") or (c.startswith("n_") and ("_ml" in c or "_cc" in c))):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X

def feats(d, sets):
    X = pd.DataFrame(index=d.index)
    if "hu" in sets:
        X = pd.concat([X, d[GS_HU].astype(float)], axis=1); X["gb_present"] = (d.gb_vox.fillna(0) >= 100).astype(float)
    if "ag" in sets:
        A = d[["ag_" + c for c in AG]].astype(float).copy(); A.columns = ["ag_" + c for c in AG]
        X = pd.concat([X, A], axis=1)
        X["gb_missing"] = (d.gb_vox.fillna(0) < 100).astype(float); X["gb_logvol"] = np.log1p(d.gb_vox.fillna(0))
        X["gb_missing_x_clip"] = X.gb_missing * np.log1p(d[["ag_peri_n800", "ag_f50_n800"]].fillna(0).sum(1))
    if "ch" in sets:
        X["chole_logit"] = d.chole_logit.astype(float)
        X["chole_x_missing"] = X["chole_logit"] * (d.gb_vox.fillna(0) < 100)
    if "nat4" in sets:
        X = pd.concat([X, d[NAT4].astype(float)], axis=1)
    if "nat" in sets:
        X = pd.concat([X, d[NAT].astype(float)], axis=1)
    if "g1" in sets:
        X = pd.concat([X, d[G1].astype(float)], axis=1)
    X["E3_late"] = d.E3_late.astype(float)
    return logc(X, X.columns)

def design(d, sets, med=None):
    X = feats(d, sets)
    if med is None: med = X.median()
    return X.fillna(med), med

def LR(C): return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
def GB(lr, leaves, msl, it): return lambda: HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, l2_regularization=1.0, random_state=0)
CANDS = {"LR_C0.1": LR(0.1), "LR_C0.3": LR(0.3), "LR_C1": LR(1.0),
         "GBM_lr.05_l15_m40_300": GB(0.05, 15, 40, 300), "GBM_lr.03_l7_m40_500": GB(0.03, 7, 40, 500),
         "GBM_lr.05_l31_m80_200": GB(0.05, 31, 80, 200), "GBM_lr.02_l15_m60_800": GB(0.02, 15, 60, 800)}
def score(m, X): return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]
auc = roc_auc_score

def boot_delta(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
    for _ in range(n):
        ix = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]); d.append(auc(y[ix], b[ix]) - auc(y[ix], a[ix]))
    d = np.array(d); return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float((d <= 0).mean())

def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    a = [auc(y[ix], s[ix]) for ix in (np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]) for _ in range(n))]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def age_matched_auc(y, s, age, edges=(0, 30, 40, 50, 60, 70, 80, 200)):
    b = pd.cut(age, edges, right=False); num = den = 0.0
    for g in b.dropna().unique():
        m = (b == g).values; n1, n0 = y[m].sum(), (1 - y[m]).sum()
        if n1 and n0: num += auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den

def sens90(y, s, thr=None):
    if thr is None: thr = np.quantile(s[y == 0], .9)
    return float((s[y == 1] > thr).mean())

# ---------------- data ----------------
lab = pd.read_csv(f"{I}/analysis_gallstones/labels_trval.csv")
lab = lab[lab.split.isin(["train", "val"])][["study_id", "split", "y_gs"]].rename(columns={"y_gs": "y"})  # split filter first
lab = lab[lab.y.notna()]
age = pd.read_csv(f"{I}/analysis_absent_gb/labels_trval.csv")[["study_id", "split", "Age"]]
age = age[age.split.isin(["train", "val"])][["study_id", "Age"]]
hu = pd.concat([pd.read_csv(f"{I}/analysis_gallstones/{f}") for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
ag = pd.concat([pd.read_csv(f"{I}/analysis_absent_gb/{f}") for f in ("feat_a6.csv", "feat_ct.csv")]).drop_duplicates("study_id")
ag = ag.drop(columns=["gb_vox"]).rename(columns={c: "ag_" + c for c in ag.columns if c != "study_id"})
g1 = pd.read_csv(F1).drop_duplicates("study_id"); G1 = [c for c in g1.columns if c.startswith("g1_")]
if NATF:
    nat = pd.read_csv(NATF).drop_duplicates("study_id")
    NAT = [c for c in nat.columns if c.startswith("n_") and c not in ("n_err", "n_dx", "n_dz", "n_gb3_vox") and pd.api.types.is_numeric_dtype(nat[c])]
    g1 = g1.merge(nat[["study_id"] + NAT], on="study_id", how="left")
df = lab.merge(hu, on="study_id", how="left").merge(ag, on="study_id", how="left").merge(g1, on="study_id", how="left").merge(age, on="study_id", how="left")
oof = pd.read_csv(f"{I}/analysis_gallstones/oof_train.csv")[["study_id", "fold", "E3_late"]]
ch_oof = pd.read_csv(f"{I}/analysis_absent_gb/oof_train.csv")[["study_id", "fold", "E3_late"]].rename(columns={"E3_late": "chole_logit", "fold": "fold_ch"})
vs = pd.read_csv(f"{I}/analysis_gallstones/scores_val_descriptive.csv")[["study_id", "E3_late"]]
ch_v = pd.read_csv(f"{I}/analysis_absent_gb/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
T = df[df.split == "train"].merge(oof, on="study_id").merge(ch_oof, on="study_id", how="left").reset_index(drop=True)
V = df[df.split == "val"].merge(vs, on="study_id").merge(ch_v, on="study_id", how="left").reset_index(drop=True)
print("train", len(T), int(T.y.sum()), "val", len(V), int(V.y.sum()), "fold agreement", float((T.fold == T.fold_ch).mean()),
      "chole missing tr/val", int(T.chole_logit.isna().sum()), int(V.chole_logit.isna().sum()),
      "g1 missing", int(T.g1_vox.isna().sum()), int(V.g1_vox.isna().sum()), flush=True)
y, yv = T.y.values.astype(int), V.y.values.astype(int)

def oof_scores(sets, mk):
    s = np.zeros(len(T))
    for k in sorted(T.fold.unique()):
        tr, te = T.fold.values != k, T.fold.values == k
        Xtr, med = design(T[tr], sets); Xte, _ = design(T[te], sets, med)
        s[te] = score(mk().fit(Xtr, y[tr]), Xte)
    return s

VARIANTS = {"V0 E3+HU (quickwins)": ["hu"], "V1 +缺失/體積/膽囊窩/手術夾": ["hu", "ag"], "V2 V1+膽囊切除 head": ["hu", "ag", "ch"],
            "V3 V0+3mm 紋理/形狀": ["hu", "g1"], "V4 全部": ["hu", "ag", "ch", "g1"]}
if NATF:
    VARIANTS = {"V4 全部": ["hu", "ag", "ch", "g1"], "V5a V4+native4": ["hu", "ag", "ch", "g1", "nat4"], "V5b V4+native全部": ["hu", "ag", "ch", "g1", "nat"]}
    print("native coverage train/val", int(T.n_md_max.notna().sum()), int(V.n_md_max.notna().sum()), flush=True)
base_tr, base_v = T.E3_late.values, V.E3_late.values
qw = pd.read_csv(f"{I}/quickwins/val_scores_gs.csv")[["study_id", "stack"]].rename(columns={"stack": "qw"})
Vq = V[["study_id"]].merge(qw, on="study_id", how="left"); qw_v = Vq.qw.values
qwo = pd.read_csv(f"{I}/quickwins/oof_train_gs.csv")[["study_id", "stack"]].rename(columns={"stack": "qw"})
qw_tr = T[["study_id"]].merge(qwo, on="study_id", how="left").qw.values
print("quickwins check: oof", auc(y, qw_tr), "val", auc(yv[~np.isnan(qw_v)], qw_v[~np.isnan(qw_v)]), flush=True)
res = {"n_train": len(T), "pos_train": int(y.sum()), "n_val": len(V), "pos_val": int(yv.sum()), "E3_late": dict(
    oof=auc(y, base_tr), val=auc(yv, base_v), val_ci=boot_ci(yv, base_v), oof_age=age_matched_auc(y, base_tr, T.Age), val_age=age_matched_auc(yv, base_v, V.Age),
    oof_sens90=sens90(y, base_tr), val_sens90_trthr=sens90(yv, base_v, np.quantile(base_tr[y == 0], .9))),
    "quickwins": dict(oof=auc(y, qw_tr), val=auc(yv, qw_v), val_ci=boot_ci(yv, qw_v), oof_age=age_matched_auc(y, qw_tr, T.Age), val_age=age_matched_auc(yv, qw_v, V.Age)), "variants": {}}
OOF = pd.DataFrame({"study_id": T.study_id, "fold": T.fold, "y": y, "E3_late": base_tr, "quickwins": qw_tr})
VAL = pd.DataFrame({"study_id": V.study_id, "y": yv, "E3_late": base_v, "quickwins": qw_v})
miss_t = (T.gb_vox.fillna(0) < 100).values; vis_t = (T.gb_maxblob100.fillna(0) >= 2).values
miss_v = (V.gb_vox.fillna(0) < 100).values; vis_v = (V.gb_maxblob100.fillna(0) >= 2).values
def sub(yy, s, msk):  # positives in subgroup vs all negatives
    k = (yy == 0) | msk; return float(auc(yy[k], s[k]))
for vn, sets in VARIANTS.items():
    cv = {cn: float(auc(y, oof_scores(sets, mk))) for cn, mk in CANDS.items()}
    best = max(cv, key=cv.get); print(vn, {k: round(v, 4) for k, v in cv.items()}, "->", best, flush=True)
    s_oof = oof_scores(sets, CANDS[best])
    X, med = design(T, sets); m = CANDS[best]().fit(X, y); Xv, _ = design(V, sets, med); sv = score(m, Xv)
    thr = float(np.quantile(s_oof[y == 0], .9))
    r = dict(sets=sets, cand=best, cv=cv, oof=float(auc(y, s_oof)), val=float(auc(yv, sv)), val_ci=boot_ci(yv, sv),
             d_val_vs_E3=boot_delta(yv, base_v, sv), d_val_vs_qw=boot_delta(yv, qw_v, sv), d_oof_vs_E3=boot_delta(y, base_tr, s_oof),
             d_oof_vs_qw=boot_delta(y, qw_tr, s_oof),
             oof_age=age_matched_auc(y, s_oof, T.Age), val_age=age_matched_auc(yv, sv, V.Age),
             oof_sens90=sens90(y, s_oof), val_sens90_trthr=sens90(yv, sv, thr),
             sub_oof=dict(nonvis=sub(y, s_oof, ~vis_t & ~miss_t), vis=sub(y, s_oof, vis_t & ~miss_t), missing=sub(y, s_oof, miss_t)),
             sub_val=dict(nonvis=sub(yv, sv, ~vis_v & ~miss_v), vis=sub(yv, sv, vis_v & ~miss_v), missing=sub(yv, sv, miss_v)))
    res["variants"][vn] = r
    print(vn, json.dumps({k: v for k, v in r.items() if k not in ("cv", "sets")}, ensure_ascii=False, default=float), flush=True)
    OOF[vn.split()[0]] = s_oof; VAL[vn.split()[0]] = sv
    joblib.dump({"model": m, "medians": med, "columns": list(X.columns), "sets": sets, "cand": best, "thr_spec90_trainoof": thr}, f"{O}/stack_gs1_{vn.split()[0]}.joblib")
for nm, s, yy, mm, vv in (("E3_late", None, None, None, None),):
    pass
res["E3_late"]["sub_oof"] = dict(nonvis=sub(y, base_tr, ~vis_t & ~miss_t), vis=sub(y, base_tr, vis_t & ~miss_t), missing=sub(y, base_tr, miss_t))
res["E3_late"]["sub_val"] = dict(nonvis=sub(yv, base_v, ~vis_v & ~miss_v), vis=sub(yv, base_v, vis_v & ~miss_v), missing=sub(yv, base_v, miss_v))
res["quickwins"]["sub_oof"] = dict(nonvis=sub(y, qw_tr, ~vis_t & ~miss_t), vis=sub(y, qw_tr, vis_t & ~miss_t), missing=sub(y, qw_tr, miss_t))
res["quickwins"]["sub_val"] = dict(nonvis=sub(yv, qw_v, ~vis_v & ~miss_v), vis=sub(yv, qw_v, vis_v & ~miss_v), missing=sub(yv, qw_v, miss_v))
res["subgroup_n"] = dict(train=dict(nonvis=int((y == 1)[~vis_t & ~miss_t].sum()), vis=int((y == 1)[vis_t & ~miss_t].sum()), missing=int((y == 1)[miss_t].sum())),
                         val=dict(nonvis=int((yv == 1)[~vis_v & ~miss_v].sum()), vis=int((yv == 1)[vis_v & ~miss_v].sum()), missing=int((yv == 1)[miss_v].sum())))
# single-feature AUROC of the new GS1 features on train (direction-free)
sf = []
for c in G1 + ["ag_" + c for c in AG] + ["chole_logit"]:
    x = T[c].astype(float); k = x.notna().values
    if k.sum() > 100 and len(np.unique(x[k])) > 1:
        a = auc(y[k], x[k]); sf.append((c, float(max(a, 1 - a)), "+" if a >= .5 else "-", int(k.sum())))
pd.DataFrame(sf, columns=["feature", "auroc_train", "dir", "n"]).sort_values("auroc_train", ascending=False).to_csv(f"{O}/gs1_single_feature_auroc_train.csv", index=False)
OOF.to_csv(f"{O}/gs1_oof_train.csv", index=False); VAL.to_csv(f"{O}/gs1_val_scores.csv", index=False)
json.dump(res, open(f"{O}/gs1_summary.json", "w"), ensure_ascii=False, indent=1, default=float)
print("saved", flush=True)
