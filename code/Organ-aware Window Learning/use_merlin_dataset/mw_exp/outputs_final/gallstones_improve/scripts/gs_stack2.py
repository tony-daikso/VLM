"""GS3–GS6 stack on top of GS1 V4 (same features, same candidates, same folds as gs1_stack.py).
Extra feature sets (all train 5-fold OOF on the GS1 folds; val = full-train head):
  win : per-window E0/E1 gallstone logits (analysis_gallstones/oof_train.csv, probe.py recipe)
  oth : 13 other findings' E3_late logits (gs4/oof13_*.csv; gallstone label never used)
  gbx : GB-dilated crop-unit probe logits (gs5/gbx_oof_train.csv / gbx_val.csv)
Override (GS6): --override feat.csv replaces hu/ag/g1 columns of the listed study_ids (full-res GB re-segmentation).
Model per variant chosen on train OOF only; val reported once; deltas vs V4 = paired stratified bootstrap (2,000).
No age/sex inputs (age only for age-matched AUROC). Test is never read.
usage: gs_stack2.py <out_dir> "<name>=set,set;<name>=..." [--override feat.csv] [--perm name,name]"""
import sys, os, json, warnings, argparse, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score as auc
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("variants")
ap.add_argument("--override", default=None); ap.add_argument("--gbx-dir", default="gs5"); ap.add_argument("--perm", default="")
ap.add_argument("--cands", default="all", help="all | V4 (only the V4 GBM) for quick runs")
A = ap.parse_args()
I = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final"; G = f"{I}/gallstones_improve"; O = A.out; os.makedirs(O, exist_ok=True)

GS_HU = ["gb_vox", "gb_p50", "gb_p90", "gb_p95", "gb_p99", "gb_max", "gb_std", "gb_n60", "gb_n100", "gb_n130", "gb_n200",
         "gb_n_med40", "gb_n_lowfat", "shell_n100", "shell_p99", "gb_maxblob100", "liver_hu_med", "aorta_hu_med"]
AG = ["gb_ncc", "gb_lcc_frac", "gb_frac_z", "gb_frac_y", "gb_frac_x", "gb_hu_med", "gb_f_fluid", "gb_f_fat", "gb_f_air", "gb_f_dense",
      "gb_liver_contact", "gb_bowel_contact", "f30_n300", "f30_n500", "f30_n800", "f30_n1200", "f30_max", "f30_blobs500", "f30_nonliver",
      "f30_f_fat", "f30_f_fluid", "f30_f_soft", "f30_gb_frac", "f50_n300", "f50_n500", "f50_n800", "f50_n1200", "f50_max",
      "body_metal1500", "peri_n800", "peri_n1200", "liver_vox"]
WIN = ["E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]

def logc(X):
    for c in X.columns:
        if c.startswith(("gb_n", "gb_vox", "shell_n", "gb_max", "f30_n", "f50_n", "peri_n", "body_metal", "liver_vox", "g1_vox", "g1_lcc", "g1_in_vox", "g1_dog_n", "g1_lc_n", "f30_blobs")) or c.endswith("_vox"):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X

EXTRA_COLS = {}
def feats(d, sets):
    X = pd.DataFrame(index=d.index)
    if "hu" in sets:
        X = pd.concat([X, d[GS_HU].astype(float)], axis=1); X["gb_present"] = (d.gb_vox.fillna(0) >= 100).astype(float)
    if "ag" in sets:
        X = pd.concat([X, d[["ag_" + c for c in AG]].astype(float)], axis=1)
        X["gb_missing"] = (d.gb_vox.fillna(0) < 100).astype(float); X["gb_logvol"] = np.log1p(d.gb_vox.fillna(0))
        X["gb_missing_x_clip"] = X.gb_missing * np.log1p(d[["ag_peri_n800", "ag_f50_n800"]].fillna(0).sum(1))
    if "ch" in sets:
        X["chole_logit"] = d.chole_logit.astype(float); X["chole_x_missing"] = X["chole_logit"] * (d.gb_vox.fillna(0) < 100)
    if "g1" in sets:
        X = pd.concat([X, d[G1].astype(float)], axis=1)
    for s in sets:
        if s in EXTRA_COLS:
            X = pd.concat([X, d[EXTRA_COLS[s]].astype(float)], axis=1)
        elif ":" in s:  # single column of an extra set, e.g. g3b:g3b_livfb
            X[s.split(":")[1]] = d[s.split(":")[1]].astype(float)
    X["E3_late"] = d.E3_late.astype(float)
    return logc(X)

def design(d, sets, med=None):
    X = feats(d, sets)
    if med is None: med = X.median()
    return X.fillna(med), med

def LR(C): return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
def GB(lr, leaves, msl, it): return lambda: HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, l2_regularization=1.0, random_state=0)
CANDS = {"LR_C0.1": LR(0.1), "LR_C0.3": LR(0.3), "LR_C1": LR(1.0),
         "GBM_lr.05_l15_m40_300": GB(0.05, 15, 40, 300), "GBM_lr.03_l7_m40_500": GB(0.03, 7, 40, 500),
         "GBM_lr.05_l31_m80_200": GB(0.05, 31, 80, 200), "GBM_lr.02_l15_m60_800": GB(0.02, 15, 60, 800)}
if A.cands == "V4": CANDS = {"GBM_lr.05_l31_m80_200": CANDS["GBM_lr.05_l31_m80_200"]}
def score(m, X): return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]

def _bidx(y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
def boot_delta(y, a, b, n=2000, seed=0, f=None):
    f = f or (lambda yy, s, ix: auc(yy, s)); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        ix = _bidx(y, rng); d.append(f(y[ix], b[ix], ix) - f(y[ix], a[ix], ix))
    d = np.array(d); return dict(mean=float(d.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))
def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); a = [auc(y[ix], s[ix]) for ix in (_bidx(y, rng) for _ in range(n))]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
EDGES = (0, 30, 40, 50, 60, 70, 80, 200)
def age_matched_auc(y, s, age):
    b = pd.cut(pd.Series(np.asarray(age, float)), EDGES, right=False).values; num = den = 0.0
    for g in pd.unique(b[~pd.isna(b)]):
        m = (b == g); n1, n0 = y[m].sum(), (1 - y[m]).sum()
        if n1 and n0: num += auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den
def sens90(y, s, thr=None):
    if thr is None: thr = np.quantile(s[y == 0], .9)
    return float((s[y == 1] > thr).mean())

# ---------------- data (split filter before labels) ----------------
lab = pd.read_csv(f"{I}/analysis_gallstones/labels_trval.csv")
lab = lab[lab.split.isin(["train", "val"])][["study_id", "split", "y_gs"]].rename(columns={"y_gs": "y"}); lab = lab[lab.y.notna()]
age = pd.read_csv(f"{I}/analysis_absent_gb/labels_trval.csv")[["study_id", "split", "Age"]]
age = age[age.split.isin(["train", "val"])][["study_id", "Age"]]
hu = pd.concat([pd.read_csv(f"{I}/analysis_gallstones/{f}") for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
ag = pd.concat([pd.read_csv(f"{I}/analysis_absent_gb/{f}") for f in ("feat_a6.csv", "feat_ct.csv")]).drop_duplicates("study_id")
ag = ag.drop(columns=["gb_vox"]).rename(columns={c: "ag_" + c for c in ag.columns if c != "study_id"})
g1 = pd.read_csv(f"{G}/gs1_feat3mm.csv").drop_duplicates("study_id"); G1 = [c for c in g1.columns if c.startswith("g1_")]
if A.override:  # GS6: replace per-case GB features (hu/ag/g1 columns) for re-segmented cases
    ov = pd.read_csv(A.override).drop_duplicates("study_id").set_index("study_id")
    for tab in (hu, ag, g1):
        tab.set_index("study_id", inplace=True)
        cols = [c for c in ov.columns if c in tab.columns]; ix = ov.index.intersection(tab.index)
        tab.loc[ix, cols] = ov.loc[ix, cols].values; tab.reset_index(inplace=True)
    print("override rows", len(ov), flush=True)
df = lab.merge(hu, on="study_id", how="left").merge(ag, on="study_id", how="left").merge(g1, on="study_id", how="left").merge(age, on="study_id", how="left")
oof = pd.read_csv(f"{I}/analysis_gallstones/oof_train.csv")
ch_oof = pd.read_csv(f"{I}/analysis_absent_gb/oof_train.csv")[["study_id", "fold", "E3_late"]].rename(columns={"E3_late": "chole_logit", "fold": "fold_ch"})
vs = pd.read_csv(f"{I}/analysis_gallstones/scores_val_descriptive.csv")
ch_v = pd.read_csv(f"{I}/analysis_absent_gb/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
T = df[df.split == "train"].merge(oof, on="study_id").merge(ch_oof, on="study_id", how="left")
V = df[df.split == "val"].merge(vs, on="study_id").merge(ch_v, on="study_id", how="left")
EXTRA_COLS["win"] = WIN
def add_extra(tag, f_tr, f_v, prefix, fold_check=True):
    global T, V
    if not os.path.exists(f_tr): print("missing", f_tr, flush=True); return
    a = pd.read_csv(f_tr); b = pd.read_csv(f_v); cols = [c for c in a.columns if c.startswith(prefix)]
    if fold_check and "fold" in a:
        chk = T[["study_id", "fold"]].merge(a[["study_id", "fold"]].rename(columns={"fold": "f2"}), on="study_id")
        print(tag, "fold agreement", float((chk.fold == chk.f2).mean()), "n", len(chk), flush=True); assert (chk.fold == chk.f2).all()
    T = T.merge(a[["study_id"] + cols], on="study_id", how="left"); V = V.merge(b[["study_id"] + cols], on="study_id", how="left")
    EXTRA_COLS[tag] = cols
    print(tag, cols[:20], "missing tr/val", int(T[cols[0]].isna().sum()), int(V[cols[0]].isna().sum()), flush=True)
add_extra("oth", f"{G}/gs4/oof13_train.csv", f"{G}/gs4/oof13_val.csv", "o13_")
add_extra("g3b", f"{G}/gs3b/gs3b_oof_train.csv", f"{G}/gs3b/gs3b_val.csv", "g3b_")
add_extra("gbx", f"{G}/{A.gbx_dir}/gbx_oof_train.csv", f"{G}/{A.gbx_dir}/gbx_val.csv", "gbx_")
T, V = T.reset_index(drop=True), V.reset_index(drop=True)
y, yv = T.y.values.astype(int), V.y.values.astype(int)
print("train", len(T), int(y.sum()), "val", len(V), int(yv.sum()), "chole fold agreement", float((T.fold == T.fold_ch).mean()), flush=True)

def oof_scores(sets, mk, return_models=False):
    s = np.zeros(len(T)); ms = []
    for k in sorted(T.fold.unique()):
        tr, te = T.fold.values != k, T.fold.values == k
        Xtr, med = design(T[tr], sets); Xte, _ = design(T[te], sets, med)
        m = mk().fit(Xtr, y[tr]); s[te] = score(m, Xte); ms.append((k, m, med))
    return (s, ms) if return_models else s

VARIANTS = {kv.split("=")[0]: kv.split("=")[1].split(",") for kv in A.variants.split(";") if kv}
assert "V4" in VARIANTS
miss_t = (T.gb_vox.fillna(0) < 100).values; vis_t = (T.gb_maxblob100.fillna(0) >= 2).values
miss_v = (V.gb_vox.fillna(0) < 100).values; vis_v = (V.gb_maxblob100.fillna(0) >= 2).values
def sub(yy, s, msk): k = (yy == 0) | msk; return float(auc(yy[k], s[k]))
def subs(yy, s, vis, miss): return dict(nonvis=sub(yy, s, ~vis & ~miss), vis=sub(yy, s, vis & ~miss), missing=sub(yy, s, miss))
res = {"n_train": len(T), "pos_train": int(y.sum()), "n_val": len(V), "pos_val": int(yv.sum()), "variants": {},
       "subgroup_n": dict(train=dict(nonvis=int(y[~vis_t & ~miss_t].sum()), vis=int(y[vis_t & ~miss_t].sum()), missing=int(y[miss_t].sum())),
                          val=dict(nonvis=int(yv[~vis_v & ~miss_v].sum()), vis=int(yv[vis_v & ~miss_v].sum()), missing=int(yv[miss_v].sum())))}
OOF = pd.DataFrame({"study_id": T.study_id, "fold": T.fold, "y": y, "E3_late": T.E3_late}); VAL = pd.DataFrame({"study_id": V.study_id, "y": yv, "E3_late": V.E3_late})
ref = {}
agef = lambda Age: (lambda yy, s, ix: age_matched_auc(yy, s, Age[ix]))
for vn, sets in VARIANTS.items():
    cv = {cn: float(auc(y, oof_scores(sets, mk))) for cn, mk in CANDS.items()} if len(CANDS) > 1 else {}
    best = max(cv, key=cv.get) if cv else next(iter(CANDS))
    s_oof, fms = oof_scores(sets, CANDS[best], True)
    X, med = design(T, sets); m = CANDS[best]().fit(X, y); Xv, _ = design(V, sets, med); sv = score(m, Xv)
    thr = float(np.quantile(s_oof[y == 0], .9))
    r = dict(sets=sets, cand=best, cv=cv, n_feat=X.shape[1], oof=float(auc(y, s_oof)), val=float(auc(yv, sv)), val_ci=boot_ci(yv, sv),
             oof_age=age_matched_auc(y, s_oof, T.Age.values), val_age=age_matched_auc(yv, sv, V.Age.values),
             oof_sens90=sens90(y, s_oof), val_sens90_trthr=sens90(yv, sv, thr), sub_oof=subs(y, s_oof, vis_t, miss_t), sub_val=subs(yv, sv, vis_v, miss_v))
    if vn == "V4": ref = dict(oof=s_oof, val=sv)
    else:
        r["d_oof_vs_V4"] = boot_delta(y, ref["oof"], s_oof); r["d_val_vs_V4"] = boot_delta(yv, ref["val"], sv)
        r["d_oof_age_vs_V4"] = boot_delta(y, ref["oof"], s_oof, n=500, f=agef(T.Age.values)); r["d_val_age_vs_V4"] = boot_delta(yv, ref["val"], sv, n=500, f=agef(V.Age.values))
        r["d_val_sub_vs_V4"] = {g: boot_delta(yv[(yv == 0) | mk_], ref["val"][(yv == 0) | mk_], sv[(yv == 0) | mk_], n=1000)
                                for g, mk_ in dict(nonvis=~vis_v & ~miss_v, vis=vis_v & ~miss_v, missing=miss_v).items()}
        r["d_oof_sub_vs_V4"] = {g: boot_delta(y[(y == 0) | mk_], ref["oof"][(y == 0) | mk_], s_oof[(y == 0) | mk_], n=1000)
                                for g, mk_ in dict(nonvis=~vis_t & ~miss_t, vis=vis_t & ~miss_t, missing=miss_t).items()}
    if vn in A.perm.split(","):  # permutation importance on the held-out folds (train OOF), 3 repeats
        rng = np.random.default_rng(1); imp = {}
        for k, mk_, md in fms:
            te = T.fold.values == k; Xte, _ = design(T[te], sets, md); b0 = auc(y[te], score(mk_, Xte))
            for c in Xte.columns:
                dl = []
                for _ in range(3):
                    Xp = Xte.copy(); Xp[c] = rng.permutation(Xp[c].values); dl.append(b0 - auc(y[te], score(mk_, Xp)))
                imp.setdefault(c, []).append(float(np.mean(dl)))
        r["perm_importance_oof"] = dict(sorted({c: float(np.mean(v)) for c, v in imp.items()}.items(), key=lambda kv: -kv[1]))
    res["variants"][vn] = r
    print(vn, json.dumps({k: v for k, v in r.items() if k not in ("cv", "sets", "perm_importance_oof")}, ensure_ascii=False, default=float), flush=True)
    if cv: print("  cv", {k: round(v, 4) for k, v in cv.items()}, "->", best, flush=True)
    if "perm_importance_oof" in r: print("  perm top", list(r["perm_importance_oof"].items())[:25], flush=True)
    OOF[vn] = s_oof; VAL[vn] = sv
    joblib.dump({"model": m, "medians": med, "columns": list(X.columns), "sets": sets, "cand": best, "thr_spec90_trainoof": thr}, f"{O}/stack_{vn}.joblib")
# V4 reproduction check against GS1
g1o = pd.read_csv(f"{G}/gs1/gs1_oof_train.csv")[["study_id", "V4"]].rename(columns={"V4": "V4_gs1"})
chk = OOF[["study_id", "V4"]].merge(g1o, on="study_id"); res["V4_repro_corr"] = float(np.corrcoef(chk.V4, chk.V4_gs1)[0, 1])
print("V4 reproduction corr vs gs1", res["V4_repro_corr"], flush=True)
# age-proxy diagnostics for the extra (non-image-of-GB) features
res["age_diag"] = {}
for s in ("oth", "win", "gbx", "g3b"):
    for c in EXTRA_COLS.get(s, []):
        x = T[c].astype(float); k = x.notna().values
        res["age_diag"][c] = dict(spearman_age=float(pd.Series(x[k]).corr(T.Age[k], method="spearman")),
                                  auroc_gs_train=float(auc(y[k], x[k])))
res["age_alone_auroc_train"] = float(auc(y[T.Age.notna().values], T.Age[T.Age.notna()]))
OOF.to_csv(f"{O}/oof_train.csv", index=False); VAL.to_csv(f"{O}/val_scores.csv", index=False)
json.dump(res, open(f"{O}/summary.json", "w"), ensure_ascii=False, indent=1, default=float)
print("saved", flush=True)
