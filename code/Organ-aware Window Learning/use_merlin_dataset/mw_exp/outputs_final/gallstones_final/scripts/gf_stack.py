"""Gallstones final: combine Exp N (G5c + GB/GB-fossa unit head U) with G6c (GS6 re-segmented GB features + GS6 gbx).
Derived from region_units/gallstones/scripts/nu_gs_stack.py (same features, candidates, GS1 folds, bootstrap code).
Two feature frames are built:
  frame 0 = original 3 mm features (as Exp N), gbx = gs5 crop-unit logits
  frame 1 = GS6 override_feat.csv replaces hu/ag/g1 columns of the 5,034 re-segmented cases (as G6c)
  extra sets: gbx (gs5/gbx_*), gbx6 (gs6/gbx/gbx_* renamed gbx6_*), g3b, gbn (Exp N unit heads, GS1 folds)
Variants (name=frame|sets):
  N_U  (REF, frame 0) = Exp N best;  G6c (frame 1);  a_G6c_U (frame 1, G6c + gbn_U);  b_N_U_ov (frame 1, Exp N sets + override)
  c_avg = mean of z-scored N_U and G6c logits (z with each model's train-OOF mean/sd; same constants applied to val; no tuning)
Model per variant chosen on train OOF only; val reported once; deltas vs N_U = paired stratified bootstrap (2,000).
Subgroups always use the ORIGINAL 3 mm masks (gb_vox<100 = GB-missing; gb_maxblob100>=2 = visible).
No age/sex inputs, no other-finding logits. Test is never read (inputs are train/val slices).
usage: gf_stack.py <data_root> <out_dir>"""
import sys, os, json, warnings, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score as auc
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
I, O = sys.argv[1], sys.argv[2]; G = f"{I}/gallstones_improve"; os.makedirs(O, exist_ok=True)

GS_HU = ["gb_vox", "gb_p50", "gb_p90", "gb_p95", "gb_p99", "gb_max", "gb_std", "gb_n60", "gb_n100", "gb_n130", "gb_n200",
         "gb_n_med40", "gb_n_lowfat", "shell_n100", "shell_p99", "gb_maxblob100", "liver_hu_med", "aorta_hu_med"]
AG = ["gb_ncc", "gb_lcc_frac", "gb_frac_z", "gb_frac_y", "gb_frac_x", "gb_hu_med", "gb_f_fluid", "gb_f_fat", "gb_f_air", "gb_f_dense",
      "gb_liver_contact", "gb_bowel_contact", "f30_n300", "f30_n500", "f30_n800", "f30_n1200", "f30_max", "f30_blobs500", "f30_nonliver",
      "f30_f_fat", "f30_f_fluid", "f30_f_soft", "f30_gb_frac", "f50_n300", "f50_n500", "f50_n800", "f50_n1200", "f50_max",
      "body_metal1500", "peri_n800", "peri_n1200", "liver_vox"]

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
        elif ":" in s:
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
def spearman_neg(y, s, age):
    m = (np.asarray(y) == 0) & ~np.isnan(np.asarray(age, float))
    return float(spearmanr(np.asarray(s)[m], np.asarray(age, float)[m]).correlation)
def sens90(y, s, thr=None):
    if thr is None: thr = np.quantile(s[y == 0], .9)
    return float((s[y == 1] > thr).mean())

# ---------------- data ----------------
lab = pd.read_csv(f"{I}/analysis_gallstones/labels_trval.csv")
lab = lab[lab.split.isin(["train", "val"])][["study_id", "split", "y_gs"]].rename(columns={"y_gs": "y"}); lab = lab[lab.y.notna()]
age = pd.read_csv(f"{I}/analysis_absent_gb/labels_trval.csv")[["study_id", "split", "Age"]]
age = age[age.split.isin(["train", "val"])][["study_id", "Age"]]
HU0 = pd.concat([pd.read_csv(f"{I}/analysis_gallstones/{f}") for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
AG0 = pd.concat([pd.read_csv(f"{I}/analysis_absent_gb/{f}") for f in ("feat_a6.csv", "feat_ct.csv")]).drop_duplicates("study_id")
AG0 = AG0.drop(columns=["gb_vox"]).rename(columns={c: "ag_" + c for c in AG0.columns if c != "study_id"})
G10 = pd.read_csv(f"{G}/gs1_feat3mm.csv").drop_duplicates("study_id"); G1 = [c for c in G10.columns if c.startswith("g1_")]
OV = pd.read_csv(f"{G}/gs6/override_feat.csv").drop_duplicates("study_id").set_index("study_id")
oof = pd.read_csv(f"{I}/analysis_gallstones/oof_train.csv")
ch_oof = pd.read_csv(f"{I}/analysis_absent_gb/oof_train.csv")[["study_id", "fold", "E3_late"]].rename(columns={"E3_late": "chole_logit", "fold": "fold_ch"})
vs = pd.read_csv(f"{I}/analysis_gallstones/scores_val_descriptive.csv")
ch_v = pd.read_csv(f"{I}/analysis_absent_gb/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
EXTRA = [("g3b", f"{G}/gs3b/gs3b_oof_train.csv", f"{G}/gs3b/gs3b_val.csv", "g3b_", None),
         ("gbx", f"{G}/gs5/gbx_oof_train.csv", f"{G}/gs5/gbx_val.csv", "gbx_", None),
         ("gbx6", f"{G}/gs6/gbx/gbx_oof_train.csv", f"{G}/gs6/gbx/gbx_val.csv", "gbx_", "gbx6_"),
         ("gbn", f"{I}/region_units/gallstones/stack/gbn_oof_train.csv", f"{I}/region_units/gallstones/stack/gbn_val.csv", "gbn_", None)]

def build(override):
    hu, ag, g1 = HU0.copy(), AG0.copy(), G10.copy()
    if override:
        for tab in (hu, ag, g1):
            tab.set_index("study_id", inplace=True)
            cols = [c for c in OV.columns if c in tab.columns]; ix = OV.index.intersection(tab.index)
            tab.loc[ix, cols] = OV.loc[ix, cols].values; tab.reset_index(inplace=True)
    df = lab.merge(hu, on="study_id", how="left").merge(ag, on="study_id", how="left").merge(g1, on="study_id", how="left").merge(age, on="study_id", how="left")
    T = df[df.split == "train"].merge(oof, on="study_id").merge(ch_oof, on="study_id", how="left")
    V = df[df.split == "val"].merge(vs, on="study_id").merge(ch_v, on="study_id", how="left")
    for tag, ftr, fv, prefix, newp in EXTRA:
        a = pd.read_csv(ftr); b = pd.read_csv(fv); cols = [c for c in a.columns if c.startswith(prefix)]
        if newp:
            ren = {c: newp + c[len(prefix):] for c in cols}; a = a.rename(columns=ren); b = b.rename(columns=ren); cols = list(ren.values())
        if "fold" in a:
            chk = T[["study_id", "fold"]].merge(a[["study_id", "fold"]].rename(columns={"fold": "f2"}), on="study_id"); assert (chk.fold == chk.f2).all(), tag
        T = T.merge(a[["study_id"] + cols], on="study_id", how="left"); V = V.merge(b[["study_id"] + cols], on="study_id", how="left")
        EXTRA_COLS[tag] = cols
    return T.reset_index(drop=True), V.reset_index(drop=True)
FR = {0: build(False), 1: build(True)}
T0, V0 = FR[0]
for k in (1,):
    assert (FR[k][0].study_id.values == T0.study_id.values).all() and (FR[k][1].study_id.values == V0.study_id.values).all()
y, yv = T0.y.values.astype(int), V0.y.values.astype(int); folds = T0.fold.values
print("train", len(T0), int(y.sum()), "val", len(V0), int(yv.sum()), "override cases changed (train gb_vox)",
      int((FR[1][0].gb_vox.fillna(-1).values != T0.gb_vox.fillna(-1).values).sum()), flush=True)

# subgroups from ORIGINAL 3 mm masks
miss_t = (T0.gb_vox.fillna(0) < 100).values; vis_t = (T0.gb_maxblob100.fillna(0) >= 2).values
miss_v = (V0.gb_vox.fillna(0) < 100).values; vis_v = (V0.gb_maxblob100.fillna(0) >= 2).values
GR_T = dict(nonvis=~vis_t & ~miss_t, vis=vis_t & ~miss_t, missing=miss_t); GR_V = dict(nonvis=~vis_v & ~miss_v, vis=vis_v & ~miss_v, missing=miss_v)
def subs(yy, s, gr): return {g: float(auc(yy[(yy == 0) | m], s[(yy == 0) | m])) for g, m in gr.items()}

def oof_scores(T, sets, mk, return_models=False):
    s = np.zeros(len(T)); ms = []
    for k in sorted(np.unique(folds)):
        tr, te = folds != k, folds == k
        Xtr, med = design(T[tr], sets); Xte, _ = design(T[te], sets, med)
        m = mk().fit(Xtr, y[tr]); s[te] = score(m, Xte); ms.append((k, m, med))
    return (s, ms) if return_models else s

SETS_N = ["hu", "ag", "ch", "g1", "gbx", "g3b:g3b_livfb", "gbn:gbn_U"]
SETS_G6c = ["hu", "ag", "ch", "g1", "gbx6", "g3b:g3b_livfb"]
VARIANTS = {"N_U": (0, SETS_N), "G6c": (1, SETS_G6c), "a_G6c_U": (1, SETS_G6c + ["gbn:gbn_U"]), "b_N_U_ov": (1, SETS_N)}
OOF = pd.DataFrame({"study_id": T0.study_id, "fold": folds, "y": y}); VAL = pd.DataFrame({"study_id": V0.study_id, "y": yv})
res = {"n_train": len(T0), "pos_train": int(y.sum()), "n_val": len(V0), "pos_val": int(yv.sum()), "variants": {},
       "subgroup_n_orig3mm": dict(train={g: int(y[m].sum()) for g, m in GR_T.items()}, val={g: int(yv[m].sum()) for g, m in GR_V.items()})}
agef = lambda Age: (lambda yy, s, ix: age_matched_auc(yy, s, Age[ix]))
AGE_T, AGE_V = T0.Age.values, V0.Age.values

def evaluate(vn, s_oof, sv, extra):
    thr = float(np.quantile(s_oof[y == 0], .9))
    r = dict(extra, oof=float(auc(y, s_oof)), val=float(auc(yv, sv)), val_ci=boot_ci(yv, sv),
             oof_age=age_matched_auc(y, s_oof, AGE_T), val_age=age_matched_auc(yv, sv, AGE_V),
             spearman_neg_oof=spearman_neg(y, s_oof, AGE_T), spearman_neg_val=spearman_neg(yv, sv, AGE_V),
             oof_sens90=sens90(y, s_oof), val_sens90=sens90(yv, sv), val_sens90_trthr=sens90(yv, sv, thr),
             sub_oof=subs(y, s_oof, GR_T), sub_val=subs(yv, sv, GR_V))
    if vn != "N_U":
        ro, rv = OOF["N_U"].values, VAL["N_U"].values
        r["d_oof_vs_N"] = boot_delta(y, ro, s_oof); r["d_val_vs_N"] = boot_delta(yv, rv, sv)
        r["d_oof_age_vs_N"] = boot_delta(y, ro, s_oof, n=500, f=agef(AGE_T)); r["d_val_age_vs_N"] = boot_delta(yv, rv, sv, n=500, f=agef(AGE_V))
        r["d_val_sub_vs_N"] = {g: boot_delta(yv[(yv == 0) | m], rv[(yv == 0) | m], sv[(yv == 0) | m], n=1000) for g, m in GR_V.items()}
        r["d_oof_sub_vs_N"] = {g: boot_delta(y[(y == 0) | m], ro[(y == 0) | m], s_oof[(y == 0) | m], n=1000) for g, m in GR_T.items()}
    res["variants"][vn] = r; OOF[vn] = s_oof; VAL[vn] = sv
    print(vn, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("cand", "n_feat", "oof", "val", "val_ci", "oof_age", "val_age", "spearman_neg_val", "sub_val")}, default=float), flush=True)
    if "d_oof_vs_N" in r: print("   dOOF", r["d_oof_vs_N"], "dVAL", r["d_val_vs_N"], flush=True)
    return thr

for vn, (fr, sets) in VARIANTS.items():
    T, V = FR[fr]
    cv = {cn: float(auc(y, oof_scores(T, sets, mk))) for cn, mk in CANDS.items()}
    best = max(cv, key=cv.get)
    s_oof = oof_scores(T, sets, CANDS[best])
    X, med = design(T, sets); m = CANDS[best]().fit(X, y); Xv, _ = design(V, sets, med); sv = score(m, Xv)
    print(vn, "cv", {k: round(v, 4) for k, v in cv.items()}, "->", best, flush=True)
    thr = evaluate(vn, s_oof, sv, dict(frame=fr, sets=sets, cand=best, cv=cv, n_feat=X.shape[1]))
    joblib.dump({"model": m, "medians": med, "columns": list(X.columns), "sets": sets, "override_gs6": bool(fr), "cand": best,
                 "thr_spec90_trainoof": thr}, f"{O}/stack_{vn}.joblib")

# (c) fixed equal-weight average of z-scored logits (z constants from train OOF of each model; applied unchanged to val)
zc = {k: (float(OOF[k].mean()), float(OOF[k].std())) for k in ("N_U", "G6c")}
co = sum((OOF[k].values - mu) / sd for k, (mu, sd) in zc.items()) / 2; cvv = sum((VAL[k].values - mu) / sd for k, (mu, sd) in zc.items()) / 2
thr = evaluate("c_avg", co, cvv, dict(frame="avg", sets=["N_U", "G6c"], cand="z-mean(N_U,G6c), equal weights", z_consts=zc))
joblib.dump({"type": "z_mean", "members": {"N_U": "stack_N_U.joblib", "G6c": "stack_G6c.joblib"}, "z_consts": zc, "thr_spec90_trainoof": thr}, f"{O}/stack_c_avg.joblib")

# reproduction checks
ru = pd.read_csv(f"{I}/region_units/gallstones/stack/oof_train.csv")[["study_id", "N_U"]].rename(columns={"N_U": "ref"})
fc = pd.read_csv(f"{G}/final_cmp_oof.csv")[["study_id", "G6c"]].rename(columns={"G6c": "ref"})
res["repro_corr"] = {"N_U_vs_expN": float(OOF[["study_id", "N_U"]].merge(ru, on="study_id")[["N_U", "ref"]].corr().iloc[0, 1]),
                     "G6c_vs_final_cmp": float(OOF[["study_id", "G6c"]].merge(fc, on="study_id")[["G6c", "ref"]].corr().iloc[0, 1])}
res["corr_N_U_G6c"] = dict(oof=float(np.corrcoef(OOF.N_U, OOF.G6c)[0, 1]), val=float(np.corrcoef(VAL.N_U, VAL.G6c)[0, 1]))
print("repro", res["repro_corr"], "corr N_U/G6c", res["corr_N_U_G6c"], flush=True)
cands = ["a_G6c_U", "b_N_U_ov", "c_avg"]
res["winner_by_train_oof"] = max(cands, key=lambda k: res["variants"][k]["oof"])
res["beats_N_on_oof"] = res["variants"][res["winner_by_train_oof"]]["oof"] > res["variants"]["N_U"]["oof"]
print("winner", res["winner_by_train_oof"], "beats N", res["beats_N_on_oof"], flush=True)
OOF.to_csv(f"{O}/oof_train.csv", index=False); VAL.to_csv(f"{O}/val_scores.csv", index=False)
json.dump(res, open(f"{O}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
rows = []
for vn, r in res["variants"].items():
    rows.append(dict(variant=vn, cand=r["cand"], n_feat=r.get("n_feat", ""), oof=r["oof"], d_oof=r.get("d_oof_vs_N", {}).get("mean"),
                     d_oof_lo=r.get("d_oof_vs_N", {}).get("ci", [None, None])[0], d_oof_hi=r.get("d_oof_vs_N", {}).get("ci", [None, None])[1],
                     val=r["val"], val_lo=r["val_ci"][0], val_hi=r["val_ci"][1], d_val=r.get("d_val_vs_N", {}).get("mean"),
                     d_val_lo=r.get("d_val_vs_N", {}).get("ci", [None, None])[0], d_val_hi=r.get("d_val_vs_N", {}).get("ci", [None, None])[1],
                     oof_age=r["oof_age"], val_age=r["val_age"], spearman_neg_oof=r["spearman_neg_oof"], spearman_neg_val=r["spearman_neg_val"],
                     val_sens90=r["val_sens90"], val_sens90_trthr=r["val_sens90_trthr"],
                     **{f"oof_{g}": v for g, v in r["sub_oof"].items()}, **{f"val_{g}": v for g, v in r["sub_val"].items()}))
pd.DataFrame(rows).to_csv(f"{O}/results.csv", index=False)
print("saved", flush=True)
