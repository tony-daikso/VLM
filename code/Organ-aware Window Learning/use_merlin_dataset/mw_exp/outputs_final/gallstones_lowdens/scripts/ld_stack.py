"""Exp N (N_U) + low-density GB features (gallstones_lowdens). Data machinery copied verbatim from
gallbladder_sludge/scripts/sl_stack.py (rows/feature defs/candidates/bootstrap/age-matched); Part A (sludge) dropped.
Train 5-fold OOF selection only; val reported; test never read (all inputs are train/val files, asserted).
usage: ld_stack.py DATA HEADS OUT feats_lowdens.csv"""

import sys, os, json, warnings, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score as auc
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
D, H, O, LDF = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]; os.makedirs(O, exist_ok=True)
QUICK = "--quick" in sys.argv

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


def feats(d, sets, e3):
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
        if s in COLS: X = pd.concat([X, d[COLS[s]].astype(float)], axis=1)
    X[e3] = d[e3].astype(float)
    return logc(X)


def design(d, sets, e3, med=None):
    X = feats(d, sets, e3)
    if med is None: med = X.median()
    return X.fillna(med), med


def LR(C, multi=False): return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
def GB(lr, leaves, msl, it): return lambda: HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, l2_regularization=1.0, random_state=0)
CANDS = {"LR_C0.1": LR(0.1), "LR_C0.3": LR(0.3), "LR_C1": LR(1.0),
         "GBM_lr.05_l15_m40_300": GB(0.05, 15, 40, 300), "GBM_lr.03_l7_m40_500": GB(0.03, 7, 40, 500),
         "GBM_lr.05_l31_m80_200": GB(0.05, 31, 80, 200), "GBM_lr.02_l15_m60_800": GB(0.02, 15, 60, 800)}


def score(m, X, cls=None):
    if cls is not None:  # multiclass: logit of P(class)
        p = np.clip(m.predict_proba(X)[:, list(m.classes_).index(cls)], 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


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


# ---------------- data (slice holds train/val only) ----------------
L = pd.read_csv(f"{D}/gallbladder_sludge/labels_trval_sludge.csv"); assert set(L.split) <= {"train", "val"}
L = L[["study_id", "split", "y_gs", "y_sludge", "y_sludge_only", "y3", "y_gs_excl_sludgeonly", "sl_group", "sl_unc", "kw_vicarious"]]
age = pd.read_csv(f"{D}/analysis_absent_gb/labels_trval.csv")[["study_id", "Age"]] if "Age" in open(f"{D}/analysis_absent_gb/labels_trval.csv").readline() else None
hu = pd.concat([pd.read_csv(f"{D}/analysis_gallstones/{f}") for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
ag = pd.concat([pd.read_csv(f"{D}/analysis_absent_gb/{f}") for f in ("feat_a6.csv", "feat_ct.csv")]).drop_duplicates("study_id")
ag = ag.drop(columns=["gb_vox"]).rename(columns={c: "ag_" + c for c in ag.columns if c != "study_id"})
g1 = pd.read_csv(f"{D}/gallstones_improve/gs1_feat3mm.csv").drop_duplicates("study_id"); G1 = [c for c in g1.columns if c.startswith("g1_")]
fl = pd.read_csv(f"{D}/analysis_gallstones/subgroups/case_flags_trval.csv")[["study_id", "sludge"]].rename(columns={"sludge": "old_sludge"})
df = L.merge(hu, on="study_id", how="left").merge(ag, on="study_id", how="left").merge(g1, on="study_id", how="left").merge(fl, on="study_id", how="left")
if age is not None: df = df.merge(age, on="study_id", how="left")
fold = pd.read_csv(f"{D}/analysis_absent_gb/oof_train.csv")[["study_id", "fold", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
chv = pd.read_csv(f"{D}/analysis_absent_gb/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
T = df[df.split == "train"].merge(fold, on="study_id", how="left"); V = df[df.split == "val"].merge(chv, on="study_id", how="left")
COLS = {}
def add(tag, ftr, fv, cols, rename=None):
    global T, V
    a = pd.read_csv(ftr); b = pd.read_csv(fv)
    if rename: a = a.rename(columns=rename); b = b.rename(columns=rename)
    if "fold" in a:
        chk = T[["study_id", "fold"]].merge(a[["study_id", "fold"]].rename(columns={"fold": "f2"}), on="study_id"); assert (chk.fold == chk.f2).all(), tag
    T = T.merge(a[["study_id"] + cols], on="study_id", how="left"); V = V.merge(b[["study_id"] + cols], on="study_id", how="left")
    COLS[tag] = cols; print(tag, cols, "missing tr/val", int(T[cols[0]].isna().sum()), int(V[cols[0]].isna().sum()), flush=True)
# gallstone heads (official gallstone label)
add("gs_e3", f"{D}/analysis_gallstones/oof_train.csv", f"{D}/analysis_gallstones/scores_val_descriptive.csv", ["E3_late"])
add("g3b", f"{D}/gallstones_improve/gs3b/gs3b_oof_train.csv", f"{D}/gallstones_improve/gs3b/gs3b_val.csv", ["g3b_livfb"])
add("gbx", f"{D}/gallstones_improve/gs5/gbx_oof_train.csv", f"{D}/gallstones_improve/gs5/gbx_val.csv", ["gbx_r0", "gbx_r1", "gbx_cat"])
add("gbn", f"{D}/region_units/gallstones/stack/gbn_oof_train.csv", f"{D}/region_units/gallstones/stack/gbn_val.csv", ["gbn_U"])
# sludge heads
for tg, short in (("y_sludge", "sl"), ("y_sludge_only", "so")):
    hv = ["h_base", "h_livfb", "h_gbx_r0", "h_gbx_r1", "h_gbx_cat", "h_U"]
    a = pd.read_csv(f"{H}/oof_{tg}_a.csv").merge(pd.read_csv(f"{H}/oof_{tg}_b.csv").drop(columns=["fold"]), on="study_id")
    b = pd.read_csv(f"{H}/val_{tg}_a.csv").merge(pd.read_csv(f"{H}/val_{tg}_b.csv"), on="study_id")
    rn = {c: f"{short}_{c[2:]}" for c in hv}
    a.to_csv(f"{O}/heads_oof_{short}.csv", index=False); b.to_csv(f"{O}/heads_val_{short}.csv", index=False)
    a = a.rename(columns=rn); b = b.rename(columns=rn)
    T = T.merge(a[["study_id"] + list(rn.values())], on="study_id", how="left"); V = V.merge(b[["study_id"] + list(rn.values())], on="study_id", how="left")
    COLS[f"{short}_e3"] = [f"{short}_base"]; COLS[f"{short}_livfb"] = [f"{short}_livfb"]
    COLS[f"{short}_gbx"] = [f"{short}_gbx_r0", f"{short}_gbx_r1", f"{short}_gbx_cat"]; COLS[f"{short}_U"] = [f"{short}_U"]
T = T.reset_index(drop=True); V = V.reset_index(drop=True)
RES = {"labels_sha256_note": "see label_rules.md", "partA": {}, "partB": {}}
AGE_OK = "Age" in T


def run_variant(Tp, Vp, ycol, sets, e3, cands=CANDS, multi_cls=None):
    """returns OOF and val scores of the train-OOF-best candidate."""
    y = Tp[ycol].values.astype(int)
    def oof(mk):
        s = np.zeros(len(Tp))
        for k in sorted(Tp.fold.unique()):
            tr, te = Tp.fold.values != k, Tp.fold.values == k
            Xtr, med = design(Tp[tr], sets, e3); Xte, _ = design(Tp[te], sets, e3, med)
            m = mk().fit(Xtr, y[tr]); s[te] = score(m, Xte, multi_cls)
        return s
    yb = (y == multi_cls).astype(int) if multi_cls is not None else y
    cv = {}; so = {}
    for cn, mk in cands.items():
        so[cn] = oof(mk); cv[cn] = float(auc(yb, so[cn]))
    best = max(cv, key=cv.get)
    X, med = design(Tp, sets, e3); m = cands[best]().fit(X, y); Xv, _ = design(Vp, sets, e3, med)
    return so[best], score(m, Xv, multi_cls), best, cv, dict(model=m, medians=med, columns=list(X.columns), sets=sets, cand=best)


def metrics(y, s_oof, yv, sv, age_t=None, age_v=None):
    thr = float(np.quantile(s_oof[y == 0], .9))
    r = dict(oof=float(auc(y, s_oof)), val=float(auc(yv, sv)), val_ci=boot_ci(yv, sv),
             val_sens90=sens90(yv, sv), val_sens90_trthr=sens90(yv, sv, thr), oof_sens90=sens90(y, s_oof), thr_trainoof_spec90=thr)
    if AGE_OK:
        r.update(oof_age=age_matched_auc(y, s_oof, age_t), val_age=age_matched_auc(yv, sv, age_v),
                 spearman_neg_oof=spearman_neg(y, s_oof, age_t), spearman_neg_val=spearman_neg(yv, sv, age_v))
    return r



# ================= low-density variants (official y_gs) =================
LD = pd.read_csv(LDF); assert set(LD.study_id) <= set(T.study_id) | set(V.study_id)
LDC = [c for c in LD.columns if c.startswith("ld_")]
LDS = ["ld_lum_p1_rel", "ld_lum_p5_rel", "ld_lum_min_rel", "ld_frac_lt_med20", "ld_frac_lt_med30", "ld_frac_lt0", "ld_frac_ltm100",
       "ld_hypo20_maxcc_ml", "ld_hypo20_maxcc_frac", "ld_hypo20_sphericity", "ld_dep_minus_nondep_med", "ld_lum_entropy", "ld_lum_std", "ld_bile_med"]
T = T.merge(LD[["study_id"] + LDC], on="study_id", how="left"); V = V.merge(LD[["study_id"] + LDC], on="study_id", how="left")
COLS["ld"] = LDC; COLS["lds"] = LDS
print("ld missing tr/val", int(T[LDC[2]].isna().sum()), int(V[LDC[2]].isna().sum()), flush=True)
_ord = pd.read_csv(f"{D}/analysis_gallstones/labels_trval.csv")["study_id"].reset_index().set_index("study_id")["index"]
Tg = T[T.y_gs.notna()].assign(_o=lambda d: d.study_id.map(_ord)).sort_values("_o").reset_index(drop=True)
Vg = V[V.y_gs.notna()].assign(_o=lambda d: d.study_id.map(_ord)).sort_values("_o").reset_index(drop=True)
y, yv = Tg.y_gs.values.astype(int), Vg.y_gs.values.astype(int)
aget = Tg.Age.values if AGE_OK else None; agev = Vg.Age.values if AGE_OK else None
NU = ["hu", "ag", "ch", "g1", "gbx", "g3b", "gbn"]
GB_N_U = {"GBM_lr.05_l31_m80_200": CANDS["GBM_lr.05_l31_m80_200"]}
B = {}
so_ref, sv_ref, best, cv, mdl = run_variant(Tg, Vg, "y_gs", NU, "E3_late", cands=GB_N_U)
st = pd.read_csv(f"{D}/gallbladder_sludge/stack/gs_oof_train.csv")[["study_id", "N_U"]]
mm = Tg[["study_id"]].assign(a=so_ref).merge(st, on="study_id")
RES["N_U_repro_corr_vs_sludge_stack"] = float(np.corrcoef(mm.a, mm.N_U)[0, 1]); print("repro corr", RES["N_U_repro_corr_vs_sludge_stack"], flush=True)
B["N_U"] = (so_ref, sv_ref, best, cv, mdl)
B["NU+ld_sameGBM"] = run_variant(Tg, Vg, "y_gs", NU + ["ld"], "E3_late", cands=GB_N_U)
B["NU+lds_sameGBM"] = run_variant(Tg, Vg, "y_gs", NU + ["lds"], "E3_late", cands=GB_N_U)
if not QUICK:
    B["NU+ld_7cand"] = run_variant(Tg, Vg, "y_gs", NU + ["ld"], "E3_late")
    B["NU+lds_7cand"] = run_variant(Tg, Vg, "y_gs", NU + ["lds"], "E3_late")
# level-2: N_U OOF logit + ld-only head (OOF) — ld-only head trained on ld features + gb presence
B_ld_only = run_variant(Tg, Vg, "y_gs", ["ld"], "E3_late", cands=GB_N_U)
RES["ld_only_plus_E3"] = dict(oof=float(auc(y, B_ld_only[0])), val=float(auc(yv, B_ld_only[1])))
OOF = Tg[["study_id", "fold", "y_gs"]].copy(); VAL = Vg[["study_id", "y_gs"]].copy()
for vn, (so, sv, best, cv, mdl) in B.items():
    r = metrics(y, so, yv, sv, aget, agev); r.update(cand=best, cv=cv)
    if vn != "N_U":
        r["d_oof_vs_NU"] = boot_delta(y, so_ref, so); r["d_val_vs_NU"] = boot_delta(yv, sv_ref, sv)
        if AGE_OK: r["d_val_age_vs_NU"] = boot_delta(yv, sv_ref, sv, n=500, f=lambda yy, s, ix: age_matched_auc(yy, s, agev[ix]))
    RES[vn] = r; OOF[vn] = so; VAL[vn] = sv
    if mdl: joblib.dump(mdl, f"{O}/gs_{vn}.joblib")
    print("GS", vn, json.dumps({k: v for k, v in r.items() if k not in ("cv",)}, default=float), flush=True)
RES["selected_by_train_oof"] = max(B, key=lambda v: RES[v]["oof"])
print("GS selected", RES["selected_by_train_oof"], flush=True)
OOF.to_csv(f"{O}/oof_train.csv", index=False); VAL.to_csv(f"{O}/val_scores.csv", index=False)
json.dump(RES, open(f"{O}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print("LD_STACK_DONE", flush=True)
