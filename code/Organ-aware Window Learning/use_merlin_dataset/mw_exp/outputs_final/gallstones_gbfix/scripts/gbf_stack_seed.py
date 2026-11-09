"""Task B stack: Exp N (N_U) exact rebuild + GB-fossa search features for GB-missing cases (3 mm TS GB < 100 vox).
Official gallstone label unchanged; train 5-fold OOF (GS1 folds) selects; val reported once (paired stratified bootstrap 2,000).
Machinery copied from gallbladder_sludge/scripts/sl_stack.py Part B (row order = analysis_gallstones/labels_trval.csv).
Variants:
  N_U            : REF (Exp N, GBM lr.05 l31 m80 200)
  F1_NU+fossa    : N_U features + fossa feats (0 for GB-present rows), 7 LR/GBM candidates, best by OOF
  F1b_NU+fossa_c : same, candidate fixed to the N_U GBM
  F2_L2          : level-2 LR on [N_U OOF logit, gbmiss, gbmiss x z(fossa feats)] (compact set), C by OOF
  F3_L2_small    : level-2 LR with 6 hand-picked fossa features only
  (+ CNN variants if --cnn oof.csv,val.csv given: column cnn)
usage: gbf_stack.py DATA FOSSA_CSV OUT [--cnn oof_csv,val_csv]"""
import sys, os, json, warnings, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score as auc
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
D, FOS, O = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(O, exist_ok=True)
CNN = sys.argv[sys.argv.index("--cnn") + 1].split(",") if "--cnn" in sys.argv else None
PERI = sys.argv[sys.argv.index("--peri") + 1] if "--peri" in sys.argv else None

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


def LR(C): return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
def GB(lr, leaves, msl, it): return lambda: HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, l2_regularization=1.0, random_state=0)
CANDS = {"LR_C0.1": LR(0.1), "LR_C0.3": LR(0.3), "LR_C1": LR(1.0),
         "GBM_lr.05_l15_m40_300": GB(0.05, 15, 40, 300), "GBM_lr.03_l7_m40_500": GB(0.03, 7, 40, 500),
         "GBM_lr.05_l31_m80_200": GB(0.05, 31, 80, 200), "GBM_lr.02_l15_m60_800": GB(0.02, 15, 60, 800)}
GB_N_U = {"GBM_lr.05_l31_m80_200": CANDS["GBM_lr.05_l31_m80_200"]}
if os.environ.get("QUICK"): CANDS = {"LR_C0.1": LR(0.1)}; GB_N_U = CANDS


def score(m, X): return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]
def _bidx(y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
def boot_delta(y, a, b, n=(50 if os.environ.get("QUICK") else 2000), seed=0, f=None):
    f = f or (lambda yy, s, ix: auc(yy, s)); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        ix = _bidx(y, rng); d.append(f(y[ix], b[ix], ix) - f(y[ix], a[ix], ix))
    d = np.array(d); return dict(mean=float(d.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))
def boot_ci(y, s, n=(50 if os.environ.get("QUICK") else 2000), seed=0):
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


# ---------------- data (train/val only, as sl_stack.py) ----------------
L = pd.read_csv(f"{D}/gallbladder_sludge/labels_trval_sludge.csv"); assert set(L.split) <= {"train", "val"}
L = L[["study_id", "split", "y_gs"]]
age = pd.read_csv(f"{D}/analysis_absent_gb/labels_trval.csv")[["study_id", "Age"]] if "Age" in open(f"{D}/analysis_absent_gb/labels_trval.csv").readline() else None
hu = pd.concat([pd.read_csv(f"{D}/analysis_gallstones/{f}") for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
ag = pd.concat([pd.read_csv(f"{D}/analysis_absent_gb/{f}") for f in ("feat_a6.csv", "feat_ct.csv")]).drop_duplicates("study_id")
ag = ag.drop(columns=["gb_vox"]).rename(columns={c: "ag_" + c for c in ag.columns if c != "study_id"})
g1 = pd.read_csv(f"{D}/gallstones_improve/gs1_feat3mm.csv").drop_duplicates("study_id"); G1 = [c for c in g1.columns if c.startswith("g1_")]
df = L.merge(hu, on="study_id", how="left").merge(ag, on="study_id", how="left").merge(g1, on="study_id", how="left")
if age is not None: df = df.merge(age, on="study_id", how="left")
pc = pd.read_csv(f"{D}/analysis_gallstones_v2/per_case_trval.csv", usecols=["study_id", "cat", "err", "y_chole"]).rename(columns={"cat": "v2_cat", "err": "v2_err", "y_chole": "v2_chole"}).drop_duplicates("study_id")
df = df.merge(pc, on="study_id", how="left")
fold = pd.read_csv(f"{D}/analysis_absent_gb/oof_train.csv")[["study_id", "fold", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
chv = pd.read_csv(f"{D}/analysis_absent_gb/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
T = df[df.split == "train"].merge(fold, on="study_id", how="left"); V = df[df.split == "val"].merge(chv, on="study_id", how="left")
COLS = {}
def add(tag, ftr, fv, cols):
    global T, V
    a = pd.read_csv(ftr); b = pd.read_csv(fv)
    if "fold" in a:
        chk = T[["study_id", "fold"]].merge(a[["study_id", "fold"]].rename(columns={"fold": "f2"}), on="study_id"); assert (chk.fold == chk.f2).all(), tag
    T = T.merge(a[["study_id"] + cols], on="study_id", how="left"); V = V.merge(b[["study_id"] + cols], on="study_id", how="left")
    COLS[tag] = cols
add("gs_e3", f"{D}/analysis_gallstones/oof_train.csv", f"{D}/analysis_gallstones/scores_val_descriptive.csv", ["E3_late"])
add("g3b", f"{D}/gallstones_improve/gs3b/gs3b_oof_train.csv", f"{D}/gallstones_improve/gs3b/gs3b_val.csv", ["g3b_livfb"])
add("gbx", f"{D}/gallstones_improve/gs5/gbx_oof_train.csv", f"{D}/gallstones_improve/gs5/gbx_val.csv", ["gbx_r0", "gbx_r1", "gbx_cat"])
add("gbn", f"{D}/region_units/gallstones/stack/gbn_oof_train.csv", f"{D}/region_units/gallstones/stack/gbn_val.csv", ["gbn_U"])

# ---------------- fossa features ----------------
F = pd.read_csv(FOS).drop_duplicates("study_id")
F = F[F.err.isna()] if "err" in F else F
CNT = [c for c in F.columns if any(c.endswith(s) for s in ("_nblob", "_nblob_metal", "_nblob_st", "_nblob_st2", "_nblob_200", "_nblob_400", "_vol_st", "_big_vol", "_mass", "_n130", "_fluid_big", "_fluid_tot"))]
HUC = [c for c in F.columns if any(c.endswith(s) for s in ("_max_st", "_big_mean", "_big_max", "_metal_max", "_big_shell"))]
DST = [c for c in F.columns if any(c.endswith(s) for s in ("_top_dA", "_top_dpv", "_min_dliv"))]
F[CNT + HUC] = F[CNT + HUC].fillna(0); F[DST] = F[DST].fillna(99.0)
for c in ("gs6_max", "gs6_n130", "gs6_n300"): F[c] = F[c].fillna(0)
F["gs6_med"] = F["gs6_med"].fillna(F["gs6_med"].median())
for c in CNT + ["gs6_vox15", "gs6_vox3", "gs6_n130", "gs6_n300", "body_metal_n", "fs50_duo_n130", "fs50_colon_n130"]:
    F[c] = np.log1p(F[c].clip(lower=0))
FOSSA = [c for c in F.columns if c not in ("study_id", "err", "gb3_vox", "fs30_vox", "fs50_vox", "blob_thr") and not c.endswith("_frac_inliver")]
F["fossa_ok"] = 1.0
T = T.merge(F[["study_id", "fossa_ok"] + FOSSA], on="study_id", how="left"); V = V.merge(F[["study_id", "fossa_ok"] + FOSSA], on="study_id", how="left")
for X_ in (T, V):
    miss = X_.gb_vox.fillna(0) < 100
    X_["gbmiss"] = miss.astype(float)
    X_["fossa_ok"] = X_["fossa_ok"].fillna(0)
    X_.loc[~miss, FOSSA] = 0.0          # features defined only for GB-missing rows
COLS["fossa"] = FOSSA + ["fossa_ok"]
def prep_feats(F, drop):
    F = F.drop_duplicates("study_id"); F = F[F.err.isna()] if "err" in F else F
    cnt = [c for c in F.columns if any(c.endswith(s) for s in ("_nblob", "_nblob_metal", "_nblob_st", "_nblob_st2", "_nblob_200", "_nblob_400", "_vol_st", "_big_vol", "_mass", "_n130", "_fluid_big", "_fluid_tot", "_n_thr", "_ext_ml"))]
    huc = [c for c in F.columns if any(c.endswith(s) for s in ("_max_st", "_big_mean", "_big_max", "_metal_max", "_big_shell", "_ext_max"))]
    dst = [c for c in F.columns if any(c.endswith(s) for s in ("_top_dA", "_top_dpv", "_min_dliv"))]
    F[cnt + huc] = F[cnt + huc].fillna(0); F[dst] = F[dst].fillna(99.0)
    for c in cnt: F[c] = np.log1p(F[c].clip(lower=0))
    cols = [c for c in F.columns if c not in drop and not c.endswith(("_frac_inliver", "_vox"))]
    return F, cols
if PERI:
    P, PC = prep_feats(pd.read_csv(PERI), ("study_id", "err", "gb3_vox"))
    P["peri_ok"] = 1.0
    T = T.merge(P[["study_id", "peri_ok"] + PC], on="study_id", how="left"); V = V.merge(P[["study_id", "peri_ok"] + PC], on="study_id", how="left")
    for X_ in (T, V):
        X_["peri_ok"] = X_["peri_ok"].fillna(0); X_.loc[X_.gbmiss == 1, PC] = 0.0
    COLS["peri"] = PC + ["peri_ok"]
    print("peri feats", len(PC), "train rows with peri", int(T.peri_ok.sum()), "val", int(V.peri_ok.sum()), flush=True)
SMALL = ["fs30_nblob_200", "fs30_max_st", "fs30_mass", "fs50_mass", "gs6_vox15", "gs6_max"]
COLS["fossa_small"] = SMALL
if CNN:
    a = pd.read_csv(CNN[0]); b = pd.read_csv(CNN[1])
    T = T.merge(a[["study_id", "cnn"]], on="study_id", how="left"); V = V.merge(b[["study_id", "cnn"]], on="study_id", how="left")
    for X_ in (T, V): X_.loc[X_.gbmiss == 0, "cnn"] = 0.0
    COLS["cnn"] = ["cnn"]
print("fossa feats", len(FOSSA), "train missing rows with feats", int(((T.gbmiss == 1) & (T.fossa_ok == 1)).sum()), "/", int((T.gbmiss == 1).sum()),
      "val", int(((V.gbmiss == 1) & (V.fossa_ok == 1)).sum()), "/", int((V.gbmiss == 1).sum()), flush=True)

_ord = pd.read_csv(f"{D}/analysis_gallstones/labels_trval.csv")["study_id"].reset_index().set_index("study_id")["index"]
Tg = T[T.y_gs.notna()].assign(_o=lambda d: d.study_id.map(_ord)).sort_values("_o").reset_index(drop=True)
Vg = V[V.y_gs.notna()].assign(_o=lambda d: d.study_id.map(_ord)).sort_values("_o").reset_index(drop=True)
assert Tg.study_id.is_unique and Vg.study_id.is_unique, "duplicate rows"
y, yv = Tg.y_gs.values.astype(int), Vg.y_gs.values.astype(int)
AGE_OK = "Age" in Tg
aget = Tg.Age.values if AGE_OK else None; agev = Vg.Age.values if AGE_OK else None
NU = ["hu", "ag", "ch", "g1", "gbx", "g3b", "gbn"]


def run_variant(Tp, Vp, ycol, sets, e3, cands=CANDS):
    yy = Tp[ycol].values.astype(int); cv = {}; so = {}
    for cn, mk in cands.items():
        s = np.zeros(len(Tp))
        for k in sorted(Tp.fold.unique()):
            tr, te = Tp.fold.values != k, Tp.fold.values == k
            Xtr, med = design(Tp[tr], sets, e3); Xte, _ = design(Tp[te], sets, e3, med)
            m = mk().fit(Xtr, yy[tr]); s[te] = score(m, Xte)
        so[cn] = s; cv[cn] = float(auc(yy, s)); print("  cand", cn, round(cv[cn], 4), flush=True)
    best = max(cv, key=cv.get)
    X, med = design(Tp, sets, e3); m = cands[best]().fit(X, yy); Xv, _ = design(Vp, sets, e3, med)
    return so[best], score(m, Xv), best, cv, dict(model=m, medians=med, columns=list(X.columns), sets=sets, cand=best)


if os.environ.get("SEEDCHK"):
    out = {}
    for vn, sets, (lr, lv, msl, it) in (("N_U", NU, (0.05, 31, 80, 200)), ("P2", NU + ["fossa", "peri"], (0.05, 15, 40, 300)),
                                        ("P1", NU + ["peri"], (0.02, 15, 60, 800)), ("F1", NU + ["fossa"], (0.02, 15, 60, 800))):
        X, med = design(Tg, sets, "E3_late"); Xv, _ = design(Vg, sets, "E3_late", med); S = []
        for seed in range(5):
            m = HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=lv, min_samples_leaf=msl, l2_regularization=1.0, random_state=seed).fit(X, y)
            S.append(m.decision_function(Xv))
        out[vn] = dict(per_seed=[float(auc(yv, s)) for s in S], ens=float(auc(yv, np.mean(S, 0))))
        print(vn, out[vn], flush=True)
    json.dump(out, open(f"{O}/seedcheck_val.json", "w"), indent=1); sys.exit(0)
B = {}
so_ref, sv_ref, best, cv, mdl = run_variant(Tg, Vg, "y_gs", NU, "E3_late", cands=GB_N_U)
st = pd.read_csv(f"{D}/region_units/gallstones/stack/oof_train.csv")[["study_id", "N_U"]]
sv_st = pd.read_csv(f"{D}/region_units/gallstones/stack/val_scores.csv")
mm = Tg[["study_id"]].assign(s=so_ref).merge(st, on="study_id")
RES = {"N_U_repro_corr_oof": float(np.corrcoef(mm.s, mm.N_U)[0, 1])}
if "N_U" in sv_st:
    mv = Vg[["study_id"]].assign(s=sv_ref).merge(sv_st[["study_id", "N_U"]], on="study_id"); RES["N_U_repro_corr_val"] = float(np.corrcoef(mv.s, mv.N_U)[0, 1])
print("repro", RES, "val auc", auc(yv, sv_ref), flush=True)
B["N_U"] = (so_ref, sv_ref, best, cv, mdl)
if os.environ.get("REFONLY"): sys.exit(0)
if not os.environ.get("SKIPF"):
    B["F1_NU+fossa"] = run_variant(Tg, Vg, "y_gs", NU + ["fossa"], "E3_late")
    B["F1b_NU+fossa_fixedGBM"] = run_variant(Tg, Vg, "y_gs", NU + ["fossa"], "E3_late", cands=GB_N_U)
    B["F1s_NU+fossa_small"] = run_variant(Tg, Vg, "y_gs", NU + ["fossa_small"], "E3_late")
if PERI:
    B["P1_NU+peri"] = run_variant(Tg, Vg, "y_gs", NU + ["peri"], "E3_late")
    B["P2_NU+fossa+peri"] = run_variant(Tg, Vg, "y_gs", NU + ["fossa", "peri"], "E3_late")
    B["P2b_NU+fossa+peri_fixedGBM"] = run_variant(Tg, Vg, "y_gs", NU + ["fossa", "peri"], "E3_late", cands=GB_N_U)
if CNN:
    B["F4_NU+fossa+cnn"] = run_variant(Tg, Vg, "y_gs", NU + ["fossa", "cnn"], "E3_late")
    B["F4s_NU+cnn"] = run_variant(Tg, Vg, "y_gs", NU + ["cnn"], "E3_late")


# level-2 LR: N_U OOF logit + gbmiss + gbmiss x fossa feats (z-scored on train GB-missing rows)
def l2(cols, tag):
    mu = Tg.loc[Tg.gbmiss == 1, cols].mean(); sd = Tg.loc[Tg.gbmiss == 1, cols].std().replace(0, 1)
    def mk(d, s):
        X = pd.DataFrame({"g": s, "miss": d.gbmiss.values, "g_x_miss": s * d.gbmiss.values})
        Z = ((d[cols] - mu) / sd).fillna(0).values * d.gbmiss.values[:, None]
        for i, c in enumerate(cols): X["m_" + c] = Z[:, i]
        return X
    X, Xv = mk(Tg, so_ref), mk(Vg, sv_ref); cv = {}; so = {}
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        s = np.zeros(len(Tg))
        for k in sorted(Tg.fold.unique()):
            tr, te = Tg.fold.values != k, Tg.fold.values == k
            m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr]); s[te] = m.decision_function(X[te])
        so[C] = s; cv[f"C{C}"] = float(auc(y, s))
    bC = max(so, key=lambda c: cv[f"C{c}"]); m = LogisticRegression(C=bC, max_iter=5000).fit(X, y)
    return so[bC], m.decision_function(Xv), f"L2_LR_C{bC}", cv, dict(model=m, cols=cols, mu=mu, sd=sd)
B["F2_L2_fossa_all"] = l2(FOSSA, "all")
B["F3_L2_fossa_small"] = l2(SMALL, "small")
if CNN: B["F5_L2_small+cnn"] = l2(SMALL + ["cnn"], "cnn")

# ---------------- metrics ----------------
miss_t, miss_v = Tg.gbmiss.values == 1, Vg.gbmiss.values == 1
OOF = Tg[["study_id", "fold", "y_gs", "gbmiss"]].copy(); VAL = Vg[["study_id", "y_gs", "gbmiss"]].copy()
for vn, (so, sv, best, cv, mdl) in B.items():
    thr = float(np.quantile(so[y == 0], .9))
    r = dict(cand=best, cv=cv, oof=float(auc(y, so)), val=float(auc(yv, sv)), val_ci=boot_ci(yv, sv),
             val_sens90=sens90(yv, sv), val_sens90_trthr=sens90(yv, sv, thr), oof_sens90=sens90(y, so))
    if AGE_OK:
        r.update(oof_age=age_matched_auc(y, so, aget), val_age=age_matched_auc(yv, sv, agev), spearman_neg_val=spearman_neg(yv, sv, agev))
    # GB-missing subgroup: subgroup positives vs ALL negatives (v2 table definition) and within-subgroup
    kt = (y == 0) | miss_t; kv = (yv == 0) | miss_v
    r["sub_gbmiss_vs_allneg"] = dict(oof=float(auc(y[kt], so[kt])), val=float(auc(yv[kv], sv[kv])), val_ci=boot_ci(yv[kv], sv[kv]),
                                     n_pos_val=int(yv[miss_v].sum()), n_pos_train=int(y[miss_t].sum()))
    r["sub_gbmiss_within"] = dict(oof=float(auc(y[miss_t], so[miss_t])), val=float(auc(yv[miss_v], sv[miss_v])), n_val=int(miss_v.sum()))
    r["sub_gbpresent_vs_allneg"] = dict(oof=float(auc(y[(y == 0) | ~miss_t], so[(y == 0) | ~miss_t])), val=float(auc(yv[(yv == 0) | ~miss_v], sv[(yv == 0) | ~miss_v])))
    r["val_sens90_trthr_gbmiss"] = float((sv[miss_v & (yv == 1)] > thr).mean())
    for cname, pre in (("E", "E "), ("F", "F "), ("C", "C ")):
        kE = Vg.v2_cat.fillna("").str.startswith(pre).values & (yv == 1); kEt = Tg.v2_cat.fillna("").str.startswith(pre).values & (y == 1)
        r[f"rescued_{cname}FN_trthr"] = dict(val=f"{int((sv[kE] > thr).sum())}/{int(kE.sum())}", train=f"{int((so[kEt] > thr).sum())}/{int(kEt.sum())}")
    if vn != "N_U":
        r["d_oof_vs_NU"] = boot_delta(y, so_ref, so); r["d_val_vs_NU"] = boot_delta(yv, sv_ref, sv)
        r["d_val_sub_gbmiss_vs_NU"] = boot_delta(yv[kv], sv_ref[kv], sv[kv])
        if AGE_OK: r["d_val_age_vs_NU"] = boot_delta(yv, sv_ref, sv, n=500, f=lambda yy, s, ix: age_matched_auc(yy, s, agev[ix]))
    RES[vn] = r; OOF[vn] = so; VAL[vn] = sv
    joblib.dump(mdl, f"{O}/model_{vn}.joblib")
    print("GS", vn, json.dumps({k: v for k, v in r.items() if k != "cv"}, default=float), flush=True)
RES["selected_by_train_oof"] = max(B, key=lambda v: RES[v]["oof"])
print("SELECTED", RES["selected_by_train_oof"], flush=True)
OOF.to_csv(f"{O}/oof_train.csv", index=False); VAL.to_csv(f"{O}/val_scores.csv", index=False)
json.dump(RES, open(f"{O}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print("GBF_STACK_DONE", flush=True)
