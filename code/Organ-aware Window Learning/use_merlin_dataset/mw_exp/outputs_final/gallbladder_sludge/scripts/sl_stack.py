"""膽泥 model + gallstone sludge-suppression variants (train 5-fold OOF selection only; val reported once; test never read).
Machinery copied from region_units/gallstones/scripts/nu_gs_stack.py (feature definitions, 7 LR/GBM candidates, GS1 folds,
paired stratified bootstrap 2,000, age-matched AUROC, Spearman on negatives).

Part A  膽泥 (y_sludge, all train/val rows) and 膽泥-only (y_sludge_only, gallstone-known rows):
  S_head : sludge E3_late GB head alone (h_base)
  S_img  : hu+ag+ch+g1 + h_base
  S_G5c  : S_img + h_gbx_* + h_livfb                       (G5c analog)
  S_NU   : S_G5c + h_U                                     (Exp N analog = same feature sets as the gallstone best)
  S_NUgs : S_NU + gallstone E3_late + gallstone g3b_livfb (full train coverage)
Part B  gallstones under the UNCHANGED official label (y_gs):
  N_U (REF, Exp N rebuilt) | c1 N_U + sludge heads (both targets) | (i) 3-class multinomial {neither, sludge-only, stones}
  score = P(stones) | (ii) level-2 stack of N_U OOF logit + sludge-only logit + 膽泥 logit + interactions (LR/GBM)
  | (iii) z(N_U) - w*z(sludge-only), w tuned on train OOF.
usage: sl_stack.py DATA HEADS OUT"""
import sys, os, json, warnings, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score as auc
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
D, H, O = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(O, exist_ok=True)
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


# ================= Part A: 膽泥 =================
BASE = ["hu", "ag", "ch", "g1"]
for tg, short in (("y_sludge", "sl"), ("y_sludge_only", "so")):
    Tp = T[T[tg].notna()].reset_index(drop=True); Vp = V[V[tg].notna()].reset_index(drop=True)
    y, yv = Tp[tg].values.astype(int), Vp[tg].values.astype(int)
    VAR = {"S_head": ([], f"{short}_base"), "S_img": (BASE, f"{short}_base"),
           "S_G5c": (BASE + [f"{short}_gbx", f"{short}_livfb"], f"{short}_base"),
           "S_NU": (BASE + [f"{short}_gbx", f"{short}_livfb", f"{short}_U"], f"{short}_base"),
           "S_NUgs": (BASE + [f"{short}_gbx", f"{short}_livfb", f"{short}_U", "g3b"], f"{short}_base")}
    out = {"n_train": len(Tp), "pos_train": int(y.sum()), "n_val": len(Vp), "pos_val": int(yv.sum()), "variants": {}}
    OOF = Tp[["study_id", "fold", tg]].copy(); VAL = Vp[["study_id", tg]].copy()
    for vn, (sets, e3) in VAR.items():
        if vn == "S_NUgs":  # gallstone E3_late as an extra column
            COLS["gs_e3x"] = ["E3_late"]; sets = sets + ["gs_e3x"]
        if vn == "S_head":
            so, sv = Tp[e3].values, Vp[e3].values; best, cv, mdl = "head", {}, None
        else:
            so, sv, best, cv, mdl = run_variant(Tp, Vp, tg, sets, e3)
        r = metrics(y, so, yv, sv, Tp.get("Age", pd.Series(np.nan, index=Tp.index)).values, Vp.get("Age", pd.Series(np.nan, index=Vp.index)).values)
        r.update(sets=sets, cand=best, cv=cv)
        # descriptive sub-AUROCs (val): sludge-only vs non-sludge, sludge+stones vs non-sludge
        for g in ("sludge_only", "sludge_stones", "sludge_gs_unclear"):
            k = (Vp.sl_group.values == g) | (yv == 0)
            if (Vp.sl_group.values == g).sum() >= 5 and g in set(Vp.sl_group): r[f"val_sub_{g}"] = float(auc(yv[k], sv[k])) if tg == "y_sludge" else None
        if tg == "y_sludge":
            gsneg = Vp.y_gs.values == 0
            r["val_auc_within_gallstone_negatives"] = float(auc(yv[gsneg], sv[gsneg]))
            r["val_auc_certain_only"] = float(auc(yv[(Vp.sl_unc.values == 0) | (yv == 0)], sv[(Vp.sl_unc.values == 0) | (yv == 0)]))
        out["variants"][vn] = r; OOF[vn] = so; VAL[vn] = sv
        if mdl: joblib.dump(mdl | {"thr_spec90_trainoof": r["thr_trainoof_spec90"]}, f"{O}/model_{short}_{vn}.joblib")
        print(tg, vn, json.dumps({k: v for k, v in r.items() if k not in ("cv", "sets")}, default=float), flush=True)
    sel = max([v for v in out["variants"] if v != "S_head"], key=lambda v: out["variants"][v]["oof"])
    out["selected_by_train_oof"] = sel
    ref = out["variants"]["S_head"]
    for vn in out["variants"]:
        if vn != "S_head":
            out["variants"][vn]["d_val_vs_head"] = boot_delta(yv, VAL["S_head"].values, VAL[vn].values)
    print(tg, "selected", sel, flush=True)
    RES["partA"][tg] = out; OOF.to_csv(f"{O}/oof_{short}.csv", index=False); VAL.to_csv(f"{O}/val_{short}.csv", index=False)
    T[f"{short}_stack"] = T.study_id.map(OOF.set_index("study_id")[sel]); V[f"{short}_stack"] = V.study_id.map(VAL.set_index("study_id")[sel])
    COLS[f"{short}_stack"] = [f"{short}_stack"]
json.dump(RES, open(f"{O}/results.json", "w"), ensure_ascii=False, indent=1, default=float)

# ================= Part B: gallstones (official y_gs) =================
# row order = analysis_gallstones/labels_trval.csv order (as nu_gs_stack.py): HistGradientBoosting's automatic early stopping
# (n > 10,000) draws its 10% holdout by row position, so the row order must match to reproduce Exp N exactly.
_ord = pd.read_csv(f"{D}/analysis_gallstones/labels_trval.csv")["study_id"].reset_index().set_index("study_id")["index"]
Tg = T[T.y_gs.notna()].assign(_o=lambda d: d.study_id.map(_ord)).sort_values("_o").reset_index(drop=True)
Vg = V[V.y_gs.notna()].assign(_o=lambda d: d.study_id.map(_ord)).sort_values("_o").reset_index(drop=True)
y, yv = Tg.y_gs.values.astype(int), Vg.y_gs.values.astype(int)
aget = Tg.Age.values if AGE_OK else None; agev = Vg.Age.values if AGE_OK else None
NU = ["hu", "ag", "ch", "g1", "gbx", "g3b", "gbn"]
SLH = ["sl_e3", "sl_livfb", "sl_gbx", "sl_U", "so_e3", "so_livfb", "so_gbx", "so_U"]
GB_N_U = {"GBM_lr.05_l31_m80_200": CANDS["GBM_lr.05_l31_m80_200"]}
B = {}
so_ref, sv_ref, best, cv, mdl = run_variant(Tg, Vg, "y_gs", NU, "E3_late", cands=GB_N_U)   # Exp N exact (its selected candidate)
st = pd.read_csv(f"{D}/region_units/gallstones/stack/oof_train.csv")[["study_id", "N_U"]]
RES["partB"]["N_U_repro_corr"] = float(np.corrcoef(Tg.merge(st, on="study_id").N_U, pd.Series(so_ref, index=Tg.study_id).reindex(Tg.merge(st, on="study_id").study_id).values)[0, 1])
B["N_U"] = (so_ref, sv_ref, best, cv, mdl)
print("N_U repro corr", RES["partB"]["N_U_repro_corr"], flush=True)
# c1: multi-input (sludge heads + stacked sludge scores as extra inputs)
B["c1_NU+sludge_heads"] = run_variant(Tg, Vg, "y_gs", NU + SLH, "E3_late")
B["c1b_NU+sludge_stack"] = run_variant(Tg, Vg, "y_gs", NU + ["sl_stack", "so_stack"], "E3_late")
# (i) 3-class multinomial, gallstone score = logit P(stones)
Tg["y3i"] = Tg.y3.astype(int); Vg["y3i"] = Vg.y3.astype(int)
B["i_3class_NU+sludge_heads"] = run_variant(Tg, Vg, "y3i", NU + SLH, "E3_late", multi_cls=2)
B["i_3class_NU"] = run_variant(Tg, Vg, "y3i", NU, "E3_late", multi_cls=2)
# (ii) level-2 stack on OOF logits: N_U + sludge-only stack + 膽泥 stack + interactions
L2 = pd.DataFrame({"g": so_ref, "so": Tg.so_stack.values, "sl": Tg.sl_stack.values}); L2v = pd.DataFrame({"g": sv_ref, "so": Vg.so_stack.values, "sl": Vg.sl_stack.values})
for X_ in (L2, L2v):
    X_["g_x_so"] = X_.g * X_.so; X_["g_x_sl"] = X_.g * X_.sl
def l2run(cols):
    cv = {}; so = {}
    for cn, mk in CANDS.items():
        s = np.zeros(len(Tg))
        for k in sorted(Tg.fold.unique()):
            tr, te = Tg.fold.values != k, Tg.fold.values == k
            m = mk().fit(L2[cols][tr], y[tr]); s[te] = score(m, L2[cols][te])
        so[cn] = s; cv[cn] = float(auc(y, s))
    bst = max(cv, key=cv.get); m = CANDS[bst]().fit(L2[cols], y)
    return so[bst], score(m, L2v[cols]), bst, cv, dict(model=m, columns=cols, cand=bst)
B["ii_L2_g+so+inter"] = l2run(["g", "so", "g_x_so"])
B["ii_L2_g+so+sl+inter"] = l2run(["g", "so", "sl", "g_x_so", "g_x_sl"])
# (iii) z(N_U) - w*z(sludge-only stack), w on train OOF
zg = lambda s: (s - so_ref.mean()) / so_ref.std(); sos = Tg.so_stack.values; zs = lambda s: (s - sos.mean()) / sos.std()
grid = {float(w): float(auc(y, zg(so_ref) - w * zs(sos))) for w in np.round(np.arange(0, 1.01, 0.05), 2)}
wb = max(grid, key=grid.get)
B["iii_adjust_w"] = (zg(so_ref) - wb * zs(sos), zg(sv_ref) - wb * zs(Vg.so_stack.values), f"w={wb}", grid, dict(w=wb, z_g=[float(so_ref.mean()), float(so_ref.std())], z_so=[float(sos.mean()), float(sos.std())]))
# (iii-b) same with the head-only sludge-only logit (no stacking)
soh = Tg.so_base.values; zh = lambda s: (s - soh.mean()) / soh.std()
grid2 = {float(w): float(auc(y, zg(so_ref) - w * zh(soh))) for w in np.round(np.arange(0, 1.01, 0.05), 2)}; wb2 = max(grid2, key=grid2.get)
B["iii_adjust_w_head"] = (zg(so_ref) - wb2 * zh(soh), zg(sv_ref) - wb2 * zh(Vg.so_base.values), f"w={wb2}", grid2, dict(w=wb2))

so_only_v = (Vg.sl_group.values == "sludge_only"); so_only_t = (Tg.sl_group.values == "sludge_only")
old_v = (Vg.old_sludge.values == "僅膽泥") & (yv == 0); old_t = (Tg.old_sludge.values == "僅膽泥") & (y == 0)
oth_v = (yv == 0) & ~so_only_v & ~old_v; oth_t = (y == 0) & ~so_only_t & ~old_t
yx = Vg.y_gs_excl_sludgeonly.values; kx = ~np.isnan(yx); yxt = Tg.y_gs_excl_sludgeonly.values; kxt = ~np.isnan(yxt)
OOF = Tg[["study_id", "fold", "y_gs", "sl_group"]].copy(); VAL = Vg[["study_id", "y_gs", "sl_group"]].copy()
for vn, (so, sv, best, cv, mdl) in B.items():
    r = metrics(y, so, yv, sv, aget, agev); r.update(cand=best, cv=cv)
    thr = r["thr_trainoof_spec90"]; thv = float(np.quantile(sv[yv == 0], .9))
    r["fp_trthr"] = dict(val_sludge_only=float((sv[so_only_v] > thr).mean()), val_old_sludge_only=float((sv[old_v] > thr).mean()),
                         val_other_neg=float((sv[oth_v] > thr).mean()), oof_sludge_only=float((so[so_only_t] > thr).mean()),
                         oof_old_sludge_only=float((so[old_t] > thr).mean()), oof_other_neg=float((so[oth_t] > thr).mean()))
    r["fp_valthr"] = dict(val_sludge_only=float((sv[so_only_v] > thv).mean()), val_old_sludge_only=float((sv[old_v] > thv).mean()), val_other_neg=float((sv[oth_v] > thv).mean()))
    r["n_fp_groups"] = dict(val_sludge_only=int(so_only_v.sum()), val_old=int(old_v.sum()), train_sludge_only=int(so_only_t.sum()), train_old=int(old_t.sum()))
    r["secondary_excl_sludgeonly"] = dict(oof=float(auc(yxt[kxt], so[kxt])), val=float(auc(yx[kx], sv[kx])), val_ci=boot_ci(yx[kx].astype(int), sv[kx]))
    r["auc_sludge_only_vs_stones_val"] = float(auc(np.r_[np.ones((yv == 1).sum()), np.zeros(so_only_v.sum())], np.r_[sv[yv == 1], sv[so_only_v]]))
    if vn != "N_U":
        r["d_oof_vs_NU"] = boot_delta(y, so_ref, so); r["d_val_vs_NU"] = boot_delta(yv, sv_ref, sv)
        if AGE_OK: r["d_val_age_vs_NU"] = boot_delta(yv, sv_ref, sv, n=500, f=lambda yy, s, ix: age_matched_auc(yy, s, agev[ix]))
    RES["partB"][vn] = r; OOF[vn] = so; VAL[vn] = sv
    if mdl: joblib.dump(mdl, f"{O}/gs_{vn}.joblib")
    print("GS", vn, json.dumps({k: v for k, v in r.items() if k not in ("cv",)}, default=float), flush=True)
RES["partB"]["selected_by_train_oof"] = max([v for v in B], key=lambda v: RES["partB"][v]["oof"])
print("GS selected", RES["partB"]["selected_by_train_oof"], flush=True)
OOF.to_csv(f"{O}/gs_oof_train.csv", index=False); VAL.to_csv(f"{O}/gs_val_scores.csv", index=False)
json.dump(RES, open(f"{O}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print("SL_STACK_DONE", flush=True)
