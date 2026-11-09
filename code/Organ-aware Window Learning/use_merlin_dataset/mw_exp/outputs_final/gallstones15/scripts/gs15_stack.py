"""gs15 stack: 1.5 mm GB crop unit on top of Exp N (official 膽結石 best N_U, val .881). Train 5-fold OOF selection only (GS1
folds); val reported once; test never read (all inputs are train/val files, labels filtered by split first).
Data loading / feature definitions / 7 LR-GBM candidates / bootstrap copied from gallbladder_sludge/scripts/sl_stack.py
(Part B), which reproduces Exp N N_U exactly (row order = analysis_gallstones/labels_trval.csv order, see the note there:
HistGradientBoosting auto early stopping draws its holdout by row position when n > 10,000).
New inputs: gs15 probe heads (out_<mode>/oof_train.csv, val_scores.csv: U, U_org, U_gb3), 1.5 mm GB HU features
(feats.csv, f15_*), optional 3D-CNN OOF (cnn_<mode>/oof_train.csv, val_scores.csv: cnn).
Variants
  REF_off : official Exp N N_U scores (region_units/gallstones/stack) -- the .881 reference for all paired deltas
  REF_rb  : N_U rebuilt on the same train rows (fixed candidate, as Exp N)
  NU+h15, NU+f15, NU+h15+f15, [NU+cnn, NU+h15+f15+cnn] : full re-stack (7 candidates, best by train OOF)
  L2_h15, L2_h15f, [L2_h15cnn] : level-2 LR/GBM on [official N_U OOF logit + new heads (+ a few HU features)]
mode tier0: train rows = tier0 train ids with an embedding (all positives + age/sex-matched negatives 1:2); val = all val.
mode full : all train rows (rows without an embedding get train-median imputation; their count is reported).
Selection: highest train OOF among the new variants (same train rows as REF). Report: ROOT/report.md (繁體中文).
usage: gs15_stack.py D(=outputs_final) ROOT MODE"""
import sys, os, json, warnings, time
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score as auc
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
D, RT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
SUF = "_tier0" if MODE == "tier0" else ""
O = f"{RT}/stack{SUF}"; os.makedirs(O, exist_ok=True)
NB = int(os.environ.get("NBOOT", "2000"))

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
        if c.startswith("f15_") and ("_ml" in c or "blob" in c):
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
    if e3: X[e3] = d[e3].astype(float)
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
LRC = {k: v for k, v in CANDS.items() if k.startswith("LR")}
def score(m, X): return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


def _bidx(y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
def boot_delta(y, a, b, n=NB, seed=0, f=None):
    f = f or (lambda yy, s, ix: auc(yy, s)); rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        ix = _bidx(y, rng); d.append(f(y[ix], b[ix], ix) - f(y[ix], a[ix], ix))
    d = np.array(d); return dict(mean=float(d.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_le0=float((d <= 0).mean()))
def boot_ci(y, s, n=NB, seed=0):
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


# ---------------- data (train/val only; split filtered before labels are used) ----------------
L = pd.read_csv(f"{D}/gallbladder_sludge/labels_trval_sludge.csv"); assert set(L.split) <= {"train", "val"}
L = L[L.split.isin(["train", "val"])][["study_id", "split", "y_gs"]]
age = pd.read_csv(f"{D}/analysis_absent_gb/labels_trval.csv")[["study_id", "split", "Age"]]
age = age[age.split.isin(["train", "val"])][["study_id", "Age"]]
hu = pd.concat([pd.read_csv(f"{D}/analysis_gallstones/{f}") for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
ag = pd.concat([pd.read_csv(f"{D}/analysis_absent_gb/{f}") for f in ("feat_a6.csv", "feat_ct.csv")]).drop_duplicates("study_id")
ag = ag.drop(columns=["gb_vox"]).rename(columns={c: "ag_" + c for c in ag.columns if c != "study_id"})
g1 = pd.read_csv(f"{D}/gallstones_improve/gs1_feat3mm.csv").drop_duplicates("study_id"); G1 = [c for c in g1.columns if c.startswith("g1_")]
fl = pd.read_csv(f"{D}/analysis_gallstones/subgroups/case_flags_trval.csv")[["study_id", "gb_missing", "img_vis"]]
df = L.merge(hu, on="study_id", how="left").merge(ag, on="study_id", how="left").merge(g1, on="study_id", how="left") \
      .merge(fl, on="study_id", how="left").merge(age, on="study_id", how="left")
fold = pd.read_csv(f"{D}/analysis_absent_gb/oof_train.csv")[["study_id", "fold", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
chv = pd.read_csv(f"{D}/analysis_absent_gb/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "chole_logit"})
T = df[df.split == "train"].merge(fold, on="study_id", how="left"); V = df[df.split == "val"].merge(chv, on="study_id", how="left")
COLS = {}
def add(tag, ftr, fv, cols, chkfold=True):
    global T, V
    a = pd.read_csv(ftr); b = pd.read_csv(fv)
    if chkfold and "fold" in a:
        chk = T[["study_id", "fold"]].merge(a[["study_id", "fold"]].rename(columns={"fold": "f2"}), on="study_id"); assert (chk.fold == chk.f2).all(), tag
    T = T.merge(a[["study_id"] + cols], on="study_id", how="left"); V = V.merge(b[["study_id"] + cols], on="study_id", how="left")
    COLS[tag] = cols; print(tag, cols, "missing tr/val", int(T[cols[0]].isna().sum()), int(V[cols[0]].isna().sum()), flush=True)
add("gs_e3", f"{D}/analysis_gallstones/oof_train.csv", f"{D}/analysis_gallstones/scores_val_descriptive.csv", ["E3_late"])
add("g3b", f"{D}/gallstones_improve/gs3b/gs3b_oof_train.csv", f"{D}/gallstones_improve/gs3b/gs3b_val.csv", ["g3b_livfb"])
add("gbx", f"{D}/gallstones_improve/gs5/gbx_oof_train.csv", f"{D}/gallstones_improve/gs5/gbx_val.csv", ["gbx_r0", "gbx_r1", "gbx_cat"])
add("gbn", f"{D}/region_units/gallstones/stack/gbn_oof_train.csv", f"{D}/region_units/gallstones/stack/gbn_val.csv", ["gbn_U"])
# official Exp N scores (the .881 reference)
off_t = pd.read_csv(f"{D}/region_units/gallstones/stack/oof_train.csv")[["study_id", "N_U"]].rename(columns={"N_U": "nu_off"})
off_v = pd.read_csv(f"{D}/region_units/gallstones/stack/val_scores.csv")[["study_id", "N_U"]].rename(columns={"N_U": "nu_off"})
T = T.merge(off_t, on="study_id", how="left"); V = V.merge(off_v, on="study_id", how="left"); COLS["nu_off"] = ["nu_off"]
# gs15 heads
H = f"{RT}/out{SUF}"
hd = pd.read_csv(f"{H}/oof_train.csv"); hv = pd.read_csv(f"{H}/val_scores.csv")
HC = [c for c in ("U", "U_org", "U_gb3") if c in hd.columns]
hd = hd.rename(columns={c: "h15_" + c for c in HC}); hv = hv.rename(columns={c: "h15_" + c for c in HC})
chk = T[["study_id", "fold"]].merge(hd[["study_id", "fold"]].rename(columns={"fold": "f2"}), on="study_id"); assert (chk.fold == chk.f2).all(), "h15 folds"
T = T.merge(hd[["study_id"] + ["h15_" + c for c in HC]], on="study_id", how="left"); V = V.merge(hv[["study_id"] + ["h15_" + c for c in HC]], on="study_id", how="left")
COLS["h15"] = ["h15_" + c for c in HC]
# HU features
F15 = pd.read_csv(f"{RT}/feats.csv")
FC = [c for c in F15.columns if c.startswith("f15_")]
F15["m15_gb"] = F15.method15.astype(str).str.endswith("gb").astype(float)
T = T.merge(F15[["study_id"] + FC + ["m15_gb"]], on="study_id", how="left"); V = V.merge(F15[["study_id"] + FC + ["m15_gb"]], on="study_id", how="left")
COLS["f15"] = FC + ["m15_gb"]
F15S = [c for c in ("f15_gb_max", "f15_gb_p99", "f15_gb_ml100", "f15_gb_ml130", "f15_gb_maxblob100", "f15_gb_ml_chol", "f15_dep_minus_nondep",
                    "f15_fos_ml130", "f15_fos_max", "f15_gb_ml") if c in FC]
COLS["f15s"] = F15S
CNN = f"{RT}/cnn_tier0" if MODE == "tier0" else f"{RT}/cnn_full"
HAS_CNN = os.path.exists(f"{CNN}/oof_train.csv") and os.path.exists(f"{CNN}/val_scores.csv")
if HAS_CNN:
    a = pd.read_csv(f"{CNN}/oof_train.csv"); b = pd.read_csv(f"{CNN}/val_scores.csv")
    T = T.merge(a[["study_id", "cnn"]], on="study_id", how="left"); V = V.merge(b[["study_id", "cnn"]], on="study_id", how="left"); COLS["cnn"] = ["cnn"]

# row order = analysis_gallstones/labels_trval.csv (reproduces Exp N exactly on the full train set)
_ord = pd.read_csv(f"{D}/analysis_gallstones/labels_trval.csv")["study_id"].reset_index().set_index("study_id")["index"]
T = T[T.y_gs.notna()].assign(_o=lambda d: d.study_id.map(_ord)).sort_values("_o").reset_index(drop=True)
V = V[V.y_gs.notna()].assign(_o=lambda d: d.study_id.map(_ord)).sort_values("_o").reset_index(drop=True)
n_tr_all = len(T); emb_have = T.h15_U.notna()
if MODE == "tier0":
    T = T[emb_have].reset_index(drop=True)
y, yv = T.y_gs.values.astype(int), V.y_gs.values.astype(int)
aget, agev = T.Age.values, V.Age.values
print(f"mode {MODE}: train rows {len(T)} (pos {y.sum()}) of {n_tr_all}; val {len(V)} (pos {yv.sum()}); val missing h15 {int(V.h15_U.isna().sum())};"
      f" train missing h15 {int(T.h15_U.isna().sum())}; cnn {HAS_CNN}", flush=True)


def run_variant(sets, e3="E3_late", cands=CANDS):
    def oof(mk):
        s = np.zeros(len(T))
        for k in sorted(T.fold.unique()):
            tr, te = T.fold.values != k, T.fold.values == k
            Xtr, med = design(T[tr], sets, e3); Xte, _ = design(T[te], sets, e3, med)
            m = mk().fit(Xtr, y[tr]); s[te] = score(m, Xte)
        return s
    cv, so = {}, {}
    for cn, mk in cands.items():
        so[cn] = oof(mk); cv[cn] = float(auc(y, so[cn]))
    best = max(cv, key=cv.get)
    X, med = design(T, sets, e3); m = cands[best]().fit(X, y); Xv, _ = design(V, sets, e3, med)
    return so[best], score(m, Xv), best, cv, dict(model=m, medians=med, columns=list(X.columns), sets=sets, cand=best)


NU = ["hu", "ag", "ch", "g1", "gbx", "g3b", "gbn"]
B = {}
B["REF_off"] = (T.nu_off.values, V.nu_off.values, "official", {}, None)
B["REF_rb"] = run_variant(NU, cands={"GBM_lr.05_l31_m80_200": CANDS["GBM_lr.05_l31_m80_200"]})
if MODE == "full":
    print("REF_rb corr with official OOF", float(np.corrcoef(B["REF_rb"][0], T.nu_off.values)[0, 1]),
          "val corr", float(np.corrcoef(B["REF_rb"][1], V.nu_off.values)[0, 1]), flush=True)
VARS = {"NU+h15": NU + ["h15"], "NU+f15": NU + ["f15"], "NU+h15+f15": NU + ["h15", "f15"]}
L2 = {"L2_h15": (["nu_off", "h15"], LRC), "L2_h15f": (["nu_off", "h15", "f15s"], CANDS)}
if HAS_CNN:
    VARS["NU+cnn"] = NU + ["cnn"]; VARS["NU+h15+f15+cnn"] = NU + ["h15", "f15", "cnn"]
    L2["L2_h15cnn"] = (["nu_off", "h15", "cnn"], LRC); L2["L2_h15fcnn"] = (["nu_off", "h15", "f15s", "cnn"], CANDS)
for vn, sets in VARS.items():
    t0 = time.time(); B[vn] = run_variant(sets); print(vn, "oof %.4f" % auc(y, B[vn][0]), B[vn][2], f"{time.time() - t0:.0f}s", flush=True)
for vn, (sets, cands) in L2.items():
    B[vn] = run_variant(sets, e3=None, cands=cands); print(vn, "oof %.4f" % auc(y, B[vn][0]), B[vn][2], flush=True)
# single heads (descriptive)
for c in COLS["h15"] + (["cnn"] if HAS_CNN else []):
    print("head", c, "train OOF %.4f" % auc(y[~T[c].isna().values], T[c].dropna()), "val %.4f" % auc(yv[~V[c].isna().values], V[c].dropna()), flush=True)

RES = {"mode": MODE, "n_train": int(len(T)), "pos_train": int(y.sum()), "n_val": int(len(V)), "pos_val": int(yv.sum()),
       "train_missing_h15": int(T.h15_U.isna().sum()), "val_missing_h15": int(V.h15_U.isna().sum()), "has_cnn": HAS_CNN, "variants": {}, "heads": {}}
for c in COLS["h15"] + (["cnn"] if HAS_CNN else []):
    kt, kv = ~T[c].isna().values, ~V[c].isna().values
    RES["heads"][c] = dict(oof=float(auc(y[kt], T[c].values[kt])), val=float(auc(yv[kv], V[c].values[kv])))
sub_gbm_v = (V.gb_missing.fillna(0).values == 1); sub_nv_v = (V.img_vis.astype(str).str.lower().isin(["0", "0.0", "false", "no", "not_visible", "nonvis"]).values)
so_ref, sv_ref = B["REF_off"][0], B["REF_off"][1]
OOF = T[["study_id", "fold", "y_gs"]].copy(); VAL = V[["study_id", "y_gs"]].copy()
for vn, (so, sv, best, cv, mdl) in B.items():
    thr = float(np.quantile(so[y == 0], .9))
    r = dict(cand=best, cv=cv, oof=float(auc(y, so)), val=float(auc(yv, sv)), val_ci=boot_ci(yv, sv),
             oof_age=age_matched_auc(y, so, aget), val_age=age_matched_auc(yv, sv, agev),
             spearman_neg_val=spearman_neg(yv, sv, agev), val_sens90=sens90(yv, sv), val_sens90_trthr=sens90(yv, sv, thr))
    for nm, k in (("gb_missing", sub_gbm_v), ("nonvis", sub_nv_v)):
        kk = k | (yv == 0)
        if (k & (yv == 1)).sum() >= 5:
            r[f"val_sub_{nm}"] = float(auc(yv[kk], sv[kk])); r[f"n_pos_{nm}"] = int((k & (yv == 1)).sum())
            if vn != "REF_off":
                r[f"d_val_sub_{nm}"] = boot_delta(yv[kk], sv_ref[kk], sv[kk], n=1000)
    if vn != "REF_off":
        r["d_oof_vs_off"] = boot_delta(y, so_ref, so, n=1000); r["d_val_vs_off"] = boot_delta(yv, sv_ref, sv)
        r["d_val_age_vs_off"] = boot_delta(yv, sv_ref, sv, n=500, f=lambda yy, s, ix: age_matched_auc(yy, s, agev[ix]))
        r["d_oof_vs_rb"] = boot_delta(y, B["REF_rb"][0], so, n=1000)
    RES["variants"][vn] = r; OOF[vn] = so; VAL[vn] = sv
    if mdl: joblib.dump(mdl | {"thr_spec90_trainoof": thr}, f"{O}/model_{vn.replace('+', '_')}.joblib")
    print("RES", vn, json.dumps({k: v for k, v in r.items() if k != "cv"}, default=float), flush=True)
new = [v for v in B if not v.startswith("REF")]
RES["selected_by_train_oof"] = sel = max(new, key=lambda v: RES["variants"][v]["oof"])
print("SELECTED", sel, flush=True)
OOF.to_csv(f"{O}/oof_train.csv", index=False); VAL.to_csv(f"{O}/val_scores.csv", index=False)
json.dump(RES, open(f"{O}/summary.json", "w"), ensure_ascii=False, indent=1, default=float)

# ---------------- report (繁體中文) ----------------
def f3(x): return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
def dci(d): return "–" if not d else f"{d['mean']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
NAMES = {"REF_off": "REF：Exp N（N_U，官方 .881）", "REF_rb": "REF 重建（同一批 train 列）", "NU+h15": "N_U＋1.5 mm 單元頭",
         "NU+f15": "N_U＋1.5 mm HU 特徵", "NU+h15+f15": "N_U＋單元頭＋HU 特徵", "NU+cnn": "N_U＋3D CNN", "NU+h15+f15+cnn": "N_U＋單元頭＋HU＋CNN",
         "L2_h15": "第二層 LR（N_U logit＋單元頭）", "L2_h15f": "第二層（N_U logit＋單元頭＋精選 HU）", "L2_h15cnn": "第二層 LR（＋CNN）",
         "L2_h15fcnn": "第二層（＋精選 HU＋CNN）"}
TRDESC = "＝tier0（全部 train 陽性＋年齡×性別配對陰性 1:2，只用有 embedding 的）" if MODE == "tier0" else f"（缺 1.5 mm embedding {RES['train_missing_h15']} 例，以 train 中位數補）"
lines = [f"## {'中期（tier0）' if MODE == 'tier0' else '全量'}結果（{time.strftime('%m-%d %H:%M')}）", "",
         f"- train 列：{len(T):,}（陽性 {int(y.sum()):,}）{TRDESC}；val：{len(V):,}（陽性 {int(yv.sum())}；缺 embedding {RES['val_missing_h15']}）。",
         "- 依 **train 5-fold OOF**（GS1 folds）選模型；val 只報告；Δ 為對官方 Exp N 分數的配對分層 bootstrap（val 2,000 次）。test 未讀取。",
         f"- **依 train OOF 選出：{NAMES.get(sel, sel)}（{RES['variants'][sel]['cand']}）**", "",
         "| 模型 | train OOF | ΔOOF vs 官方 | val [95% CI] | Δval vs .881 [CI] | 年齡配對 val（Δ） | 敏感度@90%特異度 val（train 門檻） | 膽囊遮罩缺失子群 val（Δ） | 3 mm 不可見子群 val |",
         "|---|---|---|---|---|---|---|---|---|"]
for vn in B:
    r = RES["variants"][vn]; star = " ★" if vn == sel else ""
    lines.append(f"| {NAMES.get(vn, vn)}{star} | {f3(r['oof'])} | {dci(r.get('d_oof_vs_off'))} | {f3(r['val'])} [{f3(r['val_ci'][0])}, {f3(r['val_ci'][1])}] | "
                 f"{dci(r.get('d_val_vs_off'))} | {f3(r['val_age'])}（{dci(r.get('d_val_age_vs_off'))}） | {r['val_sens90']*100:.1f}%（{r['val_sens90_trthr']*100:.1f}%） | "
                 f"{f3(r.get('val_sub_gb_missing'))}（{dci(r.get('d_val_sub_gb_missing'))}） | {f3(r.get('val_sub_nonvis'))} |")
lines += ["", "單一頭（描述用）：" + "；".join(f"{k} train OOF {v['oof']:.3f}／val {v['val']:.3f}" for k, v in RES["heads"].items()), ""]
if MODE == "tier0":
    lines += ["註：tier0 的 train 陰性是年齡×性別配對的子集，train OOF 絕對值與全量不可直接比較；同一批列上的 REF（官方 OOF、重建）才是比較基準。", ""]
open(f"{O}/report_part.md", "w").write("\n".join(lines))
parts = [p for p in (f"{RT}/stack/report_part.md", f"{RT}/stack_tier0/report_part.md") if os.path.exists(p)]
hdr = open(f"{RT}/scripts/report_header.md").read() if os.path.exists(f"{RT}/scripts/report_header.md") else "# 膽結石 1.5 mm 膽囊 crop 單元（gs15）\n"
open(f"{RT}/report.md", "w").write(hdr + "\n" + "\n\n".join(open(p).read() for p in parts) + "\n")
print("GS15_STACK_DONE", flush=True)
