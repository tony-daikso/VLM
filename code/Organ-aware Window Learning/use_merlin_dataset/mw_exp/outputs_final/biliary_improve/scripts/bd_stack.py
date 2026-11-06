"""Formal stack for biliary ductal dilation (BD1, plus BD2/BD3/BD5 additions when their files exist).
Level-1 inputs: base-model logits (train = 5-fold OOF from bd_oof.py / bd2_aux.py / bd5 probe; val = full-train refit)
+ handcrafted 3 mm features (bd_hu porta/hilum HU, BD3 dark-tube detector, extended porta/HDL CBD proxy).
Level-2 model chosen by TRAIN CV (same 5 folds): LR (standardised, log counts, missing indicators) vs histogram GBM.
VAL is only scored with the configuration selected on train.  Usage: bd_stack.py WORKDIR OUTDIR"""
import sys, os, json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bd_eval import auc, full_metrics, add_groups, fmt
warnings.filterwarnings("ignore")
W, O = sys.argv[1], sys.argv[2]
os.makedirs(O, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(W, f))
lab = add_groups(rd("labels_trval.csv")); lab = lab[lab.split.isin(["train", "val"]) & lab.y_bdd.notna()]
meta = rd("meta_trval.csv")[["study_id", "Age", "Gender"]]
oof = rd("oof_train.csv"); val = rd("scores_val_descriptive.csv")
for f, key in (("bd2_oof_train.csv", "bd2_val.csv"), ("bd5_oof_train.csv", "bd5_val.csv")):
    if os.path.exists(os.path.join(W, f)):
        a, b = rd(f), rd(key)
        oof = oof.merge(a.drop(columns=["fold"], errors="ignore"), on="study_id", how="left")
        val = val.merge(b, on="study_id", how="left")
BASES = [c for c in oof.columns if c not in ("study_id", "fold") and not c.endswith("_reg")]
sc = pd.concat([oof.assign(split_s="train"), val.assign(split_s="val")])
feats = []
for f in ("hu_all.csv", "bd3_all.csv"):
    if os.path.exists(os.path.join(W, f)):
        feats.append(rd(f).drop_duplicates("study_id"))
F = feats[0]
for g in feats[1:]:
    F = F.merge(g[[c for c in g.columns if c == "study_id" or c not in F.columns]], on="study_id", how="outer")
F = F[[c for c in F.columns if not c.startswith("err_")]]
D = lab.merge(meta, on="study_id", how="left").merge(sc, on="study_id", how="inner").merge(F, on="study_id", how="left")
D["male"] = (D.Gender == "Male").astype(float)
T = D[D.split == "train"].reset_index(drop=True); V = D[D.split == "val"].reset_index(drop=True)
assert T.fold.notna().all()
y = T.y_bdd.values.astype(int); yv = V.y_bdd.values.astype(int)
print("train", len(T), "val", len(V), "feature coverage train", round(T.liver_vox.notna().mean(), 3), "val", round(V.liver_vox.notna().mean(), 3))
HU = ["liv_low_n", "liv_low_frac", "liv_low_maxcc", "liv_low_ncc", "liv_fluid_n", "liv_air_n", "hil_low_n", "hil_low_frac",
      "hil_low_maxcc", "porta_fluid_n", "porta_fluid_frac", "porta_fluid_maxcc", "porta_air_n", "porta_metal_n", "porta_hu_med",
      "panc_low_n", "panc_low_frac", "phead_fluid_n", "phead_metal_n", "liver_hu_sd", "liver_vox", "panc_vox", "gb_vox", "pv_vox",
      "liver_hu", "aorta_hu", "pv_hu", "panc_hu"]
TUBE = [c for c in D.columns if c.startswith("tube_")]
PORTA = [c for c in D.columns if c.startswith(("hdl_", "duct_lp_", "phd_fl_"))]
HU = [c for c in HU if c in D.columns]


def prep(X, med=None):
    X = X.copy().astype(float)
    for c in X.columns:
        if c.endswith(("_n", "_vox", "_maxcc", "_ncc", "_ml", "_vol", "_skel", "_branch", "_ends")) or "maxcc" in c:
            X[c] = np.log1p(X[c].clip(lower=0))
    miss = X.isna().any(axis=1).astype(float)
    med = X.median() if med is None else med
    return X.fillna(med).assign(miss_any=miss), med


def model(kind):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=60,
                                          l2_regularization=1.0, random_state=0)


def score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


def run(cols, kind):
    p = np.zeros(len(T))
    for k in range(5):
        tr, te = (T.fold != k).values, (T.fold == k).values
        Xtr, med = prep(T.loc[tr, cols]); Xte, _ = prep(T.loc[te, cols], med)
        m = model(kind).fit(Xtr, y[tr]); p[te] = score(m, Xte)
    Xtr, med = prep(T[cols]); Xv, _ = prep(V[cols], med)
    m = model(kind).fit(Xtr, y)
    return p, score(m, Xv)


configs = {}
def add(name, cols):
    cols = [c for c in cols if c in D.columns]
    if cols: configs[name] = cols
base = "E3_late"; best_emb = "E3late_cat_+gb_duod"
add("E3_late+HU", [base] + HU)
add("BD1: E3_late+E0+HU+porta", [base, "E0"] + HU + PORTA)
add("E3_late+HU+tube", [base] + HU + TUBE)
add("E3_late+HU+tube+porta", [base] + HU + TUBE + PORTA)
add("E3_late+E0+HU+tube+porta", [base, "E0"] + HU + TUBE + PORTA)
add("cat4+HU", [best_emb] + HU)
add("cat4+HU+tube+porta", [best_emb] + HU + TUBE + PORTA)
add("cat4+E3_late+E0+HU+tube+porta", [best_emb, base, "E0"] + HU + TUBE + PORTA)
add("cat4+E3_late+E0+catwin+all12+HU+tube+porta", [best_emb, base, "E0", "E2_catwin_liver_panc", "E0_cat_all12"] + HU + TUBE + PORTA)
aux = [c for c in BASES if ("aux" in c and not c.endswith("aux0")) or ("mtl" in c and not c.endswith("mtl0"))]
auxreg = [c + "_reg" for c in aux if c + "_reg" in oof.columns]
if aux:
    add("cat4+E3_late+E0+aux+HU+tube+porta", [best_emb, base, "E0"] + aux + HU + TUBE + PORTA)
    add("cat4+E3_late+E0+aux+auxreg+HU+tube+porta", [best_emb, base, "E0"] + aux + auxreg + HU + TUBE + PORTA)
    add("cat4+E3_late+E0+auxreg+HU+tube+porta", [best_emb, base, "E0"] + auxreg + HU + TUBE + PORTA)
bd5 = [c for c in BASES if c.startswith("P5_")]
if bd5:
    add("cat4+E3_late+E0+porta_emb+HU+tube+porta", [best_emb, base, "E0"] + bd5 + HU + TUBE + PORTA)
    add("all_bases+HU+tube+porta", [best_emb, base, "E0"] + aux + auxreg + bd5 + HU + TUBE + PORTA)
NOART = [c for c in HU if c not in ("liv_air_n", "porta_air_n", "porta_metal_n", "phead_metal_n")]
add("BEST_noairmetal: cat4+E3_late+E0+catwin+all12+HU(no air/metal)+tube+porta",
    [best_emb, base, "E0", "E2_catwin_liver_panc", "E0_cat_all12"] + NOART + TUBE + PORTA)
add("HU+tube+porta only", HU + TUBE + PORTA)
add("tube only", TUBE)
add("porta only", PORTA)
only = os.environ.get("ONLY")
rows = []; P_T = pd.DataFrame({"study_id": T.study_id}); P_V = pd.DataFrame({"study_id": V.study_id})
for b in [c for c in BASES if c in T.columns] + [c for c in T.columns if c.endswith("_reg")]:
    P_T[b] = T[b]; P_V[b] = V[b]
for name, cols in configs.items():
    if only and only not in name:
        continue
    for kind in ("lr", "gbm"):
        p, pv = run(cols, kind)
        nm = f"{name}|{kind}"; P_T[nm] = p; P_V[nm] = pv
        print(f"{nm:60s} train OOF {auc(y, p):.4f}  val {auc(yv, pv):.4f}", flush=True)
P_T.to_csv(os.path.join(O, "stack_oof_train.csv"), index=False); P_V.to_csv(os.path.join(O, "stack_val.csv"), index=False)
# metrics table: every column, train OOF (vs E3_late OOF) and val (vs E3_late full-train refit)
TT = T[["study_id", "y_bdd", "Age", "g_mild", "g_extra", "g_chole", "g_stent"]].merge(P_T, on="study_id")
VV = V[["study_id", "y_bdd", "Age", "g_mild", "g_extra", "g_chole", "g_stent"]].merge(P_V, on="study_id")
out = []
for c in P_T.columns[1:]:
    if TT[c].isna().mean() > 0.01 or VV[c].isna().mean() > 0.01:
        continue
    rt = full_metrics(TT, c, base); rv = full_metrics(VV, c, base)
    out.append({"model": c, **{f"tr_{k}": v for k, v in rt.items()}, **{f"va_{k}": v for k, v in rv.items()}})
R = pd.DataFrame(out).sort_values("tr_auc", ascending=False)
R.to_csv(os.path.join(O, "stack_metrics.csv"), index=False)
pd.set_option("display.width", 250)
print(R[["model", "tr_auc", "tr_delta", "va_auc", "va_ci_lo", "va_ci_hi", "va_delta", "va_d_lo", "va_d_hi", "va_age_matched", "va_sens90"]].round(4).to_string())
