"""Tier B appendicitis RLQ unit: refit rlq_probe.py heads (E3_late, E3cs, U_rlq_cs; target y_ap) on Merlin train, score
Merlin val (check) + CG rows; then refit the 'REF＋U_rlq_cs' LR stack exactly as ib_v2_export_ap.py / rlq_eval.py and
score CG.  usage: cg_brlq.py <cg_rlq_emb_main> [<cg_rlq_emb_delay>]   (OOF loop skipped: refit is fold-independent)"""
import os, sys, re, subprocess, joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); OF = H + "mw_exp/outputs_final/"; U = OF + "appendicitis_rlq_unit/"
SB = H + "cg10k_best/bprobe/rlq/"; HR = H + "cg10k_best/heads/R/"; PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")
MAT = "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz"
os.makedirs(SB + "emb", exist_ok=True)
lab = pd.read_csv(OF + "appendicitis_split/labels_trval_split.csv")
cg = np.load(HR + MAT, mmap_mode="r")["ids"]; cg = [str(i) for i in cg if not str(i).startswith("AC")]
add = pd.DataFrame({"study_id": cg, "split": "cg"})
for c in ("y_ap", "y_ap_def", "y_ap_unc"): add[c] = 0.0
pd.concat([lab, add], ignore_index=True).to_csv(SB + "labels.csv", index=False)
for f in os.listdir(U + "emb_rlq"):
    t = SB + "emb/" + f
    if not os.path.exists(t): os.symlink(U + "emb_rlq/" + f, t)
for pre, ed in zip(("", "D_"), [os.path.abspath(e) for e in sys.argv[1:]]):
    for f in os.listdir(ed):
        t = SB + "emb/" + pre + f
        if not os.path.exists(t): os.symlink(os.path.join(ed, f), t)
code = open(U + "scripts/rlq_probe.py").read()
code = code.replace('R + "outputs_final/' + MAT + '"', f'"{HR + MAT}"')
code = code.replace('R + "outputs_final/appendicitis_split/labels_trval_split.csv"', f'"{SB}labels.csv"')
code = code.replace('assert set(lab.split) <= {"train", "val"}', 'assert set(lab.split) <= {"train", "val", "cg"}')
code = code.replace('vai_all = np.where(S == "val")[0]', 'vai_all = np.where(np.isin(S, ("val", "cg")))[0]')
code = code.replace("for k in range(5):", "for k in []:  # CG: OOF skipped")
assert HR + MAT in code and SB + "labels.csv" in code and "for k in []" in code
open(SB + "patched_rlq_probe.py", "w").write(code)
if not os.path.exists(SB + "out/val_scores.csv"):
    r = subprocess.run(["nice", "-n", "15", PY, SB + "patched_rlq_probe.py", SB + "emb", SB + "out", "--device", "cuda", "--targets", "y_ap",
                        "--variants", "E3_late,E3cs,U_rlq_cs"], stdout=open(SB + "probe.log", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, NT="4", OMP_NUM_THREADS="4"))
    assert r.returncode == 0
v = pd.read_csv(SB + "out/val_scores.csv"); o = pd.read_csv(U + "out/val_scores.csv")
m = o.merge(v, on="study_id", suffixes=("_old", "_new"))
for c in ("y_ap__E3_late", "y_ap__E3cs", "y_ap__U_rlq_cs"):
    print(c, "max|d|", np.abs(m[c + "_old"] - m[c + "_new"]).max(), "corr", np.corrcoef(m[c + "_old"], m[c + "_new"])[0, 1])
# ---- stack (same as ib_v2_export_ap.py, target y_ap)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
oof = pd.read_csv(U + "out/oof_train.csv")
hu = pd.concat([pd.read_csv(OF + "analysis_appendicitis/hu_a6.csv"), pd.read_csv(OF + "analysis_appendicitis/hu_ct.csv")]).drop_duplicates("study_id")
LOCAL = [c for c in hu.columns if c.startswith("dlr_") or c.startswith("dglob_")] + \
        [f"rlq_r{r}_{f}" for r in (15, 30) for f in ("fat_med", "fat_p75", "strand_frac", "fluid_frac", "soft_frac", "fat_frac")] + ["rlq_wall_p50", "rlq_wall_p90"]
T = oof.merge(lab, on="study_id", how="left").merge(hu[["study_id"] + LOCAL], on="study_id", how="left")
tg = "y_ap"; Tm = T[T[tg].notna() & T[f"{tg}__E3_late"].notna()].reset_index(drop=True)
cols = [f"{tg}__E3_late", f"{tg}__E3cs", f"{tg}__U_rlq_cs"] + LOCAL
X = Tm[cols]; med = X.median(); X = X.fillna(med)
mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")).fit(X, Tm[tg].values.astype(int))
Vv = o.merge(lab, on="study_id").merge(hu[["study_id"] + LOCAL], on="study_id", how="left"); Vv = Vv[Vv[tg].notna() & Vv[f"{tg}__E3_late"].notna()]
print("val stack (stored heads)", round(roc_auc_score(Vv[tg].astype(int), mdl.decision_function(Vv[cols].fillna(med))), 6), "expected .920365")
Vn = v.merge(lab, on="study_id").merge(hu[["study_id"] + LOCAL], on="study_id", how="left"); Vn = Vn[Vn[tg].notna()]
print("val stack (refit heads)", round(roc_auc_score(Vn[tg].astype(int), mdl.decision_function(Vn[cols].fillna(med))), 6))
joblib.dump(dict(model=mdl, cols=cols, median=med), SB + "stack_appendicitis.joblib")
for s, pre in (("main", ""), ("delay", "D_")):
    f = H + f"cg10k_best/feat/{s}/appendicitis_hu.csv"
    if not os.path.exists(f): continue
    hc = pd.read_csv(f); hc["study_id"] = pre + hc.study_id
    C = v[v.study_id.str.startswith(pre + "EX")] if pre else v[v.study_id.str.startswith("EX")]
    C = C.merge(hc[["study_id"] + LOCAL], on="study_id", how="left")
    out = pd.DataFrame({"study_id": C.study_id.str.replace("^D_", "", regex=True), "appendicitis": mdl.decision_function(C[cols].fillna(med))})
    out.to_csv(SB + f"cg_scores_{s}.csv", index=False); print(s, len(out))
