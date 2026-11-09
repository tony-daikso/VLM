"""Quick wins: late stacking of hand-crafted HU features with the E3_late logit (膽結石, 脂肪肝).

Model selection: train 5-fold OOF only (same folds as the E3_late OOF). Val: final report only.
Test is never read here. Age/sex are NOT model inputs (age only used for the age-matched AUROC metric).

Usage:
  python qw_stack.py fit   <in_dir> <out_dir>      # select + fit on train OOF, report on val
  python qw_stack.py apply <out_dir> <disease> <hu_csv[,hu_csv2]> <score_csv> <out_csv>
      # later, for a locked test run: hu_csv = output of gs_hu.py / hs_hu.py, score_csv has study_id,E3_late
"""
import sys, os, json, warnings, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# ---------------- features (shared by fit and apply) ----------------
GS_HU = ["gb_vox", "gb_p50", "gb_p90", "gb_p95", "gb_p99", "gb_max", "gb_std", "gb_n60", "gb_n100", "gb_n130", "gb_n200",
         "gb_n_med40", "gb_n_lowfat", "shell_n100", "shell_p99", "gb_maxblob100", "liver_hu_med", "aorta_hu_med"]
HS_HU = ["liver_er_med", "liver_mean", "liver_p10", "liver_p25", "liver_p50", "liver_p75", "liver_p90", "liver_std", "liver_er_iqr",
         "liver_frac_lt40", "liver_frac_lt0", "liver_slab_sd", "spleen_med", "LS_diff", "LS_ratio", "aorta_med", "portal_med", "ivc_med",
         "kid_med", "kid_cmd", "L_minus_aorta", "L_minus_portal", "log_liver_vox", "L3_fat_cm2", "L3_fat_outer_cm2", "L3_fat_inner_cm2",
         "fat_frac_L3", "L1_fat_cm2"]
PH = ["native", "portal_venous", "arterial", "delayed"]


def feat_gs(d):
    X = d[GS_HU].astype(float).copy()
    for c in GS_HU:
        if c.startswith(("gb_n", "gb_vox", "shell_n", "gb_max")):
            X[c] = np.log1p(X[c].clip(lower=0))
    X["gb_present"] = (d.gb_vox.fillna(0) >= 100).astype(float)
    return X


def feat_hs(d):
    d = d.copy()
    d["LS_diff"] = d.liver_er_med - d.spleen_med
    d["LS_ratio"] = d.liver_er_med / d.spleen_med.where(d.spleen_med.abs() > 5)
    d["L_minus_aorta"] = d.liver_er_med - d.aorta_med
    d["L_minus_portal"] = d.liver_er_med - d.portal_med
    d["kid_med"] = d[["kidney_left_med", "kidney_right_med"]].mean(1)
    d["kid_cmd"] = d[["kidney_left_p90", "kidney_right_p90"]].mean(1) - d[["kidney_left_p10", "kidney_right_p10"]].mean(1)
    d["log_liver_vox"] = np.log(d.liver_vox.clip(lower=1))
    d["fat_frac_L3"] = d.L3_fat_cm2 / d.L3_body_cm2
    a, p = d.aorta_med, d.portal_med  # HU-estimated phase (metadata phase is unreliable)
    est = np.select([(a < 80) & ~(p > 100), (a >= 180) & ((a - p) > 50), a < 110], ["native", "arterial", "delayed"], "portal_venous")
    est = np.where(a.isna(), "unknown", est)
    X = d[HS_HU].astype(float).copy()
    for g in PH:
        X[f"ph_{g}"] = (est == g).astype(float)
    for g in PH:  # phase interactions (used by LR; harmless for GBM)
        for f in ("liver_er_med", "LS_diff", "liver_p25", "L_minus_portal"):
            X[f"{f}_x_{g}"] = X[f] * X[f"ph_{g}"]
    return X


FEAT = {"gs": feat_gs, "hs": feat_hs}
NAME = {"gs": "膽結石", "hs": "脂肪肝"}


def design(dis, d, logit_cols, med=None):
    X = FEAT[dis](d)
    for c in logit_cols:
        X[c] = d[c].astype(float).values
    if med is None:
        med = X.median()
    return X.fillna(med), med


# ---------------- candidates ----------------
def LR(C):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def GB(lr, leaves, msl, it):
    return lambda: HistGradientBoostingClassifier(max_iter=it, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                                  l2_regularization=1.0, random_state=0)


CANDS = {"LR_C0.1": LR(0.1), "LR_C0.3": LR(0.3), "LR_C1": LR(1.0),
         "GBM_lr.05_l15_m40_300": GB(0.05, 15, 40, 300), "GBM_lr.03_l7_m40_500": GB(0.03, 7, 40, 500),
         "GBM_lr.05_l31_m80_200": GB(0.05, 31, 80, 200)}


def score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


# ---------------- metrics ----------------
def auc(y, s):
    return roc_auc_score(y, s)


def boot_delta(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; d = []
    for _ in range(n):
        ix = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        d.append(auc(y[ix], b[ix]) - auc(y[ix], a[ix]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def age_matched_auc(y, s, age, edges=(0, 30, 40, 50, 60, 70, 80, 200)):
    """AUROC over positive–negative pairs within the same 10-year age bin (pair-count weighted)."""
    b = pd.cut(age, edges, right=False); num = den = 0.0
    for g in b.dropna().unique():
        m = (b == g).values
        n1, n0 = y[m].sum(), (1 - y[m]).sum()
        if n1 and n0:
            num += auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den


def sens_at_spec(y, s, spec=0.9, thr=None):
    if thr is None:
        thr = np.quantile(s[y == 0], spec)
    return float((s[y == 1] > thr).mean()), thr


# ---------------- fit ----------------
def load(dis, I):
    S = I + "/" + dis + "/"
    lab = pd.read_csv(S + "labels_trval.csv")
    lab = lab[lab.split.isin(["train", "val"])]  # split filter before labels are used
    ycol = {"gs": "y_gs", "hs": "y_hs"}[dis]
    lab = lab[["study_id", "split", ycol]].rename(columns={ycol: "y"})
    lab = lab[lab.y.notna()]
    hu_files = ["hu_a6.csv", "hu_ct.csv"] if dis == "gs" else ["hs_a6.csv", "hs_ct.csv"]
    hu = pd.concat([pd.read_csv(S + f) for f in hu_files]).drop_duplicates("study_id")
    meta = pd.read_csv(I + "/qw_meta_trval.csv")[["study_id", "Age"]]
    df = lab.merge(hu, on="study_id", how="left").merge(meta, on="study_id", how="left")
    oof = pd.read_csv(S + "oof_train.csv")[["study_id", "fold", "E3_late"]]
    vs = pd.read_csv(S + "scores_val_descriptive.csv")[["study_id", "E3_late"]]
    if dis == "hs":
        oof = oof.merge(pd.read_csv(S + "oof_train_concat_diff.csv")[["study_id", "E3_late_concat_diff"]], on="study_id")
        vs = vs.merge(pd.read_csv(S + "scores_val_concat_diff.csv")[["study_id", "E3_late_concat_diff"]], on="study_id")
    T = df[df.split == "train"].merge(oof, on="study_id").reset_index(drop=True)
    V = df[df.split == "val"].merge(vs, on="study_id").reset_index(drop=True)
    return T, V


def oof_scores(dis, T, mk, logit_cols):
    y = T.y.values.astype(int); s = np.zeros(len(T))
    for k in sorted(T.fold.unique()):
        tr, te = T.fold.values != k, T.fold.values == k
        Xtr, med = design(dis, T[tr], logit_cols)
        Xte, _ = design(dis, T[te], logit_cols, med)
        s[te] = score(mk().fit(Xtr, y[tr]), Xte)
    return s


def fit(I, O):
    os.makedirs(O, exist_ok=True); summ = {}
    for dis in ["gs", "hs"]:
        T, V = load(dis, I)
        y, yv = T.y.values.astype(int), V.y.values.astype(int)
        variants = {"E3_late+HU": ["E3_late"]}
        if dis == "hs":
            variants["E3_late(肝‖肝−脾)+HU"] = ["E3_late_concat_diff"]
        res = {"n_train": len(T), "pos_train": int(y.sum()), "n_val": len(V), "pos_val": int(yv.sum()),
               "variants": {}}
        base_tr, base_v = T.E3_late.values, V.E3_late.values
        thr_b = np.quantile(base_tr[y == 0], .9)
        res["E3_late"] = dict(oof=auc(y, base_tr), val=auc(yv, base_v),
                              oof_age=age_matched_auc(y, base_tr, T.Age), val_age=age_matched_auc(yv, base_v, V.Age),
                              oof_sens90=sens_at_spec(y, base_tr)[0], val_sens90=sens_at_spec(yv, base_v)[0],
                              val_sens90_trthr=sens_at_spec(yv, base_v, thr=thr_b)[0])
        for vn, lc in variants.items():
            cv = {}
            for cn, mk in CANDS.items():
                cv[cn] = auc(y, oof_scores(dis, T, mk, lc)); print(dis, vn, cn, round(cv[cn], 4), flush=True)
            best = max(cv, key=cv.get)
            s_oof = oof_scores(dis, T, CANDS[best], lc)
            X, med = design(dis, T, lc); m = CANDS[best]().fit(X, y)
            Xv, _ = design(dis, V, lc, med); sv = score(m, Xv)
            tag = "main" if vn == "E3_late+HU" else "concat"
            joblib.dump({"model": m, "medians": med, "columns": list(X.columns), "logit_cols": lc, "disease": dis,
                         "cand": best, "thr_spec90_trainoof": float(np.quantile(s_oof[y == 0], .9))},
                        f"{O}/stack_{dis}_{tag}.joblib")
            lo, hi, p = boot_delta(yv, base_v, sv)
            lo_t, hi_t, p_t = boot_delta(y, base_tr, s_oof)
            thr = np.quantile(s_oof[y == 0], .9)
            res["variants"][vn] = dict(cand=best, cv=cv, oof=auc(y, s_oof), val=auc(yv, sv), d_val_ci=[lo, hi], p_val=p,
                                       d_oof_ci=[lo_t, hi_t], oof_age=age_matched_auc(y, s_oof, T.Age),
                                       val_age=age_matched_auc(yv, sv, V.Age), oof_sens90=sens_at_spec(y, s_oof)[0],
                                       val_sens90=sens_at_spec(yv, sv)[0], val_sens90_trthr=sens_at_spec(yv, sv, thr=thr)[0])
            if tag == "main":
                pd.DataFrame({"study_id": T.study_id, "fold": T.fold, "y": y, "E3_late": base_tr, "stack": s_oof}).to_csv(f"{O}/oof_train_{dis}.csv", index=False)
            pv = pd.read_csv(f"{O}/val_scores_{dis}.csv") if tag != "main" else pd.DataFrame({"study_id": V.study_id, "y": yv, "E3_late": base_v})
            pv["stack" if tag == "main" else "stack_concat"] = sv
            pv.to_csv(f"{O}/val_scores_{dis}.csv", index=False)
            if dis == "gs" or tag == "main":
                print(dis, vn, json.dumps({k: v for k, v in res["variants"][vn].items() if k != "cv"}, ensure_ascii=False), flush=True)
        summ[dis] = res
    json.dump(summ, open(f"{O}/summary.json", "w"), ensure_ascii=False, indent=1, default=float)


def apply(O, dis, hu_csvs, score_csv, out_csv):
    tag = sys.argv[7] if len(sys.argv) > 7 else "main"
    b = joblib.load(f"{O}/stack_{dis}_{tag}.joblib")
    hu = pd.concat([pd.read_csv(f) for f in hu_csvs.split(",")]).drop_duplicates("study_id")
    d = pd.read_csv(score_csv).merge(hu, on="study_id", how="left")
    X, _ = design(dis, d, b["logit_cols"], b["medians"])
    pd.DataFrame({"study_id": d.study_id, "stack": score(b["model"], X[b["columns"]])}).to_csv(out_csv, index=False)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3])
    else:
        apply(*sys.argv[2:7])
