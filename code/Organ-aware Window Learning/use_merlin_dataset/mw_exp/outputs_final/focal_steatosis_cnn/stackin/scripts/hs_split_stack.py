"""脂肪肝拆分：stack E3_late logit + HU features (+ focal local HU features) for y_diffuse / y_focal.

Selection: train 5-fold OOF only (folds = the E3_late OOF folds of that target). Val: reporting only
(paired stratified bootstrap 2,000x). Test is never read. Age/sex are NOT model inputs (age only for the
age-matched AUROC metric).

Usage:
  python hs_split_stack.py fit   <data dir> <oof dir> <out dir>
  python hs_split_stack.py apply <out dir> <target> <hu_csvs> <focal_csvs> <score_csv> <out_csv>   # later, locked test
"""
import sys, os, json, warnings, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

HS_HU = ["liver_er_med", "liver_mean", "liver_p10", "liver_p25", "liver_p50", "liver_p75", "liver_p90", "liver_std", "liver_er_iqr",
         "liver_frac_lt40", "liver_frac_lt0", "liver_slab_sd", "spleen_med", "LS_diff", "LS_ratio", "aorta_med", "portal_med", "ivc_med",
         "kid_med", "kid_cmd", "L_minus_aorta", "L_minus_portal", "log_liver_vox", "L3_fat_cm2", "L3_fat_outer_cm2", "L3_fat_inner_cm2",
         "fat_frac_L3", "L1_fat_cm2"]
FL = ["fl_std", "fl_mad", "fl_iqr", "fl_skew", "fl_p1_rel", "fl_p5_rel", "fl_p10_rel", "fl_p25_rel", "fl_sm_p1_rel", "fl_sm_p5_rel",
      "fl_sm_p10_rel", "fl_sm_std", "fl_ml_lt15", "fl_frac_lt15", "fl_blob_lt15_ml", "fl_nblob_lt15", "fl_ml_lt25", "fl_frac_lt25",
      "fl_blob_lt25_ml", "fl_nblob_lt25", "fl_ml_lt40", "fl_frac_lt40",
      "gbf_dmed", "gbf_ml_lt15", "gbf_frac_lt15", "gbf_ml_lt25", "gbf_frac_lt25",
      "porta_dmed", "porta_ml_lt15", "porta_frac_lt15", "porta_ml_lt25", "porta_frac_lt25",
      "falc_dmed", "falc_ml_lt15", "falc_frac_lt15", "falc_ml_lt25", "falc_frac_lt25",
      "blk_min_rel", "blk_p5_rel", "blk_max_rel", "blk_std", "blk_max_frac25", "blk_n_lt15"]
PH = ["native", "portal_venous", "arterial", "delayed"]
TARGETS = {"y_diffuse": "瀰漫性脂肪肝", "y_focal": "局部脂肪浸潤"}
SENS = {"y_diffuse": ["y_diffuse_excl"], "y_focal": ["y_focal_excl", "y_focal_negclean"]}


def feat_hu(d):
    d = d.copy()
    d["LS_diff"] = d.liver_er_med - d.spleen_med
    d["LS_ratio"] = d.liver_er_med / d.spleen_med.where(d.spleen_med.abs() > 5)
    d["L_minus_aorta"] = d.liver_er_med - d.aorta_med
    d["L_minus_portal"] = d.liver_er_med - d.portal_med
    d["kid_med"] = d[["kidney_left_med", "kidney_right_med"]].mean(1)
    d["kid_cmd"] = d[["kidney_left_p90", "kidney_right_p90"]].mean(1) - d[["kidney_left_p10", "kidney_right_p10"]].mean(1)
    d["log_liver_vox"] = np.log(d.liver_vox.clip(lower=1))
    d["fat_frac_L3"] = d.L3_fat_cm2 / d.L3_body_cm2
    a, p = d.aorta_med, d.portal_med
    est = np.select([(a < 80) & ~(p > 100), (a >= 180) & ((a - p) > 50), a < 110], ["native", "arterial", "delayed"], "portal_venous")
    est = np.where(a.isna(), "unknown", est)
    X = d[HS_HU].astype(float).copy()
    for g in PH:
        X[f"ph_{g}"] = (est == g).astype(float)
    for g in PH:
        for f in ("liver_er_med", "LS_diff", "liver_p25", "L_minus_portal"):
            X[f"{f}_x_{g}"] = X[f] * X[f"ph_{g}"]
    return X


def feat_fl(d):
    X = d[FL].astype(float).copy()
    for c in FL:
        if "_ml_" in c or c.endswith("_ml") or c.startswith(("fl_nblob", "blk_n_")):
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


FSETS = {  # name: (use HU, use FL, use logit)
    "HU": (1, 0, 0), "E3+HU": (1, 0, 1), "FL": (0, 1, 0), "E3+FL": (0, 1, 1), "HU+FL": (1, 1, 0), "E3+HU+FL": (1, 1, 1)}


def design(d, fs, logit, med=None):
    hu, fl, lg = FSETS[fs]; parts = []
    if hu:
        parts.append(feat_hu(d))
    if fl:
        parts.append(feat_fl(d))
    X = pd.concat(parts, axis=1)
    if lg:
        X["logit"] = d[logit].astype(float).values
    if med is None:
        med = X.median()
    return X.fillna(med), med


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


def auc(y, s):
    return roc_auc_score(y, s)


def boot(y, s, base=None, n=2000, seed=0):
    rng = np.random.default_rng(seed); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; a, d = [], []
    for _ in range(n):
        ix = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        a.append(auc(y[ix], s[ix]))
        if base is not None:
            d.append(a[-1] - auc(y[ix], base[ix]))
    r = dict(ci=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))])
    if base is not None:
        r["d_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]; r["p_le0"] = float((np.array(d) <= 0).mean())
    return r


def age_matched_auc(y, s, age, edges=(0, 30, 40, 50, 60, 70, 80, 200)):
    b = pd.Series(pd.cut(np.asarray(age, dtype=float), edges, right=False)); num = den = 0.0
    for g in b.dropna().unique():
        m = (b == g).values; y = np.asarray(y); s = np.asarray(s)
        n1, n0 = y[m].sum(), (1 - y[m]).sum()
        if n1 and n0:
            num += auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den


def sens90(y, s, thr=None):
    if thr is None:
        thr = np.quantile(s[y == 0], .9)
    return float((s[y == 1] > thr).mean()), float(thr)


def metrics(yt, st, at, yv, sv, av, bt=None, bv=None):
    s_oof, thr = sens90(yt, st)
    r = dict(oof=auc(yt, st), val=auc(yv, sv), oof_age=age_matched_auc(yt, st, at), val_age=age_matched_auc(yv, sv, av),
             oof_sens90=s_oof, val_sens90=sens90(yv, sv)[0], val_sens90_trthr=sens90(yv, sv, thr)[0])
    bb = boot(yv, sv, bv); r["val_ci"] = bb["ci"]
    if bv is not None:
        r["d_val_ci"] = bb["d_ci"]; r["d_val_p_le0"] = bb["p_le0"]
        r["d_oof_ci"] = boot(yt, st, bt, n=1000)["d_ci"]
    return r


def load(D, OD, tgt, var):
    lab = pd.read_csv(D + "/labels_trval_split.csv")
    lab = lab[lab.split.isin(["train", "val"])]
    hu = pd.concat([pd.read_csv(D + f) for f in ("/hs_a6.csv", "/hs_ct.csv")]).drop_duplicates("study_id")
    fl = pd.concat([pd.read_csv(D + f) for f in ("/fl_a6.csv", "/fl_ct.csv")]).drop_duplicates("study_id")
    meta = pd.read_csv(D + "/meta_trval.csv")[["study_id", "Age"]]
    keep = ["study_id", "split", tgt] + [c for c in SENS[tgt]]
    df = lab[keep].merge(hu, on="study_id", how="left").merge(fl, on="study_id", how="left").merge(meta, on="study_id", how="left")
    oof = pd.read_csv(f"{OD}/oof_{tgt}_{var}.csv").drop(columns="y")
    vs = pd.read_csv(f"{OD}/val_{tgt}_{var}.csv").drop(columns="y")
    T = df[df.split == "train"].merge(oof, on="study_id").reset_index(drop=True)
    V = df[df.split == "val"].merge(vs, on="study_id").reset_index(drop=True)
    T["y"], V["y"] = T[tgt].astype(int), V[tgt].astype(int)
    return T, V


def oof_scores(T, fs, logit, mk):
    y = T.y.values; s = np.zeros(len(T))
    for k in sorted(T.fold.unique()):
        tr, te = T.fold.values != k, T.fold.values == k
        Xtr, med = design(T[tr], fs, logit); Xte, _ = design(T[te], fs, logit, med)
        s[te] = score(mk().fit(Xtr, y[tr]), Xte)
    return s


def fit(D, OD, O):
    os.makedirs(O, exist_ok=True); summ = {}; rows = []
    for tgt, zh in TARGETS.items():
        res = {"probe": {}, "stack": {}}
        # 1) E3_late variant: choose by train OOF
        pv = {}
        for var in ("liver", "concat", "concat_diff"):
            T, V = load(D, OD, tgt, var); c = f"E3_late_{var}"
            pv[var] = auc(T.y.values, T[c].values)
            res["probe"][var] = metrics(T.y.values, T[c].values, T.Age, V.y.values, V[c].values, V.Age)
            rows.append(dict(finding=zh, target=tgt, model=f"E3_late（{ {'liver':'肝','concat':'肝‖脾','concat_diff':'肝‖(肝−脾)'}[var] }）",
                             **{k: v for k, v in res["probe"][var].items()}))
        bvar = max(pv, key=pv.get); res["best_probe"] = bvar
        T, V = load(D, OD, tgt, bvar); logit = f"E3_late_{bvar}"
        y, yv = T.y.values, V.y.values; bt, bv = T[logit].values, V[logit].values
        res.update(n_train=len(T), pos_train=int(y.sum()), n_val=len(V), pos_val=int(yv.sum()))
        # 2) feature sets x candidates (train OOF selection)
        oofs, vals = {"E3_late": bt}, {"E3_late": bv}
        for fs in FSETS:
            cv = {cn: auc(y, oof_scores(T, fs, logit, mk)) for cn, mk in CANDS.items()}
            best = max(cv, key=cv.get)
            s_oof = oof_scores(T, fs, logit, CANDS[best])
            X, med = design(T, fs, logit); m = CANDS[best]().fit(X, y)
            Xv, _ = design(V, fs, logit, med); sv = score(m, Xv)
            oofs[fs], vals[fs] = s_oof, sv
            r = metrics(y, s_oof, T.Age, yv, sv, V.Age, bt, bv); r["cand"] = best; r["cv"] = cv
            res["stack"][fs] = r
            joblib.dump({"model": m, "medians": med, "columns": list(X.columns), "fset": fs, "logit": logit, "target": tgt,
                         "cand": best, "thr_spec90_trainoof": sens90(y, s_oof)[1]}, f"{O}/stack_{tgt}_{fs.replace('+', '_')}.joblib")
            rows.append(dict(finding=zh, target=tgt, model=f"{fs.replace('E3', 'E3_late')}（{best}）", **{k: v for k, v in r.items() if k != "cv"}))
            print(tgt, fs, best, round(r["oof"], 4), round(r["val"], 4), flush=True)
        bfs = max(res["stack"], key=lambda k: res["stack"][k]["oof"])
        best_name = bfs if res["stack"][bfs]["oof"] > auc(y, bt) else "E3_late"
        res["best_overall"] = best_name
        # 3) sensitivity labels (evaluation only, no retraining)
        lab = pd.read_csv(D + "/labels_trval_split.csv").set_index("study_id")
        res["sensitivity"] = {}
        for sv_col in SENS[tgt]:
            yt2 = lab.loc[T.study_id, sv_col].values; yv2 = lab.loc[V.study_id, sv_col].values
            mt, mv = ~np.isnan(yt2), ~np.isnan(yv2)
            res["sensitivity"][sv_col] = {nm: dict(oof=auc(yt2[mt], oofs[nm][mt]), val=auc(yv2[mv], vals[nm][mv]),
                                                   n_train=int(mt.sum()), n_val=int(mv.sum()))
                                          for nm in ("E3_late", best_name)}
        pd.DataFrame({"study_id": T.study_id, "fold": T.fold, "y": y, **{k: v for k, v in oofs.items()}}).to_csv(f"{O}/oof_train_{tgt}.csv", index=False)
        pd.DataFrame({"study_id": V.study_id, "y": yv, **{k: v for k, v in vals.items()}}).to_csv(f"{O}/val_scores_{tgt}.csv", index=False)
        summ[tgt] = res
    json.dump(summ, open(f"{O}/summary.json", "w"), ensure_ascii=False, indent=1, default=float)
    pd.DataFrame(rows).to_csv(f"{O}/results_all.csv", index=False)


def apply(O, tgt, hu_csvs, fl_csvs, score_csv, out_csv, fs):
    b = joblib.load(f"{O}/stack_{tgt}_{fs}.joblib")
    hu = pd.concat([pd.read_csv(f) for f in hu_csvs.split(",")]).drop_duplicates("study_id")
    fl = pd.concat([pd.read_csv(f) for f in fl_csvs.split(",")]).drop_duplicates("study_id")
    d = pd.read_csv(score_csv).merge(hu, on="study_id", how="left").merge(fl, on="study_id", how="left")
    X, _ = design(d, b["fset"], b["logit"], b["medians"])
    pd.DataFrame({"study_id": d.study_id, "stack": score(b["model"], X[b["columns"]])}).to_csv(out_csv, index=False)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        apply(*sys.argv[2:9])
