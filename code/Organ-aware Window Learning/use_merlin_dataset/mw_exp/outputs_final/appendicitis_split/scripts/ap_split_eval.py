"""闌尾炎拆分：RLQ HU stacking (LR vs GBM chosen on TRAIN 5-fold OOF) + metrics table.
Selection uses TRAIN OOF only; val is reported with stratified bootstrap CIs and paired bootstrap deltas.
Metrics: AUROC, age x sex matched AUROC (1:5, +-2 y, same sex, 20 reps), AUROC within age <30 and >=30,
sensitivity at 90% specificity. Masking: rows with NaN target are excluded. Test is never loaded.
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
A0 = R + "analysis_appendicitis/"; D = R + "appendicitis_split/"; O = D + "work/"
rng = np.random.default_rng(0)
lab = pd.read_csv(D + "labels_trval_split.csv"); assert set(lab.split) <= {"train", "val"}
hu = pd.concat([pd.read_csv(A0 + "hu_a6.csv"), pd.read_csv(A0 + "hu_ct.csv")]).drop_duplicates("study_id")
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(lab.study_id))][["study_id", "Age", "Gender"]]
LOCAL = [c for c in hu.columns if c.startswith("dlr_") or c.startswith("dglob_")] + \
        [f"rlq_r{r}_{f}" for r in (15, 30) for f in ("fat_med", "fat_p75", "strand_frac", "fluid_frac", "soft_frac", "fat_frac")] + ["rlq_wall_p50", "rlq_wall_p90"]
TGT_NAME = {"y_ap_def": "確定的急性闌尾炎", "y_ap_unc": "可疑／治療中／追蹤／術後", "y_ap": "原合併闌尾炎（參考）"}


def auc(y, s):
    m = ~np.isnan(s)
    return roc_auc_score(y[m], s[m]) if len(set(y[m])) == 2 else np.nan


def ci(y, s, B=1000):
    p, n = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[s[rng.choice(p, len(p))], s[rng.choice(n, len(n))]]) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])


def dci(y, s1, s2, B=1000):
    p, n = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(B):
        i = np.r_[rng.choice(p, len(p)), rng.choice(n, len(n))]
        v.append(roc_auc_score(y[i], s2[i]) - roc_auc_score(y[i], s1[i]))
    return np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)


def matched(y, s, Ag, G, reps=20, k=5):
    v = []; ni = np.where((y == 0) & ~np.isnan(Ag))[0]; pi = np.where((y == 1) & ~np.isnan(Ag))[0]
    for r in range(reps):
        rr = np.random.default_rng(r); sel = []
        for i in pi:
            c = ni[(np.abs(Ag[ni] - Ag[i]) <= 2) & (G[ni] == G[i])]
            if len(c): sel += list(rr.choice(c, min(k, len(c)), replace=False)) + [i]
        sel = np.array(sel); v.append(roc_auc_score(y[sel], s[sel]))
    return float(np.mean(v))


def sens90(y, s):
    return float((s[y == 1] >= np.quantile(s[y == 0], 0.90)).mean())


def metrics(y, s, Ag, G, withci=True):
    d = dict(n_pos=int(y.sum()), n_neg=int((y == 0).sum()), auroc=auc(y, s))
    if withci:
        d["ci_lo"], d["ci_hi"] = ci(y, s)
    d["age_matched"] = matched(y, s, Ag, G)
    for nm, m in [("auc_lt30", Ag < 30), ("auc_ge30", Ag >= 30)]:
        d[nm] = auc(y[m], s[m]); d["npos_" + nm[4:]] = int(y[m].sum())
    d["sens90"] = sens90(y, s); d["spearman_age_neg"] = pd.Series(s[y == 0]).corr(pd.Series(Ag[y == 0]), method="spearman")
    return d


def make_X(df, med):
    X = df[LOCAL].copy().fillna(med); X["rlq_missing"] = df.rlq_r15_n.isna().astype(int).values
    return X


STACKERS = {
    "LR_C0.03": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=5000, class_weight="balanced")),
    "LR_C0.1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")),
    "LR_C0.3": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")),
    "GBM_d2": lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, class_weight="balanced", random_state=0),
    "GBM_d3": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=40, l2_regularization=1.0, class_weight="balanced", random_state=0),
}


def score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


rows, sel_rows, sub_rows = [], [], []
for tgt in ["y_ap_def", "y_ap_unc", "y_ap"]:
    oof = pd.read_csv(O + f"oof_{tgt}.csv"); val = pd.read_csv(O + f"val_{tgt}.csv")
    if tgt == "y_ap":
        o0 = pd.read_csv(A0 + "oof_train.csv")[["study_id", "E3_late", "E3_late_colon_sb"]]
        v0 = pd.read_csv(A0 + "scores_val_descriptive.csv")[["study_id", "E3_late", "E3_late_colon_sb"]]
        oof = oof.merge(o0, on="study_id"); val = val.merge(v0, on="study_id")
        z = np.load(R + "results/expB/E3_late.npz", allow_pickle=True); fi = list(z["findings"]).index("appendicitis")
        pv = pd.Series(z["prob"].mean(0)[:, fi], index=z["ids"]); sp = pd.Series(z["split"], index=z["ids"])
        val["expB_E3_late"] = val.study_id.map(pv[sp == "val"])
    T = oof.merge(lab[["study_id", "ap_group"]], on="study_id").merge(meta, on="study_id", how="left").merge(hu, on="study_id", how="left")
    V = val.merge(lab[["study_id", "ap_group"]], on="study_id").merge(meta, on="study_id", how="left").merge(hu, on="study_id", how="left")
    T = T[T[tgt].notna()].reset_index(drop=True); V = V[V[tgt].notna()].reset_index(drop=True)
    yt = T[tgt].values.astype(int); yv = V[tgt].values.astype(int)
    At, Gt, Av, Gv = T.Age.values.astype(float), T.Gender.values, V.Age.values.astype(float), V.Gender.values
    med = T[LOCAL].median(); XT, XV = make_X(T, med), make_X(V, med)
    probes = [c for c in ["E3_late", "E3_late_colon_sb", "E3cs_hn2", "E3cs_hn3", "E3cs_trap", "E3cs_trap_hn2", "E3cs_soft05"] if c in T]
    # ---- de-aging selection on TRAIN OOF: score = mean(AUROC, age-matched AUROC) among colon+sb probes
    deage = {}
    for c in [p for p in probes if p.startswith("E3cs") or p == "E3_late_colon_sb"]:
        if c == "E3cs_soft05": continue
        a1 = auc(yt, T[c].values); a2 = matched(yt, T[c].values, At, Gt); deage[c] = (a1, a2, (a1 + a2) / 2)
        sel_rows.append(dict(target=tgt, kind="deage_probe", model=c, train_auroc=a1, train_age_matched=a2, criterion=(a1 + a2) / 2))
    best_probe = max(deage, key=lambda c: deage[c][2])
    # ---- HU stacking candidates (inputs x stacker), selected on TRAIN OOF AUROC (age-matched reported)
    inputs = {"E3+E3cs+HU": ["E3_late", "E3_late_colon_sb"], "E3+HU": ["E3_late"]}
    if best_probe != "E3_late_colon_sb":
        inputs["E3+" + best_probe + "+HU"] = ["E3_late", best_probe]
    stk_oof, stk_val = {}, {}
    for iname, cols in inputs.items():
        for sname, mk in STACKERS.items():
            Xt = pd.concat([XT, T[cols]], axis=1); p = np.zeros(len(T))
            for k in range(5):
                trm, tem = (T.fold != k).values, (T.fold == k).values
                p[tem] = score(mk().fit(Xt[trm], yt[trm]), Xt[tem])
            key = f"stack[{iname}|{sname}]"; stk_oof[key] = p
            a1 = auc(yt, p); a2 = matched(yt, p, At, Gt)
            sel_rows.append(dict(target=tgt, kind="stack", model=key, train_auroc=a1, train_age_matched=a2, criterion=a1))
            stk_val[key] = score(mk().fit(Xt, yt), pd.concat([XV, V[cols]], axis=1))
    best_stack = max(stk_oof, key=lambda k: auc(yt, stk_oof[k]))
    best_lr = max([k for k in stk_oof if "|LR" in k], key=lambda k: auc(yt, stk_oof[k]))
    best_gbm = max([k for k in stk_oof if "|GBM" in k], key=lambda k: auc(yt, stk_oof[k]))
    # ---- report rows
    show = {p: (T[p].values, V[p].values) for p in probes}
    for k in dict.fromkeys([best_lr, best_gbm, best_stack]):
        show[k] = (stk_oof[k], stk_val[k])
    show["age_only"] = (-np.nan_to_num(At, nan=55.0), -np.nan_to_num(Av, nan=55.0))
    if tgt == "y_ap":
        show["expB_E3_late (val ref)"] = (np.full(len(T), np.nan), V["expB_E3_late"].values)
    ref_v = V["E3_late"].values
    for name, (st, sv) in show.items():
        r = dict(target=tgt, target_zh=TGT_NAME[tgt], model=name,
                 selected=("best_deage_probe" if name == best_probe else "best_stack" if name == best_stack else ""))
        if not np.isnan(st).all():
            mt = metrics(yt, st, At, Gt, withci=True); r.update({"tr_" + k: v for k, v in mt.items()})
        mv = metrics(yv, sv, Av, Gv, withci=True); r.update({"va_" + k: v for k, v in mv.items()})
        if name != "E3_late" and not name.startswith("age"):
            d, lo, hi = dci(yv, ref_v, sv); r.update(va_delta_vs_E3=d, va_delta_lo=lo, va_delta_hi=hi)
            if not np.isnan(st).all():
                d, lo, hi = dci(yt, T["E3_late"].values, st); r.update(tr_delta_vs_E3=d, tr_delta_lo=lo, tr_delta_hi=hi)
        rows.append(r); print(tgt, name, round(r.get("tr_auroc", np.nan), 4), round(r["va_auroc"], 4), flush=True)
    # ---- combined-label models scored against the split labels (does the old head already rank definite cases well?)
    if tgt == "y_ap":
        for sub in ["y_ap_def", "y_ap_unc"]:
            s2 = V.merge(lab[["study_id", sub]], on="study_id", suffixes=("", "_l"))
            m = s2[sub].notna().values; yy = s2[sub].values[m].astype(int)
            sub_rows.append(dict(model="E3_late trained on combined label", eval_on=sub, va_auroc=auc(yy, V.E3_late.values[m]),
                                 va_age_matched=matched(yy, V.E3_late.values[m], Av[m], Gv[m])))
pd.DataFrame(rows).to_csv(D + "results_split.csv", index=False)
pd.DataFrame(sel_rows).to_csv(D + "selection_train_oof.csv", index=False)
pd.DataFrame(sub_rows).to_csv(D + "combined_head_on_split_labels.csv", index=False)
print(pd.DataFrame(sel_rows).round(4).to_string()); print(pd.DataFrame(sub_rows).round(4).to_string())
print("EVAL_DONE")
