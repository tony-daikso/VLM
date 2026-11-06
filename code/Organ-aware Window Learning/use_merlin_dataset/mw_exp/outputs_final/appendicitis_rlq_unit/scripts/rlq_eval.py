"""RLQ crop unit: stacking + evaluation (train 5-fold OOF for selection, val for reporting). CPU.
Stack = StandardScaler + LogisticRegression(C=0.1, class_weight=balanced) on OOF probe scores (+ HU features),
same folds as the probes (analysis_appendicitis folds); val = stack fitted on all train OOF rows applied to the
full-train->val probe scores. No age/sex as inputs.
Metrics per target x model: AUROC [stratified bootstrap 95% CI, 1000], age+sex matched AUROC (|age diff|<=2 y, same
sex, up to 5 negatives per positive, 20 draws), AUROC in age<30 / >=30, sensitivity at 90% specificity, paired
bootstrap delta vs E3_late and vs the previous best stack (REF = E3_late + E3cs + RLQ HU local).
Usage: rlq_eval.py UNIT_DIR   (reads UNIT_DIR/out/{oof_train,val_scores}.csv, UNIT_DIR/crops/*.npz hu_feats)"""
import sys, os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/"
U = sys.argv[1]; O = U + "/out/"
lab = pd.read_csv(R + "appendicitis_split/labels_trval_split.csv")
assert set(lab.split) <= {"train", "val"}
oof = pd.read_csv(O + "oof_train.csv"); val = pd.read_csv(O + "val_scores.csv")
hu = pd.concat([pd.read_csv(R + "analysis_appendicitis/hu_a6.csv"), pd.read_csv(R + "analysis_appendicitis/hu_ct.csv")]).drop_duplicates("study_id")
LOCAL = [c for c in hu.columns if c.startswith("dlr_") or c.startswith("dglob_")] + \
        [f"rlq_r{r}_{f}" for r in (15, 30) for f in ("fat_med", "fat_p75", "strand_frac", "fluid_frac", "soft_frac", "fat_frac")] + ["rlq_wall_p50", "rlq_wall_p90"]
hu = hu[["study_id"] + LOCAL]
# crop-level fat-stranding channel (4 regions x strand/fluid/fat_med)
fc = O + "crop_hu_feats.csv"
if not os.path.exists(fc):
    rows = []
    for s in pd.concat([oof.study_id, val.study_id]):
        f = f"{U}/crops/{s}.npz"; d = {"study_id": s}
        if os.path.exists(f):
            q = np.load(f)
            if bool(q["present"]):
                hf = q["hu_feats"]
                for k, rn in enumerate(("ball30", "peri", "cecti", "rlq50")):
                    for j, fn in enumerate(("strand", "fluid", "fatmed")):
                        d[f"cr_{rn}_{fn}"] = float(hf[k, j])
                d["cr_method"] = str(q["method"])
        rows.append(d)
    pd.DataFrame(rows).to_csv(fc, index=False)
cf = pd.read_csv(fc); CROP = [c for c in cf.columns if c.startswith("cr_") and c != "cr_method"]
meta = pd.read_csv("/home/advantech/Desktop/Tony/VLM/merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})[["study_id", "Age", "Gender"]]
T = oof.merge(lab, on="study_id", how="left").merge(hu, on="study_id", how="left").merge(cf, on="study_id", how="left").merge(meta, on="study_id", how="left")
V = val.merge(lab, on="study_id", how="left").merge(hu, on="study_id", how="left").merge(cf, on="study_id", how="left").merge(meta, on="study_id", how="left")
assert (T.split == "train").all() and (V.split == "val").all()
print("method counts (train+val):", pd.concat([T.cr_method, V.cr_method]).value_counts(dropna=False).to_dict(), flush=True)
TARGETS = {"y_ap": "闌尾炎（原合併定義）", "y_ap_def": "確定的急性闌尾炎", "y_ap_unc": "可疑／治療中／追蹤／術後"}


def models(tg):
    s = lambda v: f"{tg}__{v}"
    return {
        "E3_late（大腸，參考）": ([s("E3_late")], []),
        "E3cs（大腸＋小腸）": ([s("E3cs")], []),
        "RLQ 單元單獨（U_rlq）": ([s("U_rlq")], []),
        "RLQ 單元＋大腸＋小腸（U_rlq_cs）": ([s("U_rlq_cs")], []),
        "REF 堆疊：E3_late＋E3cs＋RLQ HU": ([s("E3_late"), s("E3cs")], LOCAL),
        "E3_late＋E3cs＋U_rlq（無 HU）": ([s("E3_late"), s("E3cs"), s("U_rlq")], []),
        "REF＋U_rlq": ([s("E3_late"), s("E3cs"), s("U_rlq")], LOCAL),
        "REF＋U_rlq_cs": ([s("E3_late"), s("E3cs"), s("U_rlq_cs")], LOCAL),
        "完整堆疊：REF＋U_rlq＋U_rlq_cs＋crop 浸潤通道": ([s("E3_late"), s("E3cs"), s("U_rlq"), s("U_rlq_cs")], LOCAL + CROP),
    }


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))


def stack(tg, sc, feats, Tm, Vm, y):
    cols = sc + feats
    if len(cols) == 1 and not feats:
        return Tm[cols[0]].values, Vm[cols[0]].values
    X = Tm[cols]; med = X.median(); X = X.fillna(med); XV = Vm[cols].fillna(med)
    p = np.zeros(len(Tm))
    for k in range(5):
        a, b = (Tm.fold != k).values, (Tm.fold == k).values
        p[b] = lr().fit(X[a], y[a]).decision_function(X[b])
    return p, lr().fit(X, y).decision_function(XV)


def ci(y, s, B=1000, seed=0):
    rng = np.random.default_rng(seed); pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(B):
        i = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]; v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def dci(y, s1, s2, B=1000, seed=0):
    rng = np.random.default_rng(seed); pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(B):
        i = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]; v.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return roc_auc_score(y, s1) - roc_auc_score(y, s2), np.percentile(v, 2.5), np.percentile(v, 97.5)


def matched(D, y, s, reps=20, k=5):
    Ag = D.Age.values.astype(float); G = D.Gender.values; pos = np.where((y == 1) & ~np.isnan(Ag))[0]
    ni = np.where((y == 0) & ~np.isnan(Ag))[0]; v = []
    for r in range(reps):
        rr = np.random.default_rng(r); sel = []
        for i in pos:
            c = ni[(np.abs(Ag[ni] - Ag[i]) <= 2) & (G[ni] == G[i])]
            if len(c): sel += list(rr.choice(c, min(k, len(c)), replace=False)) + [i]
        sel = np.array(sel); v.append(roc_auc_score(y[sel], s[sel]))
    return float(np.mean(v))


def sub_auc(y, s, m):
    return roc_auc_score(y[m], s[m]) if (y[m] == 1).sum() >= 3 and (y[m] == 0).sum() >= 3 else np.nan


def sens90(y, s):
    thr = np.quantile(s[y == 0], 0.90); return float((s[y == 1] > thr).mean())


rows = []
for tg, tname in TARGETS.items():
    Tm = T[T[tg].notna() & T[f"{tg}__E3_late"].notna()].reset_index(drop=True)
    Vm = V[V[tg].notna() & V[f"{tg}__E3_late"].notna()].reset_index(drop=True)
    y, yv = Tm[tg].values.astype(int), Vm[tg].values.astype(int)
    S = {}
    for mn, (sc, feats) in models(tg).items():
        if not all(c in Tm.columns for c in sc):
            continue
        S[mn] = stack(tg, sc, feats, Tm, Vm, y)
    base = S["E3_late（大腸，參考）"]; refk = "REF 堆疊：E3_late＋E3cs＋RLQ HU"; ref = S[refk]
    for mn, (p, pv) in S.items():
        r = dict(target=tg, finding=tname, model=mn, n_pos_train=int(y.sum()), n_pos_val=int(yv.sum()))
        for sp, D, yy, ss in (("oof", Tm, y, p), ("val", Vm, yv, pv)):
            lo, hi = ci(yy, ss)
            age = D.Age.values.astype(float)
            r.update({f"{sp}_auroc": roc_auc_score(yy, ss), f"{sp}_lo": lo, f"{sp}_hi": hi,
                      f"{sp}_agematched": matched(D, yy, ss),
                      f"{sp}_auc_lt30": sub_auc(yy, ss, age < 30), f"{sp}_auc_ge30": sub_auc(yy, ss, age >= 30),
                      f"{sp}_sens90": sens90(yy, ss)})
            b = base[0] if sp == "oof" else base[1]; rf = ref[0] if sp == "oof" else ref[1]
            d, dl, dh = dci(yy, ss, b); r.update({f"{sp}_d_vs_E3": d, f"{sp}_d_vs_E3_lo": dl, f"{sp}_d_vs_E3_hi": dh})
            d, dl, dh = dci(yy, ss, rf); r.update({f"{sp}_d_vs_REF": d, f"{sp}_d_vs_REF_lo": dl, f"{sp}_d_vs_REF_hi": dh})
        rows.append(r)
        print(f"{tg:9s} {mn:40s} OOF {r['oof_auroc']:.3f} [{r['oof_lo']:.3f},{r['oof_hi']:.3f}] am {r['oof_agematched']:.3f} "
              f"| val {r['val_auroc']:.3f} [{r['val_lo']:.3f},{r['val_hi']:.3f}] am {r['val_agematched']:.3f} "
              f"<30 {r['val_auc_lt30']:.3f} >=30 {r['val_auc_ge30']:.3f} s90 {r['val_sens90']:.3f} "
              f"dREF val {r['val_d_vs_REF']:+.3f} [{r['val_d_vs_REF_lo']:+.3f},{r['val_d_vs_REF_hi']:+.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(O + "results.csv", index=False)
# per-method coverage / sanity: E3_late OOF by crop method (combined label)
Tm = T[T.y_ap.notna()]
print("crop method among train positives:", Tm[Tm.y_ap == 1].cr_method.value_counts(dropna=False).to_dict())
print("EVAL_DONE")
