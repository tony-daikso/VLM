"""Compare single probe logits (train OOF on GS1 folds + val) against a reference column: AUROC, val 95% CI,
paired stratified bootstrap deltas (OOF and val), age-matched AUROC, subgroups (positives in subgroup vs all negatives:
GB-missing = 3 mm GB < 100 vox, non-hyperdense = GB present & gb_maxblob100 < 2, visible), sens @ 90% spec.
usage: gs_eval_scores.py oof.csv val.csv ref_col col[,col...] out.json   (E3_late from analysis_gallstones is merged as 'E3_late_ref')"""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
I = "/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final"
fo, fv, ref, cols, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5]
lab = pd.read_csv(f"{I}/analysis_gallstones/labels_trval.csv"); lab = lab[lab.split.isin(["train", "val"])][["study_id", "y_gs"]]
lab = lab[lab.y_gs.notna()]
age = pd.read_csv(f"{I}/analysis_absent_gb/labels_trval.csv"); age = age[age.split.isin(["train", "val"])][["study_id", "Age"]]
hu = pd.concat([pd.read_csv(f"{I}/analysis_gallstones/{f}", usecols=["study_id", "gb_vox", "gb_maxblob100"]) for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
e3o = pd.read_csv(f"{I}/analysis_gallstones/oof_train.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "E3_late_ref"})
e3v = pd.read_csv(f"{I}/analysis_gallstones/scores_val_descriptive.csv")[["study_id", "E3_late"]].rename(columns={"E3_late": "E3_late_ref"})
def prep(f, e3):
    d = pd.read_csv(f).merge(lab, on="study_id").merge(age, on="study_id", how="left").merge(hu, on="study_id", how="left").merge(e3, on="study_id", how="left")
    return d.dropna(subset=[ref] + cols).reset_index(drop=True)
T, V = prep(fo, e3o), prep(fv, e3v)
def bidx(y, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
def bd(y, a, b, n=2000):
    rng = np.random.default_rng(0); d = np.array([auc(y[i], b[i]) - auc(y[i], a[i]) for i in (bidx(y, rng) for _ in range(n))])
    return dict(mean=float(d.mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
def ci(y, s, n=2000):
    rng = np.random.default_rng(0); a = [auc(y[i], s[i]) for i in (bidx(y, rng) for _ in range(n))]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
def agem(y, s, ag):
    b = pd.cut(pd.Series(ag), (0, 30, 40, 50, 60, 70, 80, 200), right=False).values; num = den = 0.0
    for g in pd.unique(b[~pd.isna(b)]):
        m = b == g; n1, n0 = y[m].sum(), (1 - y[m]).sum()
        if n1 and n0: num += auc(y[m], s[m]) * n1 * n0; den += n1 * n0
    return num / den
res = {}
for nm, D in (("oof", T), ("val", V)):
    y = D.y_gs.values.astype(int); miss = (D.gb_vox.fillna(0) < 100).values; vis = (D.gb_maxblob100.fillna(0) >= 2).values
    groups = dict(missing=miss, nonvis=~vis & ~miss, vis=vis & ~miss)
    for c in [ref] + cols:
        s = D[c].values; thr = np.quantile(T[c].values[T.y_gs.values == 0], .9)
        r = dict(auroc=float(auc(y, s)), age_matched=float(agem(y, s, D.Age.values)), sens90=float((s[y == 1] > thr).mean()),
                 sub={g: float(auc(y[(y == 0) | m], s[(y == 0) | m])) for g, m in groups.items()})
        if nm == "val": r["ci"] = ci(y, s)
        if c != ref:
            r["d_vs_ref"] = bd(y, D[ref].values, s)
            r["d_sub_vs_ref"] = {g: bd(y[(y == 0) | m], D[ref].values[(y == 0) | m], s[(y == 0) | m], n=1000) for g, m in groups.items()}
        res.setdefault(c, {})[nm] = r
    res.setdefault("_n", {})[nm] = dict(n=len(D), pos=int(y.sum()), **{g: int(y[m].sum()) for g, m in groups.items()})
json.dump(res, open(out, "w"), indent=1)
for c, r in res.items():
    if c == "_n": print(c, r); continue
    for nm in ("oof", "val"):
        x = r[nm]; dd = x.get("d_vs_ref"); print(f"{c:16s} {nm}: {x['auroc']:.4f} {x.get('ci','')} age {x['age_matched']:.4f} s90 {x['sens90']:.3f} sub "
              + " ".join(f"{g}={v:.3f}" for g, v in x["sub"].items()) + (f" | d {dd['mean']:+.4f} [{dd['ci'][0]:+.4f},{dd['ci'][1]:+.4f}]" if dd else "")
              + ("  dsub " + " ".join(f"{g}={v['mean']:+.3f}[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for g, v in x["d_sub_vs_ref"].items()) if "d_sub_vs_ref" in x else ""))
