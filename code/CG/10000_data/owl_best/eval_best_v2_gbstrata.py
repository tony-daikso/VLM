"""best_v2 gallstones: AUROC by 1.5 mm crop centring method (gb = TS 3 mm gallbladder >= 100 vox; gs6_gb / gs6_fossa = GB not
found in the 3 mm mask -> GS6 re-seg / fossa; absent = no crop, FT head imputed with the train median). Within-stratum AUROC
(positives and negatives of the stratum), definition B; C+ main (formal) and C- (nc, paired set not required). Bootstrap 1000 seed 0."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_owl_cg10k as ev, eval_best_cg10k as eb, eval_best_v2 as v
from eval_owl_cg10k import auc
cm = pd.read_csv(v.V2 / "crop_method_gallstones.csv")
rng = np.random.default_rng(ev.SEED); R = []
base, _ = eb.load("main")
for s, vs in (("main", "main"), ("nc", "nc")):
    if s == "main":
        d = base.merge(v.v2("main"), on="EXAMID")
    else:
        old = pd.read_csv(v.NR / "probs_best_nc.csv")[["study_id", "gallstones"]].rename(columns={"study_id": "EXAMID", "gallstones": "gallstones__best"})
        e3 = pd.read_csv(ev.OUT / "probs_nc.csv"); sc = [c for c in e3.columns if c.startswith("gallstones__seed")]
        if sc: e3["gallstones"] = e3[sc].mean(axis=1)
        keep = [c for c in base.columns if not (c in ev.DIS or c.endswith("__best"))]
        d = base[keep].merge(e3[["study_id", "gallstones"]].rename(columns={"study_id": "EXAMID"}), on="EXAMID").merge(old, on="EXAMID").merge(v.v2("nc"), on="EXAMID")
    d = d.merge(cm[cm.set == s][["study_id", "crop_method"]].rename(columns={"study_id": "EXAMID"}), on="EXAMID", how="left")
    d["stratum"] = d.crop_method.map(lambda m: "gb" if m == "gb" else ("absent" if m == "absent" else "gs6 (GB 未見於 3 mm)"))
    y_all = eb.labels(d, "gallstones", "B")
    for st in ["全部", "gb", "gs6 (GB 未見於 3 mm)", "absent"]:
        m = ~np.isnan(y_all) & ((d.stratum == st).values if st != "全部" else True)
        yy = y_all[m].astype(bool)
        if yy.sum() < 5 or (~yy).sum() < 5:
            R.append(dict(phase=s, stratum=st, N=int(m.sum()), n_pos=int(yy.sum()))); continue
        idx = rng.integers(0, len(yy), size=(ev.NBOOT, len(yy))); r = dict(phase=s, stratum=st, N=len(yy), n_pos=int(yy.sum()), n_neg=int((~yy).sum()))
        S = {"new": d.gallstones__v2.values[m], "old": d.gallstones__best.values[m], "E3": d.gallstones.values[m]}
        B = {k: np.array([auc(yy[i], x[i]) for i in idx]) for k, x in S.items()}
        for k, x in S.items():
            r[k] = auc(yy, x); r[k + "_ci"] = f"{np.nanpercentile(B[k], 2.5):.3f}–{np.nanpercentile(B[k], 97.5):.3f}"
        dd = B["new"] - B["old"]; r["d_new_old"] = r["new"] - r["old"]; r["d_new_old_ci"] = f"{np.nanpercentile(dd, 2.5):+.3f}, {np.nanpercentile(dd, 97.5):+.3f}"
        R.append(r)
R = pd.DataFrame(R); R.to_csv(v.V2 / "eval_best_v2_gb_strata.csv", index=False); print(R.round(3).to_string())
