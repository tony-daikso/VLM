"""(1) train: thoracic-level / FOV-edge subgroups; (2) VAL descriptive only (no tuning): same subgroup AUROCs."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from textparse import *
OUT = Path(__file__).parent
LV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
P = pd.read_pickle(OUT / "train_pos_annot.pkl"); N = pd.read_pickle(OUT / "train_neg_annot.pkl")
ns = N.bone_ens.values
pa = lambda s: float(((np.asarray(s)[:, None] > ns[None]).mean(1)).mean())
def top_present(r):
    pres = [l for l in LV if pd.notna(r.get(f"{l}_h_post"))]
    return pres[0] if pres else None
P["top_lv"] = P.apply(top_present, axis=1)
out = {}
m_th = P.lv_cov.map(lambda l: len(l) > 0 and all(LV.index(x) <= 2 for x in l))
m_edge = P.apply(lambda r: r.top_lv is not None and r.top_lv in r.lv_cov and r.top_lv != "T8", axis=1)
for nm, m in [("報告椎節只在T8–T10", m_th), ("報告椎節=FOV最上緣(截切)", m_edge), ("其他", ~m_th & ~m_edge)]:
    out[f"train_{nm}"] = dict(n=int(m.sum()), auc=pa(P.bone_ens[m]))
# VAL descriptive
d = pd.read_pickle(OUT / "table_trval.pkl"); d = d[d.split == "val"]
d["ypos"] = d["cat"].isin(["yes", "probably_yes"]) & (d.comp == "yes"); d["yneg"] = d["cat"].isin(["no", "probably_no", "not_mentioned"])
d = d[(d.ypos | d.yneg) & d.bone_ens.notna()]
vp, vn = d[d.ypos], d[d.yneg]
vs = vn.bone_ens.values
vpa = lambda s: float(((np.asarray(s)[:, None] > vs[None]).mean(1)).mean()) if len(s) else np.nan
qf = vp.quotes.fillna("").map(flags).apply(pd.Series)
mild = (qf["mild"] & ~qf["moderate_severe"]).values
msk = vn.report.map(msk_section)
out["val_all"] = dict(n_pos=len(vp), n_neg=len(vn), auc=roc_auc_score(np.r_[np.ones(len(vp)), np.zeros(len(vn))], np.r_[vp.bone_ens, vn.bone_ens]))
out["val_輕度"] = dict(n=int(mild.sum()), auc=vpa(vp.bone_ens[mild])); out["val_非輕度"] = dict(n=int((~mild).sum()), auc=vpa(vp.bone_ens[~mild]))
out["val_pos<50"] = dict(n=int((vp.Age < 50).sum()), auc=vpa(vp.bone_ens[vp.Age < 50]))
nn_ = 1 - np.array([((vp.bone_ens.values[:, None] > vn.bone_ens.values[None])[:, i]).mean() for i in range(len(vn))])
for nm, m in [("neg≥75", (vn.Age >= 75).values), ("neg_內固定/側彎", msk.str.contains(r"scolio|kyphos|spinal fusion|posterior fusion|pedicle screw|spinal hardware|laminectomy|instrumentation|fixation rod", case=False).values),
              ("neg_骨質疏鬆字眼", vn.report.fillna("").str.contains(r"osteopeni|osteopor|demineraliz", case=False).values)]:
    out[f"val_{nm}"] = dict(n=int(m.sum()), auc=float(1 - nn_[m].mean()))
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(OUT / "level_and_val_descriptive.json", "w"), indent=1, ensure_ascii=False)
