"""Integrated table + main_all (mean of the 14 original findings) with a joint study-level paired bootstrap on val.
usage: ib_summary.py <ib_root>   -> integrated_per_finding.csv, main_all.json (report.md is written separately)
Optional extra rows for the user-approved split labels: <ib_root>/split/<key>_val.csv with columns
study_id, y, expB_E3_late, stack (+ <key>.json with display metadata)."""
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ib_core import auc

R = sys.argv[1].rstrip("/") + "/"
ORDER = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "gallstones", "biliary_ductal_dilation",
         "hydronephrosis", "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "fracture", "bowel_obstruction",
         "appendicitis", "ascites", "free_air"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊已切除", "splenomegaly": "脾腫大", "gallstones": "膽結石",
      "biliary_ductal_dilation": "膽管擴張", "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤",
      "atherosclerosis": "動脈粥狀硬化", "osteopenia": "骨質減少", "fracture": "骨折", "bowel_obstruction": "腸阻塞",
      "appendicitis": "闌尾炎", "ascites": "腹水", "free_air": "腹腔游離氣體"}
lab = pd.read_csv(R + "base/labels14_trval.csv"); exb = pd.read_csv(R + "base/expB_E3late_val.csv")
val_ids = lab[lab.split == "val"].study_id.values; idx = {s: i for i, s in enumerate(val_ids)}

rows = []
for fd in ORDER:
    p = R + f"results/{fd}.json"
    if not os.path.exists(p): print("missing", fd); continue
    r = json.load(open(p))
    rows.append(dict(finding=fd, 疾病=ZH[fd], 模型=r.get("featset"), 候選=r.get("cand"), n_val=r["n_val"], pos_val=r["pos_val"],
                     train_OOF=r["oof"], train_OOF_E3=r["oof_E3"], dOOF=r["d_oof"]["mean"], dOOF_lo=r["d_oof"]["ci"][0], dOOF_hi=r["d_oof"]["ci"][1],
                     val=r["val"], val_lo=r["val_ci"][0], val_hi=r["val_ci"][1], val_E3_expB=r["val_E3_expB"],
                     d=r["d_val_expB"]["mean"], d_lo=r["d_val_expB"]["ci"][0], d_hi=r["d_val_expB"]["ci"][1],
                     val_age=r["val_age"], val_age_E3=r["val_age_E3"], d_age=r["d_val_age"]["mean"], d_age_lo=r["d_val_age"]["ci"][0], d_age_hi=r["d_val_age"]["ci"][1],
                     spearman=r["spearman_neg"], spearman_E3=r["spearman_neg_E3"], sens90=r["sens90_val"], sens90_E3=r["sens90_val_E3"],
                     sens90_oofthr=r.get("sens90_val_oofthr"), oof_age=r["oof_age"], oof_age_E3=r["oof_age_E3"],
                     val_E3_rerun=r.get("val_E3_rerun"), flag_age_drop=r["flag_age_drop"]))
D = pd.DataFrame(rows); D.to_csv(R + "integrated_per_finding.csv", index=False)

# ---- main_all: joint bootstrap over val studies
def table(fd):
    s = pd.read_csv(R + f"scores/{fd}_val.csv")
    l = lab[lab.split == "val"][["study_id", fd]].rename(columns={fd: "y"}); l = l[l.y.notna()]
    b = exb[["study_id", fd]].rename(columns={fd: "b"})
    full = l.merge(b, on="study_id")                           # official baseline on every labelled val study
    m = full.merge(s[["study_id", "stack"]], on="study_id", how="left")
    return m

TB = {fd: table(fd) for fd in D.finding}
def arrays(m, col):
    pos = np.full(len(val_ids), -1); pos[[idx[s] for s in m.study_id]] = np.arange(len(m)); return pos
POS = {fd: arrays(TB[fd], None) for fd in TB}

def main_all(sel, ix=None, which="stack"):
    a = []
    for fd in sel:
        m = TB[fd]; y = m.y.values; s = m[which].values if which == "stack" else m.b.values
        if which == "stack": s = np.where(np.isnan(s), np.nan, s)
        if ix is None: a.append(auc(y, s)); continue
        r = POS[fd][ix]; r = r[r >= 0]; a.append(auc(y[r], s[r]))
    return float(np.nanmean(a))

sel = [f for f in ORDER if f in TB]
before = main_all(sel, which="b"); after = main_all(sel)
rng = np.random.default_rng(0); B = []
for _ in range(1000):
    ix = rng.integers(0, len(val_ids), len(val_ids)); B.append((main_all(sel, ix, "b"), main_all(sel, ix)))
B = np.array(B); d = B[:, 1] - B[:, 0]
res = dict(n_findings=len(sel), before=before, before_ci=np.percentile(B[:, 0], [2.5, 97.5]).tolist(), after=after,
           after_ci=np.percentile(B[:, 1], [2.5, 97.5]).tolist(), delta=after - before, delta_ci=np.percentile(d, [2.5, 97.5]).tolist(),
           note="before = expB E3_late (3-seed mean) on all labelled val studies; after = integrated best (fracture stack covers 4,946/4,961 val; "
                "missing studies are dropped for that finding only)")
# ---- split substitution
spl = {}
for p in sorted(glob.glob(R + "split/*.json")):
    k = os.path.basename(p)[:-5]; meta = json.load(open(p)); v = pd.read_csv(R + f"split/{k}_val.csv"); spl[k] = (meta, v)
if spl:
    subs = {}
    for k, (meta, v) in spl.items():
        if meta.get("substitutes") and meta.get("primary"): subs[meta["substitutes"]] = (k, v)
    a_b, a_s = [], []
    for fd in sel:
        if fd in subs:
            k, v = subs[fd]; a_b.append(auc(v.y, v.expB_E3_late)); a_s.append(auc(v.y, v["stack"]))
        else:
            m = TB[fd]; a_b.append(auc(m.y, m.b)); a_s.append(auc(m.y, m["stack"]))
    res["split_substituted"] = dict(before=float(np.mean(a_b)), after=float(np.mean(a_s)), substituted={f: subs[f][0] for f in subs})
json.dump(res, open(R + "main_all.json", "w"), indent=1)
print(D[["疾病", "模型", "候選", "train_OOF", "val", "val_E3_expB", "d", "d_lo", "d_hi", "val_age", "val_age_E3", "spearman", "spearman_E3", "sens90", "sens90_E3"]].round(3).to_string())
print(json.dumps(res, indent=1))
