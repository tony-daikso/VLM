"""main_all v2: mean val AUROC over the 14 original findings with the LATEST formal best per finding.
Same method as ib_summary.py: official baseline = expB E3_late (3-seed mean) on every labelled val study (labels14),
model scores left-joined by study_id (missing -> dropped for that finding only), joint study-level paired bootstrap
(1,000 resamples of val studies, rng seed 0). Adds: previous integrated_best and the new best scored on the same
resamples; age-matched main_all (= mean of ib_core.age_matched_auc, 10-y bins, pair weighted) on the same resamples;
split-label variant (脂肪肝 -> 瀰漫性, 闌尾炎 -> 確定的急性) with CI. val only; test never read.
usage: ib_summary_v2.py <integrated_best root>"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ib_core import auc, age_matched_auc
R = sys.argv[1].rstrip("/") + "/"; OF = os.path.dirname(R.rstrip("/")) + "/"
ORDER = ["hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly", "gallstones", "biliary_ductal_dilation",
         "hydronephrosis", "abdominal_aortic_aneurysm", "atherosclerosis", "osteopenia", "fracture", "bowel_obstruction",
         "appendicitis", "ascites", "free_air"]
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊已切除", "splenomegaly": "脾腫大", "gallstones": "膽結石",
      "biliary_ductal_dilation": "膽管擴張", "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤",
      "atherosclerosis": "動脈粥狀硬化", "osteopenia": "骨質減少", "fracture": "骨折", "bowel_obstruction": "腸阻塞",
      "appendicitis": "闌尾炎", "ascites": "腹水", "free_air": "腹腔游離氣體"}
NEW = {  # finding -> (file, score column, model description)
    "hydronephrosis": (OF + "region_units/hydronephrosis/stack/scores_val.csv", "REF＋全部單元頭", "region_units：REF＋全部單元頭"),
    "free_air": (OF + "region_units/free_air/stack/scores_val.csv", "REF＋全部單元頭", "region_units：REF＋全部單元頭"),
    "ascites": (OF + "region_units/ascites/stack/scores_val.csv", "REF＋U_org", "region_units：REF＋U_org"),
    "appendicitis": (R + "scores_v2/appendicitis_val.csv", "stack", "appendicitis_rlq_unit：REF＋U_rlq_cs"),
    "osteopenia": (OF + "osteopenia_v15/stack_val.csv", "5win+v15", "osteopenia_v15：5 window logit＋1.5 mm（5win+v15）"),
}
B0 = R + "work/base/"
lab = pd.read_csv(B0 + "labels14_trval.csv"); exb = pd.read_csv(B0 + "expB_E3late_val.csv")
age = pd.read_csv(B0 + "age_trval.csv").drop_duplicates("study_id")
assert set(lab.split) <= {"train", "val"}
val_ids = lab[lab.split == "val"].study_id.values; idx = {s: i for i, s in enumerate(val_ids)}
AGE = dict(zip(age.study_id, age.Age))

def load(fd):
    p = pd.read_csv(R + f"scores/{fd}_val.csv")[["study_id", "stack"]].rename(columns={"stack": "prev"})
    if fd in NEW:
        f, c, _ = NEW[fd]; q = pd.read_csv(f)
        if "y" in q.columns: chk = q[["study_id", "y"]]
        else: chk = None
        n = q[["study_id", c]].rename(columns={c: "new"})
    else:
        n = p.rename(columns={"prev": "new"}); chk = None
    l = lab[lab.split == "val"][["study_id", fd]].rename(columns={fd: "y"}); l = l[l.y.notna()]
    m = l.merge(exb[["study_id", fd]].rename(columns={fd: "b"}), on="study_id").merge(p, on="study_id", how="left").merge(n, on="study_id", how="left")
    if chk is not None:
        z = m.merge(chk.rename(columns={"y": "y2"}), on="study_id"); print(fd, "label mismatches vs score file:", int((z.y != z.y2).sum()), "of", len(z))
    m["age"] = m.study_id.map(AGE).astype(float)
    return m

def load_split(key, newfile):
    v = pd.read_csv(R + f"split/{key}_val.csv").rename(columns={"expB_E3_late": "b", "stack": "prev"})
    n = pd.read_csv(newfile)[["study_id", "y", "stack"]].rename(columns={"stack": "new", "y": "y2"})
    m = v[["study_id", "y", "b", "prev"]].merge(n, on="study_id", how="left")
    print(key, "label mismatches:", int((m.y2.notna() & (m.y != m.y2)).sum()), "new missing:", int(m.new.isna().sum()))
    m["age"] = m.study_id.map(AGE).astype(float); return m.drop(columns="y2")

TB = {fd: load(fd) for fd in ORDER}
SP = {"hepatic_steatosis": ("hs_diffuse", load_split("hs_diffuse", R + "split/hs_diffuse_val.csv")),
      "appendicitis": ("ap_def", load_split("ap_def", R + "split_v2/ap_def_val.csv"))}
VERS = ["b", "prev", "new"]

def pos(m):
    p = np.full(len(val_ids), -1); p[[idx[s] for s in m.study_id]] = np.arange(len(m)); return p
POS = {fd: pos(TB[fd]) for fd in TB}; SPOS = {fd: pos(SP[fd][1]) for fd in SP}

def per_finding(m, r=None):
    """returns {ver: (auc, age_auc)} on rows r (bootstrap row indices incl. duplicates)"""
    out = {}
    for v in VERS:
        y = m.y.values; s = m[v].values; a = m.age.values
        if r is not None: y, s, a = y[r], s[r], a[r]
        k = ~np.isnan(s); out[v] = (auc(y[k], s[k]), age_matched_auc(y[k], s[k], a[k]))
    return out

def all_findings(ix=None, split=False):
    res = {}
    for fd in ORDER:
        if split and fd in SP: m, P = SP[fd][1], SPOS[fd]
        else: m, P = TB[fd], POS[fd]
        r = None
        if ix is not None: r = P[ix]; r = r[r >= 0]
        res[fd] = per_finding(m, r)
    return res

def summarize(res):
    return {v: (float(np.nanmean([res[f][v][0] for f in ORDER])), float(np.nanmean([res[f][v][1] for f in ORDER]))) for v in VERS}

pt = all_findings(); pts = all_findings(split=True)
rng = np.random.default_rng(0); BS, BSS, BF = [], [], []
for i in range(1000):
    ix = rng.integers(0, len(val_ids), len(val_ids))
    a = all_findings(ix); BF.append(a); BS.append(summarize(a)); BSS.append(summarize(all_findings(ix, split=True)))
    if i % 100 == 0: print("boot", i, flush=True)

def ci(x): return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
def block(P, BB):
    S = summarize(P); o = {}
    for j, kind in ((0, "auroc"), (1, "age_matched")):
        d = {}
        for v in VERS: d[v] = dict(value=S[v][j], ci=ci([b[v][j] for b in BB]))
        for a_, b_ in (("prev", "b"), ("new", "b"), ("new", "prev")):
            dd = np.array([b[a_][j] - b[b_][j] for b in BB]); d[f"delta_{a_}_vs_{b_}"] = dict(value=S[a_][j] - S[b_][j], ci=ci(dd))
        o[kind] = d
    return o

out = dict(n_findings=14, n_boot=1000, seed=0, method="joint study-level paired bootstrap over val studies (same as ib_summary.py); "
           "b = expB E3_late (3-seed mean) on all labelled val; prev = integrated_best/scores; new = latest formal best; "
           "age_matched = mean over findings of ib_core.age_matched_auc (10-y bins, pair weighted)",
           versions={"b": "expB E3_late", "prev": "integrated_best（2026-11-06 v1）", "new": "最新正式最佳（v2）"},
           new_sources={fd: {"file": NEW[fd][0].replace(OF, ""), "col": NEW[fd][1], "model": NEW[fd][2]} for fd in NEW},
           main_all=block(pt, BS), split_variant=dict(substituted={"hepatic_steatosis": "hs_diffuse", "appendicitis": "ap_def"}, **block(pts, BSS)))
pf = []
for fd in ORDER:
    m = TB[fd]; r = pt[fd]
    dd = np.array([b[fd]["new"][0] - b[fd]["b"][0] for b in BF]); dp = np.array([b[fd]["new"][0] - b[fd]["prev"][0] for b in BF])
    pf.append(dict(finding=fd, zh=ZH[fd], n_val=int(len(m)), n_scored_new=int(m.new.notna().sum()), pos_val=int(m.y.sum()),
                   E3=r["b"][0], prev=r["prev"][0], new=r["new"][0], d_new_vs_E3=r["new"][0] - r["b"][0], d_ci=ci(dd),
                   d_new_vs_prev=r["new"][0] - r["prev"][0], d_prev_ci=ci(dp),
                   age_E3=r["b"][1], age_prev=r["prev"][1], age_new=r["new"][1],
                   model_new=NEW[fd][2] if fd in NEW else "同 integrated_best"))
for fd, (k, m) in SP.items():
    r = pts[fd]; dd = np.array([b[fd]["new"][0] - b[fd]["b"][0] for b in [all_findings(None, True)]])
    pf.append(dict(finding=k, zh={"hs_diffuse": "瀰漫性脂肪肝", "ap_def": "確定的急性闌尾炎"}[k], n_val=int(len(m)), n_scored_new=int(m.new.notna().sum()),
                   pos_val=int(m.y.sum()), E3=r["b"][0], prev=r["prev"][0], new=r["new"][0], d_new_vs_E3=r["new"][0] - r["b"][0],
                   d_new_vs_prev=r["new"][0] - r["prev"][0], age_E3=r["b"][1], age_prev=r["prev"][1], age_new=r["new"][1], split_row=True))
out["per_finding"] = pf
json.dump(out, open(R + "main_all_v2.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("main_all", "split_variant")}, indent=1, ensure_ascii=False))
print(pd.DataFrame(pf)[["zh", "n_val", "n_scored_new", "pos_val", "E3", "prev", "new", "d_new_vs_E3", "age_E3", "age_new"]].round(4).to_string())
