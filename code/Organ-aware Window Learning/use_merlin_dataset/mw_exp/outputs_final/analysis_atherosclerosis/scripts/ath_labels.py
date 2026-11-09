"""Per-study atherosclerosis label/text table for train+val ONLY (split filtered before reading any label/report)."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_atherosclerosis/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    a = r["answer"]; g = a["atherosclerosis"]; det = g.get("detail") or {}
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], ath_cat=g["category"],
        grade=det.get("grade", "na"), aortoiliac=det.get("aortoiliac", "na"), y_ath=b(g["category"]),
        y_aaa=b(a["abdominal_aortic_aneurysm"]["category"]), q=" ".join(g.get("quotes") or []))
rep = json.load(open(R + "merlin_disease14/reports.json"))
VRE = re.compile(r"atheroscl|calcif|plaque|aort|iliac|vascular|arter|vessel|stent|graft", re.I)
VESS = r"(?:vascular|vessel|aort\w*|aorto-?iliac|iliac|arter\w*|mesenteric|celiac|splenic|renal|femoral|coronary)"
pat = {
 "kw_athero": r"atheroscl|atheromatous|atheroma",
 "kw_vasc_calc": VESS + r"\W+(?:\w+\W+){0,4}calcif|calcif\w*\W+(?:\w+\W+){0,5}" + VESS,
 "kw_calc_plaque": r"calcified plaque|calcific plaque|plaque",
 "kw_aortoiliac": r"aorto-?iliac|iliac",
 "kw_coronary": r"coronary",
 "kw_mild": r"\bmild|minimal|minor|scattered|trace|small amount|few|faint",
 "kw_moderate": r"moderate",
 "kw_severe": r"severe|extensive|dense|heavy|heavily|marked|advanced|diffuse|significant",
 "kw_neg_stmt": r"no (?:significant |substantial |appreciable )?(?:atheroscl|vascular calcif|aortic calcif|calcif)",
 "kw_aneurysm": r"aneurysm|ectatic|ectasia|dilat",
 "kw_stent_graft": r"stent|graft|evar|endovascular|bypass",
 "kw_noncalc": r"non-?calcified|soft plaque|mural thromb|thrombus",
}
for sid, d in rows.items():
    rr = rep.get(sid) or {}
    txt = rr.get("report", "") if rr.get("split", d["split"]) in OK else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if VRE.search(s)]
    vtxt = " ".join(sents)
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, vtxt, re.I)))
    # grade words specifically inside atherosclerosis/calcification sentences or quotes
    gtxt = " ".join([s for s in sents if re.search(r"atheroscl|calcif|plaque", s, re.I)]) + " " + d["q"]
    for k in ("mild", "moderate", "severe"):
        d["g_" + k] = int(bool(re.search(pat["kw_" + k], gtxt, re.I)))
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values()).drop(columns=["q"])
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict())
t = df[df.split == "train"]
print("train pos", int(t.y_ath.sum()), "neg", int((t.y_ath == 0).sum()), "nan", int(t.y_ath.isna().sum()))
kw = [c for c in df.columns if c.startswith(("kw_", "g_"))]
print("pos kw:", t[t.y_ath == 1][kw].mean().round(3).to_dict())
print("neg kw:", t[t.y_ath == 0][kw].mean().round(3).to_dict())
print("neg not_mentioned kw:", t[t.ath_cat == "not_mentioned"][kw].mean().round(3).to_dict())
print("grade x y:", pd.crosstab(t.grade, t.y_ath).to_dict())
