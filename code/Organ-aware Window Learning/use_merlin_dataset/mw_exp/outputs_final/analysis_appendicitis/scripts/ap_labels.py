"""Per-study appendicitis label/text table for train+val ONLY.
Split is filtered on study_id (from split.csv) BEFORE any label line is json-parsed or any report is accessed."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_appendicitis/"
OK = {"train", "val"}
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
okids = set(spl[spl.isin(OK)].index)
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
IDRE = re.compile(r'"study_id"\s*:\s*"([^"]+)"')
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    m = IDRE.search(line)
    if not m or m.group(1) not in okids:
        continue
    r = json.loads(line)
    if r.get("split") not in OK: continue
    a = r["answer"]; g = a["appendicitis"]; det = g.get("detail") or {}
    d = dict(study_id=r["study_id"], split=spl[r["study_id"]], ap_cat=g["category"], y_ap=b(g["category"]),
             ap_colith=det.get("appendicolith", "na"), ap_perf=det.get("perforated", "na"),
             ap_quote=" ".join(g.get("quotes") or []))
    for k in ["bowel_obstruction", "ascites", "free_air", "hydronephrosis", "fracture", "hepatic_steatosis", "gallstones", "atherosclerosis"]:
        if k in a: d["y_" + k] = b(a[k]["category"])
    rows[r["study_id"]] = d
rep = json.load(open(R + "merlin_disease14/reports.json"))
rep = {k: v for k, v in rep.items() if k in okids}
APRE = re.compile(r"append|cecum|cecal|ileocec|right lower quadrant|\bRLQ\b|periappend|mucocele|terminal ileum", re.I)
pat = {
 "kw_perf": r"perforat|abscess|phlegmon|extraluminal|ruptur|walled[- ]off",
 "kw_colith": r"appendicolith|fecalith|appendiceal calcul",
 "kw_uncertain": r"\bearly\b|\bmild(?:ly)?\b|possibl|equivocal|borderline|cannot (?:be )?(?:entirely )?exclude|questionabl|suspicious|may represent|suggest|\blikely\b|probabl|developing|indeterminate|correlat",
 "kw_appendectomy": r"appendectomy|surgically absent|status post append|s/p append|appendix (?:has been |is )?removed",
 "kw_notvis": r"appendix (?:is |was )?not (?:well |definitely |clearly |definitively |confidently )?(?:visualized|identified|seen)|non-?visuali|appendix (?:is |was )?not discretely",
 "kw_mucocele": r"mucocele|mucinous|mucocoele|lamn",
 "kw_history": r"history of|\bs/p\b|status post|\bprior\b|previous|\bknown\b|recent(?:ly)?",
 "kw_negphrase": r"no (?:ct |radiographic |definite |convincing )?(?:evidence|findings|signs?) (?:of|to suggest|suggestive of) (?:acute )?appendicitis|no (?:acute )?appendicitis|normal(?:[- ]appearing)? appendix|appendix (?:is |appears )?(?:normal|unremarkable)|without (?:evidence of )?appendicitis|unremarkable appendix|appendicitis is not",
 "kw_stranding": r"strand|inflamm|infiltrat|haz",
 "kw_fluid": r"fluid|collection",
 "kw_dilated": r"dilat|distend|enlarg|thicken|hyperemi|enhanc",
 "kw_tip": r"tip appendicitis",
 "kw_postop": r"post-?operative|\bdrain|percutaneous|surgical",
 "kw_followup": r"resolv|improv|interval|decreas|antibiotic|treated|non-?operative",
 "kw_rlq": r"right lower quadrant|\bRLQ\b",
 "kw_cecal": r"cecum|cecal|ileocec|terminal ileum|typhlitis|ileitis",
 "kw_retrocecal": r"retrocecal|pelvic appendix|pelvis",
}
for sid, d in rows.items():
    rr = rep.get(sid) or {}
    txt = rr.get("report", "") if rr.get("split", d["split"]) in OK else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if APRE.search(s)]
    gtxt = " ".join(sents) + " " + d["ap_quote"]
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, gtxt, re.I)))
    d["kw_append_in_report"] = int(bool(re.search(r"append", txt, re.I)))
    mm = [float(x) * (10 if u.lower() == "cm" else 1) for x, u in re.findall(r"(\d+(?:\.\d+)?)\s*(mm|cm)", d["ap_quote"], re.I)]
    d["ap_mm"] = max(mm) if mm else np.nan
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values()).drop(columns=["ap_quote"])
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict())
for s in ["train", "val"]:
    t = df[df.split == s]; print(s, "cat:", t.ap_cat.value_counts().to_dict())
t = df[df.split == "train"]
print("train pos kw rates:", t[t.y_ap == 1][[c for c in df.columns if c.startswith("kw_")]].mean().round(3).to_dict())
print("train neg kw rates:", t[t.y_ap == 0][[c for c in df.columns if c.startswith("kw_")]].mean().round(3).to_dict())
print("perf x y:", pd.crosstab(t.ap_perf, t.y_ap).to_dict()); print("colith x y:", pd.crosstab(t.ap_colith, t.y_ap).to_dict())
print("ap_mm pos:", t[t.y_ap == 1].ap_mm.describe().round(1).to_dict())
