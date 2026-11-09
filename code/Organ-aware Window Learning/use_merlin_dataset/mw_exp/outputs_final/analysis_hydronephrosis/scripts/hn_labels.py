"""Per-study hydronephrosis label/text table for train+val ONLY (split filtered before reading any label/report).
Output has keyword flags only (no report text)."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_hydronephrosis/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    a = r["answer"]; h = a["hydronephrosis"]; dt = h.get("detail") or {}
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], hn_cat=h["category"], y_hn=b(h["category"]),
        hn_grade=dt.get("grade", "not_stated"), hn_side=dt.get("side", "not_stated"),
        y_ascites=b(a["ascites"]["category"]), y_obstr=b(a["bowel_obstruction"]["category"]),
        hn_quote=" ".join(h.get("quotes") or []))
rep = json.load(open(R + "merlin_disease14/reports.json"))
KRE = re.compile(r"kidney|renal|ureter|hydro|pelvi|calic|calyc|collecting system|bladder|nephro|\bstents?\b|urinary|ureteral|transplant|pyel", re.I)
NEGCUE = re.compile(r"\b(?:no|without|negative for|not|nor|resolved|resolution of|interval resolution)\b", re.I)
def hit(p, t):
    for m in re.finditer(p, t, re.I):
        pre = t[max(0, m.start() - 60):m.start()]
        pre = re.split(r"[.;:]", pre)[-1]
        if not NEGCUE.search(pre):
            return 1
    return 0
pat = {
 "kw_mild": r"\bmild|minimal|slight|trace|borderline",
 "kw_moderate": r"moderate",
 "kw_severe": r"severe|marked|massive|gross",
 "kw_fullness": r"fullness|prominen|prominent",
 "kw_pelviectasis": r"pelviectasis|pyelectasis|pelvocaliectasis|pelvicaliectasis|pelvicaliceal dilat|renal pelvis dilat|dilated renal pelvis|dilatation of the renal pelvis",
 "kw_caliectasis": r"caliectasis|calyectasis|calyceal dilat|caliceal dilat|dilated calyc|dilated calic",
 "kw_hydroureter": r"hydroureter|ureteral dilat|dilated ureter|ureterectasis",
 "kw_extrarenal": r"extra-?renal pelvis",
 "kw_parapelvic": r"parapelvic|peripelvic|renal sinus cyst",
 "kw_renal_cyst": r"(?:renal|kidney)[^.]{0,40}cyst|cyst[^.]{0,40}(?:kidney|renal)|bosniak|polycystic",
 "kw_stent": r"\bstent|double[- ]?j|nephroureteral|\bj[- ]?stent|ureteral catheter",
 "kw_nephrostomy": r"nephrostom",
 "kw_transplant": r"transplant",
 "kw_stone": r"ureterolith|nephrolith|calcul|stone|obstructing",
 "kw_obstruct": r"obstruct|upj|ureteropelvic|stricture",
 "kw_bladder_distended": r"(?:bladder[^.]{0,30}distend|distended[^.]{0,20}bladder|overdistend|bladder outlet)",
 "kw_excretory": r"excretory|delayed (?:phase|images)|urographic|ct urogra|contrast[^.]{0,30}collecting system|opacif",
 "kw_pyelo": r"pyelonephritis|perinephric strand|perinephric stranding|urothelial thicken",
 "kw_mass": r"\bmass|carcinoma|tumor|neoplasm|malignan",
 "kw_preg": r"pregnan|gravid",
 "kw_atrophy": r"atroph|cortical thinning|parenchymal thinning|thinning of the",
 "kw_uncertain": r"possible|possibly|may represent|cannot (?:be )?exclude|questionable|equivocal|suspected|likely|probable|suggest|versus|vs\.",
 "kw_nephrectomy": r"nephrectomy|solitary kidney|absent (?:left|right) kidney|(?:kidney|renal)[^.]{0,20}surgically absent",
}
for sid, d in rows.items():
    rr = rep.get(sid) or {}
    txt = rr.get("report", "") if rr.get("split", d["split"]) in OK else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if KRE.search(s)]
    ktxt = " ".join(sents)
    q = d["hn_quote"]
    for k, p in pat.items():
        d[k] = hit(p, ktxt)          # non-negated mention in kidney/urinary sentences
        d["q" + k[2:]] = int(bool(re.search(p, q, re.I)))   # in the hydronephrosis label quote
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values()).drop(columns=["hn_quote"])
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict())
t = df[df.split == "train"]
print("train pos", int(t.y_hn.sum()), "neg", int((t.y_hn == 0).sum()), "nan", int(t.y_hn.isna().sum()))
print("grade x y:", pd.crosstab(t.hn_grade, t.y_hn).to_dict())
print("side x y:", pd.crosstab(t.hn_side, t.y_hn).to_dict())
kc = [c for c in df.columns if c.startswith(("kw_", "q_"))]
print("pos rates:", t[t.y_hn == 1][kc].mean().round(3).to_dict())
print("neg rates:", t[t.y_hn == 0][kc].mean().round(3).to_dict())
