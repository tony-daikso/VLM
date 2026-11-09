"""Per-study biliary-dilation label/text table for train+val ONLY (split filtered before reading any label/report).
Writes labels_trval.csv (no free text), plus counts. Nothing from test is read into the table."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_biliary/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
okids = set(spl[spl.isin(OK)].index)
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or r["study_id"] not in okids:
        continue
    a = r["answer"]; g = a["biliary_ductal_dilation"]; det = g.get("detail") or {}
    cbd = det.get("cbd_mm")
    try: cbd = float(cbd)
    except Exception: cbd = np.nan
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], bd_cat=g["category"], y_bdd=b(g["category"]),
        bd_site=det.get("site", "na"), cbd_mm=cbd,
        y_chole=b(a["surgically_absent_gallbladder"]["category"]), y_gs=b(a["gallstones"]["category"]),
        y_steat=b(a["hepatic_steatosis"]["category"]), y_ascites=b(a["ascites"]["category"]),
        bd_quote=" ".join(g.get("quotes") or []), bd_why=str(g.get("why") or ""))
rep = json.load(open(R + "merlin_disease14/reports.json"))
CRE = re.compile(r"bil|\bcbd\b|common duct|bile duct|choledoch|hepatic duct|intrahepatic|ductal", re.I)
CORE = {"kw_mild", "kw_marked", "kw_prominent", "kw_upper_normal", "kw_unchanged", "kw_intra", "kw_extra", "kw_neg_dilat", "kw_uncertain"}
BRE = re.compile(r"bil|duct|cbd|choledoch|hepatic duct|ampull|pneumobilia|cholang|porta|stent|ercp|whipple|pancrea|cholecyst", re.I)
pat = {
 "kw_mild": r"\bmild(?:ly)?\b|minimal(?:ly)?|slight(?:ly)?|borderline",
 "kw_marked": r"moderate|marked(?:ly)?|severe|significant(?:ly)? dilat|massive",
 "kw_prominent": r"prominen",
 "kw_upper_normal": r"upper limits? of normal|upper normal|top normal|high[- ]normal|within normal limits for|normal for (?:age|post)",
 "kw_unchanged": r"unchanged|stable|again (?:seen|noted|demonstrated)|similar|no (?:significant )?(?:interval )?change|redemonstrat|persistent",
 "kw_postchole": r"post[- ]?cholecystectomy|cholecystectomy|surgically absent|status post .*gallbladder",
 "kw_pneumobilia": r"pneumobilia|air (?:in|within) the bil|biliary (?:air|gas)|gas in the bil",
 "kw_stent": r"(?:biliary|cbd|common bile duct|ductal|internal[- ]external|cholecystostomy)\s+(?:stent|drain|catheter)|stent (?:in|within|traversing|extending)[^.]{0,30}(?:bile|cbd|biliary|duct|ampulla)|\bptbd\b|\bptc\b|ercp|sphincterotomy|choledochojejun|hepaticojejun|whipple|pancreaticoduoden",
 "kw_cbd_stone": r"choledocholith|stone in the (?:distal )?(?:common bile|cbd)|cbd stone|common bile duct stone|calculus (?:in|within) the (?:distal )?(?:common|cbd)",
 "kw_panc_mass": r"pancrea\w* (?:head )?(?:mass|lesion|neoplasm|cancer|carcinoma|adenocarcinoma|tumor)|(?:mass|lesion) (?:in|of|within) the (?:pancreatic )?head|ampull\w* (?:mass|lesion|carcinoma|tumor)|periampull|double[- ]duct|cholangiocarcinoma|klatskin|(?:biliary|cbd|common bile duct) (?:obstruct|stricture)|obstruct\w* (?:of the )?(?:bil|cbd|common bile)",
 "kw_pd_dilat": r"pancreatic duct (?:is |are )?(?:\w+ )?dilat|dilat\w* (?:of the )?(?:main )?pancreatic duct|double[- ]duct",
 "kw_intra": r"intrahepatic",
 "kw_extra": r"extrahepatic|common bile|\bcbd\b|common duct|common hepatic|choledoch",
 "kw_neg_dilat": r"no (?:\w+ ){0,4}(?:biliary|ductal) dilat|without (?:\w+ ){0,3}(?:biliary|ductal) dilat|not dilated|normal caliber|nondilated|non-dilated",
 "kw_uncertain": r"possible|possibly|may represent|cannot (?:be )?exclude|questionable|equivocal|suspect|likely|probabl|suggest|vs\.?|versus",
 "kw_mri_rec": r"mrcp|recommend|correlat",
}
for sid, d in rows.items():
    txt = (rep.get(sid) or {}).get("report", "") if spl.get(sid) in OK else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if BRE.search(s)]
    btxt = " ".join(sents) + " " + d["bd_quote"]
    core = " ".join(s for s in sents if CRE.search(s)) + " " + d["bd_quote"]
    qtxt = d["bd_quote"]
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, core if k in CORE else btxt, re.I)))
        d["q" + k] = int(bool(re.search(p, qtxt, re.I)))
    # report text mentions dilation at all (any sentence)
    d["rep_dilat_any"] = int(bool(re.search(r"(?:bil|duct|cbd|choledoch)\w*[^.]{0,60}dilat|dilat\w*[^.]{0,60}(?:bil|duct|cbd)", txt, re.I)))
    mm = [float(x) * (10 if u.lower() == "cm" else 1) for x, u in re.findall(r"(\d+(?:\.\d+)?)\s*(mm|cm)", qtxt, re.I)]
    d["quote_mm"] = max(mm) if mm else np.nan
    d["has_report"] = int(bool(txt))
    d["bd_why_len"] = len(d["bd_why"])
df = pd.DataFrame(rows.values()).drop(columns=["bd_quote", "bd_why"])
# training label actually used (labels_disease14.csv) -- check consistency
lab = pd.read_csv(R + "OWL/data/labels_disease14.csv").set_index("study_id")
df["y_csv"] = df.study_id.map(lab["biliary_ductal_dilation"]) if "biliary_ductal_dilation" in lab else np.nan
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict(), "label==csv:", float((df.y_bdd.fillna(-1) == df.y_csv.fillna(-1)).mean()))
t = df[df.split == "train"]
print("train pos", int(t.y_bdd.sum()), "neg", int((t.y_bdd == 0).sum()), "nan", int(t.y_bdd.isna().sum()))
print("pos kw rates:", t[t.y_bdd == 1][[c for c in df.columns if c.startswith("kw_")]].mean().round(3).to_dict())
print("neg kw rates:", t[t.y_bdd == 0][[c for c in df.columns if c.startswith("kw_")]].mean().round(3).to_dict())
print("site x cat:", pd.crosstab(t.bd_site, t.bd_cat).to_dict())
print("chole x y:", pd.crosstab(t.y_chole.fillna(-1), t.y_bdd.fillna(-1)).to_dict())
print("cbd_mm pos:", t[t.y_bdd == 1].cbd_mm.describe().round(1).to_dict())
