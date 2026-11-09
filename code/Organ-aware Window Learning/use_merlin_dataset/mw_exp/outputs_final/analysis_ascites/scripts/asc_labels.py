"""Per-study ascites label/text table for train+val ONLY (split filtered before reading any label/report line).
Writes keyword flags only (no report text)."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_ascites/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    a = r["answer"]; g = a["ascites"]; dt = g.get("detail") or {}
    d = dict(study_id=r["study_id"], split=r["split"], asc_cat=g["category"], y_asc=b(g["category"]),
             amount=dt.get("amount", "na"), hemorrhagic=dt.get("hemorrhagic", "na"),
             asc_quote=" ".join(g.get("quotes") or []))
    for k in ["splenomegaly", "hepatic_steatosis", "bowel_obstruction", "free_air", "hydronephrosis", "appendicitis",
              "biliary_ductal_dilation", "surgically_absent_gallbladder"]:
        d["y_" + k] = b(a[k]["category"])
    rows[r["study_id"]] = d
rep = json.load(open(R + "merlin_disease14/reports.json"))
FL = re.compile(r"fluid|ascites|hemoperitoneum|collection|abscess|seroma|hematoma|loculat|cul-de-sac|pelvis", re.I)
sent_pat = {   # on fluid-related sentences + Qwen quotes
 "kw_trace": r"\btrace\b|minimal|tiny|scant|small amount|minute",
 "kw_small": r"\bsmall\b|mild",
 "kw_modlarge": r"moderate|large|massive|extensive|significant|diffuse|marked",
 "kw_pelvic": r"pelvi|cul-de-sac|rectouterine|rectovesical|douglas",
 "kw_perihep": r"perihepatic|perisplenic|morison|hepatorenal|subphrenic|paracolic",
 "kw_physiologic": r"physiolog|likely physiologic|normal for|ovulat|follic",
 "kw_free_fluid": r"free fluid",
 "kw_ascites_word": r"ascites",
 "kw_loculated": r"loculat|collection|abscess|seroma|pseudocyst|lymphocele|biloma|urinoma",
 "kw_hemo": r"hemoperitoneum|hematoma|hemorrhag|blood|high[- ]density fluid|hyperdense fluid",
 "kw_postop": r"post-?operative|postsurgical|surgical bed|status post|s/p|recent surgery|drain",
 "kw_uncertain": r"possible|possibly|may represent|questionable|equivocal|suspected|probabl|cannot (?:be )?exclude",
 "kw_resolved": r"resolved|decreased|interval (?:decrease|resolution)|improved|no longer",
 "kw_history": r"history of|known",
 "kw_negated": r"\bno\b[^.]{0,40}(?:fluid|ascites)|without[^.]{0,30}(?:fluid|ascites)",
}
rep_pat = {   # whole report (confounders / mimics)
 "r_cirrhosis": r"cirrho|nodular (?:liver|hepatic)|portal hypertension|varices|recanaliz",
 "r_splenomeg": r"splenomegaly|enlarged spleen",
 "r_anasarca": r"anasarca|body wall edema|subcutaneous edema|soft tissue edema|third spacing|third-spacing",
 "r_pleural": r"pleural effusion",
 "r_carcinomatosis": r"carcinomatosis|peritoneal (?:implant|metast|nodul|thicken)|omental cak|omental (?:metast|nodul)",
 "r_mesent_edema": r"mesenteric edema|mesenteric stranding|edema of the mesentery|mesenteric congestion",
 "r_bowel_wall": r"bowel wall thicken|colitis|enteritis",
 "r_postop": r"post-?operative|postsurgical|status post|s/p|recent surgery",
 "r_transplant": r"transplant",
 "r_dialysis": r"dialysis|peritoneal dialysis",
 "r_oral_contrast": r"oral contrast",
 "r_no_oral": r"(?:no|without) oral contrast",
 "r_pregnan": r"pregnan|gravid|intrauterine",
 "r_ovarian_cyst": r"ovarian cyst|adnexal cyst|follicle|corpus luteum",
 "r_pancreatitis": r"pancreatitis",
 "r_heart_fail": r"cardiomegaly|heart failure|pericardial effusion",
}
for sid, d in rows.items():
    e = rep.get(sid) or {}
    txt = e.get("report", "") if e.get("split", d["split"]) in OK else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if FL.search(s)]
    gtxt = " ".join(sents) + " " + d["asc_quote"]
    for k, p in sent_pat.items():
        d[k] = int(bool(re.search(p, gtxt, re.I)))
    NEGS = re.compile(r"\bno\b|without|negative for|not (?:seen|identified|present|visualized)|resolved", re.I)
    pos_s = " ".join(x for x in sents if not NEGS.search(x))      # fluid-related sentences with no negation
    d["kwp_loculated"] = int(bool(re.search(r"loculat|abscess|seroma|pseudocyst|lymphocele|biloma|urinoma|(?:fluid|complex) collection", pos_s, re.I)))
    d["kwp_hemo"] = int(bool(re.search(r"hemoperitoneum|hematoma|hemorrhag|high[- ]density fluid|hyperdense fluid", pos_s, re.I)))
    d["kwp_fluid"] = int(bool(re.search(r"free fluid|ascites|fluid in the (?:pelvis|cul-de-sac)|pelvic fluid|perihepatic fluid|perisplenic fluid", pos_s, re.I)))
    for k, p in rep_pat.items():
        d[k] = int(bool(re.search(p, txt, re.I)))
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values()).drop(columns=["asc_quote"])
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict())
t = df[df.split == "train"]
print("train pos", int(t.y_asc.sum()), "neg", int((t.y_asc == 0).sum()), "unclear", int(t.y_asc.isna().sum()))
print(pd.crosstab(t.asc_cat, t.amount))
print(pd.crosstab(t.asc_cat, t.hemorrhagic))
print("pos kw:", t[t.y_asc == 1][[c for c in df.columns if c.startswith(("kw_", "r_"))]].mean().round(3).to_dict())
print("neg kw:", t[t.y_asc == 0][[c for c in df.columns if c.startswith(("kw_", "r_"))]].mean().round(3).to_dict())
