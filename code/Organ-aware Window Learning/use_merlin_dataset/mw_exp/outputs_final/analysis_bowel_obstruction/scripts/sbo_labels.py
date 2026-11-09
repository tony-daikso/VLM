"""Per-study bowel-obstruction label/text table for train+val ONLY (split filtered before reading any label/report).
Saves flags only (no report text)."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_bowel_obstruction/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    a = r["answer"]; g = a["bowel_obstruction"]
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], sbo_cat=g["category"],
        sbo_type=(g.get("detail") or {}).get("type", "na"), y_sbo=b(g["category"]),
        y_fa=b(a["free_air"]["category"]), y_asc=b(a["ascites"]["category"]), y_app=b(a["appendicitis"]["category"]),
        y_hydro=b(a["hydronephrosis"]["category"]), quote=" ".join(g.get("quotes") or []))
rep = json.load(open(R + "merlin_disease14/reports.json"))
BOWEL = re.compile(r"bowel|loop|obstruct|ileus|transition|jejun|ileum|ileal|colon|colonic|cecum|cecal|sigmoid|rectum|volvulus|hernia|distend|dilat|small intestin|stomach|gastric|duoden", re.I)
NEGW = re.compile(r"\bno\b|without|negative|not\b|resolved|absence|free of|nor\b|normal caliber|nondilated|non-dilated|unremarkable", re.I)
# flags computed on the bowel-related sentences + Qwen quote ("t"); report-wide flags on the whole report ("txt")
pat_t = {
 "kw_partial": r"partial|low[- ]grade|incomplete|early|developing|evolving",
 "kw_highgrade": r"high[- ]grade|complete|severe|marked(?:ly)? dilat",
 "kw_closedloop": r"closed[- ]loop|volvulus|internal hernia|swirl|whirl",
 "kw_transition": r"transition (?:point|zone)",
 "kw_no_transition": r"(?:no|without)(?: \w+){0,3} transition|transition point (?:is )?not (?:seen|identified)",
 "kw_ileus": r"ileus|adynamic|paralytic|pseudo-?obstruction|ogilvie|functional obstruction",
 "kw_mild": r"mild(?:ly)?|borderline|minimal(?:ly)?|slight(?:ly)?|upper limits? of normal|prominent|top normal",
 "kw_resolving": r"resolv|improv|decreas|interval (?:decrease|improvement|resolution)|less dilat",
 "kw_adhesion": r"adhesi",
 "kw_hernia": r"hernia",
 "kw_lbo": r"large bowel obstruction|colonic obstruction|obstructing (?:colon|rectal|sigmoid)|\bLBO\b",
 "kw_gastric_outlet": r"gastric outlet|duodenal obstruction|gastroparesis|gastric distension|distended stomach|afferent loop|pancreaticobiliary limb|biliopancreatic limb",
 "kw_feces": r"small bowel feces|fecali[sz]",
 "kw_airfluid": r"air[- ]fluid",
 "kw_fluidfilled": r"fluid[- ]filled",
 "kw_uncertain": r"possible|possibly|may represent|may be|cannot (?:be )?exclude|questionable|equivocal|suspect|likely|probable|suggest|versus|vs\.?|concern",
 "kw_mass": r"mass|tumou?r|carcinoma|malignan|neoplas|metasta|carcinomatosis|implant",
 "kw_intussusception": r"intussuscep",
 "kw_contrast_passes": r"contrast (?:has )?(?:passed|reach|traverse|opacif\w* (?:the )?colon|is (?:seen|present) (?:with)?in the colon)|passage of (?:oral )?contrast",
}
pat_txt = {
 "rep_postop": r"post-?operative|post-?surgical|status post|\bs/p\b|anastomo|ostomy|laparotom|resection|recent surgery|surgical (?:clips|changes|drain)",
 "rep_ostomy": r"ostomy|stoma",
 "rep_ngtube": r"nasogastric|\bng tube|\bngt\b|enteric tube|gastric tube|orogastric|feeding tube|dobhoff|\btube (?:in|within|terminat\w*|tip) (?:the )?(?:stomach|esophag)",
 "rep_gastroenteritis": r"enteritis|gastroenteritis|colitis|proctitis|ileitis|diarrhea|infectious|inflammatory bowel|crohn",
 "rep_oral_contrast": r"oral contrast|positive contrast (?:in|within)|enteric contrast",
 "rep_no_oral_contrast": r"without oral contrast|no oral contrast|oral contrast (?:was )?not",
 "rep_ascites": r"ascites|free fluid",
 "rep_pneumatosis": r"pneumatosis|portal venous gas|mesenteric venous gas",
 "rep_bowel_wall": r"wall thicken|bowel wall|mural thicken|hypoenhanc|ischemi",
}
DIL = re.compile(r"(?:dilat|distend|distension|distention)\w*", re.I)
SBX = re.compile(r"small bowel|small intestin|jejun|ileum|ileal|bowel loops|loops of bowel", re.I)
OBS = re.compile(r"obstruct", re.I)
for sid, d in rows.items():
    rr = rep.get(sid) or {}
    txt = rr.get("report", "") if rr.get("split", d["split"]) in OK else ""
    sents = re.split(r"(?<=[.;])\s+|\n", txt)
    bs = [s for s in sents if BOWEL.search(s)]
    t = " ".join(bs) + " " + d["quote"]
    for k, p in pat_t.items():
        src = d["quote"] if k in ("kw_partial", "kw_highgrade", "kw_mild", "kw_uncertain") else t
        d[k] = int(bool(re.search(p, src, re.I)))
    # affirmative sentences (no negation word); report-wide flags only count affirmative sentences
    aff = [s for s in sents if not NEGW.search(s)]
    atxt = " ".join(aff)
    for k, p in pat_txt.items():
        d[k] = int(bool(re.search(p, atxt, re.I)))
    d["rep_aff_ileus"] = int(any(re.search(pat_t["kw_ileus"], s, re.I) for s in aff))
    d["rep_aff_sb_dilated"] = int(any(DIL.search(s) and SBX.search(s) for s in aff))
    d["rep_aff_obstruct"] = int(any(OBS.search(s) and re.search(r"bowel|intestin|colon|SBO", s, re.I) for s in aff))
    d["rep_aff_colon_dilated"] = int(any(DIL.search(s) and re.search(r"colon|cecum|cecal|sigmoid|large bowel|rectum", s, re.I) for s in aff))
    d["rep_neg_sbo"] = int(bool(re.search(r"no (?:evidence of )?(?:small )?bowel obstruction|without (?:evidence of )?(?:small )?bowel obstruction|no (?:\w+ )?obstruction", txt, re.I)))
    m = re.findall(r"(?:measur\w*|up to|diameter of)\s*(\d+(?:\.\d+)?)\s*(cm|mm)", d["quote"], re.I)
    v = [float(x) * (10 if u.lower() == "cm" else 1) for x, u in m]
    v = [x for x in v if 10 <= x <= 150]
    d["quote_max_mm"] = max(v) if v else np.nan
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values())
df.drop(columns=["quote"]).to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict())
t = df[df.split == "train"]
print("train pos", int(t.y_sbo.sum()), "neg", int((t.y_sbo == 0).sum()), "unclear", int(t.y_sbo.isna().sum()))
print("pos cat:", t[t.y_sbo == 1].sbo_cat.value_counts().to_dict(), "type:", t[t.y_sbo == 1].sbo_type.value_counts().to_dict())
kc = [c for c in df.columns if c.startswith(("kw_", "rep_"))]
print(pd.DataFrame({"pos": t[t.y_sbo == 1][kc].mean(), "neg": t[t.y_sbo == 0][kc].mean(), "unclear": t[t.y_sbo.isna()][kc].mean()}).round(3).to_string())
print("quote_max_mm pos", t[t.y_sbo == 1].quote_max_mm.describe().round(1).to_dict())
print("neg with affirmative obstruct sentence:", int(t[t.y_sbo == 0].rep_aff_obstruct.sum()), "aff ileus:", int(t[t.y_sbo == 0].rep_aff_ileus.sum()),
      "aff SB dilated:", int(t[t.y_sbo == 0].rep_aff_sb_dilated.sum()))
print("has_report", t.has_report.mean())
