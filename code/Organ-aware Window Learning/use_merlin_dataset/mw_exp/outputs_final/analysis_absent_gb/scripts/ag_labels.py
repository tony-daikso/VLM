"""Per-study post-cholecystectomy (surgically_absent_gallbladder) label + keyword table, train+val ONLY.
Split filtered before reading any label/report; no report text is written out (flags only)."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_absent_gb/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    a = r["answer"]; c = a["surgically_absent_gallbladder"]
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], ch_cat=c["category"], y_chole=b(c["category"]),
        y_gs=b(a["gallstones"]["category"]), gs_cat=a["gallstones"]["category"],
        q=" ".join(c.get("quotes") or []) + " " + " ".join(a["gallstones"].get("quotes") or []))
rep = json.load(open(R + "merlin_disease14/reports.json"))
GBRE = re.compile(r"gall ?bladder|cholecyst|gallstone|cholelith|biliary|clip|fossa|sludge", re.I)
pat = {
 "kw_chole": r"cholecystectom|surgically (?:absent|removed)|status post chole|s/p chole|post-?cholecyst|gallbladder (?:has been |was )?removed|absent gallbladder|gallbladder is absent|resected gallbladder",
 "kw_not_vis": r"gall ?bladder (?:is |was )?(?:not (?:well |clearly )?(?:visualized|seen|identified|visible)|non-?visualized)|non-?visuali[sz]ation of the gall|gall ?bladder (?:is )?not definitely",
 "kw_absent_only": r"gall ?bladder (?:is )?(?:absent|not present)|absence of (?:the )?gall",
 "kw_partial": r"partial cholecyst|subtotal cholecyst|gallbladder remnant|remnant gallbladder|residual gallbladder|remnant",
 "kw_clips": r"surgical clip|cholecystectomy clip|clips? (?:in|within|at|near) the (?:gallbladder|right upper|gb) |clips\b|\bclip\b",
 "kw_fossa": r"gallbladder fossa|cholecystectomy bed|surgical bed|gb fossa",
 "kw_contracted": r"contracted|collapsed|decompressed|not distended|underdistended|non-?distended|physiologically|incompletely distended",
 "kw_porcelain": r"porcelain|wall calcif",
 "kw_filled": r"filled with|packed with|replaced by|completely filled",
 "kw_gb_present": r"gall ?bladder (?:is |appears |demonstrates )?(?:unremarkable|normal|distended|present|contains|is filled|wall)|gallstones? (?:are |is )?(?:present|noted|seen)|cholelithiasis|sludge",
 "kw_uncertain": r"possible|possibly|may (?:represent|be)|cannot (?:be )?exclude|questionable|equivocal|presumabl|likely|probabl|suspected",
 "kw_history": r"history of|prior|previous|status post|s/p|post-?surgical|post-?operative",
 "kw_cholecystitis": r"cholecystitis|pericholecystic",
 "kw_drain": r"drain|cholecystostomy|percutaneous",
 "kw_liver_tx": r"transplant|hepatectomy|whipple|pancreaticoduodenectomy|hepaticojejunostomy|liver resection",
}
for sid, d in rows.items():
    e = rep.get(sid) or {}
    txt = e.get("report", "") if e.get("split", d["split"]) in OK else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if GBRE.search(s)]
    gtxt = " ".join(sents) + " " + d.pop("q")
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, gtxt, re.I)))
    d["kw_cholecystectomy_anywhere"] = int(bool(re.search(r"cholecystectom", txt, re.I)))
    m = re.search(r"gall ?bladder(?: and biliary[^:]{0,20})?\s*:\s*(.{0,300}?)(?=\s{2,}[A-Z][A-Za-z ,/]{2,40}:|$)", txt, re.I | re.S)
    sec = m.group(1) if m else ""
    d["gb_section"] = int(bool(m))
    d["sec_absent"] = int(bool(re.search(r"surgically|absent|cholecystectom|removed|resected", sec, re.I)))
    d["sec_not_vis"] = int(bool(re.search(r"not (?:well |clearly |definitely )?(?:visuali[sz]ed|seen|identified|visible)|non-?visuali", sec, re.I)))
    d["sec_normal"] = int(bool(re.search(r"^\s*(?:normal|unremarkable)", sec, re.I)))
    d["kw_liver_tx_anywhere"] = int(bool(re.search(r"liver transplant|hepatic transplant|transplant(?:ed)? liver|whipple|pancreaticoduodenectomy|hepaticojejunostomy|hepatectomy", txt, re.I)))
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values())
meta = pd.read_csv(R + "merlin_disease14/metadata.csv").drop_duplicates("study id").rename(columns={"study id": "study_id"})
meta = meta[meta.study_id.isin(set(df.study_id))][["study_id", "Age", "Gender", "contrast", "phase"]]
df = df.merge(meta, on="study_id", how="left")
df.to_csv(OUT + "labels_trval.csv", index=False)
t = df[df.split == "train"]
print(df.groupby("split").size().to_dict(), "| train pos", int(t.y_chole.sum()), "neg", int((t.y_chole == 0).sum()), "| val pos", int(df[df.split=="val"].y_chole.sum()))
K = [c for c in df.columns if c.startswith("kw_")]
print(t.groupby("y_chole")[K].mean().round(3).T.to_string())
print(pd.crosstab(t.ch_cat, t.kw_chole)); print(pd.crosstab([t.ch_cat], [t.sec_absent, t.sec_not_vis])); print("gb_section rate", t.gb_section.mean())
