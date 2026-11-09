"""Per-study gallstone label/text table for train+val ONLY (split filtered before reading any label/report)."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_gallstones/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK:
        continue
    a = r["answer"]; g = a["gallstones"]
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], gs_cat=g["category"],
        gs_calc=(g.get("detail") or {}).get("calcified", "na"), y_gs=b(g["category"]),
        y_chole=b(a["surgically_absent_gallbladder"]["category"]), y_bdd=b(a["biliary_ductal_dilation"]["category"]),
        gs_quote=" ".join(g.get("quotes") or []))
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rep = json.load(open(R + "merlin_disease14/reports.json"))
GBRE = re.compile(r"gallbladder|gall bladder|cholelith|gallstone|choledocholith|sludge|cholecyst|biliary|stone", re.I)
pat = {
 "kw_calcified": r"calcif|radiopaque|radio-opaque|hyperdense|dense stone|high[- ]density",
 "kw_noncalc": r"non-?calcified|not calcified|isodense|faint|subtle|low[- ]density|cholesterol",
 "kw_tiny": r"tiny|punctate|small|minute",
 "kw_multiple": r"multiple|several|numerous|stones\b|gallstones\b|calculi",
 "kw_large": r"large|\bbulky|filled with|packed",
 "kw_layering": r"layering|dependent",
 "kw_sludge": r"sludge|tumefactive",
 "kw_cbd_stone": r"choledocholith|common bile duct stone|stone in the (?:distal )?(?:common bile|cbd)|cbd stone",
 "kw_history": r"history of|status post|s/p|known",
 "kw_not_vis": r"not (?:well )?visualized|contracted|collapsed|decompressed|not distended|underdistended|non-?distended",
 "kw_distended": r"distended|hydrops|distention|distension",
 "kw_uncertain": r"possible|possibly|may represent|cannot (?:be )?exclude|questionable|equivocal|suspected|likely|probable|suggest",
 "kw_ultrasound": r"ultrasound|sonograph",
 "kw_cholecystitis": r"cholecystitis|pericholecystic|wall thicken",
 "kw_porcelain": r"porcelain|wall calcif",
 "kw_absent": r"surgically absent|cholecystectomy|removed",
}
for sid, d in rows.items():
    if spl.get(sid) not in OK: continue
    txt = (rep.get(sid) or {}).get("report", "") if (rep.get(sid) or {}).get("split", d["split"]) in OK else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if GBRE.search(s)]
    gtxt = " ".join(sents) + " " + d["gs_quote"]
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, gtxt, re.I)))
    mm = [float(x) * (10 if u.lower() == "cm" else 1) for x, u in re.findall(r"(\d+(?:\.\d+)?)\s*(mm|cm)", d["gs_quote"], re.I)]
    d["stone_mm"] = max(mm) if mm else np.nan
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values()).drop(columns=["gs_quote"])
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict())
t = df[df.split == "train"]
print("train pos", int(t.y_gs.sum()), "neg", int((t.y_gs == 0).sum()))
print(t[t.y_gs == 1][[c for c in df.columns if c.startswith("kw_")]].mean().round(3).to_dict())
print("stone_mm pos:", t[t.y_gs == 1].stone_mm.describe().round(1).to_dict())
print("calc x y:", pd.crosstab(t.gs_calc, t.y_gs).to_dict())
