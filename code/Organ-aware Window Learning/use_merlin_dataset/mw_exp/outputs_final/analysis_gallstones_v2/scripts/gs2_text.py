"""Extract gallbladder-related report sentences + label quotes for TRAIN/VAL only (split filtered before reading text)."""
import json, re, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OK = {"train", "val"}
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
keep = set(spl[spl.isin(OK)].index)
lab = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    if '"test"' in line[:200] and False: pass
    r = json.loads(line)
    sid = r["study_id"]
    if sid not in keep or r.get("split") not in OK: continue
    a = r["answer"]
    g = a["gallstones"]
    lab[sid] = dict(gs_cat=g["category"], gs_quote=" | ".join(g.get("quotes") or []),
                    keys="|".join(sorted(a.keys())))
rep_all = json.load(open(R + "merlin_disease14/reports.json"))
GBW = re.compile(r"gallbladder|gall bladder|cholelith|gallstone|sludge|cholecyst|pericholecystic|vicarious|gb\b|biliary|pneumobilia|stent", re.I)
rows = []
for sid, d in lab.items():
    v = rep_all.get(sid) or {}
    if v.get("split", spl.get(sid)) not in OK: continue
    t = v.get("report", "") or ""
    ss = [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+", t) if s.strip()]
    gb = [s for s in ss if GBW.search(s)]
    d = dict(d); d["study_id"] = sid; d["split"] = spl[sid]; d["gb_text"] = " || ".join(gb)[:3000]
    d["calc_near"] = " || ".join(s for s in ss if re.search(r"(hepatic|liver|renal|kidney).{0,40}calcif|calcif.{0,40}(hepatic|liver|renal|kidney)|granuloma", s, re.I))[:600]
    d["ascites"] = " || ".join(s for s in ss if re.search(r"ascites|free fluid|pericholecystic fluid", s, re.I))[:600]
    rows.append(d)
del rep_all
pd.DataFrame(rows).to_csv("/home/advantech/gs2_text_trval.csv", index=False)
print(len(rows))
