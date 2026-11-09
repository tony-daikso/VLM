"""Prior-imaging / CT-occult evidence flags for gallstone reports. TRAIN/VAL only: split.csv filtered first; report text of other
splits is never parsed (reports.json is loaded as a dict but only train/val keys are touched). Output: per-case flags + snippet."""
import json, re, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"; OK = {"train", "val"}
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
keep = set(spl[spl.isin(OK)].index)
lab = pd.read_csv(R + "OWL/mw_exp/outputs_final/analysis_gallstones/labels_trval.csv"); assert set(lab.split) <= OK
ids = [s for s in lab.study_id if s in keep]
rep_all = json.load(open(R + "merlin_disease14/reports.json"))
CTX = re.compile(r"gallbladder|gall bladder|cholelith|gallstone|\bGB\b", re.I)
DUCT = re.compile(r"bile duct|\bCBD\b|choledocho|renal|kidney|ureter|pancrea|\bduct\b", re.I)
GSX = re.compile(r"cholelith|gallstone|stones?\b|calcul", re.I)
RECO = re.compile(r"recommend|consider|\bif\b|could|suggest|follow-?up|correlat|may be (obtained|performed|helpful)|can be", re.I)
NEG = re.compile(r"\b(?:no|without|negative for|not|absence of|free of)\b[^.;]{0,40}$", re.I)
P = {
 "known": r"\bknown\b[^.;]{0,30}(cholelith|gallstone|stones?\b|calcul)",
 "previously": r"\b(previously|prior(?:ly)?)\s+(seen|described|noted|identified|demonstrated|visualized|reported|documented)|(seen|described|noted|identified|demonstrated|visualized|reported)\s+(on|in|at)\s+(the\s+)?(prior|previous|outside|recent)",
 "other_modality": r"(ultrasound|sonograph|\bUS\b|\bMRCP\b|\bMRI\b|\bMR\b|\bHIDA\b|ERCP|cholescintigra)",
 "history": r"\bhistory of\b[^.;]{0,40}(cholelith|gallstone|stones?\b|calcul|biliary colic)",
 "as_before": r"\bas before\b|\bas previously\b",
 "not_well_seen": r"not\s+(well\s+)?(seen|visualized|evaluated|identified|demonstrated|delineated|appreciated)\s+(on|by|with|at)\s+(this\s+|the\s+)?(CT|current|present|exam)|(poorly|incompletely)\s+(seen|visualized|evaluated)|better\s+(evaluated|assessed|seen|characterized|visualized)\s+(by|with|on|at)|not\s+(well\s+)?(seen|visualized)\b|(not|less)\s+(well\s+)?(evaluated|assessed)\s+(by|on|with)\s+CT|limited\s+(evaluation|sensitivity)",
 "noncalc": r"non-?\s?calcified|isodense|iso-attenuat|radiolucent|non-?\s?radiopaque|no\s+radiopaque|not\s+radiopaque|cholesterol\s+(stone|calcul)|no\s+(CT\s+)?(evidence|visible)[^.;]{0,30}(radiopaque|calcified)|radiopaque\s+(gallstones?|stones?|calcul)\s+(is|are)\s+not|no\s+(radiodense|hyperdense|calcified)\s+(gallstone|stone|calcul)",
}
P = {k: re.compile(v, re.I) for k, v in P.items()}
rows = []
for sid in ids:
    v = rep_all.get(sid) or {}
    if spl.get(sid) not in OK or v.get("split", spl.get(sid)) not in OK: continue
    t = v.get("report", "") or ""
    ss = [s.strip() for s in re.split(r"(?<=[.;])\s+|\n+", t) if s.strip()]
    d = {"study_id": sid, "split": spl[sid]}; snip = []
    for k, p in P.items():
        hit = 0
        for s in ss:
            if not CTX.search(s): continue
            if k != "known" and DUCT.search(s) and not re.search(r"cholelith|gallstone|gallbladder", s, re.I): continue
            if k in ("noncalc", "previously", "as_before", "not_well_seen", "other_modality") and not GSX.search(s): continue
            if k == "other_modality" and RECO.search(s): continue
            if k == "noncalc" and re.search(r"bile duct|\bCBD\b|choledocho|ductal", s, re.I) and not re.search(r"cholelith|gallstone|gallbladder (stone|calcul)", s, re.I): continue
            for m in p.finditer(s):
                if k in ("known", "history", "previously", "other_modality") and NEG.search(s[:m.start()][-60:]) and k != "previously":
                    continue  # e.g. "no history of"
                hit = 1; snip.append(f"[{k}] {s[:240]}"); break
            if hit: break
        d["pf_" + k] = hit
    d["pf_snip"] = " || ".join(snip)[:1500]; rows.append(d)
del rep_all
D = pd.DataFrame(rows); assert set(D.split) <= OK
D.to_csv(R + "OWL/mw_exp/outputs_final/gallstones_lowdens/text_flags_trval.csv", index=False)
print(len(D), D.filter(like="pf_").drop(columns="pf_snip").sum().to_dict())
