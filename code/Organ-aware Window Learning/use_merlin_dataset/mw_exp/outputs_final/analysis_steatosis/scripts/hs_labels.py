"""Per-study steatosis label/text table for train+val ONLY (split filtered before reading any label/report).
No report text is written out; only keyword flags."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_steatosis/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
okids = set(spl[spl.isin(OK)].index)


def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan


rows = {}
cats = set()
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or r.get("study_id") not in okids:
        continue
    g = r["answer"]["hepatic_steatosis"]
    cats.add(g["category"])
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], hs_cat=g["category"], y_hs=b(g["category"]),
                               hs_detail=json.dumps(g.get("detail") or {}), hs_quote=" ".join(g.get("quotes") or []))
print("categories:", sorted(cats))
rep = json.load(open(R + "merlin_disease14/reports.json"))
LIV = re.compile(r"liver|hepat|steatos|fatty|attenuation|hypodens|hypoattenuat", re.I)
pat = {
    "kw_steatosis": r"steatos",
    "kw_fatty": r"fatty (?:infiltration|liver|change)|fatty",
    "kw_decr_att": r"(?:decreased|diminished|low|reduced|hypo)[- ]?(?:attenuation|density)|hypoattenuat|hypodens",
    "kw_focal": r"focal (?:fat|fatty)|focal sparing|fatty sparing|sparing|geographic|perfusion|focal (?:area|region)",
    "kw_hetero": r"heterogene|patchy|nodular|multifocal",
    "kw_mild": r"\bmild|minimal|slight|subtle|early",
    "kw_moderate": r"moderate",
    "kw_severe": r"severe|marked|significant|diffuse",
    "kw_diffuse": r"diffuse",
    "kw_uncertain": r"possible|possibly|may represent|cannot (?:be )?exclude|questionable|equivocal|suspected|likely|probable|suggest|could reflect|may reflect",
    "kw_contrast_limit": r"(?:limited|difficult|cannot|can ?not|unable).{0,40}(?:contrast|phase|enhancement)|on (?:this )?(?:contrast|enhanced)|non-?contrast (?:ct|exam|study) would",
    "kw_history": r"history of|known|clinical|indication|reason for",
    "kw_cirrhosis": r"cirrho|nodular (?:contour|liver)|fibrosis",
    "kw_hepatomegaly": r"hepatomegaly|enlarged liver|liver is enlarged",
    "kw_focal_fat": r"focal(?:ly)? (?:fat|fatty)|focal (?:area|region|zone)s? of (?:fat|fatty|decreased|low|hypo)|fatty infiltration (?:adjacent|near|along|around|at|in the region)|(?:adjacent|near|along) (?:to )?the falciform",
    "kw_sparing": r"sparing|spared",
    "kw_geo_perf": r"geographic|perfusion|patchy",
    "kw_landmark": r"falciform|gallbladder fossa|porta hepatis|segment (?:4|iv)\b|periportal|fissure",
    "kw_iron": r"iron|hemochromat|increased attenuation|hyperattenuat|hyperdense liver|amiodarone",
}
for sid, d in rows.items():
    q = d.pop("hs_quote")
    txt = (rep.get(sid) or {}).get("report", "") if sid in okids else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if LIV.search(s)]
    ltxt = " ".join(sents) + " " + q
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, ltxt, re.I)))
        d["q_" + k] = int(bool(re.search(p, q, re.I)))   # keyword inside the label quote itself
    # steatosis words appear only in indication/history-type sentences
    hist = [s for s in re.split(r"(?<=[.;:])\s+|\n", txt) if re.search(r"steatos|fatty", s, re.I)]
    d["n_steat_sent"] = len(hist)
    d["steat_only_history"] = int(len(hist) > 0 and all(re.search(r"history|indication|clinical|reason|known", s, re.I) for s in hist))
    d["has_report"] = int(bool(txt))
    d["pure_focal"] = int(d["q_kw_focal_fat"] == 1 and d["q_kw_sparing"] == 0 and d["q_kw_steatosis"] == 0 and d["q_kw_diffuse"] == 0)
    d["quote_empty"] = int(not q.strip())
df = pd.DataFrame(rows.values())
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict())
t = df[df.split == "train"]
print("train cats:", t.hs_cat.value_counts().to_dict())
print("train pos", int(t.y_hs.sum()), "neg", int((t.y_hs == 0).sum()), "nan", int(t.y_hs.isna().sum()))
print("detail keys examples:", t[t.y_hs == 1].hs_detail.value_counts().head(8).to_dict())
kc = [c for c in df.columns if c.startswith("kw_")]
qc = [c for c in df.columns if c.startswith("q_kw_")] + ["pure_focal"]
print("pos quote-kw rate:", t[t.y_hs == 1][qc].mean().round(3).to_dict())
print("pos kw rate:", t[t.y_hs == 1][kc].mean().round(3).to_dict())
print("neg kw rate:", t[t.y_hs == 0][kc].mean().round(3).to_dict())
