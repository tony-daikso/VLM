"""Per-study AAA label/text table for train+val ONLY (split filtered before reading any label/report).
Outputs only categorical labels, the Qwen diameter/repaired fields and keyword flags (no report text)."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_aaa/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    a = r["answer"]; g = a["abdominal_aortic_aneurysm"]; det = g.get("detail") or {}
    dc = det.get("diameter_cm")
    try: dc = float(dc)
    except (TypeError, ValueError): dc = np.nan
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], aaa_cat=g["category"], q_diam_cm=dc,
        q_repaired=det.get("repaired", "na"), y_aaa=b(g["category"]),
        y_ath=b(a["atherosclerosis"]["category"]), q=" ".join(g.get("quotes") or []))
rep = json.load(open(R + "merlin_disease14/reports.json"))
VRE = re.compile(r"aort|aneurys|iliac|ectat|ectasia|dilat|evar|endovascular|endograft|stent.?graft|graft|caliber|calibre", re.I)
pat = {
 "kw_aneurysm": r"aneurysm",
 "kw_ectasia": r"ectatic|ectasia",
 "kw_dilat": r"dilat|enlarg|prominent caliber|upper limits of normal|borderline",
 "kw_evar": r"\bevar\b|endovascular|endograft|endoleak|stent.?graft|aortic graft|aorto-?(?:bi)?(?:iliac|femoral) graft|graft repair|repair of (?:an |the )?(?:abdominal )?aort|status post (?:aaa|aneurysm) repair|aneurysm repair",
 "kw_iliac_aneur": r"iliac\W+(?:\w+\W+){0,4}(?:aneurysm|ectat|dilat)|(?:aneurysm|ectatic|dilat\w*)\W+(?:\w+\W+){0,4}iliac",
 "kw_thoracic": r"thoracoabdominal|thoracic aort\w*\W+(?:\w+\W+){0,4}(?:aneurysm|dilat|ectat)|descending\W+(?:\w+\W+){0,3}aort",
 "kw_suprarenal": r"suprarenal|juxtarenal|pararenal",
 "kw_infrarenal": r"infrarenal",
 "kw_thrombus": r"mural thromb|thrombus|thrombosed",
 "kw_dissection": r"dissect",
 "kw_normal_caliber": r"normal (?:in )?(?:caliber|calibre|size)|non-?aneurysmal|without aneurysm|no (?:evidence of )?(?:abdominal )?(?:aortic )?aneurysm",
 "kw_tortuous": r"tortuous|tortuosity",
}
SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x\s*\d+(?:\.\d+)?\s*)*(cm|mm)\b", re.I)
for sid, d in rows.items():
    rr = rep.get(sid) or {}
    txt = rr.get("report", "") if rr.get("split", d["split"]) in OK else ""
    sents = [s for s in re.split(r"(?<=[.;])\s+|\n", txt) if VRE.search(s)]
    vtxt = " ".join(sents) + " " + d["q"]
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, vtxt, re.I)))
    # max size (cm) mentioned in aorta/aneurysm sentences (not iliac-only sentences)
    mx = np.nan
    for s in sents + [d["q"]]:
        if not re.search(r"aort|aneurys", s, re.I):
            continue
        for v, u in SIZE.findall(s):
            v = float(v) / (10 if u.lower() == "mm" else 1)
            if 1.0 <= v <= 12:
                mx = v if np.isnan(mx) else max(mx, v)
    d["txt_max_cm"] = mx
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values()).drop(columns=["q"])
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby(["split", "aaa_cat"]).size().to_dict())
t = df[df.split == "train"]
print("train pos", int(t.y_aaa.sum()), "neg", int((t.y_aaa == 0).sum()), "nan", int(t.y_aaa.isna().sum()))
kw = [c for c in df.columns if c.startswith("kw_")]
print("pos kw:", t[t.y_aaa == 1][kw].mean().round(3).to_dict())
print("neg kw:", t[t.y_aaa == 0][kw].mean().round(3).to_dict())
print("pos q_diam:", t[t.y_aaa == 1].q_diam_cm.describe().round(2).to_dict())
print("neg q_diam notna:", int(t[t.y_aaa == 0].q_diam_cm.notna().sum()), t[t.y_aaa == 0].q_diam_cm.describe().round(2).to_dict())
print("repaired x y:", pd.crosstab(t.q_repaired, t.y_aaa).to_dict())
