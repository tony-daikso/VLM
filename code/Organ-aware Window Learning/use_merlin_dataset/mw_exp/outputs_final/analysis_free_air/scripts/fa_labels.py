"""Per-study free-air label/text table for train+val ONLY (split filtered before reading any label/report)."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_free_air/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    a = r["answer"]; g = a["free_air"]
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], fa_cat=g["category"],
        fa_postop=(g.get("detail") or {}).get("postoperative", "na"), y_fa=b(g["category"]),
        y_sbo=b(a["bowel_obstruction"]["category"]), y_asc=b(a["ascites"]["category"]), y_app=b(a["appendicitis"]["category"]),
        y_chole=b(a["surgically_absent_gallbladder"]["category"]),
        quote=" ".join(g.get("quotes") or []))
rep = json.load(open(R + "merlin_disease14/reports.json"))
AIRS = re.compile(r"\bair\b|\bgas\b|pneumo|emphysema|locule", re.I)
pat = {
 "kw_postop": r"post-?operative|post-?surgical|status post|\bs/p\b|recent(?:ly)? (?:surgery|operation|procedure|laparo)|after (?:surgery|the procedure)|surgery|surgical|laparotom|laparoscop|resection|ectomy|anastomo|ostomy|drain|incision|procedure",
 "kw_expected": r"expected|anticipated|not unexpected|compatible with (?:recent )?(?:surgery|postoperative)|consistent with (?:recent )?(?:surgery|post)",
 "kw_drain": r"drain|catheter|tube|ostomy|\bpeg\b|gastrostomy|jejunostomy|nephrostomy",
 "kw_small": r"small(?! bowel| intestin)|trace|tiny|minimal|minute|few|scattered|punctate|bubble|single (?:focus|locule)|focus of|foci of|locules? of|mild",
 "kw_large": r"large(?! bowel| intestin)|moderate|extensive|substantial|significant|massive|marked|copious|considerable",
 "kw_decreasing": r"decreas|improv|resolv|smaller|less(?:er)? (?:amount|free)|interval",
 "kw_increasing": r"increas|new\b|enlarg|worse",
 "kw_retroperi": r"retroperiton",
 "kw_subdiaph": r"subdiaphragm|perihepatic|anterior to the liver|beneath the (?:anterior )?abdominal wall|nondependent|anterior abdom",
 "kw_pelvic": r"pelvi",
 "kw_wall": r"abdominal wall|subcutaneous|incision|soft tissues? of the",
 "kw_perf": r"perforat|ruptur|leak|dehiscen",
 "kw_pneumatosis": r"pneumatosis|portal venous gas|mesenteric venous gas",
 "kw_pneumobilia": r"pneumobilia|biliary (?:air|gas)|air in the biliary",
 "kw_uncertain": r"possible|possibly|may represent|may be|cannot (?:be )?exclude|questionable|equivocal|suspect|likely|probable|suggest|versus|vs\.?",
 "kw_history": r"history of|prior|previous|known",
 "kw_retro_only": r"retroperitoneal (?:free )?(?:air|gas)",
}
GLOBAL_SURG = re.compile(r"post-?operative|post-?surgical|status post|\bs/p\b|anastomo|ostomy|surgical (?:clips|changes|drain)|staple|laparotom|resection|drain", re.I)
POS_SENT = re.compile(r"pneumoperitoneum|(?:free|extraluminal|intraperitoneal|retroperitoneal|subdiaphragmatic|intra-abdominal) (?:intraperitoneal |intra-abdominal |abdominal )?(?:air|gas)", re.I)
NEGW = re.compile(r"\bno\b|without|negative|not\b|resolved|absence|free of|nor\b", re.I)
for sid, d in rows.items():
    rr = rep.get(sid) or {}
    txt = rr.get("report", "") if rr.get("split", d["split"]) in OK else ""
    sents = re.split(r"(?<=[.;])\s+|\n", txt)
    asents = [s for s in sents if AIRS.search(s)]
    t = " ".join(asents) + " " + d["quote"]
    for k, p in pat.items():
        src = d["quote"] if k in ("kw_small", "kw_large") else t   # amount: Qwen free-air quote only
        d[k] = int(bool(re.search(p, src, re.I)))
    d["rep_surgery"] = int(bool(GLOBAL_SURG.search(txt)))
    # possibly-missed positive: report has an affirmative free-air sentence without negation words
    d["rep_affirm_fa"] = int(any(POS_SENT.search(s) and not NEGW.search(s) for s in sents))
    d["rep_neg_fa"] = int(any(POS_SENT.search(s) and NEGW.search(s) for s in sents) or bool(re.search(r"no (?:free |extraluminal )?(?:intraperitoneal )?(?:air|gas)|pneumoperitoneum", txt, re.I) and not d["rep_affirm_fa"]))
    d["has_report"] = int(bool(txt))
    d["quote_len"] = len(d["quote"])
df = pd.DataFrame(rows.values())
df.drop(columns=["quote"]).to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby("split").size().to_dict())
t = df[df.split == "train"]
print("train pos", int(t.y_fa.sum()), "neg", int((t.y_fa == 0).sum()), "unclear", int(t.y_fa.isna().sum()))
print("pos postop:", t[t.y_fa == 1].fa_postop.value_counts().to_dict())
print("pos kw:", t[t.y_fa == 1][[c for c in df.columns if c.startswith("kw_")]].mean().round(3).to_dict())
print("neg kw:", t[t.y_fa == 0][[c for c in df.columns if c.startswith("kw_")]].mean().round(3).to_dict())
print("rep_surgery pos/neg", t[t.y_fa == 1].rep_surgery.mean().round(3), t[t.y_fa == 0].rep_surgery.mean().round(3))
print("rep_affirm_fa in neg:", int(t[t.y_fa == 0].rep_affirm_fa.sum()), "by cat", t[t.y_fa == 0].groupby("fa_cat").rep_affirm_fa.sum().to_dict())
print("has_report", t.has_report.mean())
