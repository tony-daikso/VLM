"""Per-study osteopenia label + report keyword flags for train+val ONLY (split filtered before reading any label/report).
No report text is written out, only binary flags."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
OUT = R + "OWL/mw_exp/outputs_final/analysis_osteopenia/"
OK = {"train", "val"}
POS = {"yes", "probably_yes"}; NEG = {"no", "probably_no", "not_mentioned"}
def b(c): return 1.0 if c in POS else 0.0 if c in NEG else np.nan
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]
rows = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    a = r["answer"]; o = a["osteopenia"]
    rows[r["study_id"]] = dict(study_id=r["study_id"], split=r["split"], op_cat=o["category"], y_op=b(o["category"]),
        y_frac=b(a["fracture"]["category"]), frac_cat=a["fracture"]["category"], y_ath=b(a["atherosclerosis"]["category"]),
        op_quote=" ".join(o.get("quotes") or []))
rep = json.load(open(R + "merlin_disease14/reports.json"))
pat = {
 "kw_osteopenia": r"osteopeni",
 "kw_osteoporosis": r"osteopor",
 "kw_demineral": r"demineraliz|bone mineral|bone density|decreased density|osseous density|hypodense bone|low bone",
 "kw_diffuse": r"diffuse(?:ly)? (?:osteopeni|demineral|osteopor)|generalized",
 "kw_severe": r"(?:severe|marked|significant|profound|advanced)(?:ly)?\s+(?:diffuse\s+|generalized\s+)?(?:osteopeni|osteopor|demineral)",
 "kw_mild": r"(?:mild|minimal|slight|subtle)(?:ly)?\s+(?:diffuse\s+|generalized\s+)?(?:osteopeni|osteopor|demineral)",
 "kw_moderate": r"moderate(?:ly)?\s+(?:diffuse\s+|generalized\s+)?(?:osteopeni|osteopor|demineral)",
 "kw_compression": r"compression (?:fracture|deformit)|wedge|vertebral (?:body )?(?:fracture|height loss)|height loss|endplate (?:fracture|depression)|insufficiency fracture",
 "kw_kypho": r"kyphoplast|vertebroplast|cement augment|augmentation",
 "kw_degen": r"(?:spine|spinal|lumbar|thoracic|disc|facet|vertebra)[^.]{0,60}degenerative|degenerative[^.]{0,60}(?:spine|spinal|lumbar|thoracic|disc|facet)|spondylosis|osteophyt|disc space narrow|facet arthropathy|spondylolisthesis",
 "kw_scoliosis": r"scoliosis|kyphosis",
 "kw_hardware": r"\b(?:spinal fusion|pedicle screws?|rods?|laminectomy|arthroplasty|orthopedic hardware|surgical hardware)\b",
 "kw_bone_mets": r"\blytic|blastic|osseous metasta|bone metasta|skeletal metasta|myeloma",
 "kw_age_related": r"age[- ]related|senile|for age",
}
for sid, d in rows.items():
    e = rep.get(sid) or {}
    txt = e.get("report", "") if e.get("split", d["split"]) in OK else ""
    t = txt + " " + d["op_quote"]
    for k, p in pat.items():
        d[k] = int(bool(re.search(p, t, re.I)))
    d["has_report"] = int(bool(txt))
df = pd.DataFrame(rows.values()).drop(columns=["op_quote"])
df.to_csv(OUT + "labels_trval.csv", index=False)
print(df.groupby(["split", "op_cat"]).size().to_dict())
t = df[df.split == "train"]
print("train pos", int(t.y_op.sum()), "neg", int((t.y_op == 0).sum()))
print("pos kw:", t[t.y_op == 1][[c for c in df.columns if c.startswith("kw_")]].mean().round(3).to_dict())
print("neg kw:", t[t.y_op == 0][[c for c in df.columns if c.startswith("kw_")]].mean().round(4).to_dict())
