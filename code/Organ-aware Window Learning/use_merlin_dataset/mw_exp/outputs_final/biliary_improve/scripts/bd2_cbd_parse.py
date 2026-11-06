"""BD2: parse common-bile-duct diameter (mm) from TRAIN/VAL report text (split filtered BEFORE any report is read).
Combines the regex value with the Qwen label detail (labels_trval.csv cbd_mm). Output cbd_trval.csv:
study_id, split, y_bdd, cbd_qwen, cbd_rx, cbd_mm (final: regex if present else qwen), cbd_src. No free text written."""
import json, re, numpy as np, pandas as pd
R = "/home/advantech/Desktop/Tony/VLM/"
A = R + "OWL/mw_exp/outputs_final/analysis_biliary/"; O = R + "OWL/mw_exp/outputs_final/biliary_improve/"
lab = pd.read_csv(A + "labels_trval.csv")[["study_id", "split", "y_bdd", "cbd_mm", "bd_site"]]
lab = lab[lab.split.isin(["train", "val"])]
ok = set(lab.study_id)
rep = json.load(open(R + "merlin_disease14/reports.json"))
KW = re.compile(r"common bile duct|common hepatic duct|\bcbd\b|common duct|extrahepatic (?:bile |biliary )?duct|choledoch", re.I)
NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", re.I)
BAD = re.compile(r"stone|calcul|lith|mass|lesion|cyst|nodule|stent|polyp|tumou?r", re.I)
out = []
for sid in lab.study_id:
    if sid not in ok:
        continue
    txt = (rep.get(sid) or {}).get("report", "") or ""
    vals = []
    for s in re.split(r"(?<=[.;])\s+|\n", txt):
        km = KW.search(s)
        if not km:
            continue
        tail = s[km.start():]
        for nm in NUM.finditer(tail):
            pre = tail[:nm.start()]
            if len(pre) > 90:
                break
            if BAD.search(pre[-40:]) and not re.search(r"duct\w*\s+(?:measur|is|of|to|up)", pre[-40:], re.I):
                continue
            v = float(nm.group(1)) * (10 if nm.group(2).lower() == "cm" else 1)
            if 1.5 <= v <= 35:
                vals.append(v)
    out.append(dict(study_id=sid, cbd_rx=max(vals) if vals else np.nan))
d = lab.merge(pd.DataFrame(out), on="study_id", how="left").rename(columns={"cbd_mm": "cbd_qwen"})
d["cbd_mm"] = d.cbd_rx.where(d.cbd_rx.notna(), d.cbd_qwen)
d["cbd_src"] = np.where(d.cbd_rx.notna(), "regex", np.where(d.cbd_qwen.notna(), "qwen", "none"))
d.to_csv(O + "cbd_trval.csv", index=False)
both = d.cbd_rx.notna() & d.cbd_qwen.notna()
print("agreement regex vs qwen (|diff|<=0.5mm):", round(float(((d.cbd_rx - d.cbd_qwen).abs() <= 0.5)[both].mean()), 3), "n", int(both.sum()))
for s in ["train", "val"]:
    t = d[d.split == s]
    print(s, "pos with mm", int((t.y_bdd == 1)[t.cbd_mm.notna()].sum()), "/", int((t.y_bdd == 1).sum()),
          "| neg with mm", int((t.y_bdd == 0)[t.cbd_mm.notna()].sum()),
          "| median pos", t[t.y_bdd == 1].cbd_mm.median(), "neg", t[t.y_bdd == 0].cbd_mm.median())
