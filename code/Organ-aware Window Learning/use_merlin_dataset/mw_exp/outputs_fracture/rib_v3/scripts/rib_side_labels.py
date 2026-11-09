"""rib_v3: weak rib-side / rib-level labels from TRAIN reports only (val/test reports are dropped right after loading,
never inspected). Rows: train ids of rib_ft/labels.csv. For t_rib==1 rows: sentences mentioning a rib fracture ->
side L/R (left/right/bilateral) and rib levels (ordinals, digits, ranges). t_rib==0 rows: both sides 0, no levels.
Out: OUT/rib_side.csv  study_id, t_rib, sideL, sideR (1/0/NaN unknown), levL, levR ("3;4;5" or ""), n_sent
usage: rib_side_labels.py LABELS_CSV REPORTS_JSON SPLIT_CSV OUT_DIR"""
import sys, re, json, os
import numpy as np, pandas as pd
lab_p, rep_p, spl_p, out = sys.argv[1:5]
spl = pd.read_csv(spl_p); TR = set(spl.loc[spl.split == "train", "study_id"])
lab = pd.read_csv(lab_p); lab = lab[(lab.split == "train") & lab.study_id.isin(TR)].copy()
R = json.load(open(rep_p)); R = {k: R[k]["report"] for k in lab.study_id if k in R}  # train only; rest discarded
ORD = {w: i for i, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split(), 1)}
NUM = r"(?:(\d{1,2})(?:st|nd|rd|th)?|(" + "|".join(ORD) + r"))"
NUM1 = r"(?:(\d{1,2})(?:st|nd|rd|th)|(" + "|".join(ORD) + r"))"  # single level: digits need an ordinal suffix
FX = re.compile(r"fractur|callus|healed|healing|deformit|nonunion|non-union|displaced", re.I)
NEGP = re.compile(r"\bno (?:acute |displaced |definite |evidence of )*(?:rib )?fractur|without (?:acute )?fractur|negative for fractur", re.I)


def num(m):
    return int(m[0]) if m[0] else ORD[m[1].lower()]


def levels(s):
    out = set()
    for m in re.finditer(NUM + r"\s*(?:-|–|to|through|thru)\s*" + NUM, s, re.I):
        g = m.groups(); x, y = num(g[:2]), num(g[2:])
        if 1 <= x <= 12 and 1 <= y <= 12 and x <= y: out |= set(range(x, y + 1))
    for m in re.finditer(NUM1, s, re.I):
        v = num(m.groups())
        if 1 <= v <= 12: out.add(v)
    return out


rows = []
for _, r in lab.iterrows():
    sid, t = r.study_id, r.t_rib
    d = {"study_id": sid, "t_rib": t, "sideL": np.nan, "sideR": np.nan, "levL": "", "levR": "", "n_sent": 0}
    if t == 0:
        d.update(sideL=0.0, sideR=0.0)
    elif t == 1 and sid in R:
        L, Rr, lv = set(), set(), {"L": set(), "R": set()}
        for s in re.split(r"(?<=[.;])\s+|\n", R[sid]):
            if not re.search(r"\bribs?\b|costal", s, re.I) or not FX.search(s) or NEGP.search(s):
                continue
            d["n_sent"] += 1
            # split on side words so "left 5th and right 7th" assigns levels per side
            parts = re.split(r"\b(left|right|bilateral)\b", s, flags=re.I)
            cur = None
            for p in parts:
                pl = p.lower()
                if pl in ("left", "right", "bilateral"):
                    cur = pl
                    if cur in ("left", "bilateral"): L.add(1)
                    if cur in ("right", "bilateral"): Rr.add(1)
                    continue
                if cur in ("left", "right"):
                    lv["L" if cur == "left" else "R"] |= levels(p)
        if d["n_sent"]:
            if L or Rr:
                d["sideL"] = 1.0 if L else 0.0; d["sideR"] = 1.0 if Rr else 0.0
            d["levL"] = ";".join(map(str, sorted(lv["L"]))); d["levR"] = ";".join(map(str, sorted(lv["R"])))
    rows.append(d)
o = pd.DataFrame(rows); os.makedirs(out, exist_ok=True); o.to_csv(f"{out}/rib_side.csv", index=False)
p = o[o.t_rib == 1]
print("train rows", len(o), "rib+", len(p), "with sentence", int((p.n_sent > 0).sum()), "side known", int(p.sideL.notna().sum()),
      "L only", int(((p.sideL == 1) & (p.sideR == 0)).sum()), "R only", int(((p.sideL == 0) & (p.sideR == 1)).sum()),
      "both", int(((p.sideL == 1) & (p.sideR == 1)).sum()), "with levels", int(((p.levL != "") | (p.levR != "")).sum()))
