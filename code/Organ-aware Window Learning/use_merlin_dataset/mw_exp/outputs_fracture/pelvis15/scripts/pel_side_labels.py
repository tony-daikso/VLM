"""pelvis15: weak pelvic-fracture side labels from TRAIN reports only (val/test reports dropped right after loading, never
inspected). t_pel==1 rows: sentences with a pelvic/sacral/proximal-femur term + a fracture word (not negated) -> side
R/L (left/right/bilateral; sacral midline/unspecified -> unknown). t_pel==0 rows: both sides 0.
Out: OUT/pel_side.csv  study_id, t_pel, sideR, sideL (1/0/NaN), n_sent
usage: pel_side_labels.py LABELS_CSV REPORTS_JSON SPLIT_CSV OUT_DIR"""
import sys, re, json, os
import numpy as np, pandas as pd
lab_p, rep_p, spl_p, out = sys.argv[1:5]
spl = pd.read_csv(spl_p); TR = set(spl.loc[spl.split == "train", "study_id"])
lab = pd.read_csv(lab_p); lab = lab[(lab.split == "train") & lab.study_id.isin(TR)].copy()
R = json.load(open(rep_p)); R = {k: R[k]["report"] for k in lab.study_id if k in R}  # train only; rest discarded
BONE = re.compile(r"pubi|ram(us|i)\b|acetabul|sacr|ilia|iliac|ischi|femoral (neck|head)|femur|intertrochant|subcapital|"
                  r"trochant|hip\b|pelvi|symphys", re.I)
FX = re.compile(r"fractur|callus|healed|healing|deformit|nonunion|non-union|displaced|insufficiency", re.I)
NEGP = re.compile(r"\bno (?:acute |displaced |definite |evidence of )*(?:\w+ )?fractur|without (?:acute )?fractur|negative for fractur", re.I)
rows = []
for _, r in lab.iterrows():
    sid, t = r.study_id, r.t_pel
    d = {"study_id": sid, "t_pel": t, "sideR": np.nan, "sideL": np.nan, "n_sent": 0}
    if t == 0:
        d.update(sideR=0.0, sideL=0.0)
    elif t == 1 and sid in R:
        Ls, Rs = 0, 0
        for s in re.split(r"(?<=[.;])\s+|\n", R[sid]):
            if not BONE.search(s) or not FX.search(s) or NEGP.search(s) or re.search(r"\bribs?\b|vertebra|compression", s, re.I):
                continue
            d["n_sent"] += 1
            for w in re.findall(r"\b(left|right|bilateral)\b", s, re.I):
                w = w.lower(); Ls |= w in ("left", "bilateral"); Rs |= w in ("right", "bilateral")
        if Ls or Rs:
            d["sideR"], d["sideL"] = float(Rs), float(Ls)
    rows.append(d)
o = pd.DataFrame(rows); os.makedirs(out, exist_ok=True); o.to_csv(f"{out}/pel_side.csv", index=False)
p = o[o.t_pel == 1]
print("train rows", len(o), "pel+", len(p), "with sentence", int((p.n_sent > 0).sum()), "side known", int(p.sideR.notna().sum()),
      "R only", int(((p.sideR == 1) & (p.sideL == 0)).sum()), "L only", int(((p.sideR == 0) & (p.sideL == 1)).sum()),
      "both", int(((p.sideR == 1) & (p.sideL == 1)).sum()))
