"""push90 task B: posterior-element (PE) weak labels from TRAIN reports only. Official labels untouched.
split.csv -> train ids first; labels jsonl: only train lines parsed; reports.json: only train reports kept.
t_pe: 1 = official fracture positive AND report mentions a transverse/spinous process, pars/spondylolysis, lamina,
pedicle, facet/articular-process or posterior-element fracture (non-negated sentence); 0 = official negative; NaN = other
positives (ignored by the PE head). Side (sideL/sideR) and level lists (T8..L5 -> unit index 0..9) from the same sentences.
Also writes ids_trval.txt (train+val ids, train first) and ids_tier.txt (train PE+ + all other train positives(<=600) +
1500 random train negatives) for crop ordering. Val/test labels and reports never read.
usage: pe_labels.py OUTDIR"""
import json, re, sys, os, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM"); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
spl = pd.read_csv(H + "/OWL/data/split.csv"); spl = spl[spl.split.isin(["train", "val"])]
TR = set(spl.loc[spl.split == "train", "study_id"])
POS, NEG = {"yes", "probably_yes"}, {"no", "probably_no", "not_mentioned"}
rx = re.compile(r'"study_id"\s*:\s*"([^"]+)"'); Y = {}
for line in open(H + "/OWL/data/labels_disease14_all.jsonl"):
    m = rx.search(line)
    if not m or m.group(1) not in TR: continue
    r = json.loads(line); c = r["answer"]["fracture"]["category"]
    Y[r["study_id"]] = 1.0 if c in POS else 0.0 if c in NEG else np.nan
R = json.load(open(H + "/merlin_disease14/reports.json")); REP = {k: R[k]["report"] for k in TR if k in R}; del R
PE = re.compile(r"transverse process|spinous process|\bpars\b|spondylolys|posterior element|lamina|pedicle|facet|articular process", re.I)
FX = re.compile(r"fractur|\bfx\b|spondylolys|pars defect|pars interarticularis defect|lysis", re.I)
NEGX = re.compile(r"\b(no|without|negative for|not seen|no evidence of|no acute)\b[^.;]*?(fractur|fx|spondylolys|defect)", re.I)
LV = ["T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5"]
LVRX = re.compile(r"\b([TL])\s?(\d{1,2})\b(?:\s*(?:-|–|through|to)\s*([TL])?\s?(\d{1,2})\b)?", re.I)


def levels(s):
    out = set()
    for a, b, c, d in LVRX.findall(s):
        a = a.upper(); b = int(b); c = (c or a).upper(); d = int(d) if d else b
        def idx(p, n): return {"T": n - 8, "L": n + 4}[p]  # T8..T12 -> 0..4, L1..L5 -> 5..9
        i0, i1 = idx(a, b), idx(c, d)
        for i in range(min(i0, i1), max(i0, i1) + 1):
            if 0 <= i <= 9: out.add(i)
    return out


rows = []
for sid in sorted(TR):
    y = Y.get(sid, np.nan); rep = REP.get(sid, ""); L = R_ = 0; lv = {"L": set(), "R": set(), "U": set()}; hit = 0
    for s in re.split(r"(?<=[.;])\s+|\n", rep):
        if not (PE.search(s) and FX.search(s)) or NEGX.search(s): continue
        hit = 1; sl = re.search(r"\bleft\b", s, re.I) is not None; sr = re.search(r"\bright\b", s, re.I) is not None
        bil = re.search(r"bilateral", s, re.I) is not None
        if bil: sl = sr = True
        L |= sl; R_ |= sr; key = "L" if sl and not sr else "R" if sr and not sl else "U"
        lv[key] |= levels(s)
    t = (1.0 if hit else np.nan) if y == 1 else (0.0 if y == 0 else np.nan)
    rows.append(dict(study_id=sid, y=y, t_pe=t, pe_mention=hit, sideL=float(L) if t == 1 else (0.0 if t == 0 else np.nan),
                     sideR=float(R_) if t == 1 else (0.0 if t == 0 else np.nan),
                     levL=";".join(map(str, sorted(lv["L"] | lv["U"]))) if t == 1 else "",
                     levR=";".join(map(str, sorted(lv["R"] | lv["U"]))) if t == 1 else ""))
D = pd.DataFrame(rows); D.to_csv(OUT + "/pe_labels_train.csv", index=False)
pp = D[D.t_pe == 1]
print("train", len(D), "y+", int((D.y == 1).sum()), "PE+", len(pp), "sideL", int(pp.sideL.sum()), "sideR", int(pp.sideR.sum()),
      "with level", int(((pp.levL != "") | (pp.levR != "")).sum()), "PE mention among y=0", int(((D.y == 0) & (D.pe_mention == 1)).sum()))
rng = np.random.default_rng(0)
otherpos = D[(D.y == 1) & (D.t_pe != 1)].study_id.values; neg = D[D.y == 0].study_id.values
tier = list(pp.study_id) + list(rng.choice(otherpos, min(600, len(otherpos)), replace=False)) + list(rng.choice(neg, 1500, replace=False))
open(OUT + "/ids_tier.txt", "w").write("\n".join(tier) + "\n")
va = sorted(spl.loc[spl.split == "val", "study_id"]); rest = [s for s in sorted(TR) if s not in set(tier)]
open(OUT + "/ids_trval.txt", "w").write("\n".join(tier + rest + va) + "\n")
print("tier", len(tier), "trval", len(tier) + len(rest) + len(va))
