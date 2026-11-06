"""BD4 pilot: stratified TRAIN-only sample with inverse-probability weights (disjoint strata, assigned in order)."""
import sys, numpy as np, pandas as pd
W = sys.argv[1]
d = pd.read_csv(f"{W}/labels_trval.csv"); d = d[d.split == "train"]           # split filter first
d = d[d.y_bdd.notna()].copy()
stent = (d.kw_stent == 1) | (d.kw_pneumobilia == 1)
pos, neg = d.y_bdd == 1, d.y_bdd == 0
sev = np.where(d.kw_marked == 1, "marked", np.where(d.kw_mild == 1, "mild", "unstated"))
d["stratum"] = np.select([pos & (d.bd_site == "extrahepatic"), pos & stent, pos & (sev == "mild"), pos & (sev == "marked"), pos,
                          neg & stent, neg & (d.bd_cat == "probably_no"), neg & (d.y_chole == 1), neg],
                         ["P_extra_only", "P_stent_air", "P_mild", "P_marked", "P_unstated",
                          "N_stent_air", "N_probably_no", "N_chole", "N_other"], "x")
want = dict(P_extra_only=80, P_stent_air=30, P_mild=110, P_marked=50, P_unstated=40, N_stent_air=30, N_probably_no=50, N_chole=60, N_other=130)
rows = []
for s, n in want.items():
    g = d[d.stratum == s]; k = min(n, len(g)); smp = g.sample(k, random_state=0).copy(); smp["w"] = len(g) / k; rows.append(smp)
    print(s, len(g), k)
P = pd.concat(rows)[["study_id", "stratum", "w", "y_bdd"]]
P.to_csv(f"{W}/bd4_pilot_ids.csv", index=False); P.study_id.to_csv(f"{W}/bd4_pilot_ids.txt", index=False, header=False)
print(len(P), int(P.y_bdd.sum()))
