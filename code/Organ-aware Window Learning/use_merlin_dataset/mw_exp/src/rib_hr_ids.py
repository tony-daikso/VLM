"""Experiment K (rib high-res path): prioritized id list. Train+val only.

Reads outputs_fracture/analysis_rib/table_parsed.pkl (built from train/val lines only; test ids were dropped before any
label/report line was parsed) and writes outputs_fracture/ribs_hr/ids_prio.csv with columns
study_id, split, tier, group.
  tier0: all train rib-fracture positives + 3x train negatives (2x age-matched to the positives, 1x hard negatives:
         age >= 75 / osteopenia yes|probably_yes / hardware in report) + all val fracture positives + 1500 random val others
  tier1: the remaining val cases
  tier2: the remaining train fracture positives (non-rib) + 3000 more random train negatives
"""
from pathlib import Path

import numpy as np
import pandas as pd

H = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture"
OUT = H / "ribs_hr"
OUT.mkdir(parents=True, exist_ok=True)
t = pd.read_pickle(H / "analysis_rib/table_parsed.pkl")
assert set(t.split) <= {"train", "val"}
spl = pd.read_csv(Path.home() / "Desktop/Tony/VLM/OWL/data/split.csv").set_index("study_id")["split"]
allv = spl[spl == "val"].index
rng = np.random.default_rng(0)

tr = t[t.split == "train"]
pos = tr[tr.rib == 1]
neg = tr[tr.y == 0]
hard_m = (neg.Age >= 75) | neg.osteopenia.isin(["yes", "probably_yes"]) | neg.p_hardware.astype(bool)
hard = neg[hard_m]
hard_pick = rng.choice(hard.index.values, len(pos), replace=False)
rest = neg.drop(hard_pick)
# age-matched: 2 negatives per positive from the same 5-year age bin (fallback: nearest age)
bins = (rest.Age // 5).astype("Int64")
pick = []
used = set()
for a in pos.Age.values:
    b = int(a // 5) if a == a else -1
    cand = rest.index[(bins == b).fillna(False).values]
    cand = [c for c in cand if c not in used]
    if len(cand) < 2:
        cand = [c for c in rest.iloc[np.argsort(np.abs(rest.Age.values - a))[:50]].index if c not in used]
    for c in rng.choice(cand, 2, replace=False):
        used.add(c); pick.append(c)

rows = [(i, "train", 0, "rib_pos") for i in pos.index]
rows += [(i, "train", 0, "neg_hard") for i in hard_pick]
rows += [(i, "train", 0, "neg_agematch") for i in pick]
va = t[t.split == "val"]
vpos = list(va.index[va.y == 1])
others = sorted(set(allv) - set(vpos))
vo = list(rng.choice(others, 1500, replace=False))
rows += [(i, "val", 0, "val_pos") for i in vpos] + [(i, "val", 0, "val_rand") for i in vo]
rows += [(i, "val", 1, "val_rest") for i in sorted(set(others) - set(vo))]
tpos2 = tr.index[(tr.y == 1) & (tr.rib != 1)]
rows += [(i, "train", 2, "nonrib_pos") for i in tpos2]
seen = {r[0] for r in rows}
more = [i for i in rng.permutation(neg.index.values) if i not in seen][:3000]
rows += [(i, "train", 2, "neg_more") for i in more]
df = pd.DataFrame(rows, columns=["study_id", "split", "tier", "group"]).drop_duplicates("study_id")
assert (spl.reindex(df.study_id).values == df.split.values).all()
# within tier: train first, then val, shuffled
df["r"] = rng.random(len(df))
df = df.sort_values(["tier", "split", "r"], ascending=[True, True, True]).drop(columns="r")
df.to_csv(OUT / "ids_prio.csv", index=False)
df.study_id.to_csv(OUT / "ids_prio.txt", index=False, header=False)
print(df.groupby(["tier", "split", "group"]).size().to_string())
print("hard pool", len(hard), "age pos/negpick", pos.Age.mean(), t.loc[pick, "Age"].mean(), t.loc[hard_pick, "Age"].mean())
