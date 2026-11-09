"""bd15 id priority (run on A6000 in outputs_final/region_units). Train/val only: labels_bd.csv holds train/val rows only
(asserted) and is filtered by split before y is used. tier0 = all val + all train positives + 2 train negatives per positive
matched on sex x 10-y age bin (seed 0); tier1 = remaining train negatives (random order). Within tier0 the order is
interleaved (val / train pos / train neg) so a partial tier0 is still balanced.
Usage: bd15_ids.py OUT_CSV"""
import sys, numpy as np, pandas as pd
lab = pd.read_csv("probe_inputs/labels_bd.csv"); assert set(lab.split) <= {"train", "val"}
lab = lab[lab.split.isin(["train", "val"])]
meta = pd.read_csv("../biliary_improve/meta_trval.csv")[["study_id", "Age", "Gender"]]
d = lab.merge(meta, on="study_id", how="left")
d["ab"] = (d.Age.fillna(d.Age.median()) // 10).clip(1, 9).astype(int); d["sx"] = d.Gender.fillna("U")
rng = np.random.default_rng(0)
val = d[d.split == "val"].sample(frac=1, random_state=0)
tr = d[d.split == "train"]; pos = tr[tr.y == 1].sample(frac=1, random_state=1); neg = tr[tr.y == 0]
picked = []
for (ab, sx), g in pos.groupby(["ab", "sx"]):
    pool = neg[(neg.ab == ab) & (neg.sx == sx) & ~neg.study_id.isin(picked)]
    k = min(2 * len(g), len(pool)); picked += list(pool.sample(n=k, random_state=int(rng.integers(1e9))).study_id)
short = 2 * len(pos) - len(picked)
if short > 0:
    picked += list(neg[~neg.study_id.isin(picked)].sample(n=short, random_state=2).study_id)
negm = neg[neg.study_id.isin(picked)].sample(frac=1, random_state=3)
rest = neg[~neg.study_id.isin(picked)].sample(frac=1, random_state=4)
L = [val.study_id.tolist(), pos.study_id.tolist(), negm.study_id.tolist()]
order, idx = [], [0, 0, 0]; w = np.array([len(x) for x in L], float); w /= w.sum()
while any(idx[i] < len(L[i]) for i in range(3)):
    i = int(np.argmin([(idx[j] + 1) / (w[j] * 1e6) if idx[j] < len(L[j]) else np.inf for j in range(3)]))
    order.append((L[i][idx[i]], 0, ["val", "train_pos", "train_neg_matched"][i])); idx[i] += 1
order += [(s, 1, "train_neg_rest") for s in rest.study_id]
o = pd.DataFrame(order, columns=["study_id", "tier", "group"]); assert o.study_id.is_unique
o.to_csv(sys.argv[1], index=False)
print(o.groupby(["tier", "group"]).size().to_string(), len(o))
