"""H7 localization: weak instance-level rib-unit probe (case label only, no side/level supervision), 5-fold OOF on TRAIN.
argmax rib unit in held-out positives vs reported side/level. Val/test never used."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
OUT = Path(__file__).parent
B = OUT.parent / "bones"
d = pd.read_pickle(OUT / "table_parsed.pkl")
z = np.load(B / "rib_matrix.npz", allow_pickle=True)
ids = list(z["ids"]); units = list(z["units"]); P = z["P"]
E = z["E"]                                          # [N,18,3,2,448] fp16
Sx = d.reindex(ids)
tr = np.where((Sx.split == "train").values & Sx.rib.notna().values)[0]
X = np.concatenate([E[:, :, 2, 0], E[:, :, 2, 1], E[:, :, 0, 0]], -1).astype(np.float32)   # bone mean, bone max, baseline mean
del E
y = Sx.rib.values
res = {}
U = len(units)
def std(X, fitrows):
    Xs = np.zeros_like(X)
    for u in range(U):
        m = P[fitrows, u]; r = X[fitrows][m, u]
        mu, sd = r.mean(0), r.std(0) + 1e-6
        Xs[:, u] = (X[:, u] - mu) / sd * P[:, u, None]
    return Xs
oof_unit = np.full((len(ids), U), np.nan, np.float32)
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y[tr])):
    t0 = time.time(); fit, hold = tr[a], tr[b]
    Xs = std(X, fit)
    ii, uu = np.where(P[fit]); Xi = Xs[fit][ii, uu]; yi = y[fit][ii]
    # balance: weight cases so positives total == negatives total
    w = np.where(yi == 1, 0.5 / (yi == 1).sum(), 0.5 / (yi == 0).sum()) * len(yi)
    clf = LogisticRegression(C=0.002, max_iter=300).fit(Xi, yi, sample_weight=w)
    s = clf.decision_function(Xs[hold].reshape(-1, X.shape[-1])).reshape(len(hold), U)
    oof_unit[hold] = np.where(P[hold], s, np.nan)
    print(k, f"{time.time() - t0:.0f}s", flush=True)
case = np.nanmax(oof_unit, 1)
from analyze_util import auc
res["instance_probe_case_auc_rib_train"] = auc(y[tr], case[tr])
# localization
side_of = lambda u: "L" if "left" in u else "R"
lvl_of = lambda u: (1, 4) if u.endswith("1_4") else (int(u.split("_")[-1]),) * 2
rows = []
for i in tr:
    if y[i] != 1: continue
    lv = Sx.iloc[i].rib_levels
    s = oof_unit[i]
    if np.all(np.isnan(s)): continue
    j = int(np.nanargmax(s)); u = units[j]
    pres = [k for k in range(U) if P[i, k]]
    toks = [(t[0], int(t[1:])) for t in lv.split(";")] if lv else []
    sides = {t[0] for t in toks} | ({Sx.iloc[i].side} if Sx.iloc[i].side in "LRB" else set())
    def side_ok(uu): return (not sides) or ("B" in sides) or (side_of(uu) in sides) or ("?" in sides and len(sides) == 1)
    def lvl_ok(uu, tol=1):
        lo, hi = lvl_of(uu)
        return any(lo - tol <= l <= hi + tol and (sd in "B?" or sd == side_of(uu)) for sd, l in toks)
    rows.append(dict(id=ids[i], top_unit=u, sides="".join(sorted(sides)), levels=lv,
                     side_known=bool(sides - {"?", "B"}), side_ok=side_ok(u), lvl_known=bool(toks), lvl_ok=lvl_ok(u) if toks else None,
                     chance_side=np.mean([side_ok(units[k]) for k in pres]), chance_lvl=np.mean([lvl_ok(units[k]) for k in pres]) if toks else None,
                     top_score_pct=float((case[tr] < case[i]).mean())))
L = pd.DataFrame(rows); L.to_csv(OUT / "localization_train_pos.csv", index=False)
sk = L[L.side_known & ~L.sides.str.contains("B")]; lk = L[L.lvl_known]
res.update(n_pos=len(L), side_known=len(sk), side_acc=float(sk.side_ok.mean()), side_chance=float(sk.chance_side.mean()),
           lvl_known=len(lk), lvl_acc_pm1=float(lk.lvl_ok.astype(float).mean()), lvl_chance_pm1=float(lk.chance_lvl.astype(float).mean()))
hi = L[L.top_score_pct >= 0.9]
hk = hi[hi.lvl_known]; hs = hi[hi.side_known & ~hi.sides.str.contains("B")]
res.update(detected_side_acc=float(hs.side_ok.mean()) if len(hs) else None, detected_n_side=len(hs),
           detected_lvl_acc=float(hk.lvl_ok.astype(float).mean()) if len(hk) else None, detected_n_lvl=len(hk))
# per-unit AUROC: unit containing reported fracture vs same unit in negatives
pu = []
neg = tr[y[tr] == 0]
for j, u in enumerate(units):
    posj = [i for i in tr if y[i] == 1 and P[i, j] and Sx.iloc[i].rib_levels and any(
        (t[0] in "B?" or t[0] == side_of(u)) and lvl_of(u)[0] <= int(t[1:]) <= lvl_of(u)[1] for t in Sx.iloc[i].rib_levels.split(";"))]
    nj = [i for i in neg if P[i, j]]
    if len(posj) >= 5:
        pu.append(dict(unit=u, n_pos=len(posj), auc_unit=auc(np.r_[np.ones(len(posj)), np.zeros(len(nj))], np.r_[oof_unit[posj, j], oof_unit[nj, j]])))
res["per_unit_auc"] = pu
json.dump(res, open(OUT / "results_loc.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
