"""Paired val comparison of final_ft stacks (A6000 CPU). Each tag's selected predictions come from final_ft/<tag>/scores.npz
(selection inside each tag was by train CV only). Prints/writes Δ AUROC and Δ age-matched AUROC (paired bootstrap, same val rows)
for NEW vs each reference tag, plus the sub-groups of NEW. usage: compare_stacks.py NEW REF1 [REF2 ...] > out.md"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"; FT = MW / "outputs_fracture/final_ft"
sys.path.insert(0, str(MW / "outputs_fracture/sitewise/scripts")); sys.path.insert(0, str(MW / "src"))
import sitewise as S
from age_shortcut import matched_auc
from fracture_eval_oof import boot_diff
from probe import auroc
new, refs = sys.argv[1], sys.argv[2:]
df, _ = S.build_frame(); assert set(df.split) <= {"train", "val"}


def load(tag):
    z = np.load(FT / tag / "scores.npz", allow_pickle=True); v = z["split"] == "val"
    return pd.Series(z["selected"][v], index=z["ids"][v].astype(str)), pd.Series(z["selected"][~v], index=z["ids"][~v].astype(str)), str(z["selected_name"])


P = {t: load(t) for t in [new] + refs}
ids = P[new][0].index
for t in refs: ids = ids.intersection(P[t][0].index)
va = df.loc[ids]; y = va["y"].values.astype(int); age = va["age"].values.astype(float)
tri = P[new][1].index
for t in refs: tri = tri.intersection(P[t][1].index)
ytr = df.loc[tri, "y"].values.astype(int)
out = [f"## {new} vs 參考（val 同列 n={len(ids)}，骨折 {y.sum()}；train OOF/CV 同列 n={len(tri)}）\n",
       "| 模型 | 選出設定 | train CV | val [95% CI] | 年齡配對 | Δ val（new − ref）[95% CI] | Δ 年齡配對 [95% CI] |", "|---|---|---|---|---|---|---|"]
res = {}
for t in [new] + refs:
    p = P[t][0].loc[ids].values; cv = auroc(ytr, P[t][1].loc[tri].values)
    row = f"| {t} | {P[t][2]} | {cv:.4f} | {S.f3(S.boot_ci(y, p))} | {matched_auc(y, p, age, 2.0):.3f} |"
    if t == new: row += " — | — |"
    else:
        b = boot_diff(y, P[new][0].loc[ids].values, p, age, 2.0, S.NBOOT); row += f" {S.fd(b['auc'])} | {S.fd(b['matched'])} |"
        res[t] = {k: (list(v) if hasattr(v, '__len__') else v) for k, v in b.items()}
    out.append(row)
neg = y == 0; pn = P[new][0].loc[ids].values
out += ["", "### 子群組（new；Δ vs 各參考）", "| 子群 | n 陽性 | new [95% CI] | " + " | ".join(f"Δ vs {t}" for t in refs) + " |", "|---|---|---|" + "---|" * len(refs)]
for g, f in S.SUB.items():
    m = (y == 1) & np.asarray(f(va), bool); s_ = m | neg
    cells = [S.fd(boot_diff(m[s_].astype(int), pn[s_], P[t][0].loc[ids].values[s_], age[s_], 2.0, S.NBOOT)["auc"]) for t in refs]
    out.append(f"| {g} | {int(m.sum())} | {S.f3(S.boot_ci(m[s_].astype(int), pn[s_]))} | " + " | ".join(cells) + " |")
print("\n".join(out))
json.dump(res, open(FT / new / f"compare_{'_'.join(refs)}.json", "w"), indent=1, default=str)
