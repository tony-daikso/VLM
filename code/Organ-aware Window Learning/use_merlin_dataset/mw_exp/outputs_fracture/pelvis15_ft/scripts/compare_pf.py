"""pelvis15_ft: score the train-CV-selected final_ft variant on val vs rp_both_aa (.882), with_rib (.878) and S2 lse+any (.862):
AUROC + paired bootstrap Δ (2000), age-matched (±2 y) Δ, subgroups (point + CI + paired Δ). val only (test never read).
usage: compare_pf.py NEW_TAG [REF_TAG=rp_both_aa ...]  -> final_ft/NEW_TAG/compare_vs_882.md"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
sys.path.insert(0, str(MW / "outputs_fracture/sitewise/scripts")); sys.path.insert(0, str(MW / "src"))
import sitewise as S
from age_shortcut import matched_auc
from fracture_eval_oof import boot_diff
from probe import auroc
FT = MW / "outputs_fracture/final_ft"; new = sys.argv[1]; refs = sys.argv[2:] or ["rp_both_aa", "with_rib"]
def load(t, key="selected"):
    z = np.load(FT / t / "scores.npz", allow_pickle=True); sp = z["split"].astype(str)
    return pd.Series(z[key][sp == "val"], index=z["ids"].astype(str)[sp == "val"]), pd.Series(z[key][sp == "train"], index=z["ids"].astype(str)[sp == "train"])
df, _ = S.build_frame(); va = df[df.split == "val"]
nv, nt = load(new); P = {new: nv}; T = {new: nt}
for r in refs: P[r], T[r] = load(r)
P["S2·lse+any(.862)"], T["S2·lse+any(.862)"] = load(new, "ref")
ids = sorted(set.intersection(*[set(p.index) for p in P.values()])); va = va.loc[ids]
y = va.y.values.astype(int); age = va.age.values.astype(float); neg = y == 0
G = {g: (y == 1) & np.asarray(f(va), bool) for g, f in S.SUB.items()}
sel = json.load(open(FT / new / "summary.json"))
L = [f"## {new} vs 參考（val 同列 n={len(y)}，骨折 {y.sum()}；test 未讀取）\n", f"選出設定：{' · '.join(sel['selected'])}，train CV {sel['selected_train_cv']:.4f}\n",
     "| 模型 | train CV | val [95% CI] | 年齡配對 | Δ val（new − ref）[95% CI] | Δ 年齡配對 [95% CI] |", "|---|---|---|---|---|---|"]
a = P[new].loc[ids].values
for name, p in P.items():
    pv = p.loc[ids].values; cv = json.load(open(FT / name / "summary.json"))["selected_train_cv"] if (FT / name / "summary.json").exists() else sel["ref_train_cv"]
    if name == new: dd = ("—", "—")
    else: b = boot_diff(y, a, pv, age, 2.0, S.NBOOT); dd = (S.fd(b["auc"]), S.fd(b["matched"]))
    L.append(f"| {name} | {cv:.4f} | {S.f3(S.boot_ci(y, pv))} | {matched_auc(y, pv, age, 2.0):.3f} | {dd[0]} | {dd[1]} |")
L += ["\n### 子群組（new [95% CI]；Δ 為配對 bootstrap）", "| 子群 | n 陽性 | new | " + " | ".join(f"Δ vs {r}" for r in P if r != new) + " |", "|---" * (3 + len(P) - 1) + "|"]
for g, m in G.items():
    s_ = m | neg; row = [g, str(int(m.sum())), S.f3(S.boot_ci(m[s_].astype(int), a[s_]))]
    for r in P:
        if r == new: continue
        row.append(S.fd(boot_diff(m[s_].astype(int), a[s_], P[r].loc[ids].values[s_], age[s_], 2.0, S.NBOOT)["auc"]))
    L.append("| " + " | ".join(row) + " |")
L += ["\n### 子群組點估計（各模型）", "| 模型 | " + " | ".join(G) + " |", "|---" * (1 + len(G)) + "|"]
for name, p in P.items():
    pv = p.loc[ids].values; L.append(f"| {name} | " + " | ".join(f"{auroc(m[m | neg].astype(float), pv[m | neg]):.4f}" for m in G.values()) + " |")
(FT / new / "compare_vs_882.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
