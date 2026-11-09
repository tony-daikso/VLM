"""push90b cmp_val2 = cmp_val + reference rp_a6mix3 (.897, push90/integ/rp_a6mix3).
push90: final val comparison of the selected integration variant vs references (rsP_mix_repl .892, rp_both_aa .882,
S2.lse+any .862). Scores from <dir>/scores.npz ('selected' column; S2 ref = 'ref' of rsP_mix_repl). Common val rows only.
Paired bootstrap (sitewise boot_diff, 2000) for AUROC and age-matched AUROC; subgroup AUROC (positives of the subgroup vs all
negatives) with paired Δ. Test never read. usage: cmp_val.py NEW_DIR NAME OUT_MD"""
import sys, numpy as np, pandas as pd
from pathlib import Path
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"; FR = MW / "outputs_fracture"
sys.path.insert(0, str(FR / "sitewise/scripts")); sys.path.insert(0, str(MW / "src"))
import sitewise as S
from age_shortcut import matched_auc
from fracture_eval_oof import boot_diff
from probe import auroc
new_dir, new_name, out_md = Path(sys.argv[1]), sys.argv[2], sys.argv[3]


def load(d, key="selected"):
    z = np.load(d / "scores.npz", allow_pickle=True); assert set(z["split"]) <= {"train", "val"}
    v = z["split"] == "val"; return pd.Series(z[key][v], index=z["ids"][v].astype(str))


df, _ = S.build_frame(); va = df[df.split == "val"].dropna(subset=["age"])
M = {new_name: load(new_dir), "rp_a6mix3 (.897)": load(FR / "push90/integ/rp_a6mix3"), "rsP_mix_repl (.892)": load(FR / "final_ft/rsP_mix_repl"), "rp_both_aa (.882)": load(FR / "final_ft/rp_both_aa"),
     "S2·lse+any (.862)": load(FR / "final_ft/rsP_mix_repl", "ref")}
ids = va.index
for s in M.values(): ids = ids.intersection(s.index)
va = va.loc[ids]; y = va["y"].values.astype(int); age = va["age"].values.astype(float); neg = y == 0
P = {k: v.loc[ids].values for k, v in M.items()}; p0 = P[new_name]
groups = {g: (y == 1) & np.asarray(f(va), bool) for g, f in S.SUB.items()}
rows = []
for k, p in P.items():
    r = {"模型": k, "val [95% CI]": S.f3(S.boot_ci(y, p)), "年齡配對": f"{matched_auc(y, p, age, 2.0):.3f}"}
    if k != new_name:
        b = boot_diff(y, p0, p, age, 2.0, S.NBOOT); r["Δ val（new − ref）"] = S.fd(b["auc"]); r["Δ 年齡配對"] = S.fd(b["matched"])
    else: r["Δ val（new − ref）"] = r["Δ 年齡配對"] = "—"
    rows.append(r)
sub = []
for g, m in groups.items():
    s_ = m | neg; r = {"子群": g, "n 陽性": int(m.sum()), "new": S.f3(S.boot_ci(m[s_].astype(int), p0[s_]))}
    for k, p in P.items():
        if k == new_name: continue
        r[f"Δ vs {k}"] = S.fd(boot_diff(m[s_].astype(int), p0[s_], p[s_], age[s_], 2.0, S.NBOOT)["auc"])
    sub.append(r)
pt = [{"模型": k, **{g: round(auroc((m[m | neg]).astype(float), p[m | neg]), 4) for g, m in groups.items()}} for k, p in P.items()]
md = [f"## {new_name} vs 參考（val 共同列 n={len(y)}，骨折 {int(y.sum())}；test 未讀取）\n", S.to_md(pd.DataFrame(rows)),
      "\n### 子群組（new [95% CI]；Δ 為配對 bootstrap）\n", S.to_md(pd.DataFrame(sub)), "\n### 子群組點估計\n", S.to_md(pd.DataFrame(pt))]
Path(out_md).write_text("\n".join(md) + "\n"); print("\n".join(md))
