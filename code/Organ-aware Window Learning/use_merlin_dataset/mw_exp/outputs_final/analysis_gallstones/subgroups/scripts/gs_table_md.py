"""Render subgroups_train_oof.csv + subgroups_val.csv into a markdown table (table_md.txt)."""
import pandas as pd, numpy as np
t = pd.read_csv("subgroups_train_oof.csv"); v = pd.read_csv("subgroups_val.csv")
m = t.merge(v, on=["axis", "subgroup"], suffixes=("", "_v"))
m["ord"] = m.axis.str.extract(r"^(\d+[a-c]?)")[0].fillna("0"); m = m.sort_values(["ord"], kind="stable")
f = lambda x: "–" if pd.isna(x) else f"{x:.3f}"
L = ["| 軸 | 子群 | n陽性 訓練(%) | E3_late | quickwins | V4 [95% CI] | V4 漏診率@90%特異度 | 佔V4全部FN | n陽性 驗證 | V4 驗證 [95% CI] | 驗證漏診率 |", "|" + "---|" * 11]
for _, r in m.iterrows():
    ci = f" [{r.auc_V4_lo:.3f}–{r.auc_V4_hi:.3f}]" if not pd.isna(r.auc_V4_lo) else ""
    civ = f" [{r.auc_V4_lo_v:.3f}–{r.auc_V4_hi_v:.3f}]" if not pd.isna(r.auc_V4_lo_v) else ""
    L.append(f"| {r.axis} | {r.subgroup} | {r.n_pos} ({r.pct_pos}%) | {f(r.auc_E3_late)} | {f(r.auc_quickwins)} | {f(r.auc_V4)}{ci} | {r.V4_miss_rate_spec90}% | {r.share_of_V4_FN}% | {r.n_pos_v} | {f(r.auc_V4_v)}{civ} | {r.V4_miss_rate_spec90_v}% |")
open("table_md.txt", "w").write("\n".join(L)); print("\n".join(L))
