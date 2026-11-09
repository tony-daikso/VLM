import pandas as pd, numpy as np, sys
O = sys.argv[1] + "/out/"
CS = pd.read_csv(O + "category_shares.csv")
def f(x, d=3): return "–" if pd.isna(x) else f"{x:.{d}f}"
out = []
for e in ["FN", "FP"]:
    v = CS[(CS.split == "val") & (CS.err == e)].set_index("cat"); t = CS[(CS.split == "train") & (CS.err == e)].set_index("cat")
    out.append(f"\n**{e}（互斥主類別，依優先序指派）**\n\n| 類別 | val n | val 佔比 | train OOF n | train 佔比 |\n|---|---|---|---|---|")
    for c in sorted(set(v.index) | set(t.index)):
        out.append(f"| {c} | {int(v.n.get(c, 0))} | {v.share.get(c, 0)*100:.1f}% | {int(t.n.get(c, 0))} | {t.share.get(c, 0)*100:.1f}% |")
SG = pd.read_csv(O + "subgroups.csv")
for side, lab in [("陽性", "陽性子群（該子群陽性 vs 全部陰性）；漏診率 = 該子群陽性中 FN 比例"), ("陰性", "陰性子群（全部陽性 vs 該子群陰性）；誤報率 = 該子群陰性中 FP 比例")]:
    v = SG[(SG.split == "val") & (SG.side == side)].set_index("name"); t = SG[(SG.split == "train") & (SG.side == side)].set_index("name")
    out.append(f"\n**{lab}**\n\n| 子群 | val n | Exp N val [95% CI] | 舊 V4 val | {'漏診率' if side=='陽性' else '誤報率'} N／V4 | 佔 val 全部 {'FN' if side=='陽性' else 'FP'} | train n | Exp N OOF | 舊 V4 OOF |\n|---|---|---|---|---|---|---|---|---|")
    for nm in v.index:
        r = v.loc[nm]; ci = f" [{f(r.ci_lo)}, {f(r.ci_hi)}]" if not pd.isna(r.ci_lo) else ("（n<30）" if side == "陽性" else "")
        tt = t.loc[nm] if nm in t.index else None
        out.append(f"| {nm} | {int(r.n)} | {f(r.auc_N)}{ci} | {f(r.auc_V4)} | {r.miss_rate*100:.0f}%／{r.miss_rate_V4*100:.0f}% | {r.share_err*100:.1f}% | {int(tt.n) if tt is not None else '–'} | {f(tt.auc_N) if tt is not None else '–'} | {f(tt.auc_V4) if tt is not None else '–'} |")
RC = pd.read_csv(O + "recoverable.csv")
v = RC[RC.split == "val"].set_index(["cause", "side"]); t = RC[RC.split == "train"].set_index(["cause", "side"])
v = v.join(t[["n", "d_remove"]].add_suffix("_tr")).reset_index()
v = v[~v.cause.isin(["膽囊切除陰性", "遮罩缺失（陰性）"])].sort_values("d_remove", ascending=False)
out.append("\n| 排名 | 原因 | 性質 | val n | ΔAUROC 移除 [95% CI] | ΔAUROC 同分布替換 | train n | train ΔAUROC |\n|---|---|---|---|---|---|---|---|")
for i, r in enumerate(v.itertuples(), 1):
    out.append(f"| {i} | {r.cause} | {r.kind} | {int(r.n)} | {r.d_remove:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] | {r.d_parity:+.4f} | {int(r.n_tr)} | {r.d_remove_tr:+.4f} |")
open(O + "tables.md", "w").write("\n".join(out)); print("\n".join(out))
