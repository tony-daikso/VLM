"""rib_ft -> sitewise stack (A6000 CPU). Train/val only (sitewise.build_frame filters split first; test never read).
Adds R_ft (rib_ft score: train = 5-fold OOF, val = fold-model mean) to the S2 rib site head:
  S2      : official (.862) recomputed
  S2+R    : rib head = S2 rib inputs + R_ft
  S2R     : rib head = R_ft + L_r (L rib15 site)
  S2Ronly : rib head = R_ft only
combiners as sitewise (lse+any, noisyor+any, lse, noisyor, max+any). Selection = train 5-fold CV (sitewise outer folds,
nested level-1 refits). Val reported once; paired bootstrap vs official S2 lse+any.
usage: rib_stack.py RIBFT_DIR  (reads RIBFT_DIR/oof_train.csv, val_scores.csv; writes RIBFT_DIR/stack/)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
sys.path.insert(0, str(MW / "outputs_fracture/sitewise/scripts"))
import sitewise as S
from sitewise import auroc, boot_ci, boot_diff, matched_auc, sens_at_spec, f3, fd, to_md
D = Path(sys.argv[1]); OUT = D / "stack"; OUT.mkdir(parents=True, exist_ok=True)
df, sel = S.build_frame()
o = pd.read_csv(D / "oof_train.csv").set_index("study_id")["ft"]; v = pd.read_csv(D / "val_scores.csv").set_index("study_id")["ft"]
assert not set(o.index) & set(v.index)
df["R_ft"] = pd.concat([o, v]).reindex(df.index)
base = S.FEATS["S2"]
S.FEATS = {"S2": base, "S2+R": {**base, "rib": base["rib"] + ["R_ft"]}, "S2R": {**base, "rib": ["R_ft", "L_r"]},
           "S2Ronly": {**base, "rib": ["R_ft"]}}
need = sorted({c for fs in S.FEATS.values() for vv in fs.values() for c in vv} | set(S.C3M))
tr = df[df.split == "train"].dropna(subset=need).copy(); va = df[df.split == "val"].dropna(subset=need + ["age"]).copy()
nmiss = {"train": int(((df.split == "train") & df.R_ft.isna()).sum()), "val": int(((df.split == "val") & df.R_ft.isna()).sum())}
print("train", len(tr), "val", len(va), "R_ft missing", nmiss, flush=True)
kinds = ["lse+any", "noisyor+any", "max+any", "lse", "noisyor"]
res, store = [], {}
for fs in S.FEATS:
    oof, L1oof = S.run_cv(tr, fs, kinds); L1va = S.level1(fs, tr, va)
    for k in kinds:
        pv = S.combine(k, L1va); store[(fs, k)] = (pv, oof[k])
        res.append({"設定": f"{fs} · {k}", "fs": fs, "k": k, "train CV": auroc(tr["y"].values, oof[k]),
                    "rib site CV": auroc(tr["t_rib"].values, L1oof["rib"].values)})
    store[(fs, "L1va")] = L1va; store[(fs, "L1oof")] = L1oof
res = pd.DataFrame(res); print(res.to_string(), flush=True)
ref = store[("S2", "lse+any")][0]
cand = res[res.fs != "S2"]; best = cand.loc[cand["train CV"].idxmax()]
ref_cv = float(res[(res.fs == "S2") & (res.k == "lse+any")]["train CV"].iloc[0])
print("SELECTED", best["設定"], best["train CV"], "ref CV", ref_cv, flush=True)
y = va["y"].values.astype(int); agev = va["age"].values.astype(float); neg = y == 0
groups = {g: (y == 1) & np.asarray(f(va), bool) for g, f in S.SUB.items()}
rows = []
for _, r in res.iterrows():
    p = store[(r.fs, r.k)][0]
    row = {"設定": r["設定"], "train CV": r["train CV"], "rib site CV": r["rib site CV"], "val [95% CI]": f3(boot_ci(y, p)),
           "年齡配對(±2y)": matched_auc(y, p, agev, 2.0), "陰性 Spearman(分數,年齡)": spearmanr(p[neg], agev[neg])[0],
           "90%特異度敏感度": sens_at_spec(y, p)}
    if np.array_equal(p, ref): row["Δ vs S2·lse+any"] = row["Δ年齡配對"] = "—"
    else:
        b = boot_diff(y, p, ref, agev, 2.0, S.NBOOT); row["Δ vs S2·lse+any"], row["Δ年齡配對"] = fd(b["auc"]), fd(b["matched"])
    for g, m in groups.items():
        s_ = m | neg; row[g] = auroc(m[s_].astype(float), p[s_])
    rows.append(row)
T = pd.DataFrame(rows); T.to_csv(OUT / "report_all.csv", index=False)
selp = store[(best.fs, best.k)][0]; sub = []
for name, p in [(f"選出：{best['設定']}", selp), ("S2 · lse+any（官方 .862）", ref)]:
    sub.append({"模型": name, **{g: f3(boot_ci(m[m | neg].astype(int), p[m | neg])) for g, m in groups.items()}})
sub.append({"模型": "Δ 選出 − 官方（配對）", **{g: fd(boot_diff(m[m | neg].astype(int), selp[m | neg], ref[m | neg], agev[m | neg], 2.0, S.NBOOT)["auc"]) for g, m in groups.items()}})
# R_ft alone diagnostics
rd = []
for nm, p_tr, p_va in [("R_ft（rib_ft 單獨）", tr["R_ft"].values, va["R_ft"].values), ("L_r（Exp L rib site）", tr["L_r"].values, va["L_r"].values),
                       ("S2 rib head", store[("S2", "L1oof")]["rib"].values, store[("S2", "L1va")]["rib"].values),
                       (f"{best.fs} rib head", store[(best.fs, "L1oof")]["rib"].values, store[(best.fs, "L1va")]["rib"].values)]:
    rd.append({"分數": nm, "train OOF AUROC(肋骨標籤)": auroc(tr["t_rib"].values, p_tr), "val AUROC(肋骨標籤)": f3(boot_ci(*(lambda m: (va["t_rib"].values[m].astype(int), p_va[m]))(~va["t_rib"].isna().values))),
               "val 肋骨 vs 全陰性": f3(boot_ci(groups["肋骨"][groups["肋骨"] | neg].astype(int), p_va[groups["肋骨"] | neg])),
               "val 只有肋骨 vs 全陰性": f3(boot_ci(groups["只有肋骨"][groups["只有肋骨"] | neg].astype(int), p_va[groups["只有肋骨"] | neg]))})
RD = pd.DataFrame(rd)
json.dump({"selected": best["設定"], "selected_train_cv": float(best["train CV"]), "ref_train_cv": ref_cv,
           "selected_val": float(auroc(y.astype(float), selp)), "ref_val": float(auroc(y.astype(float), ref)),
           "n_train": len(tr), "n_val": len(va), "R_ft_missing": nmiss, "feats": S.FEATS}, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
np.savez(OUT / "scores.npz", ids=np.r_[tr.index.values, va.index.values].astype(str), split=np.r_[["train"] * len(tr), ["val"] * len(va)],
         selected=np.r_[store[(best.fs, best.k)][1], selp], ref=np.r_[store[("S2", "lse+any")][1], ref])
md = [f"# rib_ft → sitewise 整合（自動產生）\n", f"train N={len(tr)}，val N={len(va)}（骨折 {int(y.sum())}）；R_ft 缺值 {nmiss}。依 train CV 選（候選不含官方 S2）：**{best['設定']}**（train CV {best['train CV']:.4f} vs 官方 {ref_cv:.4f}）。\n",
      to_md(T.drop(columns=list(S.SUB))), "\n## 子群組（點估計）\n", to_md(T[["設定"] + list(S.SUB)]), "\n## 子群組 95% CI\n", to_md(pd.DataFrame(sub)),
      "\n## 肋骨分數診斷\n", to_md(RD)]
(OUT / "report_auto.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
