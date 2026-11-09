"""final_ft: rebuild the site-wise fracture stack (outputs_fracture/sitewise, S2 . lse+any, val .862) with the fine-tuned site
heads (spine_ft, pelvis_ft, rib_ft if present) replacing / augmenting the old site inputs. Train/val only; test never read.
Every candidate uses the sitewise nested CV (outer StratifiedKFold(5, rs=0) on the common train frame; site heads + C3 'any'
stack refit inside each outer-train). Selection = max train CV (overall fracture label); val reported once.
FT inputs: <branch>_ft/runs/final_*/oof_train.csv (column ft; train OOF from fold models) + val_scores.csv (fold-model mean).
usage: final_ft.py [--rib PATH_DIR] [--tag NAME]
pelvis15_ft copy (agent pelvis15_ft 11-08): + --drop NAME[,NAME] removes old site inputs (e.g. U_org_pel,U_pel) from every site head and from the any-stack."""
import sys, os, glob, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"
sys.path.insert(0, str(MW / "outputs_fracture/sitewise/scripts")); sys.path.insert(0, str(MW / "src"))
import sitewise as S
sys.path.insert(0, str(MW / "outputs_fracture/final_ft/scripts"))
from scipy.stats import spearmanr
from age_shortcut import matched_auc
from fracture_eval_oof import boot_diff
from probe import auroc
FR = MW / "outputs_fracture"
ap = argparse.ArgumentParser(); ap.add_argument("--rib", default=""); ap.add_argument("--tag", default="")
ap.add_argument("--spine", default=""); ap.add_argument("--pelvis", default="")
ap.add_argument("--drop", default=""); ap.add_argument("--add", action="append", default=[], help="NAME=DIR@site (site in vert|rib|pel); repeatable")
a = ap.parse_args()
OUT = FR / "final_ft" / (a.tag or "main"); OUT.mkdir(parents=True, exist_ok=True)


def pick(d):
    """explicit dir, else the branch's selected final run."""
    d = Path(d)
    if (d / "oof_train.csv").exists():
        return d
    b = (d / "runs/best_cfg.txt")
    if b.exists():
        f = d / f"runs/final_{b.read_text().strip()}"
        if (f / "val_scores.csv").exists():
            return f
    return None


def load_ft(d, name):
    tr = pd.read_csv(d / "oof_train.csv"); va = pd.read_csv(d / "val_scores.csv")
    col = "ft" if "ft" in tr.columns else [c for c in tr.columns if c not in ("study_id", "fold", "in_tier", "split")][-1]
    colv = "ft" if "ft" in va.columns else [c for c in va.columns if c not in ("study_id", "fold", "split")][-1]
    s = pd.concat([tr.set_index("study_id")[col], va.set_index("study_id")[colv]])
    print(name, d, "train", tr[col].notna().sum(), "val", va[colv].notna().sum(), flush=True)
    return s.rename(name)


df, sel = S.build_frame()
ft = {}
for name, d in (("FT_sp", a.spine or FR / "spine_ft"), ("FT_pv", a.pelvis or FR / "pelvis_ft"), ("FT_rib", a.rib or FR / "rib_ft")):
    p = pick(d)
    if p is not None:
        ft[name] = str(p); df = df.join(load_ft(p, name))
ADD = {"vert": [], "rib": [], "pel": []}
for spec in a.add:  # NAME=DIR@site : extra FT head (oof_train.csv + val_scores.csv) added to that site head (and to 'any' in *a)
    nm, rest = spec.split("=", 1); dd, site = rest.rsplit("@", 1); p = pick(dd) or Path(dd)
    ft[nm] = str(p); df = df.join(load_ft(p, nm)); ADD[site].append(nm)
print("FT inputs", ft, flush=True)
# rows lacking an FT score (no crop) get the train median so the common set equals sitewise's
for n in ft:
    miss = df[n].isna(); df.loc[miss, n] = df.loc[df.split == "train", n].median(); print(n, "imputed", int(miss.sum()))

DROP = [x for x in a.drop.split(",") if x]
OLD = {k: [c for c in v if c not in DROP] for k, v in S.FEATS["S2"].items()}
C3 = [c for c in S.C3M if c not in DROP]
FTS = {"vert": [n for n in ["FT_sp"] if n in ft], "rib": [n for n in ["FT_rib"] if n in ft], "pel": [n for n in ["FT_pv"] if n in ft]}
for s_ in FTS: FTS[s_] += ADD[s_]
FTN = {s_: [n for n in v if n != "FT_pv"] for s_, v in FTS.items()}  # without the weak pelvis FT head
CANDS = {  # name: (site feats, any-members)
    "S2 (ref)": (OLD, C3),
    "S3 = S2 + FT": ({s: OLD[s] + FTS[s] for s in S.SITES}, C3),
    "S3a = S2 + FT (+FT in any)": ({s: OLD[s] + FTS[s] for s in S.SITES}, C3 + sum(FTS.values(), [])),
    "S4 = main + FT": ({"vert": ["J_add"] + FTS["vert"], "rib": ["L_r"] + FTS["rib"], "pel": [c for c in ["U_pel", "U_org_pel"] if c not in DROP] + FTS["pel"]}, C3),
    "S4a = main + FT (+FT in any)": ({"vert": ["J_add"] + FTS["vert"], "rib": ["L_r"] + FTS["rib"], "pel": [c for c in ["U_pel", "U_org_pel"] if c not in DROP] + FTS["pel"]},
                                      C3 + sum(FTS.values(), [])),
}
if "FT_pv" in ft:
    CANDS["S3n = S2 + FT (no FT_pv)"] = ({s: OLD[s] + FTN[s] for s in S.SITES}, C3)
    CANDS["S3na = S2 + FT (no FT_pv, +FT in any)"] = ({s: OLD[s] + FTN[s] for s in S.SITES}, C3 + sum(FTN.values(), []))
KINDS = ["max", "noisyor", "lse", "max+any", "noisyor+any", "lse+any", "lr", "lrc3"]
need = sorted({c for f, m in CANDS.values() for v in f.values() for c in v} | set(C3) | set(ft))
tr = df[df.split == "train"].dropna(subset=need).copy()
va = df[df.split == "val"].dropna(subset=need + ["age"]).copy()
print("train", len(tr), "val", len(va), flush=True)
diag = [{"輸入": c, **{f"train AUROC({s})": auroc(tr["t_" + s].values, tr[c].values) for s in S.SITES},
         "train AUROC(整體)": auroc(tr["y"].values, tr[c].values),
         **{f"val AUROC({s})": auroc(va["t_" + s].values, va[c].values) for s in S.SITES}} for c in list(ft) + ["J_add", "L_r", "U_pel", "U_org_pel"] if c in tr]
res, store = [], {}
for cn, (feats, any_m) in CANDS.items():
    if cn != "S2 (ref)" and not ft:
        continue
    S.FEATS["X"] = feats; S.C3M[:] = any_m
    oof, L1oof = S.run_cv(tr, "X", KINDS); L1va = S.level1("X", tr, va)
    for k in KINDS:
        f2 = S.l2_fit(k, L1oof, tr["y"].values); pv = S.combine(k, L1va, f2)
        store[(cn, k)] = (pv, oof[k])
        res.append({"候選": cn, "合併": k, "train CV": auroc(tr["y"].values, oof[k]), "val": auroc(va["y"].values, pv),
                    **{f"site CV {s}": auroc(tr["t_" + s].values, L1oof[s].values) for s in S.SITES}})
        print(cn, k, round(res[-1]["train CV"], 4), flush=True)
    store[(cn, "L1va")] = L1va
S.C3M[:] = C3
res = pd.DataFrame(res)
ref_p = store[("S2 (ref)", "lse+any")][0]
best = res.loc[res["train CV"].idxmax()]; bn, bk = best["候選"], best["合併"]
sel_p = store[(bn, bk)][0]
y = va["y"].values.astype(int); agev = va["age"].values.astype(float); neg = y == 0
groups = {g: (y == 1) & np.asarray(f(va), bool) for g, f in S.SUB.items()}
rows = []
show = [(f"{r['候選']} · {r['合併']}", store[(r["候選"], r["合併"])][0], r["train CV"]) for _, r in res.iterrows()
        if r["合併"] in ("lse+any", "lse", "noisyor+any", "lrc3") or (r["候選"], r["合併"]) == (bn, bk)]
for name, p, cv in show:
    r = {"設定": name + (" ★" if name == f"{bn} · {bk}" else ""), "train CV": cv, "val [95% CI]": S.f3(S.boot_ci(y, p)),
         "年齡配對(±2y)": matched_auc(y, p, agev, 2.0), "陰性 Spearman(分數,年齡)": spearmanr(p[neg], agev[neg])[0],
         "90%特異度敏感度": S.sens_at_spec(y, p)}
    if np.array_equal(p, ref_p):
        r["Δ vs .862（S2·lse+any）"] = r["Δ年齡配對"] = "—"
    else:
        b = boot_diff(y, p, ref_p, agev, 2.0, S.NBOOT); r["Δ vs .862（S2·lse+any）"], r["Δ年齡配對"] = S.fd(b["auc"]), S.fd(b["matched"])
    for g, m in groups.items():
        s_ = m | neg; r[g] = auroc(m[s_].astype(float), p[s_])
    rows.append(r)
T = pd.DataFrame(rows)
sub = []
for name, p in [(f"選出：{bn} · {bk}", sel_p), ("S2·lse+any（.862）", ref_p)]:
    sub.append({"模型": name, **{g: S.f3(S.boot_ci(m[m | neg].astype(int), p[m | neg])) for g, m in groups.items()}})
sub.append({"模型": "Δ 選出 − .862（配對）", **{g: S.fd(boot_diff(m[m | neg].astype(int), sel_p[m | neg], ref_p[m | neg], agev[m | neg], 2.0, S.NBOOT)["auc"])
                                          for g, m in groups.items()}})
res.to_csv(OUT / "all_candidates.csv", index=False); T.to_csv(OUT / "report_table.csv", index=False)
np.savez(OUT / "scores.npz", ids=np.r_[tr.index.values, va.index.values].astype(str), split=np.r_[["train"] * len(tr), ["val"] * len(va)],
         y=np.r_[tr["y"].values, va["y"].values], selected=np.r_[store[(bn, bk)][1], sel_p],
         ref=np.r_[store[("S2 (ref)", "lse+any")][1], ref_p], selected_name=f"{bn}·{bk}")
summ = {"selected": [bn, bk], "selected_train_cv": float(best["train CV"]), "ref_train_cv": float(res[(res["候選"] == "S2 (ref)") & (res["合併"] == "lse+any")]["train CV"].iloc[0]),
        "selected_val": float(auroc(y.astype(float), sel_p)), "ref_val": float(auroc(y.astype(float), ref_p)), "ft_inputs": ft,
        "n_train": len(tr), "n_val": len(va), "val_ci": S.boot_ci(y, sel_p), "delta_vs_ref": boot_diff(y, sel_p, ref_p, agev, 2.0, S.NBOOT)["auc"],
        "age_matched": matched_auc(y, sel_p, agev, 2.0), "neg_spearman_age": float(spearmanr(sel_p[neg], agev[neg])[0]),
        "sens_at_90spec": S.sens_at_spec(y, sel_p)}
json.dump(summ, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False, default=float)
md = [f"# 骨折 final_ft：微調部位頭整合進 site-wise 堆疊（{a.tag or 'main'}）\n",
      "規則：只用 train/val（split.csv 先過濾再解析標籤）；所有選擇只依 train 5-fold 巢狀 CV（同 sitewise 外層 folds）；val 只報告一次；test 未讀取。"
      f"Bootstrap {S.NBOOT} 次，Δ 為對正式 .862（S2·lse+any，同共同集重算）的配對 bootstrap。",
      f"FT 輸入：{ft}（缺 crop 者以 train 中位數補）。train N={len(tr)}，val N={len(va)}（骨折 {int(y.sum())}）。\n",
      f"## 依 train CV 選出：**{bn} · {bk}**（train CV {best['train CV']:.4f}；參考 S2·lse+any {summ['ref_train_cv']:.4f}）\n",
      f"- val **{summ['val_ci'][0]:.3f} [{summ['val_ci'][1]:.3f}, {summ['val_ci'][2]:.3f}]**，Δ vs .862 {S.fd(summ['delta_vs_ref'])}；"
      f"年齡配對 {summ['age_matched']:.3f}；陰性 Spearman(分數,年齡) {summ['neg_spearman_age']:.3f}；90% 特異度敏感度 {summ['sens_at_90spec']:.1%}\n",
      "## 主要設定\n", S.to_md(T.drop(columns=list(S.SUB))), "\n### 子群組（點估計）\n", S.to_md(T[["設定"] + list(S.SUB)]),
      "\n## 子群組 95% CI\n", S.to_md(pd.DataFrame(sub)), "\n## 輸入診斷（原始分數；train=OOF，val 僅描述）\n", S.to_md(pd.DataFrame(diag)),
      "\n## 全部候選（train CV／val）\n", S.to_md(res)]
(OUT / "report.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:6]), flush=True)
