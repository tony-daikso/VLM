"""spine15_ft branch report (A6000 CPU): train OOF (selection metric) + val (reported once) of the true-1.5 mm vertebra head,
next to the 3 mm-upsampled spine_ft head (final3_tf2lse) on the same rows. Labels = spine_ft/in/labels.csv (train/val only).
Writes outputs_fracture/spine15_ft/report.md (+ report.json). usage: spine15_report.py"""
import json, os, sys, numpy as np, pandas as pd
from pathlib import Path
MW = Path.home() / "Desktop/Tony/VLM/OWL/mw_exp"; F = MW / "outputs_fracture"; D = F / "spine15_ft"
sys.path.insert(0, str(MW / "src"))
from probe import auroc
lab = pd.read_csv(F / "spine_ft/in/labels.csv"); assert set(lab.split) <= {"train", "val"}
lab = lab.set_index("study_id")
NEW = {"tr": pd.read_csv(D / "oof_train.csv").set_index("study_id")["ft"], "va": pd.read_csv(D / "val_scores.csv").set_index("study_id")["ft"]}
OLD = {"tr": pd.read_csv(F / "spine_ft/runs/final3_tf2lse/oof_train.csv").set_index("study_id")["ft"],
       "va": pd.read_csv(F / "spine_ft/runs/final3_tf2lse/val_scores.csv").set_index("study_id")["ft"]}
summ = json.load(open(D / "summary.json")); rng = np.random.default_rng(0); NB = 2000


def boot(y, a, b=None):
    y = np.asarray(y, int); n = len(y); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(NB):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        v.append(auroc(y[i], a[i]) - (auroc(y[i], b[i]) if b is not None else 0))
    return np.percentile(v, [2.5, 97.5])


rows = []
for split, key in (("train OOF", "tr"), ("val", "va")):
    ids = NEW[key].dropna().index.intersection(OLD[key].dropna().index)
    L = lab.loc[ids]
    for tgt, name in (("t_vert", "椎體骨折（site 標籤）"), ("y", "整體骨折")):
        m = L[tgt].notna().values
        if split == "train OOF" and tgt == "t_vert":
            for sub, mm in (("tier 列（選模指標）", m & L.tierA.values.astype(bool)), ("全部 train", m)):
                y = L[tgt].values[mm]; a = NEW[key].loc[ids].values[mm]; b = OLD[key].loc[ids].values[mm]
                rows.append(dict(資料=split, 目標=f"{name}·{sub}", n=int(mm.sum()), 陽性=int(y.sum()), spine15=auroc(y, a), spine_ft_3mm=auroc(y, b),
                                 Δ=auroc(y, a) - auroc(y, b), ΔCI=list(boot(y, a, b))))
            continue
        y = L[tgt].values[m]; a = NEW[key].loc[ids].values[m]; b = OLD[key].loc[ids].values[m]
        r = dict(資料=split, 目標=name, n=int(m.sum()), 陽性=int(y.sum()), spine15=auroc(y, a), spine_ft_3mm=auroc(y, b), Δ=auroc(y, a) - auroc(y, b), ΔCI=list(boot(y, a, b)))
        if split == "val": r["spine15 CI"] = list(boot(y, a))
        rows.append(r)
T = pd.DataFrame(rows)
nv = NEW["va"].notna().sum(); nt = NEW["tr"].notna().sum()
f = lambda x: f"{x:.3f}"
md = ["# spine15_ft：真 1.5 mm 逐節椎體 crop 微調 TS encoder（11-07）\n",
      "- 資料：原始 Merlin CT（CT-rate CTDATA＋Mac Passport 中繼）直接以三線性內插切 1.5 mm crop，T8–L5 每節 40×48×40（60×72×60 mm，與 spine_ft 同視野與方向），"
      "中心／遮罩用 Exp J seg15（Dataset292 1.5 mm 椎體標籤；缺者用 TS 3 mm 遮罩）。不再做 3 mm→×2 上採樣。腳本 `spine15_ft/scripts/spine15_crop.py`。",
      "- 模型：spine_ft 配方（TS Dataset297 encoder stage ≥2 微調，3 窗，stage3/4 mean+max+遮罩 mean → 每節 logit＋每節 bias → logsumexp MIL，8 epoch OneCycle，inner 12% holdout 選 epoch）。"
      "候選 tf2lse（原配方）與 tf2ctx（加相鄰椎體差值特徵；3080 bf16），各 1 seed，只看 train OOF（tier 列 t_vert）選：tf2lse .9382 > tf2ctx .9314 → tf2lse。"
      "依協調者 14:45 指示（使用者改專注肋骨／骨盆）不再加 seed，最終＝tf2lse 單 seed（A6000 fp32 訓練，5 個 fold 模型；val＝5 fold 平均）。",
      "- folds／labels 同 spine_ft（`spine_ft/in/labels.csv`，sitewise 外層 folds；tierA＝train 全部骨折＋4,000 陰性）。val 只在本報告評一次；test 未讀取。",
      f"- 有分數：train {nt}，val {nv}（另 3 例無任何椎體 unit）。spine_ft 對照＝3 mm×2 上採樣版 final3_tf2lse（3 seed 平均）。\n",
      "| 資料 | 目標 | n | 陽性 | spine15（1.5 mm） | spine_ft（3 mm×2，3 seed） | Δ [95% CI] |", "|---|---|---|---|---|---|---|"]
for _, r in T.iterrows():
    ci = r.get("spine15 CI"); c = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if isinstance(ci, list) else ""
    md.append(f"| {r['資料']} | {r['目標']} | {r['n']} | {r['陽性']} | {f(r['spine15'])}{c} | {f(r['spine_ft_3mm'])} | {r['Δ']:+.3f} [{r['ΔCI'][0]:+.3f}, {r['ΔCI'][1]:+.3f}] |")
md.append("\nΔ 為同列配對分層 bootstrap（2,000 次）。整合進骨折 stack 的結果見 `final_ft/<tag>/report.md` 與本目錄 `integration.md`。")
(D / "report.md").write_text("\n".join(md) + "\n"); T.to_json(D / "report.json", orient="records", force_ascii=False, indent=1)
print("\n".join(md))
