"""bd15 report (Chinese markdown) from stack results of one or more runs. Usage: bd15_report.py ROOT TAG [TAG ...]"""
import sys, os, json, glob, numpy as np, pandas as pd
R = os.path.expanduser(sys.argv[1]); tags = sys.argv[2:]
L = ["# 膽管擴張：1.5 mm 肝門／肝十二指腸韌帶／胰頭 crop 單元（bd15，A6000）", "",
     "規則：只用 train/val（先依 split 過濾再讀標籤）；選模型只看 train 5-fold OOF（與 BD5/analysis_biliary 相同 folds）；val 只報告；test 未讀取。",
     "REF = 目前正式最佳（integrated_best：stack_bd5 cat4+E3_late+E0+porta_emb+HU+tube+porta｜GBM；train OOF .910／val .911）。",
     "單元：原始 CT → 1.5 mm 等向 128³（192 mm，與 BD5 同 FOV、同中心規則）→ 凍結 TS encoder（Dataset297，與 BD5 同 5 窗，stage 1–3，mean+max）。",
     "7 區域：r0 肝門球 30 mm、r1 HDL 走廊、r2 肝門肝實質、r3 胰頭走廊（= BD5 四區）＋ 1.5 mm 新增 r4 HDL 液體密度、r5 肝門低密度管、r6 胰頭液體密度。",
     "變體：U（7 區）、U_org（＋主 TS 胰／膽囊／肝單元）、U_bd5reg（只 r0–r3，看新區域貢獻）。Level-2：[logit(REF) | 單元頭] 以 LR／GBM，train OOF 選。", ""]
for t in tags:
    f = f"{R}/{t}/results.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    L += [f"## {t}（{'只用 tier0：全部 val＋train 陽性＋年齡×性別配對陰性 1:2' if 'tier0' in t else '全部 train/val'}）", "",
          f"train n={int(d.n_tr[0])}（陽性 {int(d.pos_tr[0])}），val n={int(d.n_val[0])}（陽性 {int(d.pos_val[0])}）。Δ = 對 REF 的 paired bootstrap（2,000 次）95% CI。", "",
          "| 模型 | 選用 | train OOF | Δ OOF [95% CI] | val [95% CI] | Δ val [95% CI] | 年齡配對 val | Δ 年齡配對 | 陰性 Spearman(分數, 年齡) | 90% 特異度敏感度 val |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        dd = lambda k: f"{r[k]:+.4f} [{r[k + '_lo']:+.4f}, {r[k + '_hi']:+.4f}]" if k in r and pd.notna(r.get(k)) else "—"
        L.append(f"| {r.model} | {r.cand} | {r.oof:.4f} | {dd('d_oof')} | {r.val:.4f} [{r.val_lo:.3f}, {r.val_hi:.3f}] | {dd('d_val')} | "
                 f"{r.val_age:.4f} | {dd('d_val_age')} | {r.spearman_neg_val:+.3f} | {100 * r.sens90_val:.1f}% |")
    s = d[d.selected].iloc[0]
    L += ["", f"**依 train OOF 選出：{s.model}（{s.cand}）** train OOF {s.oof:.4f}、val {s.val:.4f}；"
          f"Δ val {s.d_val:+.4f} [{s.d_val_lo:+.4f}, {s.d_val_hi:+.4f}]；年齡配對 {s.val_age:.4f}（Δ {s.d_val_age:+.4f} [{s.d_val_age_lo:+.4f}, {s.d_val_age_hi:+.4f}]）。", ""]
info = []
for f in glob.glob(f"{R}/logs/crop_info_w*.jsonl"):
    info += [json.loads(l) for l in open(f) if l.strip()]
if info:
    I = pd.DataFrame(info)
    L += ["## crop／embedding 統計", "", "crop 狀態：" + "、".join(f"{k} {v}" for k, v in I.status.value_counts().items()) + "。"]
    ok = I[I.status == "ok"]
    if len(ok):
        L.append(f"每例 crop 時間中位數 {ok.t.median():.1f} s（mask {ok.t_mask.median():.1f}、CT 傳輸 {ok.t_ct.median():.1f}、重取樣 {ok.t_resample.median():.1f}、區域 {ok.t_regions.median():.1f}）。")
        V = np.array(ok.vox.tolist())
        L.append("各區域 voxel 中位數（1.5 mm）：" + "、".join(f"r{i} {int(np.median(V[:, i]))}（空 {100 * (V[:, i] == 0).mean():.1f}%）" for i in range(V.shape[1])) + "。")
tf = f"{R}/logs/extract_timing.csv"
if os.path.exists(tf):
    T = pd.read_csv(tf, header=None, names=["id", "s", "ts"])
    L.append(f"GPU embedding 每例中位數 {T.s.median():.2f} s（n={len(T)}）。")
open(f"{R}/report.md", "w").write("\n".join(L) + "\n")
print("REPORT_WRITTEN", f"{R}/report.md")
