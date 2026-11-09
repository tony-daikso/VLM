"""report_v2.md from main_all_v2.json. usage: ib_report_v2.py <integrated_best root>"""
import sys, json
R = sys.argv[1].rstrip("/") + "/"; J = json.load(open(R + "main_all_v2.json"))
f3 = lambda x: f"{x:.3f}"; pm = lambda x: f"{x:+.3f}"
cis = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"; dcs = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
L = ["# main_all v2：各疾病最新正式最佳（val）", "",
     "日期：2026-11-06｜A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/integrated_best/`｜**只用 val；test 未讀取、未評分**｜CPU（nice 19）", "",
     "## 0. 方法（與 `ib_summary.py` 相同）",
     "- main_all = 14 項原始 finding 的 val AUROC 平均。",
     "- 基準：expB E3_late（3 seed 平均），用全部有標籤的 val study。",
     "- 模型分數依 study_id 併入；缺分數的 study 只在該 finding 剔除（骨折 4,946／4,961）。",
     "- CI：study 層級 joint paired bootstrap，1,000 次，seed 0。三個版本與所有 Δ 都用同一批重抽樣。",
     "- 年齡配對 main_all：各 finding 的 `ib_core.age_matched_auc` 取平均（10 歲年齡層內配對，依配對數加權），用同一批 bootstrap 重抽樣。",
     "- 前版 integrated_best 重算得 0.9138，與原 `main_all.json` 完全一致。", "",
     "## 1. main_all（14 項原始 finding）", "",
     "| 版本 | main_all [95% CI] | 年齡配對 main_all [95% CI] |", "|---|---|---|"]
for k, name in (("b", "expB E3_late"), ("prev", "前版整合最佳（integrated_best v1）"), ("new", "**最新正式最佳（v2）**")):
    a, g = J["main_all"]["auroc"][k], J["main_all"]["age_matched"][k]
    L.append(f"| {name} | {f3(a['value'])} {cis(a['ci'])} | {f3(g['value'])} {cis(g['ci'])} |")
for k, name in (("delta_new_vs_b", "Δ 最新 − E3_late"), ("delta_prev_vs_b", "Δ 前版 − E3_late"), ("delta_new_vs_prev", "Δ 最新 − 前版")):
    a, g = J["main_all"]["auroc"][k], J["main_all"]["age_matched"][k]
    L.append(f"| {name} | {pm(a['value'])} {dcs(a['ci'])} | {pm(g['value'])} {dcs(g['ci'])} |")
L += ["", "## 2. 拆分標籤版（脂肪肝 → 瀰漫性脂肪肝；闌尾炎 → 確定的急性闌尾炎）", "",
      "E3_late 端用原 expB E3_late 對新標籤評分（同 ib_summary）。", "",
      "| 版本 | main_all [95% CI] | 年齡配對 main_all [95% CI] |", "|---|---|---|"]
for k, name in (("b", "expB E3_late"), ("prev", "前版整合最佳"), ("new", "**最新正式最佳（v2）**"), ("delta_new_vs_b", "Δ 最新 − E3_late"), ("delta_new_vs_prev", "Δ 最新 − 前版")):
    a, g = J["split_variant"]["auroc"][k], J["split_variant"]["age_matched"][k]
    fmt = (lambda d: f"{pm(d['value'])} {dcs(d['ci'])}") if k.startswith("delta") else (lambda d: f"{f3(d['value'])} {cis(d['ci'])}")
    L.append(f"| {name} | {fmt(a)} | {fmt(g)} |")
L += ["", "## 3. 各疾病（val）", "",
      "| 疾病 | val 陽性 | 最新正式最佳模型 | E3_late | 前版整合 | **最新** | Δ 最新 vs E3_late [95% CI] | Δ 最新 vs 前版 | 年齡配對 E3_late → 最新 |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in J["per_finding"]:
    if r.get("split_row"): continue
    dpv = "—" if abs(r["d_new_vs_prev"]) < 1e-9 else f"{pm(r['d_new_vs_prev'])} {dcs(r['d_prev_ci'])}"
    L.append(f"| {r['zh']} | {r['pos_val']} | {r['model_new']} | {f3(r['E3'])} | {f3(r['prev'])} | **{f3(r['new'])}** | {pm(r['d_new_vs_E3'])} {dcs(r['d_ci'])} | {dpv} | {f3(r['age_E3'])} → {f3(r['age_new'])} |")
L += ["", "拆分標籤列（取代用）：", "", "| 疾病 | val 陽性 | E3_late | 前版整合 | **最新** | Δ 最新 vs E3_late | 年齡配對 E3_late → 最新 |", "|---|---|---|---|---|---|---|"]
for r in J["per_finding"]:
    if not r.get("split_row"): continue
    L.append(f"| {r['zh']} | {r['pos_val']} | {f3(r['E3'])} | {f3(r['prev'])} | **{f3(r['new'])}** | {pm(r['d_new_vs_E3'])} | {f3(r['age_E3'])} → {f3(r['age_new'])} |")
L += ["", "## 4. 分數來源（最新版）",
      "- 腎水腫：`region_units/hydronephrosis/stack/scores_val.csv`「REF＋全部單元頭」（train OOF 選定）。",
      "- 腹腔游離氣體：`region_units/free_air/stack/scores_val.csv`「REF＋全部單元頭」。",
      "- 腹水：`region_units/ascites/stack/scores_val.csv`「REF＋U_org」。",
      "- 闌尾炎（原合併）／確定的急性闌尾炎：`appendicitis_rlq_unit` 的「REF＋U_rlq_cs」堆疊。原輸出沒有存每例堆疊分數，因此用 `scripts/ib_v2_export_ap.py` 依 `rlq_eval.py` 同一設定（LR C=0.1、全部 train OOF refit）重建；val AUROC 與原 results.csv 完全相同（.920365／.936213）。輸出在 `scores_v2/appendicitis_val.csv`、`split_v2/ap_def_val.csv`。",
      "- 骨質減少：`osteopenia_v15/stack_val.csv`「5win+v15」（5 個 window logit＋1.5 mm 逐節 head）。",
      "- 骨折：Exp J add 10 seed（`integrated_best/scores/fracture_val.csv`，同前版）。",
      "- 其餘（膽結石、膽管擴張、脂肪肝、動脈粥狀硬化、腸阻塞、膽囊已切除、腹主動脈瘤、脾腫大）、瀰漫性脂肪肝：同前版 `integrated_best/scores`、`split/`。",
      "- 各分數檔的標籤與 labels14 逐例比對，不一致 0 例。", "",
      "## 5. 注意",
      "- 這次更新的 5 項（腎水腫、腹腔游離氣體、腹水、闌尾炎、骨質減少）都是看過 val 之後才做的後續實驗；模型雖然只依 train OOF 選定，val 仍不是完全未見的評估。最終數字以鎖定設定後只跑一次的 test 為準。",
      "- 年齡配對 main_all 比原始 main_all 低約 .02，主要是動脈粥狀硬化（.779）與骨質減少（.798）拉低。",
      "- 檔案：`main_all_v2.json`；程式 `scripts/ib_summary_v2.py`、`scripts/ib_report_v2.py`、`scripts/ib_v2_export_ap.py`；log `logs/ib_summary_v2.log`。"]
open(R + "report_v2.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
