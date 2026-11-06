"""Exp O round-2 unit report.md = Chinese header + no_table.py output (results from UNIT/stack/results.json). usage: no_report.py UNIT"""
import sys, subprocess, time
U = sys.argv[1]
H = {
 "bile_duct_r2": ("膽管擴張", "遠端總膽管／胰頭單元：1 個 48³ crop（144 mm），中心＝距十二指腸 15 mm 內的胰臟（備援：胰臟最右 30%、十二指腸重心）；區域 r0 胰頭、r1 胰頭＋24 mm 內未標註軟組織（胰內／遠端總膽管走向）、r2 壺腹周圍十二指腸。",
                  "REF＝integrated_best 膽管擴張（stack_bd5 GBM，val .911）。結論：**沒有增益**（Δ val +.001 n.s.）；正式最佳維持 .911。"),
 "free_air_r2": ("腹腔游離氣體", "第二輪 3 個 64³ crop（192 mm）：c0 鐮狀韌帶前（左葉前緣）、c1 胃竇／十二指腸球（穿孔好發處）、c2 右結腸旁溝；區域只取非器官體素：r0 體內未標註、r1 游離氣體候選（HU < −400、離腸道 ≥ 6 mm、離肺 ≥ 3 mm）、r2 中心 45 mm 內未標註。",
                  "REF＝實驗 M 游離氣體最佳（REF＋全部單元頭，val .912）。結論：train OOF +.005 顯著，但 **val 無增益**（−.000 [−.005, +.005]）；正式最佳維持 .912。"),
 "ascites_r2": ("腹水", "第二輪 3 個 64³ crop：c0 右結腸旁溝、c1 左結腸旁溝、c2 小網膜囊；區域（同實驗 M）：r0 30 mm 球、r1 液體候選（−20～40 HU，未標註）、r2 器官殼層 12 mm、r3 50 mm 球。",
                  "REF＝實驗 M 腹水最佳（REF＋U_org，val .916）。結論：val .917，Δ +.001～+.002（CI 下界約 0），增益極小，**不列為新最佳**。"),
 "fracture_pelvis_r2": ("骨折（骨盆環）", "4 個 48³ crop：c0／c1 右／左髖臼、c2 恥骨枝／恥骨聯合、c3 薦椎（含薦翼／薦髂關節）；區域 r0 該部位骨骼（膨脹 1 vox）、r1 crop 內全部骨骼。out＝全骨折標籤訓練的頭（U、U_org）；out_pel＝只用骨盆／薦椎／髖部位標籤訓練、對所有病例打分的頭（U_pel、U_org_pel）。",
                  "REF＝integrated_best 骨折（≈ J add 10 seed，val .847）。依 train OOF 選出 REF＋U_org_pel：val .853，Δ +.006 [+.002, +.010]，年齡配對 .837（Δ +.008）；骨盆／薦椎／髖子群 .848 → .874（Δ +.027 [+.017, +.037]）。與 I＋J＋L 的合併見 `outputs_fracture/final_combo/`。"),
 "bowel_obstruction_r2": ("腸阻塞", "胃出口／十二指腸單元：2 個 64³ crop：c0 胃竇、c1 十二指腸；區域 r0 管腔（胃｜十二指腸）、r1 該部位器官（膨脹 1 vox）、r2 中心 45 mm 內非骨骼體組織。",
                  "REF＝integrated_best 腸阻塞（val .956）。依 train OOF 選出 REF＋U_org：val .958，Δ +.003 [−.000, +.006] **未顯著**；胃出口／十二指腸／輸入環子群 23 例陽性 .954 → .957。正式最佳維持 .956（n.s. 不更新）。"),
}
name, unit, concl = H[U]
tab = subprocess.run([sys.executable, "scripts/no_table.py", U], capture_output=True, text=True, check=True).stdout
md = [f"# 實驗 O（區域 crop 單元第二輪）：{name}（{U}）", "",
      f"產生時間 {time.strftime("%Y-%m-%d %H:%M")}；結果來源 `{U}/stack/results.json`（no_stack.py）與 probe log。",
      "規則：只用 train/val（標籤 csv 只含 train/val，先依 split 過濾）；模型選擇只看 train 5-fold OOF；val 只報告；test 未讀取。年齡／性別不作為輸入。",
      "管線：3 mm prep＋TS total 遮罩 → crop（A6000＋CT-rate CPU）→ 凍結 TS --fast encoder（Dataset297，5 窗，stage 1–3，mean＋max）→ E3_late 式 late head（U＝只有 crop 區域、U_org＝再加主 TS 器官單元）→ 分數層堆疊 [logit(REF)｜單元頭]，LR／GBM 依 train OOF 選。Δ＝對 REF 的配對 bootstrap 95% CI。", "",
      f"**單元**：{unit}", "", f"**結論**：{concl}", "", "## 結果（粗體＝依 train OOF 選出）", "", tab]
open(f"{U}/report.md", "w").write("\n".join(md)); print("wrote", U + "/report.md")
