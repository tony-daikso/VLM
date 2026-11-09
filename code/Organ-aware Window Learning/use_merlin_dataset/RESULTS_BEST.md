# OWL（Organ-aware Window Learning）× Merlin：各疾病目前最佳結果（val）

更新：2026-11-09｜資料來源：A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/`（下表「報告」欄都是相對這個目錄的路徑）

## 評估規則（所有疾病一致）
- **模型選擇只用 train**：train 5-fold OOF，或 stacker 的 train 巢狀 CV。沒有任何模型是依 val 挑出來的。
- **val 只報告**：每個選定模型只評一次 val。AUROC 的 95% CI 用分層 bootstrap；Δ 用配對 bootstrap。
- **test split 從未讀取**：沒有讀 test 標籤、沒有算 test 分數、沒有抽 test 特徵。程式都先依 split.csv 過濾，再讀標籤。
- **資料量**：train 15,309 例、val 5,055 例（train＋val 共 20,364 例）。各疾病實際有標籤的例數略少，例如膽結石是 train 15,113／val 4,989。骨折 val 用共同集：4,941 例，其中 553 例陽性。
- **年齡配對 AUROC**：只比較同一年齡層內的陽性與陰性配對。年齡和性別從未當作模型輸入。
- **val 不是完全沒看過**：多數後續實驗是在看過前一版 val 之後才設計的。模型雖然只依 train 選，最終數字仍以鎖定設定後只跑一次的 test 為準。
- 本檔只列彙總指標，沒有病人 ID，也沒有逐例標籤。

## 各疾病最佳（val AUROC）

「前一版」是被取代的正式版本；如果沒有中間版本，就是 expB E3_late 基準。

| 疾病 | 最佳模型：方法 | val AUROC [95% CI] | 年齡配對 | 前一版 → Δ [95% CI] | 日期 | 報告（A6000） |
|---|---|---|---|---|---|---|
| 骨折（fracture） | **a6x10w2_tv7**：site-wise 堆疊（S3n·max），成分包括 1.5 mm 椎體微調、拉直肋骨 tile 凍結 probe（a6，10 seed）、後方附件 PE 頭（t_vert）、pelvis15_ft | **.897 [.882, .912]**（train CV .9024） | .888 | rp_a6mix3 .897 → +.001 [−.002, +.003]，n.s. | 11-09 | `outputs_fracture/push90b/integ/a6x10w2_tv7/{report,compare_vs_897}.md`、`push90b/p90b_selection.json` |
| 膽結石（gallstones）⚠ | **gs15_ft 第二層 LR（＋CNN）**：Exp N 的 N_U logit、1.5 mm 膽囊 crop 單元頭，再加上 TS encoder stage 3–4 微調頭 | **.936 [.922, .949]**（tier0） | .931 | Exp N .881 → +.054 [+.039, +.070] | 11-07 | `outputs_final/gallstones15_ft/report.md` |
| 動脈粥狀硬化（atherosclerosis） | **Exp N**：REF，加上腎下主動脈、分叉、兩側總髂的 4 個 crop 單元頭，再加分段管壁鈣化 HU | **.921 [.913, .928]** | .801 | .912 → +.009 [+.006, +.012] | 11-06 | `outputs_final/region_units/atherosclerosis/report.md` |
| 骨質減少（osteopenia，報告定義） | **osteopenia_v15**：5 個 window logit 加上 1.5 mm L1–L4 逐節 head（LR） | **.890 [.870, .908]** | .798 | .884 → +.006 [+.001, +.013] | 11-06 | `outputs_final/osteopenia_v15/report.md` |
| 腸阻塞（bowel obstruction） | E3_late、胃單元 head、腸道口徑與內容物特徵（GBM） | **.956 [.946, .965]** | .956 | E3_late .950 → +.006 [+.000, +.011] | 11-06 | `outputs_final/integrated_best/report.md`（r2 版 .958 不顯著，未採用：`region_units/bowel_obstruction_r2/report.md`） |
| 膽管擴張（bile duct dilation） | **BD5**：cat4、E3_late、E0、肝門 crop embedding、HU、管狀與肝門特徵（GBM） | **.911 [.899, .922]** | .900 | E3_late .834 → +.077 [+.064, +.091] | 11-06 | `outputs_final/integrated_best/report.md`、`biliary_improve/final/report.md` |
| 腹腔游離氣體（free air） | **Exp M**：REF，加上肝頂前方橫膈下與中線前腹 2 個 crop 的單元頭（LR） | **.912 [.890, .931]** | .911 | .901 → +.011 [+.002, +.019] | 11-06 | `outputs_final/region_units/free_air/report.md` |
| 腹水（ascites） | **Exp M**：REF，加上 Morison 袋、脾周、骨盆陷凹 3 個 pocket crop 的 U_org 頭（LR） | **.916 [.905, .925]** | .915 | .906 → +.009 [+.005, +.014] | 11-06 | `outputs_final/region_units/ascites/report.md` |
| 腎水腫（hydronephrosis） | **Exp M**：REF，加上腎竇／腎盂 crop 單元頭（兩腎 MIL max，GBM） | **.957 [.945, .967]** | .957 | .944 → +.012 [+.005, +.020] | 11-06 | `outputs_final/region_units/hydronephrosis/report.md` |
| 闌尾炎（appendicitis，原合併定義） | **RLQ 單元**：REF 加上以迴盲部為中心的右下腹 crop 單元頭（U_rlq_cs） | **.920 [.891, .947]** | .885 | REF .888 → +.033 [+.015, +.051] | 11-06 | `outputs_final/appendicitis_rlq_unit/report.md` |
| 確定的急性闌尾炎（拆分標籤） | 同上：REF＋U_rlq_cs | **.936 [.912, .958]** | .901 | REF .902 → +.034 [+.021, +.048] | 11-06 | `outputs_final/appendicitis_rlq_unit/report.md` |
| 脂肪肝（hepatic steatosis，原定義） | E3_late、肝‖(肝−脾) head、肝脾 HU（GBM） | **.832 [.814, .850]** | .832 | E3_late .819 → +.013 [+.004, +.021] | 11-06 | `outputs_final/integrated_best/report.md` |
| 瀰漫性脂肪肝（拆分標籤） | E3＋HU | **.925 [.909, .940]** | .921 | E3_late .905 → +.020 [+.007, +.031] | 11-06 | `outputs_final/steatosis_split/report.md` |
| 局部脂肪浸潤（focal steatosis，次要） | **Exp N**：REF，加上鐮狀韌帶、膽囊窩、肝門 3 個 crop 的單元頭 | **.676 [.644, .708]** | .668 | .665 → +.012 [−.003, +.026]，n.s. | 11-06 | `outputs_final/region_units/focal_steatosis/report.md`（3D CNN 版 .681 不顯著，未採用：`focal_steatosis_cnn/report.md`） |
| 膽囊已切除（absent gallbladder） | E3_late、膽囊體積、膽囊窩、手術夾特徵（GBM） | **.984 [.981, .988]** | .984 | E3_late .956 → +.028 [+.024, +.032] | 11-06 | `outputs_final/integrated_best/report.md` |
| 脾腫大（splenomegaly） | E3_late、脾臟幾何、體型特徵（GBM） | **.981 [.977, .986]** | .981 | E3_late .980 → +.001 [−.001, +.004]，n.s. | 11-06 | `outputs_final/integrated_best/report.md` |
| 腹主動脈瘤（AAA） | E3_late、主動脈直徑幾何、金屬特徵（GBM） | **.990 [.985, .995]** | .977 | E3_late .975 → +.016 [+.006, +.027] | 11-06 | `outputs_final/integrated_best/report.md`（val 陽性只有 32 例，CI 寬） |
| 膽泥（gallbladder sludge，新 finding） | **S_NU**：膽結石最佳特徵組改用膽泥標籤（GBM） | **.895 [.861, .927]** | .883 | 新 finding | 11-06 | `outputs_final/gallbladder_sludge/report.md` |
| 影像骨質疏鬆（HU 定義，新 finding） | 只用 embedding（3 mm 腰椎 5 window 加 1.5 mm L1–L4），不含 HU 特徵。≤110 HU 為主要定義，≤160 HU 為篩檢定義 | **≤110：.988 [.985, .991]**；≤160：.988 [.986, .990]；HU MAE 9.5 | .977／.981 | 新 finding | 11-06 | `outputs_final/osteoporosis_hu/report.md` |

**main_all**：14 項原始 finding 的平均。integrated_best v2（11-06）為 .920 [.915, .923]，年齡配對 .898。v2 是在骨折（.847）和膽結石（.866）更新之前算的，所以目前的實際平均更高，尚未重算。報告在 `outputs_final/integrated_best/report_v2.md`。

### 註記與來源衝突
- **⚠ 膽結石**：.936 來自 tier0 中期堆疊。train 只用 4,028 列（全部陽性，加上依年齡和性別 1:2 配對的陰性），val 有 4,989 例（465 例陽性）。這個結果是依 train OOF 選出的，但全量 train 版本沒有完成（`gallstones15/CNN_full_FAILED`）。integrated_best v2 仍寫 .866，GPU_QUEUE 寫 .881（Exp N）。本表依最新的選定結果列 .936，**全量重跑前請視為暫定值**；保守正式值是 Exp N 的 .881 [.863, .898]（年齡配對 .868）。
- **骨折**：a6x10w2_tv7 和 rp_a6mix3 同為 .897，差 +.001，不顯著。依 train CV 選的是 a6x10w2_tv7（CV .9024，rp_a6mix3 是 .9014），所以它是正式版本。
- **腹水**：單元報告寫 .916，integrated_best v2 重算寫 .915（同一模型，差在 bootstrap 和共同集）。r2 版 .917 的 CI 下界約 0，未採用。
- **闌尾炎的年齡配對**：上表取單元報告的值。integrated_best v2 用另一種年齡配對計算，結果是合併定義 .910、確定急性 .926。
- **膽管擴張**：1.5 mm 肝門 crop（bd15）在 11-06 暫停後沒有完成，正式值仍是 .911。

## 骨折進度

| 日期 | 正式版本 | val AUROC [95% CI] | 主要改變 |
|---|---|---|---|
| 11-06 | Exp J add 10 seed | .847 [.829, .865] | 1.5 mm 椎體加骨骼單元 |
| 11-06 | C3（final_combo） | .854 [.836, .871] | 加入實驗 L 的 1.5 mm 肋骨和骨盆環單元頭 |
| 11-06 | sitewise S2·lse+any | .862 [.844, .878] | site-wise 堆疊 |
| 11-07 | final_ft S3a／with_rib | .872 → .878 | 椎體、骨盆 encoder 微調，加肋骨 MIL probe（中間版本） |
| 11-08 | rp_both_aa | .882 [.866, .898] | rib_v3（28 seed）加 pelvis15 |
| 11-08 | rsP_mix_repl | .892 [.876, .907] | 拉直肋骨 tile 凍結 probe，加 pelvis15_ft |
| 11-09 | rp_a6mix3 | .897 [.881, .911] | 肋骨 a6（對側和相鄰肋骨同格對比），加後方附件 PE 頭 |
| 11-09 | **a6x10w2_tv7（正式）** | **.897 [.882, .912]** | PE 頭改用 t_vert 目標，a6 改 10 seed；未達 .90 |

子群（a6x10w2_tv7，val）：

| 子群 | val 陽性 | AUROC [95% CI] | 對 .862 的 Δ |
|---|---|---|---|
| 椎體壓迫性 | 290 | .955 [.941, .967] | +.018 |
| 肋骨 | 158 | .884 [.856, .910] | +.055 |
| 只有肋骨 | 108 | .852 [.816, .883] | +.072 |
| 骨盆／薦椎／髖 | 135 | .896 [.866, .922] | +.025 |
| 急性（含混合） | 112 | .892 [.857, .924] | +.011（n.s.） |
| 陳舊 | 336 | .897 [.877, .916] | +.040 |

整體年齡配對 .888，90% 特異度下敏感度 72.7%。

## 待處理
1. **骨折標籤定義稽核（只用 train，需使用者決定）**：報告在 `outputs_fracture/rib_straight/label_audit/report.md`。
   - 高分陰性 268 例。其中有椎體成形術骨水泥、或「壓迫變形但報告沒寫骨折」的例子，屬於標籤定義問題。
   - 低分陽性 168 例。其中有後方附件骨折、FOV 外骨折、陳舊骨折。
   - 改定義的 train 上限估計約 +.008～+.015。這會改變評估定義，必須在看 test 前鎖定，而且 val 要用同一套規則重算。正式標籤目前未改。
2. **骨折停在 .897**：push90b 第二輪（11-09 09:50）沒有達到 .90，GPU 已釋放。可以再試的方向：上面的標籤定義、肋骨 a6 加 seed（邊際約 +.0002）、無位移或癒合中的肋骨骨折。
3. **膽結石全量版**：gs15 全量 CNN 和 gs15_ft 全量 train 尚未完成，完成後才能把 .936 定為正式值。
4. **main_all 重算**：要用骨折 .897、膽結石、動脈粥狀硬化 .921 的新分數重算。
5. **test**：所有設定鎖定後只跑一次；標籤定義待決事項見 `mw_exp/LABEL_DECISIONS_PENDING.md`。
