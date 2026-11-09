# 待使用者決定的標籤定義（2026-11-05 記錄）

全部都必須在**看 test 之前**鎖定，val 與 test 套用同一定義。來源是各疾病錯誤分析（train OOF；val 只描述）。
決定前：訓練與評估維持現行標籤（yes/probably_yes = 陽性；no/probably_no/not_mentioned = 陰性）。

| # | 疾病 | 待決定的問題 | 影響（估計） | 來源 |
|---|---|---|---|---|
| 1 | 脂肪肝 | ✅ **已決定（2026-11-05 23:40，使用者）：分成兩個 finding——「瀰漫性脂肪肝」與「局部脂肪浸潤」**，各自獨立標籤與評估；原合併標籤只保留作參考。val／test 一律用同一定義 | 移出局部型後瀰漫型評估約 0.90。**已實作（2026-11-06）**：規則鎖定於 `outputs_final/steatosis_split/label_rules.md`（`scripts/hs_split_labels.py`，sha256 0a2df151…）；val：瀰漫性脂肪肝 .925（E3_late 肝‖(肝−脾)+HU），局部脂肪浸潤 .665（建議列次要）。test 標籤待鎖定後以 `--splits test --allow-test` 產生；finding_organ_map 修改建議見 report.md §5 | analysis_steatosis/report.md、steatosis_split/report.md |
| 2 | 腹水 | 微量／少量積液的界線：2,802 例「微量/少量」被標 probably_no，1,623 例「少量」被標 yes。選項：三級化／模糊陰性改 unclear | 評估 +0.014；訓練時忽略無變化，改陽性 −0.015 | analysis_ascites/report.md |
| 3 | 脾腫大 | probably_no 與 borderline（邊界）病例怎麼歸類？選項：另歸一類／排除 | 評估 +0.001～0.013（定義改變，不是模型進步） | analysis_splenomegaly/report.md |
| 4 | 腸阻塞 | 腸麻痺（ileus）、「腸道擴張但未判定阻塞」、probably_no 的陰性怎麼處理？選項：訓練時忽略（val +0.005）／另立一類。**不要改成陽性**（會變差） | 評估排除後 0.944 → 0.957 | analysis_bowel_obstruction/report.md |
| 5 | 膽管擴張 | ① probably_no 陰性（5.3%）與只寫 prominent／正常上限的陽性（13.4%）：排除或降權？② 有支架／氣膽的病例是否另標「已引流」類別？ | 標籤清理 +0.007～0.016；支架/氣膽處理主要是修掉約 0.03 虛高 | analysis_biliary/、膽管擴張子頁 |
| 6 | 骨質減少 | ✅ **已決定（2026-11-06，使用者：「我真正想要的是『影像上看得到的骨質疏鬆』」）：新增以 HU 定義的 finding「影像骨質疏鬆」**；原本以報告定義的「骨質減少」維持不變（正式 val .890）。訓練端的標籤處理（B／C／D／D2）都不採用。新 finding 的定義依使用者規格（原文見表格下方）：L1 骨小樑 ROI 的 HU 為連續標籤；≤110 為陽性（主要），≤160 為篩檢陽性（次要）；L1 有骨折、硬化或量測無效時改用 L2；有骨水泥則整例排除。**已實作（2026-11-06）**：規則鎖定於 `outputs_final/osteoporosis_hu/label_rules.md`（`scripts/oh_labels.py`，sha256 bcd20667…；`oh_roi.py` 3bf0a956…；`oh_report_flags.py` 37fd3b67…；labels_hu_trval.csv sha256 92a60675…）。有標籤 train 14,801／val 4,874；≤110 陽性 1,058／328，≤160 陽性 4,151／1,354；骨水泥排除 144／63；改用 L2 520／172；無可用椎體 341／115。只用嵌入（不含 HU）的模型：val HU MAE 9.5、r .975；≤110 AUROC .988 [.985, .991]；≤160 .988。顯影相位造成約 +16 HU 的偏移；依規格門檻不改，校正方案待使用者決定（report.md §5）。test 標籤要等鎖定後才產生（`oh_roi`／`oh_labels` 需先加 `--allow-test`） | 新 finding（定義改變）；原報告模型在新標籤上 ≤110 .948 | analysis_osteopenia/report.md、osteopenia_labels/report.md、osteoporosis_hu/report.md |
| 7 | 動脈粥狀硬化 | 鈣化量高但報告沒寫的陰性（約 12%）是否改 unclear／soft label？是否用同義詞（vascular/aortic/iliac calcification、calcified plaque）重新判讀 not_mentioned？ | 評估約 +0.03（屬標籤修正） | analysis_atherosclerosis/report.md |
| 8 | 闌尾炎 | ✅ **已決定（2026-11-05，使用者）：分成兩個 finding——「確定的急性闌尾炎」與「可疑／治療中／追蹤／術後」**，各自獨立標籤與評估；原合併標籤只保留作參考。使用者之後會試把第二類當**軟標籤**（例如給 0.3–0.7 的目標值）訓練確定組。val／test 用同一定義 | 帳面約 +0.03（定義改變） | analysis_appendicitis/report.md |
| 12 | 脂肪肝（局部） | 原標籤是 no、但報告明確寫了局部脂肪的病例（train 209、val 91；抽查 12 例中 10 例確實是局部脂肪）：維持陰性，還是改成「局部脂肪浸潤」陽性？目前主定義算陰性 | 局部型目前只有 val .665，屬次要指標；改陽性會讓局部型標籤更乾淨 | steatosis_split/report.md |
| 9 | 腹主動脈瘤 | 「修補後/EVAR 病史」與「目前有動脈瘤」是否分開？ | 評估約 +0.003 | analysis_aaa/report.md |
| 11 | 膽結石 | ✅ **已決定（2026-11-06，使用者）：另外立一類「膽泥」**。膽結石定義不變（只有膽泥仍是膽結石陰性）。**已實作（2026-11-06）**：規則鎖定於 `outputs_final/gallbladder_sludge/label_rules.md`（`scripts/sl_labels.py`，sha256 e1805e26…；labels_trval_sludge.csv sha256 d940d4b1…）；膽泥 train 221／val 73（只有膽泥 112／41、膽泥＋結石 49／14、膽結石 unclear 60／18）。膽泥 S_NU val .895 [.861, .927]（年齡配對 .883）。膽結石官方仍 .881；移除只有膽泥的次要檢視 .883；用膽泥模型抑制（3 類／堆疊／減權）依 train OOF 都沒有增益（val Δ −.006～+.000），主版本不變。test 標籤待鎖定後以 `--splits test --allow-test` 產生 | 定義改變；膽結石上限只有 +.002 | gallbladder_sludge/report.md |
| 10 | 骨折 | 是否並列「臨床顯著骨折」（急性、中重度壓迫、非術後髖/股骨）為次要指標？（不改主標籤） | 該指標 AUROC .868，受年齡影響最小 | analysis_old/report.md |

**新 finding（使用者決定拆出，2026-11-06）**：膽泥（gallbladder sludge，由第 11 項拆出）。val .895 [.861, .927]，陽性 73。列為次要 finding，**不納入 main_all 14 項**，除非使用者另外決定。詳見 `outputs_final/gallbladder_sludge/report.md`。

**第 6 項使用者規格原文（2026-11-06，影像骨質疏鬆）**：

> USER'S EXACT SPEC for 影像骨質疏鬆 — replace my defaults with this: (1) Label = L1 trabecular-bone ROI HU, stored as a CONTINUOUS value (regression target; report MAE/Pearson/ICC for predicted HU from embeddings-only model). (2) Binary targets derived from it: HU ≤110 → 陽性 (high-specificity definition, primary); HU ≤160 → 篩檢陽性 (screening definition, secondary). Note ≤ (inclusive), not <. (3) If L1 has a fracture OR sclerosis → use L2 instead (define sclerosis detection objectively, e.g. abnormally high trabecular HU / focal sclerotic lesion rule; document it). (4) Exclude cases with bone cement (vertebroplasty/kyphoplasty; detect via very high HU metal-like density in L1/L2 body, e.g. >1000–1500 HU voxels fraction, and/or report mentions) — excluded entirely, not relabeled. The ROI must be trabecular only (eroded body mask avoiding cortex/endplates). Keep contrast-phase check as a reported analysis (no threshold change unless user decides). Report counts per split for each rule (L2 fallback, sclerosis, cement exclusions). Record this spec verbatim in LABEL_DECISIONS_PENDING.md #6 and in the rules file.

**新 finding（使用者決定，2026-11-06）**：影像骨質疏鬆（以 HU 定義，由第 6 項新增）。val：只用嵌入的模型 ≤110 為 .988、≤160 為 .988，HU MAE 9.5。和報告定義的「骨質減少」分開報告，**不取代** main_all 14 項中的骨質減少；是否納入主表由使用者決定。詳見 `outputs_final/osteoporosis_hu/report.md`。

另外建議（不是標籤定義，但也要在 test 前決定）：主要報告是否並列**年齡配對 AUROC**（動脈粥狀硬化、骨質減少、闌尾炎、陳舊骨折受年齡影響大）。
