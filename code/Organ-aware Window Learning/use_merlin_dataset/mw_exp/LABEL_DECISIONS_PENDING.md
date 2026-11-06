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
| 6 | 骨質減少 | 「未提及」但 L1 骨小樑 HU <110（或 <130）的高齡陰性，是否改 unclear／PU learning？是否加一個 HU 定義的客觀輔助目標？ | 清理後評估 +0.01～0.025；原始標籤約 +0.005 | analysis_osteopenia/report.md |
| 7 | 動脈粥狀硬化 | 鈣化量高但報告沒寫的陰性（約 12%）是否改 unclear／soft label？是否用同義詞（vascular/aortic/iliac calcification、calcified plaque）重新判讀 not_mentioned？ | 評估約 +0.03（屬標籤修正） | analysis_atherosclerosis/report.md |
| 8 | 闌尾炎 | ✅ **已決定（2026-11-05，使用者）：分成兩個 finding——「確定的急性闌尾炎」與「可疑／治療中／追蹤／術後」**，各自獨立標籤與評估；原合併標籤只保留作參考。使用者之後會試把第二類當**軟標籤**（例如給 0.3–0.7 的目標值）訓練確定組。val／test 用同一定義 | 帳面約 +0.03（定義改變） | analysis_appendicitis/report.md |
| 12 | 脂肪肝（局部） | 原標籤是 no、但報告明確寫了局部脂肪的病例（train 209、val 91；抽查 12 例中 10 例確實是局部脂肪）：維持陰性，還是改成「局部脂肪浸潤」陽性？目前主定義算陰性 | 局部型目前只有 val .665，屬次要指標；改陽性會讓局部型標籤更乾淨 | steatosis_split/report.md |
| 9 | 腹主動脈瘤 | 「修補後/EVAR 病史」與「目前有動脈瘤」是否分開？ | 評估約 +0.003 | analysis_aaa/report.md |
| 11 | 膽結石 | 「只有膽泥、沒有結石」目前一律標陰性（train 164、val 59）。要維持陰性、改 unclear，還是另立一類？V4 把其中 32–36% 判為陽性（其他陰性 FP 率 9.7%） | 改 unclear 會減少 FP；屬定義改變 | analysis_gallstones/subgroups/report.md |
| 10 | 骨折 | 是否並列「臨床顯著骨折」（急性、中重度壓迫、非術後髖/股骨）為次要指標？（不改主標籤） | 該指標 AUROC .868，受年齡影響最小 | analysis_old/report.md |

另外建議（不是標籤定義，但也要在 test 前決定）：主要報告是否並列**年齡配對 AUROC**（動脈粥狀硬化、骨質減少、闌尾炎、陳舊骨折受年齡影響大）。
