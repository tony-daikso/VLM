# 任務：在 Merlin 資料集上測試「方法四」報告生成

（2026-11-09 使用者交付的任務原文，作為所有 agent 的規格依據）

## 目標
用 Merlin 資料集測試這個方法：拿 OWL 的影像 embedding 和 14 類分類結果，訓練一個 LLM，讓它照 Merlin 醫師的寫法，寫出 14 類疾病的 findings 句子。陽性句、陰性句都要，屬性也要寫對（側別、程度、椎體節段、AAA 直徑、脂肪肝分級等）。
同時做一個「方法一」的 baseline：用分類結果套固定樣板句。兩個方法在同一批病人上比較。
14 類：hepatic_steatosis, surgically_absent_gallbladder, splenomegaly, hydronephrosis, abdominal_aortic_aneurysm, atherosclerosis, osteopenia, bowel_obstruction, appendicitis, ascites, free_air, gallstones, biliary_ductal_dilation, fracture。

## 參考實作（CG 專案已經做過一輪，先讀這些）
- 整體規劃：`/Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data/report_gen/PLAN.md`（第 3 節是方法一，第 5 節是方法四）
- 方法四的資料準備與模型設計：`/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/report_gen/phase3/phase3_prep.md`，程式在 `/Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data/report_gen/phase3/`
- 從報告抽出每類疾病原句的 Qwen prompt 和後處理：`/Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data/report_gen/standardize/`（std14_prompt.md、std14_lib.py）
- 方法一的句庫、門檻、組裝、評估：`/Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data/report_gen/` 底下的程式，結果說明在 `/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/report_gen/eval_phase1.md`
這些是參考用，**不要修改 CG 專案的任何檔案**。

## Merlin 這邊已有的資產（請自己確認實際位置與格式）
- OWL 規格與程式：`/Users/tony.tu/Desktop/戴承智慧/VLM/code/Organ-aware Window Learning/use_merlin_dataset/`（mw_exp/）
- OWL 特徵輸出：A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs`（另外 Spark、CT-rate 上也有部分）
- 14 類報告標籤：`/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/disease14/labels_disease14.jsonl`
- Merlin 報告原文在 `/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/` 底下。報告受 DUA 限制，**絕對不能 commit**。
- 各疾病的最佳分類頭和 E3_late 模型：A6000 mw_exp/outputs_final/models/

## 一定要注意的事
1. **分類分數不能有洩漏**：OWL 的分類頭是在 Merlin train 上訓練的，train 病人的分數是「看過答案」的，會過度自信。給 LLM 當輸入的分類分數，train 部分一定要用 out-of-fold 預測（K-fold cross-fitting），val/test 用正常推論。做不到的話先停下來跟我說。
2. **test split 鎖住**：開發和調參一律只用 val。test 要等我確認之後才能跑。
3. **訓練目標用醫師原句**：目標是模仿醫師的寫法，所以每類疾病用報告裡的原句，保留原本的用字和順序，陽性、陰性都要。
   - 比較前次的用語（stable、progression、compared with）要拿掉，因為模型看不到前次影像。
   - se/im 編號拿掉。
   - 沒有測量依據的尺寸拿掉；主動脈直徑、脾臟大小這類有測量值的可以保留。
   - 明顯錯字修正，其他用字不動。
   - Merlin 報告如果本來就依器官分段，可以先用規則抽句，抽不到的再用 Qwen。
4. **評估**
   - 疾病有無：用產生 labels_disease14.jsonl 的同一套 Qwen prompt，反抽生成文字裡的 14 類，和原報告標籤比，算 sensitivity、specificity、F1。
   - 關鍵錯誤率：原報告陽性、生成卻寫陰性。這是最重要的指標。
   - 屬性正確率。
   - 風格相似度：BLEU、ROUGE-L、BERTScore。
   - 方法一 baseline 和方法四在同一批 val 病人上比較。
   - 另外做一個只用文字的消融（只給分類分數和測量值、不給 embedding），看 embedding 到底有沒有幫助。
5. **模型建議**（照 CG 的設計，可以視情況調整）
   - 主模型：Lingshu-7B（在 Spark 上），或 Qwen 系列 7–9B。
   - 架構：OWL 單元特徵加 type embedding，用 Perceiver resampler 壓成 32 個影像 token；分類分數、測量值、院區或機型寫成文字放進 prompt。
   - 訓練：Stage 1 只訓練 projector；Stage 2 用 LoRA（r16/α32）。

## 伺服器與檔案規則（必須遵守）
- 所有伺服器上的檔案一律放 Tony 資料夾：A6000、Spark 是 `~/Desktop/Tony/...`；CT-rate 是 `/media/daikso/Tony/...`。
- Mac 上不放大檔，只放程式和小型結果。Merlin 相關檔案放在 `/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin/` 底下，建一個新的子資料夾（本資料夾：report_gen/）。
- **GPU 現在被 CG 專案占用**：A6000 `:8000` 的 vLLM，以及 Spark 的 docker 容器 `std14-qwen`（port 8005），都是 CG 專案在用。不要停、不要動它們。需要用 GPU 訓練之前，先用 nvidia-smi 確認狀況，再問我。
- A6000 只有 53 GB RAM，重負載下重開過好幾次，平行工作要節制。另外 A6000 上**不要用 DataLoader 的 `pin_memory=True`**，會在背景悄悄壞掉約 6% 的 batch。
- 不要用 CT-rate 的 RTX 3080。
- 只做這份任務要求的事，不要動既有 pipeline，不要 commit。
- 預計超過 1–2 分鐘的工作交給 background agent 或背景執行。

## 執行順序與停點
1. 盤點：確認 Merlin 的報告、標籤、OWL 特徵、各 split 的位置、格式和數量；確認分類分數能不能做 out-of-fold。
2. 建立訓練目標：抽出每類疾病的原句，算出屬性覆蓋率和清理統計。
3. 方法一 baseline：建句庫，在 train 上訂門檻，在 val 上組裝草稿並評估。
4. **停點**：把盤點結果、資料統計、方法一在 val 上的成績、方法四的訓練計畫（模型、時間、要用哪台 GPU）整理給我，等我確認後再開始訓練。
5. 訓練方法四（含只用文字的消融），在 val 上評估，和方法一比較。

## 回報
- 回覆一律用繁體中文（台灣用語）。結果表格裡的疾病名稱用中文：脂肪肝、膽囊切除術後、脾腫大、腎水腫、腹主動脈瘤、動脈粥狀硬化、骨質減少、腸阻塞、闌尾炎、腹水、腹腔游離氣體、膽結石、膽管擴張、骨折。
- 每個停點交一份 md 報告，放在 Merlin 資料夾，內容包含數字、檔案路徑、遇到的問題和需要我決定的事。
