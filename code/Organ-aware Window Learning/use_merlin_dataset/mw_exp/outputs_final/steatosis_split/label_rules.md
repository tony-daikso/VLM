# 脂肪肝拆分標籤規則：瀰漫性脂肪肝 vs 局部脂肪浸潤

- 決定：使用者，2026-11-05 23:40（LABEL_DECISIONS_PENDING.md 第 1 項）。規則在 2026-11-05 鎖定，**在任何 test 使用之前**。train／val／test 一律用同一支程式、同一組規則。
- 程式：`scripts/hs_split_labels.py`（sha256 `0a2df1510984b4d3f0f29f401ef069d64ebd1dcee7733113ad15f0428c41e25a`）。
- 輸入只有兩種：① 報告文字（`merlin_disease14/reports.json`）② 現有的 Qwen 標籤（`labels_disease14_all.jsonl` 的 `hepatic_steatosis`：category + quotes）。**完全不使用任何模型分數**。
- split 安全：先讀 `split.csv`，只保留 `--splits` 指定的 study_id。jsonl 是先用 regex 從原始行取出 study_id、確認在允許集合內，才 json 解碼；reports.json 用依 key 過濾的掃描器，不在允許集合內的值只跳過、不解碼也不保存。`--splits test` 必須加 `--allow-test`，否則程式直接拒絕（已驗證）。
- 產出 `labels_trval_split.csv`：每例只有關鍵字旗標與標籤，不含報告文字。

## 1. 文字處理

1. 報告切句（句點／分號／換行）。只看「肝臟脂肪句」：句中有肝臟脈絡字（liver、hepat、steato、falciform、gallbladder fossa、porta hepatis、periportal、ligamentum、fissure、segment 4/IV、caudate、quadrate、left/right lobe、fatty infiltration、focal fat…），而且有脂肪或低密度字（steato*、fat、fatty、hypoattenuat*、hypodens*、decreased/low attenuation、sparing）。
2. 排除：
   - 病史／適應症句（history of、indication、clinical history、evaluate for、rule out、r/o、以 history/clinical/indication/comparison/technique 開頭）。
   - 只講其他器官的句子（pancreas、kidney、bone/marrow、bowel、ileum、hernia、lipoma、fat stranding…，且句中沒有 liver/hepat/steato）。
   - 沒有脂肪字、但描述病灶的低密度句（lesion、cyst、mass、nodule、hemangioma、too small to characterize、subcentimeter、帶有 mm/cm 大小）。
3. 否定：比對位置前 80 字元內、同一子句（以 , ; : but however with which 切開）中有 no／not／without／negative for／free of／resolved／absence of／rather than，就算否定，不採用。
4. Qwen 標籤的 quotes 也用同樣規則判讀（quotes 本來就是脂肪肝證據，不要求肝臟脈絡字）。報告與 quotes 取聯集。

## 2. 兩種文字證據

- **局部（F）**：句中有局部脂肪片語，而且有脂肪字（steato／fat／fatty）：
  - focal／focally／geographic／multifocal／nodular／regional／patchy + 3 字內接脂肪或低密度字（例：focal fatty infiltration、focal fat、geographic hypoattenuation…favors fatty infiltration）
  - area／region／zone／focus／foci of + 脂肪字（例：small focus of fatty infiltration、areas of fatty replacement）
  - 脂肪字與典型位置（falciform ligament、gallbladder fossa、porta hepatis、periportal、fissure、ligamentum teres、segment 4/IV）在 9 字內（例：fatty infiltration adjacent to the falciform ligament）
  - **focal (fatty) sparing 不算局部脂肪**（sparing 表示其餘肝臟有脂肪，算瀰漫）。片語後面接腸道、fat stranding 等非肝臟字也不算。
- **瀰漫（D）**：先把局部片語從句子中移除，剩下的文字若有：
  - steatosis／steatotic／steatohepatitis、fatty liver／change／infiltration／replacement、fatty in appearance／attenuation、the liver is (mildly) fatty、fatty-replaced liver、NAFLD／NASH
  - diffuse(ly)／overall／generalized + 5 字內接脂肪或低密度字（例：diffusely decreased in attenuation）
  - 肝臟低密度描述（liver … hypoattenuating／decreased attenuation）：沒有脂肪字時，句中不能有病灶或大小字，也不能是「focus／area／region」描述（除非有 diffuse／overall／throughout／the liver is）
  - **sparing（focal fatty sparing、spared）→ 瀰漫**

## 3. 標籤（主要定義）

原標籤 P = yes／probably_yes；N = no／probably_no／not_mentioned；unclear 兩者皆 NaN。

| 原標籤 | 文字證據 | 子類型 | y_diffuse（瀰漫性脂肪肝） | y_focal（局部脂肪浸潤） |
|---|---|---|---|---|
| P | D 且 F | both | 1 | 1 |
| P | 只有 D（含 sparing） | diffuse | 1 | 0 |
| P | 只有 F | focal | 0 | 1 |
| P | 兩者皆無 | unassigned | NaN | NaN |
| N | 不論文字 | neg | 0 | 0 |
| unclear | — | unclear | NaN | NaN |

註：
- 「both」包含「瀰漫性脂肪肝 + 另有局部脂肪」（例：hepatic steatosis…7 mm hypodensity adjacent to the falciform ligament, possibly focal fat）。依使用者規則兩者皆陽性。
- unassigned（train 8、val 5）：原標籤陽性，但報告和 quotes 都沒有可判讀的字句（多半是只有病史句提到，或「Slight hepatic hypoattenuation」這類沒有脂肪字的描述）。兩個 finding 都設為 NaN（不訓練、不評估）。
- 原陰性一律保持陰性（使用者規則）。

## 4. 敏感度分析用的變體（不是主要定義，只用來評估，不重新訓練）

| 欄位 | 定義 |
|---|---|
| `y_diffuse_excl` | 瀰漫陽性 vs 原陰性；只有局部的病例設 NaN（不當陰性） |
| `y_focal_excl` | 局部陽性 vs 原陰性；只有瀰漫的病例設 NaN |
| `y_focal_negclean` | 主要定義，但「原陰性、報告卻明確寫了局部脂肪」的病例（`neg_focal_text=1`）設 NaN |

## 5. 數量

| | train | val |
|---|---|---|
| 全部 | 15,309 | 5,055 |
| 原陽性／原陰性／unclear | 2,209／12,943／157 | 715／4,294／46 |
| 子類型：只有瀰漫 | 1,452 | 444 |
| 子類型：只有局部 | 603 | 217 |
| 子類型：兩者（重疊） | 146 | 49 |
| 子類型：unassigned | 8 | 5 |
| **瀰漫性脂肪肝 陽性／陰性（盛行率）** | **1,598／13,546（10.6%）** | **493／4,511（9.9%）** |
| **局部脂肪浸潤 陽性／陰性（盛行率）** | **749／14,395（4.9%）** | **266／4,738（5.3%）** |
| 瀰漫陽性中 sparing | 87 | 38 |
| 局部陽性中提到典型位置（鐮狀韌帶／膽囊窩／肝門／fissure／IV 段） | 670（89%） | 228（86%） |
| 局部陽性中措辭不確定 | 180（24%） | 68（26%） |
| 局部陽性中原標籤是 probably_yes | 142 | 59 |
| 原陰性但報告明確寫局部脂肪（`neg_focal_text`） | 209（no 190、not_mentioned 17、probably_no 2） | 91 |
| 原陰性但報告有瀰漫字眼（`neg_diffuse_text`，多為否定以外的模糊描述） | 137 | 31 |

## 6. 人工稽核（train，約 60 例，只看產生的旗標與命中的短片語，沒有印出整份報告）

兩輪各抽約 57 例（both 14、focal 14、unassigned 9–10、diffuse 10、原陰性但有局部字 12），依稽核結果修正規則，修正完才鎖定：

- 第 1 輪發現並修正：① 「focal fatty sparing」被當成局部 → 改為 sparing 只算瀰漫；② 「Left hepatic lobe hypodensity too small to characterize」「Subcentimeter liver hypodensities」被當成瀰漫 → 沒有脂肪字的低密度句排除病灶／大小描述；③ 「steatotic」「The liver is fatty」「Diffusely decreased in attenuation」「consistent with focal fatty infiltration」（quotes 沒有 liver 字）漏抓 → 補上；④ 「Diffuse steatosis limiting evaluation for…」被 `evaluation for` 誤判為病史句 → 改成只排除 `eval for`。
- 第 2 輪發現並修正：「Focal fatty infiltration of the terminal ileum」「fat stranding … periportal edema」被當成局部 → 局部片語後面接非肝臟字就排除。
- 第 2 輪結果（修正前的 57 例）：both 14/14 合理（瀰漫 + 另述局部脂肪，或 geographic 型）；focal 14/14 正確；diffuse 10/10 正確（1 例「geographic … right hepatic lobe hypodensity likely reflecting fatty infiltration」嚴格說應算局部，屬邊界例）；unassigned 9 例中 3 例是「hypoattenuating liver parenchyma」類描述（第 3 次修正已改善）。
- **重要發現**：原陰性但報告寫了局部脂肪的 12 例中，10 例確實是局部脂肪（例：「Focal fat along the falciform ligament」「Focal fat deposition adjacent to the falciform ligament」），Qwen 把它們標成 no（大概把題目理解為「瀰漫性脂肪肝」）。依使用者規則主要定義維持陰性；這是局部脂肪浸潤陰性中的已知漏標（train 約 209 例，約佔局部陽性數的 28%）。`y_focal_negclean` 用來估計影響。**建議**：test 前決定是否把這類病例改為局部陽性（需要使用者同意，屬標籤定義改變）。
