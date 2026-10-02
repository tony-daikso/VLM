# RADAR Report Preprocess

> 論文：*RADAR: An Expert-Level Generalist AI for Abdominal CT Diagnosis*, Science（https://www.science.org/doi/10.1126/science.aec6129）
> 官方 code：https://github.com/alibaba-damo-academy/damo-radar （本機：`VLM/RADAR/CT/`）
> 報告處理程式在 `RADAR_train/preprocess_code/`，官方說明見 `docs/PREPROCESS.md`。
> 公開版是**用 Merlin 公開資料重現**的流程（英文報告）。

---

## 1. Pipeline 總覽

輸入是 `ckpt/merlin_report.json`，由 `ckpt/transform_report_to_json.py` 從 Merlin 的 `reports_final.xlsx` 產生：

```python
# 以 'IMPRESSION:' 切出 findings / impression，但後續處理只用完整的 report
info_json[pid] = {'report': report, 'findings': findings, 'impression': impression,
                  'split': split, 'fewshot': fewshot}
```

後續三個步驟都只處理 `split == 'train'` 的報告，而且都是餵**完整 `report`**（findings + impression 不分開）：

| Step | Script | LLM（DashScope） | 處理範圍 | 輸出 |
|---|---|---|---|---|
| 1. 是否提及 | `check_organ_mention.py` | `qwen_plus` | 每份報告 × 26 個器官 | `merlin_report_mention.json`：`{pid: {report, mention: {organ: "yes"/"no"}}}` |
| 2. 抽取描述 | `report_parsing.py` | `qwen_max` | Step 1 = yes 的器官 | `merlin_report_organ_report_v1.json`：`{pid: {report, organ: 描述}}` |
| 3. 正常 / 異常 | `report_parsing_normal.py` | `qwen_max` | Step 1 = yes 的器官 | `merlin_report_organ_normal_v1.json`：`{pid: {report, organ: "normal"/"abnormal"}}` |

- Step 2、3 互相獨立，都只依賴 Step 1。
- 三支程式都用 ThreadPool 平行呼叫 API（26–36 workers），遇到 429 會做 exponential backoff，每 2000 筆存檔一次，可中斷續跑。
- 沒有設定 temperature（使用 API 預設值）。
- LLM 呼叫量：26 次（Step 1）＋ 2 × 提及的器官數（Step 2、3），Merlin 報告大約 50 次 / 報告。

```bash
cd RADAR_train/preprocess_code
# 先在各 script 設定 dashscope.api_key
python check_organ_mention.py     # → merlin_report_mention.json
python report_parsing.py          # → merlin_report_organ_report_v1.json
python report_parsing_normal.py   # → merlin_report_organ_normal_v1.json
```

---

## 2. 器官清單

### LLM 查詢的 26 個器官（`check_organ_mention.py` 的 `organ_dict`）

adrenal gland, aorta, large bowel, duodenum, esophagus, gallbladder, heart, iliac artery, iliac vena, inferior vena cava, kidney, liver, lung, pancreas, portal vein, pulmonary artery, rib, sacrum, small bowel, spleen, stomach, trachea, bladder, cervical vertebrae, thoracic vertebrae, lumbar vertebrae

### 訓練時的 36 個器官（`caption_datasets.py` 的 `organs_cn`）

順序即 **器官 index（= mask label − 1）**：

```
肾上腺 adrenal gland, 主动脉 aorta, 竖脊肌 erector spinae muscle, 脑 brain, 锁骨 clavicle,
大肠 large bowel, 十二指肠 duodenum, 食管 esophagus, 面部 face, 股骨 femur,
胆囊 gallbladder, 臀肌 gluteus muscle, 心脏 heart, 髋关节 hip joint, 肱骨 humerus,
髂动脉 iliac artery, 髂静脉 iliac vena, 髂腰肌 iliopsoas muscle, 下腔静脉 inferior vena cava, 肾 kidney,
肝 liver, 肺 lung, 胰腺 pancreas, 门静脉 portal vein, 肺动脉 pulmonary artery,
肋骨 rib, 骶骨 sacrum, 肩胛骨 scapula, 小肠 small bowel, 脾 spleen,
胃 stomach, 气管 trachea, 膀胱 bladder, 颈椎 cervical vertebrae, 腰椎 lumbar vertebrae, 胸椎 thoracic vertebrae
```

- 有 10 個器官**從不查詢 LLM**：erector spinae、brain、clavicle、face、femur、gluteus、hip joint、humerus、iliopsoas、scapula。
  - 它們的 caption 永遠是預設值 `"normal."`，abnormal flag 永遠是 False。
  - **實際上不參與器官層級的 contrastive loss**：`radar_pretrain.py` L283–287 只對 batch 內至少有一個異常病例的器官算 loss，這 10 個永遠被跳過。
  - 它們仍然參與 **segmentation dice loss**（L413–416），因為 mask 還是 36 類。
  - 可能原因（推測）：腦、臉、鎖骨、肱骨、肩胛骨不在腹部 CT 範圍內；股骨、髖關節、肌肉很少在腹部報告中被單獨描述。36 類 mask 來自 TotalSegmentator 的全身分類，但 RADAR 只查報告實際會寫到的器官。
  - 限制：髖關節退化、股骨骨折、腰大肌膿瘍這類發現，**器官層級的訓練學不到**。
- **duodenum 是獨立器官**，有自己的 mask，不歸在 small bowel 裡。

---

## 3. Prompt 原文

### 共用的補充解剖知識

三支程式都附上同一段文字，用來處理子結構和同義詞（例如 appendix 要歸到 large bowel）：

```
Supplementary anatomical knowledge:
- The large bowel includes the cecum, colon, rectum, and anal canal. The cecum includes the appendix, so information related to the appendix should also be categorized under the large intestine.
- The small bowel includes the jejunum, and ileum.
- The splenic vein is part of the portal venous system.
- C1 to C7 refer to the cervical vertebrae.
- T1 to T12 refer to the thoracic vertebrae.
- L1 to L5 refer to the lumbar vertebrae.
- Pleural effusion is considered a description relating to the lungs.
```

### Step 1 — 是否提及（`check_organ_mention.py`）

```
Please determine whether the given CT report mentions the organ ({organ}).
Simply answer "yes" or "no". Do not add diagnoses or summaries.

Supplementary anatomical knowledge: ...（同上）

CT report: ({report})
```

- 回答不是 yes/no 時，記錄為 `"未处理成功：非 yes-no 回答"`，不會重試。

### Step 2 — 抽取描述（`report_parsing.py`）

```
From the given CT report, extract the description information related to the specific anatomy ({organ}). Please follow these guidelines:
- Precise extraction: Extract only the description relevant to {organ} directly from the report.
- Focus on the affected site: If the report mentions a specific part of {organ}, make sure it is included in the extracted information.
- Concise and clear: Directly extract content from the report, avoiding unnecessary explanations or background information.
- Formatting requirement: Provide the information in the format "{organ}: description". Ensure that {organ} is used consistently as the prefix for the entry. Even if the organ has multiple distinct parts or bilateral characteristics, treat it as a whole and return only one comprehensive description for {organ}.

Supplementary anatomical knowledge: ...（同上）

CT report:
{report}
```

- 後處理：把全形 `：` 換成 `:` 後切開，並檢查前綴等於 `{organ}`；不符就記為 `"未处理成功..."`。
- 左右側或多部位（例如雙腎）會合併成**一段**描述。

### Step 3 — 正常 / 異常（`report_parsing_normal.py`）

```
From the given CT report, determine whether the specified anatomy ({organ}) is normal or abnormal.
Please answer directly with "normal" or "abnormal". Do not add diagnoses or summaries.

Supplementary anatomical knowledge: ...（同上）

CT report:
{report}
```

- 以 `startswith('normal' / 'abnormal')` 判斷；其他回答記為 `"未处理成功..."`。

---

## 4. 輸出範例（`ckpt/` demo，AC4214dbd）

| organ | 描述（Step 2） | Step 3 |
|---|---|---|
| kidney | Diffuse hypoenhancement of the renal allograft in the right iliac fossa ... | abnormal |
| bladder | Decompressed with circumferential wall thickening ... likely due to cystitis. | abnormal |
| heart | Small pericardial effusion. | abnormal |
| large bowel | Normal appendix is visualized (3/204) near right upper quadrant. Negative for bowel obstruction. | normal |
| liver | Normal. | normal |
| aorta | Not specifically described in the report. | normal |

⚠️ aorta 那一列是 Step 1 誤判「有提及」後，Step 2 回傳的敘述；這種字串會**直接變成訓練 caption**。

---

## 5. 訓練端如何使用

### Caption（`caption_datasets.py` L214–224）

- 有提及的器官：caption 用 Step 2 的描述。
- **未提及的器官：caption 設為 `"normal."`**。
- `organ_abnormal_flags[i, j] = True` ⇔ Step 3 為 `"abnormal"`；其餘都視為正常（包含未提及的器官）。

### Contrastive target 矩陣（`radar_pretrain.py` L362–393）

每個器官各自在 batch 內做 image-text contrastive learning，target 矩陣的組法：

1. 對角線 = 1（同一位病人）
2. normal × normal = 1（兩位病人的該器官都正常 → 視為正樣本）
3. caption 字串完全相同 = 1（例如都是 `"normal."`）
4. abnormal × abnormal += momentum text encoder 的 `softmax(text-text 相似度)`（描述相近的異常病例給予 soft label）
5. 每列正規化成總和為 1

其他規則：
- 只有 global batch 裡**至少有一個完整且異常**的器官才會算 loss（`organ_status_world`，L283–287）；全 batch 都正常的器官會被跳過。
- 被 crop 切到邊界的器官不算 loss（`organ_intact_flags`）。

**所以 Step 3 的 normal / abnormal 判斷直接決定 contrastive label，錯誤會直接影響訓練。**

---

## 6. 換成自己的資料時

要改的地方：
- 輸入 json 格式：每位病人需要 `report`、`impression`、`split` 三個欄位。
- DashScope API key，或把 `dashscope.Generation.call` 改接其他 LLM。
- 如果報告是中文，prompt 和補充解剖知識要一起調整。
- 程式裡的路徑是寫死的（`../../ckpt/merlin_report.json`、`caption_datasets.py` 的 `vis_root`）。

建議加的品質控管：
- 過濾 `"未处理成功..."` 和 `"Not ... described"` 這類回答，把它們改回 `"normal."`，或重新查詢。
- 抽樣人工檢查 Step 3 的 normal / abnormal。
- 可以用公開的 keyword 字典（見附錄）抽樣交叉驗證 Step 1 的 mention 結果。

---

## 7. Step 1 判定標準：寬鬆 vs 嚴格（待人工確認）

Merlin 報告有很多**籠統段落**，沒有點名具體器官，例如 `Vasculature: Patent.`、`Musculoskeletal: Normal.`。這類內容要不要算「有提及」某個器官，會直接影響 caption 和 normal / abnormal 標籤。

- **寬鬆判法**（官方 qwen 的實際傾向）：籠統段落也對應到具體器官。
- **嚴格判法**：報告要**實際寫出該器官**，或寫出補充解剖知識裡列的子結構（例如 appendix → large bowel、L3 → lumbar vertebrae），才判為 yes。

### 比較例（取自 train 前 20 份的試跑）

| # | 報告原文 | 寬鬆判法 | 嚴格判法 | 決定 |
|---|---|---|---|---|
| 1 | `Vasculature: Patent.`（AC4242a2f） | aorta = yes，caption `"Patent."`【官方實際結果】 | aorta = no → `"normal."` | |
| 2 | `Few surgical clips near the right common femoral vessels`（AC4214dbd） | iliac artery = yes，**abnormal**【官方實際結果】 | iliac artery = no（股血管 ≠ 髂動脈） | |
| 3 | `Musculoskeletal: No suspicious osseous lesion present.`（AC4242a2f） | thoracic vertebrae = yes，caption 為該句【官方實際結果】 | 所有骨頭 = no | |
| 4 | `Lower thorax: Normal.`（AC4242a2f） | lung = yes，caption `"Normal."` | lung = no → `"normal."` | |
| 5 | `Musculoskeletal: Degenerative change of the spine.`（AC4240302） | lumbar / thoracic vertebrae 可能 = yes，abnormal | 都 = no（沒寫哪一段脊椎） | |
| 6 | `Gastrointestinal tract: No bowel obstruction.`（AC42426e9） | small / large bowel 可能 = yes | 都 = no（沒寫小腸或大腸） | |
| 7 | `Vasculature: Atherosclerosis without aneurysm.`（AC42426e9） | aorta = yes，abnormal | aorta = no（沒寫主動脈） | |
| 8 | `Left para-aortic lymph node … 2.2 x 1.1 cm`（AC4240302） | aorta = yes，caption 可能變成淋巴結描述 | aorta = no（淋巴結位置詞，不是主動脈本身） | |

1–3 是 `ckpt/` 官方 json 裡實際出現的結果；4–8 是依照同樣傾向推測的寬鬆判法。

**兩種判法都判 yes 的對照例**：
- `Degenerative changes in the thoracolumbar spine most prominent at L2-L3`（AC42434c5）→ thoracic、lumbar vertebrae = yes，abnormal
- `Abdominal aorta is not aneurysmal`（AC4242a55）→ aorta = yes，normal

### 取捨

| | 寬鬆 | 嚴格 |
|---|---|---|
| 提及的器官數 | 多 | 少 |
| caption 品質 | 常出現 `"Patent."`、`"Normal."`、`"Not specifically described"` 這類無資訊量的描述 | 描述都有明確對應的器官 |
| abnormal 標籤 | 可能把異常算到不確定的器官上（#2、#5、#7） | 較準，但可能**漏掉**真正的異常（#5、#7） |
| 與官方一致性 | 較接近 | 有差異 |

### 試跑紀錄

- 檔案：`data/CT/merlin/pilot_merlin_report_mention.json`（train 前 20 份，採嚴格判法，格式同官方 `merlin_report_mention.json`）
- 平均每份報告 10.8 個器官 = yes
- 與官方 `ckpt/` 中重疊的 4 份相比：2 份完全一致；另 2 份官方多判 4 個器官（即上表 #1–#3 加上 AC4214dbd 的 aorta，其 caption 為 `"Not specifically described in the report."`）

> **決定欄待人工填寫**：可以逐條決定，或整體擇一再列例外（例如「整體嚴格，但 #7 算 aorta」）。

### 目前採用：寬鬆判法，以規則實作（2026-09-30）

沒有使用 LLM，改用規則腳本取代官方的 `check_organ_mention.py`：

- Script：`data/CT/merlin/scripts/check_organ_mention_rules.py`
- 輸入：`data/CT/merlin/merlin_report.json`（由 `reports_final.xlsx` 轉出，邏輯同 `ckpt/transform_report_to_json.py`）
- 輸出：`data/CT/merlin/merlin_report_mention.json`（格式同官方，另外多一個 `mention_source` 欄位，記錄每個 yes 是 `explicit` 還是 `lenient:<rule>`）
- `--strict` 會關閉所有 lenient 規則；個別規則可在 `LENIENT_RULES` 裡開關。

```bash
cd data/CT/merlin/scripts
python check_organ_mention_rules.py \
    --input  ../merlin_report.json \
    --output ../merlin_report_mention.json
```

驗證：
- `--strict` 的結果與人工判斷的 20 份**完全一致**。
- lenient 模式涵蓋官方 `ckpt/` 那 4 份的所有 yes。
- 已排除的誤判：`para-/peri-aortic`、`pre-sacral`（位置詞）、`left gastric artery`（血管）、`portal venous phase`（CT 期相，93 份）。

全量結果（train 15,309 份）：平均每份 **14.6** 個器官 = yes（explicit 175,307 個、lenient 47,833 個）。各 lenient 規則觸發次數：

| 規則 | 次數 |
|---|---|
| musculoskeletal_to_vertebrae（#3） | 19,941 |
| bowel_to_small_large（#6） | 10,680 |
| vasculature_to_aorta（#1、#7） | 9,631 |
| lower_thorax_to_lung（#4） | 3,513 |
| hiatal_hernia_to_stomach_esophagus | 2,395 |
| spine_to_vertebrae（#5） | 769 |
| presacral_to_sacrum | 409 |
| whipple_to_duodenum | 319 |
| femoral_to_iliac_vena / artery（#2） | 94 / 64 |
| paraaortic_to_aorta（#8） | 18 |

### 嚴格判法版本（2026-09-30 新增）

同一支 script 加上 `--strict`，輸出到 `data/CT/merlin/merlin_report_mention_strict.json`：

```bash
python check_organ_mention_rules.py --strict \
    --input  ../merlin_report.json \
    --output ../merlin_report_mention_strict.json
```

兩版比較（train 15,309 份，只列差異較大的器官）：

| 器官 | 寬鬆 | 嚴格 |
|---|---|---|
| **平均 yes / 份** | **14.6** | **11.5** |
| aorta | 96.6% | 33.6% |
| large bowel | 99.9% | 83.8% |
| small bowel | 99.2% | 45.5% |
| lung | 99.4% | 76.4% |
| thoracic vertebrae | 90.4% | 13.5% |
| lumbar vertebrae | 91.5% | 33.1% |
| esophagus | 20.1% | 11.6% |
| stomach | 40.3% | 33.1% |
| sacrum | 17.8% | 15.2% |

其他器官兩版相同或差不到 1 個百分點。

⚠️ 從官方 4 份來看，官方 qwen **並沒有**套用 #4（`Lower thorax: Normal.` → lung）和 #6（`No bowel obstruction` → small bowel）。所以這裡的寬鬆判法比官方實際結果**更寬**，要更接近官方的話，可以關掉這兩條規則。

---

## 8. 規則 vs LLM 抽查（2026-09-30）

### 方法

- 從 train 15,309 份中隨機抽 **300 份**（`random.seed(42)`），每份判斷 26 個器官，共 7,800 格。
- LLM 標註：由 Claude subagent 盲測判讀（看不到規則結果），只給**官方 Step 1 prompt 與補充解剖知識**，逐格回答 yes / no。
- 抽查檔案在 `data/CT/merlin/llm_check/`：
  - `llm_report_mention_300.json`：300 份 LLM 標註，格式同官方 `merlin_report_mention.json`
  - `llm_0.json` ~ `llm_5.json`：6 批原始標註（`{pid: [判為 yes 的器官]}`）
  - `sample_pids.json`：抽樣名單
  - `disagreements.json`：規則與 LLM 不一致的案例（依版本、器官、方向分類）
  - `compare.py`：比對程式，`python3 compare.py` 可重跑

### 結果

第一輪比對後，依不一致的案例修正規則，再重新比對：

| | 寬鬆 vs LLM | 嚴格 vs LLM（修正前） | **嚴格 vs LLM（修正後）** |
|---|---|---|---|
| 逐格一致率 | 88.4% | 98.4% | **99.4%** |
| 規則 yes、LLM no | 886 | 38 | **14** |
| 規則 no、LLM yes | 16 | 86 | **33** |
| Precision / Recall（以 LLM 為準） | 79.6% / 99.5% | 98.9% / 97.5% | **99.6% / 99.1%** |
| 26 格完全相同的報告 | 24 / 300 | 194 / 300 | **257 / 300** |

**結論：嚴格判法和 LLM 的判斷非常接近；寬鬆判法多判了約 11% 的格子。** 寬鬆多出來的 886 格幾乎都來自 lenient 規則（musculoskeletal→vertebrae 390、vasculature→aorta 193、bowel→small/large 164、lower thorax→lung 66），代表 LLM 不會這樣推論。

### 依抽查修正的規則（同時影響兩版）

| 問題 | 例子 | 修正 |
|---|---|---|
| 小腸漏判 | `small and/or large bowel` | 中間有 and/or 也比對 |
| 胃漏判 | `hiatal hernia`、`gastroesophageal reflux` | 算 stomach（LLM 不算 esophagus） |
| 大腸漏判 | `colostomy`、`large and small bowel` | 補上 |
| 髂動脈漏判 | `aortoiliac calcification` | 補上 |
| sacrum 誤判 | `S2: ABNORMAL`（Stanford 摘要代碼） | 排除後面接 `:` 的 S1–S5 |
| sacrum / 脊椎漏判 | 句首大寫 `Sacrum`、`Lumbar` | 只有 L3、S1 這類節段編號區分大小寫 |
| aorta 誤判 | `aortic valvular calcification`、`aortocaval node` | 排除 |
| stomach / esophagus 誤判 | `perigastric`、`epigastric`、`paraesophageal` | 排除位置詞 |
| heart 誤判 | `ventriculo-peritoneal shunt` | 排除 |
| duodenum 誤判 | `ampulla` | 移除 |
| sacrum 漏判 | `sacro-iliac` | 補上（`presacral` 維持排除，LLM 大多判 no） |

### 剩下的差異（屬判斷差異，未再修）

- `para-aortic / periaortic / circumaortic`：LLM 常判 aorta = yes，規則視為位置詞排除（但 `aortocaval` LLM 又判 no，LLM 本身不一致）。
- `No pericardial effusion`：LLM 判 heart = no，規則判 yes。
- `presacral`：LLM 大多判 no，少數判 yes。

⚠️ 規則是用**同一批 300 份**調整的，所以 99.4% 會偏樂觀。若要更嚴謹的數字，需要另抽一批新的報告再驗證一次。

### 修正後的全量結果（train 15,309 份）

| | 寬鬆 | 嚴格 |
|---|---|---|
| 平均 yes / 份 | 14.5 | 11.5 |
| aorta | 96.5% | 31.2% |
| large bowel | 99.9% | 85.0% |
| small bowel | 99.2% | 50.3% |
| lung | 99.4% | 76.4% |
| stomach | 39.8% | 39.8% |
| thoracic vertebrae | 90.5% | 13.6% |
| lumbar vertebrae | 91.5% | 33.2% |

（第 7 節的全量數字是修正前的版本。）

---

## 9. Step 2 規則版：抽取器官描述（2026-09-30）

用規則取代官方的 `report_parsing.py`，輸入用 Step 1 嚴格版。

- Script：`data/CT/merlin/scripts/report_parsing_rules.py`（沿用 `check_organ_mention_rules.py` 的器官 regex）
- 輸入：`merlin_report_mention_strict.json`
- 輸出：`merlin_report_organ_report_strict.json`，格式同官方 `merlin_report_organ_report_v1.json`：`{pid: {"report": ..., organ: 描述}}`

```bash
cd data/CT/merlin/scripts
python report_parsing_rules.py \
    --mention ../merlin_report_mention_strict.json \
    --output  ../merlin_report_organ_report_strict.json
```

### 規則

1. 依**段落標題**分段，再切成句子。標題字典從資料中自動學出（出現 ≥ 20 次的 `Header:`，共 71 個），比對時不要求標題前面有句號。有些報告會漏句號，例如 `Adrenal glands: Normal Kidneys and ureters: ...`。
2. 標題屬於哪些器官：直接對標題文字套用 Step 1 的器官 regex，例如 `Kidneys, ureters, and bladder` → kidney + bladder。
3. 一個器官的描述 = **該器官段落內的句子**（例如 `Spleen: Normal.` → `Normal.`）＋ **其他段落或 Impression 中點名到該器官的句子**，依報告順序排列並去除重複。
4. 遇到 Impression 的編號清單（`1.`）時離開目前段落，之後的句子只靠 keyword 歸屬。
5. 只處理 Step 1 = yes 的器官。

### 結果（train 15,309 份）

- 共 176,482 個器官描述；長度中位數 9 個字，p99 134 個字，最長 424 個字（模型 `max_txt_len` = 512 tokens）。
- Step 1 = yes 但抽不到句子：**10 個**，都是報告本身的空段落（例如 `Gallbladder: Spleen: Normal.`），不寫入輸出，訓練時會補成 `"normal."`。
- 修正過的問題：自動學出的標題混入「前段內容＋標題」的假標題（`normal kidneys and ureters`、`surgically absent pancreas`），以及 `for example:` 這類句中冒號。

### 與官方 LLM 輸出對照（`ckpt/` demo：AC4214dbd、AC4240fff）

兩邊都有的器官，描述**幾乎逐字相同**。差異：
- 規則版會多帶 Impression 中的重述句（例如 bladder 會同時有 Findings 和 Impression 各一句）。
- 規則版保留影像編號，例如 `(3/216)`；官方 LLM 有時會刪掉。
- 官方有、規則版沒有的器官（aorta `"Not specifically described"`、iliac artery），都是 Step 1 嚴格版不判 yes 的情況。

---

## 10. Step 2 規則 vs LLM 抽查（2026-09-30）

### 方法

- 從 train 另外抽 **300 份**（`random.seed(2026)`，與第 8 節的樣本只重疊 3 份），共 **3,439 組**「報告 × 器官」（器官清單 = 當時 Step 1 嚴格版判為 yes 的器官）。
- LLM：6 個 Claude subagent 盲測，只給**官方 `report_parsing.py` 的 prompt 與補充解剖知識**。
- 檔案在 `data/CT/merlin/llm_check/step2/`：
  - `in/batch_*.json`：輸入
  - `out/llm_*.json`：LLM 抽取結果（`{pid: {organ: 描述}}`）
  - `compare_step2.py`：比對程式
  - `step2_pairs_sorted_by_f1.json`：每一組的 LLM／規則描述，依 F1 由低到高排序，方便人工檢查
- 指標：文字正規化（小寫、去標點、去影像編號如 `(3/216)`）後，算**完全相同比例**與 **token F1**。recall = LLM 的內容有多少被規則涵蓋；precision = 規則的內容有多少也在 LLM 裡。

### 結果

| | 修正前 | **修正後** |
|---|---|---|
| 正規化後完全相同 | 62.7% | **64.0%** |
| 平均 token F1 | 88.6% | **89.6%** |
| 規則 recall / precision | 94.3% / 87.6% | **95.5% / 88.0%** |
| F1 < 0.5 的組數 | 216 | **188** |
| 規則空、LLM 有 | 0 | 9（Step 1 修正後不再判 yes 的器官） |

各器官 F1：adrenal、bladder、spleen、pancreas、aorta、kidney 都在 92% 以上；small bowel（約 73%）、thoracic vertebrae、iliac vessels、esophagus 最低（約 65–80%）。

### 依抽查修正的規則

**共用 regex（同時影響 Step 1）**——排除位置詞與血管名稱：

| 例子 | 原本被歸到 |
|---|---|
| `splenic flexure`、`splenic artery / vein`、`portal splenic confluence` | spleen |
| `hepatic artery`、`hepaticojejunostomy` | liver、small bowel |
| `pancreatico- / choledocho-jejunostomy`、`ileocolic lymph nodes / vessels` | small bowel |
| `left gastric (artery)` | stomach |
| `peri-rectal nodes` | large bowel |
| `emphysematous cystitis / cholecystitis / pyelonephritis` | lung |
| `portal venous and delayed phases` | portal vein |
| `iliac vessels`（幾乎都是在講動脈硬化） | iliac vena |

Step 1 嚴格版重新和第 8 節的 300 份比對：一致率 99.40% → **99.37%**，幾乎沒有變化。

**Step 2 專用**：
- `Lymph nodes` 段落的句子不透過 keyword 歸給器官；`Vasculature` 段落的句子只透過 keyword 歸給血管類器官。**例外**：該器官在其他地方完全沒有句子時才用這些句子（例如只出現在 Vasculature 的 `esophageal varices`），避免 Step 1 = yes 卻抽不到描述。
- GI／Bowel 段落中**沒點名任何器官**的句子（例如 `No evidence of bowel obstruction`、`GI tract is otherwise unremarkable`），歸給 small bowel 和 large bowel（僅限 Step 1 = yes 的器官）。

修正後全量結果：176,167 個描述，Step 1 = yes 但抽不到句子的有 9 個（都是空段落）。

### 第二輪修正：F1 = 0 的案例（2026-09-30）

第一輪修正後，F1 = 0 的有 12 組，原因分三類：

| 類型 | 組數 | 原因 | 處理 |
|---|---|---|---|
| 規則空、LLM 有 | 9 | 第一輪 regex 修改後 Step 1 不再判 yes：`iliac vessels` 從 iliac vena 移除（4）、`hepatico-/pancreatico-jejunostomy` 從 small bowel 移除（5） | `iliac vessels` 改為只在動脈硬化／鈣化語境下排除（例如 `aorta and iliac vessels`）；jejunostomy 維持排除（Step 1 抽查中 LLM 對同樣寫法曾判 no，證據矛盾） |
| LLM 空、規則有 | 2 | 器官只是位置參考：`heart and lung bases demonstrate coronary calcifications`、`lifting of the aorta from the lumbosacral spine` | 未修（規則無法判斷句子主詞） |
| 完全不重疊 | 1 | **bug**：GI 段落的籠統句 `Normal.` 被當成「已有描述」，導致真正的發現（在 Vasculature／Lymph nodes 段落）沒有被備援收進來 | 籠統句不再算作具體描述；沒有具體句子時，改用「段落限制擋掉的句子＋籠統句」 |

修正後：F1 = 0 由 12 組降到 **8 組**（5 組是 jejunostomy，3 組是位置參考）；Step 1 嚴格版與 LLM 的一致率 99.40%；Step 2 平均 F1 89.6%。

這一輪只影響 6 組（+5 組變好、1 組變差），平均 F1 只增加 0.07 個百分點。真正拉低平均的是下一輪處理的大宗差異。

### 第三輪：處理大宗差異（2026-09-30）

`report_parsing_rules.py` 新增 `OPTS` 選項，用 `llm_check/step2/tune_step2.py` 在 300 份樣本上逐一比較：

| 選項 | 作用 | 結果 | 採用 |
|---|---|---|---|
| `dedup_impression` | IMPRESSION 的句子若有 ≥ 門檻比例的字已出現在該器官的 findings 描述中，就刪除（同時刪除全大寫的重複句） | 門檻 0.3 最佳（0.3–0.5 差異不大） | ✅ 0.3 |
| `own_section_priority` | 器官有自己的段落時，忽略其他段落的 keyword 句子（例如 Liver 段落中的 `gallbladder fossa`、`post-cholecystectomy`）。`"all"` 連 IMPRESSION 也忽略 | `"all"` 最佳 | ✅ `"all"` |
| `msk_generic` | Musculoskeletal 段落中沒點名器官的句子歸給脊椎 | F1 反而下降約 0.06 | ❌ |

另外修正 `Whipple disease`（一種感染症）被誤判成 pancreas（Whipple 手術）。

| 設定 | 平均 F1 |
|---|---|
| 兩個選項都關 | 89.66% |
| 只開 `dedup_impression=0.3` | 90.85% |
| 只開 `own_section_priority="all"` | 90.51% |
| **兩個都開（採用）** | **91.25%** |

採用後的全量結果（train 15,309 份）：

| | 第三輪前 | **第三輪後** |
|---|---|---|
| 正規化後完全相同 | 64.0% | **67.0%** |
| 平均 token F1 | 89.6% | **91.3%** |
| 規則 precision / recall | 88.0% / 95.6% | **92.4% / 93.6%** |
| 規則多抓（precision < 0.8） | 558 組 | **283 組** |
| 規則少抓（recall < 0.8） | 305 組 | 446 組 |
| 描述長度中位數 / p95 | 62 / 496 字元 | 57 / 367 字元 |

取捨：描述變短、重複與無關句子大幅減少，但 `own_section_priority` 也會擋掉其他段落中真正相關的句子（recall 下降）。例如 bladder 自己段落寫 `Normal.`，而 Prostate 段落有 `mass effect on the urinary bladder … bladder outlet obstruction`，LLM 會收、規則不會。

**剩下的差異是規則的極限**：要分辨器官在句子中是「主體」（`bladder outlet obstruction`）還是「位置參考」（`posterior to the iliac vessels`、`heart and lung bases demonstrate coronary calcifications`），需要語意理解，關鍵字規則無法可靠判斷。

⚠️ 選項是用同一批 300 份調的，91.3% 會略偏樂觀（但只調了 2 個參數，過度擬合的程度有限）。

### 剩下的差異（多半是寫法不同，不是內容錯誤）

- **規則版多 Impression 重述**（最大宗，約 550 組 precision < 0.8）：同一個發現在 Findings 和 Impression（有時是全大寫）各出現一次，規則版兩句都收，LLM 通常只收一句。
- **LLM 會多收同段落的籠統句子**：例如 lumbar vertebrae 多帶 `No evidence of fractures.`、`No suspicious osseous lesions.`。
- **LLM 會改寫或刪減**：刪影像編號、合併句子、只取子句。
- 這些差異對 contrastive 訓練的影響應該不大，因為兩邊的核心發現相同；最需要注意的是**規則版描述較長、內容重複**。

---

## 附錄：公開的 keyword → anatomy 字典（交叉驗證用）

RADAR 本身**不使用**規則字典。以下是其他專案中實際存在的字典（2026-09-30 查證），只在想用規則檢查 LLM 結果時參考：

| 專案 | 檔案 | 涵蓋範圍 | 形式 |
|---|---|---|---|
| **SARLE**（Draelos，RAD-ChestCT，MIT）⭐ | [`src/vocab/vocabulary_locations.py`](https://github.com/rachellea/sarle-labeler/blob/master/src/vocab/vocabulary_locations.py) | 胸部 CT 很細＋約 10 個上腹器官 | `{'Any': [substring], 'Exclude': [...]}`，句子層級 keyword 比對 |
| **Duke body-CT RBA** | [`RBA/*/RBA_*_Config.py`](https://github.com/fitushar/multi-label-annotation-text-reports-body-CT) | 肝膽、腎 / 輸尿管、肺 / 肋膜 | regex list |
| **Merlin**（StanfordMIMI） | [`documentation/report_generation_demo.py`](https://github.com/StanfordMIMI/Merlin/blob/main/documentation/report_generation_demo.py) L48–62 | 腹部 CT，13 個 organ system | 比對報告段落標題 |
| **RadGPT / AbdomenAtlas 3.0** | [`evaluate_reports/RadGPT.py`](https://github.com/MrGiovanni/RadGPT/blob/main/evaluate_reports/RadGPT.py) 的 `organ_dict` | 腹部 CT，約 40 個器官 | 器官名稱正規化 |

SARLE 腹部節錄：

```python
'esophagus':    {'Any': ['esophag']},
'stomach':      {'Any': ['stomach', 'gastro', 'gastric']},
'intestine':    {'Any': ['colon', 'intestin', 'duoden', 'jejun', 'ileum']},
'liver':        {'Any': ['liver', 'hepatic', 'caudate', 'quadrate', 'hepatis']},
'gallbladder':  {'Any': ['gallbladder', 'gallstone', ' chole']},
'kidney':       {'Any': ['kidney', ' renal', 'nephr']},   # ' renal' 前面的空白是為了避開 adrenal
'adrenal_gland':{'Any': ['adrenal', 'suprarenal']},
'spleen':       {'Any': ['spleen', 'splenic']},
'pancreas':     {'Any': ['pancrea']},
'aorta':        {'Any': ['aorta', 'aortic', ' arch ']},
'ivc':          {'Any': ['inferior vena cava', 'ivc']},
```

套用到 RADAR 時要注意：SARLE 的 `intestine` 必須拆成 duodenum / small bowel / large bowel，並補上 bladder、portal vein、iliac 血管、各段脊椎等 RADAR 有、SARLE 沒有的器官。
