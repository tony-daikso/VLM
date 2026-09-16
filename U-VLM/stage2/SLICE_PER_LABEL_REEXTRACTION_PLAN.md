# Stage 2 切片選取問題與重新抽取計畫

這份文件記錄 Stage 2 訓練/eval 完成後發現的一個資料層級問題，以及重新抽取資料的可行做法，
給要接手實作的人（本機執行）看。背景：Stage 2 已經訓練完成（macro AUROC 0.7410，細節見
[STAGE2_PLAN.md](STAGE2_PLAN.md) 跟 `checkpoints/history.json`/`eval_per_label.csv`），
這份文件處理的是「有幾類 AUROC/AUPRC 偏低，是不是資料本身的天花板問題」这个疑問所延伸出來的
資料改造工作，不影響已經訓練好的 checkpoint。

---

## 1. 問題描述

### 1.1 現象

`eval.py` 跑出來的 per-label 結果（見 `checkpoints/eval_per_label.csv`）裡，幾類 AUPRC 明顯偏低：

| label | n_pos (val) | AUROC | AUPRC |
|---|---:|---:|---:|
| Bronchiectasis | 42 | 0.650 | 0.150 |
| Interlobular septal thickening | 22 | 0.759 | 0.128 |
| Peribronchial thickening | 32 | 0.710 | 0.180 |
| Pulmonary fibrotic sequela | 87 | 0.641 | 0.360 |

人工核對了 `Bronchiectasis` / `Pulmonary fibrotic sequela` 各 2 個假陰性案例（圖 + 完整報告文字，
記錄在 `checkpoints/manual_review_results.csv`），4 例裡 3 例明確驗證是資料選片問題，不是模型學不好。

### 1.2 根本原因（已用實際資料驗證，不是猜測）

抽取腳本 [scripts/extract_dataset_rexgroundingct.py](../../scripts/extract_dataset_rexgroundingct.py)
的 `select_slice()`（第 137-142 行）：

```python
def select_slice(seg):
    # seg: (F, H, W, D)，F = 這個 volume 裡 ReXGroundingCT 標出的所有病灶（finding）數量
    total_area = (seg > 0).any(axis=0).sum(axis=(0, 1))  # 所有病灶「聯集」後每個 z 的面積
    z = int(np.argmax(total_area))  # 取聯集面積最大的那個 z
    present = [i for i in range(seg.shape[0]) if (seg[i, :, :, z] > 0).any()]
    return z, present
```

**每個 volume 只選一張切片，而且是「所有病灶聯集面積最大」的那個 z**——不是針對某個分類標籤單獨找
它自己的最佳切片。這代表：如果一個 volume 同時有一個大範圍病灶（例如瀰漫性肺炎浸潤，面積可以到
10 萬+ pixel）跟一個小範圍局部病灶（例如支氣管擴張，可能只有幾千 pixel），選片邏輯幾乎必然選中
大病灶所在的 z，小病灶所在的 z 完全沒被選到。

用實際案例驗證（4 個 `train_*` volume 的 `findings_on_slice` 欄位）：

| item_id | 分類標籤（陽性） | 這張切片實際是為了哪個 finding 選出來的 | 該 finding 面積 (pixel) |
|---|---|---|---:|
| train_527_b_2 | Bronchiectasis | Diffuse atypical pneumonic infiltrates（另一種病灶） | 114,195 |
| train_11124_a_2 | Bronchiectasis | Calcified soft tissue density...volume loss（另一種病灶） | 30,377 |
| train_6042_a_2 | Pulmonary fibrotic sequela | Mosaic attenuation differences（另一個獨立分類標籤！） | 7,229 |
| train_7346_b_2 | Pulmonary fibrotic sequela | Linear fibroatelectasis（算對上了，但面積很小） | 3,300 |

另外一層獨立問題：`classification_labels`（18 類）來自 CT-RATE 對**整份報告文字**的判讀，
跟 ReXGroundingCT 的 grounding（哪些文字片段被標成有 3D mask 的 finding）是**兩條獨立的標註管線**。
即使重抓資料、把選片邏輯改對，還是會有一部份 volume 對某個分類標籤是陽性，但 ReXGroundingCT
完全沒有為它產生對應的 finding entity——這種情況下沒有「最佳切片」可選，只能維持用現有的
全域切片頂替。

---

## 2. 為什麼現有資料沒辦法直接修

原始 3D CT volume 跟 3D 逐病灶 segmentation mask（`seg`, shape `(F,H,W,D)`）在抽取時只是暫存：
下載下來選完那一張切片後就刪除了（見腳本開頭 docstring 第 22-26 行），現在 `data/processed_rex/
masks/*.npz` 只留了「被選中那個 z」上各病灶的 2D mask，其他 z 的資訊已經不存在。想換一個 z，
**必須重新從 HuggingFace 下載完整的 3D CT + 3D mask**（`REPO_CT = "ibrahimhamamci/CT-RATE"`,
`REPO_REX = "rajpurkarlab/ReXGroundingCT"`），沒有更便宜的替代方案。

---

## 3. 解法：per-label 最佳切片重新抽取

### 3.1 整體流程

1. **重新下載** 3,142 筆（ReXGroundingCT 涵蓋的全部 volume，見腳本 docstring）的原始 CT volume +
   segmentation，這是 Stage 1 資料抽取時已經付過一次的頻寬/時間成本，這次要再付一次
2. **建立「finding 文字 → 18 類分類標籤」的對應表**——`findings_on_slice` 裡的 `category` 欄位
   （`1a`/`1b`/`2a`/`2b`/`2c`...）是 ReXGroundingCT 自己的分類系統，**跟我們的 18 類病理不是同一套**，
   沒辦法直接查表對應。需要用 finding 的 `text` 欄位（自由文字）去對應到 18 類裡的哪一類（或都不屬於）。
   兩種做法：
   - 關鍵字/規則比對（快，但報告用詞變化大，容易漏掉或誤判，例如 "bronchiectatic changes" 要對到
     `Bronchiectasis`、"fibrotic sequela"/"fibroatelectasis" 要對到 `Pulmonary fibrotic sequela"）
   - 用 LLM 對每個 volume 的所有 finding 文字（3,142 個 volume × 平均每個 volume 數個 finding，
     總量可控）做一次分類，每個 finding 文字對應到 18 類裡的 0~多類，比關鍵字規則更穩但要花 LLM API 成本
3. **改寫 `select_slice()`**：現有邏輯是「所有病灶聯集面積最大的 z」；需要改成「給定一個目標分類標籤 X，
   只看第 2 步驟裡對應到 X 的那些 finding 的 mask，取它們面積最大的 z」。同一個 volume 對不同標籤
   會選出不同的 z（甚至同一個標籤在同一個 volume 都可能對應多個 finding，一樣要先聯集再找 argmax）
4. **決定資料/模型架構怎麼跟著改**——這是最關鍵的取捨，因為同一個 volume 平均有 2~3 個分類標籤同時
   陽性，但這樣抽下來每個標籤的最佳切片可能都不一樣：
   - **方案 A（改動小）**：維持「一張圖對 18 維標籤」的現有架構，但對每個 volume 額外挑一張
     「所有已映射到某個分類標籤的 finding 之聯集面積最大」的 z（把原本「所有病灶聯集」限縮成
     「有映射到 18 類裡任一類的病灶聯集」），比現在的版本好一點，但沒有解決「不同標籤搶同一張圖」
     的根本問題，工程改動最小
   - **方案 B（改動大，較徹底）**：每個分類標籤各自建一組樣本——volume 對標籤 X 陽性時，用
     「只對應到 X 的 finding」argmax 出來的切片，該圖的 label vector 裡，X 這一維正常監督
     （target=1），**其他 17 維要標成 ignore（訓練時用 masked BCE，不計入 loss）**，因為那些標籤
     在這張切片上是否可見完全沒有保證。等於把現在的「1 volume → 1 image → 18 維多標籤」，
     改成「1 volume → 最多 18 張 image（每個陽性標籤一張）→ 各自一維有效標籤 + 17 維 ignore」。
     `dataset.py`/`losses.py`（BCE 要支援 per-sample mask）/`train.py`（eval 時 macro AUROC 還是要
     用「原本那張全域切片」算，不能直接混用 per-label 切片，否則同一 volume 的 val 集會被同一個
     volume 的不同切片污染，破壞 held-out 的意義）都要跟著改
   - 兩個方案都要保留 fallback：某個標籤在某個 volume 完全沒有對應 finding 時，就沿用現有的全域
     切片（不強求每個陽性標籤都有專屬切片）

### 3.2 建議先做的範圍

不建議一次對全部 18 類重抓，先只針對第 1.1 節列出的幾個明顯受影響的類別
（`Bronchiectasis`、`Pulmonary fibrotic sequela`、`Interlobular septal thickening`、
`Peribronchial thickening`）做 per-label 重新抽取，驗證這個方法真的能提升這幾類的 AUPRC，
再決定要不要擴大到其他類別——重新下載 3D 資料 + 建 mapping 的成本不小，值得先小範圍驗證。

---

## 4. 待決定事項（本機執行前請先拍板，不要邊做邊猜）

1. **Finding → 18 類 mapping 方法**：關鍵字規則 vs LLM 分類，還是兩者混合（關鍵字規則覆蓋不到的
   再丟給 LLM）？
2. **資料/模型架構**：方案 A（小改動、效果有限）還是方案 B（大改動、較徹底解決問題）？
3. **重抓範圍**：先做第 3.2 節列的 4 類，還是一次做全部 18 類？
4. **儲存位置**：新抽出來的圖/mask 放新目錄（例如 `data/processed_rex_per_label/`）還是覆蓋現有
   `data/processed_rex/`？建議放新目錄，保留現有 `processed_rex/` 當作 baseline，方便重跑
   `eval.py` 比較「per-label 切片 vs 原本 volume 級切片」的效果差異
5. **下載頻寬/時間預算**：重新下載 3,142 筆的 3D CT + 3D mask 預估要多久、多少頻寬，值得先跑一個
   小樣本（例如 50 筆）估算再決定要不要跑全量

---

## 5. 相關檔案

- 抽取腳本：`scripts/extract_dataset_rexgroundingct.py`
- 現有 manifest：`data/processed_rex/manifest.jsonl`（`findings_on_slice` 欄位是驗證這個問題的關鍵證據）
- Stage 2 訓練/eval：`U-VLM/stage2/{dataset,model,losses,train,eval}.py`、`checkpoints/eval_per_label.csv`
- 人工核對紀錄：`U-VLM/stage2/checkpoints/manual_review_results.csv`（目前只有 4 筆，
  `judgment=not_visible_ceiling` 的記錄支撐了第 1.2 節的根因判斷）
- 原始風險說明：[STAGE2_PLAN.md 第 1.2 節](STAGE2_PLAN.md#12-已知風險切片是否能代表-volume-level-標籤)
