# U-VLM CXR Stage 1 — Lesion Localization Pretraining（PadChest-GR）

參考論文：*U-VLM: Hierarchical Vision Language Modeling for Report Generation* (arXiv 2603.00479)。

這是跟 [U-VLM/CT/stage1](../../../CT/stage1/STAGE1_PLAN.md)（胸部 CT，CT-RATE+ReXGroundingCT+TotalSegmentator+HiPaS）**完全獨立**的一條 track，
用 **PadChest-GR**（胸部 X 光，2D 原生）驗證同一套 U-VLM 三階段架構在不同模態上是否也適用。不跟 CT 版共用
encoder、不混訓練、不共用 checkpoint。

---

## 1. 資料盤點

`data/X-ray/PadChest-GR/processed/manifest.jsonl`（由 `scripts/extract_dataset_padchest_gr.py` 產生）：

- 4,555 筆 study，1 study = 1 張正面胸部 X 光（PA/AP/AP-horizontal，PadChest-GR 只收正面片，沒有 lateral）
- 官方 patient-safe split（直接沿用，**不重新切**）：train 3,185 / validation 455 / test 915
- 每筆 study 的 `findings` 列表：每個 finding 有 `sentence_en/es`、`abnormal`、`boxes`（0~1 normalized `[x1,y1,x2,y2]`，可能 0 個、1 個或多個，例如雙側病灶）、`extra_boxes`（第二位標註者）、`labels`、`locations`、`progression`
- **`abnormal: false` 的句子沒有 boxes/labels/locations 這幾個 key**（extraction 已用 `.get()` 補空值，dataset.py 讀 manifest 時不用再防禦，manifest 裡每個 finding 的 8 個欄位保證都存在）
- `classification_labels`：26 類 `label_group`（`Normal` 1,456、`Other Entities` 1,956，其餘 24 類為實際病理，如 cardiomegaly 500、nodule 433、pleural effusion 426...），留給 Stage 2 用
- **沒有器官/肺野/心臟的 pixel mask**，只有 finding 的 box —— 這是跟 CT 版 Stage 1（TotalSegmentator 器官 + HiPaS 血管 + ReXGroundingCT 病灶三個 head）最大的差異，已跟使用者確認：**CXR 版 Stage 1 只做 lesion 一個 head**，不額外找 CXR 器官分割資料集

---

## 2. 任務定義

**單一 lesion head**，binary 前景分割（弱監督，見下）：

- Target mask：把一個 study 所有 finding 的 `boxes`（0~1 normalized rectangle）rasterize 成跟輸出解析度一樣大的 binary mask 後逐個 finding union 起來；沒有任何 box 的 study（多數是 `abnormal_study=False` 的 normal 案例，但也可能是 abnormal 但沒有可定位 box 的句子，例如 `costophrenic angle blunting` 這類描述性但沒給 box 的 finding）→ all-zero mask，一樣要進訓練提供負樣本監督
- **明確標註這是弱監督，不是真正的病灶分割**：矩形 box 轉出來的 mask 只是「finding 大概在哪個矩形範圍」，不是病灶輪廓，精細度遠不如 CT 版 ReXGroundingCT 的 pixel-level mask。這個 head 的 Dice 分數上限本來就比 CT 版低，**不能拿來跟 CT 版 lesion head 的 Dice 直接比較**，eval 時要在文件/報告裡註明這個差異
- 只有一個 decoder head（不是 CT 版的三個），forward 一次跑一個 decoder

---

## 3. 資料前處理

1. **Normalize**：**已跑過 extraction 確認，PadChest-GR 的 PNG 其實是 16-bit（PIL mode `I;16`，數值範圍約 0~65519，不是 8-bit 0~255）**，直接用 `PIL.Image.convert("L")` 會整張畫面被 clip 成幾乎全白（試過，錯的），不能假設它是 display-ready 的 8-bit 影像。正確做法：讀成 `uint16` array 後做 **per-image 0.5~99.5 百分位 clip + min-max stretch** 到 `[0,1]`（`scripts/make_qa_overlays_padchest_gr.py` 已經用這個方法驗證過，overlay 看起來是正常對比度的胸部 X 光）。這不是 CT 的固定 HU window，是 CXR 常見的資料驅動式對比度正規化，每張圖各自算自己的百分位
2. **Resize**：512×512（跟 CT 版一致，方便之後比較兩條 track 的 encoder；PadChest-GR 原生解析度因為是掃描膠片數位化的舊資料，本身解析度不一致，resize 是必要的）。Image 用 bilinear，mask 用 nearest
3. **Box → mask 的順序**：先在 0~1 normalized 座標系統上把每個 finding 的 box rasterize 成 512×512 mask（box 座標乘上 512 再畫矩形），union 起來，再對 image+mask 一起做 augmentation——這樣 box 不用額外做座標仿射轉換，跟著 mask 一起被影像級 augmentation 處理掉即可
4. **Train/val split**：manifest 裡的官方 `split` 欄位，直接用（`train`/`validation`），`test` 保留給最終 benchmark，Stage 1/2/3 訓練期間都不能碰
5. **Augmentation**：random horizontal flip、random rotation（±10°）、scale jitter、intensity jitter。**不做垂直翻轉**（會破壞解剖上下方向，跟 CT 版規則一致）

---

## 4. 模型架構

沿用 CT Stage 1 的架構家族（同一套 2D U-Net 設計，方便之後比較兩條 track 的 encoder 學到的東西是否類似），但只有一個 head：

```
image (1, 512, 512)
   │
   ▼
ResNet34Encoder（ImageNet 預訓練初始化，segmentation_models_pytorch）──▶ f0..f4
   │
   ▼
Lesion decoder（對稱上採樣 + skip connection）
   │
   ▼
1×1 conv → 1 channel（sigmoid）
```

- Encoder 架構、`out_channels`、stage 劃分建議跟 CT 版 `U-VLM/stage1/model.py` 的 `ResNet34Encoder` 完全一致（channel `[64,64,128,256,512]`，stride `[2,4,8,16,32]`），方便日後如果想做「CT encoder vs CXR encoder 在同一種下游任務上表現如何」這類比較實驗，不需要重新對齊架構
- Encoder 初始化：ImageNet 預訓練權重（跟 CT 版 Stage 1 拍板後的結論一致——CT 版一開始論文設定是從零訓練，後來因為資料量比論文小而改用 ImageNet 初始化，PadChest-GR 4,555 筆一樣屬於偏小資料量，直接沿用這個結論，不用重新驗證）

---

## 5. Loss / 訓練設定

- Loss：`L_lesion = L_dice + L_bce`（跟 CT 版 lesion head 一致）
- Optimizer：AdamW（lr=1e-3, weight_decay=1e-4）+ cosine schedule（沿用 CT 版 Stage 1 選擇）
- Batch size：單 head、512×512，記憶體用量遠小於 CT 版 118 類 organ head，估計可以到 32~64，實際上限等寫 train.py 時在目標 GPU 上實測
- Epochs：建議先跑 100~150 + early stopping（monitor val Dice）
- Checkpoint 產出：`best_encoder.pt`（給 Stage 2/3 用）+ `best_full.pt`（含 decoder，供自己 eval 用）

---

## 6. 評估指標

- Val set（官方 `validation` split，455 筆）：Dice + IoU（pixel-level binary）
- 額外記錄 per-sample Dice 分布（跟 CT 版一樣要看是否有雙峰/outlier）
- QA overlay：抽幾張 val 預測疊圖（image + GT box-mask vs 預測），存到 `qa/` 資料夾，方便肉眼檢查（`scripts/make_qa_overlays_padchest_gr.py` 已經有畫 GT box 的版本，訓練完後可以擴充成同時畫預測）

---

## 7. 產出物 / 檔案規劃

```
U-VLM/cxr/stage1/
├── STAGE1_PLAN.md   (本文件)
├── config.yaml       # 超參數、路徑設定
├── dataset.py         # 讀 data/X-ray/PadChest-GR/processed/manifest.jsonl -> box-derived mask -> Dataset
├── model.py            # ResNet34Encoder（跟 CT 版同架構）+ 單一 lesion decoder
├── losses.py            # Dice + BCE
├── train.py              # 訓練 loop + checkpoint + val Dice
├── eval.py                # 跑 val/test set，輸出 Dice 分布 + overlay 圖
└── checkpoints/
```

**這份文件只到規劃，`dataset.py`/`model.py`/`losses.py`/`train.py`/`eval.py` 尚未實作**——跟 CT 版當初的節奏一致，先確認架構沒問題才開始寫程式碼。

---

## 8. 待確認事項

已跟使用者確認（2026-09-16）：
1. **Stage 1 範疇**：✅ 只做 lesion head（box→粗略 mask），不額外找 CXR 器官/肺野分割資料集
2. **Resolution / encoder 架構**：✅ 沿用 CT 版設定（512×512、ResNet34 + ImageNet 初始化）

已用小樣本（20 筆）實測確認：
3. **原生解析度、灰階範圍**：✅ 16-bit PNG（`I;16`），解析度不一致（實測範圍約 1652×1824 到 3396×3100），確認要 resize，且 normalize 必須用 per-image 百分位 stretch（見第 3 節），不能當成 8-bit 直接除 255

還沒拍板、開始寫 `train.py` 前需要確認：
1. Batch size 上限（等在目標 GPU 上實測）
2. Box→mask 的 rasterize 細節：多個 box 重疊時是否要做特殊處理（目前計畫是直接 union，重疊區域就是 1，跟 CT 版病灶 mask 疊加的處理方式一致，應該不需要特殊 case）
