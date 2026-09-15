# U-VLM Stage 1（2D 版）— Segmentation Pretraining 計畫

參考論文：*U-VLM: Hierarchical Vision Language Modeling for Report Generation* (arXiv 2603.00479)。
原論文 Stage 1 = 用 3D U-Net encoder 在多來源 segmentation 標註上做 pretraining（ReXGroundingCT 病灶、
TotalSegmentator 器官偽標籤、ATM22 氣管、HiPaS 血管，彼此沒有統一標註），之後把 encoder 權重帶到
Stage 2（classification head）與 Stage 3（report generation，multi-layer visual injection 進 LM）。

我們的資料集是 **2D 單切片**（每個 volume/case 只抽 1 張關鍵切片），所以整個 Stage 1 要從 3D 改寫成 2D。
資料抽取已經完成（見第 1 節），這版計畫把三個標註來源怎麼一起訓練的架構定下來，確認沒問題再開始寫程式。

---

## 1. 資料盤點（已全部抽取完成）

### 1.1 CT-RATE + ReXGroundingCT + TotalSegmentator（`data/processed_rex/`，3,042 筆）

> **2026-09-14 更新**：本節與下方原本都是寫「414 筆」，是抓取初期只拿了 CT-RATE validation split 的殘留數字。實際上後來把 train split 也一起抓了，`processed_rex/manifest.jsonl` 現在是 **3,042 筆**（TotalSegmentator 器官 mask 也已補齊全部 3,042 筆，0 筆失敗）。下面沿用舊文字的地方如果還看到「414」，以本節的 3,042 為準；CT-RATE : HiPaS 的比例因此是 **3,042 : 250（約 12:1）**，不是原本設想的 414:250（約 1.6:1），第 5 節的抽樣策略已一併更新。

每個 volume 抽 1 張切片，同一張切片上同時有病灶 mask 跟器官 mask：

- `image_npy`：`(H, W)` float32 原始 HU 值，`512x512` 或 `768x768` 兩種尺寸
- `segmentation_mask_npz`（病灶，來自 ReXGroundingCT）：`mask` shape `(N, H, W)`，**N = 該切片上的 finding 數量（0~6 不等，逐樣本不同）**，每個 channel 對應一個 grounded 病灶句子，值域 0/1/2（重疊 instance 疊加）。`category`（如 `1b`/`2a`）是 ReXGroundingCT 內部句子分類碼，**不是固定跨樣本一致的病灶類別**，不能當固定分割類別用
- `organ_masks/{volume_id}.npy`（器官，來自 TotalSegmentator `total` task, `--fast` + MPS 推論）：`(H, W)` uint8，值是 TotalSegmentator 固定的 117 類 class id（背景=0）。**這是固定 taxonomy**，可以直接當多類別分割的 target 用。實測：414 筆裡出現過 78/117 種結構，最常見的是 spinal_cord/autochthon_left/right（幾乎每張都有）、肺葉、esophagus、aorta、heart、肋骨、sternum 等胸腔中段常見結構
- `classification_labels`：18 種病理的 multi-label 分類（CT-RATE 原生）
- `report_text`：Findings/Impressions 等全文
- 限制：涵蓋 CT-RATE 的 train + validation split（`volume_id` 前綴 `train_` 2,628 筆 + `valid_` 414 筆，ReXGroundingCT 標註的 3,142 個 volume 裡扣掉 100 筆上游 404 缺檔後的實際筆數），Stage 1 的 train/val split 需要自己依 `volume_id`（patient-level）重新切，不能直接沿用 CT-RATE 官方的 train/validation 分法（原本的 valid split 現在只是我們 3,042 筆裡的一部分，不是獨立留出的驗證集）

### 1.2 HiPaS（`data/processed_hipas/`，250 筆，**獨立病人、跟上面 414 筆完全不重疊**）

- 來源：HiPaS Zenodo 公開釋出版本（`zenodo.org/records/14879605`，MIT license，250 案例，可能不是論文的完整 1073 case cohort，但已確認品質沒問題）
- 每個 case 抽 1 張「動脈+靜脈合併面積最大」的切片：
  - `image_npy` / `image_png`：同樣的肺窗 windowing
  - `vessel_mask_npz`：`artery`、`vein` 兩個 `(H, W)` uint8 binary mask
- **沒有**病灶標註、沒有器官標註、沒有分類標籤、沒有報告文字——這批資料**只能拿來訓練血管分割這個子任務**，跟 CT-RATE 那 414 筆是兩個不相交的病人群、不能逐筆對齊
- 像素間距跟 CT-RATE 不完全一致（HiPaS metadata 裡在體 0.55~0.71mm/px，CT-RATE 通常也在類似範圍但個案不同），目前計畫先不做 spacing 校正、直接以像素為單位 resize 到同一輸出解析度，若後續發現血管分割效果不好可以再考慮依 spacing resample

### 1.3 ATM22（氣管樹）—— 仍未取得

需要人工到 [atm22.grand-challenge.org](https://atm22.grand-challenge.org/) 註冊、簽 Data Use Agreement 才能下載，我沒辦法自動取得。目前計畫先不含氣管這個子任務，等你申請下來後再補。

---

## 2. Stage 1 任務定義（2D 版，多來源 multi-head）

**目標不變**：訓練一個共用 encoder，讓它先學會「精細空間結構」，之後在 Stage 2/3 被重複使用（encoder 權重 + multi-scale features）。現在有 3 個標註來源、2 個不相交的病人群，所以改成**一個共用 encoder + 多個獨立分割 head**，每個 head 對應一種標註、只在有該標註的樣本上算 loss（這正是原論文說的「不需要統一標註、多來源各自訓練」在單一 stage 裡的具體做法）：

| Head | 標註來源 | 樣本範圍 | 輸出形式 |
|---|---|---|---|
| **Lesion head** | ReXGroundingCT | CT-RATE 3,042 筆 | 1 channel binary（sigmoid），N 個 finding channel 先 `mask.sum(axis=0) > 0` 合併成單一前景 mask（原因見下） |
| **Organ head** | TotalSegmentator | CT-RATE 3,042 筆 | 118 類（117 器官 + 背景）multi-class（softmax），直接用 TotalSegmentator 的固定 class id 當 target，不用另外設計 taxonomy |
| **Vessel head** | HiPaS | HiPaS 250 筆 | 2 channel binary（sigmoid，artery / vein 各自獨立，因為理論上可能重疊） |

**為什麼 lesion 還是要合併成 binary**：ReXGroundingCT 的 finding channel 是「每個樣本自己的句子列表」，`category` 欄位不是跨樣本一致的病灶類別 id，沒辦法直接組成固定的多類別 lesion head；而 organ 因為 TotalSegmentator 本身就有固定 117 類 taxonomy，所以可以直接做多類別分割，不需要簡化成 binary。

**訓練時的 batch 組成**：CT-RATE : HiPaS 實際筆數是 3,042 : 250（約 12:1），不能直接依比例抽樣（HiPaS 一個 epoch 內會被嚴重稀釋、vessel head 幾乎學不到東西），改用 `WeightedRandomSampler` 讓兩個來源在一個 epoch 內出現次數大致平衡（例如各自貢獻約一半的 batch）。每筆樣本帶一個 `source` 標記（`"ctrate"` 或 `"hipas"`），決定要對哪些 head 算 loss、哪些 head 直接 mask 掉（見第 5 節）。三個 head 共用同一個 encoder 的多尺度特徵 `{f_i}`，各自接一個獨立的輕量 decoder（而不是三個 head 共用一個 decoder 最後分岔），這樣之後 Stage 2/3 只需要拿 encoder，任何一個 head 的 decoder 要拔掉都不影響其他 head。

---

## 3. 資料前處理

1. **Windowing**：CT-RATE 跟 HiPaS 都用同一組肺窗設定（window level -600 / window width 1500，跟 `extract_dataset_rexgroundingct.py` / `extract_dataset_hipas.py` 產生 PNG 用的一致）後 normalize 到 `[0, 1]` 或 `[-1, 1]`
2. **Resize**：統一到固定解析度（建議 `512x512`；CT-RATE 的 768→512 跟 HiPaS 原生 512 都做 resize 對齊；image 用 bilinear，mask 用 nearest）。兩個來源的實際物理像素間距不完全一致（HiPaS ~0.55~0.71mm/px），第一版先不做 spacing 校正，直接以像素為單位對齊，效果不理想再考慮加 resample
3. **Mask 準備**：
   - Lesion：`(N,H,W) → (1,H,W)` binary（同前）
   - Organ：`organ_masks/{volume_id}.npy` 直接當 118 類（含背景）multi-class target，resize 用 nearest
   - Vessel：`vessel_mask_npz` 的 `artery`/`vein` 各自 resize（nearest），組成 `(2,H,W)` binary
4. **Train/Val split**：CT-RATE 3,042 筆用 `volume_id` 做 patient-level split（同一 patient 的多個 scan/recon 不能跨 train/val），HiPaS 250 筆用 `case_id` 另外切一份（兩個來源是不同病人群，split 分開做，不要混在一起切）。都建議 85/15，固定 random seed，兩份 split 都存進 `splits.json`
5. **Augmentation**（HiPaS 250 筆仍偏小、vessel head 過擬合風險高，這步很重要；CT-RATE 3,042 筆風險較低但一起做沒有壞處）：random flip（左右，注意胸腔解剖對稱性）、random rotation（±10°）、random crop/scale jitter、intensity jitter（亮度/對比小幅擾動）。不做上下翻轉（會破壞解剖方向）。CT-RATE 樣本三個 mask（lesion/organ，vessel 缺失）跟 HiPaS 樣本（vessel，lesion/organ 缺失）都要套用同一組隨機參數做 augmentation，確保 image 跟對應 mask 一致變換

---

## 4. 模型架構（2D U-Net，共用 encoder + 3 個 decoder head）

- **共用 Encoder**：原論文用 nnU-Net 風格 channel 數 `[32, 64, 128, 256, 320, 320]`（3D, 6 stage）→ 2D 版沿用相同 channel progression 但改成 2D conv，5~6 個 downsample stage（對應 512→16 的 spatial size）。每個 stage：`Conv2d → InstanceNorm2d → LeakyReLU` ×2（nnU-Net 標準 block）
- Encoder 輸出的多尺度 feature maps `{f_i}`（每個 stage 一組）即為要保留、之後餵給 Stage 3「multi-layer visual injection」的東西 —— 這是 Stage 1 architecture 與後續 stage 對接的關鍵，所以 encoder 的 stage 劃分現在就要定好，之後不能隨便改
- **三個獨立 Decoder head**，各自對稱上採樣 + skip connection 接回同一組 `{f_i}`，只有最後輸出層不同：
  - Lesion decoder → 1×1 conv → 1 channel（sigmoid）
  - Organ decoder → 1×1 conv → 118 channel（softmax，含背景）
  - Vessel decoder → 1×1 conv → 2 channel（各自獨立 sigmoid，artery/vein）
- 三個 decoder 结構一樣（只有輸出層不同），彼此不共享權重，只共享 encoder。forward 一次跑三個 decoder，但训练時哪個 head 的 loss 會被 mask 掉取決於該樣本的 `source`（見第 2、5 節）
- 輸入：1 channel（CT 灰階）

---

## 5. Loss / 訓練設定

- **每個 head 各自的 loss**：
  - Lesion（binary）：`L_lesion = L_dice + L_bce`
  - Organ（118 類）：`L_organ = L_dice(multi-class) + L_ce`（原論文的 Dice+CE，這裡剛好跟論文一致，因為 organ 本來就是多類別）
  - Vessel（2 個 binary channel）：`L_vessel = L_dice + L_bce`（artery/vein 各自算完再平均）
- **Masked 合併**：`L_total = 1[sample 有 lesion+organ 標註] * (L_lesion + L_organ) + 1[sample 有 vessel 標註] * L_vessel`，實作上用第 2 節的 `source` 標記在 batch 內逐樣本 mask（CT-RATE 樣本的 `L_vessel` 項是 0，HiPaS 樣本的 `L_lesion`/`L_organ` 項是 0），每個 loss 項只除以「該 head 實際有標註的樣本數」做平均，避免 batch 裡兩種來源比例不同時互相稀釋
- Optimizer：原論文用 SGD lr=0.01；但考量我們資料量仍小於論文的 2600+ 3D case（3,042+250 張 2D slice），**建議改用 AdamW（lr=1e-3, weight_decay=1e-4）+ cosine schedule**，SGD 在小資料集上收斂較慢也較不穩定
- Batch size：受限於 512×512，建議 8~16，用 `WeightedRandomSampler` 讓 CT-RATE（3,042）: HiPaS（250）在一個 epoch 內出現次數大致平衡（實際筆數比例約 12:1，不能直接照比例抽樣，否則 HiPaS/vessel head 每個 epoch 只會看到極少次）
- Epochs：建議先跑 100～150 epoch + early stopping（同時 monitor 三個 head 各自的 val Dice，任一個 head 都不能只看單一指標就停）
- Encoder 初始化：原論文從零訓練（因為有 2600+ 3D case）；**已確認：改用 ImageNet 預訓練權重初始化**（`segmentation_models_pytorch` 的 ResNet34/50 encoder，或維持 nnU-Net 風格架構但 stem 用預訓練權重），減少 overfitting，接受偏離論文「從零訓練」的設定
- Checkpoint 產出：儲存 encoder state_dict（給 Stage 2/3 用，三個 decoder 都不帶過去）+ 完整三頭模型（供這階段自己 eval 用）

---

## 6. 評估指標

- Val set：每個 head 分開算 —— lesion/vessel 用 Dice + IoU（pixel-level binary），organ 用 per-class Dice 再取 mean（macro）+ 常見結構（肺葉/心臟/主動脈等）的 per-class Dice 另外列出，因為稀有結構（如某幾根肋骨）樣本少、macro mean 容易被拉低但不代表模型真的學不好
- 額外記錄：per-sample Dice 分布（因為樣本少，需要看是否有嚴重 outlier）
- 之後可視化：抽幾張 val 預測疊圖（image + 三個 head 的 GT vs 預測），存到 `U-VLM/stage1/qa/` 風格的 overlay png，方便肉眼檢查，格式可以參考這次對話裡已經驗證過的 lesion/organ/vessel 三張比較圖

---

## 7. 產出物 / 檔案規劃

```
U-VLM/stage1/
├── STAGE1_PLAN.md          (本文件)
├── config.yaml              # 超參數、路徑設定
├── dataset.py                # 讀 processed_rex + processed_hipas 兩份 manifest.jsonl
│                              # -> windowing/resize/mask 準備 -> 統一成帶 source 標記的 Dataset
├── model.py                  # 共用 2D U-Net encoder + 3 個 decoder head（encoder 可獨立 export）
├── losses.py                  # 各 head 的 Dice(+CE/BCE) + masked 合併邏輯
├── train.py                    # 訓練 loop（混合抽樣兩來源）+ checkpoint + 三個 head 的 val 指標
├── eval.py                      # 跑 val set，輸出三個 head 的 Dice 分布 + overlay 圖
└── splits.json                  # {"ctrate": {train:[...], val:[...]}, "hipas": {train:[...], val:[...]}}
```

已完成、Stage 1 訓練時會直接讀取的資料抽取腳本（在 `scripts/`）：
- `extract_dataset_rexgroundingct.py` — CT-RATE 病灶切片 + 分類/報告（414 筆）
- `extract_totalseg_organs.py` — 同一批 volume 補上 TotalSegmentator 器官 mask（414 筆）
- `extract_dataset_hipas.py` — HiPaS 血管切片（250 筆，獨立病人群）

---

## 8. 待確認事項 — 已拍板（2026-09-14）

1. **Organ head 類別數**：✅ 全用 TotalSegmentator 118 類（117 器官 + 背景），不縮減子集。3,042 筆資料量已足夠支撐 multi-class 分割，之後要擴充也不用重新設計 taxonomy。
2. **Vessel head 是否現在就放進 Stage 1**：✅ 三個 head 一起訓練。用 `WeightedRandomSampler` 處理 3,042:250 的資料量差異，不另外拆成兩階段實驗。
3. **Encoder 初始化**：✅ 用 ImageNet 預訓練權重初始化（`segmentation_models_pytorch` ResNet34/50 encoder），偏離論文「從零訓練」設定，以降低 overfitting 風險。
4. **解析度**：✅ 512×512（資料量已從 414 提升到 3,042，訓練穩定度足夠支撐這個解析度）。
5. **HiPaS 的 spacing 差異**：✅ 第一版先不做 spacing 校正，直接以像素為單位 resize 對齊；之後若 vessel head 效果不理想再考慮加 resample。

以上 5 點確認完畢，開始實作 `U-VLM/stage1/` 底下的 `config.yaml` / `dataset.py` / `model.py` / `losses.py` / `train.py` / `eval.py`。

---

## 9. 訓練結果（2026-09-15，遠端 NVIDIA L40S）

環境搬遷、GPU 相容性問題（torch cu130 build 跟 driver CUDA 12.9 不合，改裝 cu126 build 解決）、
batch size 調整（32 在 118 類 organ head 上 OOM，改成 24 才穩定跑）過程見
[REMOTE_SETUP.md](REMOTE_SETUP.md)。

**訓練在 epoch 118 觸發 early stopping**（連續 20 epoch 沒有超過 epoch 98 的最佳分數，總共跑了
119 個 epoch）。Best checkpoint 存在 `checkpoints/best_encoder.pt`（給 Stage 2/3 用）跟
`checkpoints/best_full.pt`（含三個 decoder，供這階段自己 eval 用）。

### 9.1 綜合分數

Best score（epoch 98）= **0.6076**（`(lesion_dice + organ_macro_dice + (artery_dice+vein_dice)/2) / 3`）

### 9.2 各 head 的 val 表現（`eval.py --checkpoint checkpoints/best_full.pt`，445 CT-RATE + 37 HiPaS）

| Head | Mean | Median | Min~Max |
|---|---:|---:|---|
| Lesion（binary） | 0.3255 | 0.2482 | 0.00 ~ 0.95 |
| Vessel artery | 0.8565 | 0.8708 | 0.60 ~ 0.95 |
| Vessel vein | 0.7698 | 0.7953 | 0.57 ~ 0.91 |
| Organ macro（79 類實際出現於 val） | 0.6842 | — | liver/肺葉/aorta 均 >0.92，esophagus 較弱 0.73，heart/trachea 這次 val 抽樣沒出現 |

**Lesion dice 是雙峰分布**（`checkpoints/eval_lesion_dice.csv`，n=445）：
- 38.9%（173 筆）dice < 0.05，幾乎完全沒抓到
- 36.4%（162 筆）dice ≥ 0.5
- 17.5%（78 筆）dice ≥ 0.7

肉眼看 QA overlay（`qa/` 資料夾，20 張）確認出這個雙峰的原因是**病灶大小/型態**，不是隨機雜訊：
- 小型局部病灶（例：`qa/train_10008_a_2_ctrate.png`）預測位置跟大小都抓得不錯
- 大範圍瀰漫性病灶（例：`qa/train_10061_a_2_ctrate.png`，覆蓋大半個肺葉）模型明顯低估範圍，
  只抓到局部碎片，這是目前 lesion head 最大的弱點
- Vessel（HiPaS）分割視覺上幾乎跟 GT 重合（例：`qa/004_hipas.png`），質化上支持 0.86/0.77 的
  量化結果

### 9.3 給 Stage 2 的結論

Organ（118 類 macro 0.68）跟 vessel（0.77~0.86）的表現顯示 **encoder 學到的空間特徵是紮實的**；
lesion head 偏弱看起來是「小病灶可以、大範圍瀰漫病灶低估」的系統性限制，比較像是 decoder 容量、
loss 設計（dice+bce 對大面積前景的梯度行為）或病灶標註本身雜訊（見第 1.1 節 `category` 不是固定
taxonomy 的限制）造成的，不是 encoder 本身沒學好。這個結論直接支持
[STAGE2_PLAN.md](../stage2/STAGE2_PLAN.md) 的 encoder 微調策略決定（見該文件第 8 節）。
