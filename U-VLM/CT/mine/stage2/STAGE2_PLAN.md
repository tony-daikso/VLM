# U-VLM Stage 2（2D 版）— Classification Head 計畫

參考論文：*U-VLM: Hierarchical Vision Language Modeling for Report Generation* (arXiv 2603.00479)。
原論文 Stage 2 = 把 Stage 1 pretrain 好的 encoder 接上分類 head，在多病理 multi-label 分類任務上
微調，驗證 encoder 學到的空間特徵能不能遷移到「看整張影像判斷有沒有病灶類型」這種更抽象的任務，
之後 encoder（可能連同這階段微調過的權重）會被 Stage 3 的 report generation 拿去做
multi-layer visual injection。

這份文件寫於 Stage 1 訓練跑在背景的同時；**Stage 1 已經訓練完成並跑過 eval**（結果見
[STAGE1_PLAN.md 第 9 節](../stage1/STAGE1_PLAN.md)），第 8 節列的關鍵決策已經依實際結果拍板，
可以開始寫 `U-VLM/stage2/` 底下的程式碼。

---

## 1. 資料盤點

Stage 2 **只能用 `data/processed_rex/`（CT-RATE，3,042 筆）**，原因：

- `classification_labels` 這個欄位（18 種病理的 multi-label 0/1）**只有 CT-RATE 原生就有**，來自
  CT-RATE 官方標註，不是我們自己推的
- `data/processed_hipas/`（250 筆）**沒有分類標籤**（見 HANDOFF.md 第 1 節、STAGE1_PLAN.md 1.2 節：
  HiPaS 只有 vessel mask，沒有 classification/report），Stage 2 用不到這批資料

### 1.1 標籤分佈（已從 `manifest.jsonl` 實際統計，3,042 筆）

| 病理 | 陽性數 | 盛行率 |
|---|---:|---:|
| Lung nodule | 1544 | 50.8% |
| Lung opacity | 1297 | 42.6% |
| Pulmonary fibrotic sequela | 680 | 22.4% |
| Atelectasis | 661 | 21.7% |
| Consolidation | 594 | 19.5% |
| Lymphadenopathy | 561 | 18.4% |
| Emphysema | 478 | 15.7% |
| Arterial wall calcification | 361 | 11.9% |
| Coronary artery wall calcification | 346 | 11.4% |
| Bronchiectasis | 282 | 9.3% |
| Pleural effusion | 244 | 8.0% |
| Peribronchial thickening | 222 | 7.3% |
| Medical material | 215 | 7.1% |
| Interlobular septal thickening | 174 | 5.7% |
| Pericardial effusion | 143 | 4.7% |
| Mosaic attenuation pattern | 127 | 4.2% |
| Cardiomegaly | 127 | 4.2% |
| **Hiatal hernia** | **9** | **0.3%** |

- 全陰性（18 個都是 0）樣本只有 22 筆（0.7%），代表這批資料幾乎每筆都至少有一個陽性標籤，
  是「偏異常」的資料集（跟 CT-RATE 篩選過 ReXGroundingCT 病灶標註的來源有關，不是一般族群盛行率）
- 每筆樣本平均帶 2~3 個陽性標籤（分佈見附錄式統計：1 個標籤 956 筆、2 個 823 筆、3 個 500 筆...）
- **`Hiatal hernia` 只有 9 個陽性樣本**，85/15 切分後 val 可能只有 1~2 筆陽性，這個類別的 val AUROC
  幾乎不可能穩定估計，第 6 節會特別處理

### 1.2 已知風險：切片是否能代表 volume-level 標籤

`classification_labels` 是 **CT-RATE 對整個 3D volume 下的判讀**，但我們的訓練樣本是
`extract_dataset_rexgroundingct.py` 依照 **ReXGroundingCT 病灶 mask 面積最大的那個 z 切片**選出來的
單一 2D 切片（見 `select_slice()`，`selection_method: "rexgroundingct_finding_argmax"`）。

這代表：**選切片的邏輯是為了 lesion head 優化的，不是為了 18 個分類標籤優化的**。例如
`Cardiomegaly`（心臟肥大，通常要看心胸比，需要特定的軸切面）、`Pericardial effusion` 等全域/
非病灶局部的判讀，很可能在「病灶面積最大」那張切片上根本看不出來，即使 volume-level 標籤是陽性。
這是 Stage 2 效果的天花板風險，不是 bug，之後 eval 時如果看到某幾類 AUROC 特別差，先檢查是不是
這個原因，不用急著懷疑模型或程式碼寫錯。

---

## 2. 任務定義

**目標**：把 Stage 1 訓練好的 encoder（`best_encoder.pt`，ResNet34, 5 個 multi-scale feature map，
channel `[64,64,128,256,512]`，stride `[2,4,8,16,32]`，定義見 `U-VLM/stage1/model.py`）接上一個
輕量分類 head，在 18 個病理標籤上做 multi-label 分類微調。

| 項目 | 設定 |
|---|---|
| 輸入 | 跟 Stage 1 一樣：512×512 單張切片，肺窗 windowing（window level -600 / width 1500） |
| 輸出 | 18 維 logits，每維獨立 sigmoid（multi-label，不是 softmax，因為同一筆樣本可以同時有多個陽性標籤） |
| 樣本範圍 | `data/processed_rex/` 全部 3,042 筆（HiPaS 不適用，見第 1 節） |
| Encoder 來源 | `U-VLM/stage1/checkpoints/best_encoder.pt`（`ResNet34Encoder.state_dict()`） |

---

## 3. 資料前處理

跟 Stage 1 完全共用同一套邏輯，不用重寫：

1. **Windowing / normalize**：沿用 `dataset.py` 裡對 CT-RATE 那套（window level -600 / width 1500）
2. **Resize**：512×512（跟 Stage 1 一致，這樣可以直接吃 Stage 1 encoder 權重，不需要重新適應輸入解析度）
3. **Label 準備**：直接讀 `manifest.jsonl` 的 `classification_labels` dict，依固定順序（跟上表一致）
   組成 18 維 `float32` 0/1 向量
4. **Train/val split**：**沿用 Stage 1 的 `splits.json` 裡 `ctrate` 那份 split，不重新切**——原因：
   - Stage 1 的 val set 對 encoder 來說是「沒看過的資料」，如果 Stage 2 重新切一份不同的
     train/val，會讓 Stage 1 的 val 樣本混進 Stage 2 的 train，等於讓 encoder 間接看過這些樣本，
     之後 Stage 2/3 的 val 分數會失去「held-out」的意義
   - 保持同一份 split 也方便之後比較「encoder 在 Stage 1 segmentation 上表現好的 volume，
     Stage 2 分類是不是也表現好」這類分析
5. **Augmentation**：沿用 Stage 1 那套（hflip、±10° 旋轉、scale jitter、intensity jitter），
   不做上下翻轉（解剖方向）

---

## 4. 模型架構

```
image (1,512,512)
   │
   ▼
ResNet34Encoder（Stage 1 權重初始化）──▶ f0,f1,f2,f3,f4
                                              │
                                              ▼ (只用最深層 f4，512 channel, 16×16)
                                    GlobalAveragePool2d
                                              │
                                              ▼
                                    Linear(512 → 256) → LeakyReLU → Dropout(0.3)
                                              │
                                              ▼
                                    Linear(256 → 18)  ──▶ 18 個 sigmoid logits
```

- 只用 `f4`（最深層、語義最抽象的 feature map）做分類，不像 Stage 1 三個 decoder 那樣用到全部
  5 層 skip connection——分類任務要的是「有沒有」而不是「哪個位置」，全域語義特徵通常比高解析度
  細節更有用，這是分類 head 的標準做法（跟 ResNet 原始的 classification head 一致）
- Head 本身很小（2 層 FC + dropout），故意設計得輕量，不讓 head 自己就能記住訓練集，逼 encoder
  的 feature 要真的有用
- Encoder 架構、`out_channels`、stage 劃分維持跟 Stage 1 完全一致，才能直接載入
  `best_encoder.pt` 的 state_dict

---

## 5. 訓練設定

- **Loss**：`BCEWithLogitsLoss`，每個類別各自的 `pos_weight = min(neg_count / pos_count, 20)`
  （用 train split 統計出來的類別數算，並且封頂在 20 倍，避免 `Hiatal hernia` 這種極端稀有類別把
  pos_weight 推到 300+ 倍，導致訓練不穩定或該類別的 loss 主導整個 total loss）
- **Optimizer**：AdamW，encoder 跟 head 用不同 learning rate（discriminative LR）：
  - Head：`lr=1e-3`（隨機初始化，需要學快一點）
  - Encoder：`lr=1e-4`（已經有 Stage 1 pretrain 權重，微調用小 lr 避免破壞學到的特徵）
- **Encoder fine-tune 策略**（✅ 已拍板，見第 8 節第 1 點）：兩階段——前 10 epoch 凍結 encoder
  只訓練 head（linear probe，讓 head 先穩定），之後解凍 encoder 一起用 discriminative LR 微調
- **Batch size**：跟 Stage 1 一樣不用 `WeightedRandomSampler`（Stage 2 只有單一資料來源 CT-RATE，
  不需要跨來源平衡），可以用比 Stage 1 segmentation 更大的 batch size（分類 head 記憶體用量遠小於
  118 類 segmentation 的 organ head），估計可以到 64~96（實際上限等 Stage 2 開始寫程式時再用這台
  L40S 實測）
- **Epochs**：建議 50~80（分類收斂通常比多 head 分割快），early stopping monitor macro AUROC

---

## 6. 評估指標

- **主指標**：18 類各自的 val AUROC + AUPRC，取 macro mean 當綜合分數（比 accuracy 更適合
  multi-label 且類別不平衡的情境）
- **每類單獨列出**，不要只看 macro mean——尤其 `Hiatal hernia`（9 個陽性）、`Cardiomegaly`、
  `Mosaic attenuation pattern`（都在 4% 左右）這幾類 val 陽性樣本可能個位數，AUROC 估計值
  會很不穩定，報告時要附上 val 陽性樣本數，不能直接把不穩定的數字當作模型好壞的證據
- **額外記錄**：per-label threshold 掃描（Youden's J 或 F1-optimal），但正式 threshold 選定
  等看過 val 分佈後再決定，不在訓練時硬編碼 0.5
- 呼應第 1.2 節的已知風險：如果某幾類 AUROC 明顯低於其他類且低於合理猜測，先去 eval.py 裡把該類
  false negative 的切片跟報告文字拉出來人工看幾張，確認是不是「切片本身就看不出來」而不是模型的問題

---

## 7. 產出物 / 檔案規劃

```
U-VLM/stage2/
├── STAGE2_PLAN.md          (本文件)
├── config.yaml              # 超參數、路徑設定（沿用 stage1 splits.json 路徑）
├── dataset.py                # 讀 processed_rex manifest.jsonl -> image + 18 維 label 向量
├── model.py                  # 沿用 stage1 的 ResNet34Encoder + 新的 ClassificationHead
├── losses.py                  # BCEWithLogitsLoss + per-class pos_weight（封頂版本）
├── train.py                    # 兩階段訓練（凍結 head-only -> 解凍 + discriminative LR）
├── eval.py                      # 18 類 AUROC/AUPRC + per-label 陽性數 + 疑似切片不匹配的 case dump
└── checkpoints/
    ├── best_encoder.pt        # 微調後的 encoder，給 Stage 3 用
    ├── best_head.pt            # 分類 head，Stage 3 通常用不到但留著方便自己 eval
    └── history.json
```

`U-VLM/stage1/checkpoints/best_encoder.pt` 是這階段的輸入，訓練開始前要先確認 Stage 1 訓練
已經跑完、`best_encoder.pt` 檔案存在。

---

## 8. 待確認事項 — 已拍板（2026-09-15，依 Stage 1 訓練結果）

Stage 1 已跑完 119 epoch 並完成 eval（完整結果見
[STAGE1_PLAN.md 第 9 節](../stage1/STAGE1_PLAN.md)），下面 5 點依實際結果拍板：

1. **Encoder fine-tune 策略**：✅ **兩階段**——前 10 epoch 凍結 encoder 只訓練 head（linear probe），
   之後解凍全部、用 discriminative LR（head `1e-3` / encoder `1e-4`）一起微調。理由：Stage 1 的
   organ head（118 類 macro dice 0.68）跟 vessel head（artery/vein 0.86/0.77）都表現紮實，QA
   overlay 肉眼確認 vessel 預測幾乎跟 GT 重合，代表 encoder 本身學到的空間特徵是可靠的；lesion head
   偏弱（mean dice 0.33）且呈雙峰分布——小型局部病灶抓得準、大範圍瀰漫病灶低估——這個模式比較像是
   lesion decoder 容量或 loss 設計的限制，不是 encoder 沒學好的訊號。既然 encoder 品質沒問題，
   不需要用更保守的「全程凍結 linear probe」，兩階段微調可以讓 head 先在穩定特徵上收斂、再讓
   encoder 針對分類任務做小幅適應，風險可控
2. **`pos_weight` 上限 20 倍**：✅ 維持這個保守值當起始設定，第一次訓練跑完後看 loss 曲線
   （尤其 `Hiatal hernia` 那一維的 loss 是否震盪或發散）再決定要不要調整，不因為缺乏實驗依據就
   延後開始寫程式——這本來就是需要邊跑邊看的超參數，不影響其他部分的實作
3. **`Hiatal hernia`（9 個陽性）維持在 18 類裡**：✅ 保留，理由不變（跟 CT-RATE 原生 18 類一致，
   之後方便跟其他工作比較），但 `eval.py` 輸出每類指標時必須附上 val 陽性樣本數，明確標註這一類
   的 AUROC 統計上不可靠，避免之後誤讀
4. **分類 head 只用 `f4`**：✅ 先用最深層單層 GAP + FC 的簡單版本（第 4 節架構圖），不做多層
   concat。理由：Stage 1 的 organ/vessel 結果顯示 encoder 各層特徵已經學得不錯，分類任務不需要
   一開始就上複雜度，先建立簡單 baseline，如果 macro AUROC 明顯不理想再考慮加多層 pooled feature
5. **Batch size**：✅ 起跑值訂為 **64**，訓練腳本第一次跑的時候直接拿這台 L40S 實測 forward+backward
   是否 OOM——分類 head（GAP+2層FC）跟 Stage 1 118 類 segmentation decoder 的記憶體量級差非常多
   （沒有 per-pixel 118 通道的輸出張量），64 大機率跑得動，但比照 Stage 1 batch 32 在 organ head
   上 OOM 的教訓，寫 `train.py` 時要保留「OOM 就退一級」的手動調整空間，不寫死假設一定成功

以上 5 點都已拍板，可以開始實作 `U-VLM/stage2/` 底下的 `config.yaml` / `dataset.py` / `model.py` /
`losses.py` / `train.py` / `eval.py`。
