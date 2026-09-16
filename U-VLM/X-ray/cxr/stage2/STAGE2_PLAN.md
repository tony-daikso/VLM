# U-VLM CXR Stage 2 — Classification Head（PadChest-GR）

參考論文：*U-VLM: Hierarchical Vision Language Modeling for Report Generation* (arXiv 2603.00479)。
跟 [U-VLM/CT/stage2](../../../CT/stage2/STAGE2_PLAN.md)（CT 版）平行、獨立的 track，詳見
[../stage1/STAGE1_PLAN.md](../stage1/STAGE1_PLAN.md) 的整體說明。

**目標**：把 CXR Stage 1 訓練好的 encoder 接上分類 head，在 `label_group` 多標籤分類任務上微調，
驗證 encoder 學到的空間特徵能不能遷移到「看整張影像判斷有沒有某類 finding」這種更抽象的任務。

---

## 1. 資料盤點

`data/X-ray/PadChest-GR/processed/manifest.jsonl` 的 `classification_labels` 欄位：

| label_group | 陽性數 |
|---|---:|
| Other Entities | 1,956 |
| Normal | 1,456 |
| aortic elongation | 558 |
| cardiomegaly | 500 |
| nodule | 433 |
| pleural effusion | 426 |
| scoliosis | 408 |
| vertebral degenerative changes | 313 |
| hyperinflated lung | 279 |
| vascular hilar enlargement | 273 |
| atelectasis | 263 |
| aortic atheromatosis | 243 |
| pleural thickening | 242 |
| interstitial pattern | 220 |
| alveolar pattern | 205 |
| electrical device | 176 |
| hemidiaphragm elevation | 165 |
| fracture | 157 |
| hypoexpansion | 103 |
| central venous catheter | 93 |
| hiatal hernia | 86 |
| endotracheal tube | 65 |
| NSG tube | 52 |
| bronchiectasis | 52 |
| goiter | 41 |
| osteopenia | 22 |

（共 26 類，精確 key 順序見 `data/X-ray/PadChest-GR/processed/label_groups.json`，跑過
`extract_dataset_padchest_gr.py` 後才會產生）

- `Normal` 跟 `Other Entities` 都當作普通的一個 multi-label 類別，不特殊處理（跟 CT-RATE 18 類分類的作法一致，
  `Normal` 是「這個 study 沒有任何 finding」的標記、`Other Entities` 是次要/罕見 finding 的 catch-all，
  兩者都直接放進 26 維向量裡）
- `osteopenia`（22）、`goiter`（41）、`NSG tube`（52）、`bronchiectasis`（52）這幾類陽性數很少，
  85/15 或官方 split 切完後 val/test 可能只有個位數陽性，AUROC 估計值會很不穩定，跟 CT 版 `Hiatal hernia`
  的處理方式一致：eval 時要附上 val/test 陽性樣本數，不能只看 AUROC 數字判斷模型好壞
- 官方 split 直接沿用 Stage 1 的（train 3,185 / validation 455 / test 915），不重新切——理由跟 CT 版
  Stage 2 沿用 Stage 1 split 的理由一致：Stage 1 的 val/test 對 encoder 是 held-out 資料，Stage 2 重新切會
  讓這個「沒看過」的性質失效

---

## 2. 任務定義

| 項目 | 設定 |
|---|---|
| 輸入 | 跟 Stage 1 一樣：512×512 單張 X 光，16-bit 原圖 per-image 百分位 clip+stretch normalize 到 `[0,1]`（見 [STAGE1_PLAN.md 第 3 節](../stage1/STAGE1_PLAN.md)，不是固定 HU window） |
| 輸出 | 26 維 logits，每維獨立 sigmoid（multi-label） |
| 樣本範圍 | `data/X-ray/PadChest-GR/processed/` 全部 4,555 筆 |
| Encoder 來源 | `U-VLM/cxr/stage1/checkpoints/best_encoder.pt` |

---

## 3. 模型架構

跟 CT 版 Stage 2 完全一致的設計（只用最深層 `f4` feature，GAP + 2 層 FC），確保能直接沿用同一套
`ClassificationHead` 實作，不需要重新設計：

```
image (1,512,512)
   │
   ▼
ResNet34Encoder（Stage 1 CXR 權重初始化）──▶ f0,f1,f2,f3,f4
                                              │
                                              ▼ (只用 f4, 512 channel, 16×16)
                                    GlobalAveragePool2d
                                              │
                                              ▼
                                    Linear(512 → 256) → LeakyReLU → Dropout(0.3)
                                              │
                                              ▼
                                    Linear(256 → 26)  ──▶ 26 個 sigmoid logits
```

---

## 4. 訓練設定

- **Loss**：`BCEWithLogitsLoss`，每類 `pos_weight = min(neg_count / pos_count, 20)`（train split 統計，
  封頂倍數沿用 CT 版起始值）
- **Optimizer**：AdamW，discriminative LR——Head `lr=1e-3`，Encoder `lr=1e-4`
- **Encoder fine-tune 策略**：兩階段——前 10 epoch 凍結 encoder 只訓練 head，之後解凍全部一起微調（跟 CT
  版 Stage 2 拍板的策略一致；CXR Stage 1 訓練完後如果 lesion head 的 Dice 明顯偏弱，可以重新考慮要不要延長
  linear-probe 階段，屆時再依實際結果調整，這裡先沿用 CT 版已驗證的預設）
- **Batch size**：起跑值 64，第一次跑 train.py 時實測 OOM 就退一級（同 CT 版 Stage 2 的教訓）
- **Epochs**：建議 50~80，early stopping monitor macro AUROC

---

## 5. 評估指標

- 26 類各自的 val/test AUROC + AUPRC，macro mean 當綜合分數
- 每類單獨列出，附上 val 陽性樣本數（尤其上面列的幾個稀有類別）
- Per-label threshold 掃描（Youden's J 或 F1-optimal），不在訓練時硬編碼 0.5

---

## 6. 產出物 / 檔案規劃

```
U-VLM/cxr/stage2/
├── STAGE2_PLAN.md   (本文件)
├── config.yaml       # 沿用 stage1 路徑設定
├── dataset.py         # 讀 manifest.jsonl -> image + 26 維 label 向量
├── model.py            # 沿用 stage1 的 ResNet34Encoder + 新的 ClassificationHead
├── losses.py            # BCEWithLogitsLoss + per-class pos_weight
├── train.py              # 兩階段訓練
├── eval.py                # 26 類 AUROC/AUPRC + per-label 陽性數
└── checkpoints/
    ├── best_encoder.pt   # 微調後的 encoder，給 Stage 3 用
    ├── best_head.pt
    └── history.json
```

**這份文件只到規劃，程式碼尚未實作**，需要等 CXR Stage 1 訓練完成、`best_encoder.pt` 確定存在之後才開始寫。

---

## 7. 待確認事項

還沒拍板、開始寫 `train.py` 前需要確認（等 Stage 1 訓練結果出來再依實際情況決定，跟 CT 版當初的節奏一致）：
1. Encoder fine-tune 策略是否需要調整（取決於 Stage 1 lesion head 的表現）
2. `pos_weight` 封頂倍數是否需要調整（取決於第一次訓練的 loss 曲線，尤其 `osteopenia`/`goiter` 這幾個極稀有類別）
3. Batch size 實際上限（等在目標 GPU 上實測）
