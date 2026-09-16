# U-VLM CXR Stage 2.5 — Lesion Detection（PadChest-GR）

跟 [../stage1/STAGE1_PLAN.md](../stage1/STAGE1_PLAN.md)、[../stage2/STAGE2_PLAN.md](../stage2/STAGE2_PLAN.md) 平行、
獨立的一條 track。CT 版沒有對應的 stage，這是 CXR track 特有的新增階段。

**動機**：Stage 1 的 lesion head 只把 box 轉成粗糙的 union binary mask（弱監督分割），沒有還原成
真正的、每個獨立病灶各自的 bounding box + 類別。Stage 2 的分類 head 只輸出「整張影像有沒有某類
finding」，也遺失了空間位置。Stage 2.5 用類似 YOLO/CenterNet 的 anchor-free detection head，
直接在真正的 box 標註上訓練，驗證 encoder 學到的 feature 能不能支撐「每個病灶各自的位置 + 類別」
這種更貼近臨床閱片、也更貼近 Stage 3 report grounding 需求的任務。

---

## 0. 已跟使用者確認的三個核心決策（2026-09-16）

1. **偵測類別**：沿用 Stage 2 的 26 類 `label_group`（不是 finding-level 的 107+ 種細粒度標籤，那個
   長尾太嚴重；也不是 class-agnostic 單一類別）
2. **Encoder 來源**：接在 **Stage 1** 的 `best_encoder.pt` 之後（不是 Stage 2 分類微調後的 encoder），
   理由是偵測任務跟 Stage 1 的「找出病灶位置」在性質上更接近，不希望被 Stage 2 分類任務的微調影響
   空間定位能力
3. **架構複雜度**：單尺度 anchor-free（只在 f3, stride 16, 32×32 feature map 上接一個輕量 CenterNet
   風格 head：heatmap + size + offset 三個 branch），不做多尺度 FPN

---

## 1. 資料盤點與關鍵問題：fine-grained label → label_group 沒有現成映射

`grounded_reports_20240819.json` 每個 finding 的 `labels` 是細粒度字串（`"atelectasis"`、
`"pseudonodule"`、`"chronic changes"`...，全量資料裡有 155 種），`boxes` 是這個 finding 的
0~1 normalized 矩形列表（可能 0、1、多個）。但 Stage 2 用的 26 類 `classification_labels`
是**study 級別**的聚合結果（來自 `master_table.csv` 的 `label_group` 欄位），沒有 per-box 或
per-finding 的 label_group 標記 -- 這是 Stage 2.5 要解決的第一個資料問題。

**解法**：`master_table.csv` 實際上是 per-(study, image, fine-grained label) 一列，每一列同時有
`label` 跟 `label_group` 兩個欄位。實測全部 8,787 列，**每個 fine-grained label 都唯一對應到一個
label_group，沒有一個 label 同時出現在兩個不同 group 底下**，所以這是一個乾淨的多對一映射，
只取決於 label 字串本身，跟哪個 study/split 無關。已用 `scripts/build_label_group_mapping_padchest_gr.py`
從（全量）`data/X-ray/PadChest-GR/master_table.csv.zip` 建出 `label_to_label_group.json`（155 個
fine-grained label → 26 類其中一類），寫進 `data/X-ray/PadChest-GR(-small)/processed/` 底下 --
PadChest-GR-small 本身沒有帶 `master_table.csv.zip`，但這個映射是全域固定的詞彙表，不是
per-study 的東西，所以用全量的 master_table 建一次、兩邊 processed/ 都能用。已驗證 small 子集
manifest 裡出現的所有 993 個 label 都能在這份映射裡查到。

**指定給每個 box 的 class**：同一個 finding 底下可能有多個 `labels`（罕見但存在），每個都映射到
一個 label_group 後可能得到 1 個以上不同的 group -- 這個 finding 底下所有的 box 都標上這整個
class 集合（multi-hot），不是強制挑一個。`extra_boxes`（第二標註者）沿用 Stage 1 的做法，不用
在訓練上，只有 `boxes` 算數。

---

## 2. 任務定義

Anchor-free、single-scale，CenterNet 風格：

- 只用 f3（256 channel, stride 16）；輸入 512×512 → feature map 32×32
- 每個 GT box 的中心點在 26 個 class heatmap channel 裡对应的 channel 上畫一個 2D Gaussian（用
  CenterNet 論文的 `gaussian_radius` 公式決定半徑，`min_overlap=0.7`），同一 finding 對應多個
  class 就在多個 channel 上都畫
- 中心點所在的 feature map pixel 額外回歸兩個值：box 的 (w, h)（feature map pixel 單位）+
  sub-pixel (dx, dy) offset（修正 stride=16 造成的離散化誤差）
- 沒有任何 box 的 study（normal 或沒有可定位病灶的 abnormal study）→ 全 0 heatmap，一樣要進訓練
  提供負樣本監督（跟 Stage 1 的哲學一致）

---

## 3. 資料前處理

跟 Stage 1 完全共用同一套 image pipeline（16-bit PNG per-image 百分位 clip+stretch、512×512、
同一個磁碟 cache 目錄命名規則，兩個 stage 可以互相重用已經算好的圖）。

Box 的 augmentation 處理方式**跟 Stage 1 不同**：Stage 1 是先把所有 box union 成一張 mask 再整張
mask 一起做 augmentation，Stage 2.5 需要保留每個 box 各自獨立的座標（要回歸 w/h/offset，不能只有
一張 union mask）。做法：

1. 每個 box 各自 rasterize 成一張獨立的 512×512 binary mask（不 union）
2. image + 所有獨立 box mask 疊在一起，用同一組隨機參數（hflip/rotation/scale）過同一個
   `affine_grid`/`grid_sample`（image 用 bilinear，box mask 用 nearest，理由跟 Stage 1 一致）
3. 對每個 warp 後的 box mask，取其非零像素的 axis-aligned 外接矩形，反推回增強後的 box 座標
   （這樣就不用手動推導旋轉/縮放對矩形四個角座標的解析變換，直接複用跟 Stage 1 一樣、已經驗證過的
   image-level warp 機制）；box 被旋轉/縮放到完全出界（外接矩形為空）就丟棄這個 box，等同標準
   detection augmentation 的作法
4. 用增強後的 box 座標，在 32×32 feature map 上重新生成 heatmap/size/offset/mask 四個 target

---

## 4. 模型架構

```
image (1, 512, 512)
   │
   ▼
ResNet34Encoder（Stage 1 CXR 權重初始化）──▶ f0..f4
                                              │
                                              ▼ (只用 f3, 256 channel, 32×32)
                              ┌───────────────┴───────────────┐
                              ▼                                ▼
                 heatmap branch (26ch, sigmoid)     size branch (2ch, w/h)
                              │                                │
                              └──────────────┬─────────────────┘
                                              ▼
                                   offset branch (2ch, dx/dy)
```

- 三個 branch 各自是 `Conv3x3(256→256) + ReLU + Conv1x1(256→out_ch)`，沒有額外的 normalization
  layer（跟 CenterNet 原論文 head 設計一致，刻意簡單）
- Encoder 初始化：`U-VLM/X-ray/cxr/stage1/checkpoints/best_encoder.pt`（見第 0 節決策 2）

---

## 5. Loss / 訓練設定

- **Heatmap loss**：CenterNet 的 modified focal loss（`alpha=2, beta=4`），對每張圖的 GT 物件數
  取平均（至少除以 1，避免 0 物件時除以 0）
- **Size loss / Offset loss**：只在 box 中心點所在的 pixel 算 L1，其餘 pixel 不貢獻（用 `reg_mask`
  篩選），除以有效中心點數量
- **總 loss**：`L = L_heatmap + 0.1 * L_size + 1.0 * L_offset`（CenterNet 論文預設權重，沿用不重新調）
- **Encoder fine-tune 策略**：跟 Stage 2 一致的兩階段做法——前 10 epoch 凍結 encoder 只練 head，
  之後解凍、encoder 用比 head 低的 discriminative LR 一起微調
- **Optimizer**：AdamW，head lr=1e-3、encoder lr=1e-4（跟 Stage 2 一致）
- **Batch size**：64 起跑，OOM 就退一級
- **Epochs**：80，early stopping monitor val mAP（見第 6 節）

---

## 6. 評估指標

Detection 沒有現成的「AUROC」概念，用簡化版單一 IoU 門檻的 per-class AP：

1. Heatmap 解碼：3×3 max-pool 找局部極大值（peak）當候選中心點，取分數最高的前 K 個
2. 用該點的 size/offset 值還原成 box（feature map pixel → normalized 0~1 座標）
3. 對每個 class 各自做貪婪 IoU 匹配（IoU ≥ 0.3，考量到 box 標註本身跟 Stage 1 一樣有「粗略性」，
   門檻不用比照一般 detection benchmark 的 0.5）算 precision/recall，AP = PR 曲線下面積
4. Macro mAP（over 26 類）+ 每類的 GT box 數量一起報告（box 數太少的類別 AP 不可靠，跟 Stage 2
   對稀有類別的處理原則一致）
5. QA overlay：抽幾張 val 預測疊圖（image + GT box + 預測 box），肉眼檢查

---

## 7. 產出物 / 檔案規劃

```
U-VLM/X-ray/cxr/stage2_5/
├── STAGE2_5_PLAN.md   (本文件)
├── config.yaml
├── dataset.py           # manifest.jsonl + label_to_label_group.json -> box list -> heatmap/size/offset target
├── model.py              # 沿用 stage1 的 ResNet34Encoder + 新的 DetectionHead（heatmap/size/offset）
├── losses.py              # CenterNet modified focal loss + masked L1
├── target.py               # box -> heatmap/size/offset/mask 編碼、gaussian_radius（獨立成檔，train/eval/dataset 共用）
├── decode.py                # heatmap -> box 解碼（peak 找點 + IoU 貪婪匹配 AP），train/eval 共用
├── train.py                  # 兩階段訓練
├── eval.py                     # per-class AP + QA overlay
└── checkpoints/
    ├── best_encoder.pt   # 微調後的 encoder，給 Stage 3 用
    ├── best_head.pt
    └── history.json
```

---

## 8. 待確認事項

還沒拍板、真的要拿去用之前需要注意：

1. **IoU 門檻（0.3）跟 K（候選數）**：先用經驗值，等第一次訓練結果出來看 val mAP 的絕對數字
   合不合理再調
2. **稀有類別**：26 類裡不少類別 box 數量比 Stage 2 的 study 級別正樣本數更稀疏（一個 study 可能
   有正樣本但沒有可定位的 box），小子集上大概率會重演 Stage 2 的狀況（部分類別 val 完全沒有正樣本、
   AP 算不出來）
3. 目前 `config.yaml` 先指到 `PadChest-GR-small`（跟 Stage 1/2 一致），全量資料解壓完成後只要改
   `processed_dir` 一行
