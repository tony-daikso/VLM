# 任務：以 TotalSegmentator encoder 進行 per-organ multi-window 快速驗證

## 0. 給 Claude Code 的總體說明

### 背景

我們想在 RADAR 風格的流程（先切出解剖單元 → 取各器官 embedding → 下游學習）中加入 **CT multi-window 輸入**。目前**唯一可用的預訓練模型是 TotalSegmentator**（nnU-Net v2），沒有其他 foundation encoder。之後這套流程會套用到其他器官的 CT 資料，那些情境同樣沒有合適的公開 encoder，所以這個實驗也是在模擬未來的實際流程。

### 要回答的問題

> 在**凍結 TotalSegmentator encoder** 的前提下，讓每個器官的 embedding 能從多個 CT window（soft tissue / liver / bone 等）中挑選資訊，能否提升器官層級 finding 的判別能力？

### 核心做法

1. 用 TotalSegmentator 在**原始 HU** 上產生器官 mask。
2. 以每個器官為中心切出 nnU-Net 訓練時使用的 patch 大小。
3. 對每個 patch 套用多組 HU window，**拉伸回 nnU-Net 熟悉的強度範圍**後，各自送進凍結的 encoder。
4. 從多個 encoder stage 的 feature map 做 mask pooling，得到器官 embedding `[n_organs, n_windows, D]`。
5. 只訓練 linear probe 與 per-organ window gating，比較 baseline 和 multi-window。

### 必須遵守的規則

- **encoder 權重全程凍結**（`requires_grad=False`、`eval()`、`torch.no_grad()`）。
- 不要修改 TotalSegmentator 或 nnU-Net 的套件原始碼；所有自訂程式放在 `mw_exp/` 下。
- 所有昂貴的中間結果（mask、patch 座標、embedding）都要**快取到磁碟**，並支援中斷後續跑。
- 資料切分**以病人為單位**；超參數只能在 validation set 上調，test set 只在最後使用一次。
- 程式要**以設定檔驅動**（器官清單、window 清單、mask 來源都寫在 yaml），因為之後要換成其他器官的資料。
- 遇到第 2.4 節列出的情況時，**先停下來問我**，不要自行猜測。
- 每完成一個步驟，就在 `mw_exp/PROGRESS.md` 追加簡短紀錄（做了什麼、輸出在哪、有什麼異常）。

---

## 1. 設定（請先填寫或向我確認）

`mw_exp/config.yaml`：

```yaml
ct_dir: <CT volume 路徑，NIfTI，單位必須是 HU>
labels_file: <finding 標籤 csv；若沒有見第 5.3 節>
split_file: <patient-wise split；若沒有則自行建立並存檔>
mask_source: totalsegmentator   # 之後可改成 custom，從 mask_dir 讀取自訂分割
mask_dir: <若 mask_source=custom，mask 的路徑>
ts_task: total                  # TotalSegmentator 任務名稱
ts_model_dir: <TotalSegmentator 權重所在資料夾；預設在 ~/.totalsegmentator 下，請自行找出>
output_dir: mw_exp/outputs
subset_size: 2000               # 快速驗證先用子集，跑通後再擴大
device: cuda
amp: true
seed: 42
```

---

## 2. 步驟 0：讀懂 TotalSegmentator / nnU-Net 模型（不要跳過）

把以下答案寫進 `mw_exp/MODEL_NOTES.md`，附上檔案路徑與相關程式位置。

### 2.1 模型結構

1. `total` 任務實際由**哪幾個子模型**組成（TotalSegmentator v2 的 total 任務可能拆成多個部分，例如器官、脊椎、肋骨、肌肉等，各自有獨立權重）。列出每個子模型的資料夾、涵蓋的類別、輸入 spacing。
2. 決定本實驗使用哪個子模型的 encoder：
   - **主要**：涵蓋腹部器官的子模型。
   - 骨骼單元（脊椎、骨盆）若不在該子模型中，記錄下來，**先停下來問我**要不要另外使用骨骼子模型的 encoder。
3. 也確認 `--fast`（較低解析度、單一模型）版本的權重是否存在，作為備案記錄下來，但本實驗預設使用全解析度模型。

### 2.2 從 `plans.json` / `dataset_fingerprint.json` 讀出

- `3d_fullres` 設定中的 `patch_size`、`spacing`、網路架構（每個 stage 的 channel 數與下採樣倍率）。
- CT normalization 參數：`foreground_intensity_properties_per_channel` 中的 `percentile_00_5`、`percentile_99_5`、`mean`、`std`。

### 2.3 載入 encoder 並確認輸出

- 用 nnU-Net v2 的 predictor 從模型資料夾載入網路（例如 `nnUNetPredictor.initialize_from_trained_model_folder`），取出 `network.encoder`。
- 用一個隨機輸入確認 encoder 會回傳**每個 stage 的 feature map 列表**，並記錄每個 stage 的 shape。如果 encoder 不直接回傳 skip features，請用 forward hook 取得。
- 確認 TotalSegmentator 本身的前處理：方向（orientation）處理方式、resample 到哪個 spacing。本實驗的影像前處理必須與它一致。

### 2.4 需要停下來問我的情況

- 腹部器官分散在多個子模型中，無法用單一 encoder 涵蓋。
- 找不到 normalization 參數，或 CT 資料看起來不是 HU（數值範圍在 0–255 或 0–1）。
- encoder 無法單獨取出或無法得到各 stage 特徵。
- 第 5.3 節的標籤缺失或 finding → 器官對應有疑義。

---

## 3. 步驟 1：器官 mask 與 patch

### 3.1 Mask

- 對每個 volume 在**原始 HU** 上執行 TotalSegmentator（`mask_source=custom` 時改為讀取 `mask_dir`）。
- 把細類別合併成本實驗使用的解剖單元，對應表存在 `mw_exp/organ_map.yaml`。建議的初始清單如下（可依資料調整）：

```yaml
liver: [liver]
spleen: [spleen]
pancreas: [pancreas]
kidney_left: [kidney_left]
kidney_right: [kidney_right]
gallbladder: [gallbladder]
stomach: [stomach]
adrenal: [adrenal_gland_left, adrenal_gland_right]
# 骨骼單元依第 2.1 節的確認結果決定是否加入
```

- 輸出覆蓋率統計 `outputs/mask_stats.csv`（每個器官的體素數、出現比例）。體素數過小（例如 < 100）視為缺失，標記為 missing。
- 手術切除後的器官（例如膽囊切除）mask 本來就會缺失。**不要直接丟掉這些樣本**，因為「器官不存在」本身就是第 7 節正向對照組要用到的資訊。

### 3.2 器官中心 patch

- 影像先依 TotalSegmentator 的前處理 resample 到模型 spacing；mask 用 nearest neighbor 同步處理。
- 以器官 mask 的質心為中心，切出 `patch_size` 大小的區塊；超出邊界的部分以空氣值（-1024 HU）補齊。
- 器官任一軸大於 patch 時（例如肝臟），沿該軸切成**有重疊的多個 patch**（重疊約 25%），各自抽 embedding 後，以每個 patch 內的器官體素數加權平均。
- 把每個器官的 patch 座標存成 `outputs/patches/{volume_id}.json`，確保可重現。

---

## 4. 步驟 2：Window 與輸入映射

### 4.1 Window 清單 `mw_exp/windows.yaml`

| name | level | width | HU 範圍 | 用途 |
|---|---|---|---|---|
| `baseline` | — | — | nnU-Net 原始 normalization | 對照組 |
| `soft_tissue` | 40 | 400 | −160 ~ 240 | 一般腹部器官 |
| `liver` | 60 | 150 | −15 ~ 135 | 肝臟低對比病灶、脂肪肝 |
| `narrow_abd` | 40 | 250 | −85 ~ 165 | 胰臟、腎臟小病灶 |
| `bone` | 400 | 1800 | −500 ~ 1300 | 骨骼 |

### 4.2 映射到 nnU-Net 的輸入範圍（關鍵步驟）

encoder 是在 nnU-Net 的 CT normalization 下訓練的。為了減少分布偏移，每個 window 先映射到 [0, 1]，再**拉伸回 nnU-Net clip 的 HU 範圍**，最後套用原本的 z-score：

```python
def window_for_nnunet(hu, level, width, p005, p995, mean, std):
    lo, hi = level - width / 2, level + width / 2
    x = np.clip((hu - lo) / (hi - lo), 0, 1)     # window → [0, 1]
    hu_like = p005 + x * (p995 - p005)            # 拉伸回 nnU-Net 熟悉的 HU 範圍
    return (hu_like - mean) / std                 # nnU-Net 原本的 normalization

def baseline_for_nnunet(hu, p005, p995, mean, std):
    return (np.clip(hu, p005, p995) - mean) / std
```

`baseline` 必須和 nnU-Net 原生的 CT normalization **完全一致**。請在第 6 節的 sanity check 中驗證。

### 4.3 處理順序

1. 讀入原始 HU。
2. **所有空間處理（resample、crop、pad）都先在 HU 空間完成**。
3. 最後才套用 window 與映射。不可以先 window 再 resample。

---

## 5. 步驟 3：抽取 embedding 與準備標籤

### 5.1 多 stage mask pooling

- 每個器官 patch、每個 window 各跑一次凍結 encoder，取得所有 stage 的 feature map。
- 使用**中間的 2–3 個 stage**（排除解析度最高的第一個 stage，以及解析度過低的 bottleneck）。實際選哪幾個 stage，依第 2.2 節記錄的 shape 決定，並寫進 `MODEL_NOTES.md`。
- 對每個選用的 stage：把器官 mask 用 **average pooling 下採樣**到該 stage 的解析度，得到每個位置的器官比例，再以此為權重對 feature map 做加權平均。
- 若某器官在某 stage 的總權重過小（小器官在低解析度層幾乎消失），該 stage 改用 mask 的 bounding box 內平均作為替代，並記錄發生次數。
- 各 stage 的向量做 L2 normalize 後串接，得到維度 D 的器官 embedding。

### 5.2 儲存格式

每個 volume 一個檔案 `outputs/emb/{volume_id}.npz`：
- `emb`: `[n_organs, n_windows, D]`（float16）
- `organ_present`: `[n_organs]`（bool）
- `organ_names`, `window_names`, `stages_used`

先用 `subset_size` 的子集執行，記錄每個 volume 的處理時間與 GPU 記憶體峰值。

### 5.3 標籤

- 若 `labels_file` 存在：讀取為 `[volume_id, finding_1, ..., finding_K]` 二元標籤。
- 若不存在：**先停下來問我**。一個可行的模擬選項是使用 Merlin Abdominal CT Dataset（只用資料，不用其 encoder），搭配 Pillar-0 釋出的 RATE / RATE-Evals（`https://github.com/YalaLab/rate`、`https://github.com/YalaLab/rate-evals`）提供的報告標籤。
- 草擬 `mw_exp/finding_organ_map.yaml`（finding → 器官），並為每個 finding 標記：
  - `low_contrast: true/false`（例如小肝臟病灶、胰臟腫瘤、小腎臟病灶）
  - `control: density / structure / none`（見第 7 節）
- **草擬完成後停下來讓我確認**，再繼續。
- 測試集陽性數 < 10 的 finding 從主要分析中排除，另外列出。

---

## 6. 步驟 4：Sanity checks（全部通過才進入實驗）

1. **normalization 一致性**：`baseline_for_nnunet` 的輸出和 nnU-Net 原生前處理的輸出逐體素比對，最大誤差應 < 1e-4。
2. **encoder 仍能分割**：用 baseline 輸入跑完整網路（encoder + decoder），在 3 個 volume 上的 Dice 應接近 TotalSegmentator 原本的輸出（器官 Dice > 0.9）。這確認前處理和模型載入都正確。
3. **windowed 輸入的分割表現**（診斷用，不需要通過門檻）：用 `soft_tissue` window 輸入同樣跑完整網路，記錄 Dice。若 Dice 嚴重崩壞，代表分布偏移很大，要在報告中說明。
4. **視覺檢查**：隨機抽 5 個 volume，對每個 window 存一張經過肝臟的 axial 切片 PNG，並疊上 mask 輪廓，存到 `outputs/qc/`。
5. **embedding 健康度**：沒有 NaN/Inf；各 window embedding 的平均 norm 不應差距數量級。

---

## 7. 步驟 5：實驗

所有實驗都只使用快取的 embedding。每個 finding 使用其對應器官的 embedding。

| 編號 | 名稱 | 輸入 | 維度 | 目的 |
|---|---|---|---|---|
| E0 | Baseline | `baseline` embedding | D | 對照組 |
| E1 | 單一 window | 每個 window 各自一組 | D | 看每個 window 單獨的價值 |
| E2 | Concat | 所有 window 串接 | N·D | 資訊上限（維度較大，需另加強正則化） |
| E3 | Learned gating | per-organ softmax 加權合併 | D | **主要方法** |
| E4 | Fixed preset | 依醫師慣例指定（liver→liver window、bone→bone window…） | D | 比較人工指定與模型學習 |

### E3 gating 模組

```python
class OrganWindowGate(nn.Module):
    # 每個器官一組可學習 logits，長度 = n_windows，初始化為 0（等權重）
    # emb: [B, n_windows, D] -> softmax(logits[organ]) 加權和 -> [B, D]
```

- gating 與 linear head 一起訓練；同一器官的所有 finding 共享同一組 gating 權重。
- E3 有正向訊號時，才額外嘗試 E3b：由輸入 embedding 動態產生權重的小型 MLP。

### 正向對照組（必做）

凍結的分割 encoder 可能對很多病灶不敏感，所以需要預期「一定看得到」的 finding 來確認 pipeline 正常：

- **密度型對照**：脂肪肝（肝臟密度下降）。預期 `liver` window 在 E1 中明顯優於 baseline；若沒有，優先懷疑 windowing 或映射的實作。
- **結構型對照**：膽囊切除、腎臟切除、脾臟切除術後等。預期所有 window 都能判斷得很好；若連這類都做不好，問題在 embedding 抽取或 probe，不在 windowing。

對照組的結果要在報告中**單獨列出**，不要混進主要的平均分數。

### 訓練設定（所有實驗共用）

- 每個 finding 一個 linear head，class-balanced BCE。
- Adam；lr ∈ {1e-4, 1e-3, 1e-2}、weight decay ∈ {0, 1e-4, 1e-3}，依 validation mean AUROC 挑選。
- Early stopping 依 validation mean AUROC。
- 3 個 seed，報告平均值。

---

## 8. 步驟 6：評估與報告

### 8.1 指標

- 每個 finding 的 test AUROC。
- 整體 mean AUROC，以及 `low_contrast`、其他 finding、正向對照組三個子集各自的 mean AUROC。
- **以病人為單位的 bootstrap**（1000 次）95% CI。
- 成對差值：E3 − E0、E4 − E0、E3 − E4，附 95% CI。
- 每個器官的 win rate（E3 勝過 E0 的 finding 比例）。

### 8.2 可解釋性

- E3 學到的器官 × window 權重熱圖 `outputs/gate_weights.png`，數值存成 csv。
- 檢查是否符合直覺（例如肝臟偏重 liver window），並寫入報告。

### 8.3 `mw_exp/REPORT.md` 內容

1. 設定摘要：資料量、使用的 TotalSegmentator 子模型、patch 大小、選用的 stage、器官與 finding 數量、window 清單。
2. Sanity check 結果（特別是第 6 節第 3 點，windowed 輸入下的分割 Dice）。
3. 正向對照組結果。
4. 主表：E0–E4 的 mean AUROC（整體、low-contrast、其他），附 95% CI。
5. 成對差值表與顯著性。
6. 各器官結果拆解與 gating 權重解讀。
7. 異常與限制：缺失器官比例、小器官在低解析度 stage 的替代次數、分布偏移跡象。
8. 結論與下一步建議（依第 8.4 節判準）。

### 8.4 判斷準則與下一步

- **正向對照組失敗** → 先修 pipeline，其他結論都不成立。
- **值得繼續**：E3 或 E4 相對 E0 的提升 95% CI 不跨 0，且 low-contrast 子集的提升大於其他子集。
  → 下一步：**方法 2**。以 TotalSegmentator encoder 為初始化，把第一層 conv 從 1 channel 擴展為 N channel（權重複製 N 份、各除以 N），再用標籤或報告對比學習進行 finetune。
- **訊號不明**：整體提升不顯著，但 E1 中某些 window 在特定器官明顯優於 baseline，或第 6 節第 3 點顯示 windowed 輸入下 Dice 嚴重下降。
  → 凍結 encoder 的分布偏移可能掩蓋了效果，同樣建議進入方法 2 再驗證。
- **不值得繼續**：所有變體都不優於 E0，E1 中各 window 也都不如 baseline，且正向對照組正常。

---

## 9. 為了之後套用到其他器官的設計要求

- 器官清單、window 清單、finding 對應全部由 yaml 決定，程式中**不可寫死器官名稱或 HU 數值**。
- `mask_source` 支援 `totalsegmentator` 與 `custom` 兩種；`custom` 時從 `mask_dir` 讀取 NIfTI mask，類別對應同樣寫在 `organ_map.yaml`。
- encoder 載入要抽象成一個介面（輸入：已正規化的 patch；輸出：各 stage feature map），之後可以換成其他 nnU-Net 模型或自訓練的 encoder。
- 在 `README.md` 寫清楚：換一個新器官資料集時，需要修改哪些設定檔，以及要重跑哪些步驟。

---

## 10. 建議目錄結構

```
mw_exp/
├── config.yaml
├── windows.yaml
├── organ_map.yaml
├── finding_organ_map.yaml
├── MODEL_NOTES.md
├── PROGRESS.md
├── REPORT.md
├── README.md
├── src/
│   ├── encoder.py       # 載入 TotalSegmentator/nnU-Net encoder、取各 stage 特徵
│   ├── windowing.py     # window + nnU-Net 映射
│   ├── masks.py         # 執行 TotalSegmentator 或讀取自訂 mask、類別合併
│   ├── patches.py       # 器官中心 patch 與大器官切塊
│   ├── extract.py       # 多 stage mask pooling 抽 embedding，可續跑
│   ├── qc.py            # sanity checks 與視覺化
│   ├── probe.py         # E0–E4 訓練
│   └── evaluate.py      # AUROC、bootstrap、圖表、報告
└── outputs/
    ├── masks/
    ├── patches/
    ├── emb/
    ├── qc/
    └── results/
```

## 11. 執行順序總覽

1. 讀模型 → 寫 `MODEL_NOTES.md` →（有疑問就停下來問）
2. 產生 mask、patch 與覆蓋率統計
3. 建 `finding_organ_map.yaml`、low-contrast 與對照組標記 → **停下來讓我確認**
4. 抽 embedding（先跑子集）
5. Sanity checks 全部通過
6. 跑 E0–E4 與正向對照組
7. 評估、畫圖、寫 `REPORT.md`
