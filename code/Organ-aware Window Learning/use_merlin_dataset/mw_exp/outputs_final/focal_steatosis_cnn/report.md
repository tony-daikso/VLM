# 局部脂肪浸潤：多區域 3D CNN＋堆疊（focal_steatosis_cnn）

日期：2026-11-06（12:40–17:40）｜A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/focal_steatosis_cnn/`（Mac 鏡像同路徑）
- 標籤：`steatosis_split` 的 `y_focal`（規則在 test 前鎖定）；fold 用 y_focal 的 OOF fold（與實驗 N 相同）。**只用 train 15,144（陽性 749）／val 5,004（陽性 266），test 沒有讀，也沒有評分。**
- 選模：**只看 train 5-fold OOF**；val 只在最後評一次。
- 硬體：crop 在 A6000 CPU（nice 19，3 個 process）；訓練在 4090（port 23688，VRAM ≤ 7.5 GB，實際峰值 6.6 GB；2 threads；nice 10）；堆疊在 Mac CPU。A6000 GPU、Spark、CT-rate 3080 都沒有用。

## 0. 摘要
- **CNN 單獨**：選中的設定 B（3D，w24），3 個 seed 平均後 train OOF **.656**，val .646 [.613, .677]。比實驗 N 的凍結 encoder 單元頭（OOF .675）和 REF（.680）都差。
- **依 train OOF 選出：CUR＋CNN**（CUR＝實驗 N 的 REF＋全部單元頭；GBM 堆疊時多加 CNN logit）。
  - train OOF .6968 → **.6974**，Δ +.0005 [−.003, +.004]
  - val .676 → **.681 [.649, .712]**，Δ +.004 [−.001, +.009]，**不顯著**
  - 年齡配對 AUROC（val）：.668 → .671，Δ +.003 [−.003, +.008]
  - 90% 特異度下的敏感度（val）：25.9% → 25.2%；用 train OOF 門檻時 24.1% → 22.6%
- **結論**：CNN 帶來的資訊幾乎都和現有的特徵重疊。train OOF 增益只有 +.0005，在雜訊範圍內。依規則名義上選到 CUR＋CNN（val .681），但**建議正式最佳維持 .676**，局部脂肪浸潤仍列為次要指標。

## 1. 方法
- **Crop**：使用實驗 N 的 48³ 肝臟位置 crop（`nu_crop.py fst`，程式未修改）。共 3 個位置：鐮狀韌帶／IV 段前方、膽囊窩、肝門。
  - CT-rate 上原有的 14,526 個 crop 已經刪除，所以重新產生 14,379 例：4,628 例用 A6000 的 prep；9,751 例的 3 mm prep／mask 從 CT-rate NVMe 以 ionice idle 分批讀過來，crop 完就刪除。
  - train/val 20,148 例全部都有 crop。有 8 例三個位置都缺，其中 train 3 例。
- **輸入**：每個位置 8 個 channel。
  - 5 個視窗（baseline z-score、軟組織、肝臟、窄腹部、骨）
  - 肝臟相對 HU：(HU − 該例肝臟中位數) / 40
  - 位置肝臟遮罩 r0
  - crop 內肝臟遮罩 r2
- **模型**：3 個位置共用同一個 3D ResNet 式 encoder。
  - 每個位置做 GAP＋GMP＋肝臟遮罩內平均，再加上位置 embedding。
  - 3 個位置之間用 gated attention 加 max pooling 合併後，接 logit。
- **訓練**：
  - 增強：隨機 40³ 子區塊、三軸翻轉、HU 縮放 ±3%、平移 ±10 HU、雜訊、位置 dropout（15%）
  - 類別不平衡：平衡抽樣，每批陽性 25%
  - 其他：label smoothing、AdamW＋OneCycle、EMA 權重；推論時用 4 種翻轉 TTA
- **設定**（事先固定，依 train OOF 選）：
  - A：3D，w16，20×4000 樣本
  - B：3D，w24，30×6000 樣本
  - C：各向異性（stem 與 layer1 用 1×3×3 卷積）的 2.5D 式，w16
  - 每個設定跑 seed 0／1／2；每個 seed 訓練 5 個 fold 模型（產生 OOF）和 1 個全 train 模型（預測 val）。
  - 各 seed 先用 OOF 的平均與標準差標準化後再平均。
- **堆疊**（`scripts/fsc_stack.py`）：
  - 用實驗 N 的 `nu_fst_stack` 設計：E3_late＋HU＋FL＋單元頭，6 個候選模型。
  - 重建後的 CUR 為 OOF .6968／val .6762，與實驗 N 完全一致。
  - 比較三種：CUR＋CNN、REF＋CNN、blend（LR 用 [CUR 分數, CNN]）。

## 2. 結果（val 5,004／266）

| 模型 | train OOF | Δ OOF [CI] | val [CI] | Δ val [CI] | 年齡配對 train／val | Δ 年齡配對 val [CI] | 敏感度@90%特異度 val（train 門檻） |
|---|---|---|---|---|---|---|---|
| CUR（實驗 N 最佳） | .6968 | – | .676 [.644, .708] | – | .693／.668 | – | 25.9%（24.1%） |
| **CUR＋CNN ★** | **.6974** | +.0005 [−.003, +.004] | **.681 [.649, .712]** | +.004 [−.001, +.009] | .693／.671 | +.003 [−.003, +.008] | 25.2%（22.6%） |
| blend（CUR, CNN） | .6971 | +.0003 [−.004, +.004] | .680 [.648, .711] | +.004 [−.001, +.008] | .693／.670 | +.002 [−.002, +.007] | 26.3%（23.3%） |
| REF＋CNN | .6819 | −.015 [−.025, −.005] | .676 [.643, .709] | +.000 [−.014, +.015] | .677／.669 | +.001 | 26.3%（27.8%） |
| CNN B 單獨（選中） | .6557 | −.041 | .646 [.613, .677] | −.031 [−.055, −.007] | .650／.632 | −.037 | 22.2%（18.0%） |
| CNN A 單獨 | .6530 | −.044 | .634 | −.043 | .648／.622 | −.047 | 19.9% |
| CNN C 單獨（2.5D 式） | .6529 | −.044 | .638 | −.038 | .647／.625 | −.044 | 21.4% |

★ 依 train OOF 選出（候選：CUR、CUR＋CNN、blend）。

- 各 seed 的 OOF：
  - A：.650／.649／.651
  - B：.654／.651／.653
  - C：.649／.651／.651
- 各 fold 的 inner AUROC 大約落在 .63–.69。
- val 用「5 個 fold 模型平均」取代「全 train 模型」時，CNN 單獨為 .644（差異很小）。
- 陰性中分數與年齡的 Spearman：CUR −.232，CUR＋CNN −.243。

## 3. 解讀
1. 從頭訓練的小型 CNN 只有 749 個陽性，標籤雜訊又大：24% 措辭不確定、約 28% 陰性是漏標。結果 OOF 只有 .65，學到的訊號比凍結的 TS encoder 單元頭（.675）少。
2. CNN 和現有特徵高度重疊。加入後 train OOF 只多 +.0005，val +.004 不顯著，敏感度也沒有改善。3D、加寬、2.5D 式三種設定的差異都在 .003 以內。
3. 瓶頸仍然是解析度（3 mm 門脈期上，1–2 cm、只差 10–20 HU 的病灶）和標籤。要再往上，可能需要 1.5 mm 原始影像的 crop，或先清理標籤。這兩件都要使用者決定。

## 4. 執行紀錄與檔案
- 第一次執行時發現 bug 並已修正：train 中有 3 例三個位置都缺，被抽到時 max-pool 的 −1e4 會讓 loss 暴衝。修正後全部重跑，舊結果放在 4090 `~/OWL/fsc/old_bug/`，沒有使用。
- 程式：`scripts/`
  - `fsc_crop_chain.sh`：A6000 crop
  - `fsc_push.sh`：Mac 中繼
  - `fsc_pack.py`、`fsc_train.py`、`run_fsc.sh`：4090
  - `fsc_stack.py`：Mac
- 預測：`runs/{A,B,C}/pred_{0..4,full}_s{0,1,2}.csv`（只有 train/val）
- 堆疊：`stack/`（results.json、scores_oof.csv、scores_val.csv、stack.log）；`stackin/` 是已依 train/val 過濾的輸入
- log：`logs_4090/`
- 磁碟：
  - A6000 `crops/`（6.0 GB，14,379 例）
  - 4090 `~/OWL/fsc/crops`（8.3 GB）、`~/OWL/fsc/work`（19 GB memmap）
  - 都可以刪；要重跑時保留即可。
