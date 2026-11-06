# 腎水腫：腎竇／腎盂區域單元（Exp M 單元 1）

日期：2026-11-06｜主機：A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/region_units/hydronephrosis/`；crop 在 A6000 與 CT-rate 的 CPU 上產生，embedding 與 probe 在 4090 GPU 上跑，堆疊在 A6000 CPU 上跑｜**test 未讀取、未評分**：crop 與 embedding 只做了 train＋val 的 20,364 例，標籤檔也只有 train／val。

## 0. 摘要
- **新單元**：每顆腎切一個以腎竇／腎門（內側凹陷）為中心的 48³ crop（3 mm，即 144 mm），對 4 個區域做 mask pooling（mean＋max）。兩顆腎的 embedding 用 MIL max 合併。
- **與 REF 比較**：REF 是 integrated_best 的腎水腫堆疊（E3_late＋兩腎 MIL max＋腎液體 HU，GBM），train OOF 0.936，val 0.944。本次以相同程式重建，數字完全一致。
- **選定模型**：依 train OOF 選出 **REF＋全部單元頭**（U_mil、U_mean、U_org_mil，GBM）。
  - train OOF：0.936 → **0.949**，Δ +0.013 [+0.008, +0.018]。
  - val：0.944 → **0.957 [0.945, 0.967]**，Δ **+0.012 [+0.005, +0.020]**。
- **年齡配對 AUROC**（val）：0.944 → **0.957**，Δ +0.013 [+0.005, +0.021]。
- **陰性中分數與年齡的 Spearman**（val）：0.242 → 0.216，沒有變得更依賴年齡。
- **90% 特異度下的敏感度**（val）：82.8% → 86.9%；改用 train OOF 門檻時為 81.4% → 84.6%。
- **單元單獨使用**：U_mil 的 val 是 0.942，和整個 REF 堆疊（0.944）幾乎相同；但同一組 crop 改成兩腎平均（U_mean）只有 0.915。單側病灶需要用 MIL 處理。
- **結論**：腎水腫的新最佳是 **0.957**（取代 0.944）。

## 1. 方法
### 1.1 Crop（CPU，`scripts/ru_crop.py hn`）
- **輸入**：3 mm prep 的 HU，加上 TS `total` 的 3 mm mask。
- **方向判定**：與 RLQ 單元相同，由脊椎、腸道、肝臟、膀胱的位置推定，不依賴 affine。
- **腎竇**：與 `hn_hu.py` 相同的定義。先對腎臟做 3D closing（9 mm），每個 slice 補洞後減去腎臟本身，只保留未標註或腎囊腫的體素。
- **crop 中心**：腎竇的質心，相當於腎門與內側凹陷處。腎竇少於 20 個體素時，改用腎臟質心往內側移 15 mm（`medial`）。
- **四個區域**：
  - r0：腎竇／腎盂（腎竇向 free space 擴張 1 個體素）
  - r1：腎實質（含腎囊腫）
  - r2：以腎門為中心、半徑 24 mm 的球（排除骨骼）
  - r3：腎周殼層（腎外 12 mm，排除 closing 範圍，只留 free space）
- **定位結果**（train＋val）：

  | 腎 | sinus | medial | 缺失 | 無法判定方向 |
  |---|---|---|---|---|
  | 右腎 | 20,108 | 57 | 179 | 20 |
  | 左腎 | 20,016 | 60 | 268 | 20 |

  缺失多為腎切除或萎縮腎。
- **QC**：抽 20 例 train（`qc/qc_00–19.png`、`qc/montage.png`；ID 只放在 `qc/qc_index.csv`）。共 40 顆腎，其中 39 顆存在：
  - 38 顆的中心正確落在腎竇／腎門。
  - 1 顆走了 `medial` 備援，中心仍在腎內。
  - 1 顆腎缺失。

### 1.2 Embedding（4090 GPU，`scripts/ru_extract.py`）
- 使用凍結的 TS `--fast` encoder（Dataset297），5 個 HU 視窗。
- 取 stage 1–3，每個區域做 mean 與 max pooling，每個 stage 各自 L2 normalize，得到 896 維。每例輸出 [2 腎, 4 區域, 5 視窗, 896]。
- 執行條件：VRAM 上限 10 GB（實際約 0.8 GB），4 threads，nice 19。約 45 例/秒，20,364 例全部完成，0 例缺失。
- 傳輸：crop 經 Mac 中繼分批送進 4090，embedding 做完後就刪掉該 crop。

### 1.3 Probe（4090 GPU，`scripts/ru_probe.py`）
- 架構：E3_late 式的 late head，每個視窗一個線性 head，再由 gate 合併。
- 訓練設定：
  - 只用 train 做標準化，以「腎 × 區域」為單位；缺失的區域填 0。
  - class-balanced BCE；Adam，lr 1e-3、wd 1e-4。
  - early stopping 用 train 內切出的 15% 子集。
  - seed 42／43，fold 沿用腎水腫 E3_late OOF 的 5 個 fold。
- 標籤：與 integrated_best 相同（labels14 的 hydronephrosis 欄）。
- 變體：
  - U_mil：每顆腎是一個 instance，4 個區域串接；兩腎共用 head，加上側別 bias，取 MIL max。
  - U_mean：同樣的 instance，但兩腎取平均。
  - U_org_mil：每顆腎的 instance 再串接該腎原本的 TS 器官 embedding，取 MIL max。

### 1.4 堆疊（A6000 CPU，`scripts/ru_stack.py`）
- REF 的欄位（134 個）與模型設定（GBM_lr.02_l15_m60_800）取自 `integrated_best/models/hydronephrosis_features.json`，用 `ib_stack.load_feats` 重建。
- 新特徵集：REF＋各單元頭，以及 REF＋全部單元頭。
- 候選模型：REF 原本的 GBM 與 LR_C0.1。
- 選擇方式：只看 train 5-fold OOF。選好後用全部 train 重新訓練，在 val 上只評一次。
- 評估：
  - AUROC 的 CI 用 bootstrap 2,000 次。
  - Δ 與 REF 用配對 bootstrap。
  - 年齡配對 AUROC 用 10 歲分箱。
  - 另報陰性中分數與年齡的 Spearman、90% 特異度下的敏感度。
  - 年齡與性別都不作為模型輸入。

## 2. 結果

train 15,199 例（陽性 1,113）；val 5,024 例（陽性 344）。**粗體**為依 train OOF 選出的模型。

| 模型 | 分類器 | train OOF | Δ OOF vs REF [CI] | val AUROC [CI] | Δ val vs REF [CI] | 年齡配對 train／val | Δ 年齡配對 val [CI] | Spearman（陰性，val） | 敏感度@90%特異度（val） |
|---|---|---|---|---|---|---|---|---|---|
| REF（integrated_best） | GBM | 0.936 | – | 0.944 [0.931, 0.956] | – | 0.936／0.944 | – | 0.242 | 82.8% |
| U_mil 單獨 | LR | 0.932 | −0.004 [−0.012, +0.004] | 0.942 [0.929, 0.954] | −0.002 [−0.014, +0.009] | 0.932／0.944 | −0.000 | 0.222 | 84.0% |
| U_mean 單獨 | LR | 0.916 | −0.021 | 0.915 [0.896, 0.931] | −0.030 [−0.045, −0.014] | 0.917／0.913 | −0.032 | 0.118 | 79.4% |
| U_org_mil 單獨 | LR | 0.932 | −0.004 | 0.943 [0.929, 0.955] | −0.002 [−0.013, +0.009] | 0.932／0.944 | −0.000 | 0.203 | 85.5% |
| REF＋U_mil | GBM | 0.947 | +0.011 [+0.006, +0.015] | 0.956 [0.945, 0.966] | +0.012 [+0.006, +0.019] | 0.947／0.957 | +0.013 [+0.006, +0.020] | 0.237 | 88.1% |
| REF＋U_mean | GBM | 0.946 | +0.010 [+0.006, +0.014] | 0.950 [0.938, 0.961] | +0.006 [−0.000, +0.013] | 0.946／0.950 | +0.006 | 0.218 | 84.6% |
| REF＋U_org_mil | GBM | 0.946 | +0.010 [+0.006, +0.015] | 0.955 [0.943, 0.966] | +0.011 [+0.004, +0.018] | 0.946／0.955 | +0.011 [+0.005, +0.018] | 0.228 | 86.9% |
| **REF＋全部單元頭** | **GBM** | **0.949** | **+0.013 [+0.008, +0.018]** | **0.957 [0.945, 0.967]** | **+0.012 [+0.005, +0.020]** | **0.949／0.957** | **+0.013 [+0.005, +0.021]** | **0.216** | **86.9%** |

- 各模型的 LR 版本在 OOF 上都比 GBM 低約 0.006，完整列表見 `stack.log`。
- 完整數字在 `stack/results.csv`、`stack/results.json`。每例分數在 `stack/scores_oof.csv`、`stack/scores_val.csv`。probe 分數在 `out/oof_train.csv`、`out/val_scores.csv`。

## 3. 解讀
1. **以腎竇為中心的 crop，把 REF 靠手工 HU 特徵才拿到的資訊直接編進了 embedding**：U_mil 單獨使用就達到 val 0.942，與 REF 堆疊的 0.944 相當。原本的 E3_late（整顆腎 mean pooling）只有 0.897，兩腎 MIL max 是 0.922。
2. **再疊回 REF 仍有穩定的增益**：train OOF 與 val 都是約 +0.012，兩者的 CI 都不含 0，年齡配對 AUROC 的提升幅度也一樣。所以增益來自影像，不是年齡捷徑；陰性中分數與年齡的 Spearman 也從 0.24 降到 0.22。
3. **MIL 是關鍵**：同一組 crop，兩腎平均（U_mean）比 MIL max 低 0.027，這和既有結論一致：約 69% 的陽性是單側腎水腫。
4. **加入原本的器官 embedding（U_org_mil）沒有額外幫助**，因為腎竇 crop 的區域已經涵蓋腎實質（r1）。
5. **限制**：
   - QC 只看了 20 例，`medial` 備援約占 0.3%。
   - 腎竇的定義依賴 TS 腎臟 mask。腎外型腎盂與輸尿管只部分落在 r0／r3 內。
   - 單元沒有區分「腎盂旁囊腫」與真正擴張，這部分仍要靠堆疊中的囊腫特徵。
   - test 尚未評分，最終數字要等 val 鎖定後的單次 test 評估。

## 4. 檔案
- `scripts/`：`ru_crop.py`、`ru_qc.py`、`ru_extract.py`、`ru_probe.py`、`ru_stack.py`、`ru_methods.py`、`ru_feed.sh`（Mac 中繼）、`ru_post.sh`、`ru_assist_*.sh`、`ru_chain_*.sh`。共用版在 `region_units/scripts/`。
- `crops/`：A6000 這份約 3.0 GB（A6000 的 10,518 例）；CT-rate `~/OWL/mw_exp/region_units_tmp/hydronephrosis/crops/` 另有 9,846 例。4090 上的 crop 已全部刪除。
- `emb/`：20,364 個 npz，約 1.4 GB。
- `out/`：probe 分數。
- `stack/`：堆疊結果。
- `qc/`：QC 圖。
- log：`crop_a6.log`、`logs_4090/{extract,probe}.log`、`stack.log`。
