# 影像骨質疏鬆（HU 定義）標籤規則

- 決定者：使用者，2026-11-06（LABEL_DECISIONS_PENDING.md 第 6 項）。使用者原話：「我真正想要的是『影像上看得到的骨質疏鬆』」。
- 這是**新增的 finding**。原本以報告定義的「骨質減少」finding 完全不變（正式 val .890）。
- 規則在 2026-11-06 鎖定，**沒有讀取或評分任何 test**。之後 train、val、test 一律用同一組程式與常數。
- 程式與 sha256：
  - `scripts/oh_roi.py`（`3bf0a95620d984f5351fec1ec8434d2d9f2aa9f24ced2e402549a42a64659d70`）：計算骨小樑 ROI 的 HU。
  - `scripts/oh_report_flags.py`（`37fd3b6724463f71bd4dd0b43ab71b5975f0be4cd4cce7ada756a0abd7566feb`）：產生報告旗標，只用在排除規則與退回 L2 的規則。
  - `scripts/oh_labels.py`（`bcd2066752973dae843484bda7d569bc6f5bc04765546bfdac9d8fcbb1267341`）：產生標籤。
  - 產出 `labels_hu_trval.csv`（`92a60675391e11c1ae71b7eccf5337a52bf4d33a5f8f47e5e2b74400b8a04c32`）。

## 使用者規格（原文照錄，2026-11-06）

> USER'S EXACT SPEC for 影像骨質疏鬆 — replace my defaults with this: (1) Label = L1 trabecular-bone ROI HU, stored as a CONTINUOUS value (regression target; report MAE/Pearson/ICC for predicted HU from embeddings-only model). (2) Binary targets derived from it: HU ≤110 → 陽性 (high-specificity definition, primary); HU ≤160 → 篩檢陽性 (screening definition, secondary). Note ≤ (inclusive), not <. (3) If L1 has a fracture OR sclerosis → use L2 instead (define sclerosis detection objectively, e.g. abnormally high trabecular HU / focal sclerotic lesion rule; document it). (4) Exclude cases with bone cement (vertebroplasty/kyphoplasty; detect via very high HU metal-like density in L1/L2 body, e.g. >1000–1500 HU voxels fraction, and/or report mentions) — excluded entirely, not relabeled. The ROI must be trabecular only (eroded body mask avoiding cortex/endplates). Keep contrast-phase check as a reported analysis (no threshold change unless user decides). Report counts per split for each rule (L2 fallback, sclerosis, cement exclusions). Record this spec verbatim in LABEL_DECISIONS_PENDING.md #6 and in the rules file.

## 1. 標籤

- **連續值 `hu`**：選定椎體（L1，必要時改用 L2）骨小樑 ROI 的 HU 中位數，作為主要標籤與迴歸目標。
- **`y110`＝1，若 hu ≤ 110**：「影像骨質疏鬆」陽性。這是高特異度定義，為**主要**二元標籤。
- **`y160`＝1，若 hu ≤ 160**：「影像骨質疏鬆（篩檢）」陽性。這是篩檢定義，為**次要**二元標籤。
- **`cat3` 三級**：
  - 骨質疏鬆：hu ≤ 110
  - 骨量減少：110 < hu ≤ 160
  - 正常：hu > 160
- 兩個門檻都是 **≤（含等號）**。
- 被排除的病例，`hu`、`y110`、`y160` 都是 NaN，既不訓練也不評估。

## 2. 骨小樑 ROI（`oh_roi.py`）

ROI 在看任何標籤之前就已固定，只取骨小樑，避開皮質與終板。

- **資料來源**：1.5 mm seg15，也就是 Dataset292 的椎體 mask，加上正中矢狀 HU slab（11 欄，共 16.5 mm）。
- **步驟**：
  1. 椎體 mask 少於 800 voxel → `no_mask`。
  2. 取 xc＝椎體 x 中位數，使用 xc±3 的矢狀欄（±4.5 mm，避開側面皮質）。落在 slab 內的欄數少於 3 → `off_slab`。
  3. **椎體本體 footprint**：取正中矢狀占有率（xc±2）中最前方的連通區，並填補孔洞，用來排除後方附件。作法和 geom15 相同。
  4. 每一欄先取 S＝椎體 mask ∩ 本體 footprint，代表含皮質的整個椎體，用來偵測骨水泥。接著用 9×7 矩形侵蝕 S，得到骨小樑 ROI：
     - z 方向 4 px＝6 mm，避開上下終板；
     - y 方向 3 px＝4.5 mm，避開前後皮質。
  5. ROI 少於 100 voxel 時，改用較小的侵蝕（3×2 px，標記為 `small`）。仍少於 50 voxel → `small_roi`。
  6. 量測值＝ROI 的 HU 中位數。中位數不在 [−300, 1200] 範圍內 → `implausible`。
- **有效（valid）**：狀態為 `ok`。

## 3. 規則（常數在 `oh_labels.py` 開頭）

### 骨折（L）——符合任一項即算
- 報告某句同時提到該節（L1、L-1、first lumbar，或涵蓋該節的範圍，例如 T12-L2），以及 fracture、compress*、collaps*、wedg*、height loss、plana、burst，且沒有被否定。
- geom15 **中段高度** loss ≥ 0.25，即 Genant 2 級以上的雙凹或壓碎。
- 不使用前、後段高度：它們雜訊太大。若改用「三段最大 loss ≥ .25」，train 中 18% 的 L1 會被判為骨折，對報告 L1 骨折的 PPV 只有 .11。這個判斷只看 train 的分布與報告的一致性，沒有看 val。

### 硬化（L）——符合任一項即算
- 局部硬化病灶：ROI 中 >600 HU 的比例 ≥ 5%。train 的第 99 百分位是 4%。
- 瀰漫性硬化：ROI 中位數 > 400 HU。train 的第 99.9 百分位約 390。
- 相對鄰節過高：ROI 中位數減去鄰節（L1 的鄰節是 T12 與 L2，L2 的鄰節是 L1 與 L3）有效中位數的平均 ≥ 100 HU。train 中 L1 減鄰節的第 99 百分位是 64，第 99.9 百分位是 376。
- 報告某句同時提到該節與 scleros*、sclerotic、blastic、bone island、enostosis、dense、increased density，且沒有被否定。
- 上述門檻都是看 train 的分布（不看標籤）後事先定下的。

### 骨水泥（整例排除）——符合任一項即排除
- 報告提到 vertebroplasty、kyphoplasty 或 vertebral augmentation。
- 報告的同一句同時有脊椎字（vertebra、spine、lumbar、thoracic、T#、L#）與 cement 或 PMMA。這樣可以排除 cemented hip 等關節置換。
- 病史句也算，因為骨水泥會留在骨內。
- L1 或 L2 的整個椎體（含皮質，xc±3 欄）中，>1500 HU 的 voxel ≥ 20 個（約 0.07 mL）。train 的第 99 百分位是 0 個。
- 符合者**整例排除，不重新標記**。

### 選用的椎體
| 條件 | 結果 |
|---|---|
| L1 有效、沒有骨折、沒有硬化 | 用 L1 |
| 否則，L2 有效、沒有骨折、沒有硬化 | 用 L2 |
| 都不符合 | 排除（`no_valid_level`） |

L1 無效（沒有 mask、不在 slab 內、ROI 太小、數值不合理）時，也同樣退回 L2。

### seg15 不在本機時的備援
- 部分病例的 seg15 只存在 Spark（本次規定不使用 Spark）。這些病例改用 geom15 的 `hu_body`（同一個 1.5 mm 椎體正中矢狀面，侵蝕 2 px，取中位數），規則相同。
- 這類病例沒有 voxel 層級資料，所以骨水泥只能看報告，硬化只能用「絕對值」與「相對鄰節」兩條規則。
- 在兩種來源都有的 17,119 例上比較 ROI 與 geom15：ICC .966，偏差 −3.9 HU，≤110 的 κ .84。
- 欄位 `roi_src` 標示每例的來源：`seg15_roi` 或 `geom15_fallback`。
- 若之後要讓全部病例都用同一種 ROI，從 Spark 取回 seg15 再重跑同一支程式即可。

### Split 安全
- 所有程式都先讀 `split.csv`，只保留 train／val 的 id。
- 報告用 key-gated 掃描器讀取，作法和 steatosis_split 相同。
- `oh_report_flags.py --splits test` 必須加 `--allow-test`。`oh_roi.py` 與 `oh_labels.py` 目前寫死只處理 train／val，產生 test 標籤前要加上同樣的開關。

## 4. 各 split 的病例數

| | train | val |
|---|---|---|
| 總數 | 15,309 | 5,055 |
| ROI 來源：seg15 / geom15 備援 / 無 | 12,604 / 2,682 / 23 | 4,917 / 135 / 3 |
| **骨水泥排除**（報告 / HU / 兩者皆是） | **144**（89 / 100 / 45） | **63**（41 / 45 / 23） |
| L1 骨折（報告 / geom 中段高度 / 合計） | 359 / 399 / 622 | 98 / 151 / 207 |
| L1 硬化（局部 / 絕對值 / 相對鄰節 / 報告 / 合計） | 129 / 84 / 88 / 62 / 251 | 57 / 34 / 35 / 17 / 86 |
| **退回 L2**（L1 無效 / L1 骨折 / L1 硬化） | **520**（117 / 335 / 68） | **172**（48 / 107 / 17） |
| 沒有可用椎體，排除 | 341 | 115 |
| 沒有量測值，排除 | 23 | 3 |
| **有標籤** | **14,801** | **4,874** |
| y110（≤110）陽性 | 1,058（7.1%） | 328（6.7%） |
| y160（≤160）陽性 | 4,151（28.0%） | 1,354（27.8%） |
| 三級：骨質疏鬆 / 骨量減少 / 正常 | 1,058 / 3,093 / 10,650 | 328 / 1,026 / 3,520 |
