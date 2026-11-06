# 脂肪肝拆分：瀰漫性脂肪肝 vs 局部脂肪浸潤

日期：2026-11-06｜位置：A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/steatosis_split/`（Mac 鏡像：`mw_exp/outputs_final/steatosis_split/`）
決定來源：LABEL_DECISIONS_PENDING.md 第 1 項（使用者，2026-11-05 23:40）。

**test 完全沒碰**：標籤程式先依 split 過濾，才讀報告和標籤（`--splits test` 若沒有加 `--allow-test` 會直接拒絕）；模型只用 train 5-fold OOF 挑選，val 只用來報告（分層 paired bootstrap 2,000 次）；沒有產生任何 test 分數。全程只用 CPU（nice 19、≤4 threads），沒有使用 A6000／Spark 的 GPU。

## 1. 標籤（規則見 `label_rules.md`，已在 test 前鎖定）

只用報告文字與現有 Qwen 標籤，不用任何模型分數。原陽性（yes／probably_yes）依報告內容分成：

| | train | val |
|---|---|---|
| 原陽性 → 只有瀰漫 | 1,452 | 444 |
| 原陽性 → 只有局部 | 603 | 217 |
| 原陽性 → 兩者都有（重疊） | 146 | 49 |
| 原陽性 → 無法判讀（unassigned，兩者都設為 NaN） | 8 | 5 |
| 原陰性（no／probably_no／not_mentioned） | 12,943 | 4,294 |
| unclear（NaN） | 157 | 46 |
| **瀰漫性脂肪肝 陽性／陰性（盛行率）** | **1,598／13,546（10.6%）** | **493／4,511（9.9%）** |
| **局部脂肪浸潤 陽性／陰性（盛行率）** | **749／14,395（4.9%）** | **266／4,738（5.3%）** |

- 瀰漫陽性包含「瀰漫性脂肪肝，伴局部 sparing」（train 87、val 38）。
- train 的局部陽性中，89% 提到典型位置（鐮狀韌帶、膽囊窩、肝門、fissure、第 IV 段），24% 的措辭不確定。
- 稽核：在 train 抽兩輪，每輪約 57 例，只看產生的旗標和命中的短片語。修正 5 類誤判後才鎖定規則（見 label_rules.md §6）。
- **已知問題**：原標籤是 no、但報告明確寫了局部脂肪的病例，train 有 209 例、val 有 91 例。稽核抽查 12 例，其中 10 例確實是局部脂肪，被 Qwen 標成 no。依使用者規則，主要定義仍把它們算作陰性，因此局部脂肪浸潤的陰性裡有已知的漏標。

## 2. 方法

- **E3_late probe**：作法與 analysis_steatosis 相同（12 個單元合併後用 train 統計量做標準化、class-balanced BCE、Adam 1e-3／1e-4、從 train 切 15% 做 early stopping、2 個 seed），只把標籤換成新定義。輸入比較三種：肝、肝‖脾、肝‖(肝−脾)。每個 finding 依各自的標籤做 StratifiedKFold(5)。
- **HU 堆疊**：沿用 quickwins 的脂肪肝 HU 特徵（肝／脾／L−S／L/S／主動脈／門脈／腎 HU、依 HU 推定的期別與其交互項、脂肪面積）。
- **局部特徵（FL，新增 `scripts/hs_focal_hu.py`）**：
  - 肝臟先侵蝕 2 voxel，再做 3×3×3 中值濾波。
  - HU 異質性：std、MAD、IQR、skew，以及 p1／p5／p10／p25 減去中位數。
  - 低於中位數 −15／−25／−40 HU 的體積，與最大連通塊。
  - 三個典型位置的低密度體積，以及區域中位數與全肝中位數的差：膽囊窩（膽囊 15 mm 內）、肝門（門靜脈 15 mm 內）、鐮狀韌帶／第 IV 段前方（中線 ±30 mm、肝臟前半部）。
  - **max pooling**：把肝臟切成 15 mm 區塊，取區塊中位數的最小值與 p5、單一區塊低密度比例的最大值、區塊之間的標準差。
  - 共算了 20,364 例（train+val）：A6000 9,009 例、3080 11,355 例，約 14 分鐘。
- 註：organ embedding 只有 mean pooling，肝臟沒有 tile 層級的 max embedding（要重抽就得用 GPU），所以 max pooling 改在 HU 區塊層級做。
- **堆疊候選**：LR（C 0.1／0.3／1）與 HistGBM（3 組設定）。特徵組合有 HU、FL、HU+FL 三種，各自再分加不加 E3_late logit（用 OOF 最好的 probe 變體）。**模型只用 train OOF 挑選**；選定後用全部 train 重新 fit，在 val 上評估一次。模型輸入不含年齡和性別。
- 指標：
  - AUROC，附 val 的 95% CI。
  - Δ：相對於 E3_late（肝‖(肝−脾)）的 paired bootstrap 差值。
  - 年齡配對 AUROC：只比較同一個 10 歲年齡層內的陽性與陰性配對。
  - 90% 特異度下的敏感度。

## 3. 結果

### 3.1 主表

| finding | 模型（依 train OOF 挑選） | train OOF | val AUROC [95% CI] | Δval vs E3_late 肝‖(肝−脾) | 年齡配對 AUROC（OOF／val） | 90% 特異度敏感度（OOF／val） |
|---|---|---|---|---|---|---|
| 瀰漫性脂肪肝 | E3_late（肝） | 0.914 | 0.917 [0.901, 0.932] | | 0.914／0.914 | 79.1%／80.1% |
| 瀰漫性脂肪肝 | E3_late（肝‖脾） | 0.925 | 0.916 [0.900, 0.932] | | 0.925／0.913 | 81.2%／80.9% |
| 瀰漫性脂肪肝 | E3_late（肝‖(肝−脾)） | 0.926 | 0.920 [0.903, 0.935] | — | 0.926／0.916 | 82.0%／82.2% |
| 瀰漫性脂肪肝 | 只用 HU（GBM） | 0.929 | 0.919 [0.903, 0.933] | [−0.008, +0.007] | 0.929／0.914 | 81.1%／80.5% |
| 瀰漫性脂肪肝 | **E3_late（肝‖(肝−脾)）+ HU（GBM）** ★ | **0.933** | **0.925 [0.909, 0.940]** | **[+0.000, +0.011]** | 0.933／0.921 | 82.4%／81.3% |
| 瀰漫性脂肪肝 | E3_late + HU + FL（GBM） | 0.932 | 0.926 [0.911, 0.941] | [+0.002, +0.012] | 0.931／0.923 | 83.3%／82.4% |
| 瀰漫性脂肪肝 | E3_late + FL（LR） | 0.925 | 0.918 [0.902, 0.933] | [−0.005, +0.002] | 0.925／0.915 | 81.2%／81.3% |
| 瀰漫性脂肪肝 | 只用 FL（GBM） | 0.826 | 0.831 [0.811, 0.851] | — | 0.825／0.825 | 57.1%／55.8% |
| 局部脂肪浸潤 | E3_late（肝） | 0.669 | 0.674 [0.640, 0.708] | | 0.661／0.664 | 23.2%／26.7% |
| 局部脂肪浸潤 | E3_late（肝‖脾） | 0.675 | 0.656 [0.622, 0.689] | | 0.668／0.647 | 22.7%／24.8% |
| 局部脂肪浸潤 | E3_late（肝‖(肝−脾)） | 0.676 | 0.660 [0.627, 0.694] | — | 0.668／0.649 | 21.9%／25.6% |
| 局部脂肪浸潤 | 只用 HU（GBM） | 0.662 | 0.631 [0.597, 0.663] | [−0.062, +0.002] | 0.659／0.621 | 20.3%／21.1% |
| 局部脂肪浸潤 | E3_late + HU（LR） | 0.679 | 0.653 [0.621, 0.686] | [−0.023, +0.009] | 0.672／0.639 | 24.4%／21.8% |
| 局部脂肪浸潤 | 只用 FL（GBM） | 0.622 | 0.605 [0.570, 0.639] | — | 0.624／0.602 | 18.6%／18.0% |
| 局部脂肪浸潤 | E3_late + FL（GBM） | 0.679 | 0.665 [0.631, 0.698] | [−0.012, +0.023] | 0.673／0.658 | 23.9%／25.6% |
| 局部脂肪浸潤 | **E3_late + HU + FL（GBM）** ★ | **0.680** | **0.665 [0.631, 0.698]** | [−0.014, +0.024] | 0.674／0.659 | 25.1%／28.6% |
| 原合併脂肪肝（參考） | E3_late（肝） | 0.819 | 0.818 [0.799, 0.836] | | 0.823／0.818 | 58.4%／56.6% |
| 原合併脂肪肝（參考） | E3_late + HU（GBM） | 0.840 | 0.825 [0.807, 0.843] | | 0.844／0.824 | 63.1%／58.0% |
| 原合併脂肪肝（參考） | E3_late（肝‖(肝−脾)）+ HU（GBM） | 0.841 | **0.830** [0.812, 0.848] | | —／0.831 | —／59.2% |

★ = 依 train OOF 選出的最佳模型。val 的敏感度是用 val 自己的 90% 特異度閾值算的；改用 train OOF 定出的閾值，瀰漫型最佳模型的敏感度是 78.9%。局部型的閾值在 train 與 val 之間不穩定（盛行率只有 4.9%，分數分布平坦），不建議用固定閾值。

### 3.2 原合併模型（用舊標籤訓練）在新標籤上的表現（不重新訓練）

| 新 finding | E3_late（舊標籤）OOF／val | E3_late + HU（舊標籤）OOF／val |
|---|---|---|
| 瀰漫性脂肪肝 | 0.901／0.902 | 0.923／0.919 |
| 局部脂肪浸潤 | 0.590／0.622 | 0.605／0.608 |

改用新標籤訓練後，瀰漫型 E3_late 的 OOF 從 0.901 升到 0.914（肝），再升到 0.926（肝‖(肝−脾)）；局部型從 0.590 升到 0.669。

### 3.3 敏感度分析（只換標籤變體重新評估，不重新訓練）

| 變體 | 模型 | train OOF | val |
|---|---|---|---|
| 瀰漫型；只有局部的病例不算陰性（`y_diffuse_excl`） | E3_late 肝‖(肝−脾)／最佳堆疊 | 0.926／0.933 | 0.919／0.924 |
| 局部型；只有瀰漫的病例不算陰性（`y_focal_excl`） | E3_late／最佳堆疊 | 0.677／0.680 | 0.663／0.667 |
| 局部型；排除「原標 no、但報告寫了局部脂肪」的病例（`y_focal_negclean`） | E3_late／最佳堆疊 | 0.679／0.682 | 0.663／0.668 |

各變體的差異都在 ±0.005 以內，所以結論不受主要定義細節的影響。

## 4. 結論

1. **瀰漫性脂肪肝：val 0.925**（原合併 finding 是 0.830，差 +0.095；這個差距包含評估定義的改變和模型的改進）。
   - 最佳模型是 E3_late（肝‖(肝−脾)）+ HU GBM，train OOF 0.933。
   - 相對單獨的 probe：val Δ 的 CI 下限剛好約為 0；train OOF Δ 的 CI 是 [+0.004, +0.010]，不跨 0。
   - 年齡配對 AUROC 是 0.921，幾乎不受年齡影響。
   - 錯誤分析預估的「約 0.90～0.93」得到證實。
2. **局部脂肪浸潤：val 約 0.66～0.67，模型幾乎抓不到。**
   - 最佳模型是 E3_late + HU + FL，train OOF 0.680，但和單獨的 probe 差異不顯著（val Δ [−0.014, +0.024]）。
   - 局部 HU 特徵（典型位置的低密度體積、異質性、區塊 max pooling）單獨只有 0.62；單一特徵最好的「鐮狀韌帶區 < −25 HU 體積」也只有 0.53。
   - 原因有三：① 局部脂肪通常只有 1–2 cm，在 3 mm、門脈期的影像上對比只有約 10–20 HU；② 24% 的報告措辭不確定；③ 約 28% 漏標（原陰性中寫了局部脂肪）。
   - 在凍結 encoder 加 mean pooling 的框架下，**局部脂肪浸潤不適合當主要 finding**。建議只列為次要／探索性指標，或不納入。
3. 原合併 finding 只保留作參考：val 0.830。

## 5. 納入正式設定時需要的修改（**沒有改 finding_organ_map.yaml**，只記錄在這裡）

- **標籤**：`labels_disease14.csv` 要新增兩欄 `hepatic_steatosis_diffuse`、`hepatic_steatosis_focal`，由 `scripts/hs_split_labels.py` 產生。test 要等設定鎖定後，再用 `--splits test --allow-test` 跑。原本的 `hepatic_steatosis` 保留作參考。
- **`finding_organ_map.yaml`**：
  - `hepatic_steatosis_diffuse: {organs: [liver, spleen], agg: concat_diff}`。目前 probe.py 不支援 `agg: concat_diff`，要先新增這種聚合方式（liver ‖ (liver − spleen)，analysis_steatosis 已建議過）。如果不改 probe.py，改用 `organs: [liver]`，val 也有 0.917。
  - `hepatic_steatosis_focal: {organs: [liver]}`（建議標為次要／探索性）。
  - 原本的 `hepatic_steatosis` 改為參考，或直接移除。
- **堆疊**：test 時用 `scripts/hs_split_stack.py apply`。需要的輸入是 test 的 hs_hu.py 與 hs_focal_hu.py 特徵，加上 E3_late 分數。

## 6. 檔案

- `labels_trval_split.csv`：只含 train+val，只有旗標與標籤欄位（`y_diffuse`、`y_focal`、`*_excl`、`y_focal_negclean`、`subtype` 等）
- `label_rules.md`：規則、各類數量、稽核紀錄
- `scripts/`：
  - `hs_split_labels.py`：產生標籤；test 之後用同一支
  - `hs_focal_hu.py`：局部 HU 特徵
  - `hs_split_oof.py`：probe OOF
  - `hs_split_stack.py`：堆疊（fit／apply）
  - `hs_split_ref.py`：原合併 finding 的參考數字
- `results_main.csv`（主表）、`results_all.csv`（含 CI 細節）、`results_reference_combined.csv`、`summary.json`（含各候選模型的 CV AUROC 與敏感度分析）
- `oof/`：probe 的 train OOF 與 val 分數（2 個 finding × 3 種輸入）；`oof_train_y_*.csv`、`val_scores_y_*.csv`：各模型的分數
- `models/stack_y_*_*.joblib`：堆疊模型（Mac 上用 scikit-learn 1.9.1 存檔；版本不同時可以用 `fit` 重建，約 15 分鐘）
- `features/fl_a6.csv`、`features/fl_ct.csv`：局部 HU 特徵（train+val）
