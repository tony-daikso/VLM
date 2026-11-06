# 脂肪肝：錯誤分析（TotalSegmentator --fast 凍結 encoder，expB）

日期：2026-11-05。所有分析都在 **train 5-fold OOF** 上做（N = 15,152，陽性 2,209，盛行率 14.6%）；val 只列出描述性數字，沒有用 val 調任何東西；**test 完全沒碰**（id 先依 split 過濾，才讀標籤與報告）。

## 0. 重現性檢查

用 probe.py 的設定在 train 上重做 5-fold OOF：肝臟單元、12 個單元合併的 train 統計量做標準化、class-balanced BCE、Adam lr 1e-3 / wd 1e-4（expB 選定的 HP），early stopping 用 train 內切出的 15%（不碰 val），2 個 seed。與 probe.py 的差別：只訓練脂肪肝一個 finding（probe.py 是 14 個 finding 一起訓練）。

| | train OOF | val（描述性） | expB 原始 val |
|---|---|---|---|
| E0 | 0.803 | 0.793 | 0.795 |
| E3_late | 0.819 | 0.818 | 0.819/0.820 |

可以重現，下面的分析都以這組 OOF 分數為基礎。

## 1. 結論（依估計的 AUROC 損失排序）

| 排名 | 原因 | 證據（train OOF） | 估計損失 |
|---|---|---|---|
| 1 | **標籤定義混雜：「局部脂肪浸潤」被算成脂肪肝** | 陽性中有 609 例（27.6%）的標籤引文只寫 focal fatty infiltration（常在鐮狀韌帶或膽囊窩旁），沒有 steatosis 或 diffuse 字眼。這些病例在 PV 期的肝 HU 中位數是 124（陰性 116），L−S 是 −1（陰性 −3），器官層級的影像上和陰性分不開。E3_late 對這群的 AUROC 只有 **0.596**；瀰漫型則是 **0.916** | 把這群拿掉後 AUROC 0.819 → **0.904（+0.085）** |
| 2 | **肝臟單元缺少參考器官，加上顯影劑強度造成干擾** | 資料幾乎全是門脈期（依 HU 判斷：PV 93%、真正的非顯影只有 51 例，0.3%）。肝的絕對 HU 受顯影劑強度影響很大。portal HU 最高四分位的 AUROC 是 0.744，最低四分位是 0.877。embedding 可以線性解出肝臟的絕對 HU（liver window R² = 0.976），但解不出肝減脾（R² = 0.38），因為脾不在肝臟單元裡。單用手工 HU 特徵的 GBM（L−S、L/S、肝減門脈、主動脈/門脈 HU 等）OOF 就有 **0.835，比 E3_late 的 0.819 還高** | ≈ −0.02（stack HU 或加入脾臟後可以補回：0.819 → 0.840） |
| 3 | **不確定措辭與 probably_yes 陽性** | 陽性中 1,032 例的措辭不確定或屬於 probably_yes；probably_yes 的 AUROC 是 0.775，yes 是 0.826 | 拿掉後 +0.016 |
| 4 | **陰性中可能有漏標** | PV 期 L−S < −25 HU（影像上像脂肪肝）的陰性：not_mentioned 中佔 10.3%，明確寫 no 的只佔 6.0%。E3_late 前 10% 的 FP 有 20% 屬於這一類 | 拿掉後 +0.007 |
| 5 | 肝臟分割 | 肝臟偏小（< p2）的 AUROC 是 0.669，FOV 切到肝臟的是 0.795，但合計只有 617 例 | 拿掉後 +0.002 |
| — | 非顯影 vs 顯影 | 非顯影只有 0.3%。metadata 的 `phase=native` 不可靠：899 例中有 821 例依 HU 判斷其實是 PV | 無法評估，也不是主因 |
| — | 輕重度 | mild 0.841、moderate 0.928、severe 0.977。mild 並不比整體差，「沒寫程度」（0.805）才比較難 | 不是主因 |

**假設 2（normalization 把絕對 HU 洗掉）不成立**：5 個 window 的 embedding 都能以 R² 0.81–0.98 解出肝臟 HU（MAE 2–7 HU）。真正缺的是**相對參考**（脾、血池）。只用肝臟 HU 的 LR 只有 0.715，加上脾和期別後是 0.784，全部 HU 特徵用 GBM 則是 0.835。

## 2. 子群 AUROC（E0 / E3_late，train OOF）

| 子群 | n | 陽性 | E0 | E3_late |
|---|---|---|---|---|
| 全部 | 15,152 | 2,209 | 0.803 | 0.819 |
| 依 HU 判斷為 PV 期 | 14,078 | 2,050 | 0.797 | 0.814 |
| 延遲期（主動脈 < 110） | 853 | 114 | 0.885 | 0.905 |
| 動脈期 | 167 | 38 | 0.717 | 0.672 |
| 非顯影 | 51 | 7 | 0.770 | 0.821 |
| portal HU Q1（低）→ Q4（高） | 約 3,800 / 組 | | 0.861 → 0.736 | 0.877 → 0.744 |
| 瀰漫型 / steatosis 陽性 vs 全部陰性 | | 1,419 | 0.899 | 0.916 |
| 局部脂肪浸潤陽性 vs 全部陰性 | | 609 | 0.583 | 0.596 |
| yes / probably_yes | | 1,911 / 298 | 0.811 / 0.754 | 0.826 / 0.775 |
| 陰性為 no / not_mentioned | | | 0.790 / 0.824 | 0.805 / 0.842 |
| 年齡 < 40 / 55–64 / ≥ 75 | | | 0.791 / 0.832 / 0.720 | 0.806 / 0.851 / 0.734 |
| 男 / 女 | | | 0.816 / 0.792 | 0.834 / 0.805 |
| L3 脂肪面積三分位 低 / 中 / 高 | | 盛行率 7.7% / 13.5% / 22.6% | 0.697 / 0.783 / 0.831 | 0.717 / 0.790 / 0.855 |
| 肝分割有問題 / 正常 | 617 / 14,535 | | 0.710 / 0.806 | 0.755 / 0.821 |

只用干擾因子（年齡、性別、L3 脂肪面積、肝體積）的 OOF AUROC 是 0.697。糖尿病沒有資料。

## 3. Window

單一 window 的 OOF：baseline 0.803、soft_tissue 0.804、**liver 0.806**、narrow_abd 0.802、bone 0.801；E3_late 是 0.819（多 window 合併 +0.016）。window 之間差異很小。liver window 最能解出絕對 HU（R² 0.976），但對 AUROC 沒有額外幫助，因為瓶頸不在絕對 HU。「針對非顯影 CT、以 40–60 HU 為中心的窄 window」在這份資料無法評估（非顯影只有 51 例），預期幫助有限。

## 4. 錯誤類別（train OOF，90% 特異度）

E3_late 在 90% 特異度時，敏感度是 **0.584**（stack 後是 0.632）。下面的類別可以重疊。

**FN（919 例）**：局部脂肪浸潤 58.7%｜PV 期且 L−S ≥ −5（影像上肝不比脾暗）45.2%｜不確定措辭或 probably_yes 50.7%｜高顯影（portal HU Q4）36.9%｜mild 17.8%｜分割問題 3.8%

**FP（1,295 例）**：L3 脂肪面積最高三分位 54.7%｜not_mentioned 32.6%｜低顯影（portal HU Q1）22.5%｜PV 期 L−S < −25（疑似漏標）20.2%｜肝硬化 / 不均勻 / 肝腫大 11.0%｜報告其他地方提到 steatosis 或 fatty 6.3%｜分割問題 4.6%

## 5. 天花板估計

- 雜訊模型：把局部脂肪浸潤陽性（a = 27.6%）當成「不是瀰漫性脂肪肝」，陰性中的漏標估計為 b ≈ 2.7%（not_mentioned 超出 no 的 L−S < −25 比例，再加 1%）。完美偵測器在現行標籤下的 AUROC ≈ **0.85**。目前 0.819（stack 後 0.840）**已經接近現行標籤的天花板**。
- 乾淨子集（瀰漫型陽性 vs 明確寫 no 的陰性）：E3_late 0.909、stack 0.932。
- 結論：只要標籤定義不改，模型再改良最多只能多拿約 +0.01～0.03。

## 6. 前三名修正（預期收益與成本）

1. **重新定義標籤：把「局部脂肪浸潤 / focal fat」從脂肪肝陽性中移出**（另立一類，或設為 NaN）。用 Qwen 重新判斷 diffuse vs focal，或直接用規則處理（引文含 focal fat 但不含 steatosis / diffuse）。
   - 收益：現行標籤下的評估 AUROC 0.819 → **約 0.90**（評估定義改變）。
   - 收益：訓練時排除這群後，瀰漫型 vs 全部陰性的 AUROC **0.916 → 0.933**（+0.017，模型真的變好）。
   - 成本：低（規則半天；用 Qwen 重標約 1 小時 GPU）。val 和 test 也要套用同一套定義，而且要在看 test 之前鎖定。
2. **讓肝臟 head 看得到參考器官**：把肝臟和脾臟的 embedding 串接成 liver ‖ (liver − spleen)，不要用 finding_organ_map 裡的 mean，因為 mean 沒有幫助（OOF 0.8185）。
   - 收益：E3_late 的 OOF **0.819 → 0.831（+0.012）**；val 描述性 0.818 → 0.823。
   - 成本：低。probe.py 加一種聚合方式（`agg: concat_diff`），embedding 不用重抽。
3. **stack 手工 HU 特徵**：L−S、L/S、肝減門脈 / 肝減主動脈、主動脈 / 門脈 / 腎 HU、肝 HU 百分位、L3 脂肪面積，用 GBM 處理顯影劑強度的非線性。
   - 收益：OOF **0.819 → 0.840（+0.021）**；val 描述性 0.818 → 0.830。和第 2 項一起用是 0.8405，兩者大多重疊。
   - 成本：低。特徵已經算好，每例約 0.3 秒 CPU；test 也要用同一支程式產生（要等 test 解鎖後）。

附帶建議：probably_yes 改用軟標籤（例如 0.7），not_mentioned 陰性降低權重。預期再增加 ≤ +0.01，屬於次要。

## 7. 檔案

A6000 `outputs_final/analysis_steatosis/`：
- `report.md`（本檔）
- `subgroup_auroc.csv`、`model_comparison.csv`、`hu_single_feature_auc.csv`、`errors_at_spec90.csv`、`auroc_cost_drop_subgroup.csv`、`embedding_decodes_liver_hu.csv`、`label_definition_retrain.csv`、`summary.json`
- `hs_a6.csv` + `hs_ct.csv`（每例 HU / 期別 / 脂肪特徵）、`oof_train*.csv`、`per_case_train_oof.csv`
- `scripts/`（hs_labels / hs_hu / hs_oof / hs_oof_spleen / hs_oof_nofocal / hs_analyze）
