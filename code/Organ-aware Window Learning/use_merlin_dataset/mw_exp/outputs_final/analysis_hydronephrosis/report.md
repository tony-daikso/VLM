# 腎水腫：錯誤分析（train 5-fold OOF）

日期：2026-11-05｜主機：A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/analysis_hydronephrosis/`｜**test 未讀取、未評分**

## 0. 方法
- 分析資料：train（陽性 1,113／陰性 14,086，排除 unclear）。陽性中 yes 1,072、probably_yes 41；程度：輕度 596、中度 306、重度 113、未註明 98；側別：右 442、左 321、雙側 335、移植腎 9、未註明 6。
- 用 probe.py 的設定跑 5-fold OOF：腎臟單元 [kidney_left, kidney_right]、標準化、class-balanced BCE、lr 1e-3／wd 1e-4（expB 選出的超參數）。early stopping 改用 train fold 內 15% 的 holdout，不看 val。腎臟單元上只有腎水腫一個 finding。每種設定 2 個 seed 取平均。
- 另跑 `*_max`：兩顆腎臟共用 head 各自出 logit（加上 per-unit bias），再取 MIL max（也就是 probe.py 的 `agg: max`），同時得到左右兩側各自的分數。
- 重現檢查（val，只描述）：本流程 E0 0.843、E3_late 0.894；expB 原始結果是 0.838／0.897，兩者一致。
- HU 特徵：從 3 mm prep 的 HU 與 TS total 標籤圖計算，涵蓋 train＋val 全部 20,364 例（A6000 10,518 例，CT-rate CPU 9,846 例）。每側包括：腎遮罩 HU 分位數、腎內／腎竇／腎周 9 mm 殼層中的液體體素（−10～25 HU）、與腎相連的最大液體連通區（體積與前後／左右／上下徑）、>300／>1000 HU 高密度體素（排泄期顯影劑、結石、支架）、TS kidney_cyst 體積、腎臟高度。全域特徵：膀胱體積／HU、主動脈 HU（用來判斷相位）。
- 文字特徵：只用 train／val 報告中與腎、輸尿管相關的句子做關鍵字比對，只保留旗標，不含報告原文。
- 堆疊：與 OOF 共用同一組 fold，做 logistic regression（C=0.3，class-balanced），另有 GBM 版本做對照。
- 信賴區間：AUROC 差值用 paired bootstrap（陽性、陰性分層重抽，2,000 次）。

## 1. 總覽（train OOF AUROC）
| 設定 | AUROC |
|---|---|
| E0（baseline） | 0.846 |
| E1 soft_tissue / liver / narrow_abd / bone | 0.849 / 0.832 / **0.854** / 0.847 |
| E3_late | 0.888 |
| **E0_max（兩腎 MIL max）** | 0.871（+0.025） |
| **E3_late_max（兩腎 MIL max）** | **0.907（+0.020）** |
| 只用 HU 特徵（LR／GBM） | 0.892 / 0.902 |
| 各 window logit 堆疊（LR） | 0.873 |
| E3_late + HU 特徵堆疊 | 0.921（+0.034） |
| **E3_late + E3_late_max + HU 堆疊** | **0.927（+0.039）** |
| E3_late + 年齡／性別 | 0.886（年齡沒有幫助） |

MIL max 增益的 paired bootstrap 95% CI：
| 資料 | 子集 | E3_late → E3_late_max | 增益 [95% CI] |
|---|---|---|---|
| train OOF | 全部 | 0.888 → 0.907 | **+0.020 [+0.013, +0.026]** |
| train OOF | 單側陽性 vs 全部陰性 | 0.878 → 0.904 | +0.027 [+0.018, +0.035] |
| train OOF | 雙側陽性 vs 全部陰性 | 0.919 → 0.920 | +0.001 [−0.008, +0.010] |
| val（描述） | 全部 | 0.894 → 0.922 | **+0.028 [+0.017, +0.040]** |
| val（描述） | 單側陽性 | 0.875 → 0.915 | +0.040 [+0.026, +0.055] |
| val（描述） | 雙側陽性 | 0.933 → 0.942 | +0.009 [−0.004, +0.024] |

（E0 → E0_max：train +0.025 [+0.017, +0.033]，val +0.037 [+0.024, +0.052]。）增益幾乎全部來自單側腎水腫，雙側病例沒有變化，符合「平均稀釋」的機制。

val（描述用，不調參，堆疊模型以 train OOF 訓練）：E3_late 0.894 → E3_late_max 0.922 → E3_late + E3_late_max + HU **0.939**。

90% 特異度下的敏感度：E0 61.6% → E3_late 71.9% → E3_late_max 77.8% → E3_late + HU 80.2% → E3_late + E3_late_max + HU **82.7%**。

## 2. 原因排序（依估計的 AUROC 損失）
| # | 原因 | 證據（train OOF） | 估計損失 |
|---|---|---|---|
| 1 | **左右腎平均（mean pooling）稀釋了單側腎水腫** | 單側陽性佔 68.6%（右 442、左 321）。E3_late 對單側陽性 0.878、雙側 0.919。改成 MIL max 後單側升到 0.904，雙側不變。單側病例中，患側分數高於對側的比例從 0.876 升到 0.936。 | ≈0.020（train）／0.028（val），CI 不含 0 |
| 2 | **擴張的腎盂大多在 TS 腎遮罩外，embedding 看不到完整積液量** | 陽性中最大液體連通區只有中位數 **12%** 落在 TS kidney 遮罩內，其餘在腎竇／腎門（遮罩外）。單一特徵「最大液體連通區體積」AUROC 就有 **0.835**，HU 特徵 LR 0.892，已高於 E3_late。用 embedding 回歸液體量：遮罩內液體比例 R² 0.61–0.86（liver window 最好），最大液體連通區只有 0.46–0.54。HU 堆疊：+0.034（E3_late 之上）／+0.019（E3_late_max 之上）。 | ≈0.02–0.035（HU 堆疊可救回大部分） |
| 3 | **輕度腎水腫本身難分** | 輕度佔陽性 53.5%，E3_late 只有 0.846（中度 0.944、重度 0.963）。最大液體連通區中位數：輕度 126 vox、中度 299、重度 846、陰性 24。90% 特異度下的 FN 有 53% 是「輕度且影像上積液小」（在陽性中只佔 37%）。堆疊後輕度升到 0.890，仍是最大的殘差來源。 | ≈0.03（只能部分救回） |
| 4 | **腎囊腫／腎盂旁囊腫被當成積液** | FP 中 38.5% 是囊腫類（在陰性中只佔 22%）。腎盂旁囊腫陰性在 90% 特異度閾值下有 **46%** 被判陽性；陽性 vs 腎盂旁囊腫陰性 AUROC 只有 0.672。TS kidney_cyst ≥30 vox 的陰性 FP 率 32%。 | ≈0.01 |
| 5 | 標籤雜訊 | probably_yes 佔陽性 3.7%（AUROC 0.779）；引文含 fullness／prominence 4.9%（0.814）、pelviectasis 8.4%（0.802）。陰性中 probably_no 的 FP 率 28%、not_mentioned 22%（全部陰性為 10%）。另有 155 例陰性（1.0%）最大液體連通區 ≥408 vox（中重度陽性的中位數）且無囊腫紀錄，FP 率 35%，可能是漏標。可疑陽性共 10.8%，拿掉後 AUROC 0.888 → 0.897。 | ≈0.009 |
| 6 | 支架／腎造廔 | 有支架的研究中 AUROC 0.909，高於無支架的 0.879；「有支架」旗標單獨 AUROC 只有 0.562，**不是捷徑**。但有腎造廔的陰性 FP 率 62%（n=42）、有支架的陰性 21%：引流裝置旁的殘餘擴張被判陽性，這部分也可能有標籤歧義。 | 小 |
| 7 | 移植腎／腎遮罩缺失 | 移植腎陽性只有 9 例，AUROC 0.63（MIL max 後 0.76）。任一顆腎遮罩 <100 vox：陽性 3.3%、陰性 1.9%，AUROC 0.885，與整體相近。 | ≈0.002 |
| 8 | 顯影相位 | 排泄期 0.902、門脈期 0.888、未打顯影劑 0.878。排泄期 FN 佔 7.7%（陽性中 8.6%），不是問題。 | 可忽略 |
| 9 | 年齡／性別捷徑 | 年齡單獨 AUROC 0.546；年齡＋性別配對後 E3_late 0.885（原本 0.888），≥75 歲最低 0.862。 | 可忽略 |

補充：
- 單一 window：narrow_abd 最好（0.854），liver 最差（0.832）。但從 E3_late 拿掉 liver 掉最多（0.888 → 0.879），liver window 對遮罩內液體的可解碼性也最高（R² 0.86），表示它提供互補資訊。E3_late 的 gate 權重接近平均（baseline 0.20、soft 0.22、liver 0.21、narrow 0.22、bone 0.15）。
- 兩兩組合：baseline+narrow 0.877、+soft 0.874、+liver 0.867、+bone 0.849。不需要更換 window 組合。
- **更正先前的說法**：TS 腎遮罩**不是**把腎竇脂肪包進去。陰性腎臟 p10 中位數是 +100 HU（陽性 +76 HU），p10 < −30 HU 的只佔 0.35%。真正的問題相反：腎竇與擴張的腎盂多半在遮罩**外**（見原因 2）。
- 影像側別與標註側別一致：單側陽性中，較大液體連通區在標註側的比例為 88.5%。

## 3. 90% 特異度下的錯誤分類（E3_late OOF；FN 313、FP 1,409）
**FN**（括號內為該類在全部陽性中的比例）
- D 輕度且影像上積液小：53.4%（36.7%）
- B 標註不確定（probably、fullness、腎外型腎盂）：16.3%（10.3%）
- E 中重度但影像上積液小（遮罩或閾值沒抓到）：14.4%（18.4%）
- C 排泄期（集尿系統被顯影劑填滿）：7.7%（8.6%）
- A 腎遮罩缺失／移植腎：4.8%（4.0%）
- F 影像上有明顯積液但仍漏掉：3.5%（21.8%）→ 有明顯積液的病例模型幾乎都抓得到

**FP**（括號內為該類在全部陰性中的比例）
- b 腎囊腫／腎盂旁囊腫：38.5%（22.2%）
- h 其他：29.0%（53.9%）
- d 支架／腎造廔（含影像高密度）：10.1%（6.0%）
- f 有結石或阻塞字眼，但沒有擴張：8.8%（10.5%）
- a 腎遮罩缺失／單腎：6.5%（1.9%，約 3.4 倍）
- g 腎萎縮／腫瘤：3.8%（3.8%）
- e 影像上有明顯積液（疑似漏標）：2.7%（0.8%）
- c 腎外型腎盂：0.7%（0.8%）

## 4. 上限估計
- 中度＋重度 vs 全部陰性：E3_late 0.949，堆疊 **0.970**，幾乎已到頂。
- 拿掉可疑標籤後：E3_late 0.897，堆疊 0.931。
- 剩下的瓶頸是輕度（佔陽性 54%），目前最好 0.890。若輕度能做到約 0.92，整體約 0.94–0.95。
- 以凍結 encoder、3 mm 影像為前提，實際上限約 **0.93–0.95**（train OOF）。目前最佳堆疊 0.927，已接近下緣。標籤清理只能帶來約 +0.01。

## 5. 建議修正（前三名）
1. **兩腎改用 MIL max**（建議修改 `finding_organ_map.yaml`，本次**未修改**）：
   ```yaml
   hydronephrosis: {organs: [kidney_left, kidney_right], low_contrast: false, control: none, agg: max,
                    sites: {kidney: [kidney_left, kidney_right]}}
   ```
   - 預期增益：train OOF **+0.020 [+0.013, +0.026]**（0.888 → 0.907），val 描述 +0.028 [+0.017, +0.040]。
   - 成本：低。只需重訓 probe，不用重新抽 embedding，還能輸出左右側分數。
2. **加入腎臟液體 HU 特徵，做 late stacking**（最大液體連通區體積與前後徑、腎竇液體、腎內液體比例、左右不對稱、TS kidney_cyst 體積、>1000 HU 高密度體素）。
   - 預期增益：在 MIL max 之上再 +0.019（0.907 → 0.927），val 描述 0.922 → 0.939。
   - 成本：低。特徵已算好；test 只需在鎖定設定後跑一次 CPU 特徵抽取。
3. **擴大腎臟單元，涵蓋腎竇／腎盂**（例如 closing(kidney) 加上腎門，或新增 `renal_pelvis` 單元 = closing(kidney) − kidney），並把腎囊腫分開處理（TS kidney_cyst 當作排除遮罩或額外通道）。目標是輕度擴張與囊腫 FP。
   - 預期增益：約 +0.01～0.02（不確定性高）。
   - 成本：中。需要用 GPU 重新抽取 train／val 的腎臟 embedding。

不建議優先做：更換 window 組合（各 window 都有貢獻，差異小）、依顯影相位分組（沒有差異）、加入年齡／性別（沒有增益）。

## 6. 檔案
- `results_train_oof.csv`：全部子群組 AUROC 與統計
- `per_case_train_oof.csv`：每例 OOF 分數、HU 特徵、FN／FP 類別（只有 train）
- `bootstrap_mil_max_gain.csv`：MIL max 增益的 paired bootstrap CI（train OOF＋val 描述）
- `errors_at_spec90.csv`：90% 特異度下的 FN／FP 類別
- `embedding_decodes_kidney_fluid.csv`：各 window embedding 對液體量的回歸 R²
- `oof_train.csv`、`oof_winsub.csv`（window 子集與 leave-one-out）、`scores_val_descriptive.csv`、`val_winsub.csv`、`val_descriptive.csv`、`gates.json`
- `labels_trval.csv`（只有關鍵字旗標，不含報告原文）、`meta_trval.csv`、`hu_a6.csv`、`hu_ct.csv`、`expB_val_probs.csv`、`kidney_emb_trval.npz`
- 日誌：`oof_main.log`、`winsub.log`、`analyze.log`、`hu_a6.log`、`hu_ct.log`、`labels.log`
- 程式：`scripts/hn_labels.py`、`hn_hu.py`、`hn_dump.py`、`hn_valexpb.py`、`hn_oof.py`、`hn_oof_winsub.py`、`hn_analyze.py`、`hn_boot.py`
