# 快速改進：HU 特徵堆疊（膽結石、脂肪肝）

日期：2026-11-05｜位置：A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/quickwins/`｜**test 未讀取、未評分**（先依 split 過濾 train/val，才讀標籤）

## 方法
- Late stacking：手工 HU 特徵 + E3_late logit。**模型輸入不含年齡／性別**（年齡只用來算年齡配對 AUROC）。
- 候選模型：LR（C = 0.1／0.3／1，class-balanced）與 HistGradientBoosting（3 組設定）。**只用 train 5-fold OOF 挑選**，fold 與 E3_late OOF 相同。
- 最終模型：用全部 train OOF 列重新 fit，套到 val 一次（val 的 E3_late 分數來自以全部 train 訓練的 probe）。
- 膽結石特徵：膽囊最大 HU、p50/p90/p95/p99、std、>60/>100/>130/>200 HU 體素數、最大 >100 HU 連通塊、膽囊體積與是否存在、膽囊壁 shell、肝／主動脈 HU。
- 脂肪肝特徵：肝 HU（中位數、平均、百分位、<40／<0 比例、slab sd）、脾、主動脈、門脈、IVC、腎 HU、L−S、L/S、肝−主動脈、肝−門脈、肝體積、L1／L3 脂肪面積、依 HU 推定的顯影期別（含交互項）。
- 年齡配對 AUROC：只比較同一個 10 歲年齡層內的陽性與陰性配對（依配對數加權）。CI：val 分層 paired bootstrap，2,000 次。

## 結果
| 疾病 | 模型（train CV 選出） | train OOF AUROC | val AUROC | Δval（95% CI） | 年齡配對 AUROC（OOF／val） | 90% 特異度下的敏感度（OOF／val） |
|---|---|---|---|---|---|---|
| 膽結石 | E3_late | 0.782 | 0.773 | — | 0.765／0.751 | 54.3%／49.9% |
| 膽結石 | **E3_late + 膽囊 HU（GBM）** | **0.835** | **0.821** | **+0.048（+0.030～+0.067）** | 0.827／0.809 | 60.8%／59.4% |
| 脂肪肝 | E3_late | 0.819 | 0.818 | — | 0.823／0.818 | 58.4%／56.6% |
| 脂肪肝 | **E3_late + HU（GBM）** | **0.840** | **0.825** | +0.007（−0.002～+0.016） | 0.844／0.824 | 63.1%／58.0% |
| 脂肪肝 | E3_late（肝‖肝−脾 head）+ HU（GBM，選配） | 0.841 | 0.830 | +0.012（+0.003～+0.022） | 0.845／0.831 | 63.0%／59.2% |

- train OOF 的 Δ（vs E3_late）：膽結石 +0.052（+0.042～+0.063）、脂肪肝 +0.021（+0.016～+0.027）、肝‖肝−脾 版 +0.022（+0.016～+0.029）。
- 用 train OOF 定出的 90% 特異度閾值套到 val，敏感度：膽結石 50.5% → 58.3%；脂肪肝 57.6% → 59.7%（主版）／55.1%（肝‖肝−脾 版）。
- 膽結石：GBM（0.835）勝過 LR（0.827），比錯誤分析預期（OOF .827／val .810）更好；提升在年齡配對下仍然存在（+0.06）。
- 脂肪肝：不含年齡／性別時，val 只有 +0.007，CI 稍微跨 0；錯誤分析中的 val .830 包含年齡／性別。依 train OOF，「肝‖肝−脾 head + HU」版本略勝（0.841 vs 0.840），val 0.830，CI 不跨 0，但 test 需要多一個 head 的分數。

## 檔案
- `stack_gs_main.joblib`、`stack_hs_main.joblib`、`stack_hs_concat.joblib`：模型 + 補值中位數 + 欄位順序 + train OOF 的 90% 特異度閾值
- `val_scores_gs.csv`、`val_scores_hs.csv`：每例 val 分數（E3_late、stack、stack_concat）
- `oof_train_gs.csv`、`oof_train_hs.csv`：train OOF 分數；`summary.json`：所有數字（含各候選模型的 CV AUROC）
- `scripts/qw_stack.py`：`fit` 重跑選模與 fit；`apply` 給 test 用：`python qw_stack.py apply <quickwins> gs|hs <hu.csv[,hu2.csv]> <score.csv 含 study_id,E3_late[,E3_late_concat_diff]> <out.csv> [main|concat]`
- `scripts/gs_hu.py`、`scripts/hs_hu.py`：產生 HU 特徵的原始程式（test 要在設定鎖定後才跑）
- 注意：joblib 模型在 Mac 以 scikit-learn 1.9.0 存檔；若 A6000 環境版本不同，可直接 `python qw_stack.py fit <含 gs/、hs/ 與 qw_meta_trval.csv 的資料夾> <out>` 重建（約 2 分鐘，GBM random_state=0 可重現）。
