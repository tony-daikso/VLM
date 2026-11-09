# report_gen/oof：OWL 分類分數（train OOF + val）

給 Merlin 報告生成（方法一、方法四）用的 14 類 OWL 分數。只讀既有 OWL 輸出，不重訓；test 不讀。

- `assemble_scores.py`（在 A6000 執行，radar env）：從 `mw_exp/outputs_final`、`outputs_fracture` 讀各疾病正式最佳模型的
  train 5-fold OOF 分數與 val 分數（全 train refit），組成 `out/scores_trval.csv`，並輸出 `check.csv`（AUROC、覆蓋率）
  與 `sources.json`（每類分數來源）。闌尾炎 REF＋U_rlq_cs 的堆疊分數沒有存檔，依 `rlq_eval.py` 相同規則在 CPU 重算。
- `calibrate.py`：只用 train OOF 做 Platt 校正，輸出 `out/scores_trval_prob.csv`（0–1 機率）與 `calib.json`。

執行：`cd ~/Desktop/Tony/VLM/OWL/report_gen/oof && ~/miniconda3/envs/radar/bin/python assemble_scores.py out && ~/miniconda3/envs/radar/bin/python calibrate.py`
結果檔（含病例 id）只放在 A6000 與 Mac 的 data 端（`data/CT/merlin/report_gen/oof/out/`），不進 git。
