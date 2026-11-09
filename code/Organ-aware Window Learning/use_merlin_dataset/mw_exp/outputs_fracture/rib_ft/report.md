# 骨折 · 肋骨分支 rib_ft（2026-11-07）

規則：只用 train/val（labels 經 sitewise.build_frame 先依 split 過濾）；選擇只看 train 5-fold OOF（sitewise 外層 folds）；val 每個最終設定只報一次；test 未讀取。

## 結論
- 最終肋骨分數 R_ft＝**凍結** TS Dataset295 stage 0–1 特徵（1.5 mm 肋骨 ROI、真實 context、baseline＋骨窗）、肋骨×24 角度扇區 instance、attention＋max MIL probe（hid 512、60 epoch、lr 5e-4），5 seed（Spark 3＋3080 2）平均。
- R_ft 單獨：train OOF（肋骨標籤）**.835**（同列 Exp L L_r .794）；val 肋骨 .838 [.802, .872]、只有肋骨 .822（L_r .824／.799）。
- 併入 sitewise（候選依 train CV 選）：**S2+R · lse**（肋骨 site head＝S2 肋骨輸入＋R_ft；合併 lse）train CV **.8706**（官方 S2·lse+any .8626）。
- val **.870 [.852, .886]**，Δ vs .862 **+.008 [+.003, +.013]**；年齡配對 .857（Δ +.009 [+.002, +.016]）；90% 特異度敏感度 66.0%（官方 61.5%）。
- 子群組（Δ vs 官方，配對）：肋骨 .849（**+.020** [+.008, +.033]）、只有肋骨 .811（**+.031** [+.015, +.048]）、椎體壓迫性 .942（+.005 [+.001, +.010]）、骨盆/薦椎/髖 .871（±.000）、急性 .877（−.003 n.s.）、陳舊 .869（+.012）。
- 注意：S2·lse（不含 any）本身 val .865，但 train CV 低於 lse+any，不以 val 改選；S2+R 的增益中約一半來自 lse vs lse+any 的差，另一半來自 R_ft（比較 S2+R·lse+any .866 vs .862：+.004 [+.001, +.007]）。

## 過程（依 train OOF 做的決策）
1. TS encoder 微調（gs15 tf2 配方）：stage-1 稀疏 cache（只存肋骨周圍、其餘補 0）→ 扭曲 InstanceNorm；tf2 fold0 train OOF .62（同列 L_r .70）、過擬合 → 放棄。
2. 凍結特徵 probe（tier0 2,178 列比較）：cache s1 .717 > s1+s2–4 .66 → 細尺度＋真實 context 才有用；feats2（stage 0–3，12 扇區）s0+s1 .740；**feats3（stage 0–1，baseline＋骨窗，24 扇區）.770**；hid/epoch 調整 → .777（選定）。feats4（加 stage 2）因 Spark 04:35 主機 OOM 中斷，未完成、未採用。
3. 全量（train 15,053／val 4,956）5 seed → combine → rib_stack.py。

## 檔案
- A6000 `outputs_fracture/rib_ft/{oof_train.csv,val_scores.csv,summary.json}`（欄位 `ft`；train＝OOF，val＝5 fold 模型平均，seed 間 train-z 後平均）、`stack/{report_auto.md,report_all.csv,summary.json,scores.npz}`、`scripts/`。
- Spark `~/OWL/rib_ft/`（roi 20,009、feats3、runs/final_f3_a）；CT-rate `/media/daikso/Tony/VLM/data/CT/merlin/rib_ft/`（roi、seg15、feats3、runs/final_f3_b）。
