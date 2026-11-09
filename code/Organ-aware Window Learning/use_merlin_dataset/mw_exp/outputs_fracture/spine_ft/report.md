# spine_ft：椎體 crop 微調 TS encoder（11-07）

- 資料：既有 3 mm prep＋TS 3 mm mask 切 T8–L5 每節 crop（20×24×20，GPU 上 ×2 上採樣成 1.5 mm 間距）；全部 20,018 例 train/val（20,015 有 crop）。CT-rate HDD 被 rib_ft 讀滿，故沒有從原始 CT 重切 1.5 mm。
- 模型：TS Dataset297 encoder stage ≥ 2 微調（tf2），3 窗（baseline／骨窗 400/1800／軟組織 40/400），stage3/4 mean+max+遮罩 mean → 每節 logit（＋每節 bias）→ logsumexp MIL。目標＝椎體部位標籤（t_vert；其他部位骨折不訓練）。8 epoch，OneCycle，inner 12% holdout 選 epoch。
- 訓練列：tierA（train 全部骨折＋4,000 隨機陰性，t_vert 已知）；folds＝sitewise 外層 folds。每個 fold 模型對該 fold 的全部 train 列（含非 tier）打分＝OOF；val＝5 fold 模型平均。
- 只跑一個設定（時間限制）：tf2·lse。seed 0：train OOF .927（tier，t_vert）；seed 1：.925。2 seed 平均 → `runs/final_tf2lse/`。
- 參考：J_add 對 t_vert train OOF .907；FT_sp（2 seed）train .930／val .931（t_vert）。
- 進行中：seed 2（`runs/s2_tf2lse`）→ `runs/final3_tf2lse` → final_ft `main_spine3seed`（followup.sh 自動）。
