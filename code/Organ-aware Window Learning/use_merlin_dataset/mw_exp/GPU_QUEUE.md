# GPU 工作佇列（使用者最高原則：GPU 不可閒置；從最弱疾病開始提升 AUC）

規則：任何 agent 在某張 GPU 上做完工作、或發現 GPU 閒置，就從下面「待辦」由上往下取第一個適合該主機的項目，標成「進行中（主機、agent、時間）」，做完改「完成」並附結果。
只用 train/val；test 不碰；標籤定義不改（待使用者決定的在 LABEL_DECISIONS_PENDING.md）。
可用 GPU：A6000、Spark（GB10，nnU-Net 慢 5–6×，適合 embedding/小 CNN）、4090（`ssh -p 25966 root@tw-07.access.glows.ai`，24 GB，Azure 串流）。**不要用 CT-rate 3080**（另一個 session 的 cg10k）。

目前 val 最弱（正式整合，2026-11-06 01:30；04:22 實驗 N 更新膽結石／動脈粥狀硬化；10:50 骨折更新）：骨折 .854（原 .847） → 膽結石 .881（原 .866） → 骨質減少 .890 → 腹腔游離氣體 .912 → 腹水 .916 → 膽管擴張 .911 → 動脈粥狀硬化 .921（原 .912；年齡配對 .801）。脂肪肝已拆分（瀰漫 .925；局部 .665 次要）。

## 進行中
- **項目 5 膽管擴張 bd15（1.5 mm 肝門 crop）— A6000 GPU＋CPU，agent，11-06 10:35 起**：driver `outputs_final/biliary_improve/bd15/scripts/run_bd15.sh 7`（bash PID **1323252**，log `bd15/logs/driver.log`）；7 個 crop worker（CPU nice 10，原始 CT 由 CT-rate HDD 以 ssh 唯讀拉取，3 mm 遮罩讀 A6000 本機）＋1 個 GPU 抽取迴圈（extract loop PID 1323386，TS Dataset297 凍結 encoder，約 0.25 s/例）。先做 tier0（10,792＝全部 val＋train 陽性＋年齡×性別配對陰性 1:2）→ 自動中期 probe／stack／report（`stack_tier0/`）→ 再做 tier1 → 全量 probe／stack（`stack/`、`report.md`、`FINAL_DONE`）。實測約 2,100–2,300 例/h（crop 為瓶頸，CT 由 CT-rate HDD 讀取）。**ETA：tier0 約 15:45（中期結果 ~16:15）；全部約 20:15（最終結果 ~21:00）**。停止 `touch outputs_final/biliary_improve/bd15/STOP`；細節見 LONG_RUNNING。
- 11-06 09:05 起 4090 GPU 故障（需使用者重啟/釋放執行個體）；Spark 使用者使用中（不要用）；CT-rate 3080 不要用。

## 待辦（由上往下）
1. ~~膽結石：膽囊高解析 crop 單元~~ → 完成（實驗 N，見下）
2. ~~動脈粥狀硬化：主動脈/髂動脈管壁 crop 單元~~ → 完成（實驗 N，見下）
3. ~~脂肪肝（局部）：肝臟局部區塊單元~~ → 完成（實驗 N，見下）
4. **骨折：肋骨高解析 CNN（實驗 K 改良版）**— 視實驗 L 結果決定，需長時間 GPU（4090 適合）
5. **膽管擴張：1.5 mm 肝門 crop**（以原始 CT 1.5 mm 重做 BD5 單元）→ **進行中（A6000，agent，11-06 10:35 起；見上）**
6. ~~腹水/游離氣體：實驗 M 結果出來後的第二輪~~ → 完成（實驗 O，見下）

## 完成
（agent 完成後在此登記：項目、主機、val 結果、報告路徑）
- **實驗 L：1.5 mm 肋骨（A6000＋4090＋Spark，11-06 00:47–10:16）**：抽取 20,355 例；全 train 5 seed。依 stacker train CV 選 SL_L_repl-rl03-lrc＝LR(expI＋J_add＋L)：train CV .854，val **.851**，vs J add .847 +.004 [+.001, +.006]，年齡配對 .834（+.006 [+.003, +.010]）。L 單獨（repl-rl03-lrc）val .850：肋骨 .798 → .822（+.024 [+.010, +.038]）、只有肋骨 .740 → .774（+.034 [+.015, +.055]）、年齡配對（肋骨）.765 → .796；椎體壓迫性 .932（持平）、骨盆 .847（持平）。報告 A6000 `outputs_fracture/expL/final/{SL_*/report.md,lrep/lreport.md}`。
- **實驗 O：區域 crop 單元第二輪（4090→A6000，11-06 04:35–10:18）**：報告 A6000 `outputs_final/region_units/<unit>_r2/report.md`（Mac 鏡像）。依 train OOF 選，test 未用。
  - 骨折骨盆環（fracture_pelvis_r2）：REF＋U_org_pel，val .847 → **.853**，Δ +.006 [+.002, +.010]；年齡配對 .829 → .837；骨盆/薦椎/髖子群 .848 → .874（+.027 [+.017, +.037]）。
  - 腸阻塞（胃出口／十二指腸，bowel_obstruction_r2）：REF＋U_org，val .956 → .958，Δ +.003 [−.000, +.006] **n.s.**（不更新正式最佳）。
  - 膽管擴張（遠端總膽管／胰頭，bile_duct_r2）：.911 → .911（+.001 n.s.）。腹腔游離氣體（free_air_r2）：.912 → .912（train OOF +.005，val −.000 n.s.）。腹水（ascites_r2）：.916 → .917（+.001，CI 下界 ≈0，不更新）。
- **骨折最終整合（CPU，11-06 10:40–10:50）**：expI＋J_add＋L＋骨盆環單元頭，4 個候選依 stacker train CV 選出 C3＝LR(expI＋J_add＋L＋U_org_pel＋U_pel＋U_org_out)：train CV .8585，val **.854** [.836, .871]，vs .847 +.006 [+.002, +.011]、vs .851 +.003 [−.002, +.007] n.s.；年齡配對 .840；陰性 Spearman .564；子群：椎體壓迫性 .927、肋骨 .823、只有肋骨 .773、骨盆/薦椎/髖 .868、急性 .876、陳舊 .850。（C1＝只加 U_org_pel 的 val .857 較高，但 train CV 差 .0007，不以 val 改選。）**骨折正式最佳 .847 → .854**。報告 A6000 `outputs_fracture/final_combo/report.md`（Mac 鏡像）。
- 骨質減少 × Exp J 1.5 mm 逐節椎體 embedding（CPU：Mac nice 19、4 threads；切片在 A6000 產生，2026-11-06 完成）：依 train OOF 選出新的正式最佳「5 個 window logit＋1.5 mm 4 節串接 head（LR）」，train OOF .904 → .908；val .884 → **.890** [.870, .908]，Δ +.006 [+.001, +.013]；年齡配對 .784 → .798，年齡×性別層內 .776 → .795；陰性 Spearman .729 → .749。單用 1.5 mm 時 OOF .905、val .888；加 HU 沒有幫助。報告：A6000 `outputs_final/osteopenia_v15/report.md`（Mac 鏡像同路徑）。test 未使用。
- **實驗 N 區域 crop 單元（4090 GPU，2026-11-06 02:55–04:22；crop 在 A6000＋CT-rate CPU 產生，nice 19、每台 4 process；4090 每個 consumer 上限 5 GB／2 threads，合計 ≤ 10 GB／4 threads；實驗 L 未受影響；標記 `/root/OWL/EXP_N_RUNNING` 已移除）**。都依 train OOF 選模型，test 未使用。報告在 A6000 `outputs_final/region_units/{gallstones,atherosclerosis,focal_steatosis}/report.md`（Mac 鏡像同路徑）。
  - 膽結石（膽囊／膽囊窩 64³ crop，3 區域，7 窗〔＋結石窗、鈣化窗〕；GS6 遮罩用於 5,034 例缺失病例，其中 794 例改以膽囊為中心）：G5c＋U，train OOF .875 → .879；val .866 → **.881** [.863, .898]，Δ +.016 [+.008, +.024]；年齡配對 .851 → .868；陰性 Spearman .25 → .23；90% 特異度敏感度 64.3% → 68.2%；膽囊遮罩缺失子群 .689 → .776。結石窗／鈣化窗本身無貢獻。
  - 動脈粥狀硬化（腎下主動脈／分叉／兩側總髂 4 個 48³ crop，管壁殼層＋整個 crop，6 窗〔＋鈣化窗〕，加分段管壁鈣化 HU）：REF＋全部單元頭＋分段 HU，train OOF .903 → .910；val .912 → **.921** [.913, .928]，Δ +.009 [+.006, +.012]；**年齡配對 .779 → .801**，Δ +.022 [+.014, +.030]（主要來自分段 HU）；陰性 Spearman .81 → .82；90% 特異度敏感度 72.6% → 73.8%。
  - 局部脂肪浸潤（鐮狀韌帶／膽囊窩／肝門 3 個 48³ crop，3 區域，5 窗）：REF＋全部單元頭，train OOF .680 → .697（Δ +.017，顯著）；val .665 → .676 [.644, .708]，Δ +.012 [−.003, +.026] **n.s.**；年齡配對 .659 → .668；90% 特異度敏感度 28.6% → 25.9%。仍列為次要指標。
  - 註：integrated_best 的 main_all 尚未用這三項重算。crop 保留在 A6000 與 CT-rate 的 `region_units{,_tmp}/<unit>/crops/`（CT-rate 約 19 GB，可刪）。
