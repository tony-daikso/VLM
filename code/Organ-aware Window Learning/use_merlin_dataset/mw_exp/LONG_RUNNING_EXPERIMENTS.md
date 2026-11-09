# 長時間實驗紀錄（2026-11-06 12:25 更新：bd15 暫停、膽結石 gs15 在 A6000＋Mac 執行；L／O 完成）

規則：所有實驗只用 train/val，test 不碰（先依 split 過濾再讀標籤，禁止 `head` 標籤 jsonl）。
主機：A6000 = `ssh A6000`（NVIDIA IGX Orin + RTX A6000，aarch64，12 核），工作目錄 `~/Desktop/Tony/VLM/OWL/mw_exp`；
CT-rate = `ssh CT-rate`（RTX 3080 10 GB），`~/OWL/mw_exp`；Spark = `ssh sparks`（GB10，PyTorch 不支援 sm_121，nnU-Net 類運算慢 5–6×）。



## ✅ push90b（骨折 → val ≥ .90 第二輪，agent push90b，11-09 00:37 – 09:50 完成）
- **結果**：36 個整合變體依 train CV 選 **a6x10w2_tv7**（CV .9024 vs 基準 .9014）；val **.897 [.882, .912]**，年齡配對 .888；vs rp_a6mix3 Δ +.001 [−.002, +.003] n.s.→ 未達 .90。增益來源：後方附件 PE 探針改 t_vert 目標（7 runs×5 seeds，OOF .926）≈+.0007、a6 10 seeds（side .9045）≈+.0002。肋骨 tier：tw_w2 .893 勝出但 full f_w2 side 只 .889（< f_a6 .900）；tw_attn .887、tw_w2sep .883、tw_w2bias .890、tw_bias .888。骨盆 ctx tier 全輸 pc0_base .845（pc1 .818、pc2 .791、pc3 .820）→ 不跑 full。PE any 目標 3 runs 無增益。f_a6_s10 中途停止（邊際 ~.0002、需再 ~4 h）。選擇檔 `push90b/p90b_selection.json`，比較 `push90b/integ/a6x10w2_tv7/compare_vs_897.md`。STOP 已放三台，GPU 全部釋放。
- 基準：push90 rp_a6mix3（train CV .9014，val .897）。所有新輸出在 A6000 `outputs_fracture/push90b/`（不覆蓋 push90／final_ft 的選擇檔）。
- A6000 GPU：`scripts/lane.sh`（2 次重試、連續失敗即停）跑 `rs_probe2_f.py` a6 seed 5–14（`prod/f_a6_s5`、`prod/f_a6_s10`，每 run ~3.5 h）；`rs_probe3.py`（--ctxw bias/attn、--nbw 2、--nbsep）tier 於 `A/runs/tw_*`。thermal guard `scripts/tguard.sh`（>85°C SIGSTOP，≤78°C 繼續；log `logs/guard.log`）。
- CT-rate 3080：`/media/daikso/Tony/VLM/data/CT/merlin/push90b/`（lane.sh＋rs_probe2_f.py；labels `pe/pe_probe_labels_v.csv`＝pe_probe_labels＋train t_vert）：pe_ctx_s10/s15（t_pe）、pe_tv_s0（t_vert）、pe_y_s0（y）。
- Spark：`~/Desktop/Tony/OWL/push90b/`（pv15_probe2.py：--mirw 0＝同一解剖格鏡像、--nonbr、per-fold resume）pelvis tier pc0–pc3；PE feats 副本 `push90b/pe/feats`（2.7 GB）。
- 整合：`scripts/integ_run3.sh TAG …`（final_ft3 --cvonly，只算 train CV）；選定後 `FULL=1` 只跑一次 val；比較 `scripts/cmp_val2.py`。事件 log `logs/push90b.log`。停止：`touch outputs_fracture/push90b/STOP`（各主機同名 STOP）。

## ✅ push90（骨折 → val ≥ .90，agent push90，11-08 17:20–11-09 00:35 完成；無程序在跑，GPU 全部已釋放）
- 結果：選 rp_a6mix3（train CV .9014），val .897 [.881, .911]（vs .892 +.004 顯著），未達 .90。詳見 `outputs_fracture/push90/report.md`。
- 基準 rsP_mix_repl：train CV .8966，val .892 [.876, .907]。只用 train/val；選擇只依 stacker train CV（官方標籤）；test 不讀；官方標籤不改。
- A（肋骨）：`push90/scripts/rs_feat2.py`（rs_feat 同切塊／12 mm 格；4 窗×stage 0–2，cols 欄位描述）→ tier 特徵 `push90/A/feats_tier`（A6000 GPU）→ `rs_probe2.py`（rib_probe3＋--blocks／--nbmode tile〔對側同格鏡像〕／--wcsv）tier 比較（基準 tprobe_tile_sl side .843）。
- B（後方附件）：CT-rate `push90/pe15/scripts/pe15_crop.py`（T8–L5，24×72×80 @1.5 mm，從原始 CT）→ `pe_feat.py`（Dataset292 stage 0–2，base+bone，每節左右×{椎體、椎弓、橫突、棘突} instance）→ rs_probe2（t_pe＝train 報告提到橫突/棘突/pars/椎弓等骨折的陽性 255 例；陰性＝官方 0；其他陽性忽略；側別／節段弱監督）。
- C（雜訊）：audit_fn（268）→ w=0、audit_fp（168）→ w=0.3，只影響訓練；OOF／stacker CV 用官方標籤。Spark rib probe `runs/c_fn0`、`runs/c_fn0fp03`（5 seed，對照 prod/t8_sl_s0 side .8846）；stacker 端 `push90/scripts/final_ft2.py --wcsv`。
- 停止：見 GPU_QUEUE。每個 wrapper 失敗 2 次即停。

## ✅ rib_straight（肋骨拉直 tile＋encoder 微調，agent rib_straight，11-08 07:40–17:20 完成；無程序在跑，GPU 全部已釋放）
- **結果**：微調 tier 失敗（295 .788／297 fold0 .749 ＜ ROI probe .825）；拉直 tile 凍結特徵（`scripts/rs_feat.py` → `feats/`〔12 mm〕、`feats_c6/`〔9 mm〕）＋rib_probe3（`scripts/rs_prod.sh` → `prod/`）全 train t_rib OOF .886；`cand/{rs15,mix15,mix3,rs6}`；final_ft 依 train CV 選 rsP_mix_repl .8966 → val .892 [.876, .907]（vs .882 +.010）。整合腳本 `scripts/rs_integ*.sh`、選擇 `scripts/rs_select.py`（寫 `rib_straight/rs_selection.json`，不動 final_ft/rp_selection.json）。
- 目標：肋骨 site 分數（rib_v3 lev28 t_rib train OOF .859）→ final_ft 整合（rp_both_aa .882）。只用 train/val；選擇只看 train OOF（t_rib）／stacker train CV；test 不讀。
- tile：Spark `~/Desktop/Tony/OWL/rib_straight/scripts/rs_tile.py`（rib_ft/roi 1.5 mm＋Dataset295 標籤 → 每根肋骨：體素圖測地雙掃描求中心線〔碎片 ≤30 mm 橋接〕→ 1.5 mm 弧長重取樣 → 後端在前、兩端外延 8 列 → 32×32 截面〔u＝上方投影、v＝T×u〕三線性取樣；ROI 外＝最近邊緣值）→ `tiles/{id}.npz`（hu、m 肋骨遮罩 packbits、tab〔rib,start,L,m0,n,nvox〕、oor、cl）。
- 訓練：`outputs_fracture/rib_straight/scripts/rs_ft.py`（pv15_ft 改寫；chunk 64 列 stride 48、chunk 數補到 8 的倍數〔固定 cudnn 形狀，否則慢 6×〕；instance＝16 列〔24 mm〕弧段；stage3 mean/max/肋骨遮罩 mean＋stage4 mean/max → LN → linear＋肋骨 bias＋弧位 bias → lse；左右 bag side .5、層級 lev .3；每 epoch ckpt 可續跑；A6000 不用 pin_memory）。`rs_tier.sh`（每 run 最多 2 次）、`rs_guard.sh`（>85°C SIGSTOP rs_ft.py）。
- 停止：A6000 `touch outputs_fracture/rib_straight/runs/STOP`（當前 epoch 後）；Spark tile `touch ~/Desktop/Tony/OWL/rib_straight/STOP`。

## ▶ pelvis15_ft（骨盆 1.5 mm encoder 微調，agent pelvis15_ft，11-08 01:50 起；A6000 GPU＋CT-rate 3080）
- 目標：骨盆/薦椎/髖子群（rp_both_aa val .867）→ 骨盆 site 分數。只用 train/val；選擇只看 train OOF（t_pel）／stacker train CV；test 不讀。
- 程式：`outputs_fracture/pelvis15_ft/scripts/pv15_ft.py`（frac15_ft.py tf2lse 改寫：ROI 固定 176×144×240 裁切/補齊、2 窗 base+bone、stage 0–1 凍結、2–4 微調，instance＝骨（5）×24 mm 格〔x 自中線起算，左右鏡像〕，stage3/4 mean+max → LN → linear＋骨 bias → lse；側別：右 bag／左 bag lse 對 pel_side.csv BCE ×0.5；`--mir` 加對側鏡像格差值；每 epoch ckpt 可續跑；A6000 不用 pin_memory）。`tier_run.sh`（每 run 最多 2 次）、`thermal_guard.sh`（A6000 GPU > 85°C SIGSTOP、≤ 78°C 續）、`tier_cmp.py`（tier 列比較＋堆疊 CV）。
- tier：A6000 `runs/t295s`（Dataset295＋side .5，~2.8 h）；CT-rate `/media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft/`（`scripts/lane_ct.sh`：`t295sm`〔＋mir〕→ `t297s`〔Dataset297〕，bf16）。
- 事先定的判準（tier 列 n=1,804、陽性 304；參考 PV15 tier probe .849、PV15 全量 .876、U_org_pel .881、堆疊 U_org_pel+U_pel+PV15 .898）：(a) FT tier OOF ≥ .872，或 (b) 堆疊 ref3＋FT ≥ ref3 ＋.003 → 才跑全量；否則停止並回報。
- 停止：A6000 `touch outputs_fracture/pelvis15_ft/runs/STOP`；CT-rate `touch /media/daikso/Tony/VLM/data/CT/merlin/pelvis15_ft/runs/STOP`（當前 epoch 後結束；重跑同指令續跑）。
- **06:05 tier 結果（tier 列 n=1,804，陽性 304，1 seed）**：t295s .790、t295sm（＋鏡像差值）.813、**t297s（Dataset297＋side .5）.855**、t297sm .852（folds 3–4 在 3080 跑）。參考 PV15 tier probe .849、PV15 全量 .876、U_org_pel .881。堆疊 CV（U_org_pel＋U_pel＋PV15＝.898）＋t297s → **.903（+.005）**，達判準 (b) → 依 tier OOF 選 **t297s** 跑全量。表 `pelvis15_ft/runs/tier_cmp.json`。
- **06:10 全量（pool＝tierA∩t_pel 已知 4,309 列〔ROI 4,296〕，Dataset297，side .5，2 seed × 5 fold，--no-val）**：`scripts/full_run.sh`；A6000 jobs 0:0 1:0 2:0 3:1（fp32，`roi_pool/`＝roi_tier symlink＋自 CT-rate 拉的 3,560 個）；CT-rate 3080 3:0 4:0 0:1 1:1 2:1（bf16）；Spark 4:1（bf16，`~/Desktop/Tony/OWL/pelvis15_ft/`，pool ROI 21 GB）。log `logs/full.log`。之後 fold 檔＋模型集中到 CT-rate，3080 `--score` 補 val＋非 pool train 列 → 兩 seed 平均 → A6000 `pelvis15_ft/final/` → final_ft 變體（add／取代 FT_pv／取代 PV15／去掉 U_org_pel+U_pel），只依 train CV 選。停止：各主機 `touch <base>/runs/STOP`（Spark base＝`~/Desktop/Tony/OWL/pelvis15_ft`）。

## A6000 cold data → CT-rate（A6000 磁碟清理，11-07 13:50 起；慢慢搬，不趕）
- 目的：A6000 根目錄 879 GB 只剩 ~17–19 GB → 把冷資料用網路（A6000 直接 ssh CT-rate，不經 Mac、**不用 A6000 USB Passport**）複製到 CT-rate `/media/daikso/Tony/A6000/<A6000 home 下相同相對路徑>`（ext4），驗證後才刪 A6000 原檔。
- 程式（A6000）：`~/cold_move/move_cold.sh`＋`~/cold_move/lst.py`（CT-rate 端副本 `/media/daikso/Tony/A6000/_tools/lst.py`），清單 `~/cold_move/items.txt`（依序：radar_tok_data → ctrate_radar → model_polyp〔→ CT-rate `model_polyp_home`〕→ radar_hu_work、radar_tok_work、radar_work、radar_stress、RADAR_ON_MERLIN、ctrate_radar_bench、ctrate_radar_bench_fast、ctrate_par8、ctrate_prof、CT-rate → outputs_final 下 focal_steatosis_cnn/crops、gallstones_improve/gs5/crops、gallstones_improve/gs6/post/crops、biliary_improve/crops、biliary_improve/bd15/crops）。
- 溫和設定：一次一項；rsync `--bwlimit=20000`（~20 MB/s）；兩端 `nice -n 19 ionice -c3`；每 30 s 檢查，A6000 GPU > 80°C 或 load1 > 20 就停掉 rsync，等到 ≤ 77°C 且 load1 ≤ 18 再續（`--partial-dir` 接續）。不碰 spine15 訓練、CT_AI_report、ollama、~/OWL、他人目錄、conda、dotfiles。
- 每項流程：/proc 掃描確認沒有程序使用（cwd/fd/mmap/cmdline）＋30 分鐘內無修改 → rsync → 驗證（兩端逐檔 type/size/mtime 清單完全相同＋檔數／總 bytes〔python 64-bit〕＋`rsync --dry-run` 0 變更＋隨機 64 檔 md5 相同）→ 原位置上層寫 `MOVED_TO_CTRATE.txt`（例：`~/MOVED_TO_CTRATE.txt`、`outputs_final/focal_steatosis_cnn/MOVED_TO_CTRATE.txt`）→ 再確認沒人使用 → 才刪 A6000 原檔。驗證失敗或使用中 → 保留原檔、該項跳過（`state/<id>.skipped`）。
- 狀態：`tail ~/cold_move/move.log`；`cat ~/cold_move/heartbeat`（每 30 s 更新，含 GPU 溫度／load／剩餘空間）；`ls ~/cold_move/state`（`<id>.copied/.verified/.deleted/.skipped`）；全部完成後 `~/cold_move/SUMMARY.txt`（含開始前／完成後剩餘空間）。rsync 輸出 `~/cold_move/rsync_<id>.log`。
- 停止：A6000 `touch ~/cold_move/STOP`（30 s 內乾淨停止，部分檔保留）。
- 續跑：`rm -f ~/cold_move/STOP; nohup setsid nice -n 19 ionice -c3 bash ~/cold_move/move_cold.sh >> ~/cold_move/run.out 2>&1 < /dev/null &`（已刪／已驗證的項目自動跳過；flock 防重複執行）。要重試被跳過的項目：刪掉 `~/cold_move/state/<id>.skipped` 再續跑。
- 當機偵測：腳本不會開機自動重啟；若 A6000 再當機，下次啟動時 move.log 會記 `WARNING: previous run did not exit cleanly…` 與最後 heartbeat 時間。**A6000 若當機，先回報時間再決定是否續跑。**
- **11-07 21:00 第二輪（使用者：「目前用不到的資料都搬到 CT-rate Tony」）**：主機 A6000（nohup，`~/cold_move/move_cold.sh`，同樣 bwlimit 20 MB/s＋GPU > 80°C 暫停）；停止 `touch ~/cold_move/STOP`；log `~/cold_move/move.log`（狀態 `heartbeat`、`state/`、`SUMMARY.txt`）。腳本修正：items 改用 fd 3 讀（之前 do_item 裡的 ssh 吃掉了 items.txt 其餘行，所以第一輪只搬了 radar_tok_data）；「近期修改」門檻改 24 h；支援單一檔案；每項在原位置留 `<name>.MOVED_TO_CT-rate.txt`。清單（依大小）：ctrate_radar、model_polyp/hn_mining、models/Qwen3.8-27B-Q8_0.gguf、thyroid、Carotid_Plaque、outputs_final 的 region_units/*/crops、biliary_improve/crops、gs5/crops、focal_steatosis_cnn/crops、gs6/post/crops、mw_exp/outputs/ts_engine_old、~/model_polyp、radar_*／ctrate_* 小目錄。**刻意不搬**：gallstones15/crops（CNN_full_FAILED、REQ_full 未結）、bd15/crops（暫停待續）、cg10k_nc（今天有修改）、cg10k_best（cg10k_nc 腳本會讀）、mw_exp/outputs 其餘（prep/masks/emb* 等仍被腳本引用）、models/Qwen3.8-27B-FP8（vLLM 用）、outputs_fracture/*。

## ✅ fracture_rp（骨折：肋骨 rib_v3＋骨盆 pelvis15 生產與整合，agent fracture_rp，11-07 23:00–11-08 01:45 完成；無程序在跑，GPU 已釋放）
- **結果**：pelvis15 20 seed t_pel OOF .872（`pelvis15/final_m400`）；依 train CV 選 final_ft `rp_both_aa`（train CV .8871），val .882 [.866, .898]，vs .878 +.005 [+.001, +.009]；年齡配對 .872；骨盆子群 .867 未改善。報告 `final_ft/rp_both_aa/report.md`。
- 修復：CT-rate `run_feat_full.sh 292/295`（PID 3132438／3196122）因 21:35 磁碟重整後 cwd 變成 `/media/daikso/Tony/Tony/...` 每 120 s 失敗 → 已 kill（含 sleep 子程序）。`f_292`／`f_295` 全部 19,973 檔載入驗證 0 壞檔（`pelvis15/scripts/pv15_validate.py`，`logs/validate_*.txt`），已補 `SPARK_DONE`、`f_*/FULL_DONE`。全部 20,018 id = 19,973 roi＋45 fewvox，crop 完整。CT-rate 無其他程序 cwd 在 /media/daikso/Tony/Tony。Spark `spark_watch.sh`（buf/ 已空、閒置）以 `buf/DONE` 正常結束。A6000 `feat_s3.sh`／`feat_s3_295.sh` 因 <ID redacted> OOM 無限重試（1803/1804 已完成）→ 已停，補 `DONE3`。
- pelvis15 tier 補測（A6000 GPU，`outputs_fracture/pelvis15/scripts/tier_more.sh`）：295 side .5：maxinst 400 .849／600 .841／800 .817；stage 0–3＋m400 .848（無增益）；side 1.0＋m400 .850（≈）。→ 生產用 295 stage 0–1＋side .5＋maxinst 400。
- pelvis15 生產（全 train 5-fold OOF＋val fold 平均，各 5 seed）：Spark `~/Desktop/Tony/OWL/pelvis15/scripts/prod_sp.sh 400 0`、CT-rate 3080 `pelvis15/scripts/prod_ct.sh 400 10`、A6000 `outputs_fracture/pelvis15/scripts/prod_a6.sh`（seed 20；f_295 已從 CT-rate 拉到 A6000）。run 名 `runs/p295s_m400_s{0,10,20}`。停止：各主機 pelvis15 目錄 `touch STOP_PROD`＋`pkill -f pv15_probe.py`。合併 → A6000 `outputs_fracture/pelvis15/final_m400/`。
- rib_v3 生產池：依 train OOF（t_rib，ft_side）選 lev28＝v3_side_lev＋v3_sl_s10..40＋v3_s1l_s50（28 seed）.8593（sl23 .8591、side44 .8582、pool1 .8548）→ `outputs_fracture/rib_v3/cand/lev28/`。
- 整合（A6000 CPU）：`final_ft/scripts/rp_rib.sh`（rp_rv3add／rp_rv3repl）、`final_ft/scripts/rp_pel.sh`（rp_pv15*、rp_both_*），變體間只依 train CV 選。

## rib_v3（肋骨第三版，agent rib_v3/pelvis15，11-07 14:45 起；Spark GPU）
- 目標：肋骨 site 分數（R_ft，train OOF .835）再提升 → final_ft 整合。只用 train/val；選擇只看 train OOF（t_rib，sitewise 外層 folds）；val 每個最終設定只報一次。
- 弱標籤：A6000 `outputs_fracture/rib_v3/scripts/rib_side_labels.py` → `rib_v3/rib_side.csv`（只讀 train 報告；val/test 報告載入後立即丟棄、不檢視）：train 肋骨骨折 505 例中 463 例有側別（左 181／右 185／雙側 97）、310 例有肋骨層級。
- 模型：`rib_v3/scripts/rib_probe3.py`（rib_probe2＋`--ctx` 對側鏡像扇區±1／相鄰肋骨／相鄰扇區的 hidden 差值；`--side` 左右 bag 弱監督；`--lev` 肋骨層級 ±1 的 instance max 弱監督；`--ema`）。每個 run 同時輸出 ft_head／ft_side／ft_mix 三種分數。
- Spark `~/Desktop/Tony/OWL/rib_v3/`（feats3、labels.csv 為 rib_ft 的 symlink）：`scripts/abl_v3.sh`（v3_base／v3_side／v3_ctx／v3_ctx_side／v3_ctx_side_lev／v3_ctx_side_ema，各 3 seed，同時 3 個；log `logs/<run>.log`，摘要 `logs/abl_summary.txt`）。停止 `touch ~/Desktop/Tony/OWL/rib_v3/STOP`（目前 run 跑完才停；要立刻停 `pkill -f rib_probe3.py`）。
- 之後：依 train OOF 選設定 → 多 seed 全量 → combine → A6000 `outputs_fracture/rib_v3/{oof_train,val_scores}.csv` → final_ft。
- **17:25 消融結果（全 train 5-fold OOF，t_rib，3 seed，同 seed）**：base .820；ctxin（輸入空間對比）.822（無用）；ctxin+side+lev .814；**side 0.5 .849（side 分數）／side 1 .851／side 2 .844／side 0.5+lev 0.3 .854（最佳）**；side 0.5 5 seed（s10）.852。舊 R_ft（5 seed）.835、15 seed 合併 .839。→ 對側鏡像對比無用，**報告側別弱監督是主要增益**。train CV 篩選（`rib_v3/scripts/cv_only.py`，只算 train）：S3＋side1(3 seed) .8791 → .8815。
- 17:25 起生產 seed：A6000 `scripts/seeds_a6.sh`（v3_sl_s10/20/30/40＝side .5＋lev .3，各 5 seed，2 lane）；Spark `seeds_side.sh`（v3_side_s10/s20）→ `seeds_sp2.sh`（v3_s1l_s50＝side 1＋lev .3）。合併用 `scripts/combine3.py`（欄位 ft_side）。

## pelvis15（真 1.5 mm 骨盆 ROI＋凍結多 stage 特徵 MIL，agent rib_v3/pelvis15，11-07 14:46 起；CT-rate CPU＋3080）
- ① crop（CT-rate CPU）：`/media/daikso/Tony/VLM/data/CT/merlin/pelvis15/`：`scripts/run_pv.sh`（driver；spine15_crop 還在跑時 4 proc，結束後自動改 12 proc；nice 19、ionice c2/7）→ `scripts/pv15_crop.py`：原始 CT 直接三線性取樣 1.5 mm、世界座標軸對齊的骨盆 ROI（兩側髖骨＋薦椎＋股骨上 80 mm 的 bbox＋12 mm），標籤取 TS 3 mm 遮罩最近鄰後向外長 3 mm（涵蓋皮質）。輸出 `roi/{id}.npz`、`status.tsv`、`logs/{driver,crop}.log`。順序 `ids_order.txt`＝train 骨盆陽性 309 → 1,500 陰性（tier）→ 其他部位骨折 → val → 其餘 train。
- ② encoder 比較（CT-rate 3080，≤6 GB）：`scripts/run_feat_tier.sh` → `scripts/pv15_feat.py`（凍結 stage 0–1，baseline＋骨窗，instance＝骨頭×18 mm 格，x 以中線計），Dataset292／294（從 A6000 複製到 `/media/daikso/Tony/VLM/models/ts/`）／295 → `f_{292,294,295}/`；tier 完成時 `TIER_FEAT_DONE`。
- ③ probe：`scripts/pv15_probe.py`（MIL＋`--ctx` 左右鏡像格／6 鄰格差值；目標 t_pel）；tier 依 train OOF 選 encoder／設定 → 全量特徵 → 全量 OOF → final_ft `--add PV15=…@pel`。
- **17:25 tier 結果（1,804 列，t_pel，5-fold OOF，3 seed）**：參考 U_org_pel .877；凍結 stage 0–1：292 .792／294 .800／295 .795；加報告側別弱監督（`pel_side.csv`，train 報告 309 例中 250 例有側別）：292 .786／294 .795／**295 .829**。與 U_org_pel＋U_pel 的 LR 堆疊（tier CV）：.878 → **.886（295+side）**。→ 選 Dataset295。3080 全量抽取改為 `scripts/run_feat_full.sh 295`（292 全量已停）。Spark 由 Mac relay（`outputs_fracture/pelvis15/scripts/relay_pv.sh`）＋`spark_watch.sh` 切 crop 並每 5 分推回 CT-rate roi。A6000 另測 stage 0–3（f_292s3）與 maxinst 400。
- 注意：CT-rate 全量抽取迴圈的結束條件需要 `CROP_DRIVER_DONE` 與 `SPARK_DONE`（Spark crop 完成後手動 touch 到 CT-rate pelvis15/）。
- 停止：`touch …/pelvis15/STOP_ALL`（crop）＋`touch …/pelvis15/STOP_FT`（特徵）。續跑：重跑 `scripts/run_pv.sh`（已完成 roi 跳過）。

## ✅ spine15_ft（真 1.5 mm 椎體分支，agent spine15，11-07 11:10–15:55 完成；無程序在跑，A6000 GPU／3080 已釋放）
- **結果**：依 train OOF 選 tf2lse（tier t_vert .9382；tf2ctx .9314；3 mm spine_ft 同列 3 seed .934）；單 seed。val 椎體（t_vert）.939 [.922, .954]，vs 3 mm spine_ft +.008 [+.001, +.017]。整合（依 train CV）：final_ft `spine15_add`＝S3a·lse+any，train CV .8824（with_rib .8788），val **.877 [.861, .893]**；vs .862 +.016 [+.009, +.022]；vs .872（main）+.005 [+.001, +.010]；vs .878（with_rib）−.000 [−.003, +.002] n.s.；年齡配對 .865；椎體壓迫性 .955（vs with_rib +.004 [+.002, +.007]），肋骨 .840（−.007）、骨盆 .865（−.009）。報告 `spine15_ft/{report.md,integration.md}`、`final_ft/spine15_{add,repl}/report.md`（Mac 鏡像同路徑）。
- 留存：CT-rate `frac15_raw/crops/`（20,015 例 1.5 mm 椎體 crop，20 GB）＋`runs/`（fold 模型）；A6000 `spine15_ft/runs/s0_*`（fold 模型 636 MB），A6000 上的 crop 副本已刪。
- （以下為執行過程紀錄）
spine15_ft（真 1.5 mm 椎體分支，agent spine15，11-07 11:10 起；crop＝CT-rate CPU＋Mac Passport 中繼，訓練＝A6000 GPU fp32）
- 目標：以原始 CT 直接切 1.5 mm 逐節椎體 crop（T8–L5，40×48×40，同 spine_ft 視野/方向，不再 3 mm ×2 上採樣）→ TS Dataset297 stage ≥2 微調（spine_ft 配方）→ `outputs_fracture/spine15_ft/{oof_train.csv,val_scores.csv,summary.json,report.md}`（欄位 ft）→ final_ft 整合。只用 train/val；選擇只看 train OOF（tier 列 t_vert，sitewise 外層 folds）。
- ① crop（CT-rate）：`/media/daikso/Tony/VLM/data/CT/merlin/frac15_raw/`：`run_crop.sh`（`scripts/spine15_crop.py`，12 proc，nice 10，讀 CTDATA FUSE，依 ids_prio.txt 正向：tierA→val→其餘 train）＋`scripts/crop_buf.sh`（4 proc，吃 `buf/` 的中繼 CT，切完即刪）。中心／遮罩＝Exp J seg15（Dataset292 1.5 mm，`vert15_seg15/`）優先，缺者用 TS 3 mm 遮罩（`~/OWL/mw_exp/outputs/masks`＋`frac15_raw/masks3`〔自 A6000 複製 8,853 個〕）；另存 m3（3 mm 遮罩版）。輸出 `crops/{id}.npz`、`status.tsv`、`logs/crop*.log`。
  Mac 中繼（只轉送不運算）：`outputs_fracture/spine15_ft/scripts/relay_passport.sh`（Passport → CT-rate buf，順序 relay_order.txt＝tierA 反向→其餘反向，buf 上限 12；log `outputs_fracture/spine15_ft/relay.log`）。
- ② A6000 `outputs_fracture/spine15_ft/`：`scripts/sync_crops.sh`（rsync 拉 tierA crop）；driver `scripts/run_spine15.sh`（log `logs/driver.log`）：等 tierA crop 齊 → 兩設定同時跑 seed 0（tf2lse；tf2ctx＝加相鄰椎體差值特徵），只對 held-out tier 列打分、存 fold 模型、不評 val → 依 train OOF 選 → `runs/best_cfg.txt` → seed 1、2 同時跑 → `runs/TRAIN_DONE`。每個 job VRAM ≤19 GB（實測 ~11.5 GB），fp32。
- ③ 之後：`frac15_ft.py --score` 以存下的 fold 模型補打 val＋非 tier train 列（A6000 分批搬 crop 或 CT-rate 3080）→ `final_<best>`（3 seed 平均）→ 複製到 `spine15_ft/` → final_ft 整合（加 spine15 頭＋rib 頭）。
- 停止：CT-rate `touch $R/STOP $R/STOP_BUF`（R＝frac15_raw）；Mac `touch outputs_fracture/spine15_ft/scripts/STOP_RELAY`；A6000 `touch outputs_fracture/spine15_ft/runs/STOP`（當前 epoch 結束）＋`touch outputs_fracture/spine15_ft/STOP_SYNC`。續跑：重跑同一指令（已完成 crop／fold／run 自動跳過）。
- 12:20 調整：A6000 GPU 另有 e292 spine FT（~19 GB）＋gs6 seg，故 A6000 一次只跑 1 個 spine15 job（s0_tf2lse → 選完後 s1/s2）；tf2ctx 改在 CT-rate 3080（`frac15_raw/scripts/run_ct.sh`，bf16、accum 4、VRAM ≤5.5 GB，與 rib_ft2 並存；log `frac15_raw/logs/s0_tf2ctx.log`），A6000 driver 完成後自動拉回比較。之後 `scripts/score_fill.sh`（A6000，已在等 TRAIN_DONE）分批補打 val／非 tier 列並寫出 `spine15_ft/{oof_train,val_scores,summary}`。
- **14:45 收尾（使用者改方向：專注肋骨／骨盆；協調者指示不跑額外 seed）**：3080 的 s1_tf2ctx 已取消；A6000 `scripts/finish_a6.sh`（PID 33636，不用 GPU）等 s0_tf2lse 跑完（約 15:20，之後 A6000 GPU 釋放）→ 依 train OOF 在 tf2lse／tf2ctx（s0 .9314）之間選 → 把選出的 run 推到 CT-rate → CT-rate `scripts/score_ct.sh <best>`（等 crop 全部完成、停止 crop worker，3080 補打 val＋非 tier 列，完成即釋放 3080）→ A6000 拉回 `spine15_ft/{oof_train,val_scores,summary}.json` → `spine15_report.py`＋final_ft `--tag spine15_add`（S2＋FT＋rib＋spine15 頭）／`spine15_repl`（spine15 取代 3 mm FT_sp）→ `runs/INTEGRATION_DONE`。單 seed（協調者指示）。
- 13:02／13:16 A6000 兩次當機重開（協調者判斷與 A6000 上 USB exFAT rsync 同時發生）。13:31 起：frac15_ft.py 每 epoch 存 `runs/<run>/ckpt_f{k}_s{sd}.pt`（fold 完成即刪），重跑 driver 自動從最後 epoch 接續；協調者已設 A6000 GPU 硬體功率上限 200 W（重開機會重置，重開後請通知協調者重設）；driver 用 2 個 dataloader worker、`--throttle 0`。CT-rate 3080 另排 `scripts/run_ct2.sh tf2ctx 1`（s0_tf2ctx 完成後預先跑 seed 1；若最後選 tf2lse 則作廢），A6000 driver 遇到已存在的 s1/s2 summary 會跳過。
- 12:20 A6000 磁碟（協調者要求）：`spine_ft/in/pack_{a6,ct}.npz`（11.6 GB）已刪，md5 驗證過的副本在 CT-rate `/media/daikso/Tony/VLM/data/CT/merlin/frac15_crops/`（說明 `spine_ft/in/PACKS_MOVED.txt`）；e292 job 已載入 RAM 不受影響，若要重跑先 rsync 回來。A6000 剩 23 GB。

## spine_ft / pelvis_ft / final_ft（A6000 GPU，11-07 01:40 起；主體 08:07 完成，val .872；followup.sh PID 451046 仍在跑 spine seed 2＋等 rib_ft）
- crop：`spine_ft/scripts/crop_prep.py`（3 mm prep＋TS 3 mm mask，A6000 本機 10,337 → `spine_ft/in/pack_a6.npz`；CT-rate `/media/daikso/Tony/VLM/data/CT/merlin/frac15_crops/pack_ct.npz` 9,681 → 複製到 A6000 `spine_ft/in/pack_ct.npz`）。labels/folds：`spine_ft/in/labels.csv`（sitewise 外層 folds；tierA＝train 全部骨折＋4,000 隨機陰性）。
- 訓練：`spine_ft/scripts/run_branch.sh spine|pelvis "name=args;…"`（tune 1 seed train-only OOF → 依 train OOF 選 → final 2 seed＋val）。輸出 `{spine,pelvis}_ft/runs/{tune,final}_*/{oof_train.csv,val_scores.csv,summary.json}`。
- 停止：`touch outputs_fracture/{spine,pelvis}_ft/runs/STOP`（當前 epoch 結束後退出）；續跑：重跑 driver（已完成 fold 由 foldK.npz 接續）。
- 整合：`final_ft/scripts/final_ft.py`（A6000 CPU）→ `final_ft/main/report.md`；rib_ft 出現（`rib_ft/runs/best_cfg.txt`＋final 或 `rib_ft/oof_train.csv`）後以 `--tag with_rib` 重跑。


## rib_ft（肋骨，11-07 01:45–11:20，已完成；本 agent 的行程都已結束）
- 目標：肋骨骨折 site 分數 → 併入 sitewise S2（.862）。只用 train/val；選擇只看 train OOF（sitewise 外層 folds）。
- labels：A6000 `outputs_fracture/rib_ft/labels.csv`（`scripts/prep_labels.py`，經 sitewise.build_frame，先 split 過濾）。
- ① ROI prep（CPU）：CT-rate `/media/daikso/Tony/VLM/data/CT/merlin/rib_ft/`：`scripts/rib_prep.py ids_order.txt . --jobs 13`（PID 1627282，log `logs/prep.log`；原始 CT 在 CTDATA FUSE，約 1 例/s）→ `roi/{id}.npz`（Exp L 1.5 mm 肋骨 ROI＋Dataset295 標籤）。
  Spark `~/OWL/rib_ft/`：`rib_prep.py buf . --watch --jobs 10`（log `logs/prep_watch.log`）吃 Mac 中繼送來的 CT；Mac `outputs_fracture/rib_ft/scripts/relay_passport.sh`（Passport → Spark buf，從清單尾端往前；log `outputs_fracture/rib_ft/relay.log`）。
  同步：Spark `sync_roi.sh`（roi 雙向）、`sync_cache.sh`（cache Spark → CT-rate）。
- ② stage-1 cache（GPU）：Spark `scripts/rib_cache.py roi cache --ids ids_order.txt`（log `logs/cache.log`；凍結 stage 0–1 baseline 窗，只存肋骨周圍 12 mm，約 16 MB/例）。
- ③ 訓練：`scripts/run_tune.sh`（Spark：tf2e6、tf2lr3；3080：tf3e6）→ `runs/<cfg>/oof_fold*.csv`、`model_fold*.pt`；之後 `rib_ft.py --score` 產生全 train fold／val 分數。
- **03:20 改向**：cache 模式（stage-1 只存肋骨周圍、其餘補 0）會扭曲 stage 2–4 的 InstanceNorm，微調 tf2 fold0 train OOF 只有 .62（同列 L_r .70），凍結 probe 只用 stage-1 .72、加 stage 2–4 反而 .66 → 停止 cache／微調。
  改用 `scripts/rib_feat2.py`（完整 ROI、真實 context，Dataset295 stage 0–3 凍結，肋骨×12 角度扇區 instance 的 mean/max）→ `feats2/{id}.npz`；Spark part 0/3（log `logs/feat2_0.log`），3080 part 1/3＋2/3（CT-rate `logs/feat2_{1,2}.log`）；Spark `sync_feat2.sh` 每 2.5 分把 CT-rate feats2 拉到 Spark（停止 `touch ~/OWL/rib_ft/STOP_SYNC_F2`）。
  **03:55 再改**：tier0 比較 feats2 s0+s1 .740 → `scripts/rib_feat3.py --maxst 1 --bone --nsec 24`（stage 0–1、baseline＋骨窗、24 扇區）.770（同列 L_r .727）→ 全量改抽 feats3：Spark part 0/3（`logs/feat3_0.log`），3080 part 1/3、2/3（CT-rate `logs/feat3_{1,2}.log`）；feats2 已停；Spark `sync_feat3.sh` 拉回（停止 `touch ~/OWL/rib_ft/STOP_SYNC_F3`）。
  **10:25 狀態**：ROI prep 完成（20,009）；Mac relay 已停（STOP_RELAY）。Spark 04:35 主機 OOM（unified memory）殺掉所有 rib_ft 行程（feat3 part0、prep watch、sync）。3080 兩個 feat3 行程共用 10 GB 有 CUDA OOM → 已停，改單行程補抽：Spark `roi_miss/`（`logs/feat3_sweep.log`）、CT-rate `roi_miss/`（--vram-gb 9.3）；`sync_feat3.sh` 已重啟。feats4（加 stage 2）測試只做到 1,749/2,178，未採用。
  probe 設定依 tier0 train OOF 選：feats3＋`--hid 512 --epochs 60 --lr 5e-4`（.777；L_r 同列 .727）。
  模型：`scripts/rib_probe2.py`（instance MIL，train 5-fold OOF）→ `combine.py` → A6000 `outputs_fracture/rib_ft/` → `rib_stack.py`。
- 停止：Spark `touch ~/OWL/rib_ft/runs/STOP ~/OWL/rib_ft/STOP ~/OWL/rib_ft/cache/STOP ~/OWL/rib_ft/STOP_SYNC`；CT-rate `touch …/rib_ft/STOP …/rib_ft/runs/STOP`；Mac `touch outputs_fracture/rib_ft/scripts/STOP_RELAY`。續跑：重跑同一指令（已完成 roi／cache／fold 自動跳過）。


## fracture_round2（骨折第二輪，agent fracture_round2，11-07 11:10 起；目標 val ≥ .90，只依 train CV 選）
- 整合（A6000 CPU）：`outputs_fracture/final_ft/scripts/final_ft.py`（11:30 擴充：`--add NAME=DIR@site` 可加額外部位頭；新增候選 S3n/S3na＝不含 FT_pv；原檔備份 `final_ft.py.bak_1107`）。
  已完成：`final_ft/with_rib`（2-seed spine）、`final_ft/with_rib_spine3seed`（3-seed spine＋R_ft）→ 依 train CV 選 S3·lse：train CV .8791，val **.878 [.861, .893]**，Δ vs .862 +.016 [+.010, +.022]。
- A6000 GPU（≤ 20 GB）：enc292（Dataset292）spine FT 試跑 75 分鐘連 1 個 epoch 都沒完成（>15× 慢於 297），12:40 已停、放棄。
  12:31 想改跑 spine tf2·lse seed 3，但 3 mm pack 已被 spine15 agent 移到 CT-rate（A6000 磁碟），未啟動；A6000 GPU 讓給 spine15（1.5 mm 原始 crop）訓練。本 agent 在 A6000 只做 CPU 整合。
- CT-rate 3080：rib_ft2 seed 1 跑到 fold 3（OOF .743/.758/.790/.800）後 **14:05 停止**：Spark rib_ft2 seed 0 全 5 fold train OOF（t_rib）只有 .746，遠低於凍結 probe R_ft .835；z 平均 w=.2/.3/.5 → .833/.830/.817（都更差）→ 端到端微調放棄。3080 現由 spine15 agent 的 frac15_ft.py 使用。（Spark main_full 沒有產生 val_scores_seed0.csv，FT2_DONE 仍被 touch；未修。）
- Spark GPU（14:10 起）：肋骨 probe 特徵 stage 0–2（feats4，1,749 tier0）vs stage 0–1（feats3，同 1,749 列）配對比較：`~/OWL/rib_ft/scripts/abl_f4_r2.sh`（log `~/OWL/rib_ft/logs/r2_abl_f4.log`，輸出 `runs/r2_t0_{f3m,f4}`）。若 stage 2 依 train OOF 明顯較好 → 全量抽 feats4（Spark＋3080）。停止：kill abl_f4_r2.sh。
- 注意：CT-rate `frac15_raw/`（spine15_crop.py，原始 CT 真 1.5 mm 椎體 crop，12 procs，11:10 起）**不是本 agent 啟動**（另一個 agent），本 agent 不碰；其輸出完成後會納入整合。

## rib_ft2（肋骨 encoder 端到端微調，Spark GPU，11-07 11:15 起，agent 骨折肋骨）
- 不用 3080（已交給 spine agent）。資料：Spark 本機 `~/OWL/rib_ft/roi`（20,009，真實 context）＋`~/OWL/rib_ft/feats3`（凍結 stage 0–1 instance 特徵）。
- 模型：`~/OWL/rib_ft2/scripts/rib_ft2.py`：骨窗，stage 0–1 凍結、stage 2–3 訓練，肋骨×24 扇區 instance 的 s2/s3 mean+max ＋ feats3 向量 → attention＋max MIL；3 epoch、陰性 3×；sitewise folds。實測 0.33 s/例（訓練）、0.19 s/例（推論），每 fold 約 45 分。
- driver：`scripts/run_ft2.sh`（5 fold OOF → full-train 模型 → val），log `~/OWL/rib_ft2/logs/{main_folds,main_full}.log`，輸出 `runs/main/{oof_fold*.csv,val_scores_seed0.csv,FT2_DONE}`。ETA 約 16:30。
- 停止：`touch ~/OWL/rib_ft2/runs/main/STOP`（當前 epoch 結束）；續跑：重跑 `scripts/run_ft2.sh`（完成的 fold 跳過）。
- 自動收尾：Mac `outputs_fracture/rib_ft/scripts/relay_ft2.sh`（只做中繼，log `outputs_fracture/rib_ft/relay_ft2.log`）等 Spark `FT2_DONE` → Spark `post_ft2.py`（依 train OOF 在 probe／ft2／平均 中選）→ A6000 `outputs_fracture/rib_ft2/{oof_train,val_scores,summary}` → `final_ft.py --rib …/rib_ft2 --tag with_rib2` → `final_ft/with_rib2/report.md`（Mac 鏡像同路徑）。停止：kill 該 Mac 行程。
- 11-07 11:15 rib_ft（凍結 probe）結果：R_ft 單獨 train OOF .835（L_r .794）；sitewise S2+R·lse val .870 [.852,.886]（+.008）；與 spine/pelvis FT 整合 final_ft/with_rib：S3·lse train CV .8788，val **.878 [.861, .893]**（Δ +.016 [+.010, +.022]）。rib_ft2 ep0 inner .777。
- 完成後：依 train OOF（肋骨標籤，同 folds）在 rib_ft（凍結 probe）／rib_ft2／平均 中選 → A6000 `outputs_fracture/rib_ft2/{oof_train,val_scores}.csv`（欄位 `ft`）→ 重跑 `final_ft/scripts/final_ft.py`。

## ✅ 局部脂肪浸潤 3D CNN（focal_steatosis_cnn；11-06 12:40–17:40 完成，無程序在跑；4090 GPU 已釋放）
結果：CUR＋CNN（依 train OOF 選）OOF .6968→.6974、val .676→.681 [.649,.712]，Δ +.004 [−.001,+.009] n.s.；CNN 單獨 OOF .656／val .646。建議正式最佳維持 .676。報告 A6000 `outputs_final/focal_steatosis_cnn/report.md`。以下為執行方式紀錄（需重跑時參考）。
目標：局部脂肪浸潤 val .676（實驗 N：REF＋全部單元頭）再提升。只用 train/val（標籤 `focal_steatosis_cnn/work/labels_trval.csv` 由 Exp N stack 的 train fold／val 檔建立），依 train 5-fold OOF 選模。
- **A6000 CPU（只做 crop，nice 19，3 個 crop process＋1 個 rsync）**：`outputs_final/focal_steatosis_cnn/scripts/fsc_crop_chain.sh`（log `logs/chain.log`、`crop_a6.log`、`crop_ct.log`、`fetch.log`）。用 Exp N 的 `nu_crop.py fst`（未修改）補齊 14,379 例缺少的 48³ crop：4,628 例用 A6000 `outputs/prep`；9,751 例只在 CT-rate NVMe（`~/OWL/mw_exp/outputs/prep,masks`），以 200 例一批 rsync（CT-rate 端 ionice idle、只讀檔，不碰 3080）到 `tmp/`，crop 完即刪。輸出 `focal_steatosis_cnn/crops/`。完成時寫 `CROP_ALL_DONE`。停止：`touch outputs_final/focal_steatosis_cnn/STOP`（本批做完後停）；續跑：重跑同一支（已完成的 crop／批次會跳過）。
- **Mac 中繼**：`outputs_final/focal_steatosis_cnn/scripts/fsc_push.sh`（log `outputs_final/focal_steatosis_cnn/logs_push.log`），每 2 分鐘把 A6000 新 crop（只限 train/val id）用 tar 串流到 4090 `~/OWL/fsc/crops`，全部完成後在 4090 寫 `~/OWL/fsc/work/PUSH_DONE`。停止：`touch outputs_final/focal_steatosis_cnn/STOP_PUSH`；續跑：重跑（只補缺的）。
- **4090 GPU（`ssh -p 23688 glows@tw-07.access.glows.ai`；VRAM ≤ 7.5 GB、torch／OMP 2 threads、nice 10；不碰 `~/OWL/gs15`）**：driver `~/OWL/fsc/scripts/run_fsc.sh`（log `~/OWL/fsc/logs/driver.log`、`logs/train_<cfg>_s<seed>.log`）。等 PUSH_DONE → `fsc_pack.py`（memmap `work/hu.npy`、`msk.npy`，約 20 GB）→ 3 個設定（A：3D w16；B：3D w24 較長；C：各向異性 1×3×3 的 2.5D 式）× seed 0/1/2，每組 5 fold OOF＋全 train 模型→val。輸出 `~/OWL/fsc/runs/<cfg>/pred_{0..4,full}_s<seed>.csv`。完成寫 `~/OWL/fsc/ALL_TRAIN_DONE`。停止：`touch ~/OWL/fsc/STOP`（目前模型訓練完才停，約 3–6 分）；立即停：`pkill -f run_fsc.sh; pkill -f fsc_train.py`。續跑：`cd ~/OWL/fsc && setsid nohup bash scripts/run_fsc.sh >> logs/driver.log 2>&1 < /dev/null &`（已有的 pred 檔跳過）。
- **堆疊（Mac CPU）**：`outputs_final/focal_steatosis_cnn/scripts/fsc_stack.py stackin <runs> stack`；`stackin/` 是 A6000 上已依 train/val 過濾的 REF 特徵、Exp N 單元頭 OOF/val。已驗證重建 CUR = OOF .6968／val .6762（與 Exp N 相同）。報告 → A6000 `outputs_final/focal_steatosis_cnn/report.md`（Mac 鏡像）。

## ✅ 11-06 10:45 更新
- 實驗 L 最終評估、實驗 O 收尾都已完成（A6000 run_final.sh／no_rescue_a6.sh 已結束，`A6000_GPU_LOCK` 不存在）。結果見 GPU_QUEUE「完成」與 `outputs_fracture/final_combo/report.md`（骨折 I＋J＋L＋骨盆單元合併）。
- 清理：A6000 `region_units/bowel_obstruction_r2/crops{,_ct}`（3.3＋9.0 GB）與 CT-rate `~/OWL/mw_exp/region_units_tmp/*_r2/crops`（9.0 GB）已刪（刪前確認 20,364 個 emb 齊全、每個 crop id 都有 emb）。`sync_spark.sh` 兩份重複迴圈已停（`outputs_ribs15/STOP_SYNC_SP`）。
- r2 單元報告：A6000 `outputs_final/region_units/{bile_duct,free_air,ascites,fracture_pelvis,bowel_obstruction}_r2/report.md`（Mac 鏡像同路徑；產生器 `scripts/no_report.py`，`no_table.py` 已修正語法錯誤與子群欄位重複）。

## ▶ 11-06 09:00 狀態更新（restart 後 status agent）
- **4090 GPU 已故障**：約 06:58 起 `CUDA_ERROR_LAUNCH_FAILED`，`nvidia-smi` 顯示 `Unable to determine the device handle … Unknown Error`（容器內無法重置；需使用者在 glows.ai 重啟或釋放執行個體）。4090 上已無 OWL 行程（Exp L worker／下載器已因 STOP 結束；Exp O 卡住的 no_extract 已 kill）。`/root/OWL/EXP_O_RUNNING` 仍在。
- **實驗 L 抽取完成**：A6000 `outputs_ribs15/emb15` 20,355／20,355（4090 的 emb15 已由 sync 併入；Mac `sync_ribs15.py` 已不需要）。A6000／Spark worker 都已結束，Spark GPU 閒置。`sync_spark.sh`（PID 772864、776518，兩份重複）仍在空轉，可 `touch outputs_ribs15/STOP_SYNC_SP` 停。
- **實驗 L 最終評估**（A6000，09:04 起）：`outputs_fracture/expL/run_final.sh`（bash PID **1233148**，log `outputs_fracture/expL/final/run_final.log`）：重建 l_matrix（r15 20,010／20,018）→ 全 train 5 seed OOF（L_add+rl03、L_repl+rl03+lrc、L_repl+rl03、Jadd 參考；`--device cuda`，4 個 process，每 fold 約 100 s）→ `rib15_eval.py`（`final/lrep/`）→ `final_stack.py` I 集成＋J add＋L（`final/SL_<v>/report.md`，含 vs J add 10 seed .847 配對 bootstrap）；依 stacker train CV 選。ETA ~10:30–11:00。執行期間建立 `outputs_final/A6000_GPU_LOCK`（OOF 結束自動刪）。停止：`touch outputs_fracture/expL/final/STOP`（步驟間生效）；立即停：`kill 1233148; pkill -f "rib15_probe.py --stage oof"`，再刪 lock。續跑：重跑同一腳本（已存在的 OOF 檔會跳過；build 會重做）。
- **實驗 O 改在 A6000 收尾**：Mac driver（7768/7769）、feed、post 已停（`expO_stage/STOP_DRIVER`、`bowel_obstruction_r2/STOP_FEED` 已建）。A6000 `outputs_final/region_units/scripts/no_rescue_a6.sh`（bash PID **1235871**，log `region_units/rescue_o_a6.log`）：從 CT-rate（只讀 CPU/磁碟）拉 13,775 個胃出口／十二指腸 crop 到 `bowel_obstruction_r2/crops_ct/` → A6000 GPU `no_extract.py`（8 GB／3 threads，已有 emb 跳過）→ `no_probe.py frpp`（骨盆分部位）→ `no_stack.py fracture_pelvis_r2` → `no_probe.py go` → `no_stack.py bowel_obstruction_r2`。ETA ~11:00。停止：`touch outputs_final/region_units/STOP_RESCUE_O`（步驟間），立即停 `kill 1235871; pkill -f no_extract.py`；可續跑。完成後可刪 A6000 `bowel_obstruction_r2/crops{,_ct}` 與 CT-rate `region_units_tmp/*_r2/crops`。

## ▶ 膽結石 gs15：1.5 mm 膽囊 crop 單元（11-06 12:12 起，A6000 GPU＋CPU、Mac CPU；agent：gallstones15）
目標：膽結石 val .881 → ≥ .90。只用 train/val（標籤先依 split 過濾），只依 train 5-fold OOF（GS1 folds）選模型，val 只報告（對 .881 配對 bootstrap），test 未讀。
- 目錄：A6000 `outputs_final/gallstones15/`（crops/、emb/、claims/、ct/〔暫存〕、logs/、inputs/、out_tier0/、stack_tier0/、cnn_tier0/、out/、stack/、cnn_full/、feats.csv、**report.md**）；程式 `scripts/`（Mac 鏡像同路徑）。Mac 中繼 `VLM/data/CT/merlin/gs15_stage/`（root/、masks3/〔3 mm TS mask 複本〕、gs6/、ids_mac.txt、maskmap_mac.tsv、mac_driver.log）。
- 方法：從原始 CT 切 96³×1.5 mm（144 mm）crop，中心＝TS 3 mm 膽囊質心（≥100 voxel）→ 否則 GS6 1.5 mm 膽囊（≥800 voxel）→ 否則肝 bbox 比例的膽囊窩點（同 Exp N）。5 個區域（膽囊膨脹 3 mm、膽囊窩 30 mm ROI、重力依賴側 1/3、中心 20 mm 球、膽囊內 ≥90 HU）× 6 窗（baseline、軟組織 40/400、窄窗 60/150、40/250、結石窗 200/500、低密度〔膽固醇結石〕窗 10/80），TS Dataset297 凍結 encoder stage 1–3 mean＋max。另有 1.5 mm 膽囊 HU 特徵（百分位、max、≥60/100/130/200/300 HU 體積、低密度／膽固醇候選體積、最大高密度 blob、依賴層−非依賴層中位數差、膽囊壁、膽囊窩 ROI），存在每個 crop 的 `feat`，彙整到 `feats.csv`。
- **給 gs15 driver／stack 的備註（11-06 14:20，agent：gallstones_lowdens）**：低密度膽囊特徵已存在 A6000 `outputs_final/gallstones_lowdens/feats_lowdens.csv`（Mac 同路徑），共 20,364 例 train/val，含 split 欄。`ld_*`（31 欄）是 3 mm TS 遮罩的特徵，涵蓋全部 train/val；`ld15_*` 是 gs15 crop 的特徵，在 13:34 時有 3,640 例。如果 stack 要用，請 merge by study_id，**不要改 feats.csv**。crop 增加後可以重算 ld15：`python scripts/ld_feats.py 15 ids_trval.txt ../gallstones15/crops feats15.csv 2`。注意：把這些特徵加進 Exp N 時，val 是 −.003，年齡配對 −.004，所以**不建議納入**，只供參考。
- 優先順序 `inputs/ids_prio.csv`：tier0 9,084＝全部 val 5,055＋train 陽性 1,343＋年齡×性別配對陰性 2,686（1:2）；tier1＝其餘 train 11,280。每 100 例一區塊，A6000／Mac 交錯分配（`ids_a6.txt`／`ids_mac.txt`：自己的 tier0 → 對方 tier0 倒序 → 自己 tier1 → 對方 tier1 倒序；claims 每 2 分鐘雙向同步）。
- **A6000**：driver `scripts/run_gs15.sh 7`（bash PID 1430366，12:41 已停、改由 run_gs15_eval.sh 接手，見下，log `logs/driver.log`，每 5 分鐘一行進度）；7 個 crop 迴圈 PID 1430371/1430387/1430403/1430419/1430437/1430491/1430575（nice 10、2 threads、每命 300 例；原始 CT 用 ssh 從 CT-rate HDD 唯讀拉取，3 mm mask 與 GS6 讀本機）；GPU 抽取迴圈 PID 1430592（`gs15_extract.py --poll`，VRAM 上限 12 GB，實測 ~0.16 s/例、約 3 GB）。
- **Mac**：`gs15_stage/scripts/gs15_mac.sh 5`（caffeinate，driver PID **77381**，log `gs15_stage/mac_driver.log`）＋追加 w5/w6（caffeinate PID 79222/79223）；7 個 worker 讀 Passport 原始 CT（與 A6000 輸出逐 voxel 相同，已驗證），每 2 分鐘把 crop 推到 A6000 `crops/`（推完刪本機）、同步 claims／log。
- 實測（12:12–12:20）：A6000 ~29 例/min（CT-rate HDD 讀取 ~11 s/例為瓶頸）＋Mac ~23 例/min（Passport I/O 為瓶頸；12:21 加到 7 worker）＝約 2,900–3,200 例/h。**ETA：tier0 ~15:15（中期結果 ~15:45）；全部 ~19:00（最終結果 ~20:00）**。
- **12:41 改版（使用者：A6000 GPU 另有用途，不在 A6000 開新的 GPU 工作；CNN／probe 改到 4090）**：原 driver（1430366）已停（只停 driver 本身；crop 迴圈與 GPU 抽取迴圈照跑，抽取約 2.6 GB VRAM）。新的 A6000 eval driver `scripts/run_gs15_eval.sh`（PID **1473523**，log `logs/eval_driver.log`，**只用 CPU**）：tier0 全部 embedding 完 → 寫 `inputs/labels_tier0.csv`＋`REQ_tier0` → 等 4090 probe 結果（`out_tier0/PROBE_DONE`；180 分鐘沒來就在 A6000 CPU 跑 probe）→ `gs15_feats.py` → `gs15_stack.py`（CPU）→ `report.md`、`TIER0_EVAL_DONE`；全部抽完（`EXTRACT_DONE`）→ 同樣流程 full → `FINAL_DONE`。
- **4090（`ssh -p 23688 glows@tw-07.access.glows.ai`，目錄 `~/OWL/gs15_gpu/`：scripts/、emb/、in/、res/；venv `~/L/venv`）**：由 Mac 協調腳本 `gs15_stage/scripts/gs15_remote.sh`（caffeinate，PID **86624**，log `gs15_stage/remote.log`）經 Mac 中繼（A6000 與 4090 互不相通）。只傳衍生資料：crop embedding、CNN 輸入包（64³ HU＋膽囊遮罩，`gs15_cnn_pack.py`，每例約 0.3 MB）、main_gs.npz（train/val）、train/val 標籤子集；不傳 test，不碰 SAS。VRAM：probe ≤ 6 GB、CNN ≤ 10 GB，依序執行（≤ 14 GB，留給另一個 agent 的局部脂肪 CNN ≤ 8 GB）。
  1. CNN 調參（tier0 train crop ≥ 2,400 時，約 13:45）：只用 train 列（val 完全不載入），3 組設定（e14/lr1e-3、e24/lr5e-4、e10/lr2e-3）× GS1 5-fold × 1 seed，依 train OOF 選 → `gs15_stage/gpu/best_args.txt`（同步到 A6000 `cnn_tune/`）。
  2. `REQ_tier0` 出現 → emb 同步 → 4090 probe（U／U_org／U_gb3）→ A6000 `out_tier0/`＋`PROBE_DONE`；再打包 → 4090 CNN（best_args、2 seed，GS1 OOF＋全 train 評 val）→ A6000 `cnn_tier0/` → A6000 CPU 在 driver 的 stack 之後重跑 `gs15_stack.py`（含 CNN 變體，`logs/stack_tier0_cnn.log`）→ 更新 `report.md`，`CNN_tier0_DONE`。
  3. `REQ_full` → 同上（`out/`、`cnn_full/`、`logs/stack_cnn.log`、`CNN_full_DONE`）。全量 CNN 估約 2–3 h（GPU）。
  停止協調：`touch gs15_stage/STOP_REMOTE`（或 A6000 `STOP`）；4090 上當下的 python 用 `pkill -f gs15_cnn_train.py`／`gs15_probe.py`。續跑：重跑同一行 `nohup caffeinate -i bash scripts/gs15_remote.sh > remote.log 2>&1 &`（已完成的 PROBE_DONE／CNN_<mode>_DONE 會跳過；調參有 best_args.txt 就跳過）。
- **15:28 事件處理與重啟**：
  - ① A6000 的 7 個 crop 迴圈在 12:41 隨舊 driver 一起被 kill（`pkill -P`），worker 各自做完 300 例後於 13:23 結束；13:23–15:28 A6000 沒有切 crop（只有 4090 在切）。GPU 抽取 python（1430594）仍在跑。
  - ② 15:10–15:20 A6000 磁碟滿（別的 session 的暫存檔）。檢查 14:55–15:40 間寫入的 639 個 crop、645 個 emb：全部可完整載入，shape 正確、沒有 NaN；crop／emb 一一對應；4 個沒有 crop 的 claim 是 4090 正在做的。沒有刪除任何檔案。
  - ③ Mac crop worker 依使用者指示已停（`gs15_stage/root/STOP`，14:33 MAC_DONE），沒有影響 A6000／4090。Mac 的 tier0 區塊由 A6000 清單尾端接手；claims 由另一個 agent 的 `relay_4090.sh` 同步。
  - ④ 新的 A6000 worker 腳本 `scripts/run_gs15_workers.sh 7`（PID **1823860**，log `logs/workers.log`）：7 個 crop 迴圈（PID 1823863/1824009/1824094/1824122/1824208/1824294/1824372，每命 50 例），加上抽取 keepalive。**磁碟保護**：/ 剩不到 15 GB 就不開新批次，crop 程式也會在批次中途停；每 2 分鐘刪除 `ct/` 裡超過 30 分鐘的暫存檔。停止：`touch STOP_a6`（只停 crop）或 `STOP`（全部停）。
  - ⑤ 4090 CNN 調參完成（只用 train，2,404 例）：e14/lr1e-3 .796、**e24/lr5e-4 .805（選用）**、e10/lr2e-3 .789。
  - ⑥ 新 ETA（不含 Mac：A6000 ~1,700 例/h＋4090 ~870 例/h）：tier0 約 16:20、中期結果約 16:45、tier0 CNN 約 17:30；全部約 20:50、最終結果約 21:30、全量 CNN 約 23:30–24:00。
- 12:36 曾在 A6000 啟動 CNN 調參迴圈（PID 1464138），尚未用到 GPU 就在 12:38 停掉（`STOP_CNN_TUNE`），改到 4090。
- Exp N 3 mm 膽囊 crop 的 CNN：**不可行**——A6000 只剩 5,838／20,364 例（隨機約 29%，train 陽性 387／1,343），CT-rate 份已刪，無法產生全 train OOF 當堆疊輸入；要做需先在 CPU 重切 3 mm crop。
- 待辦（GPU，未實作）：在 1.5 mm 膽囊 crop 上微調 TS encoder 最後幾個 stage（train folds、OOF 當堆疊輸入、內部 train 早停）。
- 停止：A6000 `touch outputs_final/gallstones15/STOP`（crop worker 做完當前一例結束；抽取迴圈、driver 也停；Mac 同步迴圈會把 STOP 傳到 Mac）。只停 Mac：`touch gs15_stage/root/STOP`。只停 CNN：`touch outputs_final/gallstones15/STOP_CNN`（stack 照跑，不含 CNN）。要立刻停：kill 上列 PID（含 eval driver 1473523、Mac 86624），再 `pkill -f gs15_crop.py; pkill -f gs15_extract.py`（A6000 與 Mac）。
- 續跑：刪 `STOP`、`STOP_EXTRACT`；`rm ct/*`；刪除 `claims/<id>` 中沒有 `crops/<id>.npz` 也沒有 `emb/<id>.npz` 的（Mac 端 `gs15_stage/root/claims` 同樣處理，否則同步會讓它被跳過）；再在 mw_exp 執行 `setsid nohup bash outputs_final/gallstones15/scripts/run_gs15.sh 7 >> outputs_final/gallstones15/logs/driver.log 2>&1 < /dev/null &`（它會再起 crop＋抽取；之後 kill 它自己的 driver bash，改跑 `setsid nohup bash outputs_final/gallstones15/scripts/run_gs15_eval.sh >> outputs_final/gallstones15/logs/eval_driver.log 2>&1 < /dev/null &`，因為 run_gs15.sh 內建的 eval 會在 A6000 GPU 上跑 probe），Mac 執行 `cd gs15_stage && nohup caffeinate -i bash scripts/gs15_mac.sh 7 > mac_driver.log 2>&1 &`（已完成的 crop／emb 會跳過；已完成的 eval 有 `TIER0_EVAL_DONE` 會跳過）。
- 磁碟：每個 crop 約 1 MB，全部約 20 GB（A6000 12:00 剩約 80 GB）；Mac 推完即刪。完成並確認 emb 齊全後，若不再訓練 CNN 可刪 `crops/`。

### ▶ gs15 新 4090 加入 crop＋embedding（11-06 12:26 起；agent：gs15_4090）
- **主機**：新租 4090 `ssh -p 23688 glows@tw-07.access.glows.ai`（user glows，passwordless sudo；Ubuntu 24.04、kernel 6.8.0-49、20 vCPU、58 GB RAM、~180 GB 可用；**時鐘是 UTC**，台灣時間 −8 h）。舊 4090（port 25966）已不用。
- **Driver**：卸 nouveau＋blacklist（`/etc/modprobe.d/blacklist-nouveau.conf`、update-initramfs），apt `nvidia-driver-550-server`（Ubuntu 轉接套件實裝 **580.178.04-server**，CUDA 13.0），`modprobe nvidia` 成功，**未重開機**；nvidia-smi 正常（24,564 MiB）。安裝腳本／log `~/drv.sh`、`~/drv.log`。
- **環境**：`~/L/venv`（Python 3.12；torch **2.9.1+cu126**〔cuDNN 9.10〕、nnunetv2 2.8.1、TotalSegmentator 2.18.0，與 A6000 相同；pip check 只有 nnunetv2 對 torch 2.9 的警告，與舊 4090 相同）。另一個 agent 的 probe／CNN（`~/OWL/gs15_gpu/`）也用這個 venv。
- **檔案**：`~/OWL/gs15/`：`scripts/`（gs15_crop.py、**gs15_crop_buf.py**〔讀下載暫存區版本〕、gs15_extract.py、gs15_windows.yaml、dl_gs15.py、run_4090.sh、azure_watch.sh、feed_4090.py、relay_4090.sh）、`src/`（encoder.py、windowing.py，取自 A6000）、`ts/Dataset297…`（TS 3 mm 凍結 encoder，取自 A6000）、`masks3/`（23,059 個 3 mm TS mask，取自 Mac `gs15_stage/masks3`）、`gs6/`（5,035）、`maskmap_4090.tsv`、`ids_4090.txt`（**tier0 9,084 倒序 → tier1 11,280 倒序**，從 A6000／Mac 清單的尾端往前搶，靠 claims 避免重複）、`root/{crops,emb,claims,logs}`。原始 CT 暫存 `~/L/buf`（做完即刪）。Mac 中繼目錄 `VLM/data/CT/merlin/gs15_4090_stage/`（code/、claims/、emb/、crops/〔過站〕、logs/、relay.log、feed.log）。
- **Smoke test**：同一例（<ID redacted>）4090 crop 與 A6000 crop 的 HU 逐 voxel 相同、HU 特徵相同；embedding 最大差 3.7e-4（fp16）、相關 0.9999998。
- **⚠️ Azure SAS**：`~/OWL/.azure_sas` 目前只有 **1 byte**（空的；未讀內容），所以 Azure 無法用。改由 **Mac feeder** 供 CT：`feed_4090.py STAGE 8 48 passport`（Mac PID **84506**，caffeinate；8 條串流，從 Mac 的 Passport 讀原始 CT → ssh 管線寫到 4090 `~/L/buf`，buffer ≤ 48 檔）。實測：從 CT-rate HDD 中繼（12:27–12:36）會把 A6000 的 CT 讀取從 8 s 拖到 13 s（A6000 1,950→1,490 例/h，淨增益很小），所以 12:37 改從 Passport（A6000 恢復 ~1,900 例/h、Mac worker 不受影響）。`feed_4090.py` 的第 4 參數可設 `ctrate|passport|mix`。
  **SAS 一放進去就會自動切換**：4090 上 `azure_watch.sh`（PID **79504**，log `root/logs/azure_watch.log`）每分鐘檢查 SAS 檔大小（> 50 bytes），就建 `~/L/buf/AZURE_ON`（Mac feeder 看到後自己結束）並啟動 `dl_gs15.py --par 64`（Exp L 實測約 2,000 例/h）。
- **4090 行程**：`run_4090.sh 6`（bash PID **56601**，log `root/logs/run.log`，每 5 分鐘一行）；6 個 crop 迴圈 PID 56608/56631/56654/56688/56711/56784（`gs15_crop_buf.py --tag 4090`，nice 5、2 threads、每命 300 例；依 ids_4090 順序消耗 buffer 裡的 CT；已有 crop／emb／claim 的跳過；被別台 claim 超過 15 分鐘的暫存 CT 會刪掉）；GPU 抽取迴圈 PID 56858（`gs15_extract.py --poll`，cap 12 GB，實占 ~2.2 GB）。crop 2.0 s/例/worker、GPU 0.046 s/例 → 瓶頸完全在 CT 供應。
- **Mac 中繼**：`relay_4090.sh`（PID **80654**，caffeinate，log `gs15_4090_stage/relay.log`）每 2 分鐘：claims A6000 ↔ Mac ↔ 4090 聯集（只增不刪）；4090 `emb/` → Mac → A6000 `outputs_final/gallstones15/emb/`（先推 emb）；已有 emb 的 4090 crop 搬到 A6000 `crops/`（4090 與 Mac 端都刪；供 feats.csv／CNN 用）；`crop_info_4090_w*.jsonl`、`failed_4090.txt` → A6000 `logs/mac/`（status 會讀）；A6000 有 `STOP` 時傳給 4090（`root/STOP`＋`~/L/buf/STOP_DL`）。
- **實測（12:39–12:48）**：4090 ~520–860 例/h（Passport ~520；CT-rate 路徑 ~860 但拖慢 A6000）；A6000 ~1,900、Mac ~1,050；**合計約 3,450 例/h**（未加 4090 前約 3,100）。12:48 tier0 emb 2,011／9,084。**ETA：tier0 ~14:55；全部 ~18:10**。若 SAS 補上（4090 +~2,000 例/h）：tier0 約提早到 ~14:20、全部 ~16:30。
- **停止**：4090 `touch ~/OWL/gs15/root/STOP ~/L/buf/STOP_DL`（crop worker 做完當前一例結束，抽取迴圈與 run_4090.sh 也結束）；Mac `touch gs15_4090_stage/STOP_FEED gs15_4090_stage/STOP_RELAY`（relay 最後一輪做完後結束；停 relay 前先讓 4090 emb／crop 都推過去）。A6000 的 `STOP` 也會經 relay 傳到 4090。
- **續跑**：4090 `rm -f ~/OWL/gs15/root/STOP ~/L/buf/STOP_DL ~/L/buf/.*.part`；刪 `root/claims` 中沒有 `root/crops/<id>.npz` 也沒有 `root/emb/<id>.npz` 的**自家** claim（`grep -h study_id root/logs/crop_info_4090_w*.jsonl` 之外的才是別台的；A6000 與 Mac `gs15_4090_stage/claims` 也要刪同名，否則同步會讓它被跳過）；再 `cd ~/OWL/gs15 && setsid nohup bash scripts/run_4090.sh 6 >> root/logs/run.log 2>&1 < /dev/null &`、`setsid nohup bash scripts/azure_watch.sh > root/logs/azure_watch.log 2>&1 < /dev/null &`。Mac：`cd gs15_4090_stage && nohup caffeinate -i bash code/scripts/relay_4090.sh > relay.log 2>&1 &`、`nohup caffeinate -i python3 -u code/scripts/feed_4090.py $PWD 8 48 passport >> feed.log 2>&1 &`。
- **之後（bd15）**：gs15 crop 全部完成後，4090 可接 bd15（膽管 1.5 mm 肝門 crop）剩餘 id；要另寫 bd15 版 crop_buf＋claims 同步（A6000 `outputs_final/biliary_improve/bd15/claims`），尚未實作。

## ▶ gs15_ft：膽結石項目 9，TS encoder 後段 stage 在 1.5 mm 膽囊 crop 上微調（新 4090 GPU，11-06 17:41 起；agent gs15_ft）
- **主機**：4090 `ssh -p 23688 glows@tw-07.access.glows.ai`，目錄 `~/OWL/gs15_ft/`（scripts/、runs/、logs/），venv `~/L/venv`；程式 Mac 鏡像 `outputs_final/gallstones15_ft/scripts/{gs15_ft.py,run_ft.sh}`。**不碰** `~/OWL/gs15`（crop worker）與 `~/OWL/gs15_gpu`（只唯讀它的 `in/pack_tier0.npz`、`in/labels_tier0.csv`；後者只有 train/val，程式內 assert）。
- **方法**：gs15 CNN 包（96³ crop 中央 64³×1.5 mm HU＋膽囊膨脹 3 mm 遮罩）→ TS Dataset297 encoder（3 窗：baseline、40/400、60/150，nnU-Net 正規化同 gs15_extract）→ stage 3＋4 特徵［全域 mean｜max｜膽囊遮罩 mean］→ dropout＋linear。encoder lr 1e-4、head 1e-3，AdamW、OneCycle、12 epoch、bs 16、bf16、shift ±6 voxel＋HU 抖動；每次 fit 用 training folds 內 12% 做早停。VRAM 上限 10 GB（實占 ~9.2 GB），nice 10、4 threads。
- **流程**（`run_ft.sh`）：① 調參，只用 train（`--train-only`，val 不載入）：tf3（訓練 stage 3–4）、tf2（stage 2–4）、tf5（全凍結，只訓練 head）× GS1 5-fold × 1 seed，各 ~10 分 → ② 依 train OOF 選最佳 → `runs/best_cfg.txt` → ③ 最終：2 seed 5-fold OOF＋全 train 模型→val（只評一次）→ `runs/final_<cfg>/{oof_train.csv,val_scores.csv,summary.json}`，完成寫 `~/OWL/gs15_ft/FT_DONE`。log：`logs/driver.log`、`logs/tune_<cfg>.log`、`logs/final_<cfg>.log`。**ETA ~19:00**。
- **之後（CPU）**：FT 分數放到 A6000 `outputs_final/gallstones15_ft/cnn_tier0/`（欄位改名 cnn），用 `gallstones15/scripts/gs15_stack.py` 以 ROOT＝gallstones15_ft 跑 tier0 堆疊（N_U＋FT，依 train OOF 選，vs 正式 .881 配對 bootstrap），報告 `outputs_final/gallstones15_ft/report.md`；不覆蓋 gallstones15 的 report。
- **停止**：`touch ~/OWL/gs15_ft/STOP`（當前 epoch 結束即停）；立即停：`pkill -f run_ft.sh; pkill -f gs15_ft.py`。**續跑**：`rm -f ~/OWL/gs15_ft/STOP; cd ~/OWL/gs15_ft && setsid nohup bash scripts/run_ft.sh > logs/nohup.out 2>&1 < /dev/null &`（已有 summary.json 的設定跳過）。

## ⏸ 項目 5 膽管擴張 bd15 — 11-06 12:01 暫停（讓 GPU／CPU／CT-rate HDD 給膽結石 gs15）
- 以 STOP 檔正常停止（`outputs_final/biliary_improve/bd15/STOP`，12:01:26 driver 結束，所有 crop worker／抽取迴圈都已結束，沒有 kill）。所有輸出都保留。
- 停止時進度：crops 3,061、emb 3,054（7 個 crop 尚未抽 embedding，續跑時抽取迴圈會補）、claims 3,061（每個 claim 都有 crop，沒有孤兒）、`ct/` 已空、failed 0。tier0 emb 3,054／10,792。
- **續跑時機**：等膽結石 gs15 crop 全部完成（A6000 `outputs_final/gallstones15/CROP_ALL_DONE` 出現，預估 ~19:00）再開，否則兩邊搶 CT-rate HDD 與 A6000 CPU。膽管已 .911，今晚續跑即可。
- **續跑指令**（在 A6000 `~/Desktop/Tony/VLM/OWL/mw_exp`）：
  `cd outputs_final/biliary_improve/bd15 && rm -f STOP STOP_EXTRACT && rm -f ct/* && for c in claims/*; do i=$(basename $c); [ -f crops/$i.npz ] || rm -f $c; done && cd ~/Desktop/Tony/VLM/OWL/mw_exp && setsid nohup bash outputs_final/biliary_improve/bd15/scripts/run_bd15.sh 7 >> outputs_final/biliary_improve/bd15/logs/driver.log 2>&1 < /dev/null &`
  已完成的 crop／emb 會跳過。剩餘約 17,100 例，以約 2,200 例/h 估：tier0 約 3.5 h 後、全部約 8 h 後。

## ⏸（12:01 暫停，見上） 項目 5 膽管擴張 1.5 mm 肝門 crop（bd15）— 11-06 10:35 起在 A6000 執行（原 Spark 計畫 09:30 取消，改 A6000）
- 目錄 A6000 `outputs_final/biliary_improve/bd15/`（crops/、emb/、claims/、ct/〔暫存〕、logs/、inputs/、smoke/〔smoke test，可刪〕）。只用 train/val；test 未讀。
- A6000 改寫：`scripts/run_bd15.sh`（路徑改 A6000、`~/miniconda3/envs/radar`；Spark 原版留 `run_bd15.sh.spark_orig`）；`bd15_crop.py` 改為 3 mm 遮罩直接讀 A6000 本機（`inputs/maskmap.tsv` → `outputs/masks`、`outputs_vert15/masks3_extra`，20,186／20,186 都有），原始 CT 仍以 ssh 從 CT-rate HDD 唯讀拉取（`ct/` 暫存，做完即刪）。inputs：ids_prio.csv、labels_bd.csv（train/val）、main_bd.npz（連到 region_units/probe_inputs）、ref_oof/ref_val＝integrated_best 膽管擴張（stack_oof／stack，val .911）、age_trval.csv。
- Smoke test（10:34–10:35）：crop 4 例全部 ok，單 worker 5.3–7.2 s/例（CT 讀取 1.1–1.7 s、resample 1.8–3.5 s、區域 1.8 s），每個 crop 約 2.7 MB；GPU 抽取 0.25 s/例（首例 0.9 s），emb (1,7,5,896) 全為有限值。probe／stack／report 另以假 embedding smoke test（`smoke/fake/`）。
- 執行中：driver bash PID **1323252**（log `logs/driver.log`，每 5 分鐘一行進度）；crop worker 迴圈 PID 1323257/1323273/1323291/1323309/1323326/1323343/1323367（每命 300 例回收，nice 10，2 threads）；GPU 抽取迴圈 PID 1323386（`logs/extract.log`、`logs/extract_timing.csv`，--vram-gb 24，實占約 5 GB）。7 worker 實測約 2,100–2,300 例/h（CT 讀取多 worker 時約 4.4 s）。
- 順序：tier0（10,792）→ 自動中期評估（`bd15_probe.py` → `out_tier0/`、`bd15_stack.py` → `stack_tier0/`、`report.md`）→ tier1（9,394 train 陰性）→ 全量評估（`out/`、`stack/`、`report.md`、`FINAL_DONE`）。**ETA：tier0 ~15:45、中期結果 ~16:15；全部 ~20:15、最終結果 ~21:00**。
- 磁碟：crop 全保留約 54 GB（A6000 10:45 剩約 95 GB）；全部完成且 emb 齊全後可刪 `crops/`。
- 停止：`touch outputs_final/biliary_improve/bd15/STOP`（crop worker 做完當前一例後結束；抽取迴圈與 driver 看到 STOP 結束）。要立刻停：`kill 1323252 1323386 <crop 迴圈 PID>` 再 `pkill -f bd15_crop.py; pkill -f bd15_extract.py`。
- 續跑：`rm STOP STOP_EXTRACT`；`rm ct/*`；刪除沒有 `crops/<id>.npz` 的 `claims/<id>`；再在 `~/Desktop/Tony/VLM/OWL/mw_exp` 執行 `setsid nohup bash outputs_final/biliary_improve/bd15/scripts/run_bd15.sh 7 >> outputs_final/biliary_improve/bd15/logs/driver.log 2>&1 < /dev/null &`（已完成的 crop／emb 會跳過）。
- 完成後：更新 GPU_QUEUE「完成」、Notion §12（只有真的新最佳才更新；REF＝.911）。

## ✅ 實驗 N：區域 crop 單元（膽結石→動脈粥狀硬化→局部脂肪；11-06 02:55–04:22 完成；無程序在跑）
- 結果與報告：A6000 `outputs_final/region_units/{gallstones,atherosclerosis,focal_steatosis}/report.md`（Mac 鏡像同路徑）。
  - 膽結石 .866 → **.881**
  - 動脈粥狀硬化 .912 → **.921**（年齡配對 .779 → .801）
  - 局部脂肪 .665 → .676（n.s.）
  - test 未碰。
- 程式：`outputs_final/region_units/scripts/nu_*`（crop／extract／probe／stack／chain／feed／post／gs6_sync）。Mac 中繼與 log：`VLM/data/CT/merlin/expN_stage/`。
- 4090 `/root/OWL/expN/` 只剩 emb／out／logs，約 4.4 GB，可刪。標記 `/root/OWL/EXP_N_RUNNING` 已移除。
- crop 保留：
  - A6000 `region_units/<unit>/crops/`（各 5,838 例）
  - CT-rate `~/OWL/mw_exp/region_units_tmp/<unit>/crops/`（各 14,526 例，合計約 19 GB，可刪）

## ✅ 實驗 M：區域 crop 單元（11-06 01:40–02:55 完成；無程序在跑）
- 結果（val，依 train OOF 選定，相對 integrated_best REF 的配對 Δ）：腎水腫 .944→**.957**（+.012 [+.005,+.020]）；腹腔游離氣體 .901→**.912**（+.011 [+.002,+.019]）；腹水 .906→**.916**（+.009 [+.005,+.014]）。test 未碰。
- 報告：A6000 `outputs_final/region_units/{hydronephrosis,free_air,ascites}/report.md`（本機鏡像同路徑）；程式 `region_units/scripts/`；emb 在 A6000 `region_units/*/emb/`（各 20,364）。
- crop 保留：A6000 `region_units/*/crops/`（A6000 份，約 10.5k/單元）、CT-rate `~/OWL/mw_exp/region_units_tmp/*/crops/`（9,846/單元）。4090 上的 crop／emb 已刪，`/root/OWL/EXP_M_RUNNING` 已移除（`/root/OWL/expM/` 只剩程式、probe 輸入、log，616 MB）。

## ✅ 實驗 L：1.5 mm 肋骨（11-06 00:47 起；10:16 最終評估完成，無程序在跑）
TS 高解析度肋骨模型 Dataset295（part5_ribs，1.5 mm，patch 128³），ROI＝3 mm 肋骨 mask 框 ±15 mm（預先算好的世界座標 `outputs_ribs15/rib_boxes.npz`，20,355／20,364 有肋骨）→ 18 個肋骨單元（organ_map_ribs）→ 3 窗 × stage 1–3，mean＋max。程式 `src/rib15_full.py`（extract / prio / boxes）、`src/rib15_probe.py`（模型）、`src/rib15_eval.py`（報告）、`src/dl_ribs15.py`（4090 Azure 下載）、`src/sync_ribs15.py`（Mac 中繼）。
- id 順序：`outputs_ribs15/ids_prio.csv`（tier0 8,405＝train 骨折陽性 1,675〔肋骨 505 排最前〕＋陰性 1,675〔半數困難陰性〕＋全部 val 5,055）。以 100 例為一區塊，每 3 塊中 4090 取 2 塊、A6000 取 1 塊；`ids_a6.txt`／`ids_4090_v2.txt`＝自己的 tier0 區塊 → 對方 tier0 區塊 → 自己 tier1 → 對方 tier1（搶尾巴，靠 claims 同步避免重複）。
- **A6000**：4 worker，loop PID **396577–396580**（`run_ribs15.sh … outputs_ribs15/ids_a6.txt outputs_ribs15 0 3 200 - outputs/masks outputs_vert15/masks3_extra`），CT 用 ssh 從 CT-rate HDD 拉。持有 `outputs_final/A6000_GPU_LOCK`（結束後要刪）。實測單 worker 約 6.5 s/例、峰值 VRAM 8 GB。輸出 `outputs_ribs15/{emb15,seg15,claims,logs,timing_w*.csv,failed.txt}`。
  停止：`touch outputs_ribs15/STOP`（當前一例做完後結束；生成器也會檢查 STOP）。續跑：刪 STOP，清 `emb15/*.tmp*`、`ct/`、沒有 emb15 的 claims，再跑同一行 `run_ribs15.sh`。
- **4090（租用，`ssh -p 25966 root@tw-07.access.glows.ai`）**：環境 `/root/L/venv`（torch 2.9.1+cu126、nnunetv2 2.8.1、TS 2.18.0），權重與程式在 `/root/OWL/mw_exp`，輸出 `/root/OWL/out_ribs15`，原始 CT 暫存 `/root/L/buf`（Azure 下載，處理完即刪，上限 220 檔／35 GB）。benchmark：Azure 每條約 0.75 MB/s（48 條約 36 MB/s，瓶頸），seg＋embed 約 2.8 s/例/worker（GPU 約 1 s，其餘是 CPU 解壓/resample），峰值 VRAM 2.2 GB。
  **01:35 已啟動**（使用者授權後）：`/root/OWL/run_4090.sh 5` → 下載迴圈 PID **6477**（`dl_ribs15.py`，Azure par 64〔實測 128 條 26 MB/s、48 條 57 MB/s、64 條 60 MB/s ≈ 2,000 例/h〕，暫存上限 220 檔／35 GB；已完成或被他台 claim 超過 10 分鐘的暫存 CT 會自動刪除；log `/root/L/dl.log`、`out_ribs15/logs/dl.out`），worker 迴圈 PID **1716–1720**；合計（含 A6000）約 3,150 例/h，ETA tier0 ~03:40、全部 ~07:30（5 個 `--consume`，每命 300 例回收；log `out_ribs15/logs/w*.log`）。id 清單 `/root/OWL/out_ribs15/ids_4090.txt`（＝A6000 `outputs_ribs15/ids_4090_v2.txt`）。
  停止：`touch /root/OWL/out_ribs15/STOP /root/L/buf/STOP_DL`（worker 做完當前一例後結束；暫存 CT 可 `rm /root/L/buf/*`）。續跑：刪這兩個 STOP，刪 `/root/L/buf/.*.part`、`out_ribs15/emb15/*.tmp*` 與沒有 emb15 的自家 claim（也要在 A6000 刪同名 claim，否則同步會讓它被永久跳過），再跑 `/root/OWL/run_4090.sh 5`。SAS 檔在 4090 `/root/OWL/.azure_sas`（600，使用者提供；不要印出）。舊 4090（port 24814）上也曾放過 SAS 檔，節點已下線無法刪除。
- **Mac 中繼**：`python3 src/sync_ribs15.py <stage>`（PID 69063，caffeinate 包著；stage＝`VLM/data/CT/merlin/ribs15_4090_stage`，log `sync.log`）：每 2 分鐘同步兩台的 claims（只傳 id），每 10 分鐘把 4090 的 emb15/seg15 拉到 Mac 再推到 A6000 `outputs_ribs15/`。停止：`touch <stage>/STOP_SYNC`。
- **Spark**（04:25 起，GS6 已結束、無 vLLM）：3 worker，loop PID **3478419–3478421**，清單 `~/OWL/expL/ids_spark.txt`（tier0 倒序→tier1 倒序），CT 用 ssh 從 CT-rate 拉，輸出 `~/OWL/expL/out_ribs15`。停止：`touch ~/OWL/expL/out_ribs15/STOP`。A6000 `outputs_ribs15/sync_spark.sh`（PID 776518，log `logs/sync_spark.log`）每 2 分鐘同步 A6000↔Spark claims，並把 Spark 的 emb15/seg15 拉回 A6000；停止 `touch outputs_ribs15/STOP_SYNC_SP`。
- **tier0 已於 04:19 完成**（8,399／8,405；6 例 3 mm mask 無肋骨）。4090 failed.txt 中的 FileNotFoundError 都是 A6000 已做完的重複 id，不影響。中期評估：A6000 `outputs_fracture/expL/interim.sh`（PID 771507，結果 `outputs_fracture/expL/interim/`）。
- 模型／評估：A6000 `outputs_fracture/expL/`（`l_matrix.npz`、`oof/{tier0,full}/oof_L_*`、`logs/`）。

## ✅ 闌尾炎右下腹（RLQ／盲腸）crop 單元（11-05 23:55 起，11-06 已完成；無程序在跑）
- 結果：A6000 `outputs_final/appendicitis_rlq_unit/report.md`（本機鏡像同路徑）。依 train OOF 選模型；val：確定的急性闌尾炎 REF＋U_rlq_cs **0.936** [0.912, 0.958]（REF 0.902，+0.034）；原合併定義 **0.920**（REF 0.888，+0.033）；可疑／治療中組 U_rlq_cs 0.832（26 例陽性，僅描述性）。test 未碰。
- 保留：A6000 `emb_rlq/`（20,364 例，716 MB）、`out/`、`qc/`、log。A6000 的 `crops/`（6.9 GB）已於 11-06 刪除；CT-rate `~/OWL/mw_exp/analysis_appendicitis_rlq_tmp/crops/` 仍保留完整一份（20,364 例）。
- 程式：`outputs_final/appendicitis_rlq_unit/scripts/`（rlq_crop／rlq_qc／rlq_extract／rlq_probe／rlq_eval／run_rlq.sh）。

## 目前最佳（骨折，val）
- 正式最佳：實驗 F 骨骼 10 seed 集成 **0.826**／年齡配對 0.804（`outputs_fracture/bones/final_F1/groups/bone_ens.npz`）
- 候選：全骨骼 E3_late 10 seed train-OOF **0.831**／0.809（vs .826 +.004 [−.000,+.009]，未顯著）→ F2 最終整合預計採用（`outputs_fracture/bones/rerun/std/oof/`）
- 全疾病 main_all：E0 0.871、E3_late 0.884（`outputs_final/results/expB/tables_val.md`）

## ▶ 11-05 16:20–16:47 重啟（A6000＋Spark 恢復可用）— 目前在跑的工作
A6000 上別人的 vLLM（Qwen3.8-27B，約 44 GB）16:20 時還在，16:25 已自行結束（不是我們停的）。**若 vLLM 要再開，需要約 44 GB，J 目前在 A6000 用約 11 GB，兩者會撞**：屆時先停 A6000 的 J（見下方停止方式）。
另有非本批的 `scripts/sbo_feat.py`（4 個行程，PID 4062948/59–62）在 A6000 跑，沒有動。

**J（1.5 mm 椎體，三台分工，不重疊）**
- CT-rate 3080：照舊 2 worker，從 `ids_ct_todo.txt` **前面**往後做（含全部 tier0）。
- A6000：**2 worker**，`outputs_vert15/ids_a6_rev.txt`（剩餘 id 反序，每 3 個取 2 個，9,266 例）；loop PID **4099479（w0）、4099480（w1）**。原本開 3 個，但 CT-rate HDD 讀取時間從 1.3 s 升到 4.3 s（3080 慢約 40%），16:37 用 PID 停掉 w2（4099481/4099483），孤兒 claim 已清掉。
- Spark：**3 worker**，`~/OWL/expJ/ids_spark_rev.txt`（反序，每 3 個取 1 個，4,633 例；所需 3 mm mask 已從 CT-rate rsync 到 `~/OWL/expJ/masks3`）；loop PID **3953378/3953379/3953380**，輸出 `~/OWL/expJ/out_vert15`。
- **跨主機 claim 同步**：A6000 `outputs_vert15/sync_claims.sh`（PID **4099849**，每 4 分鐘把三台 `claims/` 聯集互相補上，只新增不刪；log `outputs_vert15/logs/sync_claims.log`）。所以 CT-rate 從前面、A6000/Spark 從後面做，在中間相遇時不會重複。停止：`touch outputs_vert15/STOP_SYNC`。
- worker 指令（A6000，在 mw_exp 下；Spark 同形式，`PY=~/radar_env/bin/python`、`--ids ids_spark_rev.txt --out out_vert15 --mask-dirs /home/daikso_spark/OWL/expJ/masks3`）：
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup bash -c "while [ ! -f 'outputs_vert15/STOP' ] && [ ! -f 'outputs_vert15/DONE_w$i' ]; do ~/miniconda3/envs/radar/bin/python src/vert15_full.py extract --ids outputs_vert15/ids_a6_rev.txt --out outputs_vert15 --mask-dirs outputs/masks outputs_vert15/masks3_extra --wid $i --max-cases 200 --threads 2; sleep 2; done" > outputs_vert15/logs/w$i.r1630.log 2>&1 &`（CT 經 ssh 從 CT-rate HDD 拉）
- 停止：A6000 `touch outputs_vert15/STOP`、Spark `touch ~/OWL/expJ/out_vert15/STOP`、CT-rate `touch ~/OWL/expJ_ct/out_vert15/STOP`（在當前一例做完後結束；要立刻停就依 PID kill loop 再 kill 其 python 子行程，之後刪掉沒有 emb15 的自家 claim 與 `ct/` 殘檔，並在三台都刪同名 claim，否則同步會讓該 id 永遠被跳過）。
- 實測（16:36–16:46）：CT-rate ~1,130 例/h（load 2.6 s，原本單跑 ~1,600/h）、A6000 2 worker ~900 例/h（每 worker ~8 s/例）、Spark 3 worker ~450 例/h（~22 s/例/worker）；合計 **~2,480 例/h**。0 失敗、無 OOM、每 worker GPU 峰值 ~8.1 GB（A6000 實占 ~5.5 GB/worker）。
- 進度（16:46）：ids_ct_todo 17,906 例已完成 4,792；tier0 5,947 已完成 4,347。**ETA：tier0 ~18:15（只由 CT-rate 做）；全部 ~11-05 22:00–22:30**（CT-rate 單跑原估 11-06 01:40）。之後 CT-rate 3080 可交給 cg10k。
- 完成後：合併三台 `emb15/`（A6000 `outputs_vert15/emb15`、Spark `~/OWL/expJ/out_vert15/emb15`、CT-rate `~/OWL/expJ_ct/out_vert15/emb15`），檢查各台 `failed.txt` 與「有 claim 無 emb15」的孤兒，補跑。

**I（逐節監督，A6000 CPU）**：`NQ=3 TH=2 nice -n 19 bash run_expI.sh …/expI/jobs_s1.txt`，runner PID **4096450**（佇列 bash 4096456–58），log `outputs_fracture/bones/expI/logs/queue_s1_1620.out`；已完成的 member 自動跳過。16:46 members 18→22，剩約 10 個（含 3 個 r18 長工作，每個 ~50 分）。ETA ~18:30–19:00。停止：`kill 4096450 4096456 4096457 4096458` 再 kill 各 `bone_expI.py` 子行程，刪 `logs/*.running`。

**膽管擴張（11-05 20:50 更新，agent：biliary_improve）**：
- `bd_oof.py` 已完成（OOF_DONE）。BD1–BD4 已完成，結果 A6000 `outputs_final/biliary_improve/report.md`（最佳 stack train OOF .902、val .899）。
- `bd_hu.py`（PID **4096451**，舊版 ball dilation，~14 s/例，估 11-06 才寫出 `hu_a6.csv`）**已不需要**：同一特徵已用 EDT 版（數值相同）補齊為 `analysis_biliary/hu_all_final.csv`／`biliary_improve/work/hu_all.csv`。可依 PID kill 以釋放 A6000 2 核（沒有動它）。
- ▶ **BD5（待辦，GPU）**：crop 已全部做完並放在 A6000 `outputs_final/biliary_improve/crops/`（20,364）。**等 A6000 上 Exp J 結束**（沒有 `src/vert15_full.py` 行程）後，在 `~/Desktop/Tony/VLM/OWL/mw_exp` 執行：
  `setsid nohup bash outputs_final/biliary_improve/scripts/run_bd5.sh > outputs_final/biliary_improve/run_bd5.log 2>&1 < /dev/null &`
  （腳本會先檢查 J 已結束、`outputs_final/A6000_GPU_LOCK` 不存在；GPU 抽取約 1.5–2 h → probe（CPU，NT=4）約 30 分 → 重跑 stack，結果在 `biliary_improve/stack_bd5/`、`bd5_probe.log`）。
  **與膽結石 GPU 串接的協調**：`gs_gpu_chain.sh` 在 J 結束後只等 45 分鐘讓 `bd5_extract.py` 出現；超過就先跑 GS5/GS6 並建 `A6000_GPU_LOCK`，那時 BD5 要等 lock 消失再跑。

## ▶ 膽結石 GS3–GS6（11-05 18:00 起，agent：gallstones_improve）
**✅ 11-06 04:30 全部完成，所有行程都已結束（A6000 lock 已刪；串接 CHAIN_DONE；Spark gs6_seg 已完成 5,034 例）。** 結果：G6c train OOF 0.879／val 0.870，見 `outputs_final/gallstones_improve/report.md`。下列 PID 只供紀錄用。
目錄：A6000 `outputs_final/gallstones_improve/`（gs3/ gs3b/ gs4/ gs5/ gs6/，程式在 scripts/）。全部 nice 19，只用 train/val。
- **CPU（A6000）**：`gs4_oof13.py`（13 個其他疾病 E3_late OOF，PID 4124654，log `gs4/oof13.log`）、`gs3b_oof.py`（膽囊缺失時用肝臟替代，PID 4133122，log `gs3b/oof.log`）。停止：依 PID kill。
- **CPU（GS5 crop）**：A6000 `gs5_crop.py`（PID 4128836，`gs5/crop_a6.log`）；CT-rate `~/OWL/gs5/gs5_crop.py`（PID 2989941，只讀 SSD 上的 prep/mask，`~/OWL/gs5/crop.log`）。
- **Mac（GS6 raw crop，讀 Passport）**：`gs6_rawcrop.py`（PID 14852，3 process，輸出 `VLM/data/CT/merlin/gs6_rawcrop/out/`，log `rawcrop.log`）；推送迴圈 `gs6_push.sh`（PID 15591，A6000 `gs6/inbox` 最多 600 個；停止 `touch …/gs6_rawcrop/STOP_PUSH`）。
- **GPU 串接（A6000）**：`scripts/gs_gpu_chain.sh`（PID **4133867**，log `gs_gpu_chain.log`）。順序：等 A6000 上沒有 vert15_full.py（J 完成）→ 讓膽管 bd5_extract.py 先跑（最多等 45 分鐘讓它出現，出現就等它跑完）→ 建 `outputs_final/A6000_GPU_LOCK` → GS5 `gs5_extract.py`（約 2 萬個 64³ 膽囊 crop，5 窗，mean+max）→ GS6 `gs6_seg.py`（TS 全解析度 Dataset291，約 5,034 例 1.5 mm crop，`--consume`）→ 刪除 lock。**膽管 agent：A6000_GPU_LOCK 存在時請不要同時跑 GPU 抽取。**
- **22:35 更新（使用者同意）：GS6 改在 Spark GPU 跑**。A6000 串接中的 GS6 步驟已停用（inbox 已改名 `gs6/inbox_unused_moved_to_spark`，放了 DONE_SENDING，串接跑完 GS5 後會直接結束並刪 lock）。Spark：`~/OWL/gs6/gs6_seg.py`（PID **3111251**，log `~/OWL/gs6/seg.log`，結果 `~/OWL/gs6/seg/`、`seg_log.csv`），實測 ~3.5 s/例，5,034 例 ETA ~11-06 03:30。Mac→Spark 一次 rsync 全部 1.5 mm crop（PID 47682，log `VLM/data/CT/merlin/gs6_rawcrop/push_spark.log`）。停止：`touch ~/OWL/gs6/inbox/STOP` 或 kill 3111251（可續跑）。
  GS5 仍排在 A6000：串接（PID 4133867）偵測到 bd5_extract.py 在跑，會等它結束才開始。
  停止：`touch outputs_final/gallstones_improve/STOP_CHAIN`（在兩步之間生效）；要立刻停：`kill 4133867` 後再 kill 當下的 `gs5_extract.py`／`gs6_seg.py` python PID（`pgrep -f gs5_extract.py`、`pgrep -f gs6_seg.py`），並刪 `outputs_final/A6000_GPU_LOCK`。gs6_seg 可續跑（已完成的會跳過）。

**沒有重啟（等使用者決定）**：K（肋骨）、H0、rsna-knee rsync。

## ⚠️ 13:38 已全部終止（釋放記憶體）— 重啟方式
13:38 依使用者指示，所有預計 >1 小時的工作都已 **kill**（不是暫停），A6000/CT-rate/Spark 的 RAM 與 GPU 記憶體已釋放。被殺的 PID 清單在各主機 `~/OWL_killed_pids_1338.txt`（或相近時間戳）。
下表「狀態」欄是被殺前的進度；重啟方式：
- **J（1.5 mm）**：可續跑（已完成的 id 會跳過）。先刪 `outputs_vert15/STOP`（A6000）與 `~/OWL/expJ/out_vert15/STOP`（Spark），再用 agent 原本的 worker 迴圈指令重啟（見 A6000 `ps` 紀錄 / OWL_killed_pids 檔中的 bash -c 迴圈）。可能有半寫的 tmp 檔，重啟前清掉 `*.tmp*`。
  - **11-05 13:52 改在 CT-rate 3080 續跑（A6000/Spark 不動）**：2 worker（`run_vert15.sh`，每命 200 例回收；loop PID 2862372/2862373），`--ct-local` 讀 CT-rate HDD Merlin 原檔，3 mm mask 從 `~/OWL/mw_exp/{outputs/masks,outputs_from_a6000/masks,outputs_vert15/masks3_extra}`。id 清單 `~/OWL/expJ_ct/ids_ct_todo.txt`（17,906 例 = ids_prio_all 扣掉 A6000 1,769＋Spark 707 已完成；前 5,947 為優先層 tier0，全部有 mask），輸出 `~/OWL/expJ_ct/out_vert15/`（emb15/、seg15/）。實測 ~2.4 s/例（~1,500 例/h），20 分鐘 0 失敗、峰值 ~4.8 GB。ETA：優先層 ~17:50、全部 ~11-06 01:40。停止：`touch ~/OWL/expJ_ct/out_vert15/STOP`；續跑：刪 STOP、清 `emb15/*.tmp*` 與殘留 claims（無對應 emb15 的），再跑同一行 `run_vert15.sh`（見 `~/OWL/expJ_ct/README`）。之後合併三台的 emb15 時注意重疊 id（A6000∩Spark 有 18 例）。
- **K（肋骨）**：K1 fold 0 只跑到 epoch 3（inner 肋骨 AUROC 0.60–0.62），進度丟失，需從頭重跑 `rib_hr_train.py`。Spark 前處理（`~/OWL/ribs_hr/run_prep.sh`）可續跑（已完成的 CPR 會保留）；A6000 `sync_cpr.sh` 也已停，需重啟。
- **H0（無 trap 對照）**：訓練進度丟失，需在 CT-rate 重跑 `finetune_bones.py --stage train --run H0_notrap …`（參數見 `~/OWL/data_h/runs/H0_notrap/args.json`）。H1 已完成，不受影響。
- **I**：`run_expI.sh` 與 3 個 `bone_expI.py` 已殺；已完成的 members 保留在 `outputs_fracture/bones/expI/members/`，重跑 jobs 檔中未完成的項目即可。
- **膽管擴張收尾**：`bd_oof.py`（12 個 OOF 變體，只完成 E0/E3_late/E1_soft/E1_liver）與 `bd_hu.py`（hu_a6.csv 尚未寫出，進度全丟）已殺，需重跑。
- **rsna-knee 搬移**：rsync 已送 kill，但當時在寫 USB（D 狀態），會在 I/O 完成後結束；原始 `~/rsna-knee` 完整未動。重跑同一行 rsync 會從已完成的檔案接續（做到一半的大檔會重來）。外接碟是 SMR，很慢（~5 MB/s）。
- **H1 結果（已完成）**：val .813（vs 骨骼集成 −.013 [−.024,−.001]），年齡配對 .785，Spearman .36；椎體壓迫性 .912（最高）但肋骨 .759、骨盆 .784 退步；與 E3_late 等權平均 .832（未顯著）。報告 A6000 `outputs_fracture/bones/expH/report_h.md`。Notion「實驗 H」寫 H0 約 16:00 完成 → 已過時（H0 已終止）。
- **未終止（短時間、仍在跑）**：闌尾炎、腹腔游離氣體、腹水、動脈粥狀硬化錯誤分析（A6000 CPU，`scripts/fa_*`、`ap_*` 等）；H1 評估 agent。
- **Spark 上的 vLLM（Qwen3.8-27B-FP8，root）與 `run_cg10k.py` 不是 OWL 的工作，沒有動。**

## 被終止前的長時間工作（13:30 狀態）

| 實驗 | 主機 / 路徑 | 狀態（13:30） | ETA | 完成後要做 |
|---|---|---|---|---|
| **H0** 全骨骼 finetune（無 trap 對照） | CT-rate 3080；`~/OWL/data_h/run_h.sh`，log `logs/H0_notrap.log`，輸出 `runs/H0_notrap/` | 訓練中（由 run_h.sh 自動接在 H1 後） | ~16:00 | 把 `pred_val.npz` 複製到 A6000 `outputs_fracture/bones/expH/`，跑 `src/eval_h.py --pred H1=… --pred H0=…`；更新 Notion「實驗 H」 |
| **H1 評估**（finetune＋trap） | A6000 `outputs_fracture/bones/expH/H1_pred_val.npz` | 評估 agent 執行中 | 快完成 | 看 Notion「實驗 H」段落；注意：訓練時未選中單元用凍結 cache，emb 漂移 23–26 SD（train/inference 不一致） |
| **J** 1.5 mm 椎體 encoder 全量 | A6000 `src/vert15_full.py extract`（4 worker）→ `outputs_vert15/`；Spark `~/OWL/expJ/out_vert15`（3 worker） | 已抽 ~2,100 例（A6000 1,528＋Spark 589）；優先層 `outputs_vert15/ids_prio.txt` | 優先層 ~20:00；全量（20,364）明天下午 | 可續跑（已完成的 id 會跳過）。優先層完成後做中期評估：1.5 mm 椎體單元併入依部位 MIL＋Genant 幾何，比 bone_ens .826（壓迫 .905） |
| **I** 逐節監督＋困難陰性 | A6000 CPU，`src/bone_expI.py`（--variant r18/rl1/lrc…）→ `outputs_fracture/bones/expI/` | 訓練中 | 未定 | 結果寫入 Notion「實驗 I」 |
| **K** 肋骨高解析度路徑 | 前處理 Spark `~/OWL/ribs_hr/`（`run_prep.sh`，3 HDD＋2 Azure worker）；A6000 `sync_cpr.sh` 每 3 分同步（`touch STOP_SYNC` 停）；訓練 A6000 PID 3980381 `outputs_fracture/ribs_hr/runs/K1.out`（K1 5-fold，只用 train tier0 2,003 例，`--no-val`） | 13:24 開始 K1；val tier0 前處理 ~14:40、全部 val ~16:40；每 fold 40–60 分 | K1 ~17:30 | ① val 備齊後（在 A6000 mw_exp 下）`python src/rib_hr_train.py --table outputs_fracture/analysis_rib/table_parsed.pkl --cpr outputs_fracture/ribs_hr/cpr --split ../data/split.csv --out outputs_fracture/ribs_hr/runs/K1 --lstride 2 --crop 24 --predict-ids outputs_fracture/ribs_hr/ids_prio.csv`（fold 0 第 1 epoch inner 肋骨 AUROC 0.603，162 s/epoch） ② `rib_hr_eval.py --run outputs_fracture/ribs_hr/runs/K1 --table outputs_fracture/analysis_rib/table_parsed.pkl` ③ Spark 有 5 例孤兒要重啟一個 worker 才會補 ④ 寫 Notion「實驗 K」。負責 agent 已交接停止，需再派人接手 |
| **膽管擴張錯誤分析收尾** | A6000 `outputs_final/analysis_biliary/`：PID 3909041（bd_oof.py，12 個 OOF 變體）、PID 3928591（bd_hu.py，寫 hu_a6.csv） | 執行中 | ~15:50 | `nice -n 19 env OMP_NUM_THREADS=1 ~/miniconda3/envs/radar/bin/python scripts/bd_analyze.py` → 寫 report.md → Notion 子頁「膽管擴張：錯誤分析」 |
| **錯誤分析 ×5** | A6000/CT-rate CPU；`outputs_final/analysis_{appendicitis,free_air,osteopenia,ascites,atherosclerosis}/` | 執行中 | 未定 | 各自寫 report.md＋Notion 子頁 |
| **rsna-knee 搬移** | A6000 `~/rsna-knee`（137 GB）→ `"/media/advantech/My Passport/rsna-knee/"`；log `~/rsna_knee_move.log` | rsync 12%，外接碟為 SMR，只剩 ~6 MB/s | ~18:30–19:00 | 比對檔案數/bytes＋checksum（或抽樣），全部通過才 `rm -rf ~/rsna-knee`；原始檔目前完整保留 |

## 已完成（結果都在 Notion「骨折 AUC 提升總結」與 OWL 主頁子頁）
- 骨折：實驗 B（骨骼單元，organ_map 已改，備份 `*_preB.yaml`）、C（1.5 mm 試驗）、D（腰椎 finetune，無效）、E（標籤，無效）、F 第一部分、G（單根肋骨，無效）、全骨骼重跑先前實驗。
- 錯誤分析：壓迫性骨折、肋骨/骨盆、陳舊骨折、膽結石、脂肪肝。
- 快速改進（HU 特徵堆疊，不含年齡／性別，train OOF 選模）：膽結石 E3_late＋膽囊 HU（GBM）OOF .782→.835、val .773→.821（Δ+0.048，CI +0.030～+0.067）；脂肪肝 ＋HU（GBM）OOF .819→.840、val .818→.825（CI 跨 0），肝‖肝−脾 版 val .830。A6000 `outputs_final/quickwins/`，Notion 主頁 9.7 節。

## 待決定事項
1. **脂肪肝標籤定義**：局部脂肪浸潤（佔陽性 27.6%）要不要移出陽性？（移出後約 0.819 → 0.90）必須在看 test 前鎖定。
2. **低成本改進**（都只需 CPU）：膽結石膽囊 HU 特徵堆疊（+0.045）、脂肪肝 liver‖(liver−spleen)＋HU 特徵（+0.021）、膽管擴張肝門特徵（+0.019）。
3. **磁碟**：A6000 剩 ~44 GB。可清：A6000 `outputs/ts_engine_old`（2.2 GB）、CT-rate `~/OWL/mw_exp/outputs_from_a6000`（15 GB）；`outputs/prep`（38 GB）可搬到 CT-rate `/media/daikso/CTDATA`（11 TB 空）。
4. **F2 最終整合**：等 H、I、J、K 都出來後，用 `src/final_stack.py` 合併（成員需 train OOF；沒有 OOF 的只能做固定權重平均）。
5. 外接碟是 desktop 自動掛載，重開機需登入或設 fstab（未設定）。

## 監控
- 看門狗：`/private/tmp/claude-501/.../scratchpad/watchdog.sh`（每 15 分鐘；GPU 閒置＋有 GPU 程序、磁碟 < 15 GB、主機斷線、無檔案活動時告警）。暫停時已停止。
