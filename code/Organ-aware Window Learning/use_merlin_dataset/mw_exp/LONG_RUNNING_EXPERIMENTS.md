# 長時間實驗紀錄（2026-11-06 10:45 更新：bd15 在 A6000 執行、L／O 完成；13:30 曾全部暫停）

規則：所有實驗只用 train/val，test 不碰（先依 split 過濾再讀標籤，禁止 `head` 標籤 jsonl）。
主機：A6000 = `ssh A6000`（NVIDIA IGX Orin + RTX A6000，aarch64，12 核），工作目錄 `~/Desktop/Tony/VLM/OWL/mw_exp`；
CT-rate = `ssh CT-rate`（RTX 3080 10 GB），`~/OWL/mw_exp`；Spark = `ssh sparks`（GB10，PyTorch 不支援 sm_121，nnU-Net 類運算慢 5–6×）。



## ✅ 11-06 10:45 更新
- 實驗 L 最終評估、實驗 O 收尾都已完成（A6000 run_final.sh／no_rescue_a6.sh 已結束，`A6000_GPU_LOCK` 不存在）。結果見 GPU_QUEUE「完成」與 `outputs_fracture/final_combo/report.md`（骨折 I＋J＋L＋骨盆單元合併）。
- 清理：A6000 `region_units/bowel_obstruction_r2/crops{,_ct}`（3.3＋9.0 GB）與 CT-rate `~/OWL/mw_exp/region_units_tmp/*_r2/crops`（9.0 GB）已刪（刪前確認 20,364 個 emb 齊全、每個 crop id 都有 emb）。`sync_spark.sh` 兩份重複迴圈已停（`outputs_ribs15/STOP_SYNC_SP`）。
- r2 單元報告：A6000 `outputs_final/region_units/{bile_duct,free_air,ascites,fracture_pelvis,bowel_obstruction}_r2/report.md`（Mac 鏡像同路徑；產生器 `scripts/no_report.py`，`no_table.py` 已修正語法錯誤與子群欄位重複）。

## ▶ 11-06 09:00 狀態更新（restart 後 status agent）
- **4090 GPU 已故障**：約 06:58 起 `CUDA_ERROR_LAUNCH_FAILED`，`nvidia-smi` 顯示 `Unable to determine the device handle … Unknown Error`（容器內無法重置；需使用者在 glows.ai 重啟或釋放執行個體）。4090 上已無 OWL 行程（Exp L worker／下載器已因 STOP 結束；Exp O 卡住的 no_extract 已 kill）。`/root/OWL/EXP_O_RUNNING` 仍在。
- **實驗 L 抽取完成**：A6000 `outputs_ribs15/emb15` 20,355／20,355（4090 的 emb15 已由 sync 併入；Mac `sync_ribs15.py` 已不需要）。A6000／Spark worker 都已結束，Spark GPU 閒置。`sync_spark.sh`（PID 772864、776518，兩份重複）仍在空轉，可 `touch outputs_ribs15/STOP_SYNC_SP` 停。
- **實驗 L 最終評估**（A6000，09:04 起）：`outputs_fracture/expL/run_final.sh`（bash PID **1233148**，log `outputs_fracture/expL/final/run_final.log`）：重建 l_matrix（r15 20,010／20,018）→ 全 train 5 seed OOF（L_add+rl03、L_repl+rl03+lrc、L_repl+rl03、Jadd 參考；`--device cuda`，4 個 process，每 fold 約 100 s）→ `rib15_eval.py`（`final/lrep/`）→ `final_stack.py` I 集成＋J add＋L（`final/SL_<v>/report.md`，含 vs J add 10 seed .847 配對 bootstrap）；依 stacker train CV 選。ETA ~10:30–11:00。執行期間建立 `outputs_final/A6000_GPU_LOCK`（OOF 結束自動刪）。停止：`touch outputs_fracture/expL/final/STOP`（步驟間生效）；立即停：`kill 1233148; pkill -f "rib15_probe.py --stage oof"`，再刪 lock。續跑：重跑同一腳本（已存在的 OOF 檔會跳過；build 會重做）。
- **實驗 O 改在 A6000 收尾**：Mac driver（7768/7769）、feed、post 已停（`expO_stage/STOP_DRIVER`、`bowel_obstruction_r2/STOP_FEED` 已建）。A6000 `outputs_final/region_units/scripts/no_rescue_a6.sh`（bash PID **1235871**，log `region_units/rescue_o_a6.log`）：從 CT-rate（只讀 CPU/磁碟）拉 13,775 個胃出口／十二指腸 crop 到 `bowel_obstruction_r2/crops_ct/` → A6000 GPU `no_extract.py`（8 GB／3 threads，已有 emb 跳過）→ `no_probe.py frpp`（骨盆分部位）→ `no_stack.py fracture_pelvis_r2` → `no_probe.py go` → `no_stack.py bowel_obstruction_r2`。ETA ~11:00。停止：`touch outputs_final/region_units/STOP_RESCUE_O`（步驟間），立即停 `kill 1235871; pkill -f no_extract.py`；可續跑。完成後可刪 A6000 `bowel_obstruction_r2/crops{,_ct}` 與 CT-rate `region_units_tmp/*_r2/crops`。

## ▶ 項目 5 膽管擴張 1.5 mm 肝門 crop（bd15）— 11-06 10:35 起在 A6000 執行（原 Spark 計畫 09:30 取消，改 A6000）
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
