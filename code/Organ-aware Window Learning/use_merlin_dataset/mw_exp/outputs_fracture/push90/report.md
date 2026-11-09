# push90：骨折 → val .90（agent push90，11-08 17:20 – 11-09 00:35）

規則：只用 train/val（split 先過濾再讀標籤）；所有選擇只依 stacker train CV（官方標籤）；val 只在選完後報告一次；test 未讀取；官方標籤未改。

## 結果摘要
- **依 train CV 選出 rp_a6mix3**（spine15＋肋骨 z 平均[RV3 lev28＋RS15＋A6]＋後方附件 PE ctx head@椎體＋pelvis15_ft；S3n·lse）：train CV **.9014**（rsP_mix_repl .8966）。
- val **.897 [.881, .911]**；Δ vs .892 +.004 [+.002, +.007]、vs .882 +.014 [+.009, +.020]、vs .862 +.035 [+.027, +.043]；年齡配對 .889（Δ vs .892 +.005 [+.002, +.009]）。**未達 .90**。
- 子群：椎體壓迫性 .954（−.002）、肋骨 .885（+.014）、只有肋骨 .854（+.017）、骨盆/薦椎/髖 .896（+.003）、急性 .894（−.001 n.s.）、陳舊 .896（+.006）。表：`integ/rp_a6mix3/compare_vs_892.md`；選擇：`p90_selection.json`。

## A 肋骨（拉直 tile 凍結特徵；tier 2,005 列 3 seed，t_rib OOF side；基準 .843）
| 設定 | tier |
|---|---|
| a0 base+bone st0–1（重現） | .838 |
| a1 4 窗 st0–1 | .848 |
| a3 4 窗 st0–2 | .858 |
| a2 base+bone st0–2 | .867 |
| v2 base+bone st0–3 | .862 |
| v2 st2–3 only | .825 |
| v2 st0–2 mean+max | .851 |
| a4 st0–1 ＋相鄰/對側同格對比（--ctx，tile 鄰居） | .881 |
| a5 st0–2 ＋ctx | .885 |
| **a6 st0–2 mean+max ＋ctx** | **.887** |
- soft/wide 窗無益（維度過高）；stage 2 與對側/相鄰肋骨對比是主要增益。
- 全量 a6（5 seed，20,009 例）：t_rib OOF side **.8996**（RS 5 seed .8846、RS15 .886、RV3 .859）→ `cand/a6_side`。

## B 後方附件（T8–L5 1.5 mm crop 20,312 例，Dataset292 st0–2，每節左右×{椎體、椎弓、橫突、棘突}）
- t_pe 弱標籤：train 報告提到橫突/棘突/pars/椎弓等骨折的官方陽性 255 例；陰性＝官方 0；其他陽性忽略。
- tier（2,355 列）t_pe OOF：plain .823、ctx .817、side/lev .812；tier 堆疊增益 +.009～+.012。
- 全量 t_pe OOF：pf_ctx（5 seed）head .796 / side .802；pf_plain .787；pf_ctx_s5 .761（seed 變異大）。
- 整合（rsP_mix_repl＋PE@椎體）：ctx head .8981、ctx10 head .8981、ctx side .8980、mix .8974、plain .8966。

## C train 端雜訊降權（評估用官方標籤）
- stacker：c_fn0 .8964、c_fn0fp03 .8961、c_all03 .8965（基準重現 .8966）→ 無益。rib probe 版在 Spark 太慢（~10 h）已停，未完成。

## 全部整合變體（train CV）
見 `p90_selection.json`：rp_a6mix3 .9014、rp10_a6mix3 .9014、rp_a6add .9013、rp10_a6add .9013、rp_a6repl .9008、r_a6mix3 .8999、r_a6add .8998、rp_a6only .8994、r_a6repl .8993、pe_ctx10h .8981、pe_ctxh .8981、pe_ctxs .8980、r_a6only .8977、pe_mix .8974、rsP_mix_repl .8966、pe_plain .8966、C 系列 .8961–.8965。

## 檔案
scripts/（rs_feat2.py、rs_probe2.py、rs_probe2_f.py、pe_labels.py、final_ft2.py、zmix.py、cmp_val.py、p90_select.py、pe_tier_eval.py、各 lane sh）；A/feats_v2（23 GB，st0–3 base+bone）、A/feats_tier（2.3 GB）；B/feats（PE 特徵）；prod/f_a6、B/prod；cand/；integ/。CT-rate `/media/daikso/Tony/VLM/data/CT/merlin/push90/pe15/crops`（~37 GB PE crop）。Spark `~/Desktop/Tony/OWL/push90/A/feats_v2`（23 GB 副本，可刪）。
