import sys
B = "outputs_fracture/rib_straight/logs/"
gq = open("GPU_QUEUE.md").read(); bak = open(B + "GPU_QUEUE.md.bak_1108").read()
lines = [l for l in gq.split("\n") if "rib_straight 肋骨拉直" not in l]
assert "\n".join(lines) == bak, "GPU_QUEUE changed by someone else"
entry = ("- **rib_straight 肋骨拉直 tile＋encoder 微調（agent rib_straight，11-08 07:40 起）— A6000 GPU（與 pelvis15_ft 共用，fp32，~5 GB，200 W 未動，"
         "`rs_guard.sh` >85°C 暫停）；pelvis15_ft 釋放 3080／Spark 後再接手**：Spark CPU 由 rib_ft/roi 建拉直 tile（每根肋骨中心線、1.5 mm 32×32 截面；"
         "`~/Desktop/Tony/OWL/rib_straight/tiles`，副本 A6000 `outputs_fracture/rib_straight/tiles`、CT-rate `/media/daikso/Tony/VLM/data/CT/merlin/rib_straight/tiles`）"
         "→ `rs_ft.py`（spine15 tf2lse：stage 2–4 微調，64 列 chunk、24 mm 弧段 instance lse MIL，side .5＋lev .3）→ tier（`in/ids_tier.txt` 2,005 列，t_rib，5 fold）"
         "比 rib_v3 → 有增益才全量。停止 A6000 `touch outputs_fracture/rib_straight/runs/STOP`。細節見 LONG_RUNNING「rib_straight」。\n")
k = bak.index("## 進行中\n") + len("## 進行中\n"); open("GPU_QUEUE.md", "w").write(bak[:k] + entry + bak[k:])
lr = open("LONG_RUNNING_EXPERIMENTS.md").read(); lbak = open(B + "LONG_RUNNING.md.bak_1108").read()
i0 = lr.index("## ▶ rib_straight"); i1 = lr.index("## ▶ pelvis15_ft")
assert lr[:i0] + lr[i1:] == lbak, "LONG_RUNNING changed by someone else"
sec = """## ▶ rib_straight（肋骨拉直 tile＋encoder 微調，agent rib_straight，11-08 07:40 起；Spark CPU＋A6000 GPU〔與 pelvis15_ft 共用〕）
- 目標：肋骨 site 分數（rib_v3 lev28 t_rib train OOF .859）→ final_ft 整合（rp_both_aa .882）。只用 train/val；選擇只看 train OOF（t_rib）／stacker train CV；test 不讀。
- tile：Spark `~/Desktop/Tony/OWL/rib_straight/scripts/rs_tile.py`（rib_ft/roi 1.5 mm＋Dataset295 標籤 → 每根肋骨：體素圖測地雙掃描求中心線〔碎片 ≤30 mm 橋接〕→ 1.5 mm 弧長重取樣 → 後端在前、兩端外延 8 列 → 32×32 截面〔u＝上方投影、v＝T×u〕三線性取樣；ROI 外＝最近邊緣值）→ `tiles/{id}.npz`（hu、m 肋骨遮罩 packbits、tab〔rib,start,L,m0,n,nvox〕、oor、cl）。
- 訓練：`outputs_fracture/rib_straight/scripts/rs_ft.py`（pv15_ft 改寫；chunk 64 列 stride 48、chunk 數補到 8 的倍數〔固定 cudnn 形狀，否則慢 6×〕；instance＝16 列〔24 mm〕弧段；stage3 mean/max/肋骨遮罩 mean＋stage4 mean/max → LN → linear＋肋骨 bias＋弧位 bias → lse；左右 bag side .5、層級 lev .3；每 epoch ckpt 可續跑；A6000 不用 pin_memory）。`rs_tier.sh`（每 run 最多 2 次）、`rs_guard.sh`（>85°C SIGSTOP rs_ft.py）。
- 停止：A6000 `touch outputs_fracture/rib_straight/runs/STOP`（當前 epoch 後）；Spark tile `touch ~/Desktop/Tony/OWL/rib_straight/STOP`。

"""
k = lbak.index("## ▶ pelvis15_ft"); open("LONG_RUNNING_EXPERIMENTS.md", "w").write(lbak[:k] + sec + lbak[k:]); print("docs ok")
