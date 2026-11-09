p = "GPU_QUEUE.md"; s = open(p).read(); L = s.split("\n")
i = [k for k, l in enumerate(L) if "rib_straight 肋骨拉直" in l]; assert len(i) == 1; L.pop(i[0]); s = "\n".join(L)
done = ("- **rib_straight 肋骨拉直 tile（agent rib_straight，11-08 07:40–17:20；A6000／Spark／3080 GPU 全部已釋放）**：20,009 例拉直 tile（0 失敗）。encoder 微調 tier 失敗"
        "（295 .788、297 fold0 .749 ＜ ROI probe .825）→ 改用拉直 tile 凍結特徵 probe：全 train t_rib OOF .886（rib_v3 .859）。整合依 train CV 選 **rsP_mix_repl**"
        "（spine15＋[RV3＋RS15]@rib＋pelvis15_ft 取代 PV15）：train CV .8966，val **.892 [.876, .907]**，vs .882 +.010 [+.006, +.014]；年齡配對 .884；"
        "肋骨 .872（+.019）、只有肋骨 .837（+.023）、骨盆 .893（+.026）、椎體壓迫性 .956。報告 `final_ft/rsP_mix_repl/{report.md,compare_vs_882.md}`，"
        "`rib_straight/rs_selection.json`。train-only 標籤稽核 `rib_straight/label_audit/report.md`。test 未使用。")
k = s.index("## 完成\n"); k = s.index("\n", s.index("（agent 完成後", k)) + 1; s = s[:k] + done + "\n" + s[k:]; open(p, "w").write(s)
p = "LONG_RUNNING_EXPERIMENTS.md"; s = open(p).read()
s = s.replace("## ▶ rib_straight（肋骨拉直 tile＋encoder 微調，agent rib_straight，11-08 07:40 起；Spark CPU＋A6000 GPU〔與 pelvis15_ft 共用〕）",
              "## ✅ rib_straight（肋骨拉直 tile＋encoder 微調，agent rib_straight，11-08 07:40–17:20 完成；無程序在跑，GPU 全部已釋放）\n"
              "- **結果**：微調 tier 失敗（295 .788／297 fold0 .749 ＜ ROI probe .825）；拉直 tile 凍結特徵（`scripts/rs_feat.py` → `feats/`〔12 mm〕、`feats_c6/`〔9 mm〕）＋rib_probe3"
              "（`scripts/rs_prod.sh` → `prod/`）全 train t_rib OOF .886；`cand/{rs15,mix15,mix3,rs6}`；final_ft 依 train CV 選 rsP_mix_repl .8966 → val .892 [.876, .907]（vs .882 +.010）。"
              "整合腳本 `scripts/rs_integ*.sh`、選擇 `scripts/rs_select.py`（寫 `rib_straight/rs_selection.json`，不動 final_ft/rp_selection.json）。")
open(p, "w").write(s); print("ok")
