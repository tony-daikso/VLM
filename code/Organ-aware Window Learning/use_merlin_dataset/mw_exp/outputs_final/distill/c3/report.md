# 蒸餾 C3（膽結石）：GS1-V4 stack → 只用 embedding 的膽囊 probe

**作法**：老師＝GS1-V4 stack（E3_late＋膽囊 HU＋膽囊缺失/體積/手術夾＋膽囊切除 head＋3 mm 紋理；train OOF .866、val .859）。學生＝只看膽囊單元 embedding 的 E3_late probe（`analysis_gallstones/scripts/gs_oof.py` 同一套：probe.py `late`、膽結石＋膽囊切除兩個 head、lr 1e-3／wd 1e-4、train 內 15% 早停、StratifiedKFold(5, rs=0)、seed 42/43 平均）。學生的膽結石目標＝α·真實標籤＋(1−α)·老師機率（老師 train OOF logit 以 train 做 Platt 校正，溫度 T）。α=1 即原本的 hard label。只看 train OOF 選 α；val 只報告；test 未讀。

| 模型 | train OOF | val AUROC | 年齡配對 | Δ vs hard（同程式） | Δ vs 老師 V4 |
|---|---|---|---|---|---|
| 老師 GS1-V4 stack | .866 | **.859** | .846 | +.085 [+.063, +.105] | — |
| 原 E3_late（gs_oof.py） | .782 | .773 | .749 | −.001 [−.011, +.010] | −.085 |
| hard（α=1，本程式重跑） | .785 | .774 | .753 | — | −.085 [−.105, −.063] |
| 學生 α=0.3（train OOF 最高 → 選定） | **.794** | .759 | .734 | **−.015 [−.029, +.000]** | −.100 |
| 學生 α=0.5 | .789 | .755 | .730 | −.019 [−.031, −.006] | −.103 |
| 學生 α=0.7 | .786 | .758 | .734 | −.016 [−.026, −.007] | −.101 |
| 學生 α=0.5、T=2 | .792 | .757 | .731 | −.017 [−.028, −.007] | −.102 |

val N=4,974（膽結石 464）。

**判讀**：stack 的增益（+.085）**幾乎都來自 HU／缺失／手術夾等 embedding 之外的特徵**，無法搬進只看 embedding 的 probe。學生在 train OOF 上看起來變好（.785 → .794），但 val 反而變差（−.015～−.019）。原因：老師的 train OOF 機率帶有其他 fold 標籤的資訊（巢狀洩漏），學生學到的是「模仿老師」而不是新的影像訊號。**部署上仍需要 HU 特徵（或 GS5/GS6 的高解析度膽囊路徑），蒸餾不是替代方案。**

檔案：A6000 `outputs_final/distill/c3/`（`c3_a*_T*.csv` 每例 train OOF／val 分數、`report.md`、`distill_c3.py`）；輸入 `gs1/gs1_oof_train.csv`（唯讀）。
