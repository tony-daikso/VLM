# spine15_ft：真 1.5 mm 逐節椎體 crop 微調 TS encoder（11-07）

- 資料：原始 Merlin CT（CT-rate CTDATA＋Mac Passport 中繼）直接以三線性內插切 1.5 mm crop，T8–L5 每節 40×48×40（60×72×60 mm，與 spine_ft 同視野與方向），中心／遮罩用 Exp J seg15（Dataset292 1.5 mm 椎體標籤；缺者用 TS 3 mm 遮罩）。不再做 3 mm→×2 上採樣。腳本 `spine15_ft/scripts/spine15_crop.py`。
- 模型：spine_ft 配方（TS Dataset297 encoder stage ≥2 微調，3 窗，stage3/4 mean+max+遮罩 mean → 每節 logit＋每節 bias → logsumexp MIL，8 epoch OneCycle，inner 12% holdout 選 epoch）。候選 tf2lse（原配方）與 tf2ctx（加相鄰椎體差值特徵；3080 bf16），各 1 seed，只看 train OOF（tier 列 t_vert）選：tf2lse .9382 > tf2ctx .9314 → tf2lse。依協調者 14:45 指示（使用者改專注肋骨／骨盆）不再加 seed，最終＝tf2lse 單 seed（A6000 fp32 訓練，5 個 fold 模型；val＝5 fold 平均）。
- folds／labels 同 spine_ft（`spine_ft/in/labels.csv`，sitewise 外層 folds；tierA＝train 全部骨折＋4,000 陰性）。val 只在本報告評一次；test 未讀取。
- 有分數：train 15055，val 4960（另 3 例無任何椎體 unit）。spine_ft 對照＝3 mm×2 上採樣版 final3_tf2lse（3 seed 平均）。

| 資料 | 目標 | n | 陽性 | spine15（1.5 mm） | spine_ft（3 mm×2，3 seed） | Δ [95% CI] |
|---|---|---|---|---|---|---|
| train OOF | 椎體骨折（site 標籤）·tier 列（選模指標） | 5010 | 1011 | 0.938 | 0.934 | +0.004 [-0.003, +0.011] |
| train OOF | 椎體骨折（site 標籤）·全部 train | 14391 | 1011 | 0.937 | 0.934 | +0.003 [-0.003, +0.010] |
| train OOF | 整體骨折 | 15055 | 1675 | 0.820 | 0.828 | -0.008 [-0.016, +0.000] |
| val | 椎體骨折（site 標籤） | 4726 | 322 | 0.939 [0.922, 0.954] | 0.930 | +0.008 [+0.001, +0.017] |
| val | 整體骨折 | 4960 | 556 | 0.824 [0.801, 0.844] | 0.826 | -0.003 [-0.011, +0.006] |

Δ 為同列配對分層 bootstrap（2,000 次）。整合進骨折 stack 的結果見 `final_ft/<tag>/report.md` 與本目錄 `integration.md`。
