# pelvis_ft：骨盆 crop 微調 TS encoder（11-07）

- 資料：3 mm prep＋TS mask 切 7 個 24³ crop（右/左髖臼＋股骨頭、恥骨聯合、右/左薦椎上 60%、右/左股骨近端 70 mm），GPU 上 ×2 上採樣；19,976／20,018 有 crop（缺者整合時以 train 中位數補）。
- 模型與流程同 spine_ft（tf2，3 窗，lse MIL），目標＝骨盆/薦椎/髖部位標籤（t_pel），train 陽性 309。
- seed 0 train OOF .825、seed 1 .816（tier，t_pel）；2 seed 平均 FT_pv train .830／val .873（t_pel）。單獨不如凍結的 U_org_pel（train .877），整合時只帶來很小的骨盆子群增益（+.002 n.s.）。
- 輸出：`runs/final_tf2lse/{oof_train.csv,val_scores.csv,summary.json}`。
