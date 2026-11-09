# 膽結石：1.5 mm 膽囊 crop 單元（gs15）＋小型 3D CNN

主機：A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/gallstones15/`（Mac 鏡像同路徑）；crop 由 A6000 CPU（原始 CT 以 ssh 從 CT-rate HDD 唯讀讀取）與 Mac CPU（Passport）產生，embedding／probe／CNN 用 A6000 GPU。
**test 未讀取、未評分**：標籤檔只含 train/val（程式內 assert），先依 split 過濾再用標籤；模型只依 train 5-fold OOF（GS1 folds）選，val 只報告。官方膽結石標籤定義不變。CT-rate 3080 與 Spark 都沒有用。

## 方法
- **Crop**（`scripts/gs15_crop.py`）：原始 CT → canonical RAS → 抗混疊 Gaussian → 三線性重取樣到 96³×1.5 mm（144 mm）。中心：TS 3 mm 膽囊 ≥100 voxel 用其質心；否則 GS6 1.5 mm 膽囊 ≥800 voxel 用其質心；否則肝 bbox 比例 [0.3355, 0.7002, 0.6623] 的膽囊窩點（同 Exp N）。膽囊遮罩：GS6 病例用 GS6，其餘用 TS 3 mm（線性、≥0.5）。
- **區域**（5 個）：r0 膽囊膨脹 3 mm、r1 膽囊窩 30 mm ROI（扣骨／血管／泌尿／腸與肝核心，同 GS5 fossa30）、r2 重力依賴側 1/3、r3 中心 20 mm 球、r4 膽囊內 ≥90 HU。
- **視窗**（6 個）：baseline、軟組織 40/400、窄窗 60/150、40/250、結石窗 200/500（高 L）、低密度窗 10/80（膽固醇結石）。TS Dataset297 凍結 encoder stage 1–3，mean＋max → [1, 5, 6, 896]。
- **1.5 mm HU 特徵**（`feats.csv`）：膽囊體積、百分位、max、≥60/100/130/200/300 HU 體積、≤−10 HU 與「低於膽汁中位數 25 HU」的膽固醇候選體積、最大高密度 blob、依賴層−非依賴層中位數差、膽囊壁、膽囊窩 ROI 的高密度體積。
- **單元頭**（`gs15_probe.py`）：E3_late 式 late head（同 Exp N），U（5 區域）、U_org（＋主 TS 膽囊／肝單元）、U_gb3（只 r0–r2）。
- **3D CNN**（`gs15_cnn.py`）：64³ 中心、3 通道（軟組織窗、窄窗、膽囊遮罩），小型 3D ResNet，GS1 5-fold OOF＋全 train 模型評 val，2 seed 平均。
- **堆疊**（`gs15_stack.py`，機制複製自 `nu_gs_stack.py`／`sl_stack.py`，列順序固定以完全重現 Exp N）：完整重堆疊（N_U 特徵＋新頭／HU／CNN，7 個 LR／GBM 候選）與第二層（官方 N_U OOF logit＋新頭）兩類，全部依 train OOF 選。Δ 是對官方 Exp N 分數（.881）的配對分層 bootstrap。
