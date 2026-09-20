# RADAR 移植到 X-ray（PadChest）— PoC 執行計畫

## 現況掌握

### RADAR 架構核心

程式碼位置：`RADAR/CT/RADAR_train/lavis/models/radar_models/`

- `VisionBranch`：3D UNet（`PlainConvUNetLightD`，Conv3d/BatchNorm3d），輸入 CT volume，同時輸出 36 個腹部器官的分割遮罩，以及 3 個尺度的特徵圖。用分割遮罩對特徵圖做 max-pool，取出「哪些 token 屬於哪個器官」的布林遮罩。
- 訓練時交替兩種對比學習（RADAR+ 逐 iteration 切換）：
  - **anatomy-wise ITC**：用器官遮罩把影像 token 透過 cross-attention 池化成器官級向量，跟 LLM 從報告解析出的器官級敘述做對比學習，並用 normal/abnormal flag 建構更細緻的 soft target。
  - **whole-image ITC**：整張影像 vs 整份報告的全域對比。
  - 再加上分割 Dice loss（輔助任務）。
- 器官級文字監督來自三段式 LLM pipeline：`check_organ_mention.py → report_parsing.py → report_parsing_normal.py`，從自由文本報告抽取「哪些器官被提到」「怎麼描述」「正常/異常」。

### PadChest 資料現況

資料位置：`data/X-ray/PadChest-Origin`

- 已更新為只保留 PA/AP 的完整資料（`PNG_tar/PADCHEST_PA_AP.tar`，約 37GB，共 96,270 張 PNG），取代原本只有 1/54 shard 的狀態，不再需要另外下載其餘 shard。
- Label CSV（`other/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz`）已經是**結構化資料**：`Labels`（英文 finding 清單）、`Localizations`（部位詞，如 `loc basal bilateral`）、`LabelsLocalizationsBySentence`（finding 與 location 的配對）都已經是英文、句子級對齊的結果。這點比 CT 端幸運很多——CT 端要靠三段 LLM 才能拿到的「器官級 finding + 正常異常判斷」，X-ray 端用 rule-based 規則就能從既有欄位直接組出來，**可以先跳過 LLM 那一步**。
- 原始 `Report` 欄位是西班牙文且已 lemmatize（不是流暢句子），不建議拿來當文字監督；用 `Labels`/`Localizations` 組模板句更穩，也能直接沿用英文 `bert-base-uncased`，不用換 multilingual 模型。
- `RADAR/X-ray/` 目錄目前是空的，`RADAR/CT/` 是重整後的 vendored 原始碼——repo 看起來已經為這次任務先騰好位置。

## 核心差異與挑戰

1. **3D → 2D**：`VisionBranch` 用的是 Conv3d/BatchNorm3d 的 `PlainConvUNetLightD`，需要換成 2D 版本（底層 `dynamic_network_architectures` 是 nnU-Net 系列，理論上抽換 `conv_op`/`norm_op`/`kernel_sizes`/`strides` 為 2D 即可，不用重寫 UNet 本身）。`forward()` 裡的 `max_pool3d` anatomy token pooling 也要改成 `max_pool2d`。
2. **沒有 X-ray 版的 `checkpoint_unet.pth`**：CT 端靠 TotalSegmentator 產生 36 個腹部器官 mask 去 pretrain 分割 backbone。X-ray 端沒有現成對應的胸腔解剖分割 checkpoint，需要：
   - 方案 A：用 CheXmask（公開的大規模 CXR 解剖分割資料集，涵蓋 PadChest，用 HybridGNet 產生的 lung/heart/clavicle 遮罩）取代 TotalSegmentator 產生 mask 的角色。
   - 方案 B：用現成的胸部 X 光分割模型（如 [ianpan/chest-x-ray-basic](https://huggingface.co/ianpan/chest-x-ray-basic)）現場推論。
   - 建議先用方案 A，PadChest 剛好在 CheXmask 覆蓋範圍內。
3. **解剖區域數量大幅縮減**：CT 用 36 個腹部器官，X-ray 只能提供有限的解剖分區（例如：左肺、右肺、心臟/縱膈腔），比較細的定位靠 `Localizations` 欄位的 `loc apical/basal/bilateral` 等詞輔助。
4. **報告解析大幅簡化**：PadChest 已提供 `Labels`、`Localizations`、`LabelsLocalizationsBySentence`，可以用規則式方式把 finding 分派到對應解剖區域，不需要 LLM 三段式 pipeline（非必要時可跳過）。
5. **Text encoder 語言**：直接用現有的英文 `bert-base-uncased`，靠 `Labels` 生成的模板 caption（例如 `"pulmonary fibrosis in bilateral basal lung"`）取代原始報告文字。
6. **RADAR+ 全域對比**：用 `Labels` 全部串接成一段英文句子作為 whole-image caption。

## PoC 五個階段

### Phase 0.5：只用正位片（PA/AP），排除 Lateral（已完成）

PadChest 的 `Projection` 欄位分佈：PA 57.0%、L（Lateral）30.8%、AP_horizontal 8.9%、AP 2.8%、COSTAL 0.4%。決定**排除 Lateral（約 31%）**，只用 PA/AP：

- CheXmask 等解剖分割工具幾乎只做正位（PA/AP）的肺野/心臟遮罩，側位片沒有對應的解剖 mask 可用，RADAR 的 anatomy-aware pooling 沒有 mask 就跑不動。
- Labels/Localizations 的左右肺、bilateral 等標註邏輯是以正位片為基準，混進側位片會讓「左肺/右肺」分區失去意義。
- Tradeoff：會漏掉一些只靠側位才看得出來的異常（如後肋膈角處的少量肋膜積水），PoC 階段可接受，之後有需要再另外處理側位。
- **狀態**：資料已更新為只含 PA/AP（`PNG_tar/PADCHEST_PA_AP.tar`，96,270 張），這步驟已完成。

### Phase 1：解剖遮罩（取代 TotalSegmentator）

用 CheXmask 取得 mask，先只留 3-4 個粗分區（例如左肺、右肺、心臟/縱膈腔）。若 CheXmask 拿不到對應這個 shard 的 case，備案是用 [ianpan/chest-x-ray-basic](https://huggingface.co/ianpan/chest-x-ray-basic) 現場推論取得肺部/心臟等分割遮罩。

### Phase 2：文字監督（取代三段 LLM report parsing）

寫 rule-based 腳本，把 `Labels` + `Localizations` + `LabelsLocalizationsBySentence` 轉成「每個解剖分區的 caption + normal/abnormal flag」——帶 bilateral/basal 詞的 finding 分給對應肺野，沒有 location 的 finding 當作 whole-image finding 給 RADAR+ 全域對比用。

### Phase 3：模型改 2D

把 `VisionBranch` 的 Conv3d/BatchNorm3d/kernel_sizes/strides 換成 2D，`forward()` 裡的 `max_pool3d` 換成 `max_pool2d`；`organs` 清單從 36 個腹部器官縮到 3-4 個胸腔分區。因為沒有現成的 X-ray 版 `checkpoint_unet.pth`，segmentation 這條輔助 loss 用 CheXmask mask 從頭 end-to-end 一起訓練，不單獨做分割預訓練。

### Phase 4：Dataset pipeline

仿照 `caption_datasets.py`，寫一版讀 PNG + region mask + region caption + abnormal flag 的 dataset，`radar_config.yaml` 的 batch size 可以放大（2D 影像記憶體需求遠低於 3D volume）。

### Phase 5：跑通驗證

先從完整 PA/AP 資料（96,270 張）中抽樣一小部分跑通訓練迴圈，確認 loss（seg + anatomy ITC + whole ITC）都會下降、流程無誤，再擴大到全部 PA/AP 資料訓練。用 `Labels` 欄位做 zero-shot 分類評估（比照 `calc_metrics.py` 算 AUC）驗證效果。

## 下一步

先動手做 **Phase 1**：確認 CheXmask 對 PadChest 的實際覆蓋範圍與下載方式，確認可行後再往 Phase 2、3 走。
