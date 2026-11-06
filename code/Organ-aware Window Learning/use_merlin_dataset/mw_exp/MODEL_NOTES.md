# MODEL_NOTES：TotalSegmentator / nnU-Net encoder

環境（A6000）：`~/miniconda3/envs/radar`，TotalSegmentator 2.18.0、nnunetv2 2.8.1、torch 2.9.1+cu126。
權重位置：`~/.totalsegmentator/nnunet/results/`

## 2.1 `total` 任務的子模型

| 資料夾 | 用途 | 類別 | spacing | patch | stages |
|---|---|---|---|---|---|
| Dataset291_part1_organs | total（全解析度） | 24：腹部器官、肺葉、食道、氣管、甲狀腺、腸道、膀胱、攝護腺、腎囊腫 | 1.5 mm | 128³ | 6 |
| Dataset292_part2_vertebrae | total | 26：sacrum、S1、L5–C1 | 1.5 | 128³ | 6 |
| Dataset293_part3_cardiac | total | 18：心臟、主動脈、IVC、門靜脈、髂動靜脈… | 1.5 | 128³ | 6 |
| Dataset294_part4_muscles | total | 23：肱骨、股骨、髖骨、臀肌、髂腰肌、腦… | 1.5 | 128³ | 6 |
| Dataset295_part5_ribs | total | 26：肋骨、胸骨、肋軟骨 | 1.5 | 128³ | 6 |
| **Dataset297_total_3mm** | **total `--fast`** | **117（全部類別，單一模型）** | **3.0** | **112×112×128** | **5** |
| Dataset298_total_6mm | total `--fastest` | 117 | 6.0 | 64³ | 5 |

**決定（2026-11-04，使用者指定）：使用 `--fast`（Dataset297）。** 這是單一模型，同時涵蓋腹部器官、主動脈和腰椎，所以第 2.1 節原本「骨骼在另一個子模型時要停下來問」的疑慮已經不存在。全解析度的 part1 encoder 只當備案（腹部器官全在 part1；腰椎在 part2，aorta 在 part3）。

TS 設定（`get_task_config("total", fast=True)`）：`task_id=297, resample=3.0, trainer=nnUNetTrainer_4000epochs_NoMirroring, folds=[0], crop=None`。

## 2.2 plans.json / CT normalization（Dataset297）

- `3d_fullres`：patch_size `[112, 112, 128]`，spacing `[3, 3, 3]`，PlainConvUNet，features `32-64-128-256-320`，pool `[1,2,2,2,2]`（各軸），每 stage 2 conv。
- `foreground_intensity_properties_per_channel["0"]`：
  - `percentile_00_5 = -1004`，`percentile_99_5 = 1588`
  - `mean = -50.387`，`std = 503.392`
- normalization：`CTNormalization`。clip 到 [p0.5, p99.5] 後減 mean、除以 std。
- 對照：part1 的範圍是 [−1024, 276]，mean −370，std 437，會把骨骼資訊截掉。3mm 模型的 clip 範圍涵蓋骨骼，所以 `bone` window 拉伸後仍落在模型熟悉的區間內。

## 2.3 Encoder 與前處理

- 載入方式：`nnUNetPredictor.initialize_from_trained_model_folder(folder, use_folds=(0,))`，再 `network.load_state_dict(list_of_parameters[0])`，取 `network.encoder`（`PlainConvEncoder`，`return_skips=True`，直接回傳各 stage 的 list）。程式在 [src/encoder.py](src/encoder.py)。
- 用隨機輸入 `[1,1,112,112,128]`（fp16 autocast）測出的各 stage shape：

| stage | shape | 相對 patch 的倍率 | 實際解析度 |
|---|---|---|---|
| 0 | 32 × 112×112×128 | 1 | 3 mm |
| **1** | **64 × 56×56×64** | 1/2 | 6 mm |
| **2** | **128 × 28×28×32** | 1/4 | 12 mm |
| **3** | **256 × 14×14×16** | 1/8 | 24 mm |
| 4 (bottleneck) | 320 × 7×7×8 | 1/16 | 48 mm |

- **選用 stage 1、2、3** → D = 64 + 128 + 256 = **448**。排除 stage 0（解析度最高）和 stage 4（bottleneck）。
- 可訓練參數 = 0；單次 forward 約 0.23 秒，峰值 2.85 GiB。

**TotalSegmentator 的前處理**（`totalsegmentator/nnunet.py: nnUNet_predict_image`）：
1. `as_closest_canonical(img)` → RAS。
2. `change_spacing(img, 3.0, order=1, dtype=int32)`（`resampling_order=1`，HU 四捨五入成整數）。
3. 存成 `s01_0000.nii.gz` 交給 nnU-Net；dataset.json 指定 `NibabelIOWithReorient`，讀進來後轉成 RAS 並 transpose 成 (z, y, x)。
4. nnU-Net `DefaultPreprocessor`：crop to nonzero → resample（spacing 已經是 3mm，等同 identity）→ CTNormalization。
5. `save_lowres=True` 時，輸出的 mask 留在 3mm 格點上（只做 undo_canonical）。

本實驗（[src/patches.py](src/patches.py)）完全照第 1–3 步處理；mask 先做 canonical，若格點不一致再用 nearest 對齊。sanity check 1 確認 baseline 和 nnU-Net 原生前處理的**最大誤差為 0.0**。

## 其他
- 小器官在 stage 3（24 mm）幾乎會消失：pilot 中 adrenal 和 gallbladder 會觸發 bbox fallback，次數記錄在 `qc/sanity.json` 的 `fallback_counts`。
- 器官大於 patch 時才切多塊：3mm 下只有 colon 偶爾會切成 2 塊，其餘器官都是 1 塊。

## Experiment B: bone units for fracture (2026-11-05)
- 17 bone units (organ_map.yaml tail; T8..L5 each, ribs_left/right, sacrum, hip_l/r, femur_l/r) are extracted by
  src/extract_bones.py into outputs/emb_bones/{baseline,soft_tissue_L40_W400,bone_L400_W1800}/ (keys emb = mean, emb_max).
- probe.py appends units missing from emb/ from `bone_emb_dir`; windows absent for bones (liver, narrow_abd) are filled
  from `bone_window_fallback` (default soft_tissue). --standardize keeps pooled stats for the original 12 units and
  uses per-unit stats for bone units.
- finding_organ_map `agg: max` + `sites`: each site has a shared linear head over its units (+ per-unit bias),
  site logit = max over present units, finding logit = logsumexp over sites (+ bias). Trained on the any-fracture
  label only (site sub-label auxiliary loss gave the same val AUROC in bone_probe.py: 0.827 vs 0.830).
- Mean pooling > max pooling for every aggregation; window concat (baseline+soft+bone) +0.006 over baseline.
