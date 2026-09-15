# U-VLM Stage 1 資料處理 — 交接文件

寫於 2026-09-14，目的是換一台電腦繼續跑之前，把目前進度、還缺什麼、怎麼繼續接起來講清楚。
架構設計（模型怎麼訓練）的完整規劃在 [U-VLM/stage1/STAGE1_PLAN.md](U-VLM/stage1/STAGE1_PLAN.md)，這份文件只講「資料處理到哪了、怎麼在新機器上接續」。

---

## 1. 目前整體狀態一句話總結

**三個分割標註資料來源的抽取都已完成或接近完成**，training code（dataset.py / model.py / train.py 等）**還沒開始寫**，卡在 STAGE1_PLAN.md 最後 5 個待確認事項還沒拍板。

| 來源 | 腳本 | 狀態 |
|---|---|---|
| CT-RATE + ReXGroundingCT（病灶 mask + 分類 + 報告） | `scripts/extract_dataset_rexgroundingct.py` | ✅ **完成**，3,042 筆 |
| TotalSegmentator（器官 mask） | `scripts/extract_totalseg_organs.py` | ⏸️ **暫停中，2,807/3,042 筆完成（92%）**，可續跑 |
| HiPaS（血管 artery/vein mask） | `scripts/extract_dataset_hipas.py` | ✅ **完成**，250/250 筆 |

---

## 2. 各資料夾內容

```
data/
├── processed_rex/                  # CT-RATE + ReXGroundingCT + TotalSegmentator
│   ├── manifest.jsonl              # 3,042 筆，每筆有 image_npy/image_png/segmentation_mask_npz(病灶)/
│   │                                #   classification_labels/report_text
│   ├── manifest_failed.jsonl       # 100 筆永久失敗（ReXGroundingCT 上游真的沒有這些檔案，404，
│   │                                #   不是我們這邊的問題，不用再重試）+ 少量已修復的舊紀錄（重複行，
│   │                                #   實際仍失敗的只有 100 筆不同 volume_id）
│   ├── images_npy/, images/, masks/  # 病灶相關檔案
│   ├── organ_masks/                # TotalSegmentator 器官 mask，目前 2,807/3,042 個 .npy
│   │                                #   （檔名 = volume_id，缺的就是還沒跑完的那 235 筆）
│   ├── totalseg_organs.log         # 器官分割批次的執行 log（可看已處理到哪、每筆耗時）
│   └── totalseg_organs_failed.jsonl  # 器官分割批次失敗紀錄（目前 0 筆失敗）
│
├── processed_hipas/                 # HiPaS 血管，完全獨立的病人群，跟上面對不上
│   ├── manifest.jsonl              # 250 筆，每筆有 image_npy/vessel_mask_npz(artery+vein)
│   └── images_npy/, images/, masks/
│
└── hipas_raw/                        # HiPaS 原始下載檔（annotation 已解壓，
                                        #   但 24GB 的 ct_scan.zip 目前已經不在資料夾裡了 —
                                        #   不確定是誰清掉的，但因為 processed_hipas/ 已經是完整處理好
                                        #   的結果，不影響任何事，除非之後想改切片邏輯重新抽取
                                        #   才需要重新下載那顆 24GB zip
```

`U-VLM/stage1/STAGE1_PLAN.md` — 完整架構規劃文件（共用 encoder + 3 個 decoder head：lesion/organ/vessel），最後一節列了 5 個還沒拍板的待確認事項。

---

## 3. 在新電腦上怎麼接續

### 3.1 環境設置

```bash
# Python 3.14（這台機器用的版本，torch 2.14 有對應 wheel）
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install TotalSegmentator huggingface_hub nibabel pandas pillow
```

TotalSegmentator 第一次執行時會自動下載模型權重（`total` task），不需要另外設定。

**Apple Silicon 才有 MPS 加速**（`-d mps`）；如果新電腦是 Intel Mac 或 Linux/Windows，`extract_totalseg_organs.py` 裡的 `"-d", "mps"` 要改成 `"-d", "cpu"` 或 `"-d", "gpu"`（有 NVIDIA GPU 的話），不然會直接報錯。

### 3.2 需要 Hugging Face 存取權限

- `ibrahimhamamci/CT-RATE`：gated dataset，需要先在 HF 網站上該 dataset 頁面按同意授權條款，然後 `huggingface-cli login` 或設定 `HF_TOKEN` 環境變數
- `rajpurkarlab/ReXGroundingCT`：同上可能也需要登入

### 3.3 需要搬過去的東西

把整個 `data/processed_rex/`、`data/processed_hipas/`、`scripts/`、`U-VLM/` 資料夾複製過去就好（`data/processed_rex/` 大小約 4.5GB，`data/processed_hipas/` 約 289MB，都不大）。**`data/hipas_raw/` 不需要搬**（HiPaS 已經處理完成，原始檔案用不到了）。

### 3.4 繼續跑器官分割批次（剩 235 筆）

```bash
cd VLM
source .venv/bin/activate
python3 scripts/extract_totalseg_organs.py
```

這支腳本是 **resumable** 的：開頭會讀 `data/processed_rex/organ_masks/` 裡已經有哪些 `{volume_id}.npy`，自動跳過、只處理缺的。不用加任何參數。

以這台機器的速度（M3 + MPS + `--fast`），大約 **25~35 秒/筆**，剩 235 筆估計還要 **~2 小時**。新電腦效能不同，時間會不一樣。

**背景執行建議**：
```bash
python3 scripts/extract_totalseg_organs.py &
# 如果是筆電，建議同時防止睡眠中斷任務：
caffeinate -i -w $! &
```
（睡眠中斷任務不會遺失進度，只是會浪費時間 — 每筆處理完就立刻存檔，中斷後重跑會自動接續，但如果不裝 caffeinate 且電腦常自動睡眠，實測會讓單筆耗時暴增到 5~10 分鐘，因為暫停的時間也算進去了。）

### 3.5 器官分割全部跑完之後

1. 檢查 `data/processed_rex/organ_masks/` 是否有 3,042 個檔案（等於 manifest.jsonl 筆數）
2. `data/processed_rex/totalseg_organs_failed.jsonl` 應該還是 0 筆或很少，如果有失敗可以重跑腳本重試
3. 回頭確認 `U-VLM/stage1/STAGE1_PLAN.md` 第 8 節的 5 個待確認事項：
   - organ head 用全部 118 類還是縮減子集
   - vessel head（HiPaS）要不要現在就跟 lesion/organ 一起訓練
   - encoder 要不要用 ImageNet 預訓練初始化
   - 解析度 512×512 還是 256×256
   - HiPaS 跟 CT-RATE 的 pixel spacing 差異要不要處理
4. 確認完這 5 點後，才開始寫 `U-VLM/stage1/dataset.py` / `model.py` / `losses.py` / `train.py` / `eval.py`（目前都還沒寫）

---

## 4. 已知問題 / 踩過的坑（新機器上不用重踩）

1. **`hf_hub_download` 的快取是 blob+symlink**：直接 `os.remove()` 下載回來的路徑只會刪 symlink，底層 blob 檔案還在磁碟上，不會真的釋放空間。`extract_dataset_rexgroundingct.py` 和 `extract_totalseg_organs.py` 裡都已經修好（`cleanup_download()` 函式會同時刪 symlink 跟 `os.path.realpath()` 指向的真實 blob），新機器上這兩支腳本可以直接用，不用再修。
2. **TotalSegmentator 需要完整 3D volume 才準**，不能只餵單張 2D slice（會破壞它預期的 z 方向上下文），所以 `extract_totalseg_organs.py` 是重新下載完整 volume、跑完 3D 推論後才切出對應那一張 slice，不是直接處理我們已經存好的 2D `image_npy`。
3. **ReXGroundingCT 有 3,142 個 volume 有標註**，分散在 CT-RATE 的 train（2,578+50+100=2,728）跟 validation（414）兩個 split，一開始只抓了 validation 那 414 筆，後來才發現要把 train split 也一起抓才有足夠資料量。
4. **ReXGroundingCT 有 100 個 volume 的 segmentation 檔案在 HF repo 上是真的不存在**（404，主要集中在 `train_129xx`~`train_136xx` 這段病人 ID），這是上游資料缺口，重試也沒用，已經記錄在 `manifest_failed.jsonl`。
5. **筆電背景長時間任務要注意睡眠跟電量**：這次跑批次時因為電腦自動睡眠讓單筆耗時暴增，之後改用 `caffeinate` 解決；另外也建議電量低於 20% 就先暫停、插電後再繼續，避免電池耗盡中斷任務或跑到系統強制關機。
