# Phase 1: CheXmask 解剖遮罩取得 — 結果記錄

見 `RADAR/X-ray/PLAN.md` 了解整體 5-phase 規劃。這份文件記錄 Phase 1 實際跑出來的結果。

## 最終結果

- **覆蓋率**：我們的 96,270 張 PA/AP PNG 裡，96,184 張在 CheXmask 找到對應 mask（**99.91%**），86 張沒對到（見 `missing_from_chexmask.txt`）。缺口遠低於原訂 1% 的「要不要做方案 B」門檻，**確定不需要 `ianpan/chest-x-ray-basic` 現場推論備案**。
- **Mask 全量產生**：96,184 張全部成功產生 512×512 label map（0=背景，1=left_lung，2=right_lung，3=heart），**0 筆失敗**。輸出在 `data/X-ray/PadChest-Origin/processed_chexmask/masks/`，manifest 在同目錄下的 `manifest.jsonl`。
- **品質分布**：Dice RCA (Mean) 中位數 0.858，p5=0.790，只有 1,257 張（1.31%）低於論文建議的 0.7 門檻，manifest 裡這些都已標記 `quality_ok: false`，Phase 4 的 dataset 應該濾掉這些不用。
- **QA 疊圖目視檢查（25 張抽樣）**：肺野/心臟遮罩位置目視正確，左右側跟 X 光片上的 "R"/"L" 標記或解剖形狀吻合。

## 意外發現：PA/AP tar 裡混了至少一張側位片

抽樣疊圖時發現 `216840111366964013418328332882012198173339371_01-060-132.png` 其實是**側位片**，不是正位——HybridGNet 是為正位設計的模型，套在側位片上跑出來的遮罩完全不對(見疊圖)。好消息是這張的 Dice RCA (Mean) 只有 0.617，正確低於 0.7 門檻，`quality_ok` 標記為 `false`，代表 Phase 4 只要照 manifest 的 `quality_ok` 欄位過濾，這類混進來的標籤錯誤案例會被自動排除，不需要額外處理。但這提醒我們：`PADCHEST_PA_AP.tar` 的 Projection 欄位篩選不是 100% 乾淨,Dice RCA 門檻剛好順便當了一層防呆。

## Mask 合併規則

CheXmask 提供的 Left Lung / Right Lung / Heart 三個 binary mask，合併成單通道 label map 時，畫的順序是 left lung(=1) → right lung(=2) → heart(=3)，心臟蓋在最上層(心臟輪廓在中線/肺野內側偶爾跟肺野有些微重疊，manifest 裡的 `overlap_pixel_frac` 記錄了每張圖的重疊像素比例,目前看到的樣本大概在 3-5% 左右，屬於正常範圍)。

## 跟 Phase 4 的格式約定(重要)

- Mask 是 512×512、單通道、uint8、值域 0-3 的 PNG,用 `Image.resize((512,512), Image.NEAREST)` 從 CheXmask 提供的原始解析度縮放而來(**單純縮放,沒有 crop/pad**)。
- 已確認我們 tar 裡的原圖本身也是**單純不等比例拉伸**縮到 512×512(用邊界像素統計驗證過,沒有補黑邊的痕跡),所以 Phase 4 讀圖時**必須也用同樣「單純 resize 到 512×512、不要 crop/pad」的方式**,mask 才會跟圖片逐像素對齊。
- Phase 4 的 dataset 讀取時,建議照 manifest 的 `quality_ok` 欄位過濾掉品質不好的 1.31%。

## 檔案清單

```
RADAR/X-ray/phase1/
├── README.md（本檔案）
├── rle_utils.py             # 從 ngaggion/CheXmask-Database 搬來的 RLE 解碼
├── list_padchest_ids.py     # 列出 tar 裡的 96,270 張檔名
├── check_coverage.py        # 覆蓋率/品質檢查(支援 CHEXMASK_CSV_PATH 環境變數覆寫,方便對部分下載的檔案做 dry-run)
├── extract_masks.py         # 主要 mask 產生腳本(resumable)
└── qa_overlay.py            # 抽樣疊圖

data/X-ray/PadChest-Origin/
├── PNG_tar/PADCHEST_PA_AP.tar
├── chexmask_raw/Padchest.csv                    # CheXmask OriginalResolution 版,4.6GB
└── processed_chexmask/
    ├── padchest_pa_ap_ids.txt                   # 我們的 96,270 張檔名清單
    ├── missing_from_chexmask.txt                # 86 張沒對到 CheXmask 的檔名
    ├── manifest.jsonl                            # 96,184 筆
    ├── manifest_failed.jsonl                     # 空(0 失敗)
    ├── masks/<image_id>.png                      # 96,184 個 mask 檔
    └── qa/*.png                                  # 25 張抽樣疊圖
```

Phase 1 到此結束,可以進 Phase 2(已完成,見 `RADAR/X-ray/phase2/`)/ Phase 4(dataset pipeline)。

## 資料存放位置(重要,訓練前一定要注意)

`/datadrive`(fuse.fx 掛載)寫入速度實測只有 **~14MB/s**,讀很多小檔案時更慢(掃 96,184 個 mask 檔打包成 tar 時掉到 ~1MB/s)。`/root/Desktop/VLM`(本地磁碟)有 **~829MB/s**。原始資料/一次性用完的大檔(CheXmask CSV、labels CSV)留在 `/datadrive` 當存檔即可,但**任何訓練會重複讀取的東西都複製一份到本地磁碟**:

```
/root/Desktop/VLM/data/X-ray/PadChest-Origin/
├── PNG_tar/PADCHEST_PA_AP.tar              # 從 /datadrive 複製過來的原圖 tar
└── processed_chexmask/
    ├── manifest.jsonl
    ├── padchest_pa_ap_ids.txt
    └── masks.tar                            # 96,184 個 mask 打包成單一 tar(不是 96,184 個散檔!)
                                              # 讀取方式比照圖片 tar:lazy-open + tarfile 隨機存取,
                                              # 見 phase3/pretrain_segmentation.py 的 ChexmaskSegDataset
```

`data/X-ray/PadChest-Origin/processed_captions/manifest.jsonl`(Phase 2 輸出)也比照複製了一份到本地。

`/datadrive` 上原本產生的 96,184 個散落 mask 檔(`processed_chexmask/masks/`)還留著當備份,但**新的程式都改讀本地的 `masks.tar`**,不要再直接開散檔案。
