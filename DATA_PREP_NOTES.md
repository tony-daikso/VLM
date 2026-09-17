# 新資料的預處理/存檔建議

## 結論

未來有新的胸腔 X-ray 資料要加進來時,可以直接**先正規化、再縮成 512×512 灰階存起來**,當成
「母解析度」快取,這樣可以大幅省空間,而且跟這個專案裡現有的用法完全相容(U-VLM Stage1、DINO
推論都是吃 512×512)。

## 為什麼是 512×512

- `U-VLM/X-ray/cxr/stage1/config.yaml` 跟 `U-VLM/CT/stage1/config.yaml` 的 `resolution` 都是
  512,U-VLM 從訓練到推論全程都是 512×512,沒有解析度落差問題。
- 這次 DINO+LLaVA baseline 的 `stage_mosaic/build_mosaics.py` 跟
  `stage_dino_ssl/dataset.py`(`IMG_SIZE`)在**推論/產生假圖時**用的也是 512×512。
- DINO **訓練時**用的是官方標準的 global crop 224 / local crop 96(比 512 小很多,原因見
  `stage_dino_ssl/main_dino.py` 的註解:直接用 512 當 crop 尺寸會把 46GB GPU 打 OOM)。512 當
  母解析度存起來,對 224/96 的 `RandomResizedCrop` 增強來說解析度還綽綽有餘,不會因為先 resize
  過而讓增強效果打折。

## 正確的處理順序(重要,不能跳過)

**一定要先做 percentile normalize(16-bit → 8-bit),再 resize**,不能對原始 16-bit
DICOM/PNG 直接做 naive resize/convert。

這個順序錯了會重踩我們已經踩過的坑:PadChest-GR 原圖是 16-bit PNG(PIL mode `"I;16"`),直接
`Image.open(path).convert("RGB")` 會把像素值 naive 轉換,幾乎全部夾到接近全白(實測平均值
241.95/255,肉眼看幾乎是一張全白的圖,DINO 等於在學一張壞掉的圖)。這個 bug 最早在
`stage_mosaic/build_mosaics.py` 跟 `stage_dino_ssl/dino_eval.ipynb` 裡發生過,後來才修好。

正確流程:

```
讀原始像素(np.array(Image.open(path)))
  → percentile_normalize_uint8()（0.5~99.5 分位數 clip + min-max stretch，
     見 stage_dino_ssl/dataset.py）
  → resize 到 512×512
  → 存 PNG（8-bit 灰階）
```

可以直接重用 `stage_dino_ssl/dataset.py` 裡的 `percentile_normalize_uint8()` 函式,不用重寫。

## 空間節省

- 原圖:約 1824×1652、16-bit,單張約 6MB。
- 512×512、8-bit 灰階 PNG:單張大概幾百 KB。
- 壓縮比 90%+ 以上。

## 快取檔名慣例

這個專案裡已經有兩種類似的快取,新資料建議比照同一套命名習慣,方便以後對照:

- U-VLM:`_cache_r{resolution}_p{percentile_low}_{percentile_high}`
  (例:`_cache_r512_p0.5_99.5`,見 `U-VLM/X-ray/cxr/stage3/dataset.py`)
- DINO SSL 訓練:`_cache_dino_ssl_r{CACHE_RESOLUTION}`
  (`_cache_dino_ssl_r320`,見 `stage_dino_ssl/dataset.py`)

新資料的母解析度快取可以取類似的名字,例如 `_cache_r512_p0.5_99.5`,跟 U-VLM 的慣例對齊。
