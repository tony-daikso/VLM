# DINO + LLaVA(Llama-3.2-3B) 胸腔 X-ray Baseline — 對比 U-VLM CXR Stage3

## 這是什麼

U-VLM 的 X-ray/cxr 賽道（`U-VLM/X-ray/cxr/`，ResNet34 共享 encoder → GPT-2 cross-attention 報告
生成）已經訓練完成。這個資料夾要做一個對照組：把學長之前做過的「DINO ViT + LLaVA + Llama-3.2-3B」
CT 報告生成 pipeline（`DINO_LLM/prior/`，未附權重、無訓練資料，純推論用的 demo pipeline）**忠實地**
搬到胸腔 X-ray（PadChest-GR）上重新做一次，才能跟 U-VLM CXR Stage3 在同一份資料、同一個 split 上
公平比較。

**這次只做 X-ray，CT 不管。**

## 已拍板的關鍵設計決策

1. **DINO→LLM 融合方式**：完全照抄學長的「馬賽克假圖」trick —— DINO CLS token embedding
   reshape 成小格子拼成一張假圖，餵進**完全沒改過的 LLaVA CLIP tower**（不做 patch-token
   projector 之類「更標準」的整合）。
2. **DINO 視覺骨幹權重**：學長的 `dino.pth` 沒附在 repo 裡，架構也是非標準的 `vit_large`
   （`embed_dim=1280, depth=12, num_heads=20`）。選擇**在胸腔 X-ray 資料上重新跑一次
   self-supervised DINO 訓練**，而不是換用官方 DINOv2 權重。
3. Llama-3.2-3B 是 HF gated model，需要使用者提供已同意授權條款帳號的 HF token。

## 資料與程式碼重用

- **PadChest-GR 資料**：直接重用 U-VLM 已處理好的
  `data/X-ray/PadChest-GR/processed/manifest.jsonl`（4,555 studies，official split
  train 3,185 / validation 455 / test 915，跟 `U-VLM/X-ray/cxr/stage3/dataset.py` 完全一樣）。
  不重新處理影像，只重用同一份 manifest 與 split，確保跟 U-VLM 比較時資料完全一致。
- **DINO 模型/載入程式碼**：重用 `DINO_LLM/prior/dino/{vision_transformer.py, utils.py,
  dino_model_c.py}`（原封不動複製到 `dino/`），只補一支學長沒附的 self-supervised 訓練腳本
  （`prior/` 裡只有 model 定義跟 loader，沒有官方 `main_dino.py` 那種訓練迴圈）。
- **假圖 mosaic 邏輯**：重用 `DINO_LLM/prior/inference.py::dino_inference()` 的算法（CLS token
  reshape 成 5×16×16 block，拼進 `final_size=480` 的畫布，z-score normalize 成 uint8）。**必要調整**：
  原版是給「一個 study 資料夾裡一堆 CT 切面」設計的，X-ray 一個 study 只有一張 2D 影像，所以每張
  假圖只會填進畫布左上角 5/900 格，其餘留白。
- **LLaVA 程式碼**：重用 `DINO_LLM/prior/llava/` 整包，完全不修改架構。
- **評估指標**：重用 `U-VLM/X-ray/cxr/stage3/eval.py` 的 BLEU/ROUGE（sacrebleu + rouge_score）與
  26 類 keyword 臨床有效性 proxy 邏輯，套用在同一組 validation/test split 上，才能跟 U-VLM CXR
  Stage3 的數字直接對照。

## 目錄結構

```
DINO_LLM/X-ray/
├── dino/                # 從 prior/dino/ 複製，架構不變
├── llava/                # 從 prior/llava/ 複製，架構不變
├── stage_dino_ssl/        # Phase 2：DINO self-supervised pretraining
│   ├── dataset.py          # train split multi-crop 資料集（含磁碟 cache）
│   └── main_dino.py        # 官方 DINO 訓練迴圈（student/teacher EMA + DINO loss）
├── stage_mosaic/           # Phase 3：假圖產生
│   └── build_mosaics.py
├── stage_llava_ft/         # Phase 4：LLaVA LoRA 微調
│   ├── build_llava_data.py  # manifest -> LLaVA conversation JSON
│   └── train_lora.sh        # 呼叫 llava/train/train.py 的 LoRA 訓練指令
└── eval/
    └── eval.py              # BLEU/ROUGE/keyword-proxy，跟 U-VLM stage3 同一套指標
```

Checkpoint/log/mosaic 輸出全部指到 `/datadrive/VLM/DINO_LLM/X-ray/`（持久化掛載點）。

---

## Phase 0 — 環境設置 ✅ 完成

新建 conda env `dino_llm`（`/root/miniconda3/envs/workenv/envs/dino_llm`），裝跟 driver
（CUDA 12.9）相容的 torch cu126 build（原本 `VLM` env 裡的 torch 2.14+cu130 跟 driver 對不上，
`torch.cuda.is_available()` 是 `False`），加上 transformers/peft/accelerate/bitsandbytes/
sacrebleu/rouge_score 等套件。已驗證 GPU 抓得到（NVIDIA L40S, 46GB）。

## Phase 1 — 資料準備 ✅ 完成

不需要重新處理影像，manifest.jsonl 路徑/split 筆數已核對過（train 3,185 / validation 455 /
test 915）。

## Phase 2 — DINO Self-Supervised Pretraining ✅ 完成

用 `vit_large`（架構完全不動）在 train split 的 3,185 張 X-ray 上做 self-supervised
pretraining（multi-crop + student/teacher EMA + DINO loss，單 GPU，AMP 混合精度）。

- global/local crop 解析度用官方 DINO 標準的 224/96（不是 U-VLM 那邊的 512×512 —— crop 尺寸是
  SSL 訓練超參數，跟下游影像原始解析度無關；實測 512 直接把 46GB GPU 打到 OOM）。
- 中途發現每個 sample 的資料前處理（decode 大圖 + percentile normalize）要 ~0.48 秒，導致
  GPU 大部分時間閒置（dataloader-bound）。加了磁碟快取（`_cache_dino_ssl_r320/`）後降到
  ~0.18 秒，熱快取後單一 epoch 只要 ~1 分鐘。
- **實際結果**：100 epochs 花了 98.6 分鐘，loss 從 10.85 收斂到 ~7.23。Self-attention 有
  聚焦在解剖結構上、embedding 沒有崩塌（cosine similarity mean=0.37, std=0.42，不是接近
  1.0/0），但 `dino_eval.ipynb` 的 linear probe macro AUROC 明顯低於 U-VLM Stage2 監督式訓練
  的 encoder（U-VLM 0.8007，數字詳見 notebook 執行結果）。
- **已知風險（驗證屬實）**：DINO self-supervised pretraining 官方是在百萬級圖片上做的，這裡
  只有 3,185 張，訓練出來的表徵品質確實明顯弱化。這是「忠實重現土炮流程」的必然代價。

## Phase 3 — 假圖（Mosaic）產生 ✅ 完成

對 train/validation/test 全部 4,555 筆 study 跑訓練好的 DINO，取 CLS token 產生假圖 PNG，存到
`/datadrive/VLM/DINO_LLM/X-ray/mosaics/{split}/{study_id}.png`（train 3,185 / validation 455 /
test 915，全部對齊）。

- 中途發現一個真的 bug：PadChest-GR 原圖是 16-bit PNG，`build_mosaics.py` 一開始漏掉
  percentile normalize，直接 `Image.open().convert("RGB")` 把像素值 naive 轉換、幾乎全部夾到
  接近全白（平均值 241.95/255），已修好並重跑。
- 假圖驗證正常：非零區域剛好落在左上角 80×16 像素（5 個 16×16 block），其餘 99.65% 是黑的
  ——這是必要調整（X-ray 單張 2D 影像只夠貢獻 5 個 block，不像 CT 多切面能塞滿整張畫布），
  但也代表餵給 LLaVA CLIP tower 的假圖絕大部分是空的，資訊密度極低。

## Phase 4 — LLaVA + Llama-3.2-3B LoRA 微調 ✅ 完成

- 改用 `unsloth/Llama-3.2-3B`（HF 上無 gating 限制的鏡像，權重內容跟官方一致，Llama 3.2
  Community License 本身允許重新散佈，詳見對話記錄），不用等 HF token。
- `build_llava_data.py` 把 train split 轉成標準 LLaVA conversation JSON（固定 prompt
  `"generate analysis report"`，跟學長 `inference.py` 一致）。
- `train_lora.sh` 做單階段 LoRA 訓練：mm_projector 從零學、LLM 用 LoRA 微調、CLIP vision
  tower 全程凍結。跳過 LLaVA 官方兩階段 pretrain（那需要百萬級圖文對，跟這裡的任務/資料規模
  不匹配，也沒有對應的 pretrain 資料）。
- **訓練中修掉的真實 bug（都是實測 smoke test 抓到的，不是憑空猜的）**：
  1. `use_fast=False` 在 unsloth 這份鏡像上會讓 `AutoTokenizer.from_pretrained` 直接回傳
     `True` 而不是 tokenizer 物件（沒有慢速 tokenizer 需要的 `tokenizer.model`）。
  2. `model.requires_grad_(False)` 在 LoRA 掛上去**之後**才呼叫，把剛加的 LoRA adapter
     一起關掉，等於 LoRA 完全沒在學（只有 mm_projector 在學）。
  3. `model_max_length=512` 太短：CLIP-ViT-L/14-336 一張圖展開成 576 個 patch token，加上
     文字直接超過 512，把 assistant 的回答截斷掉、labels 全部變 -100，loss 變 NaN（Trainer
     的 log 還會把 NaN 誤導性地顯示成 0.0）。改成 1024。
  4. 兩處 `maybe_zero_3()` 無條件 `import deepspeed`，單 GPU 沒裝 deepspeed 每次存檔就炸，
     改成延遲匯入。
  5. `_save_checkpoint()` 的 LoRA 分支每個 epoch 只存 `mm_projector.bin`，LoRA adapter
     權重只有整個訓練結束後存一次——導致 `--evaluation_strategy epoch` 選出來的「最佳
     epoch」根本沒有對應的完整模型可以重建。補上每個 epoch 也存 `adapter_model.safetensors`。
  6. `eval/eval.py` 一開始借用 `llava/model/builder.py` 的通用 loader，但它的 LoRA 分支假設
     vocab_size 訓練時被 resize 過一格，這裡的 tokenizer 本來就有 pad token、從沒 resize
     過，會對不上 shape。改寫成專用 loader，照訓練時的初始化流程重建。
  7. `gradient_checkpointing` 直接關掉（不是修，是繞開）：開著會讓 LoRA/projector 完全學不到
     東西（loss/grad_norm 卡在 0.0），46GB GPU 對 3B 模型 + LoRA 綽綽有餘，不值得深入除錯。
- **實際結果**：5 epochs，每個 epoch 都有 `--evaluation_strategy epoch` 存 val loss：
  epoch 1: 1.5965 / epoch 2: 1.5184 / **epoch 3: 1.4895（最低，選為最佳 epoch）** /
  epoch 4: 1.5099 / epoch 5: 1.5396（開始 overfit，跟 U-VLM CXR Stage3 自己的曲線走勢一樣）。
  訓練 5 epochs 總共 ~29 分鐘。

## Phase 5 — 評估與跟 U-VLM 對照 ✅ 完成

用 `eval/eval.py`（跟學長 `inference.py::llava_inference()` 同款生成邏輯：llama3 conv
template、temperature=0 greedy，`--llava_model` 指到 epoch 3 checkpoint）在 validation split
（n=455）上生成報告，算 BLEU/ROUGE/keyword-F1。

### 定量結果

| 指標 | 這次 DINO+LLaVA baseline | U-VLM CXR Stage3 |
|---|---|---|
| BLEU | **2.25** | 無精確數字紀錄（見下方質化比較） |
| ROUGE-1/2/L (F1) | 0.0720 / 0.0312 / 0.0699 | 無精確數字紀錄 |
| keyword-proxy macro F1 | **0.0064** | 中等偏低，但 cardiomegaly F1=0.50 等常見類別還有訊號 |

### 質化結果 —— 完全 mode collapse

抽查 `eval_generations_validation.csv` 發現：**455 筆驗證集，生成的文字開頭 60 個字元
100% 一字不差**，全部都是：

> "No significant findings. Dorsal scoliosis. Dorsal spondylosis. No other relevant
> findings. Dorsal kyphosis. Dorsal scoliosis. Dorsal spondylosis. ..."

不管餵哪張 X-ray 進去，模型完全不看影像內容，只是不斷重複同一段脊椎相關的樣板文字直到打到
`max_new_tokens=96` 上限。26 個 keyword-proxy label 裡唯一不是 0 的是 `scoliosis`
（precision=0.0901, recall=1.0000, f1=0.1653）——recall=1.0 純粹是因為模型每次都提到
「scoliosis」，precision 剛好等於該 label 在驗證集裡的盛行率（41/455≈9%），代表**零真實
分類訊號**，不是模型學會辨識脊椎側彎。

### 結論：這個 baseline 比 U-VLM CXR Stage3 明顯更差，而且是完全的 mode collapse

U-VLM CXR Stage3 至少還有部分 mode collapse（常輸出 "No significant findings."，但常見
病灶如 cardiomegaly 還有一定辨識力，F1=0.50）；這次忠實重現學長「DINO 馬賽克假圖 +
LLaVA」做法的結果是**完全**的 mode collapse——模型學到的是「不管看到什麼都講同一段話」，
對輸入影像的依賴趨近於零。

---

## 風險與限制（跑完全部 Phase 後，以下全部證實為真，不是預先猜測）

1. DINO SSL 只用 3,185 張圖，表徵品質確實偏弱（linear probe AUROC 明顯低於 U-VLM Stage2）。
2. 馬賽克假圖對單張 2D X-ray 來說 99.65% 畫布是黑的，資訊密度比 CT 多切面版本低很多。
3. LLaVA 的 CLIP tower 從沒看過這種「假圖」，這整個 trick 本身資訊損失就很大（只用 CLS
   token，丟掉所有 patch-level 空間資訊）——這是學長原始設計的已知缺陷，忠實重現不代表要
   掩蓋它。
4. **最終結果**：以上限制疊加起來，導致 Phase 4 微調出來的模型完全學不會根據影像內容生成
   對應的報告，出現 100% 一致的 mode collapse（見 Phase 5），量化指標（BLEU 2.25、
   keyword-proxy macro F1 0.0064）都明顯劣於 U-VLM CXR Stage3。這個結果本身就是這次「忠實
   重現學長做法」實驗最有價值的產出——證明了 U-VLM 論文裡的 multi-layer visual injection
   （深層特徵注入早期 LM 層、淺層特徵注入晚期 LM 層，每層都重新注入）明顯優於「馬賽克假圖
   trick + 完全沒改過的 CLIP tower」這種做法，不是靠更大的 LLM（Llama-3.2-3B 遠大於 U-VLM
   自己 from-scratch 的 GPT-2/5-layer decoder）就能彌補融合機制本身的資訊損失。

## 目前整體進度

**全部 5 個 Phase 都已完成。** Phase 0~3 完成無異狀；Phase 4/5 過程中抓到並修好 7 個真實
bug（詳見 Phase 4 段落），最終在 validation split（n=455）上得到完整的定量+質化比較結果，
結論是這個「馬賽克假圖 + 未改動 CLIP + LoRA 微調 Llama-3.2-3B」的做法明顯劣於 U-VLM 自己的
CXR Stage3（多層視覺注入 + from-scratch 小型 decoder），且出現比 U-VLM 更嚴重的完全 mode
collapse。
