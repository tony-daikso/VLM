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

## Phase 2 — DINO Self-Supervised Pretraining 🔄 進行中

用 `vit_large`（架構完全不動）在 train split 的 3,185 張 X-ray 上做 self-supervised
pretraining（multi-crop + student/teacher EMA + DINO loss，單 GPU，AMP 混合精度）。

- global/local crop 解析度用官方 DINO 標準的 224/96（不是 U-VLM 那邊的 512×512 —— crop 尺寸是
  SSL 訓練超參數，跟下游影像原始解析度無關；實測 512 直接把 46GB GPU 打到 OOM）。
- 中途發現每個 sample 的資料前處理（decode 大圖 + percentile normalize）要 ~0.48 秒，導致
  GPU 大部分時間閒置（dataloader-bound）。加了磁碟快取（`_cache_dino_ssl_r320/`）後降到
  ~0.18 秒，熱快取後單一 epoch 只要 ~1 分鐘，100 epochs 估計 **~1.5 小時**跑完。
- **已知風險**：DINO self-supervised pretraining 官方是在百萬級圖片上做的，這裡只有 3,185 張，
  訓練出來的表徵品質預期會明顯弱化。這是「忠實重現土炮流程」的必然代價，會在最終比較報告裡
  老實寫出來。

## Phase 3 — 假圖（Mosaic）產生 ⏳ 待 Phase 2 完成

對 train/validation/test 全部 study 跑訓練好的 DINO，取 CLS token 產生假圖 PNG，存到
`/datadrive/VLM/DINO_LLM/X-ray/mosaics/{split}/{study_id}.png`。

## Phase 4 — LLaVA + Llama-3.2-3B LoRA 微調 ⏳ 待 Phase 3 完成 + 需要 HF token

- 下載 `meta-llama/Llama-3.2-3B`（gated model，需要使用者提供的 HF token）。
- `build_llava_data.py` 把 train split 轉成標準 LLaVA conversation JSON（固定 prompt
  `"generate analysis report"`，跟學長 `inference.py` 一致）。
- `train_lora.sh` 做單階段 LoRA 訓練：mm_projector 從零學、LLM 用 LoRA 微調、CLIP vision
  tower 全程凍結。跳過 LLaVA 官方兩階段 pretrain（那需要百萬級圖文對，跟這裡的任務/資料規模
  不匹配，也沒有對應的 pretrain 資料）。

## Phase 5 — 評估與跟 U-VLM 對照 ⏳ 待 Phase 4 完成

用 `eval/eval.py`（跟學長 `inference.py::llava_inference()` 同款生成邏輯：llama3 conv
template、temperature=0 greedy）在 validation/test split 上生成報告，算 BLEU/ROUGE/keyword-F1，
直接對照 U-VLM CXR Stage3 已有的數字（val_loss 1.33、嚴重 mode collapse、keyword-proxy F1
多數類別接近 0、cardiomegaly F1 0.50 等），並附上質化抽查（生成文字 vs 參考報告對照）。

---

## 風險與會誠實記錄的限制

1. DINO SSL 只用 3,185 張圖，表徵品質預期偏弱。
2. 馬賽克假圖對單張 2D X-ray 來說大部分畫布是空的，資訊密度比 CT 多切面版本低很多。
3. LLaVA 的 CLIP tower 從沒看過這種「假圖」，這整個 trick 本身資訊損失就很大（只用 CLS
   token，丟掉所有 patch-level 空間資訊）——這是學長原始設計的已知缺陷，忠實重現不代表要
   掩蓋它。
4. Llama-3.2-3B 版本（base vs Instruct）、LoRA 訓練超參數（epoch 數、batch size、lr）等
   細節會在跑到 Phase 4 時先做小規模 smoke test 確認可行後再定案。

## 目前整體進度

Phase 0/1 完成，Phase 2 進行中（背景訓練，預估 ~1.5 小時內跑完），Phase 3-5 待續。
Phase 4 需要使用者提供 HF token 才能下載 Llama-3.2-3B（尚未收到）。
