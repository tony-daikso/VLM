# U-VLM CXR Stage 3 — Report Generation（PadChest-GR）

參考論文：*U-VLM: Hierarchical Vision Language Modeling for Report Generation* (arXiv 2603.00479)。
跟 [U-VLM/stage2](../stage2/STAGE2_PLAN.md)（CXR 版 Stage 2）與 CT 版的 `U-VLM/CT/stage3`（目前是空資料夾，
CT 版還沒設計到這一步）都是獨立的東西——這份文件是**目前整個專案（CT + CXR）第一份真正把 Stage 3 架構定下來
的規劃文件**，先在 CXR track 上驗證可行，之後如果要幫 CT 版補 Stage 3 可以參考這裡的設計。

**v1 範疇（已跟使用者確認，2026-09-16）**：只做整段報告生成，**不輸出 per-sentence bounding box**。
PadChest-GR 的 grounding 標註（box/labels/locations）在 manifest 裡完整保留，v1 只是先不用在輸出端，
之後如果要升級成「每句話配一個 box」的 grounded 輸出（v2），不需要重新抽資料。

---

## 1. 資料盤點

`data/X-ray/PadChest-GR/processed/manifest.jsonl` 的 `report_text` 欄位：一個 study 所有 finding 的
`sentence_en` 依原始順序串接成一段完整英文報告（西文 `sentence_es` 先不用）。

- Normal study（1,456 筆）的 `report_text` 通常是類似「No other significant alterations.」這種單句
  或極短文字（見 grounded_reports_20240819.json 的 finding 範例，`abnormal:false` 的句子沒有 box，但
  一樣有 sentence_en）
- Abnormal study（3,099 筆）的 `report_text` 是多句 finding 串起來的完整段落
- 官方 split（train 3,185 / validation 455 / test 915）直接沿用，不重新切
- `findings` 裡完整的 `boxes`/`labels`/`locations`/`progression` 資訊留著不用，供 v2 用

---

## 2. 任務定義（v1）

輸入影像的前處理跟 Stage 1/2 完全一致（16-bit 原圖 per-image 百分位 clip+stretch，見
[STAGE1_PLAN.md 第 3 節](../stage1/STAGE1_PLAN.md)）：

```
image (1,512,512)
   │
   ▼
Encoder（建議用 CXR Stage 2 微調後的權重，因為分類任務已經驗證過這組特徵在語意層面有用）
   │  multi-scale features {f_i}
   ▼
Visual injection（把 {f_i} 投影成一組 visual token / prefix，注入 LM decoder 的多層——呼應論文
"multi-layer visual injection" 的設計，不是只塞一層）
   │
   ▼
Decoder-only LM（先選現成的小型模型，量級跟 GPT-2 相近，不從零訓練 LM）
   │
   ▼
生成完整 report_text（teacher forcing 訓練，用真實 report_text 當 target）
```

- **不輸出 box**：LM 只生成文字，訓練時的 target 就是 `report_text` 這個字串，沒有額外的座標輸出頭
- Encoder 來源：CXR Stage 2 的 `best_encoder.pt`（比 Stage 1 的版本多了分類任務的微調，特徵更貼近
  「有沒有某種 finding」這種語意層級的判斷，比較適合報告生成這種需要語意理解的任務）
- 只用英文 `sentence_en`／`report_text`，西文 `sentence_es` 先不用（雙語是可以之後加的擴充，不影響
  v1 架構）

---

## 3. 評估指標

- **標準生成指標**：BLEU、ROUGE（跟 report_text 逐字比較）
- **臨床有效性代理指標**：把生成的報告文字餵回 CXR Stage 2 訓練好的分類器（或用簡單 keyword 比對
  `label_group` 關鍵詞），檢查生成內容有沒有講到跟真實 26 類標籤一致的 finding。比 BLEU 更貼近
  「這份報告有沒有講對病灶」而不是「用詞跟真報告像不像」——CT-RATE 那邊沒有做這個，是 CXR track
  因為有 Stage 2 分類器現成可用而多出來的評估手段，之後也可以回頭套用到 CT 版

---

## 4. 產出物 / 檔案規劃

```
U-VLM/cxr/stage3/
├── STAGE3_PLAN.md   (本文件)
├── config.yaml
├── dataset.py         # 讀 manifest.jsonl -> image + report_text
├── model.py            # encoder（沿用 stage2）+ visual injection + LM decoder
├── train.py              # teacher forcing 訓練
├── eval.py                # BLEU/ROUGE + 臨床有效性代理指標
└── checkpoints/
```

**這份文件只到規劃，程式碼尚未實作**，需要等 CXR Stage 1、Stage 2 都訓練完成才開始寫，
且下面的待確認事項要先拍板。

---

## 5. 待確認事項（開始寫 train.py 前需要拍板）

1. **LM decoder 的具體 backbone**：目前只定了「量級跟 GPT-2 相近的現成 decoder-only 模型」，
   還沒選定具體哪一個（例如直接用 GPT-2、或某個小型 medical/clinical 語言模型），需要先確認
   授權/取得方式跟這台機器/遠端 GPU 的資源限制
2. **Visual injection 的實際接法**：prefix tokens（把投影後的視覺特徵當成 LM input 序列最前面幾個
   token）vs cross-attention（LM 每一層額外接一個 cross-attention 去 attend 視覺特徵）——論文用的是
   後者（multi-layer 注入），但實作複雜度更高，需要先確認要不要一次做到位還是先用 prefix tokens 的
   簡化版本起步
3. **Tokenizer / vocabulary**：選定 LM backbone 後才能確認，跟第 1 點連動
4. **是否需要處理報告長度差異**：normal study 的 report_text 很短（常常一句話），abnormal study
   可能是好幾句——訓練時 batch 內長度差異大要怎麼處理（padding/截斷策略），等實際看過 report_text
   長度分布統計後再決定
