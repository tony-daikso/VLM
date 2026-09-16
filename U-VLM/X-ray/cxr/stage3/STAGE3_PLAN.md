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

## 5. 待確認事項 → 已拍板（2026-09-16）

1. **LM decoder backbone**：✅ **GPT-2 (124M)**，`transformers` 已裝（5.17.0），直接
   `GPT2LMHeadModel.from_pretrained("gpt2")`，不從零訓練
2. **Visual injection 接法**：✅ **Cross-attention，忠實於論文的 multi-layer 注入**。實測發現
   `transformers` 的 GPT2 實作原生支援這個模式（`GPT2Config(add_cross_attention=True)`，配合
   `GPT2Model.forward` 的 `encoder_hidden_states` 參數），不用手動改 GPT2Block 內部——載入
   pretrained GPT-2 權重時，self-attention/MLP/embedding 都正常從 checkpoint 載入，新增的
   12 層 `crossattention`/`ln_cross_attn`/`q_attn` 權重是隨機初始化（HF 的
   load-report 會列出這些 MISSING key，預期行為，train.py 會把它們訓練起來）
3. **Visual token 來源**：只用 f4（跟 Stage 2 分類 head 的選擇一致，512 channel, stride 32,
   輸入 512×512 → 16×16=256 個 token），沒有用多尺度——256 個 cross-attention token 對 GPT-2
   規模來說已經足夠，多尺度會讓每層 cross-attention 的 KV 長度暴增，先不做
4. **Tokenizer / padding**：GPT2Tokenizer（BPE），沒有原生 pad token，沿用標準做法把
   `pad_token = eos_token`；batch 內動態 padding 到當下 batch 最長序列，`labels` 在 padding
   位置設 -100（CrossEntropyLoss 自動忽略）；`max_length=96`（PadChest-GR report_text 實測
   median 9 字、p95 31 字、max 81 字，96 token 綽綽有餘，極端長的才會被截斷）
5. **臨床有效性代理指標**：✅ 採用計畫裡的「簡單 keyword 比對」方案（不是把文字餵回 Stage 2 圖像
   分類器——那個模型輸入是影像不是文字，兩者不相容）：檢查生成報告文字裡有沒有出現各
   label_group 的名稱關鍵字，跟這個 study 真正的 `classification_labels` 比對，算 precision/
   recall/F1，這只是粗略 proxy（有些 label_group 名稱不會逐字出現在自然語句裡），不是精確指標

---

## 6. 產出物 / 檔案規劃（實際落地，跟第 4 節一致，補上細節）

```
U-VLM/X-ray/cxr/stage3/
├── config.yaml
├── dataset.py         # manifest.jsonl -> image + tokenized report_text
├── model.py            # Stage 2 fine-tuned ResNet34Encoder（凍結）+ f4 投影 + GPT-2(cross-attn)
├── train.py              # teacher forcing 訓練
├── eval.py                # BLEU/ROUGE + keyword 臨床 proxy 指標
└── checkpoints/
```

Encoder 策略：v1 直接**凍結** Stage 2 fine-tuned 的 encoder（不像 Stage 2 有 freeze→unfreeze
兩階段），理由是 Stage 3 的重點是驗證「視覺特徵能不能餵給 LM 生成合理報告」，先固定視覺端減少
變數；如果生成品質不理想，再回頭考慮要不要解凍 encoder 一起微調。
