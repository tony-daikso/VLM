# U-VLM Stage 3（2D 版）— Report Generation 計畫

參考論文：*U-VLM: Hierarchical Vision Language Modeling for Report Generation*
(arXiv 2603.00479, Shi et al., 2026-02-28)。本文件寫作前已抓取論文全文（11 頁）逐節讀過，
架構細節（§2.1/2.2、Eq.3-5、Table 1/4/5）直接引用論文原文，不是憑印象轉述。凡是跟論文不一樣
的地方，都會在段落裡明講「這裡跟論文不同，因為 XXX」，不含糊帶過。

---

## 0. 跟論文的關鍵差異（先講清楚，避免後面看起來像是抄論文卻兜不起來）

| 項目 | 論文做法 | 這個專案的做法 | 為什麼不一樣 |
|---|---|---|---|
| Encoder 輸入 | 3D volume（256×256×192，patch 128×128×96），6-stage nnU-Net ResEncoder，channel `[32,64,128,256,320,320]` | 2D 單張切片（512×512），5-stage ResNet34，channel `[64,64,128,256,512]` | Stage 1 一開始就決定用 2D 版本（見 STAGE1_PLAN.md 第 4 節），這個決定已經鎖死、Stage 2 也跟著用，Stage 3 沒有理由再改回 3D |
| Stage 2/3 訓練資料量 | CT-RATE 全部 25,692 筆（Stage 2/3 不需要 lesion mask，用 CT-RATE 原生 report + 18 類標籤） | 只用 `data/processed_rex/` 現有 3,042 筆（ReXGroundingCT 子集） | 已跟使用者確認：擴大到全部 25,692 筆需要重新設計「沒有 lesion mask 的切片要怎麼選」，工程量太大，先用手上資料，**預期絕對數字會比論文低很多，這是主動的規模縮減，不是實作錯誤** |
| Decoder | 0.1B、8 層、hidden 512、8 heads，從零訓練 | 沿用論文做法：從零訓練的小型 Transformer decoder（不用任何預訓練 LM 權重） | 已跟使用者確認：論文的核心結論之一就是「從零訓練的小 decoder 打贏 LoRA/全微調的 Qwen3-4B」（Table 5：CT-RATE F1 0.415 vs Qwen3-4B LoRA 0.167 / 全微調 0.156），用預訓練 LM 輕量微調正是論文刻意驗證過「效果更差」的做法 |
| Encoder stage 數 / decoder 層數對應 | N=6 stages，decoder L=8 層，論文只寫「stage N-j+1 → layer j」，L>N 時多出的層怎麼處理沒有交代 | N=5 stages（f0..f4），**decoder 固定用 L=5 層**，跟 5 個 stage 一一對應，不留 ambiguity | 論文原文對 L≠N 的情況沒有說明，我們的 5-stage encoder 剛好可以做乾淨的 1:1 對應，避免猜測論文沒寫清楚的部分 |
| Reference stage r（決定 visual token 長度 K） | 論文說「typically r=N-1」 | 這裡選 **r=N（最深層 f4 當 reference）**，K=256（16×16） | 論文的 r=N-1 是配合他們 3D volume 特定的空間解析度；我們的 f4 在 512×512 輸入、stride 32 下就是 16×16=256，接近論文 ablation 找到的「384 tokens 已經夠、再多沒用」這個量級，而且選 r=N 的好處是**不會有任何 stage 比 reference 更深**，完全不需要設計 zero-padding 這個分支（Eq.4 的 `Pad(f_i)` 分支用不到），實作更乾淨 |

---

## 1. 任務定義

給定 Stage 2 微調過的 encoder（`U-VLM/stage2/checkpoints/best_encoder.pt`）跟一張 512×512 CT
切片，生成一段跟原始放射科報告 Findings/Impressions 段落相似的**自由文字報告**（不是結構化/
模板化輸出——這是跟使用者確認過的選擇，最貼近論文的 "report generation" 定位，但也代表訓練/
評估難度都是三個 stage 裡最高的）。

| 項目 | 設定 |
|---|---|
| 輸入 | 512×512 單張切片（跟 Stage 1/2 完全一致的 windowing：level -600 / width 1500） |
| 輸出 | 自由文字，格式為 `Findings: {...}\nImpressions: {...}`（跟論文 Fig.2 qualitative 範例的段落結構一致，也跟 Stage 2 `dataset.py` 的 `format_report_text()` 使用同一組欄位：`Findings_EN` + `Impressions_EN`，不含 `ClinicalInformation_EN`/`Technique_EN` 這兩段檢查方式制式描述） |
| 樣本範圍 | `data/processed_rex/` 全部 3,042 筆，沿用 Stage 1/2 的 `splits.json` ctrate split（train 2,597 / val 445），**不重新切**——理由跟 Stage 2 第 3 節一樣：保持三個 stage 的 val set 是同一批「沒被任何一個 stage 訓練看過」的樣本，held-out 才有意義 |
| Encoder 來源 | `U-VLM/stage2/checkpoints/best_encoder.pt`（Stage 2 微調後的權重，不是 Stage 1 的） |
| Decoder | 從零訓練的小型 Transformer，架構見第 4 節 |

---

## 2. 論文核心設計（原文引用，附我們的 2D 改寫）

### 2.1 Progressive Training（§2.1）

論文的三階段 loss：

```
Stage 1 (Segmentation): L_seg = L_dice(Y_seg, Ŷ_seg) + L_ce(Y_seg, Ŷ_seg)                      (Eq.1)
Stage 2 (Classification): ŷ = σ(Linear(Flatten(CrossAttn(Q, f_N)))), L_cls = BCE(y, ŷ)           (Eq.2)
Stage 3 (Report Gen):    L_gen = -Σ_t log P(w_t | w_<t, {h_j}_{j=1}^L)                            (Eq.3)
```

我們的 Stage 1/2 已經完成，但**跟論文的 Stage 2 head 設計不同**：論文 Stage 2 用 learnable query
`Q ∈ R^{M×D}` + cross-attention pooling 聚合 `f_N`；我們 Stage 2（見 `U-VLM/stage2/model.py`）
用的是更簡單的 GAP + 2 層 FC（`ClassificationHead`），這是 STAGE2_PLAN.md 第 8 節已經拍板、
訓練驗證過的決定（macro AUROC 0.7410），**不會為了跟論文的 Stage 2 一致而回頭重做**——Stage 3
只依賴 Stage 2 產出的 encoder 權重品質，不依賴 Stage 2 head 的具體架構，兩者是解耦的。

Stage 3 的 loss（Eq.3）我們直接沿用，這是標準的 autoregressive cross-entropy，沒有 domain-specific
的地方，不需要改寫。

### 2.2 Multi-Layer Visual Injection（§2.2, Eq.4-5）—— 改寫成我們的 5-stage 版本

論文的 Feature Alignment（Eq.4）：給定 reference stage `r`，K = 該 stage 的 token 長度，

```
f̃_i = Align_r(f_i) = Pool(f_i)  if i < r
                      f_i        if i = r
                      Pad(f_i)   if i > r
```

**我們的版本**（N=5，r=N=5，即 f4 自己就是 reference，K=256）：

| stage | 原始空間解析度 | 對齊後長度 K |
|---|---|---|
| f0（stride 2） | 256×256 = 65,536 | pool → 256 |
| f1（stride 4） | 128×128 = 16,384 | pool → 256 |
| f2（stride 8） | 64×64 = 4,096 | pool → 256 |
| f3（stride 16） | 32×32 = 1,024 | pool → 256 |
| f4（stride 32） | 16×16 = 256 | 恆等（reference） |

因為 r=N，**沒有任何 stage 比 reference 更深，`Pad(·)` 這個分支完全用不到**——這是跟論文
r=N-1 選擇不同、但刻意簡化的地方（見第 0 節）。Pool 用 `nn.AdaptiveAvgPool2d` 把每個 stage 的
`(C_i, H_i, W_i)` 空間網格降到 `16×16`，flatten 成 `(256, C_i)` 再各自過一層 `Proj_i: C_i → D`
（`D` = decoder hidden dim）。

Skip Connection-Style Injection（Eq.5）：

```
h_j^(v) = LM_j( h_{j-1}^(v) + Proj_j( Align_r(f_{N-j+1}) ) )     # vision token 位置
h_j^(t) = LM_j( h_{j-1}^(t) )                                     # text token 位置，不注入
```

深層 encoder stage（全局語意）接 LM **早期**層，淺層 stage（細節）接 LM **後期**層。我們的
5 層 decoder 對應表（`f_{N-j+1}` with N=5）：

| decoder layer j | 對應 encoder stage | 語意 |
|---|---|---|
| layer 1 | f4（最深，512ch，16×16） | 全局語意（有沒有異常的大方向） |
| layer 2 | f3（256ch，32×32→pool 256） | |
| layer 3 | f2（128ch，64×64→pool 256） | |
| layer 4 | f1（64ch，128×128→pool 256） | |
| layer 5 | f0（64ch，256×256→pool 256） | 細節（邊界、微小結構） |

Attention mask 沿用論文的 hybrid 設計：vision token 之間雙向注意力，text token 用 causal
attention；輸入序列排列為 `[vision tokens (K=256, bidirectional)] + [instruction tokens
(固定 prompt，例如 "Generate a report from the CT image.")] + [target report tokens
(causal, teacher-forced)]`，跟論文 Fig.1 Stage 3 示意圖的排列一致。

### 2.3 Encoder 是否凍結（Table 5 ablation）

論文原文：「Frozen Encoder (F1=0.415) outperforms fine-tuning (0.362), likely because freezing
prevents catastrophic forgetting of discriminative features learned during classification
pretraining」——**Stage 3 訓練時 encoder 全程凍結**，只訓練 decoder + 各層的 `Proj_j` 投影層。
這點直接採用論文結論，不需要自己重新做 ablation。

### 2.4 Decoder 大小 / Tokenizer（論文沒完全講清楚的地方）

論文 decoder：0.1B 參數、8 層、hidden 512、8 heads，但論文訓練資料是全部 CT-RATE 25,692 筆，
我們只有 3,042 筆（約論文的 1/8.4）。**這裡先沿用論文的 0.1B / hidden 512 / 8 heads 當起跑設定**
（decoder 層數改成 5 層對應 5 個 encoder stage，見上面），但因為資料量遠小於論文，過擬合風險更高，
訓練時要比論文更早介入 early stopping、更密集地看 val loss/BLEU 曲線，不能照搬論文的訓練 epoch
數（論文沒有寫 Stage 3 訓練了幾個 epoch）。

Tokenizer：論文完全沒提訓練了什麼 tokenizer、vocab 多大。**這裡的決定**：直接沿用一個現成的
BPE tokenizer 的詞表（例如 GPT-2 的 BPE，~50k vocab）**只借詞表，不借預訓練權重**——decoder
權重仍然是從零訓練，符合論文「decoder 從零訓練」的核心主張；但用現成詞表而不是只拿 3,042 篇
report 訓一個全新的小 BPE，可以避免詞表太小、遇到報告裡的醫學術語/罕見字被切得太碎。

---

## 3. 資料規劃

沿用 Stage 2 `dataset.py` 的 windowing/resize/split 邏輯（見 STAGE2_PLAN.md 第 3 節），差異
只在輸出的 target 從「18 維 label」換成「文字 token 序列」：

1. Image：跟 Stage 1/2 完全一致，512×512，肺窗 windowing，同一份 `splits.json` ctrate split
2. Target text：`manifest.jsonl` 的 `report_text.Findings_EN` + `report_text.Impressions_EN`，
   格式化成 `"Findings: {findings}\nImpressions: {impressions}"`
3. Tokenize：用第 2.4 節選定的 BPE tokenizer，設定一個 `max_length`（先抓 512 token，實際訓練
   前應該先跑一次全體 report 的 token 長度分佈，看 512 涵蓋了多少百分位數，再決定要不要調大——
   不要沒看過分佈就直接假設 512 夠用）
4. Augmentation：沿用 Stage 1/2 的 hflip/旋轉/scale/intensity jitter（純影像增強，不影響文字）

---

## 4. 模型架構

```
image (1,512,512)
   │
   ▼
ResNet34Encoder(frozen, Stage 2 權重) ──▶ f0,f1,f2,f3,f4
   │ (每個 stage 各自 AdaptiveAvgPool2d → 16×16 → flatten → (256, C_i))
   ▼
Proj_1..Proj_5 (Linear: C_i → D)  ──▶ 5 組 (256, D) 對齊後的 visual feature
   │
   ▼
5 層 Transformer decoder，逐層注入對應 encoder stage（見 2.2 節對應表）
   │  輸入序列 = [visual tokens(256, bidirectional)] + [instruction tokens] + [report tokens(causal)]
   ▼
autoregressive 生成報告文字
```

---

## 5. 訓練設定

- **Loss**：標準 autoregressive cross-entropy（Eq.3），只對 report token 段計算 loss，
  instruction/vision token 位置不計入
- **Optimizer**：AdamW；encoder 全程凍結（lr=0，不放進 optimizer），decoder + Proj 層的 lr
  **不直接沿用論文的 2e-5**——論文的 2e-5 是配合他們全部 25,692 筆、可能跑更多 total step 的設定，
  我們資料量少、能跑的 step 數少很多，2e-5 可能學得太慢；起跑建議 **1e-4 ~ 3e-4**（cosine
  scheduler），實際數字要看第一次訓練的 loss 曲線再調，這是需要邊跑邊看的超參數，不是可以先驗證定案的
- **Batch size**：論文用 batch=2（A100 80GB，但他們很多 ablation 是 encoder 不凍結，比較吃記憶體）。
  我們 encoder 全程凍結、decoder 只有 5 層小 Transformer，記憶體用量會小很多，起跑值可以抓高一點
  （例如 16~32），第一次跑的時候比照 Stage 1/2 的做法，實測 forward+backward 記憶體，OOM 就退一級
- **Epochs / early stopping**：monitor val loss 或 val BLEU-mean，論文沒有給 epoch 數可以參考，
  這部分要自己邊跑邊看

---

## 6. 評估指標

- **BLEU-mean**（BLEU-1~4 平均，論文的 B-mean 定義）：標準做法，可以直接用 `sacrebleu`/`nltk`
  算，不需要額外基礎設施
- **F1**（論文：從生成報告裡抽取 18 類病理標籤，跟 ground-truth `classification_labels` 比較）：
  **論文用的是「CT-RATE 自己的 text classifier」，我們沒有這個工具**，需要自己想辦法從生成的
  自由文字報告抽出 18 維 0/1 向量。**建議做法**：用 LLM 對生成報告文字做一次 18 類分類抽取
  （跟 `U-VLM/stage2/SLICE_PER_LABEL_REEXTRACTION_PLAN.md` 裡提到的「用 LLM 把 finding 文字對應
  到 18 類」是同一種做法，這個專案已經有這個模式的先例），只在 val set（445 筆）上跑，控制 LLM
  API 成本
- **質化樣本**：比照論文 Fig.2，固定抽幾筆 val 樣本，每個 epoch（或訓練完後）把「生成報告 vs
  原始 Findings/Impressions」並排存下來，方便人工核對生成品質，尤其要對照 Stage 2 已經發現的
  「切片選取天花板」問題（`U-VLM/stage2/SLICE_PER_LABEL_REEXTRACTION_PLAN.md`）——如果生成報告
  對某些病理描述得不準，先確認是不是那個病灶本來就不在這張切片上，不要一律怪 decoder

---

## 7. 產出物 / 檔案規劃

```
U-VLM/stage3/
├── STAGE3_PLAN.md          (本文件)
├── config.yaml              # 超參數、路徑設定（沿用 stage1/2 splits.json 路徑）
├── dataset.py                # 讀 processed_rex manifest.jsonl -> image + tokenized report
├── model.py                  # 沿用 stage1/2 的 ResNet34Encoder(frozen) + multi-layer injection decoder
├── losses.py                  # autoregressive cross-entropy（只算 report token 段）
├── train.py                    # encoder 凍結、decoder 從零訓練
├── eval.py                      # BLEU-mean + LLM 抽取 18 類做 F1 + 質化樣本 dump
└── checkpoints/
    ├── best_decoder.pt         # 訓練好的 decoder + Proj 層
    ├── tokenizer/               # 借用的 BPE tokenizer 檔案
    └── history.json
```

`U-VLM/stage2/checkpoints/best_encoder.pt` 是這階段的輸入，訓練開始前要先確認它存在
（這台機器上已經有，見 Stage 2 訓練紀錄）。

---

## 8. 待確認事項

以下幾點在動手寫 `U-VLM/stage3/` 程式碼之前，建議再看一次，其中前兩點已經跟使用者確認過，
其餘是本文件起草時依論文結論 + 專案現況做的預設決定，不是已經跑過實驗驗證的數字：

1. **Decoder 從零訓練，不用預訓練 LM**：✅ 已跟使用者確認（見第 0 節），依論文 Table 5 的實驗結果
2. **只用現有 3,042 筆，不擴大到全部 CT-RATE 25,692 筆**：✅ 已跟使用者確認（見第 0 節）
3. **Tokenizer 詞表借用現成 BPE（例如 GPT-2）**：目前是本文件的預設建議，還沒有跟使用者確認過，
   之後開始寫 `dataset.py` 前可以再討論一次
4. **Decoder 學習率 1e-4~3e-4（不用論文的 2e-5）**：這是邊跑邊調的超參數，第一次訓練前只是起跑值
5. **Reference stage 選 r=N（f4，K=256），decoder 固定 5 層**：這是為了讓 5-stage encoder 跟
   decoder 層數乾淨對應所做的簡化，跟論文原文的 r=N-1 不同，原因見第 0 節表格
6. **F1 用 LLM 抽取生成報告的 18 類標籤**：可行但需要 LLM API 存取權限跟额外成本，是否要做、
   要用哪個 LLM，需要再確認
