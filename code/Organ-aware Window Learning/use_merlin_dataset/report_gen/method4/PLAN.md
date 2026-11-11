# 方法四（Merlin）：OWL 特徵 + 分類分數 → LLM 寫醫師原句

更新：2026-11-09 17:10（agent C）｜狀態：**程式骨架完成；CPU smoke test 通過；A6000 上用真的 Lingshu-7B + 合成資料做過 GPU 短測（各 6 step，不存權重）；正式 Stage 1/2 訓練尚未開始，等使用者同意**

本資料夾只放程式與說明，不放報告原文、例句或病人 id。資料、訓練 jsonl、結果都在 `VLM/data/CT/merlin/report_gen/`（Mac）或 A6000 `~/Desktop/Tony/VLM/OWL/report_gen/m4/`。

---

## 0. Stage 1 狀態（2026-11-09 18:00）

**Stage 1（lingshu7b_s1）已於 11-09 17:48 在 A6000 開跑**（使用者同意）。不開任何實驗開關，test 沒碰。

| 項目 | 內容 |
|---|---|
| 訓練目標 | B 的 targets **v2**（17:43；train md5 `bd06411d…`、val `e45c6364…`），凍結複本在 A6000 `m4/inputs/targets_v2/`；之後的 v3 不影響這次 Stage 1 |
| prompt v1 | 期相、機型；**年齡、性別**（metadata，缺就略過）；32 影像 token；14 類 OOF 機率；測量值：腹主動脈最大徑、脾臟上下徑與體積、**肝 HU、脾 HU**（OWL `analysis_biliary/hu_all_final.csv`）、**主動脈到髂動脈管壁鈣化**（OWL `region_units/atherosclerosis/hu_feats.csv` 四段管壁 shell 的最大 HU 與 ≥300 HU 體素數）；缺值寫 unknown，HU 取整數 |
| 新欄位覆蓋率（train / val） | 年齡 99.7% / 99.7%、性別 99.7% / 99.7%、肝 HU 99.98% / 99.92%、脾 HU 98.7% / 98.8%、管壁鈣化 99.95% / 99.88% |
| 筆數 | train 15,306（排除 3 例空白目標）、val 5,055 |
| 長度 | prompt 379 → **432 token**（只用文字版 344 → 397）；目標平均 159（p95 256、最長 440）；全長平均 591、最長 874 |
| 資料版本 | `m4/data/`＝targets v2 + prompt v1（使用中）；`m4/data_v1targets/`＝targets v1 + prompt v1；`m4/data_v0/`＝targets v1 + prompt v0。各資料夾都有 `TARGETS_MD5.txt`，`data/build_stats.json` 記錄所有輸入 md5 |
| 前段 loss（每 20 個 optimizer step 平均） | 1.270 → 1.012 → 0.882（step 20 / 40 / 60），沒有 NaN |
| 資源 | GPU 24.7 GB、100%；RAM 約 3 GB |
| 速度 | 每 optimizer step（16 例）約 6.8 秒，共 956 步，**約 1.8 小時**（比短測慢約 25%：prompt 長 54 token，加上真實資料 I/O） |
| 中止紀錄 | 17:37 第一次啟動（用到有 bug 的 targets v1），載入模型中就停掉，沒有任何 step、沒有權重；log 改名為 `logs/lingshu7b_s1_aborted_badtargets.log` |

啟動指令（已執行）：`ssh A6000 'bash ~/Desktop/Tony/VLM/OWL/report_gen/m4/code/start_stage1.sh lingshu7b_s1'`；log `m4/logs/lingshu7b_s1.log`；權重 `m4/runs/lingshu7b_s1/{epoch1,best,final}/`。
結束後執行 `post_stage1.sh lingshu7b_s1 200`：用 best adapter 對 val 前 200 例產生草稿，再走 A6000 → Spark 的通道做 Qwen 反抽（8 並行）、算文字指標與 BERTScore；結果在 `m4/eval/`。

依實測修正的時間（每 micro-step 約 1.7 s）：Stage 1 約 1.8 h；Stage 2 每 epoch 約 2.4 h（短測 1.87 s × 1.25）；只用文字每 epoch 約 2.2 h。主模型（Stage 2 兩個 epoch）＋消融（兩個 epoch）合計約 11.5–12 h，加推論。

## 0B. Stage 1 結果（11-09 19:40 完成）

- 訓練：17:48 → 19:39（6,642 s，956 個 optimizer step）。train loss 每 20 步平均：1.270（step 20）→ 0.826（140）→ 0.764（300）→ 約 0.73–0.80 持平（step 400 之後）→ 0.757（956）。**val loss（val 前 400 例）0.711**。只跑 1 個 epoch，所以 best = epoch1 = final。沒有 NaN；GPU 24.7 GB。
- 權重：A6000 `m4/runs/lingshu7b_s1/{best,epoch1,final}/adapter.pt`（169 MB）+ `manifest.json`。
- val 前 200 例推論（`post_stage1.sh`）：200 例都有輸出、tag 全部能解析，平均每例 9.4 類、7.7 行、46 個字（醫師目標 72 個字）。草稿與結果在 A6000 `m4/eval/`。
- 評估（Qwen 走 Spark 18003、8 並行，200/200 反抽成功；owl 定義、嚴格）：

| 指標 | 數值 |
|---|---|
| 疾病有無 micro 敏感度／特異度／PPV／F1 | .423 / .988 / .770 / .546（macro F1 .426） |
| 各類 F1 | 膽囊切除術後 .870、脾腫大 .778、骨折 .643、動脈粥狀硬化 .617、腸阻塞 .560、腹主動脈瘤 .500（2 例陽性）、骨質減少 .400、腹水 .400、膽結石 .381、膽管擴張 .316、脂肪肝 .273、腎水腫 .222、闌尾炎 0、游離氣體 0 |
| **關鍵錯誤率**（原報告陽性、生成寫陰性） | **97 / 246 = 39.4%**；漏寫 40 / 246。最嚴重：闌尾炎 6/6、腎水腫 6/7、脂肪肝 30/37、膽管擴張 12/15、膽結石 13/17、腹水 13/22 |
| 屬性（兩邊都陽性時） | 樣本很少：腸阻塞類型 7/7、腎水腫側別 1/1、AAA 直徑 1/1；骨折部位 4/9、椎體節段 0/7（生成多半沒寫節段）；動脈粥狀硬化程度、脂肪肝分級、脾長度幾乎沒寫 |
| 文字相似度 | BLEU-4 .228、ROUGE-L .486、BERTScore F .900（P .915 / R .885） |

解讀：Stage 1 只訓 adapter、LLM 凍結，模型學會醫師的版面與陰性樣板句（BERTScore .90），但多半寫陰性，陽性句與屬性還沒學起來，關鍵錯誤率很高。這符合「Stage 1 只對齊影像 token」的預期；疾病判斷與屬性要靠 Stage 2（LoRA）。Stage 2 與只用文字的消融要等使用者看過停點報告才開始。

## 1. 架構

```
每例 OWL 單元特徵 tok [1082, 448] fp16 + mask        （pack_features.py，A6000）
  → LayerNorm + Linear 448→1024，加 type embedding（group / unit / region / window / pool 五種 embedding 相加）
  → 加一個永遠有效的 null token（避免單元全缺時注意力變 NaN）
  → Perceiver resampler：32 個 learned latent，2 層 cross-attention（latent 同時看自己，Flamingo 式），8 head，FFN×4
  → LayerNorm → MLP（1024→3584→3584）= 32 個影像 token
  → 填進 prompt 中 32 個影像 placeholder 的位置（masked_scatter 進 LLM 的 input embedding）
文字 prompt：掃描期相、機型；14 類 OWL 機率（A 的 train OOF / val 全 train 模型，已校正）；
            影像測量值（腹主動脈最大徑、脾臟上下徑換算成 cm，脾臟體積）
輸出：醫師原句，一句一行，行首用 [疾病 tag] 標出這句涵蓋哪幾類，例如「[ascites,free_air] <原句>」
     → infer.py 解析成 {case_id, findings: {disease: sentence}}，直接給 eval/evaluate.py
```

- 只用 LLM，不載入任何原本的 vision encoder（OWL 特徵經 resampler 直接進 LLM embedding 空間）。
- 參數量：adapter（type embedding + resampler + projector）約 42 M；LoRA r16 套在 q/k/v/o/gate/up/down，7B 約 40 M。
- 只用文字的消融（`--text-only`）：同一個 LLM、同一份 prompt（只少了影像區塊）、同一套 LoRA 設定，不跑 Stage 1。
- tag 格式的理由：同一句常同時涵蓋多類（例如腹膜腔一句同時講腹水與游離氣體），tag 讓輸出能對回 14 類，而句子本身維持醫師原文。tag 不會進評估文字（evaluate.py 用 findings 去重組裝）。

### 1.1 OWL 單元特徵（A6000 唯讀確認，2026-11-09）

| 群組 | 每例 shape | 來源（`~/Desktop/Tony/VLM/OWL/mw_exp/` 底下） | token 數 |
|---|---|---|---|
| organ3：12 器官 × 5 window（mean） | emb [12,448] ×5 檔 | `outputs_final/emb/<window>/`（25,489 例） | 60 |
| bone3：17 骨骼單元 × 3 window × mean/max | [17,448] ×2 | `outputs/emb_bones/<window>/`（25,489） | 102 |
| rib3：18 肋骨單元 × 3 window × mean/max | [18,448] ×2 | `outputs/emb_ribs/<window>/`（20,364 = train+val） | 108 |
| vert15：T8–L5 1.5 mm × 3 window × mean/max | [10,3,448] ×2 | `outputs_vert15/merged_final/emb15/`（20,363） | 60 |
| rib15：18 肋骨 1.5 mm × 3 window × mean/max | [18,3,448] ×2 | `outputs_ribs15/emb15/`（20,355） | 108 |
| 區域單元（9 種 crop：腎竇、腹膜腔 pocket、血管壁、橫膈下/前腹、膽囊、肝門、腸道、骨盆、右下腹） | [K,R,W,896=mean‖max] | `outputs_final/region_units/*/emb/`、`appendicitis_rlq_unit/emb_rlq/`（各 20,364） | 644 |
| **合計** | | | **1,082** |

- 每個向量是 stage 1–3 各自 L2 normalize 後串接（64+128+256），抽查向量範數 = √3。
- 1.5 mm 椎體 / 肋骨與區域單元只有 train+val，test 本來就沒抽，正好符合 test 鎖住的規則。
- 打包：每例一個 npz（tok fp16 [1082,448] + mask），寫在 A6000 `~/Desktop/Tony/VLM/OWL/report_gen/m4/features/`。**已完成（11-09）：20,364 例（train 15,309 / val 5,055），共 19 GB**；每例有效 token 中位數 1,034（平均 1,031，最少 26；少於 500 的 19 例）。單一 process、nice 19 + ionice idle，約 35 分鐘；log `m4/logs/pack.log`。有些區域單元檔只有 present=False、沒有 emb，當作該單元缺（mask=False）。

### 1.2 Backbone（可替換，`m4_common.BACKBONES`，train/infer 用 `--backbone` 選）

| 項目 | **lingshu7b（主模型）** | **lingshu_i8b（備選）** |
|---|---|---|
| 來源 | lingshu-medical-mllm/Lingshu-7B（Spark `~/Desktop/Tony/models/Lingshu-7B`，16 GB） | lingshu-medical-mllm/Lingshu-I-8B（尚未下載，約 16 GB，MIT） |
| 原架構 | Qwen2.5-VL-7B（`Qwen2_5_VLForConditionalGeneration`） | InternVL3-8B，HF 上是 transformers 原生格式（`InternVLForConditionalGeneration`，transformers 4.52.4 存檔） |
| LLM | Qwen2.5-7B 架構：hidden 3584、28 層、28 head / 4 KV head、vocab 152,064 | Qwen2ForCausalLM：hidden 3584、28 層、28 / 4 head、vocab 151,674 |
| 權重 key | `model.layers.*`、`model.embed_tokens`、`lm_head`（與 Qwen2ForCausalLM 相同；另有 384 個 `visual.*`） | `language_model.model.*`、`language_model.lm_head`（另有 `vision_tower.*`、`multi_modal_projector.*`） |
| 取 LLM 的方式 | loader `qwen2_from_vlm`：用 text 欄位建 Qwen2Config，在 meta device 建模型，從原 safetensors 只讀 LLM key（不複製權重、不需 trust_remote_code） | 同一個 loader，加 key 對應 `language_model.model.→model.`、`language_model.lm_head.→lm_head.`，config 取 `text_config` |
| 位置編碼 | 原本是 M-RoPE（mrope_section 16/24/24）；純文字位置三段相同，等同 1-D RoPE（θ=1e6），所以改用一般 RoPE | text_config 帶 dynamic RoPE scaling ×2（只在超過 32k token 時才有作用，本任務約 600 token，不影響） |
| Chat 模板 | ChatML（`<|im_start|>` / `<|im_end|>`） | 也是 ChatML（HF 的 chat_template.jinja），模板本身沒有預設 system |
| 影像 placeholder | `<|vision_start|>` + 32×`<|image_pad|>` + `<|vision_end|>` | `<img>` + 32×`<IMG_CONTEXT>` + `</img>`（原本一張圖是 256 個 IMG_CONTEXT，我們只用 32 個，因為 resampler 自己產生 token） |
| 醫療預訓練 | 醫療多模態 SFT（含報告生成） | 同團隊、同資料，InternVL 系 |

- 協調者提到 Lingshu-I-8B 要 trust_remote_code、模板是 InternVL 格式：我查了 HF 上的 config.json / chat_template.jinja / safetensors index（只抓這三個小檔），**目前 HF 版本是 transformers 原生 InternVL 格式，不需要 remote code，模板是 ChatML**。為保險另外保留 `lingshu_i8b_remote` loader（`AutoModel(trust_remote_code=True).language_model`），如果拿到的是舊版 InternVLChatModel 格式就用它。
- 我們自己寫 ChatML（`m4_data.chatml`），不依賴各模型的 chat_template，所以訓練和推論的格式完全一致；換 backbone 只差影像 placeholder 字串、tokenizer 和 loader。
- 另有 `qwen3_1p7b`（A6000 已有，Qwen3 用 `<think></think>` 空區塊關掉思考）供除錯，`tiny_random` 只給 smoke test。
- `prepare_llm.py`：可選，替 Lingshu-7B 做一個「只有 LLM」的資料夾（config + tokenizer 複本 + safetensors symlink），之後用 `causal_lm` loader 或 vLLM 載入；訓練本身不需要它。

## 2. 資料流

| 步驟 | 程式 | 在哪跑 | 輸入 → 輸出 |
|---|---|---|---|
| 1. 打包特徵 | `pack_features.py` | A6000（CPU） | mw_exp 各特徵（唯讀，先用 split.csv 濾掉 test）→ `m4/features/main/<id>.npz`、`token_meta.csv`、`manifest.csv` |
| 2. 組訓練資料 | `build_jsonl.py` | A6000 或 Mac scratch（產物含醫師原句，不放 code 資料夾、不 commit） | B 的 `targets_{train,val}.jsonl` + A 的 `oof/out/scores_trval_prob.csv`（train = 5-fold OOF、val = 全 train 模型）+ `measure_trval.csv` + `method1/measure_calibration.json` + metadata → `train.jsonl` / `val.jsonl`（case_id, split, feature_path, prompt, target, target_diseases） |
| 3. Stage 1 | `train.py --stage 1` | GPU（待確認） | LLM 凍結，只訓練 adapter → `runs/<name>/{epochN,best,final}/adapter.pt + manifest.json` |
| 4. Stage 2 | `train.py --stage 2 --init-adapter ...` | GPU | LoRA r16/α32/dropout 0.05 + adapter → `adapter.pt`、`lora/`、`manifest.json` |
| 5. 消融 | `train.py --stage 2 --text-only` | GPU | 同上但沒有影像 token |
| 6. 推論 | `infer.py` | GPU | val.jsonl → `gen_val.jsonl`（{case_id, findings, raw}），greedy、最多 320 token |
| 7. 評估 | `../eval/evaluate.py` | Mac 或 A6000（Qwen 透過 ssh tunnel） | 與方法一同一批 val 病人 |

- 分類分數洩漏：train 的分數一律用 A 的 OOF；build_jsonl.py 只讀 A 的檔案，遇到 split=test 的列直接拒絕。
- 測量值換算：`measure_calibration.json` 是 A 在 train 上擬合的兩參數線性換算（影像量測 → 報告 cm），只用 train，參數少，洩漏可忽略。
- 每個 epoch 都存 `epochN/`，另存 `best/`（val loss 最低）和 `final/`；`manifest.json` 記錄所有參數、backbone 設定、LLM 路徑、token_meta 與 train/val jsonl 的 md5，可直接拿去推論或之後的 test。
- DataLoader 一律 `pin_memory=False`。train.py / infer.py 用 GPU 前會讀 nvidia-smi，空閒記憶體不足（例如 CG 的 vLLM 還在）就拒絕執行，除非加 `--force-gpu`。

### 訓練設定（預設值，沿用 CG 設計）

| 項目 | Stage 1 | Stage 2 |
|---|---|---|
| 可訓練 | adapter（type emb + resampler + projector） | LoRA（q/k/v/o/gate/up/down）+ adapter |
| lr | 1e-3 | LoRA 2e-4、adapter 1e-4 |
| 排程 | warmup 3% + cosine | 同左 |
| epoch | 1 | 3（建議先 2） |
| batch | micro 4 × accumulation 4 = 16 | 同左 |
| 長度 | max 1,024（prompt 從不截斷；目標超長才截） | 同左 |
| 其他 | bf16 autocast、gradient checkpointing、loss 只算目標 token、每 epoch 算 400 例 val loss | 同左 |

## 3. CPU smoke test（已完成，沒有用 GPU）

`smoke_test.py`：tiny 隨機 Qwen2（hidden 64、2 層）+ Lingshu-7B 的 tokenizer（只複製 tokenizer 檔到 scratch）+ 合成資料（6 例、50 token 的假特徵，句子是自己寫的通用句，不是報告內容）。結果全部通過：

| 檢查 | 結果 |
|---|---|
| 兩種 backbone loader（Lingshu-7B 式 key、Lingshu-I-8B 式 `language_model.*` key） | 載入權重與原 tiny 模型完全相同 |
| batch shape | input_ids [4, 468]、tok [4, 50, 448]、mask [4, 50]；每列剛好 32 個影像 placeholder |
| 某例所有單元都缺（mask 全 False） | loss 有限，沒有 NaN |
| Stage 1 | adapter 有梯度、LLM 0 個參數有梯度 |
| Stage 2 | LoRA 與 adapter 都有梯度 |
| 只用文字 | prompt 348 token（有影像版 383），沒有 placeholder |
| 生成（left padding）與 tag 解析 | 正常；8 類合成句全部對回 |
| adapter / LoRA 存讀 | 一致 |
| train.py stage 1 → stage 2 → infer.py，以及 text-only stage 2 → infer.py | 各跑 2 個 optimizer step（CPU、tiny 模型），全部成功，epoch/best/final 權重與 manifest 都有寫出 |

實際 7B 權重在 Mac 上沒有載入（不下載大模型到 Mac）。Lingshu-7B 的 key 對應是用 Spark 上的 index 檔確認；Lingshu-I-8B 用 HF 上的 index 檔確認。

## 4. 時間與記憶體（A6000 實測 + 預估）

### 4.1 A6000 GPU 短測（2026-11-09，`gpu_smoke.py`）

條件：真的 Lingshu-7B（`qwen2_from_vlm` loader，bf16）、32 例**合成資料**（隨機 OWL 向量、合成 prompt 與自己寫的通用句，不含任何報告原文）、micro-batch 4、序列平均 604 token（text-only 569）、gradient checkpointing、AdamW；每個設定 6 個 optimizer step，取第 2–6 步中位數。環境：A6000 的 `~/Desktop/Tony/tools/vllm-env`（torch 2.13 cu130 + transformers 5.19，CUDA 13.4 forward-compat）＋ `~/Desktop/Tony/tools/m4_pylib`（peft 0.21.2、accelerate 1.15，`pip --target` 裝在 Tony 資料夾，沒動任何既有環境）。測完沒存任何權重。

| 設定 | 可訓練參數 | 每 micro-step（4 例） | GPU 峰值（allocated / reserved） | 一個 epoch（15,309 例 = 3,827 step） |
|---|---|---|---|---|
| Stage 1（只訓 adapter） | 42.3 M | 1.35 s | 20.2 / 21.7 GB | **1.44 h** |
| Stage 2（LoRA r16 + adapter） | 82.7 M（LoRA 40.4 M） | 1.87 s | 20.7 / 22.2 GB | **1.98 h** |
| 只用文字 Stage 2（LoRA） | 40.4 M | 1.71 s | 19.7 / 21.5 GB | **1.81 h** |

- 載入模型 5–13 秒（page cache 熱的時候）；loss 正常下降（合成資料每例目標相同，只用來確認梯度有作用）。
- 記憶體只用了 48 GB 的一半不到；需要時可把 micro-batch 提到 8（每 step 時間約加倍，總時間差不多）。
- 生成：短測裡 batch 16 × 64 個新 token 花 11.5 秒，但那時模型開著 gradient checkpointing（HF 會關掉 KV cache），是上限值。infer.py 另外載入、KV cache 有開，預估 val 5,055 例（每例約 180 個輸出 token）約 **45 min–1 h**／每個系統（未實測）。

### 4.2 總時數預估（A6000，依實測換算；實際序列長度依 B 的目標句而定，誤差約 ±20%）

| 項目 | Stage 2 兩個 epoch | Stage 2 三個 epoch（CG 原設計） |
|---|---|---|
| 主模型 Lingshu-7B：Stage 1（1 epoch）+ Stage 2 | **5.4 h** | 7.4 h |
| 只用文字的消融（沒有 Stage 1） | **3.6 h** | 5.4 h |
| val 推論 2 個系統 | 約 1.5–2 h | 約 1.5–2 h |
| **主模型 + 消融合計** | **約 10.5–11 h** | 約 14–15 h |
| **再加 Lingshu-I-8B 一組**（LLM 同為 7.6 B Qwen2，每 step 時間應相同） | **+5.4 h 訓練 + 約 0.8 h 推論**，另需下載約 16 GB 到 A6000 | +7.4 h + 0.8 h |
| Qwen3-1.7B 除錯（1/3 epoch） | 約 15–20 min | — |

- Qwen 反抽評估另計：Spark `merlin-qwen-fp8`（:8003）並行 8，600 例子集每個系統約 15–30 分鐘（依 Spark 速度，見 eval/README）。
- Spark 依指示不跑訓練或 smoke test（共享記憶體，曾 OOM），沒有實測；只在 A6000 上規劃訓練。
- CPU RAM：A6000 53 GB，短測期間系統 used 約 2 GB（權重經 safetensors mmap 讀入）；特徵打包同時在跑（單一 process、nice/ionice），沒有互相影響。

## 4A. 實驗開關（使用者 11-09 同意；預設全部關閉，Stage 1 baseline 不受影響）

| 開關 | 作用 | 實作 |
|---|---|---|
| `--aux-unit-loss <w>`（需 `--unit-labels-dir`） | resampler 之前，每個 OWL 單元 token 接一個 2 層 MLP（1024→256→14），預測「這個單元有沒有某類異常」；BCE 只算有標籤的（單元, 疾病）格；總 loss = LM loss + w × aux loss | `m4_model.ImageAdapter(aux_unit=True)`；標籤由 `build_unit_labels.py` 產生 |
| `--lr-flip-prob <p>` | 以機率 p 對調左右成對單元（左右腎、腎竇 crop、左右肋骨各段、左右髖/股骨、左右髂總動脈、左右髖臼），同時把 prompt 與目標句的 left/right、Left/Right、lt/rt、單獨的 L/R 對調；`L1`、`L-2`、`L 3` 這類椎體名稱不動；單元標籤一起對調 | `m4_common.LR_PAIRS / swap_lr`、`m4_data.lr_permutation`；只在 train 生效 |
| `--attr-token-weight <w>` | 屬性 token（側別、節段、分級、尺寸數字等）的 LM loss 乘以 w；用 tokenizer 的 offset 對到 B 的 attributes 值（加同義詞，例如 marked→severe）在目標句中的位置，另外用規則抓「數字＋cm/mm」與椎體節段 | `build_jsonl.attr_terms` → jsonl 的 `attr_terms`；`m4_data.attr_char_spans`；loss 用加權平均 Σwᵢ·CEᵢ / Σwᵢ |
| `--resampler-groups` | 32 個 latent 依解剖區域分組，每組只看自己群組的單元（加 null token 與同組 latent）：肝膽 5、腎 3、脾 2、主動脈 3、腸 4、腹膜腔 3、脊椎 5、肋骨 4、骨盆 3 | `m4_common.RES_GROUPS`、`m4_model.group_attn_mask`（cross-attention 的 attn_mask） |

- 四個開關都可以疊加；`--text-only` 不能配 aux / groups（沒有影像 token）。validation loss 一律只算一般 LM loss，才能跨設定比較。
- 存檔的 `manifest.json` 會記錄開關，`infer.py` 依 manifest 重建同一個 adapter（含 aux head、分組遮罩）。

### build_unit_labels.py（單元層級標籤）

從 B 的 targets（status 與 attributes）回推，status pos → 1、neg → 0、uncertain / 沒提到 → 不給標籤：
- **腎水腫**：側別 left/right → 該側腎（organ3 kidney_*、腎竇 crop）=1、對側=0；bilateral → 兩側 1；沒寫側別或移植腎 → 無法對應；陰性 → 兩側 0。
- **骨折**：椎體節段 T8–L5 → vert15 與 bone3 對應節段；肋骨描述有側別＋第幾根 → rib3 / rib15 的 rib_<側>_<n>（1–4 共用一個單元）與 bone3 ribs_<側>；部位含薦椎 → bone3 sacrum 與骨盆 crop k3。陽性病例其他單元不給標籤（可能有沒寫到的骨折）；陰性 → 所有椎體、肋骨、骨盆單元 0。
- **腸阻塞**：SBO → 小腸（+ 十二指腸 crop）、LBO → 大腸、both → 兩者；沒寫類型 → 無法對應。
- **其他 11 類**：器官層級，把病例標籤給該器官或區域 crop（肝、脾、膽囊、主動脈＋血管壁 crop、腰椎＋椎體、右下腹、腹膜腔 pocket、橫膈下、肝門）。這是粗標籤，只是「單元能對應的盡量對」。
- 輸出 `unit_labels_{train,val}.jsonl` 與 `unit_label_stats.{json,md}`（每類陽性中能對應到特定單元的比例）。實際比例要等 B 的 targets 完成後在 A6000 上跑（`python build_unit_labels.py --targets ... --meta features/token_meta.csv --out m4/unit_labels`）。

### OWL 現有單元與分類頭能涵蓋哪些屬性（A6000 唯讀確認）

| 屬性 | 輸入端有沒有位置／數值資訊 | 現有分類頭或分數 |
|---|---|---|
| 腎水腫側別 | **有**：左右腎 organ 單元、左右腎竇 crop（k0=右、k1=左）各自是獨立 token | **沒有左右分開的分類頭**：正式最佳是兩腎 MIL max（`region_units/hydronephrosis/out/oof_train.csv` 只有 U_mil / U_mean / U_org_mil 的病例層級分數），每顆腎的分數沒存 |
| 骨折椎體節段 | **T8–L5 有**（3 mm bone3 與 1.5 mm vert15 各 10 節）；**T1–T7、頸椎沒有任何單元**；薦椎只有一個整體單元 | 正式最佳 a6x10w2_tv7 是 site-wise 堆疊（肋骨／骨盆／椎體三個部位分數取 max），A 匯出的分數只有病例層級；部位層級分數只是中間檔（例如 `push90b/prod/*/score_fold*.csv` 的肋骨分數也是病例層級），**沒有每節椎體或每根肋骨的分數** |
| 骨折肋骨側別／第幾根 | **有**：左右各 9 段（1–4 合併、5–12 各一）× 3 mm 與 1.5 mm | 同上，沒有每根肋骨的分數 |
| 骨盆／髖部骨折側別 | 部分：左右髖臼 crop、左右髖骨與股骨單元 | 只有病例層級 |
| 腹主動脈瘤直徑 | **有數值**：prompt 內的腹主動脈最大徑（A 用 train 校正成 cm） | 分類分數是病例層級 |
| 脾臟大小 | **有數值**：上下徑（校正）與體積 | 同上 |
| 脂肪肝分級 | 只有 embedding；肝／脾 HU 特徵在 OWL 有算（`analysis_steatosis/hs_*.csv`），**目前沒放進 prompt** | 病例層級 |
| 腹水量 | 只有三個 pocket crop（Morison、脾周、骨盆）的 embedding，沒有體積數值（CG 那邊的 fluid_ml 不可靠） | 病例層級 |
| 腸阻塞類型（SBO/LBO） | 小腸、大腸、胃、十二指腸各自是單元 | 病例層級，沒有分類型的頭 |
| 膽管擴張部位／總膽管徑 | 肝門 crop 與肝單元；**沒有管徑數值** | 病例層級 |
| 動脈粥狀硬化程度 | 血管壁 crop；分段管壁鈣化 HU 在 OWL 有算，**沒放進 prompt** | 病例層級 |
| 骨折新舊、壓迫性、闌尾穿孔／糞石、膽結石鈣化／大小、AAA 是否修補、游離氣體是否術後 | **沒有專門的輸入**（只能靠 embedding；術後與否還要臨床資訊，影像上不一定看得出來） | 沒有 |

結論：側別與節段（T8–L5）、肋骨位置在輸入特徵裡有位置資訊，但分類頭全部是病例層級，模型只能自己從單元 token 學位置，這正是 `--aux-unit-loss` 和 `--resampler-groups` 想補的地方。T1–T7 與頸椎骨折節段、總膽管徑、腹水量、骨折新舊在輸入端沒有直接資訊，寫錯時不能怪模型。若要補，最便宜的是把肝／脾 HU 和管壁鈣化 HU 加進 prompt（OWL 已經算好）。

### smoke test（CPU，tiny 隨機模型 + 合成資料，2026-11-09）

四個開關全部通過：
- `swap_lr`：「Moderate left hydronephrosis. R hip. L1 and L-2, L 3. Right-sided. lt kidney」→ 左右對調，L1 / L-2 / L 3 不動。
- 左右配對：合成 token 表中找到 12 對（左右腎、腎竇 crop、肋骨、髖骨等），椎體不會被對調；翻轉後特徵、單元標籤都依同一個排列對調，目標句變成「Moderate right hydronephrosis」「left 5th rib」「L hip」，L1 不變。
- 單元標籤：左腎水腫 → 左腎=1、右腎=0，翻轉後變成右腎=1；骨折 L1 與右第 5 肋 → vert15 L1、rib3 rib_right_5 都是 1。
- 屬性加權：被加權的 token 是 Mild、diffuse、Moderate、right、L、1、left、5、th、rib 等。
- 分組 resampler：只改肝的特徵時，只有肝膽組的 5 個 latent 輸出改變，其他 27 個完全不變。
- aux loss：有限值、aux head 有梯度。
- train.py 四個開關全開跑 Stage 1 → Stage 2 → infer.py：成功；預設（全關）的 Stage 1 / Stage 2 / text-only / infer 也再跑一次，全部成功。

## 4B. 提示來源對照（OOF vs 真實標籤＋dropout）與錯誤提示測試（11-09 加入；預設關閉）

背景：DCP-PD（CT-RATE 胸部報告）用分類器預測當訓練提示，F1 0.438，跟不給提示一樣；改用真實標籤當訓練提示、每個 finding 以 0.3 機率丟掉、推論時換成分類器預測，F1 0.538（不確定它們的預測是不是 OOF）。這裡在 Merlin 上直接比較兩種做法。TASK.md 第 1 條（給 LLM 的 train 分數一律用 OOF、不能用 in-sample 分數）仍然有效：`oof` 就是現行做法；`gt_dropout` 是刻意的 teacher forcing，用的是報告標籤、不是分類器分數，不違反該條。**val／推論一律用分類器分數，絕不用真實標籤**（程式強制，見下）。

### 開關（`train.py`，只作用在 train 列；val loss 與推論永遠用 jsonl 裡原本的分類器那一行）

| 開關 | 預設 | 作用 |
|---|---|---|
| `--cue-source oof\|gt_dropout` | `oof` | `oof`＝現狀（jsonl 的 prompt：train 5-fold OOF、val 全 train 模型）。`gt_dropout`＝每次讀到一筆 train 資料時，用該例的報告標籤重寫「Image classifier probabilities (0-1): …」那一行，每個 finding 以 `--cue-dropout` 機率拿掉；每個 epoch 重新抽 |
| `--cue-gt <cue_gt.json>` | — | `build_cue_gt.py` 的輸出（gt_dropout 必填） |
| `--cue-dropout p` | 0.3 | 只在 gt_dropout 用 |
| `--cue-gt-map sample\|hard` | `sample` | 標籤 → 數字的方式（見下） |
| `--cue-gt-hi / --cue-gt-lo` | 0.95 / 0.02 | `hard` 用 |
| `--meas-dropout p` | 0 | 每個量化項目（主動脈徑、脾臟長度與體積、肝／脾 HU、管壁鈣化）以機率 p 改成「unknown」（就是現有的缺值寫法）；兩種 cue source 都能用。人口學與期相、機型不動 |

**真實標籤怎麼變成和 OOF 一樣的格式**（`m4_common.gt_cue_probs`）：
- 標籤二元化和 OWL 分類器訓練時相同：yes / probably_yes → 陽性；no / probably_no / not_mentioned → 陰性；unclear → 不寫（OOF 分數在 unclear 病例本來就是空的，prompt 本來就不列，兩邊一致）。
- 被 dropout 的 finding 直接從那一行拿掉，跟現有「沒有分數就不列」的寫法一樣；14 類全被丟掉時整行拿掉（等於現有「完全沒有分數」的格式）。
- `sample`（預設）：陽性 → 從 train OOF 中「標籤陽性且分數 ≥ 方法一門檻 t_f1」的分數隨機抽一個；陰性 → 從「標籤陰性且分數 < t_f1」抽。所以數字長得像真的分類器輸出（同樣兩位小數、同樣的分布形狀），但一定落在標籤那一側。門檻與分數池只用 train（`method1/thresholds.csv` 的 t_f1、`oof/out/scores_trval_prob.csv` 的 train 列），不含 val 資訊。
- `hard`：陽性一律 0.95、陰性 0.02（較接近 DCP-PD 的二元提示，但推論時會看到訓練沒見過的中間值）。
- 改寫只換掉那一行，用和 `build_jsonl.make_prompt` 一模一樣的字串格式（`m4_common.render_cue_line`）；真實資料 20,364 個 prompt 解析後重組逐字相同。

**防呆**：`cue_gt.json` 只存 train split 的標籤（val、test 的標籤根本沒寫進去）；`M4Dataset` 帶 gt 提示時遇到非 train 列直接停止；推論模式（`train=False`）一律忽略 gt 提示；train.py 的 val 資料集不接這些參數；`manifest.json` 記錄 cue 開關與 `cue_gt_md5`。

### 錯誤提示測試（`infer.py`＋`evaluate.py`，只在 val）

- `infer.py --cue-gt cue_gt.json --cue-flip-pos2neg 0.5 --cue-flip-neg2pos 0.05 [--cue-flip-seed 20261109]`：用方法一門檻 t_f1 判斷分類器提示是陽還是陰；陽性提示以機率 0.5 改成「陰性樣子」的數值（從 train 陰性池抽），陰性提示以機率 0.05 改成「陽性樣子」的數值（從 train 陽性池抽），再生成。亂數用 `seed:case_id` 逐例設定，跟 batch 順序、`--limit`、子集都無關；同一個 seed 機率調高只會多翻、不會換掉原本翻的。非 val 列拒絕執行。
- 輸出 `<out>.cues.jsonl`（每例原本 14 類提示＋被翻的 finding：[原值, 新值, 方向]）與 `<out>.cues.meta.json`（門檻、機率、seed、翻轉數）。不加翻轉、只給 `--cue-gt` 時也會寫 log（記錄原本提示），方便算「分類器自己就錯」的情形。
- `evaluate.py --cue-log <out>.cues.jsonl`：每類分別算
  - 陽→陰：翻轉數、其中提示錯（原報告陽性）的格數、報告仍寫陽性（與原報告一致）的比例、其中寫成明確陰性（關鍵錯誤）的數量；
  - 陰→陽：翻轉數、其中提示錯（原報告陰性）的格數、報告仍寫陰性的比例；
  - 沒翻、但分類器本身就錯的提示（自然 FN／FP）的一致率。
  - 表格裡脂肪肝、骨折、膽結石標「分類器較弱」。參考標籤用 owl 定義（unclear 排除），生成用嚴格定義（yes / probably_yes）。
- 用 (0.5, 0.05) 在真實 val 提示上試算（只算翻轉、沒有生成）：全部 val 5,055 例，陽→陰 3,760 格（其中原報告陽性 2,618）、陰→陽 3,046 格（原報告陰性 2,935），平均每例 1.35 格；600 例開發子集，陽→陰 605（原陽性 446）、陰→陽 359（原陰性 337），每例 1.6 格。子集中腹主動脈瘤、闌尾炎的陽→陰只有 6、7 格，這兩類要看全部 val。

### 實驗矩陣（全部在 val；test 鎖住）

| 組別 | 輸入 | 訓練提示 | 指令重點（A6000，`m4/code`） | 估計訓練時間 |
|---|---|---|---|---|
| 方法一 | 樣板句 | — | 已完成（`method1/`） | — |
| T-oof | 提示＋量化（不給 embedding） | OOF | `train.py --stage 2 --text-only --out runs/T_oof` | 2 epoch 約 4.4 h（1 epoch 2.2 h） |
| T-gt | 提示＋量化 | 真實＋dropout 0.3 | 同上 ＋ `--cue-source gt_dropout --cue-gt m4/cue/cue_gt.json` | 約 4.4 h |
| U-oof | 提示＋量化＋32 單元 token | OOF | `train.py --stage 2 --init-adapter runs/lingshu7b_s1/best/adapter.pt --out runs/U_oof` | 約 4.8 h（Stage 1 已在跑，1.8 h） |
| U-gt | 提示＋量化＋32 單元 token | 真實＋dropout 0.3 | 同上 ＋ `--cue-source gt_dropout --cue-gt …` | 約 4.8 h（共用同一個 Stage 1；另跑 gt 版 Stage 1 再加 1.8 h） |

- 其他參數全部相同：Lingshu-7B、LoRA r16/α32、micro-batch 4 × accumulation 4、Stage 2 lr 2e-4（adapter 1e-4）、max-len 1024、bf16、pin_memory=False；四個舊開關（aux、左右翻轉、屬性加權、分組 resampler）都不開。
- 時間依 Stage 1 實測（每 micro-step 約 1.7 s）換算：Stage 2 每 epoch 約 2.4 h、只用文字約 2.2 h。gt_dropout 每步多做的只是字串改寫，prompt 反而短一點，時間視為相同。**四組 Stage 2 依序跑（2 epoch）約 18.4 h；其中 T-oof、U-oof 原本就在計畫內，新增的 T-gt、U-gt 約 9.2 h**。
- 推論：每組 val 推論一次＋錯誤提示推論一次。全部 val 每次約 45 min–1 h（**待量測**，Stage 1 結束後的 `post_stage1.sh` 200 例會先量到）；600 例子集每次估約 6–8 min（待量測）。Qwen 反抽：Spark `merlin-qwen-fp8`（:8003）8 並行，600 例約 15–30 min／次；全部 val 約 3–4.5 h／次。
- GPU：只用 **A6000**（48 GB；實測 Stage 1 占 24.7 GB，同時跑兩個 7B 訓練放不下，所以依序跑）。Spark 不訓練（共享記憶體，曾 OOM）；CT-rate 的 3080 不用。**A6000 `:8000` 的 vLLM 與 Spark 的 `std14-qwen` 容器（:8005）是 CG 專案的，不可停、不可動**；train.py / infer.py 開跑前會檢查 GPU 空閒記憶體，CG 的 vLLM 若重新占用 A6000 就會拒絕執行。
- 建議順序：Stage 1 結束 → U-oof → U-gt →（看差異再決定）T-oof → T-gt。若想先快速看方向，可四組都先跑 1 epoch（共約 9.2 h），再挑有差異的補第 2 個 epoch。

### 評估（每組相同，全部在 val、按 14 類拆開）

| 指標 | 工具 |
|---|---|
| 疾病有無 sens / spec / F1（owl 與 B 兩種參考定義） | `evaluate.py`（開發用 600 例子集 `--ids method1/val_qwen_subset_ids.txt`，最終用全部 val） |
| 關鍵錯誤率（原報告陽性 → 生成寫陰性，**最重要**）、漏寫率 | 同上 |
| 屬性正確率 | 同上 |
| BLEU-1～4、ROUGE-L、BERTScore | 同上（`--targets`、`--bertscore-model`） |
| 錯誤提示測試 | `infer.py --cue-gt … --cue-flip-pos2neg 0.5 --cue-flip-neg2pos 0.05` → `evaluate.py --cue-log …` |

- 特別看分類器較弱的三類：**脂肪肝（val AUROC .832）、骨折（.897）、膽結石（暫定 .936／目前採用保守 .881）**。train 上 OOF 提示落在標籤另一側的比例（＝ T-oof 訓練時看到的錯誤提示率）：動脈粥狀硬化 17.1%、腹水 11.4%、脂肪肝 11.1%、膽管擴張 9.4%、骨折 8.0%、骨質減少 6.7%、膽結石 6.5%，其餘 0.4–4.8%。
- 判讀：gt 組若在弱類別的關鍵錯誤率較低、但錯誤提示測試的「陽→陰仍一致率」也很低，代表它是照抄提示（沒有從影像學到東西）；U 組與 T 組在錯誤提示下的一致率差距，就是 embedding 實際提供的資訊量。
- 方法一與四組都在同一批 val 病人上比；Qwen 反抽本身的雜訊底約 0–1%（eval/README 第 5 節），差異小於此不要過度解讀。

### 驗證（CPU，2026-11-09，沒用 GPU）

- 開關全關時，新程式產生的 dataset item／collate batch 與修改前的程式**逐位元相同**（合成資料；有影像、只用文字、val、推論模式、四個舊開關中的左右翻轉＋屬性加權＋單元標籤，5 種設定，每種把 2–4 筆重讀 5 次，固定 seed）；train.py Stage 1、Stage 2（含左右翻轉＋屬性加權）、只用文字各跑 2 個 optimizer step（CPU、tiny 模型、固定 seed），train／val loss、adapter、LoRA 權重完全相同；infer.py 生成文字相同。另外在真實資料上，開關關閉與明確傳入關閉值的 2,000 筆 train item 相同。
- 真實資料（Mac，用分數／量測／metadata 重組 prompt、目標留空，不含報告文字）：20,364 個 prompt 分類器那一行解析重組 0 個不同；gt_dropout p=0.3 對全部 train 15,309 例：212,477 個有標籤的格子丟掉 30.18%（各類 29.7–30.8%），顯示出來的 148,359 格 100% 在標籤那一側，unclear 一律沒顯示；量化 dropout p=0.3 實際 30.1%；val 列帶 gt 提示被拒絕；推論模式 val prompt 與不帶 gt 時完全相同；cue_gt.json 內 val 病例 0 例。
- 翻轉函式：同 seed 兩次結果相同、換 seed 不同；機率設 1 時所有有分數的格子都被翻、方向與門檻一致。
- smoke_test.py 全部通過（含新增的 train.py gt_dropout〔有影像與只用文字〕與 infer.py 翻轉端到端）。evaluate.py 的錯誤提示統計用合成反抽結果比對手算一致；實際 Qwen 反抽沒跑。

## 5. 建議執行順序（待確認後）

1. ✅ A6000 全量打包特徵完成；B 的 targets 完成、A 確認 OOF 無洩漏後，在 Mac 執行 `bash build_data.sh`（把小檔輸入傳到 A6000，在 A6000 上產生 `m4/data/{train,val}.jsonl`，不放 Mac）。
2. ✅ Lingshu-7B 已從 Spark 直接 rsync 到 A6000 `~/Desktop/Tony/models/Lingshu-7B`（16 GB，A6000 從 Spark 拉，不經 Mac）；`prepare_llm.py` 已產生 `~/Desktop/Tony/models/Lingshu-7B-llm`（339 個 LLM tensor，390 個 visual 丟掉；symlink，不佔空間）。訓練環境：vllm-env + `m4_pylib`（見 4.1），執行時要設 `LD_LIBRARY_PATH=~/Desktop/Tony/tools/cuda-compat-13-4/usr/local/cuda-13.4/compat`、`PYTHONPATH=~/Desktop/Tony/tools/m4_pylib`。
3. （可省）Qwen3-1.7B 小跑 1/3 epoch 檢查流程與 loss。
4. 主模型（Lingshu-7B）Stage 1 → Stage 2（2 epoch）→ val 推論 → 評估。
5. 只用文字的消融（同 epoch 數）。
6. 看結果再決定要不要做 Lingshu-I-8B 比較（多約 7.5–10 h）。

## 6. 需要使用者確認的事

1. **開始正式訓練**：A6000 上主模型 + 只用文字消融，2 epoch 版約 10.5–11 h（3 epoch 版約 14–15 h）。要不要加 Lingshu-I-8B 比較（再加約 6 h，並需下載約 16 GB 到 A6000 `~/Desktop/Tony/models/`）？
2. **環境**：目前用 vllm-env + `m4_pylib`（peft/accelerate 以 `--target` 裝在 Tony 資料夾）就能訓練，可以照用嗎？還是要另建乾淨的 venv？
3. **epoch 數**：Stage 2 先跑 2 epoch（省 2 h）還是照 CG 的 3 epoch？
4. **prompt 內容**：目前只放期相、機型、14 類機率、主動脈徑與脾臟大小。要不要加年齡、性別（OWL 分類器本身沒用過）？要不要加入方法一的 neg/verify/pos 判定文字？
5. **完全沒提到 14 類的報告**：目前訓練目標為空時仍保留（模型學會「不寫」），要不要排除？
6. **評估子集**：Qwen 反抽用 600 例子集（與方法一同一批）還是全部 val（每個系統約 3–4.5 h）？
7. **特徵範圍**：區域單元目前放 9 種（含沒有採用為正式最佳的 r2 版腸道、骨盆、肝門 crop）；骨折與膽結石的正式最佳是 encoder 微調版本，那些微調特徵沒有放進來（只用凍結 encoder 的單元特徵，分數則已經含微調模型的 OOF）。這樣可以嗎？
8. **提示來源對照（4B 節）**：是否加跑 T-gt、U-gt（新增約 9.2 h，A6000 依序）；gt 標籤 → 數字用 `sample`（預設）還是 `hard`；dropout 0.3 照 DCP-PD 還是另掃；U-gt 是否共用 OOF 版 Stage 1；錯誤提示測試機率 (0.5, 0.05) 與開發子集是否可以。

## 7. 檔案

| 檔案 | 用途 |
|---|---|
| `m4_common.py` | 14 類、prompt 英文名、tag 格式、`BACKBONES` 設定、輸出解析 |
| `pack_features.py` | A6000 特徵打包（只讀 mw_exp，只處理 train+val） |
| `build_jsonl.py` | 組 prompt 與 tag 目標句 |
| `m4_model.py` | TypeEmbedding、Perceiver resampler、projector、把影像 token 填進 LLM |
| `m4_data.py` | Dataset / collate（手寫 ChatML、pin_memory=False） |
| `train.py` | Stage 1 / Stage 2 / text-only；backbone loader；存權重與 manifest；GPU 占用檢查 |
| `infer.py` | 產生 {case_id, findings, raw}；`--cue-gt --cue-flip-*` 錯誤提示測試（只限 val） |
| `build_cue_gt.py` | `--cue-source gt_dropout` 與錯誤提示測試用的 `cue_gt.json`（只有 train 標籤、t_f1 門檻、train OOF 分數池；含病例 id，不進 git） |
| `prepare_llm.py` | （可選）Lingshu-7B 的 LLM-only 資料夾（symlink，不複製權重） |
| `smoke_test.py` | CPU tiny 隨機模型 smoke test（含四個實驗開關與提示來源開關、錯誤提示翻轉） |
| `build_unit_labels.py` | `--aux-unit-loss` 用的單元層級標籤與對應率統計 |
| `build_data.sh` | Mac 端：輸入傳到 A6000、在 A6000 跑 build_jsonl.py |
| `start_stage1.sh` | A6000 端：Stage 1 啟動腳本（nohup，log 在 `m4/logs/`；有前置檢查，使用者同意後才執行） |
| `gpu_smoke.py` | 真 backbone + 合成資料的 GPU 短測（顯存、每 step 秒數、每 epoch 估計） |

## 8. 改進實驗（agent F，2026-11-11 起；A6000 依序；test 未讀取）

程式在 `improve/`（A6000 複本 `m4/code_f/`，與 agent E 使用的 `code_v2/` 分開）；資料、產物在 A6000 `m4/improve/`、`m4/gen_f/`、`m4/eval/results_f/`；Qwen 快取 `m4/eval/qwen_extract_cache_F.jsonl`（不和 E 共寫）。

| 實驗 | 做法 | 開關／程式 |
|---|---|---|
| 改進項 4：偏敏感度門檻（只推論） | U-gt、T-gt 原權重，val600 推論時把分類器那一行改寫：原始分數 ≥ 新門檻 → 顯示成 ≥ t_f1 的數值（模型學到的陽性那一側），< 新門檻 → < t_f1；單調分段線性。新門檻只用 train OOF 訂：s80（train 敏感度 ≥ .80）、s90（≥ .90）；方法一 1A 的陽性門檻 t_neg 就是 s95 | `improve/make_variants.py thresholds / remap` |
| 改進項 3：U-nocue | 完全拿掉分類器那一行（量測、HU、年齡性別照給），接 lingshu7b_s1 Stage 1 adapter 跑 Stage 2 2 epoch | `make_variants.py nocue` → `train.py --data improve/data_nocue` |
| 改進項 2：逐單元結構化提示 | 凍結 OWL 單元 token 訓練輕量 probe：每顆腎（organ3 腎 5 token＋腎竇 crop 40 token）的腎水腫、T8–L5 每節椎體（vert15 6＋bone3 6 token）的骨折；train 5-fold OOF（腎水腫用 region_units/hydronephrosis 的 fold、骨折用 sitewise 外層 fold），val 用全 train refit；logistic 與小 MLP 依 train OOF AUROC 擇一。prompt 在分類器那一行後加「Unit-level probe scores」一行；gt 版訓練時該行由報告側別／節段標籤重建（從 train OOF 分數池抽、每部分 dropout 0.3），其餘同 T-gt | `improve/unit_probe.py`、`improve/build_unit_data.py`、`train.py --unit-cue-gt unit_cue.json --unit-cue-dropout 0.3`（預設關閉；關閉時與原本逐位元相同） |

- 佇列：`improve/gpu_queue.sh`（讀 `m4/improve/jobs.txt`，一次一個 GPU 工作，NaN／停滯看門狗）、`improve/eval_queue.sh`（讀 `evals.txt`，自己的通道 A6000:18023 → Spark :8003，8 並行）；`improve/summarize.py` 把 metrics.json 轉成 RESULTS 表的列。
- 結果見 `queue_cue/RESULTS_interim.md`「改進實驗」段落。
