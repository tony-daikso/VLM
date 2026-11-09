# Merlin 報告生成：共用評估工具（evaluate.py）

更新：2026-11-09（agent C）

方法一（樣板句）、方法四（LLM）與只用文字的消融，都用這支程式在同一批 val 病人上評估。
本資料夾只放程式與說明；生成檔、結果、Qwen 反抽快取都放在資料夾 `VLM/data/CT/merlin/report_gen/eval/`（含報告片段，受 DUA 限制，不進 git）。

## 1. 用法

```bash
# Qwen 走 ssh tunnel（vLLM 只綁 localhost）。11-09 17:00 起用 Spark 的 merlin-qwen-fp8（:8003）
ssh -N -L 18003:127.0.0.1:8003 sparks &

python evaluate.py --gen <生成檔.jsonl> --name <系統名> \
    --targets $MERLIN_DATA/report_gen/targets/targets_val.jsonl \
    --ids $MERLIN_DATA/report_gen/method1/val_qwen_subset_ids.txt \
    --url http://127.0.0.1:18003/v1 --jobs 8
```

- `MERLIN_DATA` 預設是 Mac 的 `/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/merlin`；在其他機器上設環境變數或直接給 `--labels --meta --disease14-dir --cache --out-root`。
- 結果寫到 `$MERLIN_DATA/report_gen/eval/results/<系統名>/`：`metrics.json`、`report.md`（中文疾病名表格）、`per_case_categories.jsonl`（每例每類 [參考類別, 生成反抽類別]，沒有文字）。
- 常用選項：`--no-qwen`（只算 BLEU / ROUGE / BERTScore）、`--no-request`（只用快取，不送任何請求）、`--limit N`、`--ids 檔案`、`--bertscore-model <本機模型資料夾>`。
- **test split 鎖住**：輸入裡 split=test 的病例一律丟掉；`--split test` 要加 `--allow-test` 才會執行，只能在使用者解鎖 test 之後用。
- **Qwen 規則**：本程式只當 client，不會啟動、停止或修改任何 vLLM。並行數上限寫死為 8（預設 2）；伺服器同時有其他專案在用時請用 ≤4。11-09 下午 A6000 `:8000` 與 Spark `std14-qwen :8005`（CG 專案）已關閉，現在用 Spark `merlin-qwen-fp8`（`127.0.0.1:8003`，max-model-len 16384）。
- 程式裡沒有任何密碼或 token。
- 參考原句：B 的 targets 列有 `target_text`（target_lines 的 clean 一行一句）時直接當整段參考文字；每類句子用 `diseases.<d>.sentence_clean`。B 的完整 targets 在 A6000 `~/Desktop/Tony/VLM/OWL/report_gen/targets/`，評估時把 `targets_val.jsonl` 拿來用（含醫師原句，只能放 data/ 或 A6000）。

## 2. 輸入格式

| 檔案 | 格式 |
|---|---|
| 生成檔（`--gen`） | 每行一例：`{"case_id", "findings": {disease: sentence}}`，或 `{"case_id", "text": "..."}`（也接受 `report` / `draft` / `generated` 欄位）。findings 會依 Merlin 器官順序組成文字，同一句只出現一次（例如一句同時講腹水與游離氣體）。 |
| 參考標籤（`--labels`） | `disease14/labels_disease14.jsonl`（預設），每例 14 類的 category + detail + quotes |
| 參考原句（`--targets`，可省略） | agent B 的 `targets_{split}.jsonl`：每類 `{status, sentence_raw, sentence_clean, attributes, source}`；放在頂層或 `diseases` / `findings` / `per_disease` / `targets` 底下都可以。若有 `target_text` / `target` 字串欄位就直接當整段參考文字。 |
| metadata（`--meta`） | `disease14/metadata.csv`，用來組 Qwen user 訊息的 patient line（和產生標籤時完全一樣） |

## 3. Qwen 反抽（與標籤同一套）

- 直接 import `disease14/run_disease14.py`（`PROMPT` = `prompt.md`、`schema()`、`ask()`、`patient_line()`），沒有複製或改寫 prompt。結果檔記錄兩個檔案的 md5（prompt.md `01620827…`、run_disease14.py `51a44dc3…`，與 A6000 上產生標籤的版本相同）。
- 模型 Qwen3.8-27B-FP8（vLLM served name `qwen`）、temperature 0、thinking 關閉、guided JSON、max_tokens 6000；user 訊息 = patient line + `# Report` + 生成文字。
- **快取**：`$MERLIN_DATA/report_gen/eval/cache/qwen_extract_cache.jsonl`，key = sha256(prompt md5, 模型標記, max_tokens, 完整 user 訊息)。同一段文字不會重送；失敗的請求不算快取命中，重跑會重送。多個程式同時寫同一個快取也能讀（壞行略過）。

## 4. 指標定義

**疾病有無**（每類、micro 合計、macro 平均）：生成文字反抽的 category 和參考標籤比較。
- 生成「陽性」：嚴格 = yes / probably_yes；寬鬆 = 再加 unclear。
- 參考標籤兩種定義都算：
  - **owl**（主要）：陽性 yes / probably_yes；陰性 no / probably_no / not_mentioned；unclear 排除。這是 OWL 分類器訓練用的二元化（mw_exp/src/make_labels.py）。
  - **B**：陰性只算 no / probably_no（明確寫沒有），not_mentioned 與 unclear 排除。與 CG eval_phase1 相同。
- 報敏感度、特異度、PPV、F1。

**關鍵錯誤率**（最重要）：原報告陽性（yes / probably_yes），生成反抽為 no / probably_no（明寫陰性）。分母 = 原報告陽性數。另列「漏寫」：原報告陽性、生成 not_mentioned。

**屬性正確率**：只看原報告與生成都陽性的病例；比較兩邊 Qwen 抽出的 detail 欄位。分母 = 參考有寫該屬性的病例，生成沒寫算錯；另列「生成有寫時的正確率」與「參考沒寫但生成寫了」（可能是編造）。
| 疾病 | 屬性 | 判定 |
|---|---|---|
| 腎水腫 | 側別；程度 | 完全一致；程度另報差一級內 |
| 脂肪肝 | 分級 | 一致／差一級內 |
| 脾腫大 | 長度 | ±1 cm |
| 腹主動脈瘤 | 直徑；是否修補 | ±0.3 cm；一致 |
| 膽管擴張 | 部位；總膽管徑 | 一致；±2 mm |
| 腹水 | 量（trace < small < moderate < large）；出血性 | 一致／差一級內；一致 |
| 腸阻塞 | SBO / LBO | 一致 |
| 闌尾炎 | 穿孔；糞石 | 一致 |
| 腹腔游離氣體 | 術後 | 一致 |
| 膽結石 | 鈣化 | 一致 |
| 動脈粥狀硬化 | 程度；主髂動脈 | 一致／差一級內；一致 |
| 骨折 | 部位集合；椎體壓迫性；急性/陳舊；**椎體節段** | 集合完全一致（另報有交集）；一致；一致；節段集合完全一致 |

椎體節段不在 disease14 的 detail 裡，改用規則從骨折句抽（C/T/L/S + 數字，範圍如 T11-L1、L2-4 會展開）：參考用 B 的骨折原句（沒有時用標籤 quotes），生成用 findings 的骨折句（整段文字輸入時用反抽的 quotes）。

**風格相似度**（需要 `--targets`）：生成文字 vs 同一例的醫師原句（B 的 sentence_clean 依器官順序組成）。
- BLEU-1～4：corpus BLEU（Papineni），單一參考，小寫、英數字與標點切詞，自己實作（不需要 sacrebleu）。
- ROUGE-L：LCS 的 F1，逐例平均。
- BERTScore：自己實作 Zhang et al. 2020 的貪婪 cosine 配對，**等同 bert_score 套件英文預設**：`roberta-large` 第 17 層、無 idf、無 baseline rescale；去掉 CLS/SEP，最長 510 token。模型放在 A6000 `~/Desktop/Tony/models/roberta-large`（1.4 GB，2026-11-09 下載；Mac 沒放）。CPU 即可（`--bertscore-threads 4`），36 對約 1–2 分鐘。載入時 transformers 會提示 pooler 權重「newly initialized」，BERTScore 不用 pooler，無影響。
- findings 格式另報每類句子層級的 BLEU-4 / ROUGE-L。

## 5. Sanity check（2026-11-09，只用 val）

**A. 參考報告本身當「生成」**：36 例 val（14 類各取 2 例陽性 + 8 例隨機，seed 固定；id 清單在資料夾 `eval/sanity_ids.txt`），送 A6000 `:8000`（當時還在），3 並行。
| 項目 | 結果 |
|---|---|
| 反抽成功 | 36 / 36（全部 finish=stop） |
| 6 類別完全一致（例×類 = 504） | **501 / 504 = 99.4%** |
| 二元一致（owl 定義，可比較的 500 格） | **499 / 500 = 99.8%** |
| 疾病有無（owl 定義，嚴格） | micro 敏感度 .990、特異度 1.000、F1 .995；13 類 F1 = 1.000，腹水 F1 .968（16 例陽性中 1 例 probably_yes → probably_no） |
| 關鍵錯誤率 | 1 / 103 = 1.0%（就是上面那 1 例腹水）；漏寫 0 |
| 屬性 | 20 個有參考值的屬性全部 100% 一致（膽結石鈣化在這 36 例中參考都沒寫）；「參考沒寫但生成寫了」1 例（骨折急性/陳舊） |
| 其餘 2 個不一致 | 骨折、游離氣體各 1 例 not_mentioned ↔ no（二元結果相同） |

**A2. 同一批 36 例改送 Spark `merlin-qwen-fp8`（:8003，4 並行）**：Spark 與 A6000 的 Qwen3.8-27B-FP8 模型檔 md5 完全相同（config.json `fa7068d0…`、safetensors index `238a4bb9…`、crc32.txt `5d556fa2…`；Spark 容器掛載的 `~/models` 是 `~/Desktop/Tony/models` 的 symlink），產生 labels_disease14 時三台用的都是這份權重。
| 項目 | 結果 |
|---|---|
| 6 類別完全一致 | 500 / 504 = 99.2%（4 格都是 not_mentioned ↔ no / unclear） |
| 二元一致（owl 定義） | 499 / 499 = 100% |
| 疾病有無（owl 與 B 定義，嚴格） | micro 敏感度、特異度、F1 都是 1.000 |
| 關鍵錯誤率 | 0 / 103；屬性全部 100% |
| A6000 與 Spark 兩次反抽彼此一致 | 499 / 504 = 99.0% |

解讀：同一段文字重抽的雜訊約 0.6–1%（類別）、0–0.2%（二元）；關鍵錯誤率的雜訊底約 0–1%。比較方法時，差異小於這個量級的不要過度解讀。端點換成 Spark 不影響結果；快取 key 含模型標記，Spark 這次用 `--qwen-model-tag "Qwen3.8-27B-FP8@spark-merlin-qwen-fp8"` 另外存（正式評估可以直接用預設標記，因為權重相同）。

**B. 文字指標**（A6000 CPU，36 例；用標籤 quotes 當暫時的參考原句，B 的 targets 完成後再換）：
| 生成 | BLEU-4 | ROUGE-L | BERTScore F |
|---|---|---|---|
| 與參考完全相同 | 1.000 | 1.000 | 1.000 |
| 整份原報告（遠長於參考：平均 494 vs 72 字） | 0.120 | 0.239 | 0.850 |

**對 CG 的負載（A6000 :8000 還在時）**：送請求前 60 秒 CG 的生成吞吐 374 token/s（32 個請求在跑）；sanity 期間 382 token/s（35 個在跑），CG 自己分到約 349 token/s，約慢 7%。每次反抽約 3.1k prompt + 1.2k completion token，CG 滿載時每份約 85–115 秒。Spark 上 4 並行時每份約 50–70 秒（牆鐘，與其他 agent 共用）。完整 val 5,055 例一個系統估計要數小時，建議 Qwen 指標用固定子集（例如方法一的 600 例，`--ids`），文字指標用全部 val。

## 6. 檔案

| 檔案 | 用途 |
|---|---|
| `evaluate.py` | 評估主程式 |
| `sanity_make_inputs.py` | 產生 sanity A 的輸入（參考報告當生成；輸出含報告原文，只能寫到 scratch 或資料夾 data/） |
