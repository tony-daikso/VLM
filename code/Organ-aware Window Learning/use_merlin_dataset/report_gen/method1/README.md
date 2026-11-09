# report_gen/method1：方法一（分類結果＋固定樣板句）baseline

資料與結果都在 data 端 `data/CT/merlin/report_gen/method1/`（路徑可用環境變數 `MERLIN_DATA` 改）；本資料夾只有程式。

| 步驟 | 程式 | 說明 |
|---|---|---|
| 門檻 | `m1_thresholds.py` | 只用 train OOF 機率訂門檻：A＝雙門檻（敏感度 ≥ .95 的陰性門檻、特異度 ≥ .95 的陽性門檻，中間寫陽性句＋[please verify]）；B＝單一 F1 最佳門檻。門檻用定義 A 標籤（not_mentioned 算陰性） |
| 句庫 | `m1_extract_targets.py`（A6000）→ `m1_bank.py <bank_src_train.jsonl>` | 只用 train 的醫師原句（agent B 的 targets；不給參數時退回 labels_disease14 的 quotes），依器官分組與陽陰性組合選最常見寫法；陰性幾乎不寫的疾病（骨質減少、動脈粥狀硬化）陰性時不寫句子 |
| 組裝 | `m1_assemble.py [val]` | 依 val 分數＋門檻＋句庫組成草稿；AAA 直徑與脾臟長度用分割量測值（在 train 上線性校正）。test 被鎖住 |
| 一致性 | `m1_consistency.py <results dir> <A|B>` | Qwen 反抽結果是否等於樣板想寫的陽/陰性 |
| 評估 | `m1_eval_decisions.py` | 組裝決策層級的 sens/spec/F1、關鍵錯誤（原陽性→寫陰性）、量測屬性；Qwen 反抽與 BLEU/ROUGE/BERTScore 用 `../eval/evaluate.py` |

執行順序：A6000 上 `python3 m1_extract_targets.py bank_src_train.jsonl`，複製到 Mac 暫存後 `python m1_thresholds.py && python m1_bank.py <暫存>/bank_src_train.jsonl && python m1_assemble.py val && python m1_eval_decisions.py`，用完刪掉暫存（含醫師原句）。
