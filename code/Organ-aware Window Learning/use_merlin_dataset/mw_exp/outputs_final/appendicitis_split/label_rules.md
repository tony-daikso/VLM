# 闌尾炎拆分：標籤規則（2026-11-05 鎖定，test 使用前）

依據：`LABEL_DECISIONS_PENDING.md` 第 8 項（使用者決定把闌尾炎拆成兩個 finding）。
程式：`scripts/ap_split_labels.py`。規則**只用報告文字＋既有 Qwen 闌尾炎標籤**，完全不看模型分數。val／test 用同一套程式與規則。

## 1. 兩個 finding 與遮罩（主要定義）
| 組別 `ap_group` | 條件 | `y_ap_def`（確定的急性闌尾炎） | `y_ap_unc`（可疑／治療中／追蹤／術後） | `y_ap_soft`（建議軟標籤，僅供日後實驗） |
|---|---|---|---|---|
| definite | 原分類 `yes`，且**沒有**任何治療／追蹤／術後旗標 | 1 | 遮罩（NaN，不訓練、不評估） | 1 |
| uncertain_followup | 原分類 `probably_yes`；或 `yes` 但有治療／追蹤／術後旗標 | 遮罩（NaN） | 1 | 0.5 |
| neg | 原分類 `no`／`probably_no`／`not_mentioned`（與原本陰性相同） | 0 | 0 | 0 |
| unclear | 原分類 `unclear` | NaN | NaN | NaN |

`y_ap` 是原合併標籤，只保留作參考。

## 2. 文字旗標（只在「闌尾／右下腹」語境的句子中比對）
- 語境句：含 append、periappend、cecum／cecal、ileocec、right lower quadrant／RLQ、terminal ileum、right abdomen／pelvis／paracolic 的句子，以及同時提到 drain／abscess／collection 與 right／append／cec／pelv 的句子。報告全文和 Qwen 擷取的引文（quotes）都會掃描，任一處命中即算。
- `t_postop`（術後）：post-operative、post-surgical、status post／s/p／after／following／recent／prior … appendectomy、appendectomy bed／site、post-appendectomy、surgical bed、appendiceal stump、stump appendicitis、appendix removed。有否定詞（no／not／without…）時不算。
- `t_drain`（已放引流）：drainage catheter／tube、pigtail、drain、status post／after／interval … (percutaneous) drainage、percutaneously drained。前面若是 amenable／consider／recommend 等建議語氣，或被否定，則不算。
- `t_treat`（藥物治療）：antibiotic、abx、non-operative、conservative management、treated with／for。被否定時不算。Merlin 報告中實際命中 0 例。
- `t_followup`（追蹤、演變）：resolving／resolution（排除前面是 following／after／until 的建議句）、improving、interval decrease／improvement／resolution／regression／evolution／organization／increase／worsening／progression／enlargement、decreased in size、smaller than、redemonstrated、again seen／noted、persistent、residual（排除 residual contrast）、evolving、healing、worsening、progression、has／have organized、previously demonstrated／identified／reported／diagnosed／measuring、unchanged、stable、known／recent／prior／treated appendicitis、history of appendicitis、follow-up of appendicitis。這組**不做否定檢查**，因為「no residual abscess」也代表之前有過一次發作。
- `t_abscess_postappx`：同一句同時有 abscess／collection／phlegmon 與 appendectomy（闌尾切除後膿瘍）。
- `treat_followup_any` = 上述任一旗標。

## 3. 稽核（train，人工檢視 67 例＋所有 definite 的比較用語）
- 檢視了 train 中所有被旗標標為 uncertain 的 `yes`（第一版 43 例）與隨機 30 例 definite 的闌尾語境句。
- 第一版發現 6 處錯誤，已全部修正：
  - 「residual contrast」被當成追蹤（2 例）。
  - 「Following resolution of acute episode, recommend colonoscopy」這種建議句被當成追蹤（1 例）。
  - 「previously described」其實指同一份報告前文（1 例）。現在只接受 previously demonstrated／identified／reported／diagnosed／measuring。
  - 「recurrent acute appendicitis」是一次新的急性發作，改回 definite（2 例）。
  - 「interval organization of previously seen inflammation」漏判（1 例），已補上規則。
- 另外把所有 302 例 train definite 中含 prior／compar／since／again／drain／antibiotic／interval／appendectomy 的句子全部列出檢查，沒有發現漏判的追蹤或術後病例。命中多為「no drainable collection」或「preliminary results were reviewed」之類的句子。
- 修正後，在 38 例 uncertain 的 `yes` 中仍有 2 例存疑：
  - AC4213f12：只命中 persistent。
  - AC4215537：post-operative 是指婦科手術，不是闌尾切除。
  - 估計誤分率約 2/67（3%）。

## 4. 各組數量
| split | definite | uncertain_followup（probably_yes／yes＋旗標） | neg | unclear |
|---|---|---|---|---|
| train | 302 | 105（67／38） | 14,790 | 112 |
| val | 92 | 26（18／8） | 4,902 | 35 |
| test | 未讀取（之後用 `--splits test --allow-test` 產生） | | | |

uncertain 組在 val 只有 26 例陽性，所以該 finding 的 val AUROC 95% CI 寬度約 ±0.08–0.10，只能當描述性結果。

## 5. 防止讀到 test
- 先讀 `split.csv`；labels jsonl 先用 regex 取出 study_id，確認屬於允許的 split 後才做 json 解析。
- `reports.json` 用依 key 過濾的掃描器，不屬於允許 split 的 value 直接跳過，不解碼也不儲存。
- 沒有加 `--allow-test` 時，程式會拒絕處理 test。
- 輸出的 CSV 只有旗標與標籤，沒有報告原文。
