# 膽泥（gallbladder sludge）標籤規則

- 決定：使用者，2026-11-06（LABEL_DECISIONS_PENDING.md 第 11 項）：「另外立一類『膽泥』」。**膽結石的定義不變**（yes／probably_yes＝陽性；只有膽泥的病例在膽結石仍是陰性）。
- 規則在 2026-11-06 鎖定，**在任何 test 使用之前**。train／val／test 一律用同一支程式、同一組規則。
- 程式：`scripts/sl_labels.py`（sha256 `e1805e26211ba2f80d58e3e29db1274b0419dd2dccd3d067fb6cb03e5824eb0b`）。
- 產出：`labels_trval_sludge.csv`（sha256 `d940d4b19fd072c8b8fc74dc621216e282d139b354eb1834996d2a81bdbfcdf3`），每例只有旗標與標籤，不含報告文字。
- 輸入只有兩種：① 報告文字（`merlin_disease14/reports.json`）② 現有 Qwen 標籤 `labels_disease14_all.jsonl` 的 `gallstones`（category＋quotes）與 `surgically_absent_gallbladder`。**不使用任何模型分數**。
- split 安全（讀取器與 steatosis_split 的 `hs_split_labels.py` 相同）：先讀 `split.csv`，只保留 `--splits` 指定的 study_id；jsonl 先用 regex 取出 study_id、確認在允許集合內才 json 解碼；reports.json 用依 key 過濾的掃描器，其他 key 的值不解碼也不保存。`--splits test` 沒有加 `--allow-test` 時程式直接拒絕。

## 1. 文字處理
1. 報告切句（句點／分號／換行）。報告句與 gallstones quotes 都判讀，兩者取聯集（quotes 不要求膽囊脈絡字）。
2. 命中詞：`sludge*`（含 biliary sludge、layering sludge、**tumefactive sludge、sludge ball**）、`microlithiasis`。
3. 句子必須有膽囊／膽道脈絡（gallbladder、gall bladder、GB、cholecyst、cholelith、gallstone，或 sludge 本身）；只講腎／輸尿管／膀胱（urinary bladder）的句子排除。
4. 依序排除（不算陽性，只記旗標）：
   - **病史句**（history of、indication、clinical history、以 history／clinical／indication／comparison／technique 開頭）→ `sl_hist`
   - **否定**：命中詞前 120 字元、同一子句（以 ; : but however which with 及句點切開）內有 no／not／without／negative for／free of／resolved／absence of／no evidence of／nor／rather than；或命中詞後 60 字元內有 has resolved／no longer seen／is not seen／is absent → `sl_neg`。例：「No radiopaque gallstones or sludge」「without evidence for calculi or sludge」都是否定。「not (significantly) changed／increased」不算否定詞。若否定詞與命中詞之間隔著逗號、且逗號後沒有 or／and／nor，也不算否定。
   - **假設語氣**：命中詞前 60 字元同一子句有 evaluate for／evaluation for／assess for／to exclude／rule out／r/o／if clinically／could be obtained／recommend → `sl_hypo`（例：「Evaluation for dependent sludge … is compromised」）
   - **只在膽管**：句中有膽管字（common bile duct、CBD、cystic duct、hepatic duct、bile duct、biliary tree、intrahepatic、choledoch、ampulla）而**沒有**膽囊字 → `sl_duct_only`（膽管內膽泥不算膽囊膽泥）
5. 措辭不確定（possible、likely、may represent、suggest、versus、or、and/or、if any…）**仍算陽性**，另記 `sl_unc`（與原 probably_yes 算陽性的慣例一致）。「sludge versus vicarious excretion of contrast」也算陽性（不確定），`kw_vicarious` 另記；單獨的 vicarious excretion、debris、hyperdense bile 不算膽泥（只記 `kw_vicarious`／`kw_debris`）。

## 2. 標籤
| 欄位 | 定義 |
|---|---|
| `y_sludge`（**膽泥**，主要定義） | 報告或 quotes 有未否定的膽囊膽泥 = 1，否則 0（有沒有結石都算）；報告與 quotes 都空白 → NaN |
| `sl_group` | none／sludge_only（膽泥且膽結石標籤陰性）／sludge_stones（膽泥且膽結石陽性）／sludge_gs_unclear（膽泥但膽結石標籤 unclear） |
| `y_sludge_only`（輔助目標，用於 3(c) 抑制實驗） | sludge_only = 1；其他膽結石已知的病例 = 0；膽結石 unclear → NaN |
| `y3`（3 類） | 0 兩者皆無／1 只有膽泥／2 結石（不論有無膽泥）；膽結石 unclear → NaN |
| `y_gs` | 原膽結石標籤（**不變**） |
| `y_gs_excl_sludgeonly`（次要檢視） | 同 `y_gs`，但 sludge_only 病例設 NaN（從膽結石陰性中移除） |

## 3. 數量
| | train | val |
|---|---|---|
| 全部 | 15,309 | 5,055 |
| **膽泥陽性（y_sludge）／盛行率** | **221（1.44%）** | **73（1.44%）** |
| 　只有膽泥（膽結石陰性） | 112（no 10、probably_no 102） | 41（no 4、probably_no 37） |
| 　膽泥＋結石（膽結石陽性） | 49（yes 37、probably_yes 12） | 14（yes 13、probably_yes 1） |
| 　膽泥、膽結石 unclear | 60 | 18 |
| 膽泥陽性中措辭不確定（sl_unc） | 131（59%） | 50（68%） |
| 膽泥陽性中同時提到 vicarious excretion | 30 | 13 |
| 膽泥陽性中 tumefactive／sludge ball | 2／1 | 1／1 |
| 只被否定（sl_neg，全為陰性） | 7 | 2 |
| 只在膽管（sl_duct_only） | 7 | 0 |
| 膽結石次要檢視（移除只有膽泥）陽性／總數 | 1,343／15,001 | 465／4,948 |

與先前子群分析（analysis_gallstones/subgroups，train 164／val 59「僅膽泥」）的差異：舊定義把 vicarious excretion、tumefactive 單獨也算膽泥，且以「有無結石字」而不是膽結石標籤區分；本定義只認 sludge／microlithiasis 字並以官方膽結石標籤分組，所以只有膽泥的例數較少（112／41）。

## 4. 稽核（train，只看旗標與命中的短片語）
- 兩輪：第 1 輪隨機 45 例＋全部否定／假設／膽管例；修正後第 2 輪再抽 40 例＋全部特殊例。
- 第 1 輪發現並修正：① 「…is not significantly changed and may reflect sludge」被當否定 → 「not (significantly) changed」不算否定詞；② 「likely sludge within the biliary tree」「Layering hyperdensity within the CBD may represent biliary sludge」被算成膽囊膽泥 → biliary tree／intrahepatic 加入膽管字，且 biliary sludge 在膽管句中也算膽管；③ 「…8-mm gallstone a small amount of sludge」被假設語氣誤排除 → 假設語氣只看前 60 字元、移除 consider／follow-up。
- 第 2 輪：隨機陽性 25／25 合理（其中約 6 例為「sludge vs vicarious excretion」或「sludge or stones」類不確定描述，依規則算陽性）；否定 7／7 正確；膽管 7／7 正確；假設語氣 1／1 正確。
- 邊界例（依規則算陽性，記在旗標中）：「minimal if any sludge」、超音波看到膽泥但 CT 正常（「recent ultrasound noted layering sludge」）。
