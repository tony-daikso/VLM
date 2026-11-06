# 實驗 O（區域 crop 單元第二輪）：腹水（ascites_r2）

產生時間 2026-11-06 10:38；結果來源 `ascites_r2/stack/results.json`（no_stack.py）與 probe log。
規則：只用 train/val（標籤 csv 只含 train/val，先依 split 過濾）；模型選擇只看 train 5-fold OOF；val 只報告；test 未讀取。年齡／性別不作為輸入。
管線：3 mm prep＋TS total 遮罩 → crop（A6000＋CT-rate CPU）→ 凍結 TS --fast encoder（Dataset297，5 窗，stage 1–3，mean＋max）→ E3_late 式 late head（U＝只有 crop 區域、U_org＝再加主 TS 器官單元）→ 分數層堆疊 [logit(REF)｜單元頭]，LR／GBM 依 train OOF 選。Δ＝對 REF 的配對 bootstrap 95% CI。

**單元**：第二輪 3 個 64³ crop：c0 右結腸旁溝、c1 左結腸旁溝、c2 小網膜囊；區域（同實驗 M）：r0 30 mm 球、r1 液體候選（−20～40 HU，未標註）、r2 器官殼層 12 mm、r3 50 mm 球。

**結論**：REF＝實驗 M 腹水最佳（REF＋U_org，val .916）。結論：val .917，Δ +.001～+.002（CI 下界約 0），增益極小，**不列為新最佳**。

## 結果（粗體＝依 train OOF 選出）

train 15085 例／陽性 3015；val 4978 例／陽性 949

| 模型 | 堆疊器 | train OOF | Δ OOF [CI] | val [CI] | Δ val [CI] | 年齡配對 train／val | Δ 年齡配對 val [CI] | Spearman（陰性） | 敏感度@90%特異度 val |
|---|---|---|---|---|---|---|---|---|---|
| REF | - | 0.919 | – | 0.915 [0.905, 0.925] | – | 0.920／0.915 | – | -0.003 | 75.8% |
| U 單獨 | probe | 0.886 | -0.033 [-0.038, -0.028] | 0.883 [0.870, 0.896] | -0.032 [-0.041, -0.024] | 0.888／0.884 | -0.030 [-0.039, -0.022] | +0.120 | 70.5% |
| U_org 單獨 | probe | 0.894 | -0.025 [-0.029, -0.020] | 0.891 [0.878, 0.903] | -0.024 [-0.033, -0.016] | 0.896／0.892 | -0.023 [-0.031, -0.015] | +0.133 | 72.4% |
| REF＋U | LR_C0.1 | 0.920 | +0.001 [+0.000, +0.002] | 0.917 [0.906, 0.926] | +0.001 [-0.000, +0.003] | 0.921／0.916 | +0.001 [-0.000, +0.003] | +0.018 | 75.8% |
| REF＋U_org | LR_C0.1 | 0.920 | +0.001 [+0.000, +0.002] | 0.917 [0.906, 0.927] | +0.002 [+0.000, +0.003] | 0.921／0.916 | +0.001 [-0.000, +0.003] | +0.021 | 75.6% |
| **REF＋全部單元頭** | LR_C1 | 0.920 | +0.001 [+0.000, +0.002] | 0.917 [0.906, 0.926] | +0.001 [+0.000, +0.003] | 0.921／0.916 | +0.001 [-0.000, +0.003] | +0.018 | 76.0% |

**probe（單元頭本身）**

- asc2 U: train OOF 0.8863 | val 0.8834
- asc2 U_org: train OOF 0.8943 | val 0.8913

區域出現率（crop × 區域）：
```
[[0.9992 0.9992 0.9992 0.9992]
 [0.9992 0.9992 0.9992 0.9992]
 [0.9987 0.9987 0.9987 0.9987]]
asc2 U: train OOF 0.8863 | val 0.8834
asc2 U_org: train OOF 0.8943 | val 0.8913
RU_PROBE_DONE

```

定位方式（train＋val，A6000＋CT-rate）：asc_colon|desc_colon|lessersac 5801、asc_colon_flipR|desc_colon_flipR|lessersac_flipR 16、asc_colon|desc_colon|stomach_post 16、asc_colon|desc_colon|absent 4、absent|absent|absent 1
