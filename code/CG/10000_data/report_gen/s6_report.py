#!/usr/bin/env python3
"""Write report_gen/eval_phase1.md (繁體中文) from the S1-S6 outputs. Run after s6_eval.py."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import DIS, OUT, PROBS, ZH, defb_labels
from s3_thresholds import load_meas

RULE_ZH = {"standard": "標準", "rare_silent": "罕見類（中間區不寫句子）"}
STATE_ZH = {"pos": "陽性句", "verify": "陽性句＋[please verify]", "neg": "陰性句", "none": "未寫",
            "implied_neg": "未寫（已寫膽囊切除）"}


def f3(x):
    return "–" if pd.isna(x) else f"{x:.3f}"


def pct(x):
    return "–" if pd.isna(x) else f"{100 * x:.1f}%"


def main():
    sp = pd.read_csv(OUT / "split.csv")
    th = pd.read_csv(OUT / "thresholds.csv")
    guards = json.load(open(OUT / "guards.json"))
    dm = pd.read_csv(OUT / "s3_dev_metrics.csv")
    R = pd.read_csv(OUT / "eval_phase1.csv")
    A = pd.read_csv(OUT / "s6_assembly_test.csv")
    Cn = pd.read_csv(OUT / "s6_consistency.csv")
    V = pd.read_csv(OUT / "s6_coverage.csv")
    can = pd.read_csv(OUT / "canonical_sentences.csv")
    cov = pd.read_csv(OUT / "s1_coverage.csv")
    n_ex = sum(1 for l in open(OUT / "labels_drafts_test.jsonl") if json.loads(l).get("parse_ok"))
    n_sub = sum(1 for l in open(OUT / "ids_s6_subsample.txt") if l.strip())

    # measurement AUROC on dev
    dev = sp[sp.split == "dev"].EXAMID
    pr = pd.read_csv(PROBS).set_index("study_id")
    lab = defb_labels()
    m = load_meas()
    mrows = []
    for k, c, nm in [("splenomegaly", "spleen_ml", "脾臟體積"), ("abdominal_aortic_aneurysm", "aorta_eqd_mm", "腹主動脈最大徑（等效圓）")]:
        y, s, o = lab.loc[dev, k], m.reindex(dev)[c], pr.loc[dev, k]
        ok = y.notna().values & s.notna().values
        mrows.append({"疾病": ZH[k], "定量值": nm, "dev N/陽性": f"{ok.sum()}/{int(y[ok].sum())}",
                      "AUROC 定量值": f3(roc_auc_score(y[ok], s[ok])), "AUROC OWL": f3(roc_auc_score(y[ok], o[ok]))})
    mtab = pd.DataFrame(mrows)

    L = ["# CG CT 報告草稿生成 — 第一階段（S1–S6）評估", "",
         "更新：2026-11-09　｜　規劃：`VLM/code/CG/10000_data/report_gen/PLAN.md`　｜　S7（Qwen 潤稿）與第二、三階段未做。", "",
         "## 0. 重點摘要", ""]
    Rs = R[(R["mode"] == "strict") & (~R["疾病"].str.contains("報告標籤"))].set_index("disease")
    allrow = V[V.disease == "ALL"].iloc[0]
    L += [f"- 草稿用和 `disease14_qwen.csv` 完全相同的 Qwen 抽取流程（`run_cg10k.py`＋`prompt.md`，md5 相同）反抽 14 類；"
          f"test 抽樣 {n_sub} 份、成功解析 {n_ex} 份。",
          f"- **最重要的錯誤（原報告陽性、草稿寫陰性）**：反抽結果 {int(allrow.critical)}/{int(allrow.report_pos)}"
          f"（{pct(allrow.critical_rate)}）；只看組裝決策 {int(allrow.critical_assembly)}/{int(allrow.report_pos)}"
          f"（{pct(allrow.critical_rate_assembly)}）。",
          f"- 原報告明確寫陰性的項目，草稿也寫成陰性的比例（陰性句涵蓋率）：{pct(allrow.neg_coverage)}"
          f"（{int(allrow.neg_covered)}/{int(allrow.report_neg)}）。",
          "- 「請確認」區很大：脂肪肝、脾腫大約八成檢查落在中間區，這是 OWL 在 CG 上 AUROC 有限（.77/.86）＋雙門檻（敏感度/特異度 ≥ .95）的直接結果。",
          "- 組裝決策和反抽標籤的一致性見 §6.3；不一致主要來自自動測量的主動脈直徑（< 3 cm 時 Qwen 會把動脈瘤判成 probably_no）。", ""]

    L += ["## 1. 資料與切分（S2）", "",
          f"- 6,970 例 C+ 主序列（OWL `probs_best_main.csv`），每個 PID 只有一筆檢查；依 PID 隨機切分（seed 20261109）。", ""]
    L += [sp.groupby(["LOC", "split"]).size().unstack().reset_index().to_markdown(index=False), ""]
    L += ["- 標籤一律用定義 B：陽性 = yes/probably_yes；陰性只算報告明確寫沒有（no/probably_no）；not_mentioned/unclear 排除。"
          "判斷邏輯和 `code/CG/10000_data/eval_owl_cg10k.py` 相同。",
          "- 骨質減少改用影像 L1 HU 標籤：≤110 HU 陽性、>160 HU 陰性、110–160 HU 視同 unclear 排除（比照定義 B 的排除規則；≤160 原本是篩檢門檻）。", ""]

    L += ["## 2. 句庫（S1）", "",
          "- 10,000 份報告取 Findings 段（IMP 之前、略過 indication/technique 行），用規則＋關鍵字分到 25 個欄位（14 類對應欄位＋膽囊正常、"
          "肝臟病灶、門靜脈、胰臟、腎上腺、淋巴結、膀胱、下胸部、腸繫膜、骨骼破壞、腎臟其他），再判斷陰性/陽性。沒有用 Qwen。",
          "- 每個院區 × 欄位 × 極性挑最常見、只對應單一欄位、不含數字/se-im/比較/分級字眼的句子；院區沒有合適句子時用通用寫法（標 fallback）。", ""]
    L += [cov.rename(columns={"n_findings_sentences": "Findings 句數", "n_slotted": "分到欄位", "frac_slotted": "比例"}).to_markdown(index=False), ""]
    cw = can[can.slot_type == "model"].copy()
    cw["欄位"] = cw.slot.map(lambda s: ZH.get(s, "膽囊正常" if s == "gallbladder_normal" else s))
    cw["句子"] = cw.sentence + np.where(cw.source == "fallback", "（fallback）", "")
    piv = cw.pivot_table(index=["欄位", "polarity"], columns="LOC", values="句子", aggfunc="first").reset_index()
    L += ["### 模型判讀欄位的句子", "", piv.rename(columns={"polarity": "極性"}).to_markdown(index=False), ""]
    cn = can[can.slot_type == "na"].pivot_table(index="slot", columns="LOC", values="sentence", aggfunc="first").reset_index()
    L += ["### 模型未判讀欄位（加 [NOT ASSESSED]）", "", cn.rename(columns={"slot": "欄位"}).to_markdown(index=False), ""]

    L += ["## 3. 門檻（S3，只用 dev）", "",
          "- 陰性門檻 t_neg：dev 陽性中敏感度 ≥ 0.95；陽性門檻 t_pos：dev 陰性中特異度 ≥ 0.95。分數 < t_neg 寫陰性句；> t_pos 寫陽性句；中間寫陽性句＋`[please verify]`，不寫陰性句。",
          "- **陽性少的類別（dev 陽性 < 20：腸阻塞 8、闌尾炎 4、腹腔游離氣體 9）**：容許漏掉的陽性數 floor(0.05×n) = 0，t_neg 就是 dev 最低的陽性分數（dev 敏感度 1.0，但 95% 下界只有 .40–.66）。"
          "這三類盛行率 1–3%，特異度 .95 時 PPV 只有幾 %，中間區又佔 30–70% 檢查；若照標準規則，七成草稿會出現「Acute appendicitis [please verify]」。"
          "因此改用「罕見類」規則：> t_pos 只寫陽性句＋`[please verify]`（不寫確定的陽性句），中間區不寫任何句子，< t_neg 才寫陰性句（仍然不會在中間區寫陰性）。",
          "- 動脈粥狀硬化 dev 陰性只有 54 例（大多數報告沒寫），t_pos 的特異度估計不穩。", ""]
    t = th.copy()
    t["疾病"] = t.disease.map(ZH)
    t["規則"] = t.rule.map(RULE_ZH)
    t = t[["疾病", "label", "dev_n_pos", "dev_n_neg", "t_neg", "t_pos", "dev_sens_at_tneg", "dev_sens_cp_lower", "dev_spec_at_tpos", "規則"]]
    t["label"] = t.label.map({"report def B": "報告定義 B", "L1 HU<=110 pos / >160 neg": "L1 HU"})
    t = t.rename(columns={"label": "標籤", "dev_n_pos": "dev 陽性", "dev_n_neg": "dev 陰性", "dev_sens_at_tneg": "dev 敏感度@t_neg",
                          "dev_sens_cp_lower": "敏感度 95% 下界", "dev_spec_at_tpos": "dev 特異度@t_pos"})
    L += [t.round(3).to_markdown(index=False), "", "（門檻是 OWL logit。）", ""]

    L += ["## 4. 定量數值（S4）", "",
          "- CT-rate 上已有 OWL 用的 TotalSegmentator（fast，3 mm）mask：`/media/daikso/CTDATA/cg10k/owl/main/masks/`（6,970 例全有）。"
          "在 CT-rate 用 CPU 計算（約 2 分鐘），只帶回小 CSV；腳本與結果放在 `/media/daikso/Tony/VLM/CG10k_reportgen/`。",
          "- 脾臟體積 = 體素數 × 體素體積（脾臟碰到掃描上下緣時視為截斷，不寫）；腹主動脈最大徑 = T12 中心以下到分叉、每層最大主動脈連通區的等效圓直徑取最大值（3 mm 格點，誤差約 ±3 mm；層數 < 10 不寫）。"
          "L1 骨小樑 HU 用既有的 `osteo_hu_label_main.csv`（L1 不可用時用 L2，並在草稿註明層級）。",
          "- 草稿最後一行寫「Auto-measurements (segmentation-based, please verify): …」。", ""]
    L += [mtab.to_markdown(index=False), ""]
    L += [f"- 安全規則（dev 上決定）：模型判陰性但主動脈徑 ≥ 30 mm → 改成請確認；模型判陰性但脾臟體積 > {guards['spleen_ml']:.0f} mL（dev 陰性的 97.5 百分位）→ 改成請確認；"
          "骨質減少模型判陰性但 L1 ≤ 110 HU、或判陽性但 L1 > 160 HU → 改成請確認。",
          "- 注意：骨質減少的評估標籤本身就是 L1 HU，加了 HU 規則後骨質減少的組裝敏感度會變成 1.0，這部分是循環的，只代表「草稿不會和自己寫的 HU 矛盾」。",
          "- 觀察：脾臟體積在 dev 的 AUROC 比 OWL 高，且低分端分得開（用體積當分數時 dev 陰性區約 30%，OWL 只有 4%）。第一階段照規劃用 OWL 決定，建議下一版脾腫大改用體積（或兩者合併）決定，可以大幅縮小請確認區。", ""]

    L += ["## 5. 草稿格式（S5）", "",
          "- LOC(8)：Indications / Techniques / Comparison / Findings（第一行留「[To be completed by radiologist]」給腫瘤病灶）/ IMP，一句一行。",
          "- LOC(2)：Clinical information / Technique / Comparison exam / `<INDEX LESIONS>`（留白）/ `<OTHER FINDINGS>` / IMPRESSION。",
          "- LOC(3)：Clinical Information / Comparison / `<Main findings>`（留白）/ `<Other findings>` 依器官編號 / IMP。",
          "- 標記放在句點前：`Fatty liver [please verify].`、`Unremarkable change of the pancreas [NOT ASSESSED].`。",
          "- 膽囊：膽囊切除判陽性時不寫膽結石句；切除與膽結石都判陰性時寫該院區的「膽囊正常」句（同時代表兩者陰性）。", "",
          "### test 全體（4,879 例）的決策分布", ""]
    At = A[A.subset == "test_all"].copy()
    dist = pd.DataFrame({"疾病": At["疾病"], "陰性句": At.frac_neg.map(pct), "請確認": At.frac_verify.map(pct),
                         "陽性句": At.frac_pos.map(pct), "不寫（罕見類中間區）": At.frac_omit.map(pct)})
    L += [dist.to_markdown(index=False), ""]

    L += ["## 6. 評估（S6，test）", "",
          f"- A6000 只有一台 vLLM（Qwen3.8-27B-FP8，且和另一個任務共用）。全部 4,879 份預估單獨就要約 4.2 小時，所以依協調指示抽樣：test 中罕見類（闌尾炎、腸阻塞、腹腔游離氣體、腹主動脈瘤）定義 B 陽性全部納入，"
          f"再依院區比例隨機補到 {n_sub} 份（`ids_s6_subsample.txt`；抽樣只用來決定要反抽哪些檢查，門檻完全沒有看 test）。成功解析 {n_ex} 份。",
          "- 罕見類陽性全收，所以這幾類的敏感度用到 test 全部陽性；但抽樣集的盛行率比 test 全體高，PPV/F1 會偏高。§6.2 另列 test 全體 4,879 例只看組裝決策（不需 Qwen）的結果供對照。", "",
          "### 6.1 草稿反抽 vs 原報告（定義 B）", "",
          "嚴格：草稿反抽為 yes/probably_yes 才算陽性；寬鬆：unclear 也算陽性（`[please verify]` 句 Qwen 多半判 unclear）。", ""]
    rows = []
    for k in DIS:
        s_ = R[(R.disease == k) & (R["mode"] == "strict") & (~R["疾病"].str.contains("報告標籤"))].iloc[0]
        l_ = R[(R.disease == k) & (R["mode"] == "lenient")].iloc[0]
        rows.append({"疾病": ZH[k], "N/陽性": f"{int(s_.N)}/{int(s_.n_pos)}",
                     "嚴格 敏感度": f3(s_.sens), "嚴格 特異度": f3(s_.spec), "嚴格 PPV": f3(s_.PPV), "嚴格 F1": f3(s_.F1),
                     "寬鬆 敏感度": f3(l_.sens), "寬鬆 特異度": f3(l_.spec), "寬鬆 F1": f3(l_.F1)})
    L += [pd.DataFrame(rows).to_markdown(index=False), ""]
    ro = R[R["疾病"].str.contains("報告標籤")]
    if len(ro):
        r0 = ro.iloc[0]
        L += [f"- 骨質減少若改用報告標籤（嚴格）：N/陽性 {int(r0.N)}/{int(r0.n_pos)}，敏感度 {f3(r0.sens)}、特異度 {f3(r0.spec)}、F1 {f3(r0.F1)}。", ""]

    L += ["### 6.2 組裝決策 vs 原報告（陽性 = 陽性句或請確認）", ""]
    rows = []
    for k in DIS:
        a = A[(A.subset == "test_all") & (A.disease == k)].iloc[0]
        b = A[(A.subset == "test_extracted") & (A.disease == k)].iloc[0]
        rows.append({"疾病": ZH[k], "test 全體 N/陽性": f"{int(a.N)}/{int(a.n_pos)}", "敏感度": f3(a.sens), "特異度": f3(a.spec),
                     "F1": f3(a.F1), "陽性卻寫陰性": f"{int(a.neg_stmt_on_pos)}（{pct(a.crit_rate)}）",
                     "抽樣集 敏感度": f3(b.sens), "抽樣集 特異度": f3(b.spec)})
    L += [pd.DataFrame(rows).to_markdown(index=False), ""]

    L += ["### 6.3 一致性（組裝時寫了什麼 vs Qwen 從草稿讀出什麼）", "",
          "預期：陽性句 → yes/probably_yes；請確認 → yes/probably_yes/unclear；陰性句 → no/probably_no；未寫 → not_mentioned；"
          "已寫膽囊切除而未寫膽結石 → no/not_mentioned。", ""]
    c = Cn.copy()
    c["寫法"] = c.rendered.map(STATE_ZH)
    c = c[["疾病", "寫法", "n", "consistent", "yes", "probably_yes", "unclear", "probably_no", "no", "not_mentioned"]]
    c["consistent"] = c.consistent.map(pct)
    L += [c.rename(columns={"consistent": "一致率"}).to_markdown(index=False), ""]
    tot = Cn.n.sum()
    L += [f"- 整體一致率（加權）：{pct((Cn.n * Cn.consistent).sum() / tot)}（{tot} 個 檢查×類別）。", ""]

    L += ["### 6.4 陰性句涵蓋率與嚴重錯誤", "",
          "- 陰性句涵蓋率 = 原報告明確寫陰性（no/probably_no；骨質減少用 L1 > 160 HU）中，草稿反抽也是陰性的比例。",
          "- 嚴重錯誤 = 原報告陽性，但草稿反抽為 no/probably_no（反抽）／組裝寫了陰性句（組裝）。", ""]
    v = V.copy()
    v = v[["疾病", "report_neg", "neg_covered", "neg_coverage", "report_pos", "critical", "critical_rate", "critical_assembly", "critical_rate_assembly"]]
    v["neg_coverage"] = v.neg_coverage.map(pct)
    v["critical_rate"] = v.critical_rate.map(pct)
    v["critical_rate_assembly"] = v.critical_rate_assembly.map(pct)
    L += [v.rename(columns={"report_neg": "報告陰性", "neg_covered": "草稿也寫陰性", "neg_coverage": "涵蓋率", "report_pos": "報告陽性",
                            "critical": "嚴重錯誤(反抽)", "critical_rate": "比例(反抽)", "critical_assembly": "嚴重錯誤(組裝)",
                            "critical_rate_assembly": "比例(組裝)"}).to_markdown(index=False), ""]
    L += ["- 人工抽查：`side_by_side_samples.md`（每院區 10 份，原報告與草稿並排，附原報告的定義 B 陽性項目）。", ""]

    L += ["## 7. 已知限制", "",
          "1. 請確認區太大（脂肪肝、脾腫大約 80%，膽結石、腹水、骨折、動脈粥狀硬化約 50%）；草稿對醫師的實際節省有限。這是雙門檻設計在 AUROC .77–.90 的分類器上必然的結果。下一步：脾腫大改用體積、腹主動脈瘤改用直徑，或把敏感度目標依類別調整（需要醫師決定可接受的漏報）。",
          "2. 特異度 .95 的陽性門檻在盛行率低的類別（腎水腫、膽管擴張、腹主動脈瘤）PPV 只有 10–20%，「確定的陽性句」仍有大量偽陽性；罕見類只寫請確認。",
          "3. 罕見類（腸阻塞、闌尾炎、腹腔游離氣體）dev 陽性 4–9 例，t_neg 估計很不穩（敏感度 95% 下界 .40–.66），test 上的敏感度也只有 11–27 個陽性可以驗證。",
          "4. 自動測量的主動脈直徑寫進草稿後，Qwen 依定義（< 3 cm 為 probably_no）會覆蓋模型的動脈瘤判斷；部分反抽的嚴重錯誤來自這裡（組裝時其實寫的是請確認）。3 mm mask、等效圓直徑對迂曲或有血栓的動脈瘤會低估。",
          "5. 骨質減少的 HU 規則與評估標籤同源（循環）。",
          "6. 很多陰性句院區原本不常寫（例：No fatty liver、No abnormal bone density、No dilated appendix 是 fallback 通用句），草稿會比原報告多出一些陰性描述；動脈粥狀硬化、骨質減少的報告陰性本來就很少。",
          "7. 抽取用的 prompt 原本是為 Stanford 急診報告寫的（和 `disease14_qwen.csv` 相同，所以可比較），對 `[please verify]`、`[NOT ASSESSED]` 標記沒有特別說明；Qwen 多把請確認句判成 unclear。",
          "8. 只評估 14 類與陰性句；腫瘤病灶、比較前次、IMP 都留白，沒有評估。C- 檢查與延遲期沒有用。",
          "9. S6 是抽樣（見 §6），罕見類陽性全收，PPV/F1 不能直接外推到全體。", ""]

    L += ["## 8. 檔案", "",
          "程式（Mac）：`/Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data/report_gen/`",
          "- `common.py`（路徑、定義 B）、`s1_sentence_bank.py`、`s2_split.py`、`s3_thresholds.py`、`s4_measure_ctrate.py`（在 CT-rate 執行）、`s5_assemble.py`、`s6_subsample.py`、`s6_eval.py`、`s6_report.py`", "",
          "產出（Mac）：`/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/report_gen/`",
          "- S1：`sentence_bank.csv`、`canonical_sentences.csv`、`s1_coverage.csv`",
          "- S2：`split.csv`",
          "- S3：`thresholds.csv`、`guards.json`、`s3_dev_metrics.csv`",
          "- S4：`measurements_ts.csv`（脾臟體積、主動脈徑，CT-rate 計算）、`measurements.csv`（草稿實際使用的值，含 L1 HU）",
          "- S5：`decisions.csv`、`drafts/<EXAMID>.txt`（6,970 份）、`drafts_dev.jsonl`、`drafts_test.jsonl`",
          "- S6：`ids_s6_subsample.txt`、`labels_drafts_test.jsonl`（Qwen 反抽原始輸出）、`eval_phase1.csv`、`s6_assembly_test.csv`、`s6_consistency.csv`、`s6_coverage.csv`、`side_by_side_samples.md`、本檔 `eval_phase1.md`", "",
          "伺服器：A6000 `~/Desktop/Tony/VLM/CG10k_reportgen/`（反抽輸入/輸出與 log）；CT-rate `/media/daikso/Tony/VLM/CG10k_reportgen/`（S4 腳本與 CSV）。", ""]
    (OUT / "eval_phase1.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
