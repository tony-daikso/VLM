"""Option A: aggregate tables (no case ids, no labels per case) from out/*_metrics.json -> out/RESULTS_optionA.md.
Verdict rule (fixed before looking at val): paired 95% CI of the delta entirely > 0 -> 'Pillar 有幫助'; entirely < 0 ->
'變差'; otherwise '無差異'.  usage: report.py [--dev]"""
import json
import os
import sys

W = os.path.expanduser("~/Desktop/Tony/VLM/OWL/pillar_optA/")
TAG = "_dev" if "--dev" in sys.argv else ""
ZH = {"hepatic_steatosis": "脂肪肝", "surgically_absent_gallbladder": "膽囊已切除", "splenomegaly": "脾腫大",
      "hydronephrosis": "腎水腫", "abdominal_aortic_aneurysm": "腹主動脈瘤", "atherosclerosis": "動脈粥狀硬化",
      "osteopenia": "骨質減少", "bowel_obstruction": "腸阻塞", "appendicitis": "闌尾炎", "ascites": "腹水",
      "free_air": "腹腔游離氣體", "gallstones": "膽結石", "biliary_ductal_dilation": "膽管擴張", "fracture": "骨折"}
UNITS_DOC = {"hepatic_steatosis": "肝、脾", "surgically_absent_gallbladder": "膽囊、肝、十二指腸", "splenomegaly": "脾、肝",
             "hydronephrosis": "右腎、左腎、膀胱", "abdominal_aortic_aneurysm": "主動脈、兩側髂動脈(合併)",
             "atherosclerosis": "主動脈、兩側髂動脈(合併)", "osteopenia": "L1、L2、L3、L4",
             "bowel_obstruction": "小腸、大腸、胃、十二指腸", "appendicitis": "大腸、小腸、右髂腰肌(右下腹代理)",
             "ascites": "肝、脾、小腸、大腸、膀胱", "free_air": "肝、胃、大腸、小腸",
             "gallstones": "膽囊、肝", "biliary_ductal_dilation": "肝、膽囊、胰、十二指腸、門靜脈/脾靜脈",
             "fracture": "T8–L5 逐節向量的平均＋逐元素最大、肋骨 24 條合併、骨盆(薦骨+S1+兩側髖骨)合併"}


def f(x, d=3):
    return "—" if x is None or x != x else f"{x:.{d}f}".replace("0.", ".", 1) if abs(x) < 1 else f"{x:.{d}f}"


def ci(v, d=3, key="auroc"):
    return f"{f(v[key], d)} [{f(v['ci'][0], d)}, {f(v['ci'][1], d)}]"


def dci(v, d=3):
    s = lambda x: ("+" if x >= 0 else "−") + f(abs(x), d)  # noqa: E731
    return f"{s(v['delta'])} [{s(v['ci'][0])}, {s(v['ci'][1])}]"


def verdict(v, higher_better=True):
    lo, hi = v["ci"]
    if not higher_better:
        lo, hi = -hi, -lo
    return "Pillar 有幫助" if lo > 0 else "變差" if hi < 0 else "無差異"


L = []
P = L.append
P("# Pillar-0 × OWL on Merlin：Option A 結果（val）" + ("（DEV，非正式）" if TAG else ""))
P("")
info = json.load(open(W + "in/prep_info.json"))
P("## 規則")
P("- train 一律 5-fold OOF，**每一類用該類正式 OWL 堆疊自己的 fold**（各類 fold 彼此不同，和肝脂肪的一致率約 20%＝隨機），"
  "所以 Pillar OOF 分數和 OWL OOF 分數屬於同一個切分，堆疊是 OOF-clean。val 只報告一次。test 從未讀取，也沒有 test 的 Pillar 特徵。")
P("- Pillar probe：特徵標準化＋logistic regression；C ∈ {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1}，以 train 內層 CV（外層每折用其餘 4 折做 leave-one-fold-out）"
  "選；val 用全 train 重訓（C 由 5-fold CV 選）。堆疊＝logit(OWL 機率) 與 Pillar OOF 分數的 LR（C=1）。")
P("- 主要堆疊「OWL+P」＝OWL＋Pillar global＋Pillar 器官單元三個分數；另列 OWL＋global、OWL＋單元。")
P("- CI：val AUROC 分層 bootstrap 2,000 次；Δ 為配對 bootstrap。單元層級（每腎、每節椎體）用病例叢集 bootstrap。")
P("- 判定（事前固定）：Δ 的 95% CI 全部 > 0 →「Pillar 有幫助」；全部 < 0 →「變差」；其餘「無差異」。")
P("")

c = json.load(open(W + f"out/cls14_metrics{TAG}.json"))
P("## 1. 14 類疾病（val AUROC [95% CI]）")
P("")
P("| 疾病 | val n（陽性） | OWL | Pillar global | Pillar 器官單元 | OWL+P 堆疊 | Δ 堆疊−OWL | 判定 |")
P("|---|---|---|---|---|---|---|---|")
for d, o in c.items():
    v = o["val"]
    P(f"| {ZH[d]} | {o['n_val']:,}（{o['pos_val']}） | {ci(v['owl'])} | {ci(v['pg'])} | {ci(v['pu'])} | {ci(v['st'])} | "
      f"{dci(v['st-owl'])} | {verdict(v['st-owl'])} |")
P("")
P("次要堆疊（Δ 對 OWL）：")
P("")
P("| 疾病 | OWL+global | Δ | OWL+單元 | Δ | 單元−global | train OOF：OWL / global / 單元 / 堆疊 |")
P("|---|---|---|---|---|---|---|")
for d, o in c.items():
    v, t = o["val"], o["train_oof"]
    P(f"| {ZH[d]} | {ci(v['st_g'])} | {dci(v['st_g-owl'])} | {ci(v['st_u'])} | {dci(v['st_u-owl'])} | {dci(v['pu-pg'])} | "
      f"{f(t['owl'])} / {f(t['pg'])} / {f(t['pu'])} / {f(t['st'])} |")
P("")
P("器官單元（TS total 遮罩，vec64＝64³ token 的遮罩加權平均；同組多個標籤用 w 精確合併，組間串接；遮罩不存在的單元補 0）：")
P("")
for d, u in UNITS_DOC.items():
    P(f"- {ZH[d]}：{u}")
P("")

NAMES = {"owl": "OWL 單元（同儕 probe）", "pglob": "Pillar global（弱基準）", "p64": "Pillar vec64", "p16": "Pillar vec16",
         "p64_16": "Pillar vec64+vec16", "m4_raw": "M4 量測單一值", "m4": "M4 量測 LR", "st_owl_p": "OWL+Pillar",
         "st_owl_m4": "OWL+M4", "st_owl_p_m4": "OWL+Pillar+M4", "st_p_m4": "Pillar+M4"}
MET = {"side_exact_": "側別完全正確", "uni_argmax_": "單側 argmax", "auc_": "單元 AUROC", "top1_": "top-1 節段"}
for task, title, m2, m2n in (("kidney", "2a. 腎水腫側別（每腎）", "side_exact", "病例側別完全正確"),
                             ("vert", "2b. 骨折節段（T8–L5 每節）", "top1", "top-1 節段命中")):
    p = W + f"out/{task}_unit_metrics{TAG}.json"
    if not os.path.exists(p):
        continue
    u = json.load(open(p))
    v = u["val"]
    P(f"## {title}")
    P("")
    if task == "kidney":
        P(f"val 病例 {u['n_val_cases']:,}；側別評估 {u['n_side_cases']} 例（雙側佔 {f(u['bilateral_share'])}），單側 {u['n_unilateral']} 例。"
          "M4 單一值＝flcc_ml（腎盂/集尿系統液體代理）。側別完全正確＝分數 ≥ train OOF 的 F1 門檻的側（皆不過門檻則取較高側），與同儕定義相同。")
        P("")
        P("| 方法 | 每腎 AUROC | 病例側別完全正確 | 單側病例 argmax 正確 | 病例 AUROC（兩腎取 max） | train OOF 每腎 AUROC |")
        P("|---|---|---|---|---|---|")
        for k in u["methods"]:
            se = ci(v[f"side_exact_{k}"], key="value") if f"side_exact_{k}" in v else "—"
            P(f"| {NAMES.get(k, k)} | {ci(v[f'auc_{k}'], key='value')} | {se} | {ci(v[f'uni_argmax_{k}'], key='value')} | "
              f"{ci(v[f'caseauc_{k}'], key='value')} | {f(u['train_oof_auc'][k])} |")
    else:
        P(f"val 病例 {u['n_val_cases']:,}；節段評估 {u['n_level_cases']} 例（隨機 top-1 機率 {f(u['chance_top1'])}）。"
          "M4 單一值＝g15_loss（相對鄰節的高度損失）。")
        P("")
        P("| 方法 | 每節 AUROC | top-1 節段命中 | 病例 AUROC（各節取 max） | train OOF 每節 AUROC |")
        P("|---|---|---|---|---|")
        for k in u["methods"]:
            P(f"| {NAMES.get(k, k)} | {ci(v[f'auc_{k}'], key='value')} | {ci(v[f'top1_{k}'], key='value')} | "
              f"{ci(v[f'caseauc_{k}'], key='value')} | {f(u['train_oof_auc'][k])} |")
    P("")
    P("配對 Δ：")
    P("")
    P("| 比較 | Δ [95% CI] | 判定 |")
    P("|---|---|---|")
    for k, x in v.items():
        if "-" in k and "delta" in x:
            a, b = k.split("-")
            mt = next(m for m in MET if a.startswith(m))
            nm = lambda s: NAMES.get(s[len(mt):], s)  # noqa: E731
            P(f"| {MET[mt]}：{nm(a)} − {nm(b)} | {dci(x)} | {verdict(x)} |")
    P("")

p = W + f"out/cheap_metrics{TAG}.json"
if os.path.exists(p):
    ch = json.load(open(p))
    TN = {"hs_grade": "脂肪肝分級（輕度 vs 中重度）", "asc_amount": "腹水量（微量/少量 vs 中量/大量）",
          "ath_grade": "動脈粥狀硬化程度（輕度 vs 中重度）", "aaa_diam_mm": "腹主動脈瘤直徑（mm）", "sp_len_mm": "脾臟長度（mm）"}
    P("## 2c. 其他屬性（只在報告陽性且有寫屬性的病例上）")
    P("")
    P("| 屬性 | train / val n（val 陽性） | OWL 類別分數 | Pillar global | Pillar 器官單元 | 量測 | 堆疊 | Δ 單元−OWL | Δ 堆疊−OWL | 判定（堆疊−OWL） |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for t, o in ch.items():
        v = o["val"]
        if "owl" in v:
            P(f"| {TN[t]} | {o['n_train']} / {o['n_val']}（{o['pos_val']}） | {ci(v['owl'])} | {ci(v['pg'])} | {ci(v['pu'])} | "
              f"{ci(v['meas']) if 'meas' in v else '—'} | {ci(v['st'])} | {dci(v['pu-owl'])} | {dci(v['st-owl'])} | {verdict(v['st-owl'])} |")
    P("")
    P("回歸屬性（val；量測＝分割量測經 train 線性校正；堆疊＝量測＋Pillar 單元 ridge）：")
    P("")
    P("| 屬性 | train / val n | MAE：Pillar global / 單元 / 量測 / 堆疊 | Spearman：Pillar 單元 / 量測 / 堆疊 | Δ MAE 堆疊−量測 | 判定 |")
    P("|---|---|---|---|---|---|")
    for t, o in ch.items():
        v = o["val"]
        if "mae_st" in v:
            P(f"| {TN[t]} | {o['n_train']} / {o['n_val']} | {ci(v['mae_pg'], 1, 'value')} / {ci(v['mae_pu'], 1, 'value')} / "
              f"{ci(v['mae_meas_cal'], 1, 'value')} / {ci(v['mae_st'], 1, 'value')} | {ci(v['rho_pu'], 2, 'value')} / "
              f"{ci(v['rho_meas_cal'], 2, 'value')} / {ci(v['rho_st'], 2, 'value')} | {dci(v['mae_st-mae_meas_cal'], 1)} | "
              f"{verdict(v['mae_st-mae_meas_cal'], higher_better=False)} |")
    P("")
open(W + f"out/RESULTS_optionA{TAG}.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
