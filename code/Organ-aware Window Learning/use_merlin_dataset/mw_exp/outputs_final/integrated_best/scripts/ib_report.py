"""Write integrated_best/report.md (繁體中文) from results/*.json, integrated_per_finding.csv and main_all.json."""
import os, sys, json, glob, pandas as pd
R = sys.argv[1].rstrip("/") + "/"
D = pd.read_csv(R + "integrated_per_finding.csv"); M = json.load(open(R + "main_all.json"))
FEAT = {"hydronephrosis": "腎臟液體 HU 特徵（左右 max/min/差）", "atherosclerosis": "主動脈＋髂動脈壁鈣化 HU",
        "free_air": "腸腔外空氣 HU（全量補齊）", "ascites": "腹膜腔積液 HU（全量補齊）", "abdominal_aortic_aneurysm": "主動脈直徑幾何＋金屬",
        "surgically_absent_gallbladder": "膽囊體積／膽囊窩／手術夾", "bowel_obstruction": "腸道口徑／內容物（全量補齊）",
        "splenomegaly": "脾臟幾何＋體型", "osteopenia": "椎體骨小樑 HU＋骨骼幾何（全量補齊）", "hepatic_steatosis": "肝脾 HU",
        "appendicitis": "RLQ 局部 HU（右下腹）"}
HEAD = {"E3": "E3_late", "E3max": "兩腎 MIL max head", "E3+E3max": "E3_late＋兩腎 MIL max head", "all12": "all-12 單元 head",
        "E3+all12": "E3_late＋all-12 單元 head", "E3+st": "E3_late＋加胃單元 head", "win": "5 個 window logit（E0、E1×4、E3_late）",
        "E3+cd": "E3_late＋肝‖(肝−脾) head", "csb": "大腸＋小腸 head", "E3+csb": "E3_late＋大腸＋小腸 head"}
def desc(r):
    fd, fs, cand = r.finding, str(r["模型"]), str(r["候選"])
    if fd == "gallstones": return f"GS5 {fs}（膽囊 HU＋膽囊窩／手術夾＋3 mm 紋理＋膽囊 crop probe＋肝回填 head，GBM）"
    if fd == "biliary_ductal_dilation": return "BD5：cat4＋E3_late＋E0＋肝門 crop embedding＋HU＋管狀＋肝門特徵（GBM）"
    if fd == "fracture": return "Exp J add 10 seed（1.5 mm 椎體＋骨骼單元）"
    base, _, tail = fs.partition("+feat") if "+feat" in fs else fs.partition("+core")
    kind = "GBM" if cand.startswith("GBM") else "LR"
    add = "" if fs in HEAD else ("＋" + FEAT.get(fd, "特徵") + ("（精選子集）" if "+core" in fs else ""))
    return f"{HEAD.get(base, base)}{add}（{kind}）"
f3 = lambda x: f"{x:.3f}"; pm = lambda x: f"{x:+.3f}"
rows = []
for _, r in D.iterrows():
    star = "" if r.d_lo > 0 else "（n.s.）"
    oofE3 = "0.837＊" if r.finding == "fracture" else f3(r.train_OOF_E3)
    rows.append(f"| {r['疾病']} | {desc(r)} | {oofE3} → {f3(r.train_OOF)} | **{f3(r.val)}** [{f3(r.val_lo)}, {f3(r.val_hi)}] | "
                f"{f3(r.val_E3_expB)} | {pm(r.d)} [{pm(r.d_lo)}, {pm(r.d_hi)}]{star} | {f3(r.val_age_E3)} → {f3(r.val_age)}"
                f"{' ⚠' if r.flag_age_drop else ''} | {r.spearman_E3:.2f} → {r.spearman:.2f} | {r.sens90_E3*100:.1f}% → {r.sens90*100:.1f}% |")
extra = []
for p in sorted(glob.glob(R + "results/split_*.json")) + [R + "results/fracture_expI.json"]:
    if not os.path.exists(p): continue
    r = json.load(open(p)); zh = r.get("zh", "骨折（Exp I 集成，只用 3 mm）")
    oe = "0.837＊" if "expI" in p else f3(r['oof_E3'])
    extra.append(f"| {zh} | {r['featset']} | {oe} → {f3(r['oof'])} | **{f3(r['val'])}** [{f3(r['val_ci'][0])}, {f3(r['val_ci'][1])}] | "
                 f"{f3(r['val_E3_expB'])}" + (f"（新標籤重訓 E3_late {f3(r['val_E3_rerun'])}）" if r.get('val_E3_rerun') else "") +
                 f" | {pm(r['d_val_expB']['mean'])} [{pm(r['d_val_expB']['ci'][0])}, {pm(r['d_val_expB']['ci'][1])}] | "
                 f"{f3(r['val_age_E3'])} → {f3(r['val_age'])}{' ⚠' if r['val_age'] < r['val_age_E3'] else ''} | {r['spearman_neg_E3']:.2f} → {r['spearman_neg']:.2f} | "
                 f"{r['sens90_val_E3']*100:.1f}% → {r['sens90_val']*100:.1f}% |")
H = ("| 疾病 | 正式最佳模型（train OOF 選出） | train OOF：E3_late → 最佳 | val AUROC [95% CI] | val E3_late（expB） | Δval vs E3_late [95% CI] | "
     "年齡配對 AUROC（val）E3 → 最佳 | Spearman(分數, 年齡｜陰性) E3 → 最佳 | 90% 特異度敏感度（val）E3 → 最佳 |\n|---|---|---|---|---|---|---|---|---|")
ss = M.get("split_substituted", {})
txt = f"""# 各疾病正式整合最佳版本（val）

日期：2026-11-06｜A6000 `~/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/integrated_best/`｜**test 未讀取、未評分、未抽特徵**

## 0. 規則（所有疾病一致）
- 只用 train＋val：先依 split.csv 過濾，再讀標籤；expB 的 probe npz 只取 val 列（test 列立即丟棄、從未使用）。
- **選模只看 train 5-fold OOF**，fold 沿用各疾病自己的 E3_late OOF fold（同一個疾病內一致）。候選：LR（C 0.1／0.3／1，class-balanced）與 4 組 HistGBM（與 gallstones_improve 相同的格點）；特徵組合 = 各 head 組合 × {{只有 head、head＋影像特徵（全部或精選子集）}}。選定後用全部 train refit，val 只套用一次。
- val 指標：AUROC＋分層 bootstrap 95% CI（2,000 次）；Δ 為相對 **expB E3_late val（3 seed 平均機率）** 的 paired bootstrap；年齡配對 AUROC（10 歲年齡層內配對，依配對數加權）；陰性中分數與年齡的 Spearman；90% 特異度敏感度（val 自身閾值）。
- **年齡／性別從未作為模型輸入**；骨質減少 bundle 裡的 metadata 欄位一律剔除。標籤定義未更動（脂肪肝、闌尾炎拆分列為額外列）。
- 計算：CPU only（nice 19、≤4 threads）；train/val 切片在 Mac／CT-rate 上計算後已刪除。

## 1. 各疾病（14 項原始 finding）
{H}
""" + "\n".join(rows) + f"""

⚠ = 年齡配對 AUROC 低於 E3_late（14 項原始疾病都沒有觸發；額外列中「可疑等闌尾炎」觸發）。n.s. = Δ 的 95% CI 跨 0。＊骨折的 train OOF 基準取全骨骼 E3_late 10 seed（Exp J 報告 .837）。

### 額外列（使用者已決定的標籤拆分、骨折替代版）
{H}
""" + "\n".join(extra) + f"""

## 2. main_all（14 項原始 finding 平均，val；study 層級 joint paired bootstrap 1,000 次）
| 版本 | main_all [95% CI] |
|---|---|
| expB E3_late | {M['before']:.3f} [{M['before_ci'][0]:.3f}, {M['before_ci'][1]:.3f}] |
| **正式整合最佳** | **{M['after']:.3f}** [{M['after_ci'][0]:.3f}, {M['after_ci'][1]:.3f}] |
| Δ | **{M['delta']:+.3f}** [{M['delta_ci'][0]:+.3f}, {M['delta_ci'][1]:+.3f}] |
| 拆分標籤取代後（{', '.join(ss.get('substituted', {}).values())}）：E3_late → 整合最佳 | {ss.get('before', float('nan')):.3f} → {ss.get('after', float('nan')):.3f} |

- 骨折 val 用 Exp J 共同集（4,946／4,961 例）；其餘疾病用全部有標籤的 val。
- 拆分取代版：脂肪肝 → 瀰漫性脂肪肝、闌尾炎 → 確定的急性闌尾炎（主要定義）；E3_late 端用原 expB E3_late 對新標籤評分。
"""
open(R + "report_body.md", "w").write(txt)
print(txt)
