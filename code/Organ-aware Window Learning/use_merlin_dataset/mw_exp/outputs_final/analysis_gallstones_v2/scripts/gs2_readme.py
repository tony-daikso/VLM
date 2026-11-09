import json, sys
S = sys.argv[1]; R = json.load(open(S + "/out/export_recs.json"))
def n(x, d=0):
    try:
        x = float(x); return "—" if x != x else f"{x:.{d}f}"
    except Exception: return "—"
L = ["膽結石 Exp N 錯誤病例檢查包（v2）", "=" * 34,
"資料：全部是 VAL 病例（沒有任何 test）。模型 = 目前膽結石最佳 Exp N（G5c＋膽囊／膽囊窩 crop 單元 U），val AUROC .881。",
"閾值：logit −2.059（train OOF 陰性的 90% 特異度；val 直接沿用）。val 在此閾值下敏感度 67.1%、特異度 90.7%。",
"挑選方式：FN = 分數最低的 8 例陽性；FP = 分數最高的 5 例陰性；TP = 2 例分數最高、膽囊內明顯高密度的陽性。",
"完整分析：mw_exp/outputs_final/analysis_gallstones_v2/report.md",
"",
"檔案（每個病例 3 個）：",
"  <編號>_<ID>_ct.nii.gz       原始 Merlin CT（未經處理；從外接硬碟 My Passport/Tony/VLM/data/CT/merlin/merlin_data 唯讀複製，與 CT_FN_check 同來源）",
"  <編號>_<ID>_gb_mask.nii.gz  TotalSegmentator 3 mm 遮罩，以最近鄰重採樣到原始 CT 網格（shape／affine 與 CT 相同）",
"                               標籤：1 = 膽囊，2 = 肝臟（定位參考）。遮罩缺失病例只有肝臟。",
"  <編號>_<ID>_gallery.png     上排腹部窗 W400/L40、下排結石窄窗 W200/L50；",
"                               左：原始解析度軸切面（膽囊最大截面），中：冠狀切面，右：3 mm 軸切面（模型輸入解析度）。綠線 = 膽囊遮罩。",
"                               遮罩缺失病例改用 Exp N crop 的膽囊窩推估點（肝 bbox 比例）置中。",
"",
"注意：「原始解析度」的 HU 數值是用 3 mm 遮罩放大後計算，邊緣會包含膽囊壁或肝的部分容積，最大 HU 會偏高。",
"",
"病例一覽", "--------"]
for r in R:
    L.append(f"{r['key']}  {r['study_id']}｜標籤：{r['label']}｜模型分數 sigmoid {r['score_p']:.3f}（logit {r['score_logit']:+.2f}）｜{r['phase']}")
    L.append(f"      類別：{r['category']}")
    L.append(f"      3 mm 膽囊：{n(r['gb_vox3'])} vox，最大 {n(r['gb_max3'])} HU，>130 HU {n(r['gb_n130_3'])} vox，>200 HU {n(r['gb_n200_3'])} vox")
    if r.get("nat_max") is not None:
        L.append(f"      原始解析度：膽囊 {r['gb_ml_native']:.1f} mL，平均 {r['nat_mean']:.0f} HU，最大 {r['nat_max']:.0f} HU，>200 HU {r['nat_n200']} vox")
    else:
        L.append("      原始解析度：TS 沒有膽囊遮罩（GS6 1.5 mm 重分割也沒有膽囊 label），影像以膽囊窩推估點置中")
    L.append(f"      報告（膽囊相關句子）：{r['report_gb']}")
    L.append("")
L += ["類別說明", "--------",
"FN：A 標籤弱（疑似措辭／僅病史或超音波）｜B 同時標膽囊切除｜C 膽囊遮罩缺失｜D 收縮／被結石充滿｜E 3 mm 無 >100 HU 小塊（等密度／微小）｜F 只有 100–200 HU 微弱小塊｜G 有 >200 HU 仍漏",
"FP：d 膽囊壁病變（瓷器／腺肌症／息肉）｜g0 報告只寫 no radiopaque gallstones（不能排除等密度結石）｜h2 肝硬化背景、膽囊沒有其他發現",
"", "在 ITK-SNAP 開啟", "----------------",
"1. File → Open Main Image… → 選 <編號>_<ID>_ct.nii.gz",
"2. Segmentation → Open Segmentation… → 選同一病例的 <編號>_<ID>_gb_mask.nii.gz（1 = 膽囊、2 = 肝臟）",
"3. Tools → Image Contrast → Contrast Adjustment：腹部 L40/W400；結石 L50/W200（找等密度結石可再窄到 L40/W150）；鈣化 L400/W1800",
"", "請協助判斷（可直接回覆編號）",
"- FN-2、FN-7（E 類）：原始解析度看得到結石嗎？若完全看不到，代表標籤來自其他資訊（超音波或前次檢查），屬於 CT 上無法偵測的陽性。",
"- FN-1、FN-6：收縮的膽囊在哪裡？可以用來設計膽囊窩偵測。",
"- FP-1（腺肌症＋壁內鈣化）、FP-2（瓷器膽囊）：要維持陰性嗎？（目前定義是陰性）",
"- FP-5：「No radiopaque gallstones」要當陰性、unclear，還是 soft label？"]
open("/Users/tony.tu/Desktop/CT_GS_errors/README.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L[:30]))
