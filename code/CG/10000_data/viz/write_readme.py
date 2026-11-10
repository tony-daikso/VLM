"""(Mac) owl_viz/README.md + append example list to ~/Desktop/OWL_ITK/README.txt (keeps the existing text)."""
import json
from pathlib import Path

from common_viz import UNIT_ZH

HERE = Path(__file__).resolve().parent
VIZ = Path("/Users/tony.tu/Desktop/戴承智慧/VLM/data/CT/CG/10000_data/owl_viz")
ITK = Path("/Users/tony.tu/Desktop/OWL_ITK")
UNITS = ['liver', 'spleen', 'pancreas', 'kidney_left', 'kidney_right', 'gallbladder', 'stomach', 'adrenal', 'aorta',
         'small_bowel', 'colon', 'lumbar_spine', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5',
         'ribs_left', 'ribs_right', 'sacrum', 'hip_left', 'hip_right', 'femur_left', 'femur_right']


def main():
    spec = json.load(open(HERE / "work/viz_spec.json"))
    rec = json.load(open(HERE / "work/recommended.json")) if (HERE / "work/recommended.json").exists() else []
    lines = ["# OWL 單元歸因視覺化（E3_late，CG10k main）", "",
             "每個 `<疾病>/<EXAMID>/` 是一個「例子 × 疾病」，資料都在 3 mm prep 網格（TotalSegmentator 前處理，RAS canonical）。",
             "同一套檔案也複製在 `~/Desktop/OWL_ITK/<疾病>/<EXAMID>/`（workspace 內路徑指向桌面那份）。", "",
             "## 檔案", "",
             "| 檔案 | 說明 |", "|---|---|",
             "| `ct.nii.gz` | CT（HU，int16），3 mm 等向 |",
             "| `units_label.nii.gz` | 單元標籤（uint8），id 對照見下表；骨骼單元覆寫在主器官單元上（只有「腰椎整體」與 L1–L5 重疊，所以 id 12 在圖上是空的） |",
             "| `contrib_heatmap.nii.gz` | float32，voxel 值＝所屬單元對此疾病的貢獻，mask 外為 0。一般疾病＝單元對 logit 的精確線性貢獻（相對於 Merlin train 平均器官，可為負）；骨折＝MIL 單元分數減去該例骨骼單元最小值（≥0，越高越像骨折所在） |",
             "| `labels.txt` | ITK-SNAP label description，名稱為「中文單元名 (貢獻值)」；貢獻前 5 名暖色不透明，其餘冷色半透明，未用於此病的單元灰色淡化 |",
             "| `workspace.itksnap` | ITK-SNAP workspace：主影像 CT（W400/L40）、segmentation＝units_label 並帶 label 表、overlay＝熱圖（Hot） |",
             "| `overview.png` | 軸切＋冠狀切（過貢獻最高單元的質心），單元依貢獻上色，前 5 名編號與數值 |",
             "| `meta.json` | 例子資訊與單元清單（id、key、中文名、貢獻、質心與 bbox，RAS mm），依貢獻排序 |",
             "| `grid.json` | 網格資訊（affine、voxel 質心／bbox） |", "",
             "## 用 ITK-SNAP 開", "",
             "1. 最快：雙擊或 File > Open Workspace… 選 `workspace.itksnap`。",
             "2. 若 workspace 讀不進來（不同 ITK-SNAP 版本），手動四步：",
             "   - File > Open Main Image → `ct.nii.gz`（再用 Tools > Image Contrast 設 W400/L40；骨折用 W1500/L300）",
             "   - Segmentation > Open Segmentation → `units_label.nii.gz`",
             "   - Segmentation > Import Label Descriptions → `labels.txt`",
             "   - File > Add Another Image → `contrib_heatmap.nii.gz`，選「as overlay」，Layer Inspector 裡 color map 選 Hot、透明度約 0.5",
             "3. 游標：workspace 格式沒有存游標位置的欄位，開啟後在 Label 清單雙擊排名第 1 的單元，或用 `meta.json` 中第一個單元的 `centroid_vox`（x y z，以 1 起算請 +1）填入 Tools > Cursor Inspector。", "",
             "## label id 對照", "", "| id | key | 中文 |", "|---|---|---|"]
    lines += [f"| {i + 1} | {u} | {UNIT_ZH[u]} |" for i, u in enumerate(UNITS)]
    lines += ["", "## 例子", "", "| 疾病 | EXAMID | 類型 | 模型機率 | 報告 |", "|---|---|---|---|---|"]
    for s in spec:
        q = s["quote"].replace("|", "/")
        lines.append(f"| {s['disease_zh']} | [{s['id']}]({s['disease_zh']}/{s['id']}/overview.png) | {'真陽性' if s['kind'] == 'TP' else '偽陽性'} | {s['prob']:.3f} | {q[:100]} |")
    if rec:
        lines += ["", "## 建議先看", ""] + [f"- `{r['path']}`：{r['why']}" for r in rec]
    lines += ["", "產生程式：`/Users/tony.tu/Desktop/戴承智慧/VLM/code/CG/10000_data/viz/`（attr_units.py → validate_loc.py → select_viz.py → make_itk.py (CT-rate) → build_local.py → write_readme.py）。"]
    (VIZ / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # desktop README.txt: keep original header, (re)write the example list section
    p = ITK / "README.txt"
    txt = p.read_text(encoding="utf-8")
    mark = "\n==== 例子清單（自動產生）===="
    txt = txt.split(mark)[0].rstrip("\n") + "\n" + mark + "\n"
    txt += "游標：ITK-SNAP workspace 不存游標，開啟後在 Label 清單雙擊第 1 名單元，或參考 meta.json 的 centroid_vox。\n"
    txt += "骨折的熱圖＝MIL 單元分數減去該例最小值；其他疾病＝單元對 logit 的線性貢獻（可為負，熱圖只顯示 0 以上）。\n"
    cur = None
    for s in spec:
        if s["disease_zh"] != cur:
            cur = s["disease_zh"]
            txt += f"\n[{cur}]\n"
        q = s["quote"].replace("\n", " ")
        txt += f"  {s['id']}  {'真陽性' if s['kind'] == 'TP' else '偽陽性'}  機率 {s['prob']:.3f}  報告：{q[:110]}\n"
    if rec:
        txt += "\n建議先看：\n" + "".join(f"  {r['desk']}\n    {r['why']}\n" for r in rec)
    p.write_text(txt, encoding="utf-8")
    print("README written")


if __name__ == "__main__":
    main()
