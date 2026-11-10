"""CG10k external validation: per-disease best OWL model vs uniform E3_late (expB).
Same A/B/C definitions, bootstrap (1000, seed 0, exam level) and paired DeLong as eval_owl_cg10k.py.
Inputs (data/CT/CG/10000_data/): owl_infer/probs_main.csv (E3_late), owl_infer/best/probs_best_main.csv + methods_main.csv,
disease14_qwen.csv/.jsonl, disease14_split_rules.csv, nifti_3mm/series_selection.csv.
Outputs: owl_infer/best/eval_best_*.csv and a section appended (replaced) in owl_infer/report.md."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_owl_cg10k as ev
from eval_owl_cg10k import DIS, ZH, POS, NEG_EXPLICIT, auc, auc_ci, boot_idx, delong_paired, MIN_POS

ROOT = ev.ROOT; OUT = ev.OUT; BD = OUT / "best"
TAG = "<!-- BEST_PER_DISEASE -->"
# Merlin val (official, Notion §12 / integrated_best v1 for the tier-A stand-ins), E3_late expB val
MERLIN_E3 = {"hepatic_steatosis": .819, "surgically_absent_gallbladder": .956, "splenomegaly": .980, "gallstones": .783,
             "biliary_ductal_dilation": .834, "hydronephrosis": .897, "abdominal_aortic_aneurysm": .975, "atherosclerosis": .888,
             "osteopenia": .881, "fracture": .823, "bowel_obstruction": .950, "appendicitis": .853, "ascites": .883, "free_air": .861}
MERLIN_USED = {"gallstones": .783, "biliary_ductal_dilation": .834, "fracture": .823, "hepatic_steatosis": .832, "surgically_absent_gallbladder": .984, "splenomegaly": .981, "hydronephrosis": .944,
               "abdominal_aortic_aneurysm": .990, "atherosclerosis": .912, "osteopenia": .884, "bowel_obstruction": .956,
               "appendicitis": .879, "ascites": .906, "free_air": .901}
MERLIN_FORMAL = {"hepatic_steatosis": .832, "surgically_absent_gallbladder": .984, "splenomegaly": .981, "gallstones": .881,
                 "biliary_ductal_dilation": .911, "hydronephrosis": .957, "abdominal_aortic_aneurysm": .990, "atherosclerosis": .921,
                 "osteopenia": .890, "fracture": .854, "bowel_obstruction": .956, "appendicitis": .920, "ascites": .916, "free_air": .912}


def load(s="main"):
    e3 = pd.read_csv(OUT / f"probs_{s}.csv")
    for k in DIS:
        sc = [c for c in e3.columns if c.startswith(k + "__seed")]
        if sc: e3[k] = e3[sc].mean(axis=1)
    be = pd.read_csv(BD / f"probs_best_{s}.csv"); meth = pd.read_csv(BD / f"methods_{s}.csv").set_index("finding").method
    lab = pd.read_csv(ROOT / "disease14_qwen.csv", encoding="utf-8-sig")
    sel = pd.read_csv(ROOT / "nifti_3mm/series_selection.csv"); sel = sel[sel.status == "ok"]; fl = sel["flags"].fillna("")
    sel = sel.assign(chest=fl.str.contains("chest_series"), thick=fl.str.contains("thick_slices"))
    sp = pd.read_csv(ROOT / "disease14_split_rules.csv").rename(columns={"study_id": "EXAMID"})
    q = {}
    for l in open(ROOT / "disease14_qwen.jsonl"):
        r = json.loads(l); a = r["answer"]
        f = a["fracture"]; d = f.get("detail") or {}
        q[r["study_id"]] = dict(fx_comp=str(d.get("vertebral_compression")), fx_sites="|".join(sorted(d.get("sites") or [])),
                                fx_acute=str(d.get("acute")), hs_grade=str((a["hepatic_steatosis"].get("detail") or {}).get("grade")),
                                hs_quote=" ".join(a["hepatic_steatosis"].get("quotes") or []))
    q = pd.DataFrame.from_dict(q, orient="index").rename_axis("EXAMID").reset_index()
    oh = ROOT / f"osteo_hu_label_{s}.csv"                     # image-derived L1/L2 trabecular HU label (cg_l1hu.py)
    if oh.exists():
        o = pd.read_csv(oh).rename(columns={"study_id": "EXAMID"})[["EXAMID", "hu", "level_used", "y_hu110", "y_hu160", "excl_reason"]]
        q = q.merge(o, on="EXAMID", how="left")
    else:
        q = q.assign(hu=np.nan, level_used="", y_hu110=np.nan, y_hu160=np.nan, excl_reason="")
    df = e3[["study_id"] + DIS].rename(columns={"study_id": "EXAMID"})
    df = df.merge(be[["study_id"] + DIS].rename(columns={"study_id": "EXAMID", **{k: k + "__best" for k in DIS}}), on="EXAMID")
    df = df.merge(lab[["EXAMID"] + [k + "_cat" for k in DIS]], on="EXAMID").merge(sel[["EXAMID", "chest", "thick"]], on="EXAMID", how="left")
    df = df.merge(sp[["EXAMID", "y_diffuse", "y_focal", "hs_subtype"]], on="EXAMID", how="left").merge(q, on="EXAMID", how="left")
    df = df.merge(sel[["EXAMID", "phase"]], on="EXAMID", how="left")
    return df, meth


def labels(df, k, mode):
    """returns y (float, NaN = excluded) for finding/extra row k under A or B."""
    if k in DIS:
        cat = df[k + "_cat"].astype(str); y = cat.isin(POS).astype(float)
        if mode == "B": y[~(cat.isin(POS) | cat.isin(NEG_EXPLICIT))] = np.nan
        return y.values
    base = labels(df, k.split(":")[0], mode)
    if k == "hepatic_steatosis:diffuse":            # Merlin locked rule: focal-only -> 0, unassigned/unclear -> NaN
        y = df.y_diffuse.values.astype(float); y[np.isnan(base)] = np.nan; return y
    if k == "hepatic_steatosis:diffuse_metamorph":  # sensitivity: unassigned + 'metamorph' in quote -> diffuse positive
        y = df.y_diffuse.values.astype(float)
        m = (df.hs_subtype == "unassigned") & df.hs_quote.fillna("").str.contains("metamor", case=False)
        y[m.values] = 1.0; y[np.isnan(base)] = np.nan; return y
    if k in ("osteopenia:hu110", "osteopenia:hu160"):  # image label, same under A and B (not report-derived)
        return df["y_" + k.split(":")[1]].values.astype(float)
    if k in SUBS:                                   # positive subgroup vs all negatives of the definition
        y = base.copy(); y[(base == 1) & ~SUBS[k][1](df).values] = np.nan; return y
    if k.startswith("fracture:"):                   # site/compression positives vs all fracture negatives; other positives excluded
        pos = base == 1
        sub = (df.fx_comp == "yes").values if k == "fracture:compression" else df.fx_sites.fillna("").str.contains("vertebra").values
        y = base.copy(); y[pos & ~sub] = np.nan; return y
    raise KeyError(k)


SITES = [("vertebra", "椎體"), ("rib", "肋骨"), ("pelvis", "骨盆"), ("sacrum", "薦骨"), ("hip_femur", "髖股骨"), ("other", "其他")]
SUBS = {"hepatic_steatosis:sub_diffuse": ("脂肪肝子群－瀰漫性（規則 hs_subtype diffuse/both）", lambda d: d.hs_subtype.isin(["diffuse", "both"])),
        "hepatic_steatosis:sub_focal": ("脂肪肝子群－局部（規則 hs_subtype focal/both）", lambda d: d.hs_subtype.isin(["focal", "both"]))}
for gk, gz in (("mild", "輕度"), ("moderate", "中度"), ("severe", "重度"), ("not_stated", "未註明")):
    SUBS[f"hepatic_steatosis:diffuse_{gk}"] = (f"脂肪肝子群－瀰漫性・{gz}（Qwen grade）", lambda d, gk=gk: d.hs_subtype.isin(["diffuse", "both"]) & (d.hs_grade == gk))
SUBS["fracture:sub_compression"] = ("骨折子群－椎體壓迫性（vertebral_compression=yes）", lambda d: d.fx_comp == "yes")
for sk, sz in SITES:
    SUBS[f"fracture:site_{sk}"] = (f"骨折子群－部位含{sz}", lambda d, sk=sk: d.fx_sites.fillna("").str.split("|").apply(lambda x: sk in x))
    SUBS[f"fracture:only_{sk}"] = (f"骨折子群－只有{sz}", lambda d, sk=sk: d.fx_sites.fillna("") == sk)
for ak, az in (("yes", "急性"), ("no", "陳舊"), ("mixed", "混合"), ("not_stated", "未註明")):
    SUBS[f"fracture:acute_{ak}"] = (f"骨折子群－{az}（Qwen acute）", lambda d, ak=ak: d.fx_acute == ak)
ROWS = DIS + ["hepatic_steatosis:diffuse", "hepatic_steatosis:diffuse_metamorph", "fracture:compression", "fracture:vertebra",
              "osteopenia:hu110", "osteopenia:hu160"] + list(SUBS)
RZH = {**ZH, "hepatic_steatosis:diffuse": "瀰漫性脂肪肝（Merlin 鎖定規則）", "hepatic_steatosis:diffuse_metamorph": "瀰漫性脂肪肝（fatty metamorphosis 算瀰漫，敏感度）",
       "fracture:compression": "骨折－椎體壓迫性", "fracture:vertebra": "骨折－椎體（部位含 vertebra）",
       "osteopenia:hu110": "骨質減少－影像標籤 L1 HU≤110", "osteopenia:hu160": "骨質減少－影像標籤 L1 HU≤160（篩檢）",
       **{k: v[0] for k, v in SUBS.items()}}


def evaluate(df, tag, mode, rng):
    rows = []
    for k in ROWS:
        f = k.split(":")[0]; y = labels(df, k, mode); ok = ~np.isnan(y)
        yy = y[ok].astype(bool); sb = df[f + "__best"].values[ok]; se = df[f].values[ok]
        ok2 = ~np.isnan(sb); yy, sb, se = yy[ok2], sb[ok2], se[ok2]
        n, npos = len(yy), int(yy.sum())
        if npos == 0 or npos == n: continue
        idx = boot_idx(n, rng)
        ab, ae, dd, lo, hi, p = delong_paired(yy, sb, se)
        bl = np.array([auc(yy[i], sb[i]) for i in idx]); el = np.array([auc(yy[i], se[i]) for i in idx]); bd = bl - el
        rows.append(dict(set=tag, row=k, 疾病=RZH[k], N=n, n_pos=npos, n_neg=n - npos, AUROC_best=ab, best_lo=np.nanpercentile(bl, 2.5), best_hi=np.nanpercentile(bl, 97.5),
                         AUROC_E3=ae, E3_lo=np.nanpercentile(el, 2.5), E3_hi=np.nanpercentile(el, 97.5), diff=dd, boot_lo=np.nanpercentile(bd, 2.5),
                         boot_hi=np.nanpercentile(bd, 97.5), DeLong_p=p, small=npos < MIN_POS))
    return pd.DataFrame(rows)


def main(s="main"):
    rng = np.random.default_rng(ev.SEED)
    df, meth = load(s)
    qc = df[~(df.chest.fillna(False) | df.thick.fillna(False))].reset_index(drop=True)
    R = pd.concat([evaluate(df, "A", "A", rng), evaluate(df, "B", "B", rng), evaluate(qc, "C-A", "A", rng), evaluate(qc, "C-B", "B", rng)])
    R.to_csv(BD / f"eval_best_{s}_long.csv", index=False)
    f3 = lambda a, l, h: f"{a:.3f} ({l:.3f}–{h:.3f})"
    def table(sets):
        W = []
        for k in ROWS:
            f = k.split(":")[0]; r = {"疾病": RZH[k], "CG 使用方法": meth.get(f, "")}
            for t in sets:
                x = R[(R.set == t) & (R.row == k)]
                if not len(x): r[f"{t} N/陽性/陰性"] = "–"; continue
                x = x.iloc[0]; star = "*" if x.small else ""
                r[f"{t} N/陽性/陰性"] = f"{x.N}/{x.n_pos}/{x.n_neg}" + ("（陰性樣本少）" if x.n_neg < 300 else "")
                r[f"{t} 最佳"] = f3(x.AUROC_best, x.best_lo, x.best_hi) + star
                r[f"{t} E3_late"] = f3(x.AUROC_E3, x.E3_lo, x.E3_hi) + star
                r[f"{t} Δ(最佳−E3) [boot CI] p"] = f"{x['diff']:+.3f} [{x.boot_lo:+.3f}, {x.boot_hi:+.3f}] p={x.DeLong_p:.3g}"
            if k in DIS:
                mm = str(meth.get(f, "")); used = MERLIN_FORMAL[f] if ("(tier B)" in mm or "C 級" in mm) else MERLIN_USED.get(f, MERLIN_E3[f])
                r["Merlin val：E3_late → CG 用的模型（正式最佳）"] = f"{MERLIN_E3[f]:.3f} → {used:.3f} ({MERLIN_FORMAL[f]:.3f})"
            W.append(r)
        return pd.DataFrame(W).fillna("")
    W = table(("B", "C-B"))                      # primary: B (neg = explicit no/probably_no only), user decision 2026-11-06
    table(("A", "C-A")).to_csv(BD / f"eval_best_{s}_table_appendixA.csv", index=False)
    W.to_csv(BD / f"eval_best_{s}_table.csv", index=False)
    # osteopenia vs continuous HU: Spearman, overall and by contrast phase (C+ vs C-); known contrast bias
    from scipy.stats import spearmanr
    O = []
    for ph, g in [("全部", df)] + [(p, g) for p, g in df.groupby(df.phase.map(lambda x: "C-" if x == "noncontrast" else "C+"))]:
        g = g[g.hu.notna()]
        if len(g) < 30: continue
        r = dict(期別=ph, N=len(g), HU中位數=g.hu.median(), 陽性110=int(g.y_hu110.sum()), 陽性160=int(g.y_hu160.sum()),
                 Spearman_best=spearmanr(g["osteopenia__best"], g.hu).correlation, Spearman_E3=spearmanr(g["osteopenia"], g.hu).correlation)
        for t in ("hu110", "hu160"):
            y = g["y_" + t].values.astype(bool)
            if 0 < y.sum() < len(y):
                idx = boot_idx(len(y), rng)
                for nm, col in (("best", "osteopenia__best"), ("E3", "osteopenia")):
                    sc = g[col].values; b = np.array([auc(y[i], sc[i]) for i in idx])
                    r[f"{t}_{nm}"] = f"{auc(y, sc):.3f} ({np.nanpercentile(b, 2.5):.3f}–{np.nanpercentile(b, 97.5):.3f})"
        O.append(r)
    O = pd.DataFrame(O); O.to_csv(BD / f"eval_best_{s}_osteo_hu.csv", index=False)
    return W, R, df


E3_COORD = {"hepatic_steatosis:sub_diffuse": .781, "hepatic_steatosis:sub_focal": .572, "hepatic_steatosis:diffuse_mild": .698,
            "hepatic_steatosis:diffuse_moderate": .831, "hepatic_steatosis:diffuse_severe": .952, "fracture": .764,
            "fracture:sub_compression": .830, "fracture:only_rib": .680, "fracture:site_pelvis": .658, "fracture:site_hip_femur": .684,
            "fracture:acute_no": .717}


def report(W, s="main"):
    meth = pd.read_csv(BD / f"methods_{s}.csv")
    O = pd.read_csv(BD / f"eval_best_{s}_osteo_hu.csv")
    cols = ["疾病", "CG 使用方法", "B N/陽性/陰性", "B 最佳", "B E3_late", "B Δ(最佳−E3) [boot CI] p", "C-B Δ(最佳−E3) [boot CI] p"]
    main_rows = W[W.疾病.isin([ZH[k] for k in DIS])]
    L = [TAG, "", "## 每病最佳 vs 統一 E3_late（CG10k main，主要定義 B）", "",
         "- 主要定義 **B**：陽性 = yes/probably_yes；陰性只限報告明確寫沒有（no/probably_no）；not_mentioned/unclear 排除。A（陰性 = 其餘全部）放附錄 `best/eval_best_main_table_appendixA.csv`。C-B = 排除 chest_series / thick_slices 後的 B。",
         "- Δ = 最佳 − E3_late，同一批 1,000 次 bootstrap（exam 層級，seed 0）95% CI；p = 配對 DeLong。`*` 陽性 < 30。陰性 < 300 註明「陰性樣本少」。",
         "- 「最佳」的每個模型都是 Merlin 正式最佳，用**原程式**在 Merlin train refit（CG 列以 split 'cg'/'val' 附加、從不參與訓練），影像特徵與 crop／embedding 程式先在 Merlin 病例上重現。CG 的 E3_late 是同一個 expB E3_late。",
         "- 14 病全部使用 Merlin 正式最佳（方法欄）。重現：所有 head／stack 在 Merlin val 逐值或 ≤5e-5 重現；例外：骨折 J_add 為 GPU 訓練（10 seed 平均 val .8477 vs 存檔 .8481，相關 .9997），C3 stacker 以存檔 Merlin 成員 refit（val 差 2e-14）；膽管擴張 45 例無肝門 ROI（原程式 present=False）視為缺失。CG 輸入為 3 mm NIfTI（1.5 mm 單元由 3 mm 上採樣，與 Merlin 原生解析度不同，屬已知偏差）。", "",
         "- **脂肪肝**：正式方法仍是 Merlin val（train OOF）選定的 integrated stack（E3_late＋肝‖(肝−脾) head＋肝脾 HU）；不因 CG 結果改回 E3_late（那等於用外部驗證資料選模型）。CG 上此 stack **未改善**（B Δ −.003，n.s.），且**中度子群顯著變差**（見子群表），兩者並列報告。",
         "- 骨質減少：主要以**影像標籤**（L1 小樑骨 HU）評估；報告標籤列為參考。L2 替代／硬化／骨水泥規則為本次新增，**待使用者確認**。", "",
         "### 14 個原始疾病", "", main_rows[cols + ["Merlin val：E3_late → CG 用的模型（正式最佳）"]].to_markdown(index=False), "",
         "### 子群與替代標籤（B 定義：子群陽性 vs 報告明確陰性；其他陽性排除）", ""]
    sub = W[~W.疾病.isin([ZH[k] for k in DIS])].copy()
    inv = {v: k for k, v in RZH.items()}
    sub["協調者 E3_late 基準"] = [E3_COORD.get(inv.get(z, ""), "") for z in sub.疾病]
    L += [sub[["疾病", "B N/陽性/陰性", "B 最佳", "B E3_late", "B Δ(最佳−E3) [boot CI] p", "協調者 E3_late 基準"]].to_markdown(index=False), "",
          "### 骨質減少：影像標籤（L1 小樑骨 HU；使用者 11-06 新定義）", "",
          "- HU = Merlin Exp J `geom15.py` 的 `hu_body`（1.5 mm 椎體正中矢狀面、侵蝕 2 px 的 HU 中位數，原程式不變）。L1 骨折／硬化改用 L2、骨水泥排除的規則是本次新增（`remote/cg_l1hu.py`，參數事先固定）：骨折 = Qwen 骨折引文提到該節，或（報告有椎體骨折且該節高度損失 ≥ 0.40）；硬化 = 比另一節高 > 100 HU 或 > 500 HU；骨水泥 = ROI ≥ 5% 體素 > 1000 HU。",
          "- 涵蓋全部 v15（1.5 mm）病例。顯影劑會讓小樑骨 HU 上升（C+ 比 C- 高約 10–30 HU），**閾值是否依期別校正：待使用者決定**（此處未校正）。", "",
          O.to_markdown(index=False), "", "### CG 各病使用的方法", "", meth.to_markdown(index=False), ""]
    rp = OUT / "report.md"; txt = rp.read_text(encoding="utf-8")
    if TAG in txt: txt = txt[:txt.index(TAG)].rstrip() + "\n"
    rp.write_text(txt + "\n" + "\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    W, R, df = main("main")
    report(W)
    print(W.to_markdown(index=False))
