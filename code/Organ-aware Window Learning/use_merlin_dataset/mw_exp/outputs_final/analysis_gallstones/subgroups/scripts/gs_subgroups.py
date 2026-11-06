"""Gallstone subgroup analysis (train OOF primary, val descriptive). Train+val ONLY: split filtered before reading any label/report.
Stratification only - no label changes. Outputs: subgroups_train_oof.csv, subgroups_val.csv, case_flags_trval.csv (no text), subgroup_counts.json."""
import json, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = "/home/advantech/Desktop/Tony/VLM/"
I = R + "OWL/mw_exp/outputs_final/"
O = I + "analysis_gallstones/subgroups/"
OK = {"train", "val"}
spl = pd.read_csv(R + "OWL/data/split.csv").set_index("study_id")["split"]

# ---------- 1. labels + text (train/val only) ----------
lab = {}
for line in open(R + "OWL/data/labels_disease14_all.jsonl"):
    r = json.loads(line)
    if r.get("split") not in OK or spl.get(r["study_id"]) not in OK:
        continue
    g = r["answer"]["gallstones"]
    lab[r["study_id"]] = dict(gs_cat=g["category"], gs_calc=(g.get("detail") or {}).get("calcified", "na"),
                              quote=" ".join(g.get("quotes") or []),
                              bdd_cat=r["answer"]["biliary_ductal_dilation"]["category"])
rep_all = json.load(open(R + "merlin_disease14/reports.json"))
rep = {}
for sid in lab:
    v = rep_all.get(sid) or {}
    if v.get("split", spl.get(sid)) in OK:
        rep[sid] = v.get("report", "") or ""
del rep_all

GBW = re.compile(r"gallbladder|gall bladder|cholelith|gallstone|choledocholith|sludge|cholecyst|biliary|cystic duct|common bile|common duct|\bcbd\b|pericholecystic", re.I)
STONE = re.compile(r"stone|cholelith|calcul", re.I)
NONGB = re.compile(r"kidney|renal|ureter|urinary|\bbladder|pyelo|nephro|appendic|ureteropelvic|ureterovesic|\bujv|\buvj|prostat|salivary|pancreatic duct|tonsil", re.I)
GB = type("G", (), {"search": staticmethod(lambda s: bool(GBW.search(s)) or (bool(STONE.search(s)) and not NONGB.search(s)))})
NEG = re.compile(r"\b(?:no|without|negative for|not|nor|absence of|free of|resolved|rather than)\b", re.I)


def sents(t):
    return [s for s in re.split(r"(?<=[.;])\s+|\n+", t) if s.strip()]


def pos_hit(pattern, ss, neg_window=150):
    """True if pattern matches in any sentence without a negation cue in the preceding clause text."""
    p = re.compile(pattern, re.I)
    for s in ss:
        for m in p.finditer(s):
            pre = s[max(0, m.start() - neg_window):m.start()]
            pre = re.split(r"[:;]|\bbut\b|\bhowever\b|\bwith\b|\bwhich\b|\band\b(?= (?:there|a|an|mild|small|moderate|large))", pre)[-1]
            if not NEG.search(pre):
                return True
    return False


rows = []
for sid, d in lab.items():
    ss = [s for s in sents(rep.get(sid, "")) if GB.search(s)] + sents(d["quote"])
    st = [s for s in ss if STONE.search(s) and not NONGB.search(s)]
    txt = " ".join(ss)
    f = dict(study_id=sid)
    # size (mm only from stone sentences that are not about duct calibre)
    mm = []
    for s in ss:
        if STONE.search(s) and not re.search(r"\b(?:cbd|common bile|common duct|hepatic duct|ductal|kidney|renal|ureter)", s, re.I):
            mm += [float(x) * (10 if u.lower() == "cm" else 1) for x, u in re.findall(r"(\d+(?:\.\d+)?)\s*(mm|cm)\b", s, re.I)]
    mm = [x for x in mm if 0.5 <= x <= 60]
    f["stone_mm"] = max(mm) if mm else np.nan
    w_small = pos_hit(r"\b(?:tiny|punctate|small|minute|millimetric|subcentimeter)\s+(?:[\w-]+\s+){0,3}?(?:gallstone|stone|calcul|cholelith|calcification|hyperdens|densit|foc)|microlith", st)
    w_large = pos_hit(r"\b(?:large|bulky|huge|giant|dominant)\s+(?:[\w-]+\s+){0,3}?(?:gallstone|stone|calcul)|(?:filled|packed) with (?:\w+ ){0,2}(?:gallstone|stone|calcul)", st)
    if mm:
        f["size"] = "微小/小(<5mm)" if f["stone_mm"] < 5 else "中(5-10mm)" if f["stone_mm"] < 10 else "大(≥10mm)"
    else:
        f["size"] = "大(≥10mm)" if w_large else "微小/小(<5mm)" if w_small else "未記載"
    f["size_explicit_mm"] = int(bool(mm))
    # number
    multi = pos_hit(r"\b(?:multiple|numerous|several|innumerable|many|few|two|three|at least \w+|\d+) (?:\w+ ){0,3}?(?:stones|gallstones|calculi|calcifications)\b|\b(?:stones|gallstones|calculi|cholelithiases)\b", ss)
    single = pos_hit(r"\b(?:single|solitary|one|an?|1)\s+(?:\w+\s+){0,3}?(?:gallstone|stone|calculus)\b(?!s)|\b(?:gallstone|calculus)\b(?!s)", ss)
    f["number"] = "多顆" if multi else "單顆" if single else "未記載(僅 cholelithiasis)"
    # calcification (report)
    calc_kw = pos_hit(r"calcif|radiopaque|radio-opaque|hyperdens|high[- ]density|dense stone|lamellated|rim[- ]calcif", st)
    noncalc_kw = pos_hit(r"non-?calcified|not calcified|isodense|faint|subtle|low[- ]density|hypodense (?:\w+ )?(?:gall)?stone|cholesterol|radiolucent|(?:not|poorly|better) (?:well )?(?:seen|visualized) on (?:ct|ultrasound|us)|seen on (?:prior |outside |recent )?(?:ultrasound|sonograph|mri)", ss)
    f["calc_rep"] = ("非鈣化/等密度/模糊" if (d["gs_calc"] == "no" or noncalc_kw) else
                     "鈣化/高密度" if (d["gs_calc"] == "yes" or calc_kw) else "未記載")
    # location / configuration (overlapping flags)
    f["loc_neck"] = int(pos_hit(r"\bneck\b|impacted|lodged|cystic duct|infundibul|hartmann", st))
    f["loc_layer"] = int(pos_hit(r"layering|dependent|layer(?:s|ed)?\b", st))
    f["loc_fundus"] = int(pos_hit(r"fundus|fundal", st))
    f["loc_filled"] = int(pos_hit(r"fill(?:ing|s)? (?:the |much of the |nearly the entire )?gallbladder|gallbladder (?:is )?(?:completely |nearly )?(?:filled|packed)|(?:filled|packed) with (?:\w+ ){0,2}(?:gallstone|stone|calcul)|gallbladder\W+(?:\w+\W+){0,4}?(?:contracted|collapsed|decompressed)|(?:contracted|collapsed|decompressed) gallbladder|wall[- ]echo", ss))
    f["loc_porcelain"] = int(pos_hit(r"porcelain|wall calcif|calcified wall|mural calcif|calcification of the (?:gallbladder )?wall", ss))
    # complications
    f["cx_cholecystitis"] = int(pos_hit(r"cholecystitis|pericholecystic|gallbladder wall (?:is )?(?:\w+ )?thicken|wall (?:is )?(?:\w+ )?thicken|thickened (?:gallbladder )?wall|hyperemi|mucosal (?:hyper)?enhancement|emphysematous|gangren", [x for x in ss if not NONGB.search(x) or GBW.search(x)]))
    f["cx_distended"] = int(pos_hit(r"gallbladder (?:is |appears )?(?:\w+ )?(?:distended|distention|distension)|(?:distended|hydropic) gallbladder|gallbladder (?:\w+ )?hydrops|gallbladder: distended", ss))
    f["cx_cbd"] = int(pos_hit(r"choledocholith|(?:stone|calcul)\w* (?:\w+ ){0,4}(?:common bile|common duct|\bcbd\b)|(?:cbd|common bile duct|common duct) (?:\w+ ){0,3}(?:stone|calcul)|mirizzi", ss))
    f["y_bdd"] = int(d["bdd_cat"] in ("yes", "probably_yes"))
    # certainty
    unc = re.search(r"possible|possibly|may represent|cannot (?:be )?exclude|questionable|equivocal|suspected|suspicious|likely|probable|suggest|versus|vs\.?|could represent|cannot be excluded|not excluded", d["quote"], re.I)
    f["certainty"] = "可能/疑似" if (d["gs_cat"] == "probably_yes" or unc) else "確定"
    # sludge
    sludge = bool(re.search(r"sludge|tumefactive|vicarious", txt, re.I)) and pos_hit(r"sludge|tumefactive|vicarious", ss)
    stone_words = pos_hit(r"stone|cholelith|calcul|lith\b", ss)
    f["sludge"] = "僅膽泥" if (sludge and not stone_words) else "膽泥+結石" if sludge else "無膽泥"
    f["gs_cat"] = d["gs_cat"]; f["gs_calc"] = d["gs_calc"]; f["has_report"] = int(bool(rep.get(sid)))
    rows.append(f)
F = pd.DataFrame(rows)

# ---------- 2. image features + scores ----------
hu = pd.concat([pd.read_csv(I + f"analysis_gallstones/{x}") for x in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")[["study_id", "gb_vox", "gb_maxblob100", "gb_max"]]
F = F.merge(hu, on="study_id", how="left")
F["gb_missing"] = (F.gb_vox.fillna(0) < 100).astype(int)
F["img_vis"] = ((F.gb_maxblob100.fillna(0) >= 2) & (F.gb_missing == 0)).astype(int)
F["img_axis"] = np.where(F.gb_missing == 1, "膽囊遮罩缺失/過小", np.where(F.img_vis == 1, "影像可見高密度(>100HU)", "影像無高密度"))
F["gb_mask"] = np.where(F.gb_missing == 1, "缺失/過小(<100 vox)", "存在(≥100 vox)")
cx = np.where((F.cx_cholecystitis == 1) & ((F.cx_cbd == 1) | (F.y_bdd == 1)), "膽囊炎徵象+膽管結石/擴張",
     np.where(F.cx_cholecystitis == 1, "合併膽囊炎徵象",
     np.where((F.cx_cbd == 1) | (F.y_bdd == 1), "合併膽管結石/膽管擴張", "單純膽結石")))
F["complication"] = cx
F.drop(columns=[]).to_csv(O + "case_flags_trval.csv", index=False)

tr = pd.read_csv(I + "gallstones_improve/gs1/gs1_oof_train.csv")[["study_id", "y", "E3_late", "quickwins", "V4"]]
va = pd.read_csv(I + "gallstones_improve/gs1/gs1_val_scores.csv")[["study_id", "y", "E3_late", "quickwins", "V4"]]
tr = tr.merge(F, on="study_id", how="left"); va = va.merge(F, on="study_id", how="left")
assert tr.gb_vox.notna().mean() > .95 and va.gb_vox.notna().mean() > .95
thr = {m: float(np.quantile(tr.loc[tr.y == 0, m], .9)) for m in ("E3_late", "quickwins", "V4")}
MODELS = ["E3_late", "quickwins", "V4"]

# ---------- 3. subgroup definitions (masks on positives) ----------
def defs(D):
    p = D.y == 1
    S = [("全部", "全部陽性", p)]
    for ax, col in [("1 大小", "size"), ("2 數目", "number"), ("3a 鈣化(報告)", "calc_rep"), ("3b 影像高密度(3mm)", "img_axis"),
                    ("5 合併症", "complication"), ("6 確定度", "certainty"), ("7 膽泥", "sludge"), ("8 膽囊遮罩", "gb_mask")]:
        for v in sorted(D.loc[p, col].dropna().unique()):
            S.append((ax, v, p & (D[col] == v)))
    S.append(("1 大小", "有明確 mm/cm 記載", p & (D.size_explicit_mm == 1)))
    for k, nm in [("loc_neck", "頸部/嵌頓/膽囊管"), ("loc_layer", "分層/重力依賴"), ("loc_fundus", "底部"),
                  ("loc_filled", "充滿膽囊/膽囊收縮含石"), ("loc_porcelain", "瓷器膽囊/壁鈣化")]:
        S.append(("4 位置/型態", nm, p & (D[k] == 1)))
    S.append(("4 位置/型態", "未記載位置", p & (D[["loc_neck", "loc_layer", "loc_fundus", "loc_filled", "loc_porcelain"]].sum(axis=1) == 0)))
    S.append(("5 合併症(重疊)", "任何膽囊炎徵象", p & (D.cx_cholecystitis == 1)))
    S.append(("5 合併症(重疊)", "膽囊擴張(單獨記載)", p & (D.cx_distended == 1)))
    S.append(("5 合併症(重疊)", "報告提及膽管結石", p & (D.cx_cbd == 1)))
    S.append(("5 合併症(重疊)", "膽管擴張標籤陽性", p & (D.y_bdd == 1)))
    for rc in ["鈣化/高密度", "非鈣化/等密度/模糊", "未記載"]:   # report x image
        for ic in ["影像可見高密度(>100HU)", "影像無高密度", "膽囊遮罩缺失/過小"]:
            S.append(("3c 報告×影像", f"{rc} × {ic}", p & (D.calc_rep == rc) & (D.img_axis == ic)))
    return S


def evaluate(D, split, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    neg = (D.y == 0).values
    fn_all = {m: ((D.y == 1) & (D[m] <= thr[m])).values for m in MODELS}
    S = defs(D)
    # bootstrap: resample negatives once per rep (shared), positives per subgroup
    negs = {m: D.loc[neg, m].values for m in MODELS}
    bidx = [rng.integers(0, neg.sum(), neg.sum()) for _ in range(B)]
    bneg = {m: [np.sort(negs[m][ix]) for ix in bidx] for m in ("E3_late", "V4")}
    out = []
    for ax, nm, mk in S:
        mk = mk.values; n = int(mk.sum())
        r = dict(axis=ax, subgroup=nm, n_pos=n, pct_pos=round(100 * n / max(1, (D.y == 1).sum()), 1))
        for m in MODELS:
            if n >= 3:
                sel = mk | neg
                r[f"auc_{m}"] = roc_auc_score(D.y.values[sel], D[m].values[sel])
            else:
                r[f"auc_{m}"] = np.nan
            if m in ("E3_late", "V4") and n >= 30:
                sp = D.loc[mk, m].values; a = []
                for b in range(B):
                    ps = sp[rng.integers(0, n, n)]; ns = bneg[m][b]
                    lo = np.searchsorted(ns, ps, "left"); hi = np.searchsorted(ns, ps, "right")
                    a.append(((lo + hi) / 2).mean() / len(ns))
                r[f"auc_{m}_lo"], r[f"auc_{m}_hi"] = np.quantile(a, [.025, .975])
        miss = mk & fn_all["V4"]
        r["V4_miss_rate_spec90"] = round(100 * miss.sum() / n, 1) if n else np.nan
        r["share_of_V4_FN"] = round(100 * miss.sum() / max(1, fn_all["V4"].sum()), 1)
        r["E3_miss_rate_spec90"] = round(100 * (mk & fn_all["E3_late"]).sum() / n, 1) if n else np.nan
        out.append(r)
    res = pd.DataFrame(out)
    res.insert(0, "split", split)
    return res


rt = evaluate(tr, "train_oof"); rv = evaluate(va, "val")
rt.round(4).to_csv(O + "subgroups_train_oof.csv", index=False); rv.round(4).to_csv(O + "subgroups_val.csv", index=False)

# sludge-only labelling & report x image crosstab (train)
cnt = {}
t = tr.merge(F[["study_id"]], on="study_id")
cnt["thr_spec90_train"] = thr
cnt["n_train_pos"] = int((tr.y == 1).sum()); cnt["n_val_pos"] = int((va.y == 1).sum())
cnt["V4_FN_train"] = int(((tr.y == 1) & (tr.V4 <= thr["V4"])).sum()); cnt["V4_FN_val"] = int(((va.y == 1) & (va.V4 <= thr["V4"])).sum())
cnt["sludge_x_label_train"] = {f"{k}": v for k, v in pd.crosstab(tr.sludge, tr.gs_cat).stack().to_dict().items()}
cnt["sludge_x_label_val"] = {f"{k}": v for k, v in pd.crosstab(va.sludge, va.gs_cat).stack().to_dict().items()}
cnt["calc_rep_x_img_train_pos"] = {f"{k}": v for k, v in pd.crosstab(tr[tr.y == 1].calc_rep, tr[tr.y == 1].img_axis).stack().to_dict().items()}
cnt["img_axis_in_negatives_train"] = tr[tr.y == 0].img_axis.value_counts().to_dict()
cnt["size_mm_train_pos"] = tr[tr.y == 1].stone_mm.describe().round(1).to_dict()
json.dump(cnt, open(O + "subgroup_counts.json", "w"), ensure_ascii=False, indent=1, default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
cols = ["axis", "subgroup", "n_pos", "pct_pos", "auc_E3_late", "auc_quickwins", "auc_V4", "auc_V4_lo", "auc_V4_hi", "V4_miss_rate_spec90", "share_of_V4_FN"]
print(rt[cols].round(3).to_string()); print(rv[cols].round(3).to_string()); print(json.dumps(cnt, ensure_ascii=False, default=str, indent=1))
