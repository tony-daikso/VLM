"""Gallstones error analysis v2 for Exp N (N_U). TRAIN OOF + VAL only. Test never read (all inputs are pre-filtered train/val files)."""
import re, json, numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
S = sys.argv[1]; I = S + "/in/"; O = S + "/out/"
import os; os.makedirs(O, exist_ok=True)
MW = "/Users/tony.tu/Desktop/戴承智慧/VLM/code/Organ-aware Window Learning/use_merlin_dataset/mw_exp/outputs_final/"
rng = np.random.default_rng(0)
# ---------- scores ----------
tr = pd.read_csv(MW + "gallbladder_sludge/stack/gs_oof_train.csv")[["study_id", "y_gs", "N_U", "sl_group"]].assign(split="train")
va = pd.read_csv(MW + "gallbladder_sludge/stack/gs_val_scores.csv")[["study_id", "y_gs", "N_U", "sl_group"]].assign(split="val")
D = pd.concat([tr, va], ignore_index=True)
assert not set(D.study_id) & set(pd.read_csv("/dev/null" if False else I + "labels_trval.csv").query("split not in ['train','val']").study_id)
v4 = pd.concat([pd.read_csv(I + "gs1_oof_train.csv")[["study_id", "V4"]], pd.read_csv(I + "gs1_val_scores.csv")[["study_id", "V4"]]])
D = D.merge(v4, on="study_id", how="left")
thr = np.quantile(D.query("split=='train' and y_gs==0").N_U, 0.9)
thr4 = np.quantile(D.query("split=='train' and y_gs==0").V4.dropna(), 0.9)
print("thr N_U", thr, "thr V4", thr4)
# ---------- flags ----------
L = pd.read_csv(I + "labels_trval.csv"); assert set(L.split) <= {"train", "val"}
C = pd.read_csv(I + "case_flags_trval.csv")
T = pd.read_csv(I + "gs2_text_trval.csv"); assert set(T.split) <= {"train", "val"}
H = pd.concat([pd.read_csv(I + "hu_a6.csv"), pd.read_csv(I + "hu_ct.csv")]).drop_duplicates("study_id")
M = pd.read_csv("/Volumes/My Passport/Tony/VLM/data/CT/merlin/metadata.csv").rename(columns={"study id": "study_id"})[["study_id", "contrast", "phase", "slicethickness"]]
SL = pd.read_csv(MW + "gallbladder_sludge/labels_trval_sludge.csv")[["study_id", "y_sludge", "sl_unc", "kw_vicarious"]]
D = D.merge(L[["study_id", "gs_cat", "y_chole", "kw_history", "kw_ultrasound", "kw_not_vis", "kw_tiny", "kw_noncalc"]], on="study_id", how="left")
D = D.merge(C[["study_id", "size", "number", "calc_rep", "loc_layer", "loc_filled", "loc_neck", "loc_porcelain", "cx_cholecystitis", "cx_distended", "cx_cbd", "y_bdd", "certainty", "img_axis"]], on="study_id", how="left")
D = D.merge(T[["study_id", "gb_text", "gs_quote", "calc_near", "ascites"]], on="study_id", how="left")
D = D.merge(H[["study_id", "gb_vox", "gb_max", "gb_p99", "gb_n130", "gb_n200", "gb_maxblob100", "aorta_hu_med"]], on="study_id", how="left")
D = D.merge(M, on="study_id", how="left").merge(SL, on="study_id", how="left")
for c in ["gb_text", "gs_quote", "calc_near", "ascites"]: D[c] = D[c].fillna("")
NEG = re.compile(r"\b(?:no|without|negative for|not|nor|absence of|free of|resolved|rather than)\b", re.I)
def pos_hit(pattern, text, neg=True):
    p = re.compile(pattern, re.I)
    for s in re.split(r" \|\| | \| ", text):
        for m in p.finditer(s):
            pre = s[max(0, m.start() - 150):m.start()]
            pre = re.split(r"[:;]|\bbut\b|\bhowever\b|\bwith\b|\bwhich\b", pre)[-1]
            if not neg or not NEG.search(pre): return True
    return False
tx = (D.gb_text + " || " + D.gs_quote)
F = {}
GBONLY = lambda t: " || ".join(x for x in re.split(r" \|\| | \| ", t) if re.search(r"gallbladder|gall bladder|cholelith|gallstone", x, re.I))
F["contracted"] = tx.map(lambda t: pos_hit(r"contracted|collapsed|decompressed", GBONLY(t), neg=False) and not re.search(r"\bnot (?:\w+ )?(?:contracted|collapsed|decompressed)", GBONLY(t), re.I))
F["nondist"] = tx.map(lambda t: pos_hit(r"underdistended|under-distended|poorly distended|minimally distended|nondistended|non-distended|not distended", GBONLY(t), neg=False))
F["cirrhosis"] = tx.map(lambda t: pos_hit(r"cirrho|nodular (?:liver|hepatic) (?:contour|surface|morphology)|hepatic surface nodularity|portal hypertension", t))
F["gb_mass"] = tx.map(lambda t: pos_hit(r"gallbladder (?:\w+ ){0,3}(?:mass|carcinoma|cancer|malignan|neoplasm|tumor)|(?:mass|tumor|metasta\w*|carcinoma) (?:\w+ ){0,6}(?:gallbladder|gallbladder fossa)", t))
F["ct_limited_neg"] = tx.str.contains(r"no (?:\w+ ){0,2}(?:radiopaque|radio-opaque|radiodense|calcified|ct evidence of|hyperdense) (?:\w+ ){0,2}(?:gallstone|stone|cholelith|calcul)|no ct evidence of (?:\w+ ){0,2}(?:gallstone|cholelith|stone)|radiolucent|noncalcified (?:gallstone|stone)s? (?:cannot|can not) be excluded|cannot exclude (?:non-?calcified|radiolucent)", case=False, regex=True)
F["gb_normal"] = tx.str.contains(r"gallbladder: (?:normal|unremarkable)\.?(?: \|\||$)", case=False, regex=True)
GBSTONE = re.compile(r"gallbladder|gallstone|cholelith|stone|calcul|sludge", re.I); NONGB2 = re.compile(r"hepat|liver|renal|kidney|cyst\b|cysts|lesion|nodule|hypodens|lung|pulmonary|adrenal|pancrea|spleen|splenic|lymph", re.I)
def gbs(t): return [x for x in re.split(r" \|\| | \| ", t) if GBSTONE.search(x) and not NONGB2.search(x)]
F["too_small"] = tx.map(lambda t: any(re.search(r"too small to (?:characteriz|charact|definitive|accurately)|(?:tiny|punctate|small) (?:\w+ ){0,3}(?:hyperdens|densit|foc)", x, re.I) for x in gbs(t)))
HEDGE = re.compile(r"possible|possibly|may represent|cannot (?:be )?exclude|questionable|equivocal|suspected|suspicious|probable|suggest|versus|\bvs\.?|could represent|cannot be excluded|not excluded|likely (?!cholesterol)|appear to be|faintly", re.I)
F["hedge_gb"] = tx.map(lambda t: any(HEDGE.search(x) and re.search(r"stone|cholelith|calcul|hyperdens|densit|foc", x, re.I) for x in gbs(t)))
F["adenomyo"] = tx.map(lambda t: pos_hit(r"adenomyomat|adenomyosis|comet tail", t))
F["polyp"] = tx.map(lambda t: pos_hit(r"polyp", t))
F["porcelain"] = tx.map(lambda t: pos_hit(r"porcelain|wall calcif|calcified wall|mural calcif|calcification (?:of|within|in|along) the (?:gallbladder )?wall", t))
F["stent_air"] = tx.map(lambda t: pos_hit(r"pneumobilia|biliary (?:stent|drain)|(?:cbd|common bile duct|biliary|plastic|metallic) stent|stent (?:\w+ ){0,3}(?:cbd|common bile|biliary|duct)|air (?:in|within) the (?:biliary|bile|common|gallbladder)|sphincterotomy|cholecystostomy", t))
F["vicarious"] = tx.map(lambda t: pos_hit(r"vicarious|excreted contrast|contrast (?:material )?(?:layering )?(?:within|in) the gallbladder|layering contrast|hyperdense bile|high[- ]density bile|milk of calcium", t)) | (D.kw_vicarious.fillna(0) == 1)
F["wall_thick"] = tx.map(lambda t: pos_hit(r"cholecystitis|pericholecystic|wall (?:is )?(?:\w+ )?thicken|thickened (?:gallbladder )?wall|thickening of the gallbladder|mural thicken|edema(?:tous)? (?:of )?(?:the )?gallbladder|gallbladder wall edema|hyperemi", t))
F["calc_near"] = D.calc_near.map(lambda t: pos_hit(r"calcif|granuloma", t))
F["ascites"] = D.ascites.map(lambda t: pos_hit(r"ascites|pericholecystic fluid|perihepatic fluid", t) and not re.search(r"ascites: (?:none|no|absent)", t, re.I))
STN = re.compile(r"stone|cholelith|calcul|sludge", re.I)
F["hist_us"] = tx.map(lambda t: any(STN.search(x) and re.search(r"(?:seen|demonstrated|visualized|better seen|identified|evident) (?:\w+ ){0,4}(?:ultrasound|sonogra|\bus\b|mri|mrcp)|on (?:\w+ ){0,2}(?:ultrasound|sonogra|mri|mrcp)|not (?:well |clearly )?(?:seen|visualized|identified) on (?:this |the current |current )?ct|history of (?:\w+ ){0,2}(?:cholelith|gallstone)|known (?:\w+ )?(?:cholelith|gallstone)", x, re.I) for x in re.split(r" \|\| | \| ", t)))
for k, v in F.items(): D["f_" + k] = v.astype(int)
D["f_distended"] = D.cx_distended.fillna(0).astype(int)
D["f_mask_missing"] = (D.gb_vox.fillna(0) < 100).astype(int)
D["f_mask_small"] = ((D.gb_vox.fillna(0) >= 100) & (D.gb_vox.fillna(0) < 300)).astype(int)
D["f_vis"] = ((D.gb_maxblob100.fillna(0) >= 2) & (D.f_mask_missing == 0)).astype(int)
D["f_hd200"] = ((D.gb_n200.fillna(0) >= 1) & (D.f_mask_missing == 0)).astype(int)
D["f_chole"] = (D.y_chole == 1).astype(int)
D["f_sludge"] = (D.y_sludge == 1).astype(int)
D["f_noncontrast"] = (D.contrast == False).astype(int)
D["f_arterial"] = D.phase.astype(str).str.startswith("arterial").astype(int)
D["f_tiny"] = ((D["size"] == "微小/小(<5mm)") | (D.kw_tiny == 1)).astype(int)
D["f_large"] = (D["size"] == "大(≥10mm)").astype(int)
D["f_single"] = (D.number == "單顆").astype(int)
D["f_multi"] = (D.number == "多顆").astype(int)
D["f_calc"] = (D.calc_rep == "鈣化/高密度").astype(int)
D["f_noncalc"] = (D.calc_rep == "非鈣化/等密度/模糊").astype(int)
D["f_layer"] = D.loc_layer.fillna(0).astype(int)
D["f_filled"] = D.loc_filled.fillna(0).astype(int)
D["f_probyes"] = (D.gs_cat == "probably_yes").astype(int)
D["f_probno"] = (D.gs_cat == "probably_no").astype(int)
D["f_notment"] = (D.gs_cat == "not_mentioned").astype(int)
D["f_hedge_pos"] = ((D.gs_cat == "probably_yes") | (D.f_hedge_gb == 1)).astype(int)
D["pred"] = (D.N_U >= thr).astype(int); D["pred4"] = (D.V4 >= thr4).astype(int)
D["err"] = np.where((D.y_gs == 1) & (D.pred == 0), "FN", np.where((D.y_gs == 0) & (D.pred == 1), "FP", np.where(D.y_gs == 1, "TP", "TN")))
# ---------- primary (mutually exclusive) categories ----------
def fn_cat(r):
    if r.f_probyes or r.f_hist_us or r.f_hedge_pos: return "A 標籤弱（疑似措辭／僅病史或超音波可見）"
    if r.f_chole: return "B 標籤衝突（同時標膽囊切除）"
    if r.f_mask_missing: return "C 膽囊遮罩缺失（<100 vox）"
    if r.f_contracted or r.f_filled: return "D 膽囊收縮／被結石充滿"
    if not r.f_vis: return "E 3 mm 無高密度（等密度／微小結石）"
    if r.gb_n200 >= 1: return "G 有 >200 HU 結石仍漏"
    return "F 只有 100–200 HU 微弱高密度"
def fp_cat(r):
    if r.f_chole and not r.f_mask_missing: return "a 膽囊切除標籤但遮罩仍在（誤分割／標籤衝突）"
    if r.f_probno or r.f_notment: return "b 標籤弱（probably_no／報告未提及）"
    if r.f_sludge or r.f_vicarious: return "c 膽泥／顯影劑排泄（vicarious）"
    if r.f_porcelain or r.f_adenomyo or r.f_polyp: return "d 膽囊壁病變（瓷器／腺肌症／息肉）"
    if r.f_stent_air: return "e 膽道支架／膽道積氣"
    if r.f_hd200: return "f 膽囊內 >200 HU（疑漏報結石／顯影劑／鄰近鈣化）"
    if r.f_ct_limited_neg: return "g0 報告只否定「高密度／CT 可見」結石（no radiopaque）"
    if r.f_gb_mass: return "g1 膽囊或膽囊窩腫瘤／轉移"
    if r.f_wall_thick: return "g 膽囊炎／壁增厚"
    if r.f_contracted or r.f_nondist or r.f_mask_missing: return "h1 膽囊收縮／未擴張／遮罩缺失"
    if r.f_cirrhosis: return "h2 肝硬化（無其他膽囊發現）"
    if r.f_vis: return "h3 膽囊內 100–200 HU 小塊（部分容積／壁強化）"
    return "h4 其他（膽囊內無高密度、無特殊發現）"
D["cat"] = ""
m = D.err == "FN"; D.loc[m, "cat"] = D[m].apply(fn_cat, axis=1)
m = D.err == "FP"; D.loc[m, "cat"] = D[m].apply(fp_cat, axis=1)
D.drop(columns=["gb_text", "gs_quote", "calc_near", "ascites"]).to_csv(O + "per_case_trval.csv", index=False)
D[["study_id", "gb_text", "gs_quote"]].to_csv(O + "text_trval_private.csv", index=False)
# ---------- helpers ----------
def auc(d, s="N_U"):
    d = d.dropna(subset=[s])
    return roc_auc_score(d.y_gs, d[s]) if d.y_gs.nunique() == 2 else np.nan
def sens_spec(d, s="N_U", t=thr):
    p = d[s] >= t; return (p[d.y_gs == 1].mean(), 1 - p[d.y_gs == 0].mean())
OUT = {"thr_logit": thr, "thr_V4": thr4}
for sp in ["train", "val"]:
    d = D[D.split == sp]
    OUT[sp] = dict(n=len(d), pos=int(d.y_gs.sum()), auc_N=auc(d), auc_V4=auc(d, "V4"), sens_spec=sens_spec(d),
                   nFN=int((d.err == "FN").sum()), nFP=int((d.err == "FP").sum()),
                   sens_spec_V4=sens_spec(d, "V4", thr4))
print(json.dumps(OUT, indent=1, default=float))
# ---------- category shares ----------
rows = []
for sp in ["train", "val"]:
    d = D[D.split == sp]
    for e in ["FN", "FP"]:
        de = d[d.err == e]
        for c, n in de.cat.value_counts().items():
            rows.append(dict(split=sp, err=e, cat=c, n=n, share=n / len(de)))
CS = pd.DataFrame(rows); CS.to_csv(O + "category_shares.csv", index=False)
# ---------- subgroup table (overlapping flags) ----------
POSF = [("f_tiny", "微小／小結石"), ("f_large", "大結石（≥10 mm）"), ("f_single", "單顆"), ("f_multi", "多顆"), ("f_calc", "報告：鈣化／高密度"),
        ("f_noncalc", "報告：非鈣化／等密度"), ("f_layer", "分層"), ("f_too_small", "報告：微小高密度點／太小無法定性"), ("f_vis", "3 mm 可見 >100 HU"),
        ("f_distended", "膽囊擴張"), ("f_contracted", "膽囊收縮／塌陷／未擴張"), ("f_filled", "膽囊被結石充滿／收縮含石"),
        ("f_mask_missing", "膽囊遮罩缺失（<100 vox）"), ("f_mask_small", "膽囊遮罩偏小（100–299 vox）"), ("f_noncontrast", "非顯影"), ("f_arterial", "動脈期"),
        ("f_wall_thick", "膽囊炎／壁增厚"), ("f_adenomyo", "腺肌症"), ("f_polyp", "息肉"), ("f_porcelain", "瓷器膽囊／壁鈣化"),
        ("f_stent_air", "膽道支架／積氣"), ("f_chole", "同時標膽囊切除"), ("f_sludge", "合併膽泥"), ("f_vicarious", "顯影劑排泄／高密度膽汁"),
        ("f_calc_near", "肝／腎鈣化（報告）"), ("f_nondist", "膽囊未擴張（nondistended）"), ("f_cirrhosis", "肝硬化"), ("f_gb_mass", "膽囊／膽囊窩腫瘤"), ("f_ct_limited_neg", "報告 no radiopaque／no CT evidence"), ("f_gb_normal", "報告「Gallbladder: Normal」"), ("f_ascites", "腹水／膽囊周圍積液"), ("f_hist_us", "僅病史／超音波可見"), ("f_probyes", "probably_yes"),
        ("f_hedge_pos", "疑似措辭（含 probably_yes）")]
NEGF = [(k, n) for k, n in POSF if k not in ("f_tiny", "f_large", "f_single", "f_multi", "f_calc", "f_noncalc", "f_layer", "f_too_small", "f_filled", "f_probyes", "f_hedge_pos")] + \
       [("f_hd200", "膽囊內 >200 HU"), ("f_probno", "probably_no"), ("f_notment", "報告未提及膽囊結石（not_mentioned）")]
def boot_auc_ci(dpos, dneg, s="N_U", B=1000):
    a, b = dpos[s].values, dneg[s].values
    if len(a) < 30: return (np.nan, np.nan)
    out = []
    for _ in range(B):
        x = rng.choice(a, len(a)); y = rng.choice(b, len(b))
        out.append(roc_auc_score(np.r_[np.ones(len(x)), np.zeros(len(y))], np.r_[x, y]))
    return tuple(np.percentile(out, [2.5, 97.5]))
SG = []
for sp in ["train", "val"]:
    d = D[D.split == sp]; P, N = d[d.y_gs == 1], d[d.y_gs == 0]
    nFN, nFP = (d.err == "FN").sum(), (d.err == "FP").sum()
    for k, nm in POSF:
        g = P[P[k] == 1]
        if len(g) == 0: continue
        a = auc(pd.concat([g, N])); a4 = auc(pd.concat([g, N]), "V4")
        ci = boot_auc_ci(g, N, B=500) if sp == "val" else (np.nan, np.nan)
        SG.append(dict(split=sp, side="陽性", flag=k, name=nm, n=len(g), auc_N=a, ci_lo=ci[0], ci_hi=ci[1], auc_V4=a4,
                       miss_rate=(g.err == "FN").mean(), miss_rate_V4=((g.V4 < thr4)).mean(), share_err=(g.err == "FN").sum() / nFN))
    for k, nm in NEGF:
        g = N[N[k] == 1]
        if len(g) == 0: continue
        a = auc(pd.concat([P, g])); a4 = auc(pd.concat([P, g]), "V4")
        SG.append(dict(split=sp, side="陰性", flag=k, name=nm, n=len(g), auc_N=a, ci_lo=np.nan, ci_hi=np.nan, auc_V4=a4,
                       miss_rate=(g.err == "FP").mean(), miss_rate_V4=(g.V4 >= thr4).mean(), share_err=(g.err == "FP").sum() / nFP))
SG = pd.DataFrame(SG); SG.to_csv(O + "subgroups.csv", index=False)
# ---------- recoverable AUROC ----------
NOISE = {"f_ct_limited_neg", "f_probyes", "f_hedge_pos", "f_too_small", "f_hist_us", "f_chole", "f_probno", "f_notment", "f_sludge", "f_vicarious"}
def delta(d, k, side, B=1000):
    base = auc(d)
    if side == "陽性": rm = (d.y_gs == 1) & (d[k] == 1)
    else: rm = (d.y_gs == 0) & (d[k] == 1)
    dr = d[~rm]; rem = auc(dr) - base
    # parity fix: replace group scores with draws from same-class non-group cases
    same = d[(d.y_gs == (1 if side == "陽性" else 0)) & ~rm].N_U.values
    par = []
    for _ in range(100):
        dd = d.copy(); dd.loc[rm, "N_U"] = rng.choice(same, rm.sum()); par.append(auc(dd) - base)
    # bootstrap for removal delta (stratified)
    bs = []
    P_idx, N_idx = np.where(d.y_gs == 1)[0], np.where(d.y_gs == 0)[0]
    y, s, r = d.y_gs.values, d.N_U.values, rm.values
    for _ in range(B):
        ix = np.r_[rng.choice(P_idx, len(P_idx)), rng.choice(N_idx, len(N_idx))]
        a0 = roc_auc_score(y[ix], s[ix]); k2 = ix[~r[ix]]; bs.append(roc_auc_score(y[k2], s[k2]) - a0)
    return rm.sum(), rem, np.percentile(bs, 2.5), np.percentile(bs, 97.5), np.mean(par)
RC = []
CAUSES = [("f_mask_missing", "陽性", "膽囊遮罩缺失（陽性）"), ("f_contracted", "陽性", "膽囊收縮／塌陷（陽性）"), ("f_filled", "陽性", "膽囊被結石充滿（陽性）"),
          ("nonvis_mask", "陽性", "3 mm 無高密度（遮罩存在，陽性）"), ("f_tiny", "陽性", "微小／小結石（陽性）"), ("f_noncalc", "陽性", "報告非鈣化（陽性）"),
          ("f_hedge_pos", "陽性", "疑似措辭陽性（含 probably_yes）"), ("f_too_small", "陽性", "太小無法定性（陽性）"), ("f_hist_us", "陽性", "僅病史／超音波（陽性）"),
          ("f_chole", "陽性", "膽囊切除標籤衝突（陽性）"), ("f_chole", "陰性", "膽囊切除陰性"), ("chole_mask", "陰性", "膽囊切除陰性但遮罩仍在"),
          ("f_probno", "陰性", "probably_no 陰性"), ("f_notment", "陰性", "not_mentioned 陰性"), ("f_sludge", "陰性", "膽泥（膽結石陰性）"),
          ("f_vicarious", "陰性", "顯影劑排泄／高密度膽汁（陰性）"), ("f_hd200", "陰性", "膽囊內 >200 HU 陰性（疑漏報結石）"), ("f_wall_thick", "陰性", "膽囊炎／壁增厚（陰性）"),
          ("f_porcelain", "陰性", "瓷器膽囊／壁鈣化（陰性）"), ("f_adenomyo", "陰性", "腺肌症（陰性）"), ("f_polyp", "陰性", "息肉（陰性）"), ("f_stent_air", "陰性", "支架／膽道積氣（陰性）"),
          ("f_calc_near", "陰性", "肝／腎鈣化（陰性）"), ("f_cirrhosis", "陰性", "肝硬化（陰性）"), ("f_gb_mass", "陰性", "膽囊／膽囊窩腫瘤（陰性）"),
          ("f_ct_limited_neg", "陰性", "報告只否定高密度結石（陰性）"), ("f_nondist", "陰性", "膽囊未擴張（陰性）"), ("f_contracted", "陰性", "膽囊收縮／塌陷（陰性）"), ("f_mask_missing", "陰性", "遮罩缺失（陰性）"), ("f_ascites", "陰性", "腹水（陰性）"), ("f_noncontrast", "陽性", "非顯影（陽性）"), ("f_noncontrast", "陰性", "非顯影（陰性）")]
D["nonvis_mask"] = ((D.f_vis == 0) & (D.f_mask_missing == 0)).astype(int)
D["chole_mask"] = ((D.f_chole == 1) & (D.f_mask_missing == 0)).astype(int)
for sp in ["train", "val"]:
    d = D[D.split == sp].reset_index(drop=True)
    for k, side, nm in CAUSES:
        n, rem, lo, hi, par = delta(d, k, side, B=500)
        kind = "標籤雜訊" if (k in NOISE or k == "chole_mask" and False) else "模型能力"
        if k in ("f_hd200",): kind = "混合（可能漏報）"
        if k in ("chole_mask",): kind = "混合（誤分割／標籤）"
        RC.append(dict(split=sp, cause=nm, flag=k, side=side, n=n, kind=kind, d_remove=rem, lo=lo, hi=hi, d_parity=par))
RC = pd.DataFrame(RC); RC.to_csv(O + "recoverable.csv", index=False)
# combined
for sp in ["train", "val"]:
    d = D[D.split == sp]; base = auc(d)
    noise_rm = ((d.y_gs == 1) & ((d.f_hedge_pos + d.f_too_small + d.f_hist_us + d.f_chole) > 0)) | ((d.y_gs == 0) & ((d.f_probno + d.f_notment + d.f_sludge) > 0))
    OUT[sp]["d_all_noise_removed"] = auc(d[~noise_rm]) - base; OUT[sp]["n_noise_removed"] = int(noise_rm.sum())
    cap = (d.y_gs == 1) & ((d.f_mask_missing == 1) | (d.nonvis_mask == 1))
    OUT[sp]["auc_visible_only"] = auc(d[~cap]); OUT[sp]["d_visible_only"] = auc(d[~cap]) - base
# ---------- HU check on FN ----------
HU = []
for sp in ["train", "val"]:
    d = D[(D.split == sp) & (D.y_gs == 1) & (D.f_mask_missing == 0)]
    for e in ["FN", "TP"]:
        g = d[d.err == e]
        HU.append(dict(split=sp, grp=e, n=len(g), gb_max_med=g.gb_max.median(), gb_max_q25=g.gb_max.quantile(.25), gb_max_q75=g.gb_max.quantile(.75),
                       p_max_lt130=(g.gb_max < 130).mean(), p_any130=(g.gb_n130 > 0).mean(), p_any200=(g.gb_n200 > 0).mean(),
                       p_blob100=(g.gb_maxblob100 >= 2).mean(), n200_med=g.gb_n200.median(), vox_med=g.gb_vox.median(),
                       p_tiny=g.f_tiny.mean(), p_noncalc=g.f_noncalc.mean(), p_calc=g.f_calc.mean(), p_contracted=g.f_contracted.mean()))
pd.DataFrame(HU).to_csv(O + "hu_fn_tp.csv", index=False)
# ---------- old (V4) vs N: FN transitions on val/train ----------
TRN = []
for sp in ["train", "val"]:
    d = D[(D.split == sp) & (D.y_gs == 1)]
    old = d.V4 < thr4; new = d.N_U < thr
    TRN.append(dict(split=sp, fn_both=int((old & new).sum()), rescued=int((old & ~new).sum()), new_fn=int((~old & new).sum()), fn_old=int(old.sum()), fn_new=int(new.sum())))
    dn = D[(D.split == sp) & (D.y_gs == 0)]
    o2 = dn.V4 >= thr4; n2 = dn.N_U >= thr
    TRN[-1].update(fp_old=int(o2.sum()), fp_new=int(n2.sum()), fp_both=int((o2 & n2).sum()))
OUT["transitions"] = TRN
json.dump(OUT, open(O + "summary.json", "w"), indent=1, default=float)
print(CS.to_string()); print(pd.DataFrame(TRN))
print(RC[RC.split == "val"].sort_values("d_remove", ascending=False).to_string())
print(pd.DataFrame(HU).to_string())
