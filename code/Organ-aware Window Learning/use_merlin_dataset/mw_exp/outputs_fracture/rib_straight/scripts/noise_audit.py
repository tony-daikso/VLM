"""Fracture label-noise audit — TRAIN ONLY (agent rib_straight, 11-08). Official labels are NOT changed.
Inputs: split.csv (test ids dropped before anything else), labels_disease14_all.jsonl (only train lines parsed),
merlin_disease14/reports.json (train reports only, rest discarded), final_ft/rp_both_aa/scores.npz (train = stacker
nested-CV OOF), site OOFs (rib_v3 lev28, spine15, pelvis15 final_m400).
Outputs (outputs_fracture/rib_straight/label_audit/): audit_fn.csv (label 0, top OOF), audit_fp.csv (label 1, bottom OOF),
summary.json, report.md (pattern counts + AUROC sensitivity to excluding uncertain categories)."""
import json, re, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
H = os.path.expanduser("~/Desktop/Tony/VLM"); MW = H + "/OWL/mw_exp"; FR = MW + "/outputs_fracture"
OUT = FR + "/rib_straight/label_audit"; os.makedirs(OUT, exist_ok=True)
spl = pd.read_csv(H + "/OWL/data/split.csv"); TR = set(spl.loc[spl.split == "train", "study_id"])   # train only
rx = re.compile(r'"study_id"\s*:\s*"([^"]+)"'); L = {}
for line in open(H + "/OWL/data/labels_disease14_all.jsonl"):
    m = rx.search(line)
    if not m or m.group(1) not in TR: continue
    r = json.loads(line); f = r["answer"]["fracture"]; d = f.get("detail", {}) or {}
    L[r["study_id"]] = dict(cat=f["category"], quotes=" | ".join(f.get("quotes", []) or []), why=f.get("why", ""),
                            sites=";".join(d.get("sites", []) or []), acute=d.get("acute", ""), comp=d.get("vertebral_compression", ""))
L = pd.DataFrame.from_dict(L, orient="index")
z = np.load(FR + "/final_ft/rp_both_aa/scores.npz", allow_pickle=True)
assert set(z["split"]) <= {"train", "val"}
t = z["split"] == "train"; S = pd.DataFrame({"y": z["y"][t], "oof": z["selected"][t]}, index=z["ids"][t].astype(str))
S = S[S.index.isin(TR)].join(L, how="inner")
for nm, p in (("rib", FR + "/rib_v3/cand/lev28/oof_train.csv"), ("spine", FR + "/spine15_ft/oof_train.csv"),
              ("pel", FR + "/pelvis15/final_m400/oof_train.csv")):
    d = pd.read_csv(p).set_index("study_id"); c = "ft" if "ft" in d else d.columns[-1]
    s = d[c]; S["z_" + nm] = ((s - s.mean()) / s.std()).reindex(S.index)
R = json.load(open(H + "/merlin_disease14/reports.json")); REP = {k: R[k]["report"] for k in S.index if k in R}; del R
S["rank"] = S.oof.rank(pct=True)
base = auc(S.y, S.oof); npos = int(S.y.sum())
print(f"train N {len(S)} pos {npos} OOF AUROC {base:.4f}", flush=True)
KW = re.compile(r"[^.]*\b(fractur\w*|compression|wedg\w*|height loss|loss of height|deformit\w*|schmorl\w*|endplate|callus|healed|"
                r"vertebroplast\w*|kyphoplast\w*|cement|fixation|hardware|orif|screw\w*|rod\w*|plate\w*|spondylolys\w*|pars|fx)\b[^.]*\.", re.I)
PAT = {  # pattern name: regex on the fracture-related report sentences (+ quotes)
    "壓迫變形/高度減少（未稱骨折）": r"compression deformit|height loss|loss of height|wedg|endplate (deformit|depression)|biconcav",
    "Schmorl 結節": r"schmorl",
    "陳舊/癒合骨折": r"healed|old|remote|chronic|callus|prior fracture",
    "術後/骨水泥/內固定": r"vertebroplast|kyphoplast|cement|fixation|hardware|orif|screw|rod|plate|arthroplast|prosthe",
    "不確定措辭（possible/cannot exclude）": r"possibl|cannot be excluded|cannot exclude|may represent|questionable|suspicious for|equivocal|versus|vs\.|nondisplaced|subtle",
    "FOV/ROI 外部位（上肢、胸骨、鎖骨、肩胛、頭頸、股骨幹以下）": r"humer|clavic|scapul|stern|skull|nasal|orbit|mandib|cervical|femoral shaft|tibia|fibula|radius|ulna|wrist|hand|foot",
    "橫突/棘突/後方附件": r"transverse process|spinous process|pars|spondylolys|posterior element|facet",
    "肋骨": r"\brib",
    "骨盆/薦椎/髖": r"pelvi|sacr|iliac|ischi|pubic|ramus|rami|acetab|femoral neck|hip|coccy",
    "椎體": r"vertebra|\bT\d{1,2}\b|\bL\d\b|spine|lumbar|thoracic",
}


def sents(cid):
    r = REP.get(cid, ""); return " ".join(m.group(0).strip() for m in KW.finditer(r))[:1200]


def tag(df):
    df = df.copy(); df["report_fx_sentences"] = [sents(c) for c in df.index]
    txt = (df.report_fx_sentences + " " + df.quotes.fillna("")).str.lower()
    for k, p in PAT.items(): df[k] = txt.str.contains(p, regex=True)
    df["報告未提及任何骨折相關字"] = df.report_fx_sentences.str.len() == 0
    return df


neg = S[S.y == 0]; pos = S[S.y == 1]
thr = neg.oof.quantile(0.98)
FN = tag(neg[neg.oof >= thr].sort_values("oof", ascending=False))            # label 0, top 2 % of negatives
FP = tag(pos[pos.oof <= pos.oof.quantile(0.10)].sort_values("oof"))          # label 1, bottom 10 % of positives
cols = ["y", "oof", "rank", "cat", "sites", "acute", "comp", "z_rib", "z_spine", "z_pel", "why", "quotes", "report_fx_sentences"] + list(PAT) + ["報告未提及任何骨折相關字"]
FN[cols].to_csv(OUT + "/audit_fn.csv"); FP[cols].to_csv(OUT + "/audit_fp.csv")
cnt = lambda d: {k: int(d[k].sum()) for k in list(PAT) + ["報告未提及任何骨折相關字"]}
# AUROC sensitivity to uncertain label categories (train OOF; no retraining)
sens = {"全部": (base, len(S))}
for nm, m in (("去掉 probably_yes", S.cat != "probably_yes"), ("去掉 probably_no", S.cat != "probably_no"),
              ("去掉 probably_*", ~S.cat.isin(["probably_yes", "probably_no"])), ("去掉 not_mentioned 以外的陰性", ~S.cat.isin(["no", "probably_no"]) | (S.y == 1))):
    sens[nm] = (float(auc(S.y[m], S.oof[m])), int(m.sum()))
# upper bound if the FN candidates that carry an image-visible fracture-equivalent cue were relabelled positive
cue = FN[["壓迫變形/高度減少（未稱骨折）", "陳舊/癒合骨折", "術後/骨水泥/內固定"]].any(axis=1)
y2 = S.y.copy(); y2.loc[cue[cue].index] = 1
fpx = FP[FP["FOV/ROI 外部位（上肢、胸骨、鎖骨、肩胛、頭頸、股骨幹以下）"] | FP["橫突/棘突/後方附件"]].index
m3 = ~S.index.isin(fpx)
summ = dict(n_train=len(S), n_pos=npos, oof_auroc=base, fn_threshold_oof=float(thr), n_fn=len(FN), n_fp=len(FP),
            fn_category=FN.cat.value_counts().to_dict(), fp_category=FP.cat.value_counts().to_dict(),
            all_category=S.cat.value_counts().to_dict(), fn_patterns=cnt(FN), fp_patterns=cnt(FP),
            fp_sites=FP.sites.value_counts().head(15).to_dict(), auroc_sensitivity=sens,
            relabel_fn_cue_auroc=float(auc(y2, S.oof)), n_fn_cue=int(cue.sum()),
            drop_fp_outside_auroc=float(auc(S.y[m3], S.oof[m3])), n_fp_outside=int(len(fpx)),
            pos_by_cat_auroc={c: float(auc(np.r_[np.ones((S.cat == c).sum()), np.zeros((S.y == 0).sum())],
                                           np.r_[S.oof[S.cat == c], S.oof[S.y == 0]])) for c in ("yes", "probably_yes") if (S.cat == c).any()},
            neg_by_cat_auroc={c: float(auc(np.r_[np.ones(npos), np.zeros((S.cat == c).sum())], np.r_[S.oof[S.y == 1], S.oof[S.cat == c]]))
                              for c in ("no", "probably_no", "not_mentioned") if (S.cat == c).any()})
json.dump(summ, open(OUT + "/summary.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))
