"""Complete the report-gen OWL score table, step 3: stack-level scoring of the missing cells + Platt + checks.
Inputs: heads of the missing studies from complete_heads.py (heads/out/<job>.csv, full-train refit) or the STORED head files
(multi-task heads whose stored train OOF / val already cover the unclear studies), unit heads from complete_units.py
(full-train refit), and the formal stack of each finding:
  integrated_best stack (hs, absent GB, spleno, AAA, SBO)   : saved models/<fd>.joblib (= full-train refit), ib_apply rules
  region-unit stacks (hydro, free air, ascites, athero)     : refit exactly as ru_stack/nu_stack (bstack_apply recipe) on the
                                                              stored train OOF, checked against the stored val scores
  osteopenia 5win+v15 / biliary BD5                         : ORIGINAL ov_run.py / bd_stack.py in a sandbox with the
                                                              missing studies appended as split 'val' (dummy y 0)
  appendicitis RLQ REF＋U_rlq_cs                            : assemble_scores.appendicitis() LR refit (C 0.1, balanced)
  gallstones N_U                                            : saved region_units/gallstones/stack/stack_N_U.joblib
  fracture a6x10w2_tv7                                      : saved full-train site heads (models/fracture_a6x10w2_tv7/
                                                              integ/stacker.pkl) on the stored member inputs (24 labelled
                                                              studies dropped by dropna rules; unclear studies have no member
                                                              scores at all -> left empty)
Raw score s -> p = sigmoid(a s + b) with out/calib.json (calibrate.py). Old cells are copied unchanged.
Checks: (1) every cell present in the old file is identical in the new file; (2) val studies that already had a score are
re-scored through the same path (refit heads / unit heads where they were recomputed) and compared with the stored raw score.
Test is never read. usage (A6000): complete_scores.py osteo | biliary | assemble"""
import os, sys, json, glob, pickle, shutil, subprocess, joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); MW = H + "mw_exp/"; OF = MW + "outputs_final/"; IB = OF + "integrated_best/"
RU = OF + "region_units/"; C = H + "report_gen/oof/complete/"; OUTD = H + "report_gen/oof/out/"
PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")
sys.argv_save = list(sys.argv); sys.argv = [sys.argv[0], IB]; sys.path.insert(0, IB + "scripts")
import ib_stack as ibs
from ib_core import CANDS, Prep, score
sys.argv = sys.argv_save
M = pd.read_csv(C + "missing_cells.csv")
SPL = pd.read_csv(OUTD + "scores_trval.csv")[["study_id", "split"]]; assert set(SPL.split) <= {"train", "val"}
RAW0 = pd.read_csv(OUTD + "scores_trval.csv")


def miss(fd, split=None):
    m = M[M.disease == fd]; return list(m.study_id if split is None else m[m.split == split].study_id)


def have_val(fd):
    """val studies that already have a score (re-scored for the reproduction check)"""
    v = RAW0[(RAW0.split == "val") & RAW0[fd].notna()]; return v[["study_id", fd]].rename(columns={fd: "stored"})


# ------------------------------------------------------------------ heads
STORED_AN = {"absent_gb", "aaa", "bowel_obstruction", "free_air", "atherosclerosis", "gallstones"}
NEWJOB = {("steatosis", "E3_late"): ("hs", "analysis_steatosis/scores_val_descriptive.csv"),
          ("steatosis", "E3_late_concat_diff"): ("hs_cd", "analysis_steatosis/scores_val_concat_diff.csv"),
          ("splenomegaly", "E3_late"): ("sp", "analysis_splenomegaly/scores_val.csv"),
          ("hydronephrosis", "E3_late"): ("hn", "analysis_hydronephrosis/scores_val_descriptive.csv"),
          ("hydronephrosis", "E3_late_max"): ("hn", "analysis_hydronephrosis/scores_val_descriptive.csv"),
          ("ascites", "E3_late"): ("asc", "analysis_ascites/scores_val_descriptive.csv"),
          ("ascites", "E3_late_all12"): ("asc", "analysis_ascites/scores_val_descriptive.csv")}


def head(an, col, ids, verify=False):
    """head score of studies `ids`. stored heads: train rows from the stored OOF (the unclear studies are OOF rows of these
    multi-task heads), val rows from the stored full-train val file. recomputed heads: complete_heads.py output
    (missing studies) / its refit val file (verify=True: val studies that already had a score)."""
    if an in STORED_AN:
        o = pd.read_csv(OF + f"analysis_{an}/oof_train.csv")[["study_id", col]]
        v = pd.read_csv(OF + f"analysis_{an}/scores_val_descriptive.csv")[["study_id", col]]
        s = pd.concat([o, v]).drop_duplicates("study_id").set_index("study_id")[col]
    else:
        j, fv = NEWJOB[(an, col)]
        f = C + f"heads/R/{fv}" if verify else C + f"heads/out/{j}.csv"
        s = pd.read_csv(f).drop_duplicates("study_id").set_index("study_id")[col]
    return s.reindex(ids)


def ib_feats(c):
    """ib_stack.load_feats with the explicit file locations of ib_apply / infer_best (analysis dir or integrated_best/features)"""
    parts = []
    for f in c["feats"]:
        d = pd.read_csv(IB + "features/" + f[1:] if f.startswith("+") else OF + f"analysis_{c['an']}/{f}")
        if "err" in d: d = d[d.err.fillna(0) == 0]
        parts.append(d)
    blocks = {}
    for d in parts: blocks.setdefault(tuple(sorted(d.columns)), []).append(d)
    F = None
    for ds in blocks.values():
        b = pd.concat(ds).drop_duplicates("study_id", keep="first")
        F = b if F is None else F.merge(b, on="study_id", how="outer", suffixes=("", "_dup"))
    F = F[[x for x in F.columns if not x.endswith("_dup")]]
    F = F[["study_id"] + [x for x in F.columns if x not in ibs.DROP and x not in ibs.META_COLS and pd.api.types.is_numeric_dtype(F[x])]]
    return ibs.derive(F, c)


IBH = {"hepatic_steatosis": ["E3_late", "E3_late_concat_diff"], "splenomegaly": ["E3_late"], "hydronephrosis": ["E3_late", "E3_late_max"],
       "ascites": ["E3_late", "E3_late_all12"], "surgically_absent_gallbladder": ["E3_late"], "abdominal_aortic_aneurysm": ["E3_late"],
       "bowel_obstruction": ["E3_late", "E3_late_st"], "free_air": ["E3_late"], "atherosclerosis": ["E3_late"]}


def ref_frame(fd, ids, verify=False):
    c = ibs.CFG[fd]; F = ib_feats(c)
    X = pd.DataFrame({"study_id": ids})
    for h in IBH[fd]: X[h] = head(c["an"], h, ids, verify).values
    return X.merge(F, on="study_id", how="left")


def ib_tier(fd):
    Mo = joblib.load(IB + f"models/{fd}.joblib")
    out = {}
    for tag, ids, ver in (("fill", miss(fd), False), ("verify", list(have_val(fd).study_id), True)):
        X = ref_frame(fd, ids, ver); assert not [x for x in Mo["cols"] if x not in X.columns]
        out[tag] = pd.Series(score(Mo["model"], Mo["prep"].transform(X[Mo["cols"]]).values), index=X.study_id)
    return out, f"integrated_best/models/{fd}.joblib（{Mo['featset']} | {Mo['cand']}，全 train refit）"


# ------------------------------------------------------------------ region-unit stacks (bstack_apply recipe, refit)
RUS = {"hydronephrosis": ("hydronephrosis", "hn", "REF＋全部單元頭", None, None),
       "free_air": ("free_air", "fa", "REF＋全部單元頭", ["U", "U_org"], None),     # stored stack predates the U_noair column
       "ascites": ("ascites", "asc", "REF＋U_org", None, None),
       "atherosclerosis": ("atherosclerosis", "ath", "REF＋全部單元頭＋HU", None, RU + "atherosclerosis/hu_feats.csv")}


def ru_tier(fd):
    unit, code, SEL, uh_only, HUF = RUS[fd]; UD = RU + unit + "/"
    c = ibs.CFG[fd]; A = RU + f"ibroot/an/{c['an']}/"
    lab = pd.read_csv(IB + "work/base/labels14_trval.csv")[["study_id", "split", fd]].rename(columns={fd: "y"})
    oof = pd.read_csv(A + "oof_train.csv"); F = ib_feats(c)
    uo = pd.read_csv(UD + "out/oof_train.csv").drop(columns=["fold"])
    uv = pd.read_csv(C + f"units/{code}/out/val_scores.csv")                       # refit unit heads: val + missing studies
    UH0 = [x for x in uo.columns if x != "study_id" and (uh_only is None or x in uh_only)]
    ren = {x: "ru_" + x for x in UH0}; UH = list(ren.values())
    uo = uo[["study_id"] + UH0].rename(columns=ren); uv = uv[["study_id"] + UH0].rename(columns=ren)
    T = lab[lab.split == "train"].merge(oof, on="study_id").merge(F, on="study_id", how="left").merge(uo, on="study_id", how="left")
    T = T[T.y.notna()].reset_index(drop=True)
    fr = {"fill": ref_frame(fd, miss(fd)), "verify": ref_frame(fd, list(have_val(fd).study_id), True)}
    fr = {k: v.merge(uv, on="study_id", how="left") for k, v in fr.items()}
    HC = []
    if HUF:
        hf = pd.read_csv(HUF).drop_duplicates("study_id"); HC = [x for x in hf.columns if x != "study_id"]
        T = T.merge(hf, on="study_id", how="left"); fr = {k: v.merge(hf, on="study_id", how="left") for k, v in fr.items()}
    for h in UH:
        md = T[h].median(); T[h] = T[h].fillna(md)
        for D in fr.values(): D[h] = D[h].fillna(md)
    for h in HC:
        lg = h.endswith(("_n300", "_n130"))
        for D in [T] + list(fr.values()): D[h] = np.log1p(D[h].clip(lower=0)) if lg else D[h]
        md = T[h].median()
        for D in [T] + list(fr.values()): D[h] = D[h].fillna(md)
    RC = json.load(open(IB + f"models/{fd}_features.json"))["columns"]
    r = pd.read_csv(UD + "stack/results.csv"); cn = r[r.model == SEL].iloc[0].cand
    cols = {"REF＋全部單元頭": RC + UH, "REF＋全部單元頭＋HU": RC + UH + HC}.get(SEL) or RC + ["ru_" + SEL.split("＋", 1)[1]]
    pr = Prep().fit(T[cols]); m = CANDS[cn]().fit(pr.transform(T[cols]).values, T.y.values.astype(int))
    os.makedirs(C + "models", exist_ok=True)
    joblib.dump(dict(finding=fd, model=m, prep=pr, cols=cols, cand=cn, selected=SEL), C + f"models/{fd}_ru_stack.joblib")
    out = {k: pd.Series(score(m, pr.transform(D[cols]).values), index=D.study_id) for k, D in fr.items()}
    return out, f"region_units/{unit} {SEL}（{cn}，全 train refit；單元頭 = 全 train refit ru/nu probe）"


# ------------------------------------------------------------------ appendicitis (assemble_scores.appendicitis rule)
def appendicitis():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    U = OF + "appendicitis_rlq_unit/"
    oof = pd.read_csv(U + "out/oof_train.csv"); new = pd.read_csv(C + "units/rlq/out/val_scores.csv")
    alab = pd.read_csv(OF + "appendicitis_split/labels_trval_split.csv")
    hu = pd.concat([pd.read_csv(OF + "analysis_appendicitis/hu_a6.csv"), pd.read_csv(OF + "analysis_appendicitis/hu_ct.csv")]).drop_duplicates("study_id")
    LOCAL = [c for c in hu.columns if c.startswith("dlr_") or c.startswith("dglob_")] + \
            [f"rlq_r{r}_{f}" for r in (15, 30) for f in ("fat_med", "fat_p75", "strand_frac", "fluid_frac", "soft_frac", "fat_frac")] + ["rlq_wall_p50", "rlq_wall_p90"]
    hu = hu[["study_id"] + LOCAL]
    T = oof.merge(alab, on="study_id", how="left").merge(hu, on="study_id", how="left")
    tg = "y_ap"; Tm = T[T[tg].notna() & T[f"{tg}__E3_late"].notna()].reset_index(drop=True)
    cols = [f"{tg}__E3_late", f"{tg}__E3cs", f"{tg}__U_rlq_cs"] + LOCAL
    X = Tm[cols]; med = X.median(); X = X.fillna(med)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")).fit(X, Tm[tg].values.astype(int))
    out = {}
    for k, ids in (("fill", miss("appendicitis")), ("verify", list(have_val("appendicitis").study_id))):
        D = pd.DataFrame({"study_id": ids}).merge(new, on="study_id", how="left").merge(hu, on="study_id", how="left")
        assert D[cols[:3]].notna().all().all()
        out[k] = pd.Series(lr.decision_function(D[cols].fillna(med)), index=D.study_id)
    return out, "appendicitis_rlq_unit REF＋U_rlq_cs LR（C 0.1，全 train refit；RLQ 頭 = 全 train refit rlq_probe）"


# ------------------------------------------------------------------ gallstones N_U (cg_gs.stack recipe on Merlin inputs)
def gallstones():
    G = OF + "gallstones_improve/"; Mo = joblib.load(RU + "gallstones/stack/stack_N_U.joblib")
    sys.argv = [sys.argv[0], C + "_nu_dummy", "x", "--gbn", RU + "gallstones/out"]
    src = open(RU + "scripts/nu_gs_stack.py").read(); src = src[:src.index("# ---------------- data")]
    g = {}; exec(compile(src, "nu_gs_stack_defs", "exec"), g)
    g["EXTRA_COLS"].update({"gbx": ["gbx_r0", "gbx_r1", "gbx_cat"]})
    g["G1"] = [c for c in pd.read_csv(G + "gs1_feat3mm.csv", nrows=1).columns if c.startswith("g1_")]
    hu0 = pd.concat([pd.read_csv(OF + f"analysis_gallstones/{f}") for f in ("hu_a6.csv", "hu_ct.csv")]).drop_duplicates("study_id")
    ag0 = pd.concat([pd.read_csv(OF + f"analysis_absent_gb/{f}") for f in ("feat_a6.csv", "feat_ct.csv")]).drop_duplicates("study_id").drop(columns=["gb_vox"])
    ag0 = ag0.rename(columns={c: "ag_" + c for c in ag0.columns if c != "study_id"}); g10 = pd.read_csv(G + "gs1_feat3mm.csv").drop_duplicates("study_id")
    st = lambda f, c: pd.concat([pd.read_csv(f[0])[["study_id", c]], pd.read_csv(f[1])[["study_id", c]]]).drop_duplicates("study_id")
    ch = st((OF + "analysis_absent_gb/oof_train.csv", OF + "analysis_absent_gb/scores_val_descriptive.csv"), "E3_late").rename(columns={"E3_late": "chole_logit"})
    e3 = st((OF + "analysis_gallstones/oof_train.csv", OF + "analysis_gallstones/scores_val_descriptive.csv"), "E3_late")
    g3 = st((G + "gs3b/gs3b_oof_train.csv", G + "gs3b/gs3b_val.csv"), "g3b_livfb")
    gx = pd.read_csv(C + "units/gbx/out/gbx_val.csv")[["study_id", "gbx_r0", "gbx_r1", "gbx_cat"]]           # refit (val + missing)
    gn = pd.read_csv(C + "units/gs/out/val_scores.csv")[["study_id", "U"]].rename(columns={"U": "gbn_U"})    # refit (val + missing)
    out = {}
    for k, ids in (("fill", miss("gallstones")), ("verify", list(have_val("gallstones").study_id))):
        d = pd.DataFrame({"study_id": ids})
        for t in (hu0, ag0, g10, ch, e3, g3, gx, gn): d = d.merge(t, on="study_id", how="left")
        assert d[["chole_logit", "E3_late", "g3b_livfb", "gbx_r0", "gbn_U"]].notna().all().all(), k
        X = g["feats"](d.set_index("study_id"), Mo["sets"])[Mo["columns"]].fillna(Mo["medians"])
        out[k] = pd.Series(g["score"](Mo["model"], X.values), index=X.index)
    return out, "region_units/gallstones stack_N_U.joblib（G5c＋膽囊單元頭，全 train refit；gbx / gbn_U = 全 train refit probe）"


# ------------------------------------------------------------------ osteopenia v15 (ov_run.py in a sandbox)
def osteo():
    A = OF + "analysis_osteopenia/"; D = C + "osteo/data/"; O = C + "osteo/out/"; os.makedirs(D, exist_ok=True)
    t = D + "ov_emb15_L1L4_trval.npz"          # ov_build.py output on Merlin train/val (built by cg_osteo_v15.py, read only)
    if not os.path.lexists(t): os.symlink(H + "cg10k_best/osteo_v15/data/merlin_ov.npz", t)
    lab = pd.read_csv(A + "labels_trval.csv"); assert set(lab.split) <= {"train", "val"}
    u = lab.study_id.isin(set(miss("osteopenia"))); assert lab.loc[u, "y_op"].isna().all()
    lab.loc[u, "split"] = "val"; lab.loc[u, "y_op"] = 0.0; lab.to_csv(D + "labels_trval.csv", index=False)   # scored only, never fitted
    for f in (A + "oof_train.csv", A + "bundle_trval.csv", IB + "scores/osteopenia_oof.csv", IB + "scores/osteopenia_val.csv", IB + "scripts/ib_core.py"):
        shutil.copy(f, D)
    sv = pd.read_csv(A + "scores_val_descriptive.csv"); hx = pd.read_csv(C + "heads/out/op.csv")
    pd.concat([sv, hx[[c for c in sv.columns if c in hx.columns]]], ignore_index=True).to_csv(D + "scores_val_descriptive.csv", index=False)
    # geom15 HU only enters the HU stacks, which are not run here: NaN placeholder (the geom15 file is not read)
    pd.DataFrame([(i, l, np.nan) for i in lab.study_id for l in ("L1", "L2", "L3", "L4")], columns=["id", "level", "hu_body"]).to_csv(D + "geom15_levels.csv", index=False)
    code = open(OF + "osteopenia_v15/scripts/ov_run.py").read()
    rep = [('base["v15_LR_mean"] = cv_lr(MT, MV, Cs, "LR mean L1-L4 (3 win)")\n', ""),
           ('base["v15_LR_mean_base"] = cv_lr(MT[:, :448], MV[:, :448], Cs, "LR mean L1-L4 (baseline win)")\n', ""),
           ('base["v15_LR_meanmax"] = cv_lr(np.c_[MT, MmT], np.c_[MV, MmV], Cs, "LR mean+max-pool L1-L4")\n', ""),
           ('for mode in ["mean", "max", "attn"]:', "for mode in []:"),
           ("\nres = {}\n", '\nstacks = {k: v for k, v in stacks.items() if k == "5win+v15"}\nres = {}\n'),
           ("# ---------------------------------------------------------------- evaluation", "sys.exit(0)\n# evaluation skipped")]
    for a, b in rep:
        assert code.count(a) == 1, a; code = code.replace(a, b)
    # only the selected v15 variant (v15_LR_levelcat, deterministic) and the selected stack (5win+v15) are computed
    open(C + "osteo/patched_ov_run.py", "w").write(code)
    r = subprocess.run(["nice", "-n", "19", PY, C + "osteo/patched_ov_run.py", D, O], stdout=open(C + "osteo/run.log", "w"),
                       stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="6", CUDA_VISIBLE_DEVICES=""))
    print("ov_run exit", r.returncode)


def osteo_scores():
    s = pd.read_csv(C + "osteo/out/stack_val.csv").set_index("study_id")["5win+v15"]
    return {"fill": s.reindex(miss("osteopenia")), "verify": s.reindex(have_val("osteopenia").study_id)}, \
        "osteopenia_v15 5win+v15（LR_C0.1 on 6 window 頭＋v15_LR_levelcat，全 train refit；window 頭 = 全 train refit op_oof）"


# ------------------------------------------------------------------ biliary BD5 (bd_stack.py in a sandbox)
COLB = "cat4+E3_late+E0+porta_emb+HU+tube+porta|gbm"


def biliary():
    BI = OF + "biliary_improve/"; W = C + "biliary/work/"; O = C + "biliary/out/"; os.makedirs(W, exist_ok=True)
    for f in glob.glob(BI + "work/*.csv"): shutil.copy(f, W)
    ids = miss("biliary_ductal_dilation")
    lab = pd.read_csv(W + "labels_trval.csv"); u = lab.study_id.isin(set(ids)); assert lab.loc[u, "y_bdd"].isna().all()
    lab.loc[u, "split"] = "val"; lab.loc[u, "y_bdd"] = 0.0; lab.to_csv(W + "labels_trval.csv", index=False)
    hd = pd.read_csv(C + "heads/out/bd.csv"); x = pd.read_csv(W + "scores_val_descriptive.csv")
    hd = hd[~hd.study_id.isin(x.study_id)]
    pd.concat([x, hd[[c for c in hd.columns if c in x.columns]]], ignore_index=True).to_csv(W + "scores_val_descriptive.csv", index=False)
    rec = pickle.load(open(C + "units/bd5/x_bd5.pkl", "rb")); b5 = pd.DataFrame(rec).reset_index().rename(columns={"index": "study_id"})
    x = pd.read_csv(W + "bd5_val.csv"); b5 = b5[~b5.study_id.isin(x.study_id)]
    pd.concat([x, b5[[c for c in b5.columns if c in x.columns]]], ignore_index=True).to_csv(W + "bd5_val.csv", index=False)
    os.makedirs(O, exist_ok=True)
    r = subprocess.run(["nice", "-n", "19", PY, BI + "scripts/bd_stack.py", W, O], stdout=open(C + "biliary/stack.log", "w"), stderr=subprocess.STDOUT,
                       env=dict(os.environ, ONLY="cat4+E3_late+E0+porta_emb+HU+tube+porta", OMP_NUM_THREADS="6"))
    print("bd_stack exit", r.returncode)


def biliary_scores():
    s = pd.read_csv(C + "biliary/out/stack_val.csv").set_index("study_id")[COLB]
    return {"fill": s.reindex(miss("biliary_ductal_dilation")), "verify": s.reindex(have_val("biliary_ductal_dilation").study_id)}, \
        "biliary_improve BD5 stack（cat4+E3_late+E0+porta_emb+HU+tube+porta | gbm，全 train refit；頭 = 全 train refit bd_oof / bd5_probe）"


# ------------------------------------------------------------------ fracture: saved full-train site heads, stored inputs
def fracture():
    FR = MW + "outputs_fracture/"; MOD = OF + "models/fracture_a6x10w2_tv7/"
    sys.path.insert(0, FR + "sitewise/scripts"); sys.path.insert(0, MW + "src")
    import sitewise as S
    df, sel = S.build_frame()                       # train/val only (split filter first); y-NaN (unclear) rows are dropped here
    STORED = {"FT_sp": FR + "spine_ft/runs/final_tf2lse", "FT_pv": FR + "pelvis_ft/runs/final_tf2lse", "FT_rib": FR + "rib_ft",
              "FT_sp15": FR + "spine15_ft", "RVRS": FR + "push90b/cand/mix3_a6x10w2", "P15FT": FR + "pelvis15_ft/final",
              "PE": FR + "push90/cand/pe_ctx_head", "PE2": FR + "push90b/cand/pe_tv7r_head"}
    for n, d in STORED.items():                     # = integrate.py / final_ft3.py
        tr_ = pd.read_csv(d + "/oof_train.csv"); va_ = pd.read_csv(d + "/val_scores.csv")
        c = "ft" if "ft" in tr_.columns else [x for x in tr_.columns if x not in ("study_id", "fold", "in_tier", "split")][-1]
        cv = "ft" if "ft" in va_.columns else [x for x in va_.columns if x not in ("study_id", "fold", "split")][-1]
        df = df.join(pd.concat([tr_.set_index("study_id")[c], va_.set_index("study_id")[cv]]).rename(n))
    MED = {n: float(df.loc[df.split == "train", n].median()) for n in STORED}
    for n in STORED: df[n] = df[n].fillna(MED[n])
    K = pickle.load(open(MOD + "integ/stacker.pkl", "rb")); HEADS, FEATS = K["heads"], K["feats"]
    assert all(abs(K["medians"][n] - MED[n]) < 1e-9 for n in MED)
    need = sorted({c for v in FEATS.values() for c in v} | set(S.C3M) | set(STORED))
    tr = df[df.split == "train"].dropna(subset=need)
    def sc(F):
        L = pd.DataFrame({s: HEADS[s][1].decision_function(HEADS[s][0].transform(F[FEATS[s]].values)) for s in S.SITES}, index=F.index)
        return L[S.SITES].max(1)
    ids = [i for i in miss("fracture") if i in df.index]                              # labelled but dropped (dropna rules)
    F = df.loc[ids].copy(); nan_cols = {c: int(F[c].isna().sum()) for c in need if F[c].isna().any()}
    for c in nan_cols: F[c] = F[c].fillna(float(tr[c].median()))                        # integrate.py rule: S2 NaN -> train median
    va = df[df.split == "val"].dropna(subset=need + ["age"])
    v = have_val("fracture"); ver = sc(va.reindex(v.study_id))
    print("fracture: fill", len(ids), "of", len(miss("fracture")), "| imputed", nan_cols, "| unclear (no member scores)", len(miss("fracture")) - len(ids))
    return {"fill": sc(F), "verify": ver}, "outputs_final/models/fracture_a6x10w2_tv7 integ/stacker.pkl（S3n·max 三部位頭，全 train refit）於已存成員輸入"


# ------------------------------------------------------------------ assemble
def assemble():
    cal = json.load(open(OUTD + "calib.json")); P0 = pd.read_csv(OUTD + "scores_trval_prob.csv")
    D14 = [c for c in P0.columns if c not in ("study_id", "split")]
    P = P0.copy().set_index("study_id"); src, rows, rawfill = {}, [], []
    jobs = {fd: (lambda fd=fd: ib_tier(fd)) for fd in ("hepatic_steatosis", "surgically_absent_gallbladder", "splenomegaly",
                                                       "abdominal_aortic_aneurysm", "bowel_obstruction")}
    jobs.update({fd: (lambda fd=fd: ru_tier(fd)) for fd in RUS})
    jobs.update({"appendicitis": appendicitis, "gallstones": gallstones, "osteopenia": osteo_scores,
                 "biliary_ductal_dilation": biliary_scores, "fracture": fracture})
    sig = lambda s, d: 1 / (1 + np.exp(-(cal[d]["a"] * s + cal[d]["b"])))
    for fd in D14:
        try:
            out, desc = jobs[fd]()
        except Exception as e:
            print(fd, "FAILED", repr(e)[:300], flush=True); src[fd] = "FAILED: " + repr(e)[:200]; continue
        f = out["fill"].dropna(); assert P.loc[f.index, fd].isna().all()
        P.loc[f.index, fd] = sig(f.values, fd)
        rawfill.append(pd.DataFrame({"disease": fd, "study_id": f.index, "raw": f.values}))
        v = have_val(fd).set_index("study_id").join(out["verify"].rename("new"))
        dr = (v.stored - v.new).abs(); dp = (sig(v.stored, fd) - sig(v.new, fd)).abs()
        m = M[M.disease == fd]
        for sp in ("train", "val"):
            ids = set(m[m.split == sp].study_id)
            rows.append(dict(disease=fd, split=sp, missing=len(ids), filled=len(ids & set(f.index)), still_empty=len(ids - set(f.index)),
                             val_rescored_n=int(dr.notna().sum()) if sp == "val" else None,
                             val_rescored_max_abs_diff_raw=float(dr.max()) if sp == "val" else None,
                             val_rescored_max_abs_diff_prob=float(dp.max()) if sp == "val" else None))
        src[fd] = desc; print(rows[-2], "\n", rows[-1], flush=True)
    P = P.reset_index()[P0.columns]
    # check (1): every old cell unchanged
    a = P0[D14].values; b = P[D14].values; old = ~np.isnan(a)
    assert (P.study_id.values == P0.study_id.values).all() and np.array_equal(a[old], b[old]), "old cells changed"
    P.to_csv(OUTD + "scores_trval_prob_complete.csv", index=False, float_format="%.5f")
    Q = pd.read_csv(OUTD + "scores_trval_prob_complete.csv"); Q0 = pd.read_csv(OUTD + "scores_trval_prob.csv")
    n_diff = int((np.abs(Q[D14].values - Q0[D14].values) > 0)[~np.isnan(Q0[D14].values)].sum())
    print("old cells differing after write:", n_diff, "| empty cells", int(np.isnan(Q[D14].values).sum()), "->", "old", int(np.isnan(Q0[D14].values).sum()))
    pd.DataFrame(rows).to_csv(C + "complete_check.csv", index=False)
    pd.concat(rawfill).to_csv(C + "filled_raw_scores.csv", index=False)
    json.dump(dict(sources=src, old_cells_changed=n_diff), open(C + "complete_sources.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    {"osteo": osteo, "biliary": biliary, "assemble": assemble}[sys.argv[1]]()
