"""CG10k: refit the integrated_best probe heads with the ORIGINAL analysis OOF scripts (same seeds/folds/recipe,
trained on Merlin train only) and score Merlin val (reproduction check) + CG10k rows (split 'cg').
Merlin test is never loaded (notest matrix; labels_trval only). CG rows get dummy label 0 (never trained on: split 'cg').
usage: cg_heads.py prep                     -> build sandbox inputs
       cg_heads.py run <job> [<job> ...]    -> run patched analysis scripts
       cg_heads.py check                    -> compare val head scores to stored ones, export CG heads"""
import os, sys, re, glob, subprocess, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/")
OF = H + "mw_exp/outputs_final/"; SRC = H + "mw_exp/src"; AC = OF + "integrated_best/scripts/analysis_code/"
SB = H + "cg10k_best/heads/"; R = SB + "R/"
MAT = "results/emb_matrix_baseline+soft_tissue_L40_W400+liver_L60_W150+narrow_abd_L40_W250+bone_L400_W1800+bones_emb_notest.npz"
CG = {"main": H + "cg10k_best/mat/cg_main.npz", "delay": H + "cg10k_best/mat/cg_delay.npz"}
PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")
ANA = ["steatosis", "splenomegaly", "hydronephrosis", "ascites", "bowel_obstruction", "osteopenia", "aaa", "absent_gb",
       "atherosclerosis", "free_air", "appendicitis"]

def cg_ids():
    out = []
    for s, p in CG.items():
        if os.path.exists(p):
            z = np.load(p); out.append((s, [("D_" if s == "delay" else "") + str(i) for i in z["ids"]], z))
    return out

def lab_comb(src, dst):
    lab = pd.read_csv(src); ids = sum([c[1] for c in cg_ids()], [])
    add = pd.DataFrame({"study_id": ids, "split": "cg"})
    for c in lab.columns:
        if c.startswith("y_"): add[c] = 0.0
    pd.concat([lab, add], ignore_index=True).to_csv(dst, index=False)

def prep():
    os.makedirs(R + "results", exist_ok=True)
    z = np.load(OF + MAT); units = list(z["units"])
    cg = cg_ids()
    E = np.concatenate([z["E"]] + [c[2]["E"] for c in cg]); P = np.concatenate([z["P"]] + [c[2]["P"] for c in cg])
    ids = np.concatenate([z["ids"].astype("<U12")] + [np.array(c[1], "<U12") for c in cg])
    for c in cg: assert list(c[2]["units"]) == units
    np.savez(R + MAT, E=E, P=P, ids=ids, units=np.array(units)); print("matrix", E.shape, flush=True)
    spl = pd.read_csv(H + "data/split.csv")
    spl = spl[spl.split != "test"]                                     # test never needed
    pd.concat([spl, pd.DataFrame({"study_id": ids[len(z["ids"]):], "split": "cg"})]).to_csv(SB + "split.csv", index=False)
    for a in ANA:
        os.makedirs(R + f"analysis_{a}", exist_ok=True)
    def u(n): return units.index(n)
    # steatosis / splenomegaly: float32 per-unit dumps + fixed pooled std stats
    for a, sub, files in (("steatosis", "", ("liver", "spleen")), ("splenomegaly", "data/", ("spleen",))):
        d = R + f"analysis_{a}/{sub}"; os.makedirs(d, exist_ok=True)
        for un in files:
            m = np.load(OF + f"analysis_{a}/{sub}{un}_emb_trval.npz")
            np.savez(d + f"{un}_emb_trval.npz", E=np.concatenate([m["E"]] + [c[2]["E"][:, u(un)].astype(np.float32) for c in cg]),
                     P=np.concatenate([m["P"]] + [c[2]["P"][:, u(un)].astype(np.float32) for c in cg]),
                     ids=np.concatenate([m["ids"].astype("<U12")] + [np.array(c[1], "<U12") for c in cg]))
        os.system(f"cp {OF}analysis_{a}/{sub}std_stats_main12.npz {d}")
        lab_comb(OF + f"analysis_{a}/{sub}labels_trval.csv", d + "labels_trval.csv")
    os.makedirs(R + "analysis_steatosis/data", exist_ok=True)       # hs_oof reads D/data/
    for f in ("liver_emb_trval.npz", "spleen_emb_trval.npz", "std_stats_main12.npz", "labels_trval.csv"):
        if not os.path.exists(R + "analysis_steatosis/data/" + f): os.symlink(R + "analysis_steatosis/" + f, R + "analysis_steatosis/data/" + f)
    # multi-unit slices (same layout as the dump scripts)
    for a, fn, un, dt in (("hydronephrosis", "kidney_emb_trval.npz", ["kidney_left", "kidney_right"], np.float16),
                          ("ascites", "emb12_trval.npz", units[:12], np.float16),
                          ("bowel_obstruction", "emb_sb_co_st_trval.npz", ["small_bowel", "colon", "stomach"], None),
                          ("osteopenia", "emb_lumbar_L1_trval.npz", ["lumbar_spine", "L1"], np.float16)):
        lab = set(pd.read_csv(OF + f"analysis_{a}/labels_trval.csv").study_id); k = np.array([i in lab or not i.startswith("AC") for i in ids])
        ui = [u(x) for x in un]; e = E[k][:, ui]
        np.savez(R + f"analysis_{a}/{fn}", E=e.astype(dt) if dt else e, P=P[k][:, ui], ids=ids[k], units=np.array(un))
    for a in ANA:
        if a in ("splenomegaly",): continue
        lab_comb(OF + f"analysis_{a}/labels_trval.csv", R + f"analysis_{a}/labels_trval.csv")
    os.makedirs(R + "analysis_ascites/scripts", exist_ok=True)
    if not os.path.exists(R + "analysis_ascites/scripts/probe.py"): os.symlink(SRC + "/probe.py", R + "analysis_ascites/scripts/probe.py")
    if not os.path.exists(R + "analysis_aaa/emb.npz"): os.symlink(R + MAT, R + "analysis_aaa/emb.npz")
    print("prep done", flush=True)

# job -> (script, argv, stored dir, out files)
JOBS = {
    "hs": ("steatosis/hs_oof.py", [SRC, R + "analysis_steatosis"], "analysis_steatosis", "oof_train.csv", "scores_val_descriptive.csv"),
    "hs_cd": ("steatosis/hs_oof_spleen.py", [SRC, R + "analysis_steatosis", "concat_diff"], "analysis_steatosis", "oof_train_concat_diff.csv", "scores_val_concat_diff.csv"),
    "sp": ("splenomegaly/sp_oof.py", [SRC, R + "analysis_splenomegaly"], "analysis_splenomegaly", "oof_train.csv", "scores_val.csv"),
    "hn": ("hydronephrosis/hn_oof.py", [R + "analysis_hydronephrosis", SRC], "analysis_hydronephrosis", "oof_train.csv", "scores_val_descriptive.csv"),
    "asc": ("ascites/asc_oof.py", ["E3_late", "E3_late_all12"], "analysis_ascites", "oof_train.csv", "scores_val_descriptive.csv"),  # per-spec rng(0): subset == full run
    "sbo": ("bowel_obstruction/sbo_oof.py", [R + "analysis_bowel_obstruction/emb_sb_co_st_trval.npz", R + "analysis_bowel_obstruction/labels_trval.csv", SRC, R + "analysis_bowel_obstruction"],
            "analysis_bowel_obstruction", "oof_train.csv", "scores_val_descriptive.csv"),
    "sbo_st": ("bowel_obstruction/sbo_oof.py", [R + "analysis_bowel_obstruction/emb_sb_co_st_trval.npz", R + "analysis_bowel_obstruction/labels_trval.csv", SRC, R + "analysis_bowel_obstruction_st", "E3_late_st"],
            "analysis_bowel_obstruction_st", "oof_train.csv", "scores_val_descriptive.csv"),   # variant: E3_late_st run alone (shared rng) - kept if it matches the stored head
    "op": ("osteopenia/op_oof.py", [], "analysis_osteopenia", "oof_train.csv", "scores_val_descriptive.csv"),
    "aaa": ("aaa/aaa_oof_local.py", [], "analysis_aaa", "oof_train.csv", "scores_val_descriptive.csv"),
    "ag": ("absent_gb/ag_oof.py", [], "analysis_absent_gb", "oof_train.csv", "scores_val_descriptive.csv"),
    "ath": ("atherosclerosis/ath_oof.py", [], "analysis_atherosclerosis", "oof_train.csv", "scores_val_descriptive.csv"),
    "fa": ("free_air/fa_oof.py", [], "analysis_free_air", "oof_train.csv", "scores_val_descriptive.csv"),
    "ap": ("appendicitis/ap_oof.py", [], "analysis_appendicitis", "oof_train.csv", "scores_val_descriptive.csv"),
    "bd": ("biliary/bd_oof.py", ["E0", "E3_late", "E3late_cat_+gb_duod"], "analysis_biliary", "oof_train.csv", "scores_val_descriptive.csv"),  # per-spec rng(0)
}

def patch(code):
    code = code.replace("/home/advantech/Desktop/Tony/VLM/OWL/mw_exp/outputs_final/", R)
    code = code.replace("/home/advantech/Desktop/Tony/VLM/OWL/data/split.csv", SB + "split.csv")
    code = re.sub(r'R = "/private/tmp/[^"]*/aaa/"', f'R = "{R}analysis_aaa/"', code)
    code = code.replace('S == "val"', 'np.isin(S, ("val", "cg"))')
    code = code.replace('("train", "val")', '("train", "val", "cg")')
    code = code.replace('sys.path.insert(0, A + "scripts")', f'sys.path.insert(0, "{SRC}")')
    code = code.replace("/Users/tony.tu/Desktop/戴承智慧/VLM/code/Organ-aware Window Learning/use_merlin_dataset/mw_exp/src", SRC)
    code = code.replace('specs["L1unit_E0"] = (1, {"mode": "single", "win_idx": 0}); specs["L1unit_E3_late"] = (1, {"mode": "late"})', "")  # op: unused trailing specs
    return code

def run(jobs):
    os.makedirs(SB + "patched", exist_ok=True); os.makedirs(SB + "logs", exist_ok=True)
    env = dict(os.environ, ADIR=R + "analysis_ascites/", OP_DIR=R + "analysis_osteopenia/", OMP_NUM_THREADS="4")
    for j in jobs:
        sc, argv = JOBS[j][0], JOBS[j][1]
        p = SB + "patched/" + j + "_" + os.path.basename(sc); open(p, "w").write(patch(open(AC + sc).read()))
        print(j, "start", flush=True)
        r = subprocess.run(["nice", "-n", "19", PY, p] + argv, stdout=open(SB + f"logs/{j}.log", "w"), stderr=subprocess.STDOUT, env=env)
        print(j, "exit", r.returncode, flush=True)

HEADS = {"hs": ["E3_late"], "hs_cd": ["E3_late_concat_diff"], "sp": ["E3_late"], "hn": ["E3_late", "E3_late_max"],
         "asc": ["E3_late", "E3_late_all12"], "sbo": ["E3_late", "E3_late_st"], "sbo_st": ["E3_late_st"], "op": ["E3_late", "E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"],
         "aaa": ["E3_late"], "ag": ["E3_late"], "ath": ["E3_late"], "fa": ["E3_late"], "ap": ["E3_late", "E3_late_colon_sb"], "bd": ["E0", "E3_late", "E3late_cat_+gb_duod"]}

def check():
    os.makedirs(SB + "cg", exist_ok=True); rows = []
    for j, (sc, argv, d, fo, fv) in JOBS.items():
        new = R + d + "/" + fv
        if not os.path.exists(new): print(j, "no output"); continue
        ref = OF + re.sub(r"_st2?$", "", d) + "/" + fv
        n = pd.read_csv(new); o = pd.read_csv(ref)
        m = o.merge(n, on="study_id", suffixes=("_old", "_new"))
        for h in HEADS[j]:
            if h + "_old" in m:
                df = (m[h + "_old"] - m[h + "_new"]).abs()
                rows.append(dict(job=j, head=h, n_val=len(m), max_abs_diff=df.max(), corr=m[[h + "_old", h + "_new"]].corr().iloc[0, 1]))
        o = o[["study_id"] + [h for h in HEADS[j] if h in o]]
        c = n[~n.study_id.str.startswith("AC")]
        c[["study_id"] + [h for h in HEADS[j] if h in c]].to_csv(SB + f"cg/{j}.csv", index=False)
    print(pd.DataFrame(rows).to_string())

def prep_extra(a):
    os.makedirs(R + f"analysis_{a}", exist_ok=True)
    lab_comb(OF + f"analysis_{a}/labels_trval.csv", R + f"analysis_{a}/labels_trval.csv"); print("labels", a)

if __name__ == "__main__":
    if sys.argv[1] == "prep_extra": prep_extra(sys.argv[2]); sys.exit()
    {"prep": lambda: prep(), "run": lambda: run(sys.argv[2:]), "check": lambda: check()}[sys.argv[1]]()
