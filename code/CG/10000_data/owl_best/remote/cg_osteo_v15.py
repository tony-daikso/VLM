"""C-tier osteopenia (formal best '5win+v15', val .890): run the ORIGINAL osteopenia_v15/scripts/ov_run.py on a sandbox
data dir = the original Merlin train/val inputs + CG rows appended as split 'val' with dummy y_op = 0 (never trained on;
Merlin val rows unchanged -> stack_val.csv Merlin rows must equal the stored ones). CG inputs: L1-L4 1.5 mm emb15
(owl_v15, same Exp J encoder), 5 window heads (cg_heads 'op'), integrated stack (infer_best), geom15 levels (cg_l1hu).
usage: cg_osteo_v15.py build_cg <emb15_dir> <prefix> <out.npz>   (on CT-rate)
       cg_osteo_v15.py run                                        (on A6000)"""
import os, sys, subprocess, numpy as np, pandas as pd
LV = ["L1", "L2", "L3", "L4"]

def build(emb, pre, out):
    ids = sorted(f[:-4] for f in os.listdir(emb) if f.endswith(".npz") and ".tmp" not in f)
    N = len(ids); E = np.zeros((N, 4, 3, 448), np.float16); Em = np.zeros_like(E); P = np.zeros((N, 4), bool); NV = np.zeros((N, 4), np.int64)
    for n, cid in enumerate(ids):                       # identical slicing to ov_build.py
        z = np.load(f"{emb}/{cid}.npz"); li = [list(z["levels"]).index(l) for l in LV]
        E[n] = z["emb"][li]; Em[n] = z["emb_max"][li]; P[n] = z["present"][li]; NV[n] = z["nvox"][li]
    np.savez(out, E=E, Emax=Em, P=P, nvox=NV, ids=np.array([pre + i for i in ids])); print(out, N, P.mean(0))

def run():
    H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); OF = H + "mw_exp/outputs_final/"; B = H + "cg10k_best/"; D = B + "osteo_v15/data/"; O = B + "osteo_v15/out/"
    os.makedirs(D, exist_ok=True)
    if not os.path.exists(D + "merlin_ov.npz"):
        subprocess.run([os.path.expanduser("~/miniconda3/envs/radar/bin/python"), OF + "osteopenia_v15/scripts/ov_build.py", D + "merlin_ov.npz"], check=True)
    zs = [np.load(D + f) for f in ("merlin_ov.npz", "cg_main_ov.npz", "cg_delay_ov.npz")]
    np.savez(D + "ov_emb15_L1L4_trval.npz", **{k: np.concatenate([z[k].astype("<U12") if k == "ids" else z[k] for z in zs]) for k in ("E", "Emax", "P", "nvox", "ids")})
    cg = list(zs[1]["ids"]) + list(zs[2]["ids"])
    A = OF + "analysis_osteopenia/"
    lab = pd.read_csv(A + "labels_trval.csv"); add = pd.DataFrame({"study_id": cg, "split": "val", "y_op": 0.0})
    pd.concat([lab, add], ignore_index=True).to_csv(D + "labels_trval.csv", index=False)
    os.system(f"cp {A}oof_train.csv {A}bundle_trval.csv {OF}integrated_best/scores/osteopenia_oof.csv {OF}integrated_best/scripts/ib_core.py {D}")
    sv = pd.read_csv(A + "scores_val_descriptive.csv"); hc = pd.read_csv(B + "heads/R/analysis_osteopenia/scores_val_descriptive.csv")
    hc = hc[~hc.study_id.str.startswith("AC")]
    pd.concat([sv, hc[[c for c in sv.columns if c in hc.columns]]], ignore_index=True).to_csv(D + "scores_val_descriptive.csv", index=False)
    iv = pd.read_csv(OF + "integrated_best/scores/osteopenia_val.csv")
    ib = []
    for s, pre in (("main", ""), ("delay", "D_")):
        x = pd.read_csv(B + f"best/{s}/osteopenia.csv")[["study_id", "osteopenia_stack"]].rename(columns={"osteopenia_stack": "stack"})
        x["study_id"] = pre + x.study_id; ib.append(x)
    pd.concat([iv] + ib, ignore_index=True).to_csv(D + "osteopenia_val.csv", index=False)
    g = [pd.read_csv(H + "mw_exp/outputs_vert15/final_eval/geom15/geom15_levels.csv", usecols=["id", "level", "hu_body"])]
    for s, pre in (("main", ""), ("delay", "D_")):
        x = pd.read_csv(B + f"osteo_v15/data/l1hu_levels_{s}.csv", usecols=["id", "level", "hu_body"]); x["id"] = pre + x.id; g.append(x)
    pd.concat(g, ignore_index=True).to_csv(D + "geom15_levels.csv", index=False)
    r = subprocess.run(["nice", "-n", "15", os.path.expanduser("~/miniconda3/envs/radar/bin/python"), OF + "osteopenia_v15/scripts/ov_run.py", D, O],
                       stdout=open(B + "osteo_v15/run.log", "w"), stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="4"))
    print("ov_run exit", r.returncode)
    out = pd.read_csv(O + "stack_val.csv"); ref = pd.read_csv(OF + "osteopenia_v15/stack_val.csv")
    m = ref[["study_id", "5win+v15"]].merge(out[["study_id", "5win+v15"]], on="study_id", suffixes=("_o", "_n"))
    print("Merlin val rows", len(m), "max|d|", np.abs(m["5win+v15_o"] - m["5win+v15_n"]).max())
    c = out[out.study_id.isin(cg)][["study_id", "5win+v15"]].rename(columns={"5win+v15": "osteopenia"})
    for s, pre in (("main", ""), ("delay", "D_")):
        x = c[c.study_id.str.startswith(pre + "EX")] if pre else c[c.study_id.str.startswith("EX")]
        x.assign(study_id=x.study_id.str.replace("^D_", "", regex=True)).to_csv(B + f"osteo_v15/cg_scores_{s}.csv", index=False); print(s, len(x))

if __name__ == "__main__":
    build(*sys.argv[2:5]) if sys.argv[1] == "build_cg" else run()
