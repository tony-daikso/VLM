"""CG10k: per-disease best OWL models (tier A = integrated_best saved stacks) -> probs_best_<set>.csv.
  heads : cg10k_best/heads/cg/<job>.csv  (refit with the original analysis scripts, cg_heads.py)
  feats : cg10k_best/feat/<set>/*.csv     (same frozen feature scripts, reproduced bit-exact on Merlin)
  stacks: mw_exp/outputs_final/integrated_best/models/<finding>.joblib via ib_apply.py (unchanged)
usage: infer_best.py valcheck            -> apply stacks to Merlin val with the refit heads (must match stored scores)
       infer_best.py run <set> [...]     -> CG scores; findings without a tier-A model keep E3_late (expB) and are flagged."""
import os, sys, subprocess, numpy as np, pandas as pd
H = os.path.expanduser("~/Desktop/Tony/VLM/OWL/"); OF = H + "mw_exp/outputs_final/"; IB = OF + "integrated_best/"
B = H + "cg10k_best/"; HD = B + "heads/"; PY = os.path.expanduser("~/miniconda3/envs/radar/bin/python")
sys.argv_save = list(sys.argv); sys.argv = [sys.argv[0], IB]; sys.path.insert(0, IB + "scripts"); import ib_stack; sys.argv = sys.argv_save
# finding -> (head columns: (job, col, name in stack)), CG feature file
TA = {
    "hepatic_steatosis": ([("hs", "E3_late", "E3_late"), ("hs_cd", "E3_late_concat_diff", "E3_late_concat_diff")], "steatosis_hu.csv"),
    "surgically_absent_gallbladder": ([("ag", "E3_late", "E3_late")], "absent_gb_feat.csv"),
    "splenomegaly": ([("sp", "E3_late", "E3_late")], "splenomegaly_geom.csv"),
    "hydronephrosis": ([("hn", "E3_late", "E3_late"), ("hn", "E3_late_max", "E3_late_max")], "hydronephrosis_hu.csv"),
    "abdominal_aortic_aneurysm": ([("aaa", "E3_late", "E3_late")], "aaa_geom.csv"),
    "atherosclerosis": ([("ath", "E3_late", "E3_late")], "atherosclerosis_hu.csv"),
    "osteopenia": ([("op", c, c) for c in ["E3_late", "E0", "E1_soft_tissue", "E1_liver", "E1_narrow_abd", "E1_bone"]], "aaa_geom.csv"),  # heads-only model ('win'); feature file unused
    "bowel_obstruction": ([("sbo", "E3_late", "E3_late"), ("sbo", "E3_late_st", "E3_late_st")], "sbo_feat.csv"),
    "appendicitis": ([("ap", "E3_late", "E3_late"), ("ap", "E3_late_colon_sb", "E3_late_colon_sb")], "appendicitis_hu.csv"),
    "ascites": ([("asc", "E3_late", "E3_late"), ("asc", "E3_late_all12", "E3_late_all12")], "ascites_hu.csv"),
    "free_air": ([("fa", "E3_late", "E3_late")], "free_air_air.csv"),
}
JOBDIR = {"hs": ("analysis_steatosis", "scores_val_descriptive.csv"), "hs_cd": ("analysis_steatosis", "scores_val_concat_diff.csv"),
          "sp": ("analysis_splenomegaly", "scores_val.csv"), "hn": ("analysis_hydronephrosis", "scores_val_descriptive.csv"),
          "asc": ("analysis_ascites", "scores_val_descriptive.csv"), "sbo": ("analysis_bowel_obstruction", "scores_val_descriptive.csv"),
          "op": ("analysis_osteopenia", "scores_val_descriptive.csv"), "aaa": ("analysis_aaa", "scores_val_descriptive.csv"),
          "ag": ("analysis_absent_gb", "scores_val_descriptive.csv"), "ath": ("analysis_atherosclerosis", "scores_val_descriptive.csv"),
          "fa": ("analysis_free_air", "scores_val_descriptive.csv"), "ap": ("analysis_appendicitis", "scores_val_descriptive.csv")}

def heads(fd, rows):
    """rows: 'val' (Merlin val, refit run) or 'cg'"""
    out = None
    for j, c, nm in TA[fd][0]:
        d = pd.read_csv(HD + "R/" + "/".join(JOBDIR[j]))
        d = d[d.study_id.str.startswith("AC")] if rows == "val" else d[~d.study_id.str.startswith("AC")]
        d = d[["study_id", c]].rename(columns={c: nm})
        out = d if out is None else out.merge(d, on="study_id", how="outer")
    return out

def apply(fd, hcsv, feats, out):
    r = subprocess.run([PY, IB + "scripts/ib_apply.py", IB, fd, hcsv, feats, out], capture_output=True, text=True)
    if r.returncode: print(r.stdout, r.stderr); raise SystemExit(fd)

def valcheck(only=None):
    os.makedirs(B + "valcheck", exist_ok=True); res = []
    for fd in (only or TA):
        c = ib_stack.CFG[fd]
        fe = ",".join(IB + "features/" + f[1:] if f.startswith("+") else OF + f"analysis_{c['an']}/{f}" for f in c["feats"])
        h = heads(fd, "val"); hc = B + f"valcheck/{fd}_heads.csv"; h.to_csv(hc, index=False)
        apply(fd, hc, fe, B + f"valcheck/{fd}.csv")
        n = pd.read_csv(B + f"valcheck/{fd}.csv"); o = pd.read_csv(IB + f"scores/{fd}_val.csv")
        m = o.merge(n, on="study_id")
        from ib_core import auc
        res.append(dict(finding=fd, n=len(m), stored_val=auc(m.y, m["stack"]), refit_val=auc(m.y, m[f"{fd}_stack"]),
                        max_abs_diff=(m["stack"] - m[f"{fd}_stack"]).abs().max(), corr=np.corrcoef(m["stack"], m[f"{fd}_stack"])[0, 1]))
        print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(B + "valcheck/summary.csv", mode="a", header=not os.path.exists(B + "valcheck/summary.csv"), index=False)

def run(s):
    os.makedirs(B + f"best/{s}", exist_ok=True)
    pre = "D_" if s == "delay" else ""
    e3 = pd.read_csv(B + f"e3late/probs_{s}.csv")                        # expB E3_late probs from infer_head.py
    out = e3[["study_id"]].copy(); src = {}
    for fd in TA:
        try:
            h = heads(fd, "cg")
            assert all(c in h for _, _, c in TA[fd][0]) and len(h) > 0
        except Exception as e:
            print(fd, "heads not ready -> E3_late placeholder", repr(e)[:80]); continue
        h = h; h = h[h.study_id.str.startswith("D_")] if pre else h[~h.study_id.str.startswith("D_")]
        h["study_id"] = h.study_id.str.replace("^D_", "", regex=True)
        hc = B + f"best/{s}/{fd}_heads.csv"; h.to_csv(hc, index=False)
        apply(fd, hc, B + f"feat/{s}/{TA[fd][1]}", B + f"best/{s}/{fd}.csv")
        d = pd.read_csv(B + f"best/{s}/{fd}.csv")[["study_id", f"{fd}_stack"]].rename(columns={f"{fd}_stack": fd})
        out = out.merge(d, on="study_id", how="left"); src[fd] = "integrated_best stack (tier A)"
    for fd in [c for c in e3.columns if c != "study_id" and "__" not in c and c not in src]:
        out = out.merge(e3[["study_id", fd]], on="study_id", how="left"); src[fd] = "E3_late expB (placeholder until tier B/C)"
    # tier B overrides (region-crop-unit stacks refit & saved by bstack_apply.py / cg_brlq.py, val reproduced)
    TB = {"hydronephrosis": "Exp M 腎竇單元 REF＋全部單元頭 (tier B)", "free_air": "Exp M 橫膈下/前腹單元 REF＋全部單元頭 (tier B)",
          "ascites": "Exp M 腹膜腔 pocket REF＋U_org (tier B)", "atherosclerosis": "Exp N 管壁單元 REF＋全部單元頭＋HU (tier B)"}
    for fd, nm in TB.items():
        f = B + f"bstack/{fd}/cg_scores.csv" if s == "main" else B + f"bstack/{fd}_d/cg_scores.csv"
        if os.path.exists(f):
            d = pd.read_csv(f)[["study_id", fd]].rename(columns={fd: fd + "_b"}); out = out.merge(d, on="study_id", how="left")
            out[fd] = out[fd + "_b"]; out = out.drop(columns=fd + "_b"); src[fd] = nm
    f = B + f"bprobe/rlq/cg_scores_{s}.csv"
    if os.path.exists(f):
        d = pd.read_csv(f).rename(columns={"appendicitis": "ap_b"}); out = out.merge(d, on="study_id", how="left")
        out["appendicitis"] = out["ap_b"]; out = out.drop(columns="ap_b"); src["appendicitis"] = "RLQ 單元 REF＋U_rlq_cs (tier B)"
    for fd, f, nm in (("osteopenia", B + f"osteo_v15/cg_scores_{s}.csv", "5 window logit＋1.5 mm L1–L4 head (C 級 v15)"),
                      ("biliary_ductal_dilation", B + f"bprobe/cg_scores_biliary_{s}.csv", "BD5 肝門單元 stack (tier B)"),
                      ("fracture", B + f"frac/cg_scores_{s}.csv", "骨折最終版 C3 LR(expI+J_add+L+骨盆單元) (C 級)"),
                      ("gallstones", B + f"gs/cg_scores_{s}.csv", "實驗 N：G5c＋膽囊單元 U (tier B)")):
        if os.path.exists(f):
            d = pd.read_csv(f)[["study_id", fd]].rename(columns={fd: fd + "_b"}); out = out.merge(d, on="study_id", how="left")
            out[fd] = out[fd + "_b"]; out = out.drop(columns=fd + "_b"); src[fd] = nm
    miss = {fd: int(out[fd].isna().sum()) for fd in src}
    out.to_csv(B + f"best/probs_best_{s}.csv", index=False)
    pd.DataFrame({"finding": list(src), "method": list(src.values()), "n_missing": [miss[f] for f in src]}).to_csv(B + f"best/methods_{s}.csv", index=False)
    print(s, out.shape, miss, flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "valcheck": valcheck(sys.argv[2:] or None)
    else:
        for s in sys.argv[2:]: run(s)
