#!/bin/bash
# (Mac relay, 17:22) Exp I member r18 seed 1 on Spark CPU in parallel with A6000 (seed 0): copy the sandbox bones dir
# (bone_matrix / rib_matrix with CG rows, expI meta, E3_late hp json) -> unchanged bone_expI.py with the label wrapper
# and patched B/OUT -> copy the member npz back to A6000 frac/bones/expI/members_spark/.  Device: Spark CPU (logged).
A='$HOME/Desktop/Tony/VLM/OWL/cg10k_best'
ssh sparks 'mkdir -p ~/OWL/cg10k_fracI/bones/expI ~/OWL/cg10k_fracI/bones/rerun/std'
for f in bone_matrix.npz rib_matrix.npz expI/meta.csv; do ssh A6000 "cat $A/frac/bones/$f" | ssh sparks "cat > ~/OWL/cg10k_fracI/bones/$f"; done
ssh A6000 "cat ~/Desktop/Tony/VLM/OWL/mw_exp/outputs_fracture/bones/rerun/std/E3_late.json" | ssh sparks "cat > ~/OWL/cg10k_fracI/bones/rerun/std/E3_late.json"
ssh sparks 'cat > ~/OWL/cg10k_fracI/run.py <<PY
import sys, numpy as np, pandas as pd, os, pathlib
SRC = os.path.expanduser("~/OWL/cg10k_ribs/code/src"); sys.path.insert(0, SRC)
import bone_probe as BP
_ll = BP.load_labels
def ll(path, ids):
    ids = [str(i) for i in ids]; df = _ll(path, [i for i in ids if i.startswith("AC")])
    cg = [i for i in ids if not i.startswith("AC")]
    add = pd.DataFrame({"y": 0.0, "compression": False, "sites": [[] for _ in cg], "acute": "not_stated"}, index=cg)
    return pd.concat([df, add])
BP.load_labels = ll
import bone_expI as _M
FB = os.path.expanduser("~/OWL/cg10k_fracI/bones/")
_M.B = pathlib.Path(FB); _M.OUT = pathlib.Path(FB + "expI")
sys.argv = ["bone_expI.py", "--variant", "r18", "--seed-off", sys.argv[1], "--outdir", FB + "expI/members", "--threads", "12"]
_M.main()
PY
cd ~/OWL/cg10k_ribs/code/src && OMP_NUM_THREADS=12 nice -n 10 ~/radar_env/bin/python ~/OWL/cg10k_fracI/run.py 1 > ~/OWL/cg10k_fracI/r18_s1.log 2>&1; tail -1 ~/OWL/cg10k_fracI/r18_s1.log'
ssh sparks "cat ~/OWL/cg10k_fracI/bones/expI/members/oof_bone_E_r18_s1.npz" | ssh A6000 "mkdir -p $A/frac/bones/expI/members_spark && cat > $A/frac/bones/expI/members_spark/oof_bone_E_r18_s1.npz"
echo SPARK_R18_DONE
