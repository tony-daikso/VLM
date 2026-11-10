#!/bin/bash
# (Mac relay, 17:20) Fracture member L (rib15_probe L_repl+rl03+lrc) seeds 3,4 on Spark GPU in parallel with A6000 (seeds 0-2):
# wait for A6000 l_matrix.npz -> stream to Spark (internal LAN via Mac) -> run the unchanged rib15_probe oof with the same
# label wrapper (CG ids y=0, split 'val') -> copy the oof npz back to A6000 frac/L/oof/full/. Device: Spark GB10 cuda (logged).
A='$HOME/Desktop/Tony/VLM/OWL/cg10k_best'
until ssh A6000 "grep -q '^N=' $A/frac/logs/L_build.log 2>/dev/null"; do sleep 30; done
ssh A6000 "cat $A/frac/L/l_matrix.npz" | ssh sparks "cat > ~/OWL/cg10k_fracL/l_matrix.npz"
ssh sparks 'cat > ~/OWL/cg10k_fracL/run.py <<PY
import sys, numpy as np, pandas as pd, os
SRC = os.path.expanduser("~/OWL/cg10k_ribs/code/src"); sys.path.insert(0, SRC)
import bone_probe as BP
_ll = BP.load_labels
def ll(path, ids):
    ids = [str(i) for i in ids]; df = _ll(path, [i for i in ids if i.startswith("AC")])
    cg = [i for i in ids if not i.startswith("AC")]
    add = pd.DataFrame({"y": 0.0, "compression": False, "sites": [[] for _ in cg], "acute": "not_stated"}, index=cg)
    return pd.concat([df, add])
BP.load_labels = ll
import rib15_probe as _S
sys.argv = ["rib15_probe.py", "--stage", "oof", "--out", os.path.expanduser("~/OWL/cg10k_fracL"), "--variants", "L_repl+rl03+lrc",
            "--seeds", sys.argv[1], "--device", "cuda", "--threads", "4"]
_S.main()
PY
cd ~/OWL/cg10k_ribs/code/src; for s in 3 4; do nice -n 10 ~/radar_env/bin/python ~/OWL/cg10k_fracL/run.py $s > ~/OWL/cg10k_fracL/L_s$s.log 2>&1 & done; wait; ls ~/OWL/cg10k_fracL/oof/full/'
for s in 3 4; do ssh sparks "cat ~/OWL/cg10k_fracL/oof/full/oof_L_L_repl-rl03-lrc_s$s.npz" | ssh A6000 "mkdir -p $A/frac/L/oof/full_spark && cat > $A/frac/L/oof/full_spark/oof_L_L_repl-rl03-lrc_s$s.npz"; done
echo SPARK_L_DONE
