"""Pack the CNN input (central 64^3 HU int16 + GB-mask r0 bits) of the crops listed in a train/val labels csv into one
compressed npz (derived data only; no labels inside) for the 4090 CNN. usage: gs15_cnn_pack.py ROOT LABELS OUT.npz"""
import sys, os, numpy as np, pandas as pd
R, LAB, OUT = sys.argv[1:4]
lab = pd.read_csv(LAB); assert set(lab.split) <= {"train", "val"}
C, S0 = 64, 96; o = (S0 - C) // 2; sl = (slice(o, o + C),) * 3
ids, H, G = [], [], []
for s in lab.study_id:
    f = f"{R}/crops/{s}.npz"
    if not os.path.exists(f): continue
    z = np.load(f)
    if "hu" not in z: continue
    rs = tuple(z["rshape"]); Rg = np.unpackbits(z["regions"])[:int(np.prod(rs))].reshape(rs)
    ids.append(s); H.append(z["hu"][0][sl]); G.append(np.packbits(Rg[0, 0][sl].ravel()))
np.savez_compressed(OUT + ".tmp.npz", ids=np.array(ids), hu=np.stack(H), gb=np.stack(G)); os.replace(OUT + ".tmp.npz", OUT)
print("packed", len(ids), "of", len(lab), flush=True)
