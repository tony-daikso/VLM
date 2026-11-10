"""ON the L40S: print nc ids in out/nc that are done (5 emb + 3 bone npz) and ids in state_nc/nc/failed.jsonl."""
import json, sys
from pathlib import Path
G = Path("/root/g4090"); O = G / "out/nc"
W = ["baseline", "soft_tissue_L40_W400", "liver_L60_W150", "narrow_abd_L40_W250", "bone_L400_W1800"]
B = ["baseline", "soft_tissue_L40_W400", "bone_L400_W1800"]
ids = sys.argv[1:]
print("D: " + " ".join(c for c in ids if all((O / "emb" / w / f"{c}.npz").exists() for w in W)
               and all((O / "emb_bones" / w / f"{c}.npz").exists() for w in B)))
f = G / "state_nc/nc/failed.jsonl"
fl = {json.loads(l)["id"] for l in open(f)} if f.exists() else set()
print("F: " + " ".join(c for c in ids if c in fl))
