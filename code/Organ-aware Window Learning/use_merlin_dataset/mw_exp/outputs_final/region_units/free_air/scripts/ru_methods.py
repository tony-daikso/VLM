"""count per-crop localisation methods over a crops dir. usage: ru_methods.py crops_dir -> prints json Counter per crop slot"""
import sys, glob, json, numpy as np, collections
C = None
for f in glob.glob(sys.argv[1] + "/*.npz"):
    if ".tmp" in f: continue
    m = [str(x) for x in np.load(f)["method"]]
    if C is None: C = [collections.Counter() for _ in m]
    for k, x in enumerate(m): C[k][x] += 1
print(json.dumps([dict(c) for c in C]))
