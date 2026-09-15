"""One-off demo: overlay a ReXGroundingCT finding-level mask on the original
full-resolution CT-RATE image, to compare against the RadGenome-based pipeline."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from huggingface_hub import hf_hub_download

VOLUME_ID = "valid_1016_c_1"

ct_path = hf_hub_download(
    "ibrahimhamamci/CT-RATE",
    f"dataset/valid_fixed/valid_1016/valid_1016_c/{VOLUME_ID}.nii.gz",
    repo_type="dataset",
)
seg_path = hf_hub_download(
    "rajpurkarlab/ReXGroundingCT", f"segmentations/{VOLUME_ID}.nii.gz", repo_type="dataset"
)
meta_path = hf_hub_download("rajpurkarlab/ReXGroundingCT", "dataset.json", repo_type="dataset")
meta = json.load(open(meta_path))
entry = next(x for x in meta["train"] if x["name"] == f"{VOLUME_ID}.nii.gz")

ct = np.asarray(nib.load(ct_path).dataobj).astype(np.float32)
seg = np.asarray(nib.load(seg_path).dataobj)  # (F, H, W, D)

total = (seg > 0).any(axis=0).sum(axis=(0, 1))
z = int(np.argmax(total))
present = [i for i in range(seg.shape[0]) if (seg[i, :, :, z] > 0).any()]

lo, hi = -600 - 1500 / 2, -600 + 1500 / 2
disp = np.clip(ct[:, :, z], lo, hi)
disp = (disp - lo) / (hi - lo)
disp = np.rot90(disp)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(disp, cmap="gray")
axes[0].set_title(f"{VOLUME_ID}\noriginal CT-RATE slice {z} (full res {ct.shape[:2]})", fontsize=9)
axes[0].axis("off")

axes[1].imshow(disp, cmap="gray")
rng = np.random.RandomState(1)
colors = rng.rand(len(present), 3) * 0.6 + 0.4
labels = []
for c, i in zip(colors, present):
    m = np.rot90(seg[i, :, :, z] > 0)
    overlay = np.zeros((*m.shape, 4))
    overlay[..., :3] = c
    overlay[..., 3] = m * 0.55
    axes[1].imshow(overlay)
    labels.append(f"[{i}] {entry['findings'][str(i)][:45]}")
axes[1].set_title("ReXGroundingCT finding masks\n" + "\n".join(labels), fontsize=6.5)
axes[1].axis("off")

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "qa", f"{VOLUME_ID}_rexgroundingct_demo.png")
fig.savefig(out_path, dpi=140)
print("wrote", out_path)
